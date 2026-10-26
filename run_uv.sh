#!/bin/bash
set -euo pipefail

# ==== Settings ====
DATE=$(date +%m%d) # auto complete
MODAL=proprio # vision/proprio

UV_ACTIVATE=/mnt/phx2/DynamicTruncation/.venv/bin/activate
PROCS_PER_GPU=2 # Maximum concurrent training jobs on each GPU.

enable_dny_trunc=true
val_ens_num=5
VE_metric=wasser
#wasser, ucb
imag_hori=32
METHOD=VE_${VE_metric}_H${imag_hori} # Value Ensemble

GPU_IDs=(
    4
    5
)
seeds=(
    1
    42
    100
)
tasks=(
    dmc_finger_spin
    dmc_hopper_hop
    dmc_quadruped_run
)



# ==== Loop ====
command -v tmux >/dev/null || { echo "Error: tmux is required." >&2; exit 1; }

source "$UV_ACTIVATE"
PYTHON=$(command -v python) || { echo "Error: python is required." >&2; exit 1; }
printf 'uv environment: %s (Python: %s)\n' "$UV_ACTIVATE" "$PYTHON"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (( ${#GPU_IDs[@]} == 0 || ${#tasks[@]} == 0 || ${#seeds[@]} == 0 )); then
    echo "Error: GPU_IDs, tasks and seeds must all be non-empty." >&2
    exit 1
fi
if [[ ! $PROCS_PER_GPU =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: PROCS_PER_GPU must be a positive integer." >&2
    exit 1
fi
declare -A seen_gpus=()
for gpu in "${GPU_IDs[@]}"; do
    if [[ ! $gpu =~ ^[0-9]+$ || -n ${seen_gpus[$gpu]:-} ]]; then
        echo "Error: GPU_IDs must contain unique, non-negative GPU indices: $gpu" >&2
        exit 1
    fi
    seen_gpus[$gpu]=1
done

# Each generated worker runs its assigned jobs sequentially inside tmux.
train_job() {
    local task=$1 seed=$2 status
    if [[ -z ${TMUX:-} ]]; then
        echo "Error: training must run inside tmux." >&2
        exit 1
    fi
    printf '\n[GPU %s] Starting task=%s seed=%s\n' "$GPU_ID" "$task" "$seed"
# -------------------------------python cmd-----------------------------
    if CUDA_VISIBLE_DEVICES="$GPU_ID" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU_ID" \
        "$PYTHON" train.py \
            "env=dmc_${MODAL}" \
            "env.task=$task" \
            "logdir=logdir/${METHOD}/${task}_${MODAL}_s${seed}_${DATE}" \
            model.compile=True \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            model.rep_loss=dreamer \
            "model.dynamic_trunc.enabled=${enable_dny_trunc}" \
            "model.value_ensemble_size=${val_ens_num}" \
            "model.VE_metric=${VE_metric}" \
            "model.imag_horizon=${imag_hori}" \
            "seed=$seed"; then
        printf '[GPU %s] Finished task=%s seed=%s\n' "$GPU_ID" "$task" "$seed"
    else
        status=$?
        printf '[GPU %s] FAILED task=%s seed=%s (exit %s)\n' "$GPU_ID" "$task" "$seed" "$status" >&2
        FAILED=1
    fi
}

WORKER_DIR=$(mktemp -d /tmp/r2dreamer-workers.XXXXXX)
SESSION="r2dreamer_${DATE}_$(date +%H%M%S)_$$"
gpu_count=${#GPU_IDs[@]}
total_jobs=$(( ${#tasks[@]} * ${#seeds[@]} ))
worker_count=$(( gpu_count * PROCS_PER_GPU ))
if (( worker_count > total_jobs )); then
    worker_count=$total_jobs
fi
worker_names=()

printf 'Distributing %s jobs across %s GPUs, up to %s concurrent jobs per GPU (%s workers):\n' \
    "$total_jobs" "$gpu_count" "$PROCS_PER_GPU" "$worker_count"
for ((worker_index = 0; worker_index < worker_count; worker_index++)); do
    gpu_index=$(( worker_index % gpu_count ))
    gpu=${GPU_IDs[$gpu_index]}
    slot=$(( worker_index / gpu_count + 1 ))
    worker_name="gpu_${gpu}"
    if (( PROCS_PER_GPU > 1 )); then
        worker_name+="_worker_${slot}"
    fi
    worker_names+=("$worker_name")
    worker="$WORKER_DIR/${worker_name}.sh"
    {
        printf '#!/bin/bash\nset -euo pipefail\n'
        printf 'cd -- %q\n' "$SCRIPT_DIR"
        # Restore caller settings, then activate the uv environment in this worker.
        for name in PATH LD_LIBRARY_PATH PYTHONPATH VIRTUAL_ENV https_proxy http_proxy all_proxy no_proxy; do
            if [[ -v $name ]]; then
                printf 'export %s=%q\n' "$name" "${!name}"
            fi
        done
        printf 'source %q\n' "$UV_ACTIVATE"
        printf 'GPU_ID=%q\nPYTHON=%q\nMODAL=%q\nMETHOD=%q\nDATE=%q\nFAILED=0\n' \
            "$gpu" "$PYTHON" "$MODAL" "$METHOD" "$DATE"
        printf 'enable_dny_trunc=%q\nval_ens_num=%q\n' \
            "$enable_dny_trunc" "$val_ens_num"
        printf 'VE_metric=%q\nimag_hori=%q\n' \
            "$VE_metric" "$imag_hori"
        declare -f train_job
    } > "$worker"

    assigned=$(( total_jobs / worker_count + (worker_index < total_jobs % worker_count) ))
    printf '\nGPU %s, worker %s: %s jobs\n' "$gpu" "$slot" "$assigned"
    job_index=0
    for task in "${tasks[@]}"; do
        for seed in "${seeds[@]}"; do
            if (( job_index % worker_count == worker_index )); then
                printf '  task=%s seed=%s\n' "$task" "$seed"
                printf 'train_job %q %q\n' "$task" "$seed" >> "$worker"
            fi
            job_index=$((job_index + 1))
        done
    done
    printf 'printf "\\n[GPU %%s] All assigned jobs completed; failure flag=%%s\\n" "$GPU_ID" "$FAILED"\nexit "$FAILED"\n' >> "$worker"
done

# Keep finished GPU windows until every worker exits, then remove this session.
cleanup="$WORKER_DIR/cleanup.sh"
{
    printf '#!/bin/bash\nset -euo pipefail\nSESSION=%q\n' "$SESSION"
    cat <<'CLEANUP'
while tmux has-session -t "=$SESSION" 2>/dev/null; do
    panes=$(tmux list-panes -s -t "=$SESSION" -F '#{window_name} #{pane_dead}') || exit 0
    running=false
    while read -r name dead; do
        if [[ $name == gpu_* && $dead == 0 ]]; then
            running=true
            break
        fi
    done <<< "$panes"
    if [[ $running == false ]]; then
        tmux kill-session -t "=$SESSION"
        exit 0
    fi
    sleep 2
done
CLEANUP
} > "$cleanup"

# Prepare a window per concurrent worker before launching any training jobs.
tmux new-session -d -s "$SESSION" -n "${worker_names[0]}" -c "$SCRIPT_DIR" "$BASH"
for worker_index in "${!worker_names[@]}"; do
    worker_name=${worker_names[$worker_index]}
    window="$SESSION:$worker_name"
    if (( worker_index > 0 )); then
        tmux new-window -d -t "$SESSION:" -n "$worker_name" -c "$SCRIPT_DIR" "$BASH"
    fi
    tmux set-window-option -t "$window" remain-on-exit on
done
for worker_name in "${worker_names[@]}"; do
    tmux respawn-window -k -t "$SESSION:$worker_name" "$BASH" "$WORKER_DIR/${worker_name}.sh"
done
tmux new-window -d -t "$SESSION:" -n cleanup -c "$SCRIPT_DIR" "$BASH" "$cleanup"
printf '\nStarted tmux session: %s\nAttach: tmux attach -t %s\n' "$SESSION" "$SESSION"
printf 'Worker scripts: %s\nThis session will be deleted automatically after all GPU workers exit.\n' "$WORKER_DIR"

# ==== Tasks ====
# tasks=(
#     dmc_acrobot_swingup
#     dmc_ball_in_cup_catch
#     dmc_cartpole_balance
#     dmc_cartpole_balance_sparse
#     dmc_cartpole_swingup
#     dmc_cartpole_swingup_sparse
#     dmc_cheetah_run
#     dmc_finger_spin
#     dmc_finger_turn_easy
#     dmc_finger_turn_hard
#     dmc_hopper_hop
#     dmc_hopper_stand
#     dmc_pendulum_swingup
#     dmc_quadruped_run
#     dmc_quadruped_walk
#     dmc_reacher_easy
#     dmc_reacher_hard
#     dmc_walker_run
#     dmc_walker_stand
#     dmc_walker_walk
# )
