#!/bin/bash
set -euo pipefail

# ==== Settings ====
DATE=$(date +%m%d) # auto complete
MODAL=proprio # vision/proprio
METHOD=baseline # Value Ensemble
CONDA_ENV=r2dreamer

enable_dny_trunc=false
val_ens_num=1

GPU_IDs=(
    6
    7
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

# Non-interactive Bash scripts must initialize Conda before activating an env.
CONDA_BIN=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z $CONDA_BIN ]]; then
    for conda_root in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        if [[ -x $conda_root/bin/conda ]]; then
            CONDA_BIN="$conda_root/bin/conda"
            break
        fi
    done
fi
if [[ -z $CONDA_BIN || ! -x $CONDA_BIN ]]; then
    echo "Error: Conda was not found. Add conda to PATH or set CONDA_EXE." >&2
    exit 1
fi
CONDA_BASE=$("$CONDA_BIN" info --base)
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
PYTHON=$(command -v python) || { echo "Error: python is required." >&2; exit 1; }
printf 'Conda environment: %s (Python: %s)\n' "$CONDA_ENV" "$PYTHON"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (( ${#GPU_IDs[@]} == 0 || ${#tasks[@]} == 0 || ${#seeds[@]} == 0 )); then
    echo "Error: GPU_IDs, tasks and seeds must all be non-empty." >&2
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

printf 'Distributing %s jobs across %s GPUs (GPUs run in parallel):\n' "$total_jobs" "$gpu_count"
for gpu_index in "${!GPU_IDs[@]}"; do
    gpu=${GPU_IDs[$gpu_index]}
    worker="$WORKER_DIR/gpu_${gpu}.sh"
    {
        printf '#!/bin/bash\nset -euo pipefail\n'
        printf 'cd -- %q\n' "$SCRIPT_DIR"
        # Restore caller settings, then let Conda set its own activation variables.
        for name in PATH LD_LIBRARY_PATH PYTHONPATH VIRTUAL_ENV https_proxy http_proxy all_proxy no_proxy; do
            if [[ -v $name ]]; then
                printf 'export %s=%q\n' "$name" "${!name}"
            fi
        done
        printf 'source %q\nconda activate %q\n' "$CONDA_SH" "$CONDA_ENV"
        printf 'GPU_ID=%q\nPYTHON=%q\nMODAL=%q\nMETHOD=%q\nDATE=%q\nFAILED=0\n' \
            "$gpu" "$PYTHON" "$MODAL" "$METHOD" "$DATE"
        printf 'enable_dny_trunc=%q\nval_ens_num=%q\n' \
            "$enable_dny_trunc" "$val_ens_num"
        declare -f train_job
    } > "$worker"

    assigned=$(( total_jobs / gpu_count + (gpu_index < total_jobs % gpu_count) ))
    printf '\nGPU %s: %s jobs\n' "$gpu" "$assigned"
    job_index=0
    for task in "${tasks[@]}"; do
        for seed in "${seeds[@]}"; do
            if (( job_index % gpu_count == gpu_index )); then
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

# Prepare a window per GPU, then start all workers without waiting for training.
tmux new-session -d -s "$SESSION" -n "gpu_${GPU_IDs[0]}" -c "$SCRIPT_DIR" "$BASH"
for gpu_index in "${!GPU_IDs[@]}"; do
    gpu=${GPU_IDs[$gpu_index]}
    window="$SESSION:gpu_$gpu"
    if (( gpu_index > 0 )); then
        tmux new-window -d -t "$SESSION:" -n "gpu_$gpu" -c "$SCRIPT_DIR" "$BASH"
    fi
    tmux set-window-option -t "$window" remain-on-exit on
done
for gpu in "${GPU_IDs[@]}"; do
    tmux respawn-window -k -t "$SESSION:gpu_$gpu" "$BASH" "$WORKER_DIR/gpu_${gpu}.sh"
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
