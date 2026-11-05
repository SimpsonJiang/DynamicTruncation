#!/bin/bash
set -euo pipefail

# ==== Settings ====
METHOD_TEMPLATE='VE_{metric}_symlog_w{trunc_warmup_steps/1000}k_f64_H{horizon}' # Placeholders expand per job.
# Maximum concurrent training jobs per GPU: [GPU index]=job limit.
declare -A GPU_MAX_JOBS=(
    [6]=4
    [7]=5
)

MODAL=proprio # vision/proprio
imag_horis=(
    15
)

enable_dny_trunc=true
trunc_warmup_steps=50000 # trainer Step; 0 enables truncation immediately
distance_space=symlog # raw / symlog; affects wasser / cramer / sqrt_cramer
uncer_metrics=(
    wasser
)
# wasser / ucb / sigma / cramer / sqrt_cramer

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


DATE=$(date +%m%d) # auto complete
UV_ACTIVATE=/mnt/phx2/DynamicTruncation/.venv/bin/activate
# ==== Loop ====
command -v tmux >/dev/null || { echo "Error: tmux is required." >&2; exit 1; }

source "$UV_ACTIVATE"
PYTHON=$(command -v python) || { echo "Error: python is required." >&2; exit 1; }
printf 'uv environment: %s (Python: %s)\n' "$UV_ACTIVATE" "$PYTHON"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Keep compilation artifacts and temporary files off a potentially full /tmp.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRIPT_DIR/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRIPT_DIR/.cache/triton}"
export TMPDIR="${TMPDIR:-$SCRIPT_DIR/.cache/tmp}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"
printf 'Inductor cache: %s\nTriton cache: %s\nTemporary directory: %s\n' \
    "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

if (( ${#GPU_MAX_JOBS[@]} == 0 || ${#tasks[@]} == 0 || ${#seeds[@]} == 0 \
    || ${#uncer_metrics[@]} == 0 || ${#imag_horis[@]} == 0 )); then
    echo "Error: GPU_MAX_JOBS, tasks, seeds, uncer_metrics and imag_horis must all be non-empty." >&2
    exit 1
fi
total_capacity=0
for gpu in "${!GPU_MAX_JOBS[@]}"; do
    if [[ ! $gpu =~ ^[0-9]+$ ]]; then
        echo "Error: GPU_MAX_JOBS keys must be non-negative GPU indices: $gpu" >&2
        exit 1
    fi
    if [[ ! ${GPU_MAX_JOBS[$gpu]} =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: GPU $gpu job limit must be a positive integer: ${GPU_MAX_JOBS[$gpu]}" >&2
        exit 1
    fi
    total_capacity=$((total_capacity + GPU_MAX_JOBS[$gpu]))
done
# Associative arrays are unordered; use numeric GPU order for reproducible assignment.
mapfile -t GPU_IDs < <(printf '%s\n' "${!GPU_MAX_JOBS[@]}" | sort -n)

# Each generated worker runs its assigned jobs sequentially inside tmux.
train_job() {
    local task=$1 seed=$2 uncer_metric=$3 imag_hori=$4 status
    local method=${METHOD_TEMPLATE//\{metric\}/$uncer_metric}
    method=${method//\{horizon\}/$imag_hori}
    method=${method//\{trunc_warmup_steps\/1000\}/$((trunc_warmup_steps / 1000))}
    if [[ -z ${TMUX:-} ]]; then
        echo "Error: training must run inside tmux." >&2
        exit 1
    fi
    printf '\n[GPU %s] Starting task=%s seed=%s uncer_metric=%s imag_hori=%s\n' \
        "$GPU_ID" "$task" "$seed" "$uncer_metric" "$imag_hori"
# -------------------------------python cmd-----------------------------
    if CUDA_VISIBLE_DEVICES="$GPU_ID" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU_ID" \
        "$PYTHON" train.py \
            "env=dmc_${MODAL}" \
            "env.task=$task" \
            "logdir=logdir/${method}/${task}_${MODAL}_s${seed}_${DATE}" \
            model.compile=True \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            model.rep_loss=dreamer \
            "model.dynamic_trunc.enabled=${enable_dny_trunc}" \
            "model.dynamic_trunc.warmup_steps=${trunc_warmup_steps}" \
            "model.uncer_metric=${uncer_metric}" \
            "model.distance_space=${distance_space}" \
            "model.imag_horizon=${imag_hori}" \
            "seed=$seed"; then
        printf '[GPU %s] Finished task=%s seed=%s uncer_metric=%s imag_hori=%s\n' \
            "$GPU_ID" "$task" "$seed" "$uncer_metric" "$imag_hori"
    else
        status=$?
        printf '[GPU %s] FAILED task=%s seed=%s uncer_metric=%s imag_hori=%s (exit %s)\n' \
            "$GPU_ID" "$task" "$seed" "$uncer_metric" "$imag_hori" "$status" >&2
        FAILED=1
    fi
}

WORKER_DIR=$(mktemp -d "${TMPDIR%/}/r2dreamer-workers.XXXXXX")
SESSION="r2dreamer_${DATE}_$(date +%H%M%S)_$$"
gpu_count=${#GPU_IDs[@]}
total_jobs=$(( ${#tasks[@]} * ${#seeds[@]} * ${#uncer_metrics[@]} * ${#imag_horis[@]} ))
worker_count=$total_capacity
if (( worker_count > total_jobs )); then
    worker_count=$total_jobs
fi
worker_names=()
worker_gpus=()
worker_slots=()
# Fill one slot per GPU per round, skipping GPUs that have reached their limit.
for ((slot = 1; ${#worker_gpus[@]} < worker_count; slot++)); do
    for gpu in "${GPU_IDs[@]}"; do
        if (( slot <= GPU_MAX_JOBS[$gpu] )); then
            worker_gpus+=("$gpu")
            worker_slots+=("$slot")
            if (( ${#worker_gpus[@]} == worker_count )); then
                break
            fi
        fi
    done
done

printf 'Distributing %s jobs across %s GPUs (%s workers):\n' \
    "$total_jobs" "$gpu_count" "$worker_count"
for gpu in "${GPU_IDs[@]}"; do
    printf '  GPU %s: at most %s concurrent jobs\n' "$gpu" "${GPU_MAX_JOBS[$gpu]}"
done
for ((worker_index = 0; worker_index < worker_count; worker_index++)); do
    gpu=${worker_gpus[$worker_index]}
    slot=${worker_slots[$worker_index]}
    worker_name="gpu_${gpu}"
    if (( GPU_MAX_JOBS[$gpu] > 1 )); then
        worker_name+="_worker_${slot}"
    fi
    worker_names+=("$worker_name")
    worker="$WORKER_DIR/${worker_name}.sh"
    {
        printf '#!/bin/bash\nset -euo pipefail\n'
        printf 'cd -- %q\n' "$SCRIPT_DIR"
        # Restore caller settings, then activate the uv environment in this worker.
        for name in PATH LD_LIBRARY_PATH PYTHONPATH VIRTUAL_ENV https_proxy http_proxy all_proxy no_proxy \
            TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TMPDIR; do
            if [[ -v $name ]]; then
                printf 'export %s=%q\n' "$name" "${!name}"
            fi
        done
        printf 'source %q\n' "$UV_ACTIVATE"
        printf 'GPU_ID=%q\nPYTHON=%q\nMODAL=%q\nDATE=%q\nFAILED=0\n' \
            "$gpu" "$PYTHON" "$MODAL" "$DATE"
        printf 'enable_dny_trunc=%q\n' "$enable_dny_trunc"
        printf 'trunc_warmup_steps=%q\n' "$trunc_warmup_steps"
        printf 'distance_space=%q\n' "$distance_space"
        printf 'METHOD_TEMPLATE=%q\n' "$METHOD_TEMPLATE"
        declare -f train_job
    } > "$worker"

    assigned=$(( total_jobs / worker_count + (worker_index < total_jobs % worker_count) ))
    printf '\nGPU %s, worker %s: %s jobs\n' "$gpu" "$slot" "$assigned"
    job_index=0
    for task in "${tasks[@]}"; do
        for seed in "${seeds[@]}"; do
            for uncer_metric in "${uncer_metrics[@]}"; do
                for imag_hori in "${imag_horis[@]}"; do
                    if (( job_index % worker_count == worker_index )); then
                        printf '  task=%s seed=%s uncer_metric=%s imag_hori=%s\n' \
                            "$task" "$seed" "$uncer_metric" "$imag_hori"
                        printf 'train_job %q %q %q %q\n' \
                            "$task" "$seed" "$uncer_metric" "$imag_hori" >> "$worker"
                    fi
                    job_index=$((job_index + 1))
                done
            done
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
