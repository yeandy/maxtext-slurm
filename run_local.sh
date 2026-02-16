#!/bin/bash

# Run a MaxText training job locally (single node, no Slurm).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_JOB_START=$SECONDS

# ============================================================================
# Interactive mode (no args → shell inside the container)
# ============================================================================

if [[ $# -eq 0 ]]; then
    exec bash "$SCRIPT_DIR/_container.sh"
fi

# ============================================================================
# Preflight warnings (non-fatal; device-agnostic)
# ============================================================================
# No GPU or container cleanup here — run_local.sh must not touch other
# processes, as it could accidentally kill a running job's rank on this node.
# If GPUs are occupied, the training job will fail on its own.
# To manually clean up stale containers, run: bash utils/release_gpu.sh

# Warn if GPUs are occupied (best-effort, does not abort).
if [[ -e /dev/kfd ]] && command -v rocm-smi &>/dev/null; then
    rocm-smi --showpids 2>/dev/null | grep -qE '^[0-9]+' && echo "WARNING: GPUs appear occupied (rocm-smi)." >&2
elif command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q . && echo "WARNING: GPUs appear occupied (nvidia-smi)." >&2
fi

# Warn if on a Slurm cluster without an allocation.
[[ -z "${SLURM_JOB_ID:-}" ]] && command -v sinfo &>/dev/null && \
    echo "WARNING: No Slurm allocation — consider: srun -N 1 --exclusive --pty /bin/bash" >&2

# ============================================================================
# Parse arguments (same format as submit.sh, minus sbatch args)
# ============================================================================

source "$SCRIPT_DIR/utils/split_script_args.sh"
split_script_args "$@"

source "$SCRIPT_DIR/utils/parse_model_spec.sh"
parse_model_spec

source "$SCRIPT_DIR/utils/resolve_model_name.sh"
if ! MODEL_NAME=$(resolve_model_name "$SCRIPT_DIR/configs" "$MODEL_NAME"); then
    echo "!!! Model name resolution failed. Exiting..." >&2
    exit 1
fi

echo "MODEL_NAME=$MODEL_NAME"
[[ -n "$MODEL_NAME_ALIAS" ]] && echo "MODEL_NAME_ALIAS=$MODEL_NAME_ALIAS"
[[ -n "$EXP_TAG" ]] && echo "EXP_TAG=$EXP_TAG"

# parse_model_spec sets SBATCH_ARGS from extra positional args. Local runs
# don't use sbatch, so warn if the user accidentally passed sbatch flags.
if [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
    echo "WARNING: Ignoring sbatch args (${SBATCH_ARGS[*]}) — not applicable to local runs." >&2
    echo "         Did you mean to use submit.sh instead?" >&2
fi

# ============================================================================
# Environment for _container.sh (generic names, no SLURM_* needed)
# ============================================================================

JOB_NAME="JAX-${MODEL_NAME}${EXP_TAG:+-$EXP_TAG}"

export MAXTEXT_SLURM_DIR="$SCRIPT_DIR"
export JAX_COORDINATOR_IP=127.0.0.1
export JOB_ID="local_$(date +%Y%m%d_%H%M%S)_$(printf '%04x' $RANDOM)"
export JOB_NAME
export NNODES=1
export NODE_RANK=0
source "$SCRIPT_DIR/utils/detect_ip.sh"
export LOGIN_NODE_HOSTNAME="${USER}@$(hostname -f 2>/dev/null || hostname)"
export LOGIN_NODE_IP="${USER}@$(detect_ip)"
export JOB_WORKSPACE="${JOB_WORKSPACE:-$SCRIPT_DIR/outputs}"
source "$SCRIPT_DIR/utils/job_dir.sh"
JOB_DIR=$(make_job_dir "$JOB_ID" "$JOB_NAME")

if [[ "${RAY:-0}" == "1" || "${RAY:-}" == "true" ]]; then
    export USE_RAY=true
fi

mkdir -p "$JOB_WORKSPACE/$JOB_DIR"; chmod a+w "$JOB_WORKSPACE/$JOB_DIR"
echo "JOB_NAME=$JOB_NAME  JOB_ID=$JOB_ID  JOB_WORKSPACE=$JOB_WORKSPACE"

# ============================================================================
# Launch via _container.sh
# ============================================================================

source "$SCRIPT_DIR/utils/pick_port.sh"

JAX_PORT=$(pick_free_port)
echo "JAX_COORDINATOR_PORT=$JAX_PORT"

# Pick a random port for Ray head GCS (only when Ray is enabled).
if [[ "${USE_RAY:-false}" == "true" ]]; then
    RAY_PORT=$(pick_free_port "$JAX_PORT")
    export RAY_PORT
    echo "RAY_PORT=$RAY_PORT"
fi

# Log to file (mirrors Slurm's --output) while still printing to terminal.
LOG_FILE="$JOB_WORKSPACE/$JOB_DIR.log"
echo "LOG_FILE=$LOG_FILE"

# Symlink to the job log inside the job directory for easy access.
ln -snf "../$JOB_DIR.log" "$JOB_WORKSPACE/$JOB_DIR/log"

# Write log header in KEY=VALUE format (matches _job.sbatch;
# tgs_tagger.py reads NNODES= from the header).
{
    echo "JOB_ID=$JOB_ID"
    echo "JOB_NAME=$JOB_NAME"
    echo "NNODES=$NNODES"
    echo "NODE_RANK=$NODE_RANK"
    echo "HOSTNAME=$(hostname)"
} | tee "$LOG_FILE"

# Job summary — always printed (success or failure) via EXIT trap.
_fmt_elapsed() {
    local s=$1
    if (( s >= 3600 )); then printf '%dh %dm %ds' $((s/3600)) $((s%3600/60)) $((s%60))
    elif (( s >= 60 )); then printf '%dm %ds' $((s/60)) $((s%60))
    else printf '%ds' "$s"
    fi
}
_print_summary() {
    local rc=${_CONTAINER_RC:-$?}
    local status
    local green red reset
    green=$(tput setaf 2 2>/dev/null) || green=""
    red=$(tput setaf 1 2>/dev/null) || red=""
    reset=$(tput sgr0 2>/dev/null) || reset=""
    if [[ $rc -eq 0 ]]; then
        status="${green}SUCCESS (exit 0)${reset}"
    else
        status="${red}FAILED (exit $rc)${reset}"
    fi
    local summary
    summary="========================== JOB SUMMARY ==========================
  Job:    $JOB_ID ($JOB_NAME)
  Model:  $MODEL_NAME
  Wall:   $(_fmt_elapsed $(( SECONDS - _JOB_START )))
  Status: $status
================================================================="
    echo "$summary"
    [[ -n "${LOG_FILE:-}" ]] && echo "$summary" >> "$LOG_FILE" 2>/dev/null
}
trap _print_summary EXIT

bash "$SCRIPT_DIR/_container.sh" \
    "$JAX_PORT" "$MODEL_NAME" "$EXP_TAG" "$MODEL_NAME_ALIAS" \
    -- "${PASSTHROUGH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
# Capture the container's exit code (not tee's) so _print_summary reports correctly.
_CONTAINER_RC=${PIPESTATUS[0]}
exit "${_CONTAINER_RC:-1}"
