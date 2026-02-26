#!/bin/bash

# Run a MaxText training job locally (single node, no Slurm)
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
# Preflight (host-specific)
# ============================================================================

# Warn if on a Slurm cluster without an allocation
[[ -z "${SLURM_JOB_ID:-}" ]] && command -v sinfo &>/dev/null && \
    echo "WARNING: No Slurm allocation — consider: srun -N 1 --exclusive --pty /bin/bash" >&2

# ============================================================================
# Common setup: arg parsing, env, ports, logging, job summary
# ============================================================================

source "$SCRIPT_DIR/utils/run_setup.sh"

# ============================================================================
# Launch via _container.sh
# ============================================================================

bash "$SCRIPT_DIR/_container.sh" \
    "$JAX_PORT" "$MODEL_NAME" "$EXP_TAG" "$MODEL_NAME_ALIAS" \
    -- "${PASSTHROUGH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
# Capture the container's exit code (not tee's) so _print_summary reports correctly
_RUN_RC=${PIPESTATUS[0]}
exit "${_RUN_RC:-1}"
