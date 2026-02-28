#!/bin/bash

# Run a MaxText training job inside the container (no Docker launch)
# Use when already inside the training container (interactive shell, docker exec,
# Kubernetes pod, or any environment with JAX pre-installed)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_JOB_START=$SECONDS

# ============================================================================
# Common setup: arg parsing, env, ports, logging, job summary
# ============================================================================

source "$SCRIPT_DIR/utils/run_setup.sh"

# Override the EXIT trap (run_setup.sh installs _print_summary) to include
# Ray cleanup.  Ensures stop_ray_cluster runs even when set -eo pipefail
# aborts before inline cleanup — prevents leaked watchdog/metrics_exporter
# processes that hold NFS file handles.
_cleanup_and_summary() {
    local _trap_rc=${_RUN_RC:-$?}
    if [[ "${USE_RAY:-false}" == "true" ]]; then
        source "$SCRIPT_DIR/utils/ray_cluster.sh" 2>/dev/null || true
        stop_ray_cluster 2>/dev/null || true
    fi
    if declare -f wait_for_coredump &>/dev/null; then
        wait_for_coredump 2>/dev/null || true
    fi
    _RUN_RC=$_trap_rc
    _print_summary
}
trap _cleanup_and_summary EXIT

# ============================================================================
# Container-internal setup (replaces _container.sh's SETUP_CMDS)
# ============================================================================

MAX_NOFILE=$(cat /proc/sys/fs/nr_open 2>/dev/null || echo 1048576)
NOFILE_LIMIT=$((MAX_NOFILE < 1048576 ? MAX_NOFILE : 1048576))
ulimit -Sn "$NOFILE_LIMIT" -Hn "$NOFILE_LIMIT" 2>/dev/null || true

# Core dump setup (best-effort; /coredump may not exist outside the container)
if [[ -f "$SCRIPT_DIR/utils/coredump.sh" ]]; then
    source "$SCRIPT_DIR/utils/coredump.sh"
    setup_coredump "/coredump/core.${JOB_ID:+${JOB_ID}.}%t.${NODE_RANK:+${NODE_RANK}.}%h.%e.%p" 2>/dev/null || true
fi

# Optional pip installs (skip if already present to avoid startup delay)
command -v py-spy &>/dev/null \
    || pip3 install py-spy 2>/dev/null \
    || echo '[WARN] Failed to install py-spy (optional diagnostic tool)'
python3 -c "import google_cloud_mldiagnostics" 2>/dev/null \
    || pip3 install google_cloud_mldiagnostics 2>/dev/null \
    || echo '[WARN] Failed to install google_cloud_mldiagnostics (optional)'

pip list 2>/dev/null | grep jax
ls /opt 2>/dev/null || true

# Resolve MaxText repo directory from container_env.sh (only MAXTEXT_REPO_DIR
# and MAXTEXT_PATCH_BRANCH are relevant; Docker-specific vars are ignored)
source "$SCRIPT_DIR/container_env.sh"
MAXTEXT_REPO_DIR="${MAXTEXT_REPO_DIR:-/workspace/maxtext}"
cd "$MAXTEXT_REPO_DIR"

if [[ -n "${MAXTEXT_PATCH_BRANCH:-}" ]]; then
    echo "[INFO] Checking out $MAXTEXT_PATCH_BRANCH..."
    if git fetch origin "$MAXTEXT_PATCH_BRANCH" && git checkout "origin/$MAXTEXT_PATCH_BRANCH"; then
        echo "[OK] Checked out $MAXTEXT_PATCH_BRANCH in the local maxtext repo."
    else
        echo "[FAIL] Failed to check out $MAXTEXT_PATCH_BRANCH in the local maxtext repo." >&2
        exit 1
    fi
else
    echo "[SKIP] No MAXTEXT_PATCH_BRANCH set, using image default."
fi

# ============================================================================
# Launch training
# ============================================================================

# Use Ray-enabled script if USE_RAY=true
if [[ "${USE_RAY:-false}" == "true" ]]; then
    MAXTEXT_RUNNER="$SCRIPT_DIR/_train_with_ray.sh"
else
    MAXTEXT_RUNNER="$SCRIPT_DIR/_train.sh"
fi

"$MAXTEXT_RUNNER" "$MODEL_NAME" -- "${PASSTHROUGH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
# Capture training exit code (not tee's) so _print_summary reports correctly
_RUN_RC=${PIPESTATUS[0]}

exit "${_RUN_RC:-1}"
