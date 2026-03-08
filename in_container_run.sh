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

# Mirror all subsequent stdout/stderr to the job log so setup/runtime messages
# are captured consistently (not only training output).
if [[ -z "${_IN_CONTAINER_LOG_TEE_ACTIVE:-}" ]]; then
    export _IN_CONTAINER_LOG_TEE_ACTIVE=1
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1
fi

# ============================================================================
# Container-internal setup (replaces _container.sh's SETUP_CMDS)
# ============================================================================

MAX_NOFILE=$(cat /proc/sys/fs/nr_open 2>/dev/null || echo 1048576)
NOFILE_LIMIT=$((MAX_NOFILE < 1048576 ? MAX_NOFILE : 1048576))
ulimit -Sn "$NOFILE_LIMIT" -Hn "$NOFILE_LIMIT" 2>/dev/null || true

# Core dump setup (best-effort; /coredump may not exist outside the container)
if [[ -f "$SCRIPT_DIR/utils/coredump.sh" ]]; then
    # In k8s/direct-container runs, /coredump may not be mounted/writable.
    # Keep caller-provided COREDUMP_DIR when valid; otherwise choose a fallback.
    if [[ -z "${COREDUMP_DIR:-}" || ! -d "${COREDUMP_DIR:-}" || ! -w "${COREDUMP_DIR:-}" ]]; then
        for _dir in /coredump "${JOB_WORKSPACE:-}" /tmp; do
            [[ -n "$_dir" ]] || continue
            if [[ -d "$_dir" && -w "$_dir" ]]; then
                COREDUMP_DIR="$_dir"
                break
            fi
        done
        export COREDUMP_DIR
    fi

    source "$SCRIPT_DIR/utils/coredump.sh"
    setup_coredump "${COREDUMP_DIR:-/tmp}/core.${JOB_ID:+${JOB_ID}.}%t.${NODE_RANK:+${NODE_RANK}.}%h.%e.%p" 2>/dev/null || true
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
[[ -n "${MAXTEXT_PATCH_BRANCH:-}" ]] && export MAXTEXT_PATCH_BRANCH
cd "$MAXTEXT_REPO_DIR"

# ============================================================================
# Launch training
# ============================================================================

# Use Ray-enabled script if USE_RAY=true
if [[ "${USE_RAY:-false}" == "true" ]]; then
    MAXTEXT_RUNNER="$SCRIPT_DIR/_train_with_ray.sh"
else
    MAXTEXT_RUNNER="$SCRIPT_DIR/_train.sh"
fi

set +e
"$MAXTEXT_RUNNER" "$MODEL_NAME" -- "${PASSTHROUGH_ARGS[@]}"
_RUN_RC=$?
set -e

exit "${_RUN_RC:-1}"
