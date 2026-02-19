#!/bin/bash

# Resolve script directory (MAXTEXT_SLURM_DIR is set by submit.sh;
# falls back to BASH_SOURCE for standalone use, e.g. interactive debugging)
SCRIPT_DIR="${MAXTEXT_SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DOCKER_SCRIPT_DIR="/$(basename "$SCRIPT_DIR")"

# ==== Container config (edit container_env.sh to switch images/paths) ====
source "$SCRIPT_DIR/container_env.sh"

# Map user-facing RAY flag to internal USE_RAY.
# For script mode, run_setup.sh already did this; for interactive mode
# (run_local.sh with no args), run_setup.sh is skipped so we do it here.
if [[ "${RAY:-0}" == "1" || "${RAY:-}" == "true" ]]; then
    USE_RAY=true
fi

if [[ "$DOCKER_IMAGE_HAS_AINIC" == "true" ]]; then
    echo "AINIC-enabled image detected: $DOCKER_IMAGE"
else
    echo "Standard image (no AINIC): $DOCKER_IMAGE"
fi

# ============================================================================
# Required environment (callers must set these, or SLURM_* fallbacks are used)
# ============================================================================
JAX_COORDINATOR_IP="${JAX_COORDINATOR_IP:-${SLURM_LAUNCH_NODE_IPADDR:-}}"
JOB_ID="${JOB_ID:-${SLURM_JOB_ID:-${SLURM_JOBID:-unknown}}}"
JOB_NAME="${JOB_NAME:-${SLURM_JOB_NAME:-unknown}}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
LOGIN_NODE_HOSTNAME="${LOGIN_NODE_HOSTNAME:-${USER}@${SLURM_SUBMIT_HOST:-$(hostname -s)}}"
LOGIN_NODE_IP="${LOGIN_NODE_IP:-${USER}@${SLURM_SUBMIT_HOST:-$(hostname -s)}}"

# Optional: comma-separated expanded node list (for Prometheus scrape targets)
if [[ -z "${NODELIST_EXPANDED:-}" && -n "${SLURM_JOB_NODELIST:-}" ]]; then
    NODELIST_EXPANDED=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
fi
NODELIST_EXPANDED="${NODELIST_EXPANDED:-}"

# Derived paths (job directory naming defined in utils/job_dir.sh).
OUTPUTS_DIR="${JOB_WORKSPACE:-$SCRIPT_DIR/outputs}"
source "$SCRIPT_DIR/utils/job_dir.sh"
JOB_DIR=$(make_job_dir "$JOB_ID" "$JOB_NAME")

# Paths configured in container_env.sh:
#   MAXTEXT_REPO_DIR, DATASET_DIR, COREDUMP_EXTRA_DIRS

# NOTE: All dataset-related mount options go here!
if [[ -d "$DATASET_DIR" ]]; then
    echo "Dataset directory found, mounting read-only."
    DATASET_MOUNT_OPTIONS=(
        -v "$DATASET_DIR:/datasets:ro"
    )
else
    echo "Dataset directory not found, running without datasets."
    DATASET_MOUNT_OPTIONS=()
fi

HOSTNAME=$(hostname -s)

# Ensure the outputs root exists before the coredump-mount check.
# (In script mode the caller pre-creates $OUTPUTS_DIR/$JOB_DIR; in interactive
# mode only $OUTPUTS_DIR itself exists — used as a fallback below.)
mkdir -p "$OUTPUTS_DIR"; chmod a+w "$OUTPUTS_DIR"

# NOTE: mount directory for core dumps (requires >500GB free space)!
# $OUTPUTS_DIR/$JOB_DIR is preferred (created by run_local.sh / submit.sh).
# $OUTPUTS_DIR is a fallback so interactive mode still gets a persistent mount.
COREDUMP_DIR_CANDIDATES=(
    "$OUTPUTS_DIR/$JOB_DIR"
    "$OUTPUTS_DIR"
    "${COREDUMP_EXTRA_DIRS[@]}"
)
COREDUMP_MOUNT_OPTIONS=()
for dir in "${COREDUMP_DIR_CANDIDATES[@]}"; do
    if [ -d "$dir" ] && \
       avail=$(df -BG "$dir" | awk 'NR==2 {print $4}' | sed 's/G//') && \
       [[ "$avail" =~ ^[0-9]+$ ]] && [ "$avail" -gt 500 ]; then
        echo "$HOSTNAME: COREDUMP_DIR=$dir"
        COREDUMP_MOUNT_OPTIONS=(-v "$dir:/coredump")
        break
    fi
done

NUM_ARGS=$#
# Split args first
source "$SCRIPT_DIR/utils/split_script_args.sh"
split_script_args "$@"
echo "PASSTHROUGH_ARGS=\"${PASSTHROUGH_ARGS[*]}\""

# Parse from SCRIPT_ARGS
JAX_COORDINATOR_PORT=${SCRIPT_ARGS[0]:-0}
MODEL_NAME=${SCRIPT_ARGS[1]:-}
# SCRIPT_ARGS[2] is EXP_TAG — passed through but unused here.
MODEL_NAME_ALIAS=${SCRIPT_ARGS[3]:-}

# ==== Determine execution mode ====
if [[ "$JAX_COORDINATOR_PORT" == "pull-only" ]]; then
    MODE="pull-only"
elif [ $NUM_ARGS -eq 0 ]; then
    MODE="interactive"
else
    MODE="script"
fi

echo "[INFO] Execution mode: $MODE"

# Determine safe nofile limit
MAX_NOFILE=$(cat /proc/sys/fs/nr_open 2>/dev/null || echo 1048576)
NOFILE_LIMIT=$((MAX_NOFILE < 1048576 ? MAX_NOFILE : 1048576))
echo "NOFILE_LIMIT=$NOFILE_LIMIT"

# Commands that should always run inside the container (interactive or not),
# *before* running MaxText.
SETUP_CMDS="
    ulimit -Sn $NOFILE_LIMIT -Hn $NOFILE_LIMIT

    # Core dump setup: pattern, ulimit, SIGTERM trap, wait-for-write helper.
    source $DOCKER_SCRIPT_DIR/utils/coredump.sh
    setup_coredump \"/coredump/core.${JOB_ID:+${JOB_ID}.}%t.${NODE_RANK:+${NODE_RANK}.}%h.%e.%p\"

    pip3 install py-spy || echo '[WARN] Failed to install py-spy (optional diagnostic tool)'
    pip3 install google_cloud_mldiagnostics || echo '[WARN] Failed to install google_cloud_mldiagnostics (optional)'

    pip list | grep jax
    ls /opt

    cd \"$MAXTEXT_REPO_DIR\"

    if [[ -n \"${MAXTEXT_PATCH_BRANCH:-}\" ]]; then
        echo \"[INFO] Checking out $MAXTEXT_PATCH_BRANCH...\"
        if git fetch origin \"$MAXTEXT_PATCH_BRANCH\" && git checkout \"origin/$MAXTEXT_PATCH_BRANCH\"; then
            echo \"[OK] Checked out $MAXTEXT_PATCH_BRANCH in the local maxtext repo.\"
        else
            echo \"[FAIL] Failed to check out $MAXTEXT_PATCH_BRANCH in the local maxtext repo.\" >&2
            exit 1
        fi
    else
        echo \"[SKIP] No MAXTEXT_PATCH_BRANCH set, using image default.\"
    fi
"

# ==== Configure based on mode ====
case "$MODE" in
    interactive)
        INTERACTIVE_TTY=(-it)
        FINAL_CMD="$SETUP_CMDS
            sudo apt-get update
            sudo apt-get install -y gdb vim

            # Fallback: if /coredump isn't mounted, use /tmp so interactive
            # debugging can still capture core dumps (ephemeral, lost on exit).
            if [[ ! -d /coredump ]]; then
                export COREDUMP_DIR=/tmp
                setup_coredump
            else
                export COREDUMP_DIR  # propagate to the exec bash shell below
            fi

            # NOTE: add useful utility commands to bash history!
            echo '$DOCKER_SCRIPT_DIR/debug_repro.sh' >> ~/.bash_history
            echo 'gdb python3 \"\$(ls -t \$COREDUMP_DIR/core*py* | head -n1)\"' >> ~/.bash_history

            echo 'Setup done. Entering interactive shell...'
            exec bash"
        ;;
    script)
        INTERACTIVE_TTY=()
        # Use Ray-enabled script if USE_RAY=true
        if [[ "${USE_RAY:-false}" == "true" ]]; then
            MAXTEXT_RUNNER="$DOCKER_SCRIPT_DIR/_train_with_ray.sh"
        else
            MAXTEXT_RUNNER="$DOCKER_SCRIPT_DIR/_train.sh"
        fi
        FINAL_CMD="$SETUP_CMDS
            $MAXTEXT_RUNNER '$MODEL_NAME' -- $(printf '%q ' "${PASSTHROUGH_ARGS[@]}") 2>&1
            _train_rc=\$?
            wait_for_coredump
            exit \$_train_rc"
        ;;
    pull-only)
        # Will be handled later, no container commands needed
        INTERACTIVE_TTY=()
        FINAL_CMD=""
        ;;
esac

# ==== Single merged DOCKER_ARGS definition ====
DOCKER_ARGS=(/bin/bash -lcx "$FINAL_CMD")

source "$SCRIPT_DIR/utils/docker_utils.sh"
DOCKER_BIN="$(get_docker_bin)" || exit 1
read -ra DOCKER_CMD <<< "$DOCKER_BIN"
echo "DOCKER_BIN=$DOCKER_BIN"

# Resolve the image to run: either a local tarball or a registry image
IMAGE_TO_RUN="$DOCKER_IMAGE"

# Detect local tarball: absolute/relative path and file exists
if [[ -f "$DOCKER_IMAGE" ]] && [[ "$DOCKER_IMAGE" == *.tar ]]; then
    echo "[INFO] Detected local tarball: $DOCKER_IMAGE"
    echo "[INFO] Loading image from tarball..."
    LOAD_OUTPUT="$("${DOCKER_CMD[@]}" load -i "$DOCKER_IMAGE" 2>&1)"
    echo "$LOAD_OUTPUT"

    # Extract the last loaded image reference if present
    # Docker outputs "Loaded image: ...", podman may output "Loaded image(s): ..."
    IMAGE_REF="$(echo "$LOAD_OUTPUT" | grep -E 'Loaded image(\(s\))?:' | tail -n1 | sed -E 's/^Loaded image(\(s\))?:\s+//')"
    if [[ -n "$IMAGE_REF" ]]; then
        IMAGE_TO_RUN="$IMAGE_REF"
        echo "[INFO] Using loaded image: $IMAGE_TO_RUN"
    else
        # Fallback to image ID if available
        IMAGE_ID="$(echo "$LOAD_OUTPUT" | grep -E 'Loaded image ID:' | tail -n1 | sed -E 's/^Loaded image ID:\s+//')"
        if [[ -n "$IMAGE_ID" ]]; then
            IMAGE_TO_RUN="$IMAGE_ID"
            echo "[INFO] Using loaded image ID: $IMAGE_TO_RUN"
        else
            echo "[ERROR] Unable to determine image name or ID from docker load output."
            echo "[ERROR] Load output was:"
            echo "$LOAD_OUTPUT"
            exit 1
        fi
    fi
else
    # Not a local tarball; proceed with original inspect/pull logic on DOCKER_IMAGE
    if ! "${DOCKER_CMD[@]}" image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        echo "[INFO] Image not found locally. Freeing up space and pulling $DOCKER_IMAGE ..."
        docker_smart_prune "$DOCKER_BIN" 120  # 120GB

        # Try anonymous pull first (works for public images).
        if ! "${DOCKER_CMD[@]}" pull "$DOCKER_REGISTRY/$DOCKER_IMAGE" 2>/dev/null; then
            # Anonymous pull failed — attempt authenticated pull
            # if credentials are configured.
            if [[ -n "${DOCKER_TOKEN:-}" && -n "${DOCKER_USERNAME:-}" ]]; then
                echo "[INFO] Anonymous pull failed. Logging in to $DOCKER_REGISTRY as $DOCKER_USERNAME ..."
                if ! echo "$DOCKER_TOKEN" | "${DOCKER_CMD[@]}" login -u "$DOCKER_USERNAME" --password-stdin "$DOCKER_REGISTRY"; then
                    echo "[ERROR] Login to $DOCKER_REGISTRY failed. Check DOCKER_USERNAME/DOCKER_TOKEN" >&2
                    echo "        in container_env.local.sh." >&2
                    exit 1
                fi
                if ! "${DOCKER_CMD[@]}" pull "$DOCKER_REGISTRY/$DOCKER_IMAGE"; then
                    echo "[ERROR] Authenticated pull failed for $DOCKER_IMAGE." >&2
                    echo "        Verify the image name in container_env.sh and" >&2
                    echo "        credentials in container_env.local.sh." >&2
                    exit 1
                fi
            else
                echo "[ERROR] Pull failed for $DOCKER_IMAGE." >&2
                echo "        For private images, run: $DOCKER_BIN login $DOCKER_REGISTRY" >&2
                exit 1
            fi
        fi
    else
        echo "[INFO] Image already exists locally. Skipping pull of $DOCKER_IMAGE ."
    fi
fi

# ----------------------------
# Docker pull-only mode: exit early
# ----------------------------
if [[ "$MODE" == "pull-only" ]]; then
    echo "[INFO] Docker pull-only mode requested. Image is ready on this node: $IMAGE_TO_RUN"
    exit 0
fi

# ---- Container naming & cleanup on cancellation ----
# Unique name lets us 'docker stop' the container when the Slurm job is
# cancelled (scancel).  Without this, cancelling kills srun/bash but the
# Docker container (managed by dockerd) keeps running until the next job.
CONTAINER_NAME="maxtext-slurm-${JOB_ID}-node${NODE_RANK}"
# Interactive runs all share JOB_ID=unknown; append PID to avoid collisions.
[[ "$JOB_ID" == "unknown" ]] && CONTAINER_NAME+="-$$"
# Remove any leftover container with the same name (e.g., from a prior SIGKILL).
"${DOCKER_CMD[@]}" rm -f "$CONTAINER_NAME" 2>/dev/null || true

# NOTE: All IB/ANP-related mount options go here!
if [[ "$DOCKER_IMAGE_HAS_AINIC" == "true" ]] || [[ "$MODE" == "interactive" ]]; then
    IB_MOUNT_OPTIONS=(
        # NOTE: has no effect unless ANP is installed in the container
#        -e NCCL_NET_PLUGIN=librccl-anp.so
    )
else
    # Detect and configure host IB-related mounts (bnxt_re driver present on host)
    if [[ -e "/etc/libibverbs.d/bnxt_re.driver" ]]; then
        echo "Detected bnxt_re driver on host: enabling /etc/libibverbs.d mounts."
        IB_MOUNT_OPTIONS=(
            -v /etc/libibverbs.d:/etc/libibverbs.d:ro
            -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro
        )
    else
        echo "No /etc/libibverbs.d/bnxt_re.driver found: disabling IB mounts."
        IB_MOUNT_OPTIONS=()
    fi
fi

# InfiniBand device passthrough and HCA detection (safe if IB is absent)
IB_DEVICE_OPTIONS=()
NCCL_IB_HCA=""
if [[ -d /sys/class/infiniband ]]; then
    NCCL_IB_HCA=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ',' | sed 's/,$//')
fi
if [[ -e /dev/infiniband ]]; then
    IB_DEVICE_OPTIONS=(--device /dev/infiniband)
fi
source "$SCRIPT_DIR/utils/choose_nccl_socket_ifname.sh"
if nccl_nic=$(choose_nccl_socket_ifname); then
    NCCL_SOCKET_IFNAME="${nccl_nic}"
    echo "NCCL INFO $HOSTNAME: NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
else
    # Handle detection failure based on execution mode
    if [[ "$MODE" == "script" && "$NNODES" -gt 1 ]]; then
        # Multi-node script mode: fail fast (cross-node NCCL needs a socket interface)
        echo "NCCL FATAL $HOSTNAME: Failed to auto-detect NCCL_SOCKET_IFNAME; ABORTING..." >&2
        exit 1
    else
        # Single-node or interactive mode: warn and continue
        echo "NCCL WARN $HOSTNAME: Could not auto-detect NCCL_SOCKET_IFNAME; leaving it unset" >&2
    fi
fi

# Only pass NCCL network vars to the container when they have a value;
# an empty --env KEY= would override NCCL's internal auto-detection.
NCCL_ENV_ARGS=()
[[ -n "$NCCL_IB_HCA" ]] && NCCL_ENV_ARGS+=(--env "NCCL_IB_HCA=$NCCL_IB_HCA")
[[ -n "${NCCL_SOCKET_IFNAME:-}" ]] && NCCL_ENV_ARGS+=(--env "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")

# Print code provenance for this run.
if [[ -f "$SCRIPT_DIR/git_summary.txt" ]]; then
    echo "[INFO] Running from artifact: $SCRIPT_DIR"
else
    (cd "$SCRIPT_DIR" && bash "$SCRIPT_DIR/utils/git_summary.sh")
fi

REPO_MOUNT_OPTIONS=()
if [[ "$MODE" == "interactive" && ! -d "$SCRIPT_DIR/.git" ]]; then
    # SCRIPT_DIR is a subdirectory of a larger repo — mount the parent for browsing.
    # When SCRIPT_DIR is the repo root (has .git/), the main -v mount below
    # already maps the entire repo into the container.
    REPO_MOUNT_OPTIONS=(-v "$(dirname "$SCRIPT_DIR")":"${DOCKER_SCRIPT_DIR}_repo")
fi

# ---- GPU device auto-detection (device-agnostic: AMD or NVIDIA) ----
if [[ -e /dev/kfd ]]; then
    GPU_DEVICE_ARGS=(--device=/dev/kfd --device=/dev/dri)
elif command -v nvidia-smi &>/dev/null; then
    # Docker uses --gpus; podman uses CDI device notation
    if [[ "${DOCKER_CMD[*]}" == *podman* ]]; then
        GPU_DEVICE_ARGS=(--device nvidia.com/gpu=all)
    else
        GPU_DEVICE_ARGS=(--gpus all)
    fi
else
    GPU_DEVICE_ARGS=()
    echo "WARNING: No GPU devices detected (no /dev/kfd, no nvidia-smi)" >&2
fi

echo "==STARTING DOCKER CONTAINER== ($CONTAINER_NAME)"

# $OUTPUTS_DIR was already created above (before the coredump-mount check).

# Preserve the caller's supplementary groups inside the container.
GROUP_ADD_ARGS=()
for gid in $(id -G); do GROUP_ADD_ARGS+=(--group-add "$gid"); done

# ---- Cancellation cleanup (scancel / SIGTERM) ----
# EXIT trap: safety net — stops the container if it's still running.
# TERM/INT traps (script mode only): respond to scancel immediately via
# background 'wait'; interactive mode leaves signals to docker run.
_CONTAINER_STOPPED=0
_cleanup_container() {
    [[ "$_CONTAINER_STOPPED" -eq 1 ]] && return
    _CONTAINER_STOPPED=1
    echo "[INFO] $HOSTNAME: Cleaning up container $CONTAINER_NAME ..."
    bash "$SCRIPT_DIR/utils/release_gpu.sh" --container "$CONTAINER_NAME"
}
trap _cleanup_container EXIT

# ---- Env vars passed to the container ----
# Training-specific vars are only needed in script mode.  In interactive mode,
# in_container_run.sh → run_setup.sh computes them from scratch.  Passing
# sentinel values (JAX_COORDINATOR_PORT=0, JOB_ID=unknown) would just force
# run_setup.sh to detect and unset them.
TRAIN_ENV_ARGS=()
if [[ "$MODE" == "script" ]]; then
    TRAIN_ENV_ARGS=(
        --env "JAX_COORDINATOR_IP=$JAX_COORDINATOR_IP"
        --env "JAX_COORDINATOR_PORT=$JAX_COORDINATOR_PORT"
        --env "JOB_DIR=$JOB_DIR"
        --env "MODEL_NAME_ALIAS=${MODEL_NAME_ALIAS}"
        --env "LOGIN_NODE_HOSTNAME=$LOGIN_NODE_HOSTNAME"
        --env "LOGIN_NODE_IP=$LOGIN_NODE_IP"
        --env "NODELIST_EXPANDED=$NODELIST_EXPANDED"
    )
fi

# Only pass Ray env vars when Ray is enabled.
# Pass RAY (user-level flag), not USE_RAY (internal) — so RAY=0 at command
# level inside the container can always override.
RAY_ENV_ARGS=()
if [[ "${USE_RAY:-false}" == "true" ]]; then
    RAY_ENV_ARGS=(--env "RAY=1")
    [[ -n "${RAY_PORT:-}" ]] && RAY_ENV_ARGS+=(--env "RAY_PORT=$RAY_PORT")
fi

DOCKER_RUN_ARGS=(
    --rm
    --name "$CONTAINER_NAME"
    --label maxtext_slurm=1
    --ulimit "nofile=$NOFILE_LIMIT:$NOFILE_LIMIT"
    --cap-add=SYS_PTRACE
    --ipc=host
    --network=host
    --privileged
    "${GPU_DEVICE_ARGS[@]}"
    "${IB_DEVICE_OPTIONS[@]}"
    "${TRAIN_ENV_ARGS[@]}"
    "${RAY_ENV_ARGS[@]}"
    "${NCCL_ENV_ARGS[@]}"
    --env "NNODES=$NNODES"
    --env "NODE_RANK=$NODE_RANK"
    --env "JOB_ID=$JOB_ID"
    "${GROUP_ADD_ARGS[@]}"
    "${IB_MOUNT_OPTIONS[@]}"
    "${DATASET_MOUNT_OPTIONS[@]}"
    "${COREDUMP_MOUNT_OPTIONS[@]}"
    "${REPO_MOUNT_OPTIONS[@]}"
    -v /boot:/boot:ro
    -v "$SCRIPT_DIR":"$DOCKER_SCRIPT_DIR"
    -v "$OUTPUTS_DIR":/outputs
    -w "$DOCKER_SCRIPT_DIR"
    "${INTERACTIVE_TTY[@]}"
    "$IMAGE_TO_RUN"
    "${DOCKER_ARGS[@]}"
)

if [[ "$MODE" == "script" ]]; then
    # TERM/INT + background/wait: bash can only run traps while 'wait' is the
    # current command.  A foreground docker-run blocks trap delivery, so scancel's
    # SIGTERM would never trigger container cleanup before Slurm escalates to SIGKILL.
    trap '_cleanup_container; exit 143' TERM
    trap '_cleanup_container; exit 130' INT
    "${DOCKER_CMD[@]}" run "${DOCKER_RUN_ARGS[@]}" &
    wait $!
else
    "${DOCKER_CMD[@]}" run "${DOCKER_RUN_ARGS[@]}"
fi
_rc=$?
_CONTAINER_STOPPED=1  # Container exited normally; --rm already cleaned up.
exit "$_rc"
