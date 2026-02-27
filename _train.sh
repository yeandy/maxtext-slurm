#!/bin/bash

# Launch MaxText training.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running _train.sh"
echo "args: $@"

# Split args first
source "$SCRIPT_DIR/utils/split_script_args.sh"
split_script_args "$@"
echo "PASSTHROUGH_ARGS=\"${PASSTHROUGH_ARGS[*]}\""

# Extract environment variables from PASSTHROUGH_ARGS (pattern: _env_KEY=VALUE).
# Values may arrive with surrounding quotes when users write _env_KEY='"value"'
# — the shell keeps the inner quotes, printf '%q' in _container.sh preserves them,
# and the inner bash reconstructs them. Strip one layer of matching outer quotes.
EXTRACTED_ENVS=()
FILTERED_PASSTHROUGH_ARGS=()
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [[ "$arg" =~ ^_env_([^=]+)=(.*)$ ]]; then
        env_key="${BASH_REMATCH[1]}"
        env_value="${BASH_REMATCH[2]}"
        # Strip one layer of matching surrounding quotes (double or single).
        if [[ "$env_value" =~ ^\"(.*)\"$ ]]; then
            env_value="${BASH_REMATCH[1]}"
        elif [[ "$env_value" =~ ^\'(.*)\'$ ]]; then
            env_value="${BASH_REMATCH[1]}"
        fi
        EXTRACTED_ENVS+=("$env_key=$env_value")
        echo "Extracted env: $env_key=$env_value"
    else
        FILTERED_PASSTHROUGH_ARGS+=("$arg")
    fi
done
PASSTHROUGH_ARGS=("${FILTERED_PASSTHROUGH_ARGS[@]}")

: "${JOB_DIR:?JOB_DIR must be set (exported by _container.sh)}"
MODEL_NAME=${SCRIPT_ARGS[0]:?model_name is required}
if [[ ! -f "$SCRIPT_DIR/configs/$MODEL_NAME.gpu.yml" ]]; then
    echo "!!! Unknown model: $MODEL_NAME (no configs/$MODEL_NAME.gpu.yml)." >&2
    echo "    Callers must resolve via resolve_model_name.sh before calling _train.sh." >&2
    exit 1
fi
echo "MODEL_NAME=$MODEL_NAME"

# Resolve output path (logic lives in job_dir.sh — single source of truth).
source "$SCRIPT_DIR/utils/job_dir.sh"
export OUTPUT_PATH=$(resolve_output_path "$JOB_DIR" "$MODEL_NAME" "${MODEL_NAME_ALIAS:-}")
echo "OUTPUT_PATH=$OUTPUT_PATH"
mkdir -p -v "$OUTPUT_PATH"

# Build associative array from extracted envs so train_env.sh can look up
# config inputs (e.g. ENABLE_XLA_DUMP) without needing them exported yet.
declare -A EXTRACTED_ENV_MAP
for env_pair in "${EXTRACTED_ENVS[@]}"; do
    EXTRACTED_ENV_MAP["${env_pair%%=*}"]="${env_pair#*=}"
done

# ---- Load environment configuration (edit train_env.sh to customize) ----
source "$SCRIPT_DIR/train_env.sh"

# Export extracted environment variables (after train_env.sh so overrides win).
if [ ${#EXTRACTED_ENVS[@]} -gt 0 ]; then
    echo "Exporting extracted environment variables:"
    for env_pair in "${EXTRACTED_ENVS[@]}"; do
        echo "  export $env_pair"
        export "$env_pair"
    done
fi

# Handle LD_PRELOAD (after extracting envs so _env_MAXTEXT_LD_PRELOAD works)
if [ -n "${MAXTEXT_LD_PRELOAD:-}" ]; then
    export LD_PRELOAD="$MAXTEXT_LD_PRELOAD"
else
    [ -n "${LD_PRELOAD:-}" ] && echo "[WARNING] LD_PRELOAD='$LD_PRELOAD'; unsetting..."
    unset LD_PRELOAD
fi

# MaxText expects NNODES for JAX distributed init.
export NNODES="${NUM_NODES:-1}"

echo "Show all environment variables:"
printenv | sort

# ============================================================================
# Launch Training (direct or via Ray actor)
# ============================================================================

# Build training arguments
TRAIN_ARGS=(
    "$SCRIPT_DIR/configs/$MODEL_NAME.gpu.yml"
    base_output_directory=$OUTPUT_PATH
    "${PASSTHROUGH_ARGS[@]}"
)

# Unbuffered output for real-time log streaming
export PYTHONUNBUFFERED=1

if [[ "${USE_RAY:-false}" == "true" ]]; then
    # Ray Actor Mode: actor launches training in a subprocess (no GIL contention)
    # Enables: GPU monitoring, flame graphs via py-spy --subprocesses
    echo "Launching via Ray actor..."
    export RAY_DEDUP_LOGS=0
    python3 -u "$SCRIPT_DIR/_ray_actor.py" "${TRAIN_ARGS[@]}"
else
    # Direct Mode (with MFU tracking)
    echo "Launching MaxText.train directly..."
    python3 -u "$SCRIPT_DIR/utils/mfu_tracker.py" "${TRAIN_ARGS[@]}"
fi
