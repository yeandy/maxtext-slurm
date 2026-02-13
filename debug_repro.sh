#!/bin/bash

# Crash reproduction: run a 1-step training job in a loop until it crashes.
# Intended to be run inside Docker via `run_local.sh` (interactive mode).
#
# No set -e: exit codes are captured and checked explicitly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/utils/resolve_model_name.sh"
source "$SCRIPT_DIR/utils/coredump.sh"
MODEL_NAME=${1:-llama2-70b}
if ! MODEL_NAME=$(resolve_model_name "$SCRIPT_DIR/configs" "$MODEL_NAME"); then
    echo "!!! Model name resolution failed. Exiting..." >&2
    exit 1
fi
echo "MODEL_NAME=$MODEL_NAME"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "========================================"
    echo "Starting iteration #$ITERATION"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    START_TIME=$(date +%s)

    NNODES=1 \
    NODE_RANK=0 \
    JAX_COORDINATOR_IP=127.0.0.1 \
    JAX_COORDINATOR_PORT=20002 \
    JOB_DIR=debug_maxtext \
    "$SCRIPT_DIR/_train.sh" "$MODEL_NAME" -- steps=1 2>&1

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "========================================"
    echo "Iteration #$ITERATION finished"
    echo "Exit code: $EXIT_CODE"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Duration: ${DURATION}s ($(printf '%02d:%02d:%02d' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60))))"
    echo "========================================"

    # If exit code is non-zero => crash => wait for coredump, then break loop
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "_train.sh crashed with exit code $EXIT_CODE. Stopping loop."
        wait_for_coredump
        break
    fi
    echo "_train.sh exited normally, restarting..."
    echo ""
done

echo "========================================"
echo "Loop terminated after $ITERATION iteration(s)"
echo "========================================"
