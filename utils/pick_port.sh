#!/bin/bash

# Pick a random port (20000-52767) that isn't already in use.
#
# Usage:
#   source utils/pick_port.sh
#   JAX_PORT=$(pick_free_port)              || { echo "FATAL: no free port" >&2; exit 1; }
#   RAY_PORT=$(pick_free_port "$JAX_PORT")  || { echo "FATAL: no free port" >&2; exit 1; }
#
# Args (optional): ports to exclude (collision avoidance).
# Prints the chosen port to stdout; returns 1 after 10 failed attempts.

pick_free_port() {
    local exclude=("$@")
    local port
    for _attempt in {1..10}; do
        port=$((20000 + RANDOM))
        # Skip excluded ports (e.g. a port already assigned to JAX).
        for ex in "${exclude[@]}"; do
            [[ -n "$ex" && "$port" -eq "$ex" ]] 2>/dev/null && continue 2
        done
        ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | grep -q ":${port}$" || { echo "$port"; return 0; }
    done
    echo "[ERROR] pick_free_port: failed to find a free port after 10 attempts" >&2
    return 1
}
