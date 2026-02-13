#!/usr/bin/env bash

# coredump.sh — Helpers for core dump capture inside Docker containers.
#
# Core dumps for JAX/XLA processes can be 100+ GB. Writing to shared storage
# at ~200 MB/s takes many minutes. If the container exits (and Docker
# unmounts /coredump) before the kernel finishes, the core file is lost.
#
# Usage (inside the container):
#   source /path/to/coredump.sh
#   setup_coredump          # set core_pattern, ulimit, SIGTERM trap
#   ...run training...
#   wait_for_coredump       # poll until core file size stabilises
#
# Environment variables:
#   COREDUMP_DIR             Directory where core files are written (default: /coredump)
#   COREDUMP_WAIT_TIMEOUT    Max seconds to wait for write (default: 900)

: "${COREDUMP_DIR:=/coredump}"

# Portable file-size helper (GNU stat uses -c%s, BSD/macOS uses -f%z).
_file_size() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

# ---------------------------------------------------------------------------
# setup_coredump — configure core_pattern, ulimit, and SIGTERM trap.
#
# Args: $1 = core_pattern template (optional, uses $COREDUMP_DIR/core.%t.%h.%e.%p)
# ---------------------------------------------------------------------------
setup_coredump() {
    if [[ -d "${COREDUMP_DIR:-}" ]]; then
        local pattern="${1:-${COREDUMP_DIR}/core.%t.%h.%e.%p}"
        echo "$pattern" | sudo tee /proc/sys/kernel/core_pattern
        ulimit -c unlimited
        echo "[coredump] core_pattern=$(cat /proc/sys/kernel/core_pattern)"
    else
        echo "[coredump] No COREDUMP_DIR — core dumps disabled"
    fi

    # Trap SIGTERM so we wait for coredumps when Slurm cancels the step.
    trap '_coredump_sigterm_handler' TERM
}

_coredump_sigterm_handler() {
    echo "[SIGTERM] Received — waiting for coredump before exit..."
    wait_for_coredump
    exit 143
}

# ---------------------------------------------------------------------------
# wait_for_coredump — block until core files stop growing or timeout.
#
# No-op if COREDUMP_DIR is unset or no core files exist.
# ---------------------------------------------------------------------------
wait_for_coredump() {
    local coredump_dir="${COREDUMP_DIR:-}"
    [[ -d "$coredump_dir" ]] || return 0

    # Any core files present?
    local found=0
    for f in "$coredump_dir"/core*; do
        [[ -f "$f" ]] && { found=1; break; }
    done
    [[ $found -eq 1 ]] || return 0

    local max_wait="${COREDUMP_WAIT_TIMEOUT:-900}"  # default 15 min
    local poll=10
    local elapsed=0 prev_bytes=-1 stable=0

    echo "[coredump] Core dump detected in $coredump_dir — waiting for write to finish (timeout ${max_wait}s)..."

    while (( elapsed < max_wait )); do
        local total=0
        for f in "$coredump_dir"/core*; do
            [[ -f "$f" ]] || continue
            total=$(( total + $(_file_size "$f") ))
        done

        if (( total == prev_bytes )); then
            (( ++stable ))
            if (( stable >= 3 )); then
                echo "[coredump] Write complete: $(( total / 1048576 )) MB after ${elapsed}s"
                return 0
            fi
        else
            stable=0
            echo "[coredump] Writing... $(( total / 1048576 )) MB (${elapsed}s elapsed)"
        fi

        prev_bytes=$total
        sleep $poll
        elapsed=$(( elapsed + poll ))
    done

    echo "[coredump] WARNING: Timeout after ${max_wait}s — core dump may be incomplete"
}
