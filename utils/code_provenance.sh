#!/bin/bash

# Emit code provenance exactly once per run.
# Usage:
#   source utils/code_provenance.sh
#   emit_code_provenance "$SCRIPT_DIR" "$LOG_FILE"   # append by path
#   emit_code_provenance "$SCRIPT_DIR" "" "$LOG_APPEND_FD"  # append via open FD
#   emit_code_provenance "$SCRIPT_DIR"               # stdout only
emit_code_provenance() {
    local script_dir="$1"
    local log_file="${2:-}"
    local log_fd="${3:-}"
    local out_msg

    if [[ "${MAXTEXT_GIT_SUMMARY_EMITTED:-0}" == "1" ]]; then
        return 0
    fi

    if [[ -f "$script_dir/git_summary.txt" ]]; then
        out_msg=$(
            {
                echo "[INFO] Running from artifact: $script_dir"
                cat "$script_dir/git_summary.txt"
            }
        )
    elif [[ -x "$script_dir/utils/git_summary.sh" ]]; then
        out_msg=$(
            {
                echo "[INFO] Capturing git summary..."
                bash "$script_dir/utils/git_summary.sh"
            }
        )
    else
        out_msg="[WARN] git summary unavailable: $script_dir/utils/git_summary.sh"
    fi

    if [[ -n "$log_file" ]]; then
        echo "$out_msg" | tee -a "$log_file"
    elif [[ -n "$log_fd" ]]; then
        # Write to current stdout and to the already-open log FD. Using an FD
        # keeps appends on the same inode even if the log path is renamed.
        echo "$out_msg"
        if [[ "$log_fd" =~ ^[0-9]+$ ]]; then
            echo "$out_msg" >&$log_fd
        else
            echo "[WARN] Invalid log fd '$log_fd' for emit_code_provenance"
        fi
    else
        echo "$out_msg"
    fi

    export MAXTEXT_GIT_SUMMARY_EMITTED=1
}
