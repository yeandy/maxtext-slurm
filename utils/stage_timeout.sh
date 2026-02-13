#!/bin/bash

# stage_timeout.sh — Generic per-stage timeout for job pipelines.
#
# Wraps any command with a configurable per-stage timeout. Provides friendly
# logging (resolved values at startup, elapsed time on completion, actionable
# hints on failure). Launcher-agnostic — works with any blocking command.
#
# Usage:
#   source utils/stage_timeout.sh
#
#   # Register stages with defaults, apply STAGE_TIMEOUTS overrides, print table.
#   # Values: positive integer (seconds) or "none" / "-1" (no timeout).
#   stage_timeout_init preflight:900 pull:900 ecc:300 train:none
#
#   # Run a command under its stage timeout.
#   run_stage preflight "Preflight" bash cleanup.sh
#   run_stage train     "Training"  bash train.sh

declare -A _STAGE_TIMEOUT=()   # stage name -> timeout value ("none" or seconds)
declare -a _STAGE_ORDER=()     # ordered stage names for display

# Normalize and validate a timeout value.
# Echoes the canonical value on success; prints error and returns 1 on failure.
_normalize_timeout() {
    case "$1" in
        none|NONE|None|-1) echo "none" ;;
        0)  echo "FATAL: Timeout '0' is ambiguous. Use 'none' to disable or a positive integer for seconds." >&2
            return 1 ;;
        *[!0-9]*)
            echo "FATAL: Timeout '$1' is not valid. Use a positive integer or 'none'." >&2
            return 1 ;;
        *) echo "$1" ;;
    esac
}

# stage_timeout_init <stage:default> [stage:default ...]
#   Register stages, apply STAGE_TIMEOUTS env var overrides, and print the table.
stage_timeout_init() {
    _STAGE_TIMEOUT=()
    _STAGE_ORDER=()

    for _spec in "$@"; do
        local name="${_spec%%:*}"
        local val="${_spec#*:}"
        val="$(_normalize_timeout "$val")" || exit 1
        _STAGE_TIMEOUT[$name]="$val"
        _STAGE_ORDER+=("$name")
    done

    if [[ -n "${STAGE_TIMEOUTS:-}" ]]; then
        IFS=',' read -ra _overrides <<< "$STAGE_TIMEOUTS"
        for _entry in "${_overrides[@]}"; do
            local name="${_entry%%:*}"
            local val="${_entry#*:}"
            if [[ -z "${_STAGE_TIMEOUT[$name]+x}" ]]; then
                echo "FATAL: Unknown stage '$name' in STAGE_TIMEOUTS."
                echo "Valid stages: ${_STAGE_ORDER[*]}"
                exit 1
            fi
            val="$(_normalize_timeout "$val")" || exit 1
            _STAGE_TIMEOUT[$name]="$val"
        done
    fi

    echo "==STAGE TIMEOUTS (override: STAGE_TIMEOUTS=\"stage:seconds,...\")=="
    for _name in "${_STAGE_ORDER[@]}"; do
        local t="${_STAGE_TIMEOUT[$_name]}"
        printf "  %-12s %s\n" "$_name" "$([[ "$t" == "none" ]] && echo "none" || echo "${t}s")"
    done
    echo "================================================================="
}

# run_stage <stage_name> <description> <command> [args...]
#   Run a command under its stage timeout. Exits on failure with actionable hint.
run_stage() {
    local stage="$1" desc="$2"; shift 2
    local t="${_STAGE_TIMEOUT[$stage]}"

    echo "== $desc BEGIN ($([[ "$t" == "none" ]] && echo "no timeout" || echo "timeout=${t}s")) =="
    local start=$SECONDS

    if [[ "$t" != "none" ]]; then
        timeout "$t" "$@"
    else
        "$@"
    fi
    local rc=$?
    local elapsed=$(( SECONDS - start ))

    if [[ $rc -eq 0 ]]; then
        echo "== $desc END (${elapsed}s) =="
    elif [[ $rc -eq 124 ]]; then
        echo "== $desc TIMEOUT (${elapsed}s/${t}s) =="
        echo "  Hint: STAGE_TIMEOUTS=\"$stage:<seconds>\" to adjust."
        _STAGE_FAILURE="$desc TIMEOUT (${elapsed}s/${t}s)"
        exit 1
    else
        echo "== $desc FAILED (exit=$rc, ${elapsed}s) =="
        _STAGE_FAILURE="$desc FAILED (exit=$rc)"
        exit 1
    fi
}
