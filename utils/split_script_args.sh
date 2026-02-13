#!/bin/bash

# Usage: source utils/split_script_args.sh
#        split_script_args "$@"
#        # Then use: SCRIPT_ARGS and PASSTHROUGH_ARGS arrays

split_script_args() {
    SCRIPT_ARGS=()
    PASSTHROUGH_ARGS=()
    local found_sep=false

    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            found_sep=true
        elif [[ "$found_sep" == false ]]; then
            SCRIPT_ARGS+=("$arg")
        else
            if [[ -n "$arg" ]]; then
                PASSTHROUGH_ARGS+=("$arg")
            fi
        fi
    done
}
