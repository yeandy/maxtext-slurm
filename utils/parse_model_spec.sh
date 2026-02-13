#!/bin/bash

# Usage: source utils/parse_model_spec.sh
#        split_script_args "$@"        # sets SCRIPT_ARGS, PASSTHROUGH_ARGS
#        parse_model_spec              # sets MODEL_NAME, MODEL_NAME_ALIAS,
#                                      #      EXP_TAG, SBATCH_ARGS

# Format passthrough args (--flag=val -> flag_val) and return the joined string.
_format_exp_tag_suffix() {
    local formatted_args=()
    local formatted_arg
    for arg in "${PASSTHROUGH_ARGS[@]}"; do
        formatted_arg="${arg#--}"
        formatted_arg="${formatted_arg#-}"
        formatted_arg="${formatted_arg//=/_}"
        formatted_args+=("$formatted_arg")
    done
    # Join with -
    (IFS='-'; echo "${formatted_args[*]}")
}

# Parse the model spec from SCRIPT_ARGS and PASSTHROUGH_ARGS (globals).
#
# Format: model_name[[:model_name_alias]:exp_tag]
#
#   model_name                            Config-only
#   model_name:exp_tag                    Config + experiment tag
#   model_name:model_name_alias:          Config + custom checkpoint dir (trailing colon)
#   model_name:model_name_alias:exp_tag   Config + custom checkpoint dir + experiment tag
#
# MODEL_NAME        -> selects the run config (.gpu.yml file) and Slurm job name
# MODEL_NAME_ALIAS  -> overrides MODEL_NAME in the checkpointing output directory
#                      (checkpoint, tensorboard) so that parallel experiments using
#                      the same config get isolated directories; has no effect when
#                      enable_checkpointing is not set (dirs are already unique per job)
#
# Sets:  MODEL_NAME, MODEL_NAME_ALIAS, EXP_TAG, SBATCH_ARGS
parse_model_spec() {
    EXP_TAG=""
    MODEL_NAME=""
    MODEL_NAME_ALIAS=""

    # Parse model name from SCRIPT_ARGS (if not starting with -)
    if [[ ${#SCRIPT_ARGS[@]} -eq 0 || ${SCRIPT_ARGS[0]} == -* ]]; then
        echo "!!! model_name is required." >&2
        echo "    Usage: submit.sh <model_name> [sbatch_args...] -- [passthrough_args...]" >&2
        echo "           run_local.sh <model_name> -- [passthrough_args...]" >&2
        exit 1
    else
        # Peel colon-separated parts from right to left:
        #   1st peel: exp_tag        (rightmost)
        #   2nd peel: model_name_alias (middle, if present)
        MODEL_NAME="${SCRIPT_ARGS[0]}"
        if [[ "$MODEL_NAME" == *:* ]]; then
            EXP_TAG="${MODEL_NAME##*:}"
            MODEL_NAME="${MODEL_NAME%:*}"
        fi
        if [[ "$MODEL_NAME" == *:* ]]; then
            MODEL_NAME_ALIAS="${MODEL_NAME##*:}"
            MODEL_NAME="${MODEL_NAME%:*}"
        fi
        if [[ "$MODEL_NAME" == *:* ]]; then
            echo "!!! Invalid model spec '${SCRIPT_ARGS[0]}': too many ':' separators." >&2
            echo "    Expected format: model_name[[:model_name_alias]:exp_tag]" >&2
            exit 1
        fi
        SBATCH_ARGS=("${SCRIPT_ARGS[@]:1}")  # Rest are sbatch args
    fi

    # Append formatted passthrough args to EXP_TAG
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
        local suffix
        suffix="$(_format_exp_tag_suffix)"
        if [[ -n "$EXP_TAG" ]]; then
            EXP_TAG="${EXP_TAG}-${suffix}"
        else
            EXP_TAG="${suffix}"
        fi
    fi
}
