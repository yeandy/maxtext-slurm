#!/bin/bash

# Usage: resolve_model_name <dir> <model_name>
# Echoes the resolved full model name to stdout, or exits with error.

resolve_model_name() {
  local configs_dir="$1"
  local input_model="$2"
  local _prev_nullglob=$(shopt -p nullglob)
  shopt -s nullglob
  local matches=("$configs_dir"/*"$input_model"*".gpu.yml")

  if (( ${#matches[@]} == 1 )); then
    local base="${matches[0]##*/}"
    local full_name="${base%.gpu.yml}"
    $_prev_nullglob
    echo "$full_name"
  else
    if (( ${#matches[@]} == 0 )); then
      echo "Error: no model file matches pattern '$configs_dir/*${input_model}*.gpu.yml'." >&2
    else
      echo "Error: ambiguous model name '${input_model}'. Multiple matches:" >&2
      for m in "${matches[@]}"; do
        echo "  ${m##*/}" >&2
      done
    fi
    echo "Supported models:" >&2
    local all=("$configs_dir"/*.gpu.yml)
    for f in "${all[@]}"; do
      echo "  ${f##*/}" | sed 's/.gpu.yml$//'
    done | sort -u >&2
    $_prev_nullglob
    return 1  # <-- fail cleanly
  fi
}
