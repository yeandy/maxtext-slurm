#!/bin/bash

# Resolve default outputs directory relative to script location
_TAIL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

tail_job_log() {
  local arg="${1:-}"
  local default_dir="${JOB_WORKSPACE:-$_TAIL_SCRIPT_DIR/outputs}"
  local latest=""

  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    cat >&2 <<'EOF'
Usage: tail_job_log [JOB_ID | FILE | DIRECTORY]

Follow (tail -f) the most recent job log file (.log or legacy .out).

Modes:
  (no argument)   Latest log in the default output directory
  JOB_ID          Latest log matching <JOB_ID>-* in the default output directory
  FILE            Tail the given file directly
  JOB_DIR         Job directory containing a log symlink (tab-completion friendly)
  DIRECTORY       Latest log in the given directory

The default output directory is $JOB_WORKSPACE if set, otherwise outputs/ next to the scripts.

Examples:
  tail_job_log              # latest log in outputs/
  tail_job_log 12345        # latest log for Slurm job 12345
  tail_job_log run.log      # tail a specific file
  tail_job_log 12345-JAX-llama2-70b/   # job dir (follows log symlink)
  tail_job_log logs/        # latest log in logs/
EOF
    return 0
  fi

  # Helper: find the most recently modified .log or .out file in a directory,
  # optionally filtered by a glob prefix (e.g. "12345-*").
  _find_latest_log() {
    local dir="$1" prefix="${2:-*}"
    ls -t "$dir"/${prefix}.log "$dir"/${prefix}.out 2>/dev/null | head -n 1
  }

  if [[ -z "$arg" ]]; then
    # No argument — find latest log in default directory
    if [[ ! -d "$default_dir" ]]; then
      echo "Default outputs directory not found: $default_dir" >&2
      return 1
    fi
    latest=$(_find_latest_log "$default_dir")
    if [[ -z "$latest" ]]; then
      echo "No log files found in $default_dir" >&2
      return 1
    fi
  elif [[ "$arg" =~ ^[0-9]+$ ]]; then
    # Pure integer — treat as a Slurm job ID
    latest=$(_find_latest_log "$default_dir" "$arg-*")
    if [[ -z "$latest" ]]; then
      echo "No log files found for job ID $arg in $default_dir" >&2
      return 1
    fi
  elif [[ -f "$arg" ]]; then
    # Direct file path
    latest="$arg"
  elif [[ -d "$arg" ]]; then
    # Directory — if it's a job dir with a log symlink, follow it;
    # otherwise find the latest log inside it.
    if [[ -e "$arg/log" || -L "$arg/log" ]]; then
      latest="$arg/log"
    else
      latest=$(_find_latest_log "$arg")
      if [[ -z "$latest" ]]; then
        echo "No log files found in $arg" >&2
        return 1
      fi
    fi
  else
    echo "Not a file, directory, or job ID: $arg" >&2
    echo "Run with --help for usage." >&2
    return 1
  fi

  echo "Tailing: $latest"
  tail -f "$latest"
}

tail_job_log "$@"
