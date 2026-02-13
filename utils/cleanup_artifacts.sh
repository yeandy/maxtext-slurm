#!/bin/bash

# Remove orphaned artifacts that no job directory references.
#
# Usage:
#   utils/cleanup_artifacts.sh                # list all artifacts (dry-run)
#   utils/cleanup_artifacts.sh -c             # remove orphans (with confirmation)
#   utils/cleanup_artifacts.sh -c -y          # remove orphans (skip confirmation)
#   utils/cleanup_artifacts.sh /path/to/workspace -c

_CLEAN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

workspace="${JOB_WORKSPACE:-$_CLEAN_SCRIPT_DIR/outputs}"
delete=false
yes=false

for arg in "$@"; do
    case "$arg" in
        -c|--clean) delete=true ;;
        -y|--yes)   yes=true ;;
        *)          workspace="$arg" ;;
    esac
done

artifact_base="$workspace/.artifacts"

RED='\033[0;31m'
GREEN='\033[0;32m'
RESET='\033[0m'

if [[ ! -d "$artifact_base" ]]; then
    echo "No .artifacts/ directory found in $workspace"
    exit 0
fi

# Collect referenced artifact dirs and which job(s) reference them.
declare -A referenced        # artifact_dir -> 1
declare -A referencing_jobs  # artifact_dir -> "job1, job2, ..."
for link in "$workspace"/*/artifact; do
    [[ -L "$link" ]] || continue
    target="$(readlink -f "$link" 2>/dev/null)" || continue
    # The symlink may point to the artifact root (scripts at repo root) or to
    # a subdirectory within it (scripts in a subdir).  Detect by checking if
    # the target's basename matches the artifact naming convention.
    if [[ "$(basename "$target")" == artifact_* ]]; then
        artifact_dir="$target"
    else
        artifact_dir="$(dirname "$target")"
    fi
    referenced["$artifact_dir"]=1
    job_name="$(basename "$(dirname "$link")")"
    if [[ -n "${referencing_jobs[$artifact_dir]:-}" ]]; then
        referencing_jobs["$artifact_dir"]+=", $job_name"
    else
        referencing_jobs["$artifact_dir"]="$job_name"
    fi
done

# List each artifact.
orphan_count=0
kept_count=0
orphan_dirs=()
for artifact in "$artifact_base"/artifact_*/; do
    [[ -d "$artifact" ]] || continue
    artifact="${artifact%/}"  # strip trailing slash
    canonical="$(readlink -f "$artifact")"
    artifact_name="$(basename "$artifact")"
    if [[ -n "${referenced[$canonical]:-}" ]]; then
        ((kept_count++))
        echo -e "${GREEN}REFERENCED${RESET}  $artifact_name  <- ${referencing_jobs[$canonical]}"
    else
        ((orphan_count++))
        orphan_dirs+=("$artifact")
        echo -e "${RED}ORPHAN${RESET}      $artifact_name"
    fi
done

echo ""
echo "Artifacts: $kept_count referenced, $orphan_count orphaned."

if (( orphan_count == 0 )); then
    exit 0
fi

if ! $delete; then
    echo "Run with -c to remove orphaned artifacts."
    exit 0
fi

# Delete orphans (confirm one-by-one unless -y).
removed=0
for artifact in "${orphan_dirs[@]}"; do
    artifact_name="$(basename "$artifact")"
    if ! $yes; then
        read -r -p "Remove $artifact_name? [y/N] " answer
        [[ "$answer" =~ ^[Yy]$ ]] || continue
    fi
    echo -e "${RED}REMOVING${RESET}    $artifact_name"
    rm -rf "$artifact"
    ((removed++))
done
echo "Done. Removed $removed of $orphan_count orphaned artifact(s)."
