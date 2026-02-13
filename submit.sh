#!/bin/bash

# Submit a MaxText training job to Slurm.
set -e

# Resolve the directory containing this script (works from any working directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAXTEXT_SLURM_DIR="$SCRIPT_DIR"

# Master output directory (configurable via env var; default: $SCRIPT_DIR/outputs).
# Usage: JOB_WORKSPACE=/shared/maxtext_jobs ./submit.sh ...
export JOB_WORKSPACE="${JOB_WORKSPACE:-$SCRIPT_DIR/outputs}"
mkdir -p -v "$JOB_WORKSPACE"; chmod a+w "$JOB_WORKSPACE"

# Warn if JOB_WORKSPACE doesn't appear to be on a shared filesystem.
_fs_type=$(df -T "$JOB_WORKSPACE" 2>/dev/null | awk 'NR==2 {print $2}')
case "$_fs_type" in
    nfs|nfs4|lustre|gpfs|beegfs|cifs|panfs|pvfs2|orangefs|fuse.*) ;;
    "") ;;  # df failed — skip check
    *) echo "WARNING: JOB_WORKSPACE ($JOB_WORKSPACE) is on filesystem '$_fs_type'," \
            "which may not be shared across cluster nodes." >&2 ;;
esac

# ------- Resolve Slurm reservation -------
source "$SCRIPT_DIR/utils/reservation.sh"
RESERVATION_NAME="$(resolve_reservation "${USER:-}")"
if [[ -n "${RESERVATION_NAME}" ]]; then
  RESERVATION_OPT=(--reservation="${RESERVATION_NAME}")
else
  RESERVATION_OPT=()
fi

# ------- Split script args first -------
source "$SCRIPT_DIR/utils/split_script_args.sh"
split_script_args "$@"
echo "PASSTHROUGH_ARGS=\"${PASSTHROUGH_ARGS[*]}\""

# ------- Parse model spec + experiment tag -------
source "$SCRIPT_DIR/utils/parse_model_spec.sh"
parse_model_spec

# ------- Resolve model name -------
source "$SCRIPT_DIR/utils/resolve_model_name.sh"
if ! MODEL_NAME=$(resolve_model_name "$SCRIPT_DIR/configs" "$MODEL_NAME"); then
    echo "!!! Model name resolution failed. Exiting..." >&2
    exit 1
fi

echo "MODEL_NAME=$MODEL_NAME"
[[ -n "$MODEL_NAME_ALIAS" ]] && echo "MODEL_NAME_ALIAS=$MODEL_NAME_ALIAS"
echo "EXP_TAG=$EXP_TAG"
echo "SBATCH_ARGS=\"${SBATCH_ARGS[*]}\""

# ------- Build job name -------
JOB_NAME="JAX-${MODEL_NAME}"
if [[ -n "$EXP_TAG" ]]; then
    JOB_NAME="${JOB_NAME}-${EXP_TAG}"
fi
echo "JOB_NAME=$JOB_NAME"

# ------- Slurm settings -------------
export SLURM_TREE_WIDTH=128

# ------- Ray export (if enabled) -------
RAY_EXPORT=()
if [[ "${RAY:-0}" == "1" || "${RAY:-}" == "true" ]]; then
    source "$SCRIPT_DIR/utils/ray_cluster.sh"
    RAY_EXPORT=("$(build_ray_export)")
fi

# ------- Build artifact -------
# Build an artifact of the repo so that pending/queued jobs are isolated from later edits.
# The artifact lives under .artifacts/ inside JOB_WORKSPACE (shared filesystem).
#
# Layout detection: scripts may live at the repo root (open-source) or in a
# subdirectory (internal).  A .git/ directory in SCRIPT_DIR means repo root.
if [[ -d "$SCRIPT_DIR/.git" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
else
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
fi
ARTIFACT_BASE_DIR="$JOB_WORKSPACE/.artifacts"
ARTIFACT_ID="artifact_$(date +%Y%m%d_%H%M%S)_$(printf '%04x' $RANDOM)"
ARTIFACT_DIR="$ARTIFACT_BASE_DIR/$ARTIFACT_ID"

source "$SCRIPT_DIR/utils/artifact.sh"
build_artifact "$REPO_ROOT" "$ARTIFACT_DIR" "$SCRIPT_DIR"

# Path to the scripts directory inside the artifact.
if [[ "$REPO_ROOT" == "$SCRIPT_DIR" ]]; then
    ARTIFACT_SCRIPTS="$ARTIFACT_DIR"
    ARTIFACT_SYMLINK="../.artifacts/$ARTIFACT_ID"
else
    ARTIFACT_SCRIPTS="$ARTIFACT_DIR/$(basename "$SCRIPT_DIR")"
    ARTIFACT_SYMLINK="../.artifacts/$ARTIFACT_ID/$(basename "$SCRIPT_DIR")"
fi

# Persist the submit command for traceability.
{
    # Env vars consumed by submit.sh (only non-default ones).
    [[ "$JOB_WORKSPACE" != "$SCRIPT_DIR/outputs" ]] && printf 'JOB_WORKSPACE=%q ' "$JOB_WORKSPACE"
    [[ -n "${RAY:-}" ]] && printf 'RAY=%q ' "$RAY"
    _cmd=$(printf '%q ' "$0" "$@")
    echo "${_cmd% }"
} > "$ARTIFACT_SCRIPTS/submit_cmd.txt"

# Tell the batch job to run from the artifact (not the live repo).
# MAXTEXT_SLURM_DIR propagates through: sbatch -> _job.sbatch
#   -> srun --export=ALL -> _container.sh -> Docker mounts.
# JOB_WORKSPACE tells _container.sh where the real outputs/
#   dir is, so Docker mounts it at /outputs (separate from the scripts mount).
export MAXTEXT_SLURM_DIR="$ARTIFACT_SCRIPTS"
echo "MAXTEXT_SLURM_DIR=$MAXTEXT_SLURM_DIR"

# ------- Submit job -------
SBATCH_OUTPUT=$(sbatch "${RESERVATION_OPT[@]}" -J "$JOB_NAME" \
    --output="$JOB_WORKSPACE/%j-$JOB_NAME.log" \
    "${RAY_EXPORT[@]}" \
    "${SBATCH_ARGS[@]}" \
    "$SCRIPT_DIR/_job.sbatch" \
    "$MODEL_NAME" "$EXP_TAG" "$MODEL_NAME_ALIAS" -- "${PASSTHROUGH_ARGS[@]}" 2>&1) || {
    echo "$SBATCH_OUTPUT"
    echo "[ARTIFACT] sbatch failed — cleaning up artifact: $ARTIFACT_DIR"
    rm -rf "$ARTIFACT_DIR"
    exit 1
}
echo "$SBATCH_OUTPUT"

# ------- Post-submit: create symlinks -------
# The artifact directory ($ARTIFACT_DIR) never moves, so MAXTEXT_SLURM_DIR
# (captured by Slurm at submit time) is always valid — no race condition.
JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '/Submitted batch job/ {print $NF}')
if [[ -n "$JOB_ID" ]]; then
    source "$SCRIPT_DIR/utils/job_dir.sh"
    JOB_DIR=$(make_job_dir "$JOB_ID" "$JOB_NAME")

    # Per-job directory with artifact symlink.
    # ../.artifacts/<artifact>/<basename> resolves correctly both on the host
    # and inside Docker (where real outputs/ is mounted at /outputs).
    mkdir -p "$JOB_WORKSPACE/$JOB_DIR"; chmod a+w "$JOB_WORKSPACE/$JOB_DIR"
    ln -snf "$ARTIFACT_SYMLINK" "$JOB_WORKSPACE/$JOB_DIR/artifact"
    echo "[ARTIFACT] $JOB_WORKSPACE/$JOB_DIR/artifact -> ${ARTIFACT_SYMLINK#../}"

    # Symlink to the job log (may dangle until Slurm starts the job).
    ln -snf "../$JOB_DIR.log" "$JOB_WORKSPACE/$JOB_DIR/log"
else
    echo "[ARTIFACT] WARNING: Could not parse job ID. Artifact at: $ARTIFACT_DIR"
fi
