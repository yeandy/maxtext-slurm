#!/bin/bash

# Submit a MaxText training job to Kubernetes.
#
# K8s equivalent of submit.sh — replaces the Slurm orchestration tier.
# Everything below _k8s_job.sh (container setup, training) runs unmodified.
#
# Prerequisites:
#   - kubectl configured with cluster access
#   - A PVC with the repo and outputs directory (set K8S_PVC_NAME)
#   - AMD or NVIDIA GPU device plugin installed in the cluster
#
# Usage:
#   k8s_submit.sh <model_name> -N <nodes> -- [passthrough_args...]
#
# Examples:
#   k8s_submit.sh 70b -N 4
#   k8s_submit.sh 70b -N 4 -- steps=20 dataset_type=synthetic
#   K8S_NAMESPACE=training k8s_submit.sh 70b -N 8
#
# Environment variables:
#   K8S_NAMESPACE          Namespace (default: default)
#   K8S_NODE_SELECTOR      Node selector as key=value (optional)
#   K8S_PVC_NAME           PersistentVolumeClaim for shared storage (required)
#   K8S_PVC_MOUNT          Mount path for the PVC inside pods (default: /mnt/shared)
#   REPO_PATH_IN_PVC       Path to this repo within the PVC (default: auto-detected)
#   DOCKER_IMAGE           Override container image
#   JOB_WORKSPACE          Output directory path inside pods
#   JAX_COORDINATOR_PORT   Coordinator port (default: 29500)
#   GPUS_PER_NODE          GPUs per node (default: 8)
#   GPU_RESOURCE_TYPE      K8s GPU resource name (default: amd.com/gpu)
#   RAY                    Enable Ray observability stack (RAY=1)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAXTEXT_SLURM_DIR="$SCRIPT_DIR"

# -------- Parse args (same interface as submit.sh) --------
source "$SCRIPT_DIR/utils/parse_job_args.sh"

echo "MODEL_NAME=$MODEL_NAME"
[[ -n "$MODEL_NAME_ALIAS" ]] && echo "MODEL_NAME_ALIAS=$MODEL_NAME_ALIAS"
echo "EXP_TAG=$EXP_TAG"

NNODES=1
NAMESPACE="${K8S_NAMESPACE:-default}"
JAX_PORT="${JAX_COORDINATOR_PORT:-29500}"
GPUS="${GPUS_PER_NODE:-8}"
GPU_TYPE="${GPU_RESOURCE_TYPE:-amd.com/gpu}"

i=0
while [[ $i -lt ${#SBATCH_ARGS[@]} ]]; do
    case "${SBATCH_ARGS[$i]}" in
        -N|--nodes)     NNODES="${SBATCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        -n|--namespace) NAMESPACE="${SBATCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        *) echo "WARNING: Ignoring unknown flag: ${SBATCH_ARGS[$i]}" >&2; i=$((i+1)) ;;
    esac
done

# -------- PVC and paths --------
PVC_NAME="${K8S_PVC_NAME:?Set K8S_PVC_NAME to your shared-storage PersistentVolumeClaim.}"
PVC_MOUNT="${K8S_PVC_MOUNT:-/mnt/shared}"

if [[ -n "${REPO_PATH_IN_PVC:-}" ]]; then
    REPO_IN_POD="${PVC_MOUNT}/${REPO_PATH_IN_PVC}"
else
    REPO_IN_POD="${PVC_MOUNT}/$(basename "$SCRIPT_DIR")"
    echo "[INFO] Assuming repo is at ${REPO_IN_POD} inside the PVC."
    echo "       Set REPO_PATH_IN_PVC if this is wrong."
fi

# -------- Container image --------
source "$SCRIPT_DIR/container_env.sh"
IMAGE="${DOCKER_IMAGE}"

# -------- Job identity --------
JOB_TS=$(date +%Y%m%d-%H%M%S)
JOB_RAND=$(printf '%04x' $RANDOM)
JOB_ID="k8s-${JOB_TS}-${JOB_RAND}"
JOB_NAME="JAX-${MODEL_NAME}${EXP_TAG:+-$EXP_TAG}"

# K8s names: lowercase, alphanumeric + hyphens, max 63 chars
K8S_NAME=$(echo "jax-${MODEL_NAME}-${JOB_TS}-${JOB_RAND}" | tr '[:upper:]_.' '[:lower:]--' | cut -c1-63 | sed 's/-$//')

# -------- Paths (inside the pod) --------
JOB_WORKSPACE="${JOB_WORKSPACE:-${REPO_IN_POD}/outputs}"
COORD_DIR="$JOB_WORKSPACE/${JOB_ID}-${JOB_NAME}/.coord"

# -------- Build artifact --------
# Snapshot the repo so queued jobs are isolated from later edits,
# matching submit.sh behavior.
if [[ -d "$SCRIPT_DIR/.git" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
else
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
fi
ARTIFACT_BASE_DIR="$SCRIPT_DIR/outputs/.artifacts"
ARTIFACT_ID="artifact_${JOB_TS}_${JOB_RAND}"
ARTIFACT_DIR="$ARTIFACT_BASE_DIR/$ARTIFACT_ID"

source "$SCRIPT_DIR/utils/artifact.sh"
build_artifact "$REPO_ROOT" "$ARTIFACT_DIR" "$SCRIPT_DIR"

if [[ "$REPO_ROOT" == "$SCRIPT_DIR" ]]; then
    ARTIFACT_SCRIPTS="$ARTIFACT_DIR"
    ARTIFACT_SYMLINK="../.artifacts/$ARTIFACT_ID"
    ARTIFACT_SCRIPTS_IN_POD="${REPO_IN_POD}/outputs/.artifacts/${ARTIFACT_ID}"
else
    ARTIFACT_SCRIPTS="$ARTIFACT_DIR/$(basename "$SCRIPT_DIR")"
    ARTIFACT_SYMLINK="../.artifacts/$ARTIFACT_ID/$(basename "$SCRIPT_DIR")"
    ARTIFACT_SCRIPTS_IN_POD="${REPO_IN_POD}/outputs/.artifacts/${ARTIFACT_ID}/$(basename "$SCRIPT_DIR")"
fi

# Persist the submit command for traceability.
{
    printf 'K8S_PVC_NAME=%q ' "$PVC_NAME"
    [[ "$PVC_MOUNT" != "/mnt/shared" ]] && printf 'K8S_PVC_MOUNT=%q ' "$PVC_MOUNT"
    [[ -n "${REPO_PATH_IN_PVC:-}" ]] && printf 'REPO_PATH_IN_PVC=%q ' "$REPO_PATH_IN_PVC"
    [[ "$JOB_WORKSPACE" != "${REPO_IN_POD}/outputs" ]] && printf 'JOB_WORKSPACE=%q ' "$JOB_WORKSPACE"
    [[ "$NAMESPACE" != "default" ]] && printf 'K8S_NAMESPACE=%q ' "$NAMESPACE"
    [[ "$GPU_TYPE" != "amd.com/gpu" ]] && printf 'GPU_RESOURCE_TYPE=%q ' "$GPU_TYPE"
    [[ -n "${RAY:-}" ]] && printf 'RAY=%q ' "$RAY"
    _cmd=$(printf '%q ' "$0" "$@")
    echo "${_cmd% }"
} > "$ARTIFACT_SCRIPTS/submit_cmd.txt"

echo "NNODES=$NNODES"
echo "NAMESPACE=$NAMESPACE"
echo "JOB_ID=$JOB_ID"
echo "K8S_NAME=$K8S_NAME"
echo "IMAGE=$IMAGE"
echo "PVC=$PVC_NAME -> $PVC_MOUNT"
echo "REPO_IN_POD=$REPO_IN_POD"
echo "ARTIFACT=$ARTIFACT_SCRIPTS_IN_POD"
echo "JOB_WORKSPACE=$JOB_WORKSPACE"
echo "PASSTHROUGH_ARGS=\"${PASSTHROUGH_ARGS[*]}\""

# -------- Build passthrough args string --------
PT_ARGS=""
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    PT_ARGS=$(printf '%s ' "${PASSTHROUGH_ARGS[@]}")
fi

# -------- Node selector YAML fragment --------
NODE_SELECTOR_YAML=""
if [[ -n "${K8S_NODE_SELECTOR:-}" ]]; then
    IFS='=' read -r NS_KEY NS_VAL <<< "$K8S_NODE_SELECTOR"
    NODE_SELECTOR_YAML=$(cat <<-NSEOF
      nodeSelector:
        ${NS_KEY}: "${NS_VAL}"
NSEOF
)
fi

# -------- Ray observability --------
RAY_ENV_YAML=""
if [[ "${RAY:-0}" == "1" || "${RAY:-}" == "true" ]]; then
    RAY_PORT="${RAY_PORT:-29501}"
    RAY_ENV_YAML=$(cat <<-RAYEOF
        - name: RAY
          value: "1"
        - name: RAY_PORT
          value: "${RAY_PORT}"
RAYEOF
)
fi

# -------- Generate manifest --------
MANIFEST_FILE="/tmp/${K8S_NAME}.yaml"

cat > "$MANIFEST_FILE" <<MANIFEST
apiVersion: batch/v1
kind: Job
metadata:
  name: ${K8S_NAME}
  labels:
    app: maxtext-training
    model: ${MODEL_NAME}
    job-id: ${JOB_ID}
spec:
  completionMode: Indexed
  completions: ${NNODES}
  parallelism: ${NNODES}
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: maxtext-training
        job-name: ${K8S_NAME}
    spec:
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      restartPolicy: Never
${NODE_SELECTOR_YAML}
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                job-name: ${K8S_NAME}
            topologyKey: kubernetes.io/hostname
      tolerations:
      - operator: Exists
      containers:
      - name: training
        image: ${IMAGE}
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-c"]
        args:
        - |
          exec ${ARTIFACT_SCRIPTS_IN_POD}/_k8s_job.sh ${MODEL_NAME} -- ${PT_ARGS}
        env:
        - name: JOB_ID
          value: "${JOB_ID}"
        - name: JOB_NAME
          value: "${JOB_NAME}"
        - name: NNODES
          value: "${NNODES}"
        - name: JAX_COORDINATOR_PORT
          value: "${JAX_PORT}"
        - name: JOB_WORKSPACE
          value: "${JOB_WORKSPACE}"
        - name: COORD_DIR
          value: "${COORD_DIR}"
        - name: MAXTEXT_SLURM_DIR
          value: "${ARTIFACT_SCRIPTS_IN_POD}"
${RAY_ENV_YAML}
        securityContext:
          privileged: true
        resources:
          limits:
            ${GPU_TYPE}: "${GPUS}"
          requests:
            ${GPU_TYPE}: "${GPUS}"
        volumeMounts:
        - name: shared-storage
          mountPath: ${PVC_MOUNT}
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: shared-storage
        persistentVolumeClaim:
          claimName: ${PVC_NAME}
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 128Gi
MANIFEST

echo ""
echo "Manifest saved: $MANIFEST_FILE"

# -------- Apply --------
if command -v kubectl &>/dev/null; then
    echo "Applying manifest..."
    kubectl apply -n "$NAMESPACE" -f "$MANIFEST_FILE"

    # Post-submit: create job directory with artifact symlink (matches submit.sh)
    source "$SCRIPT_DIR/utils/job_dir.sh"
    JOB_DIR=$(make_job_dir "$JOB_ID" "$JOB_NAME")
    mkdir -p "$SCRIPT_DIR/outputs/$JOB_DIR"; chmod a+w "$SCRIPT_DIR/outputs/$JOB_DIR"
    ln -snf "$ARTIFACT_SYMLINK" "$SCRIPT_DIR/outputs/$JOB_DIR/artifact"
    echo "[ARTIFACT] $SCRIPT_DIR/outputs/$JOB_DIR/artifact -> ${ARTIFACT_SYMLINK#../}"

    echo ""
    echo "Job submitted: $K8S_NAME (namespace: $NAMESPACE)"
    echo ""
    echo "Monitor:"
    echo "  kubectl -n $NAMESPACE get pods -l job-name=$K8S_NAME -w"
    echo "  kubectl -n $NAMESPACE logs -l job-name=$K8S_NAME -f --prefix"
    echo ""
    echo "Cancel:"
    echo "  kubectl -n $NAMESPACE delete job $K8S_NAME"
else
    echo ""
    echo "kubectl not found. Apply the manifest manually:"
    echo "  kubectl apply -n $NAMESPACE -f $MANIFEST_FILE"
fi
