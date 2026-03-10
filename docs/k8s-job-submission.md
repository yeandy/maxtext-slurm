# Kubernetes job submission

Submit multi-node training jobs to [Kubernetes](https://kubernetes.io/) using `k8s_submit.sh` — the k8s equivalent of `submit.sh`.

## Prerequisites

- `kubectl` configured with cluster access
- A **PersistentVolumeClaim (PVC)** backed by a shared filesystem (NFS, Lustre, CephFS) containing the repo and outputs
- AMD or NVIDIA **GPU device plugin** installed in the cluster

## Quick start

```bash
export K8S_PVC_NAME=shared-data
export K8S_PVC_MOUNT=/mnt/shared
export REPO_PATH_IN_PVC=yihuang/maxtext-slurm

./k8s_submit.sh 70b -N 4
./k8s_submit.sh 70b -N 4 -- steps=20 dataset_type=synthetic
RAY=1 ./k8s_submit.sh 70b -N 4                                # with observability
./k8s_submit.sh 70b -N 4 -- enable_checkpointing=true         # with checkpointing
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `K8S_PVC_NAME` | Yes | — | PVC name for shared storage |
| `K8S_PVC_MOUNT` | No | `/mnt/shared` | Mount path inside pods |
| `REPO_PATH_IN_PVC` | No | `$(basename $SCRIPT_DIR)` | Repo path relative to PVC root |
| `K8S_NAMESPACE` | No | `default` | Kubernetes namespace |
| `K8S_NODE_SELECTOR` | No | — | Node selector as `key=value` |
| `DOCKER_IMAGE` | No | from `container_env.sh` | Override container image |
| `JOB_WORKSPACE` | No | `<repo>/outputs` | Output directory inside pods |
| `JAX_COORDINATOR_PORT` | No | `29500` | JAX coordinator port |
| `GPUS_PER_NODE` | No | `8` | GPUs per node |
| `GPU_RESOURCE_TYPE` | No | `amd.com/gpu` | K8s GPU resource name (`nvidia.com/gpu` for NVIDIA) |
| `RAY` | No | — | Enable Ray observability (`RAY=1`) |
| `K8S_BARRIER_TIMEOUT` | No | `120` | Seconds to wait for coordinator/barrier |

## How it works

```
k8s_submit.sh                         # user runs this
  ├─ parse args, resolve model        # same utils as submit.sh
  ├─ build artifact                   # freeze repo snapshot
  ├─ generate Indexed Job manifest
  └─ kubectl apply

        k8s schedules pods → pulls image → starts containers

_k8s_job.sh                           # runs inside each pod
  ├─ coordinator discovery            # rank 0 writes IP to shared filesystem
  ├─ barrier                          # all pods wait for each other
  ├─ dataset alias                    # /datasets symlink
  └─ in_container_run.sh              # shared code from here down
       └─ _train.sh or _train_with_ray.sh
```

## Monitoring and cancellation

The script prints these commands after submission:

```bash
kubectl -n default get pods -l job-name=<K8S_NAME> -w       # watch pods
kubectl -n default logs -l job-name=<K8S_NAME> -f --prefix  # stream all logs
kubectl -n default delete job <K8S_NAME>                    # cancel
```

## Log layout

Rank 0 writes the primary job log (parsed by `tgs_tagger.py`, `analyze_job.py`). Non-rank-0 pods write per-rank logs inside the job directory for debugging.

```
outputs/
├── k8s-xxx-JAX-llama2-70b.log                   ← rank 0 (tools parse this)
└── k8s-xxx-JAX-llama2-70b/
    ├── rank-1.log                               ← rank 1
    ├── rank-N.log                               ← rank N
    ├── .coord/                                  ← coordinator discovery files
    ├── artifact -> ../.artifacts/...            ← frozen repo snapshot
    └── llama2-70b_train_test/                   ← tensorboard, metrics
```

For real-time per-pod output, use `kubectl logs <pod-name>`.

## Differences from Slurm

| | `submit.sh` (Slurm) | `k8s_submit.sh` (Kubernetes) |
|---|---|---|
| Scheduler | `sbatch` | `kubectl apply` (Indexed Job) |
| Node selection | `-p partition`, `-w nodelist` | `K8S_NODE_SELECTOR=label=value` |
| Cancel | `scancel <id>` | `kubectl delete job <name>` |
| Job log | All ranks via `srun -l` (rank-prefixed) | Rank 0 primary + per-rank files |
| Job queuing | Slurm scheduler (priority, fair-share) | First-come-first-served (add [Kueue](https://kueue.sigs.k8s.io/) for queuing) |
| Preflight / ECC | `preflight.sh`, `check_ecc.sh` via `srun` | Not implemented |
| Image pull barrier | `srun` pull-only stage | Ready-file barrier |

Features at parity: model configs, per-model env overrides (`configs/*.env.sh`), passthrough args (`-- key=value`), `_env_` overrides (e.g., `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.90`), `MAXTEXT_PATCH_BRANCH`, artifacts, observability (`RAY=1`), checkpointing, dataset alias, coredump setup, all analysis tools. See [Job Submission](job-submission.md) for full details on these shared features.

## Interactive / direct-container runs

When you're already inside a running pod (via `kubectl exec` or a pre-existing container), use `in_container_run.sh` or `debug_repro.sh` directly:

```bash
kubectl exec -it <pod> -- bash

# inside the pod:
export JOB_WORKSPACE=/mnt/shared/maxtext-slurm/outputs
./in_container_run.sh 70b -- steps=5
```

Multi-node additionally requires: `NNODES`, `NODE_RANK`, `JAX_COORDINATOR_IP`, `JAX_COORDINATOR_PORT`.

## Coredumps in Kubernetes

- `COREDUMP_DIR` is honored when valid and writable.
- If unset or invalid, fallback order is: `/coredump` → `$JOB_WORKSPACE` → `/tmp`.
- Coredump setup is best-effort. In restricted pods, writing `/proc/sys/kernel/core_pattern` may be blocked.

For persistent coredumps, set `COREDUMP_DIR` to a path on the PVC:

```bash
export COREDUMP_DIR=/mnt/shared/coredumps
```

## PVC setup example

One-time admin setup:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: shared-nfs-pv
spec:
  capacity:
    storage: 200Ti
  accessModes: [ReadWriteMany]
  storageClassName: nfs
  persistentVolumeReclaimPolicy: Retain
  nfs:
    server: 10.0.0.1
    path: /exports/shared
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-data
spec:
  accessModes: [ReadWriteMany]
  storageClassName: nfs
  resources:
    requests:
      storage: 200Ti
  volumeName: shared-nfs-pv
```
