# Kubernetes direct-container runs

This repo is Slurm-first. This note documents the minimal setup for running directly inside [Kubernetes](https://kubernetes.io/) containers using [`in_container_run.sh`](../in_container_run.sh) or [`debug_repro.sh`](../debug_repro.sh).

## Scope

- Use this only when the container is already running (pod shell, init command, or `kubectl exec`).
- For multi-node, Kubernetes must provide rank/orchestration env vars; these scripts are per-pod runners.

## Required environment

For Kubernetes direct-container runs, recommended:

- `JOB_WORKSPACE` — writable persistent path (PVC recommended). If unset, defaults to `outputs/` in the repo.

Multi-node additionally requires:

- `NNODES`
- `NODE_RANK`
- `JAX_COORDINATOR_IP`
- `JAX_COORDINATOR_PORT`

Recommended metadata:

- `JOB_ID`
- `JOB_NAME`

Example:

```bash
export JOB_WORKSPACE=/mnt/pvc/maxtext
export JOB_ID=k8s_$(date +%Y%m%d_%H%M%S)
export JOB_NAME=JAX-llama2-70b
export NNODES=2
export NODE_RANK=0
export JAX_COORDINATOR_IP=maxtext-rank0.default.svc.cluster.local
export JAX_COORDINATOR_PORT=30000
./in_container_run.sh 70b -- steps=10
```

## Coredumps in Kubernetes

- `COREDUMP_DIR` is honored when valid and writable.
- If unset or invalid, fallback order is: `/coredump` -> `$JOB_WORKSPACE` -> `/tmp`.
- Coredump setup is best-effort. In restricted pods, writing `/proc/sys/kernel/core_pattern` may be blocked and host policy may control destination.
- Runtime logs include `setup_mode=best_effort`, requested/active `core_pattern`, and apply status to make this explicit.

If you need persistent coredumps, mount a PVC and set:

```bash
export COREDUMP_DIR=/mnt/pvc/coredumps
```
