# Architecture

This document describes the layered architecture that makes each component (scheduler, container runtime, training framework) independently adaptable. For concrete adaptation paths, see [Extensibility](extensibility.md).

## Training flow

```
submit.sh                      run_local.sh              in_container_run.sh
    │  (Slurm)                      │  (local)                │  (already inside container)
    ▼                               │                         │
_job.sbatch                         │                         │
    │                               │                         │
    └──► _container.sh ◄────────────┘                         │
              │  (sources container_env.sh)                   │
              │  (env overrides: DOCKER_IMAGE=… etc.)         │
              │  (launches Docker/Podman)                     │
              │                                               │
              └──────────────────────┬────────────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │ (default)             │ RAY=1
                         │                       ▼
                         │             _train_with_ray.sh
                         │                       │
                         │             ray_cluster.sh + prometheus.sh (Ray Dashboard + Prometheus + TensorBoard)
                         │                       │
                         ▼                       ▼
            _train.sh (shared entry point; sources train_env.sh + per-model env)
                         │                       │
                         ▼                       ▼
                  mfu_tracker.py       _ray_actor.py
                  (CLI wrapper)                  │
                         │             mfu_tracker.setup()
                         │                       │
                         └───────────┬───────────┘
                                     ▼
                           MaxText.train.main()
```

`run_local.sh` with a model name verifies GPU availability first; with no args it drops into an interactive shell. `in_container_run.sh` provides the same interface for use inside the container (skips the [Docker](https://www.docker.com/) launch). All three entry points share argument parsing via `utils/parse_job_args.sh`; the two `run` scripts additionally share environment setup, logging, and job summary via `utils/run_setup.sh`. See [Job Submission](job-submission.md) for usage.

`mfu_tracker` auto-detects GPU + dtype and wraps stdout/stderr to append `MFU: X.XX%` to TFLOP/s/device log lines.

### Performance: subprocess isolation

Running `MaxText.train.main()` directly inside a [Ray](https://www.ray.io/) actor causes measurable throughput degradation — Ray's internal threads (raylet, dashboard agent, log monitors) contend with [JAX](https://jax.dev/)'s training loop for the GIL. To eliminate this, `_ray_actor.py` launches training in a **subprocess** (`subprocess.Popen`), giving the training process its own interpreter with zero Ray threads. The actor itself is a thin launcher (`num_gpus=0`, `num_cpus=0`) that routes logs and collects the exit code.

**Steady-state:** with subprocess isolation, `RAY=1` has no measurable throughput impact compared to `RAY=0`.

**Initialization:** Ray cluster bootstrap, [Prometheus](https://prometheus.io/), and [TensorBoard](https://www.tensorflow.org/tensorboard) add extra startup time — negligible for production runs but noticeable for short benchmarks. Use `RAY=0` (the default) for quick benchmarks; `RAY=1` for long-running jobs or when debugging and monitoring are needed.

**Debuggability:** training runs in a subprocess, so `py-spy` must use `--subprocesses` to inspect it — this is pre-configured in `ray_cluster.sh`. The Ray Dashboard's live stack traces and CPU flame graphs work normally.

## Artifact system

Each `submit.sh` submission builds an immutable snapshot of the repository before submitting the [Slurm](https://slurm.schedmd.com/) job. This enables:

- **Rapid multi-submit** — edit code and submit more jobs without affecting pending or running ones.
- **Full traceability** — each job records exactly which code, configs, and submit command it ran with.

How it works:

1. `submit.sh` calls `build_artifact()` (in `utils/artifact.sh`), which uses `rsync` to copy the repo tree into `$JOB_WORKSPACE/.artifacts/artifact_<timestamp>_<rand>/`. The copy respects `.gitignore` at every level and excludes `.git/`, `outputs/`, and `*.trace.json*` files, keeping artifacts small and fast to build.

2. `utils/git_summary.sh` captures the current git state (branch, last commit, diffs, untracked files) into `git_summary.txt` inside the artifact. The exact submit command is saved to `submit_cmd.txt`.

3. The Slurm job runs from the artifact (via `MAXTEXT_SLURM_DIR`), not the live repo. A symlink `artifact -> ../.artifacts/artifact_...` is created in the per-job output directory for easy access.

```
$JOB_WORKSPACE/
  .artifacts/
    artifact_20260206_143052_a3f2/           # immutable copy of the repo at submit time
      submit_cmd.txt                         # exact submit command
      git_summary.txt                        # git status at submit time
  12345-JAX-llama2-70b.log                   # job log (stdout + stderr)
  12345-JAX-llama2-70b/
    artifact -> ../.artifacts/artifact_...   # symlink to the artifact
    log -> ../12345-JAX-llama2-70b.log       # symlink to the job log
    <training output files>
```

Interactive mode (`run_local.sh` with no args) and local runs do not build artifacts — they use the live code.

## Observability pipeline (Ray mode)

When `RAY=1` is set, the system persists two classes of observability data to the shared filesystem, both surviving hard kills (preemption, OOM, node failure):

- **Logs** — Ray session logs written to `$JOB_WORKSPACE/<job>/ray_logs/<hostname>/`, capturing low-level failures (NCCL hangs, C++ fatal errors) that never reach Python-level tracebacks.
- **Metrics** — A plugin-based exporter runs on every node, feeding GPU, host, network, and training metrics into a single Prometheus TSDB at `$JOB_WORKSPACE/<job>/prometheus/`. Ray's built-in exporters provide the broad baseline; `*_metrics_plugin.sh` scripts in `utils/` extend coverage automatically — no registration needed.

The stack is self-contained: Ray is `pip install`'d and Prometheus downloaded inside the container at startup — no cluster-wide pre-installation required. See [Observability](observability.md) for dashboards, alerts, the unified TSDB design, and the plugin authoring guide.

## Stage timeouts

Each job stage (preflight, pull, ecc, train) is wrapped with a configurable timeout (`utils/stage_timeout.sh`) to prevent unresponsive nodes from blocking the job forever. Defaults: preflight 900s, pull 900s, ecc 300s, train none. Override per-job: `STAGE_TIMEOUTS="preflight:600,train:86400"`. The resolved values and override syntax are printed at the top of every job log.

The utility is launcher-agnostic (pure bash) — see the header comment in `utils/stage_timeout.sh` for the API.

## Preflight and environment

Before training starts, `_job.sbatch` runs per-host preflight checks (`preflight.sh`) that clean up leftover GPU processes, prune stale containers, and tune system settings (THP, NUMA). `docker_utils.sh` auto-detects a usable container runtime — prefers [Podman](https://podman.io/) (rootless), falls back to [Docker](https://www.docker.com/). `choose_nccl_socket_ifname.sh` deterministically selects the same network interface on all nodes for NCCL/RCCL communication.

`train_env.sh` centralizes runtime environment variables (XLA, NCCL, ROCm, Transformer Engine). `_train.sh` sources it before launching training; `_env_KEY=VALUE` passthrough args override it per-run. See [Job Submission: Environment Configuration](job-submission.md#environment-configuration) for details.

## Container boundary

The default path runs training inside a container (Docker or Podman, auto-detected), with coupling confined to `_container.sh`. Container settings (image, registry, host mount paths, hotfix branch) are defined in `container_env.sh`; all variables use `${VAR:-default}` and can be overridden from the command line (e.g. `DOCKER_IMAGE=my/image:tag ./submit.sh ...`). See [Job Submission: `container_env.sh`](job-submission.md#container_envsh-docker-image-and-paths) for the full variable reference. `in_container_run.sh` provides an alternative that bypasses the container launch entirely — for use when already inside the container or any environment with JAX pre-installed. Everything upstream (`submit.sh`, `_job.sbatch`, `run_local.sh`) and downstream (`_train.sh`, `train_env.sh`) is container-agnostic. See [Extensibility: Execution Environment](extensibility.md#axis-2-execution-environment-container--native) for adaptation paths.

## Orchestrator extensibility

The codebase is layered so that scheduler coupling is confined to the orchestration tier:

| Tier | Files | Scheduler coupling |
|------|-------|--------------------|
| **Orchestration** | `submit.sh`, `_job.sbatch`, `reservation.sh`, `slurm_job_monitor.sh` | Slurm-specific (`sbatch`, `srun`, `#SBATCH`). `_job.sbatch` maps Slurm variables to generic env vars (`JOB_ID`, `NNODES`, `NODE_RANK`, etc.) before calling downstream scripts. Replace this tier for a different scheduler. |
| **Container boundary** | `_container.sh` | Scheduler-agnostic. Uses only generic env vars (set by the orchestration layer); maps `RAY` → `USE_RAY` and conditionally passes training/Ray env vars to Docker. |
| **Training** | `_train.sh`, `train_env.sh`, all Python code | Zero scheduler awareness. |
| **Utilities** | `parse_job_args.sh`, `run_setup.sh`, `artifact.sh`, `preflight.sh`, `docker_utils.sh`, `stage_timeout.sh` | No scheduler dependency. |

To add a new scheduler (e.g., Kubernetes), only the orchestration tier needs a parallel implementation — write a new entry point that sets the same generic env vars and calls `_container.sh`. See [Extensibility](extensibility.md) for details on each adaptation axis.
