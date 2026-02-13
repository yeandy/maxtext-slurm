# Debugging

Workflows for diagnosing training job failures. For system-level and training metrics (GPU thermals, network stats, loss curves, throughput) and log analysis, see [Observability](observability.md) — including [post-run diagnostics](observability.md#post-run-diagnostics) for querying Prometheus and Ray logs after a job ends. For a real-world case study where observability-driven debugging uncovered a [JAX](https://jax.dev/) heartbeat bug, see the [heartbeat false-positive post-mortem](jax-heartbeat-false-positive-postmortem.md).

## Core dumps

**For multi-node crashes or crashes that occur after the first training step, core dumps are the only diagnostic.** Make sure the core dump path is properly configured before running production jobs.

### Where core dumps go

Core dumps are written to the first directory with >500 GB free, checked in order: the per-job output directory, the outputs root (`$JOB_WORKSPACE/`), then any paths in `COREDUMP_EXTRA_DIRS` (configured in `container_env.sh`). The selected directory is bind-mounted as `/coredump` inside the container.

To add a cluster-specific coredump directory, append it to `COREDUMP_EXTRA_DIRS` in `container_env.sh`.

Core dumps are enabled automatically — no manual setup required. Notes:

- The path selection is made independently on each node.
- If the path points to local disk, SSH into the corresponding compute node to access the core files.

### Waiting for writes

Core dumps for JAX/XLA processes are large (35–120 GB per process). The system automatically waits for writes to finish before the container exits — no data is lost even on NFS. If no core files exist, there is zero delay.

| Variable | Default | Description |
|----------|---------|-------------|
| `COREDUMP_DIR` | `/coredump` | Directory inside the container (mounted from the job dir) |
| `COREDUMP_WAIT_TIMEOUT` | `900` (15 min) | Max seconds to wait for core dump writes to finish |

### Inspect a core dump

Core dump files must be inspected **inside the same [Docker](https://www.docker.com/) image** used by the crashed job.

```bash
# Step 0: Launch an interactive Docker container.
run_local.sh

# Step 1: Find the core dump from the crashed job.
# Core files from script-mode jobs are inside per-job subdirectories under /outputs.
ls -lt /outputs/*/core*py* 2>/dev/null

# Step 2: Use gdb to inspect it.
gdb python3 "<path from step 1>"
```

### Crash reproduction

Only works for single-node crashes that happen before the first training step.

```bash
# Step 0: Reserve a compute node (skip if not on a Slurm cluster).
srun -N 1 --exclusive --pty /bin/bash

# Step 1: Launch an interactive Docker container.
run_local.sh

# Step 2: Inside the Docker container, run the debug job.
# This runs a 1-step training job in an infinite loop until a crash occurs.
# (Pre-loaded in bash history — press ↑ to recall.)
debug_repro.sh

# Step 3: After the crash, inspect the generated core dump.
# (Also pre-loaded in bash history — press ↑ to recall.)
gdb python3 "$(ls -t $COREDUMP_DIR/core*py* | head -n1)"
```

If you need to use a different Docker image for debugging, update `DOCKER_IMAGE` in `container_env.sh`. For private images, see [Job Submission: `container_env.sh`](job-submission.md#container_envsh-docker-image-paths) for credential setup.

## Unresponsive nodes

**Symptom.** A stage times out or hangs. Stage timeouts catch this automatically — the job exits with a `FATAL` message identifying the node list.

**Diagnosis.** Find the missing task number in the job output (each node prefixes output with its task id, e.g. `16:`). Map task number to hostname and check node state for signs of unresponsiveness.

**Recovery.** Drain the bad node and resubmit. See [Job Submission: Stage Timeouts](job-submission.md#stage-timeouts) to tune timeout values.
