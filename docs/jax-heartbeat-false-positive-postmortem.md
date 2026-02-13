# JAX Distributed Heartbeat False-Positive Kill: Post-Mortem

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why Debugging Is So Hard](#2-why-debugging-is-so-hard)
3. [Building Observability to Break the Deadlock](#3-building-observability-to-break-the-deadlock)
4. [Root Cause Analysis](#4-root-cause-analysis)
5. [Conclusion](#5-conclusion)
6. [Practical Workarounds](#6-practical-workarounds)

**Appendix**

- [Source Code Inspection and Suspicion](#appendix-source-code-inspection-and-suspicion)

---

## 1. Problem Statement

We run large-scale distributed training using
**[MaxText](https://github.com/AI-Hypercomputer/maxtext)** on
**[JAX](https://jax.dev/)** across 24 nodes (192 GPUs), orchestrated by
**[Ray](https://www.ray.io/)** on a **[Slurm](https://slurm.schedmd.com/)** cluster.
Most jobs complete successfully, but two jobs with identical configurations were
killed mid-training with the same mysterious error:

```
UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats):
  /job:jax_worker/replica:0/task:X
  /job:jax_worker/replica:0/task:Y
The tasks have crashed. Check the task logs for an earlier error,
or scheduler events (e.g. preemption, eviction) to debug further.
```

| | Job A | Job B |
|---|---|---|
| Crash time | 19:58:10 UTC | 23:48:08 UTC |
| Step reached | 71 | 398 |
| "Dead" tasks | 10, 16 | 4, 5, 18 |
| Heartbeat timeout | 900 s | 900 s |
| Checkpointing | Restored from ckpt 0 | Restored from ckpt 0 + saved ckpt 200 |

Both jobs ran with `jax_distributed_heartbeat_timeout_seconds=900` and
`enable_checkpointing=true`, targeting 5,000 training steps.

The error says the tasks "crashed." That turned out to be **false** — they were alive
and actively training when they were killed. This post-mortem documents how we proved
that, what the actual root cause is, and what to do about it.

---

## 2. Why Debugging Is So Hard

### The log tells you almost nothing

The only error is the heartbeat timeout message above. It appears simultaneously on
every node — including the "dead" tasks themselves. There is:

- **No earlier error** on the accused tasks. They were running normally.
- **No stack trace** from the heartbeat mechanism.
- **No timing information** about when heartbeats actually stopped — only when the
  timeout fired, 900 seconds later.
- **No differentiation** between "process crashed" and "heartbeat RPC failed." The
  message reads "The tasks have crashed" in both cases.
- **No heartbeat thread diagnostics.** The heartbeat runs in a C++ thread inside
  [XLA](https://openxla.org/). It emits no log when it sends a heartbeat, when an
  RPC is slow, or when it blocks.

### The misleading error message

The coordinator says "The tasks have crashed." But in both jobs, the accused tasks
**received and logged their own death notification** via the `PollForError` RPC:

```
# Task 10 (job A) — supposedly "crashed" — logging its own death:
E0212 19:58:10 coordination_service_agent.cc:310] Polled error from coordination service
F0212 19:58:10 client.h:77] Terminating process because the JAX distributed service
  detected fatal errors.
  /job:jax_worker/replica:0/task:16
  /job:jax_worker/replica:0/task:10   ← task 10 reports itself as "crashed"
```

A truly crashed process cannot log its own death. This was the first clue that the
error message was wrong.

### The 900-second blind spot

With `heartbeat_timeout_seconds=900`, the heartbeat interval is `900 / 2 = 450`
seconds (7.5 minutes), per the XLA implementation (`heartbeat_interval =
timeout / 2`). This means:

- Heartbeats are sent only **~2 times per timeout window**.
- The actual failure occurred **up to 15 minutes before** the crash log appears.
- By the time you see the error, whatever caused the heartbeat to stop is long gone.
- There is zero observability into what happened during those 15 minutes.

Without system-level metrics, you are left guessing.

---

## 3. Building Observability to Break the Deadlock

When we first hit this issue, we had no way to investigate. The error message was all
we had, and it was misleading. We could not answer basic questions: Were the "dead"
tasks actually doing work? Was the network healthy? Was the coordinator under load?

The application logs are silent on all of this. The heartbeat runs in a C++ thread
deep inside XLA — there is no Python traceback, no application-level logging, no
metric. We were stuck.

This gap is what motivated us to build the
[Prometheus-based observability stack](observability.md) that now ships with every
job. It collects system-level metrics — GPU utilization, host memory, TCP health,
I/O pressure, RDMA counters — from all nodes into a persistent TSDB, queryable both
live and post-hoc. When the application logs have nothing useful, you need an
independent source of truth.

For this investigation, the critical metrics were:

| Metric | Question it answers |
|---|---|
| `ray_node_gpus_utilization` | Were the "dead" tasks still training? |
| `ray_node_mem_used` | Did checkpoint D2H transfers stress the system? |
| `hw_io_pressure_full_pct` | Was checkpoint I/O blocking processes? |
| `hw_tcp_retransmits_total` | Was the network degraded when heartbeats stopped? |
| `hw_tcp_listen_drops_total` | Was the coordinator's [gRPC](https://grpc.io/) server overloaded? |

Using [`prometheus.sh view`](tooling.md#inspect-prometheus-metrics-after-a-job-ends)
to spin up read-only Prometheus instances against the completed jobs' TSDBs, we
queried each metric at the exact time the heartbeats stopped. The answer was the same
every time: **everything was clean.** The absence of any anomaly was itself the most
important clue — it ruled out infrastructure and pointed squarely at the heartbeat
mechanism itself.

The observability stack was built to debug this specific crash. But the
infrastructure it provides — per-node system metrics persisted with every job,
queryable post-hoc — turns out to be broadly useful for any future failure mode:
NCCL hangs, GPU thermal throttling, NFS latency spikes, network degradation, or
silent performance regressions. See [Observability](observability.md) for the full
design.

---

## 4. Root Cause Analysis

### 4.1 Job A: Tasks 10 and 16 declared dead at step 71

**Step 1 — Read the logs.**

The coordinator declared tasks 10 and 16 dead at 19:58:10 for "stopped sending
heartbeats." All 24 tasks across all nodes were killed.

**Step 2 — Check if the "dead" tasks were actually dead.**

Both tasks logged receiving their own death notification via `PollForError`. A
crashed process cannot do this — they were alive.

Their stdout confirmed training proceeded normally through step 71 at ~30 s/step.

**Step 3 — Reconstruct the heartbeat timeline.**

- Crash at 19:58:10, timeout = 900 s → last successful heartbeat at ~19:43.
- Checkpoint restore (D2H transfer) completed by ~19:17.
- Training started at ~19:20.
- At 19:43, training had been running cleanly for ~23 minutes.
- The next heartbeat attempt at ~19:50 (450 s after the last success) never arrived.

**Step 4 — Query Prometheus at the failure time (~19:43–19:50).**

| Metric at ~19:50 | task 10 | task 16 | coordinator |
|---|---|---|---|
| GPU utilization | 100% | 100% | 100% |
| TCP retransmit rate | ~0/s | ~0/s | ~0/s |
| I/O pressure | 0% | 0% | 0% |
| Memory | 205 GB | 204 GB | 207 GB |
| TCP listen drops | flat | flat | flat |

**Everything was clean.** No network issues, no I/O stalls, no memory pressure.
The heartbeat simply stopped arriving for no externally observable reason.

### 4.2 Job B: Tasks 4, 5, and 18 declared dead at step 398

**Step 1 — Read the logs.**

Same pattern. Tasks 4, 5, and 18 were declared dead at 23:48:08. The job had
completed step 398 and was entering the step-400 checkpoint.

**Step 2 — Check if the "dead" tasks were actually dead.**

- Task 4: completed step 398, entered checkpoint, logged its own death notification.
  **Alive.**
- Task 18: identical. **Alive.**
- Task 5: no worker log files (Ray logging issue), but Prometheus showed 100% GPU
  utilization until 23:44 and 65% I/O pressure during checkpoint. **Active.**

**Step 3 — Discover the checkpoint memory pattern.**

Job B saved a checkpoint at step 200, revealing a dramatic memory split:

| Node group | Count | Steady-state memory | Role |
|---|---|---|---|
| DP replica #0 (tasks 0–7) | 8 | 1,350–1,395 GB | D2H transfer for checkpoint save |
| Others (tasks 8–23) | 16 | ~204 GB | Normal training |

The save caused a **1.47 TB peak** on DP#0 nodes, and the D2H memory was **retained**
(~1.37 TB) for the rest of training. Step 201 took **772 seconds** (vs normal 30 s)
because the checkpoint blocked the main thread.

By contrast, job A only restored from step 0 — the memory spike was transient
(peak 249 GB, freed back to ~95 GB immediately). Checkpoint **saves** create a
persistent ~1.37 TB footprint on DP#0 nodes; **restores** do not.

**Step 4 — Query Prometheus at the heartbeat failure time.**

Crash at 23:48:08 → heartbeats stopped at ~23:33. Metrics at 23:33:

| Metric | task 4 | task 5 | task 18 | coordinator |
|---|---|---|---|---|
| GPU utilization | 100% | 100% | 100% | 100% |
| TCP retransmit rate | 0.15/s | 0.03/s | 0.12/s | ~0/s |
| I/O pressure | 0% | 0% | 0% | 0% |
| Memory | 1,379 GB | 1,393 GB | 204 GB | 1,350 GB |
| CPU utilization | 25.9% | 24.3% | 25.9% | 26.9% |

**Again, everything was clean.** Heartbeats stopped during normal training with no
observable system-level anomaly.

**Step 5 — Rule out checkpoint I/O as the trigger.**

The step-400 checkpoint started at ~23:45 (I/O pressure spike, TCP retransmit storm
of 780–2,134/s). But heartbeats had already stopped at ~23:33 — **12 minutes before**
the checkpoint began. The checkpoint I/O was a consequence of training reaching
step 400, not the cause of the heartbeat failure.

### 4.3 Cross-job comparison

| | Job A | Job B |
|---|---|---|
| Failed tasks | 2 (tasks 10, 16) | 3 (tasks 4, 5, 18) |
| Tasks alive at death? | Yes (both) | Yes (at least 2 of 3) |
| System metrics at HB stop | All clean | All clean |
| Checkpoint save before failure? | No (restore only) | Yes (step 200, 772 s stall) |
| Failed tasks on DP#0? | No (both non-DP#0) | 2 of 3 on DP#0 |
| Heartbeat cycles before failure | ~7 | ~29 |

The failures share the same signature: heartbeats silently stop on a random subset of
tasks while every system metric looks healthy. The accused tasks prove they were alive
by logging their own death notifications.

---

## 5. Conclusion

**These are false-positive heartbeat timeout kills**, not actual task crashes. The JAX
coordination service's heartbeat mechanism has design flaws that cause it to declare
healthy, actively-training tasks as dead. The consequence is catastrophic: **all 24
tasks across all nodes are killed**, destroying training progress since the last
checkpoint.

The false-positive rate is high. With `heartbeat_interval = timeout/2 = 450 s`, only
~2 heartbeats fit in the timeout window. A single failed or hung RPC is
unrecoverable — the task is killed 900 seconds later with no retry.

Estimating from our two incidents (~7 and ~29 heartbeat cycles respectively), the
per-heartbeat failure probability is on the order of **5–15%**, making it virtually
certain that any long-running job will eventually be killed:

| Job duration | Heartbeat cycles | P(false-positive kill) |
|---|---|---|
| 30 minutes | 4 | ~34% |
| 1 hour | 8 | ~57% |
| 3 hours | 24 | ~92% |
| 12 hours | 96 | ~100% |

Checkpointing-enabled jobs are disproportionately affected because they:

1. **Run longer** — more heartbeat cycles, more chances to hit the bug.
2. **Create gRPC contention** during checkpoint save/restore — D2H
   transfers, barrier coordination, and heavy I/O all compete on the shared gRPC
   channel.
3. **Leave persistent memory footprints** on DP replica #0 nodes (1.37 TB vs 204 GB)
   after saves, altering the operating environment for the rest of training.

These factors compound: longer runtime increases exposure, while checkpoint operations
elevate the per-heartbeat failure probability.

---

## 6. Practical Workarounds

Since modifying JAX/XLA source code is not feasible for us (see
[Appendix](#appendix-source-code-inspection-and-suspicion) for the suspected bugs),
we work around the issue at the operational level.

### 6.1 Set an extremely large heartbeat timeout

The timeout should be large enough that a false-positive kill never destroys
irrecoverable training progress. At minimum, it should cover one full checkpoint
cycle:

```
T_hb >= alpha * (W + C_r + P * S + C_s)
```

| Symbol | Meaning | Example value |
|---|---|---|
| W | Warmup overhead (compilation, init) | ~120 s |
| C_r | Checkpoint restore time | ~180 s |
| P | Checkpoint period (steps) | 200 |
| S | Time per training step | 30 s |
| C_s | Checkpoint save time | 772 s |
| alpha | Safety factor | >= 2 |

**Example:** `T_hb >= 2 * (120 + 180 + 200*30 + 772) = 2 * 7,072 = 14,144 s` (~4
hours).

With a 4-hour timeout, `heartbeat_interval = timeout/2 = 2 hours`. Heartbeats become
essentially cosmetic — but since the mechanism is unreliable, this prevents it from
killing productive training.

**Trade-off:** a truly crashed task won't be detected for 4 hours. This is acceptable
because we have independent hang detection (see below).

### 6.2 Independent hang detection

The heartbeat cannot detect actual hangs (e.g., NCCL/RCCL deadlocks) because the
process remains alive during a hang. We use
[`slurm_job_monitor.sh`](tooling.md#monitor-a-slurm-job-with-telegram-alerts) as an
independent monitor: it watches training log `mtime` and fires a Telegram alert if
output stalls for 30 minutes. This provides the hang detection the heartbeat was
supposed to deliver, without the false-positive kill risk.

### 6.3 Defense in depth

| Threat | Detection | Response |
|---|---|---|
| Task crash | JAX heartbeat (very large timeout) | Delayed kill (acceptable) |
| NCCL/RCCL hang | `slurm_job_monitor.sh` (mtime-based) | Telegram alert → manual intervention |
| Heartbeat false-positive | Large timeout prevents the kill | Training continues safely |
| Checkpoint failure | I/O pressure metrics in [Prometheus](observability.md) | Post-hoc diagnosis |
| Hardware error | GPU RAS / PCIe metrics in [Prometheus](observability.md) | Post-hoc diagnosis |

By setting the heartbeat timeout to several hours and relying on
`slurm_job_monitor.sh` for hang detection, we decouple the broken heartbeat mechanism
from training availability while maintaining full observability through Prometheus for
post-hoc root cause analysis.

---

## Appendix. Source Code Inspection and Suspicion

We examined the open-source JAX/XLA coordination service to understand the heartbeat
mechanism and identify potential bugs. Versions inspected: **JAX
0.8.2**, **jaxlib 0.8.2** (self-built), XLA headers from
**[TensorFlow](https://www.tensorflow.org/) 2.19.1**.

### Architecture

```
Python: jax.distributed.initialize()
  → C++: DistributedRuntimeCoordinationServiceClient (client.cc)
    → C++: CoordinationServiceAgent (coordination_service_agent.cc)
      → gRPC channel → CoordinationService (coordination_service.cc) on coordinator
```

All coordination RPCs — **Heartbeat, PollForError, Barrier, KeyValueGet/Set** — share
a **single `grpc::Channel`** (one TCP connection via HTTP/2
multiplexing).

### Bug 1: Single heartbeat failure is instantly fatal

In `coordination_service_agent.cc`, the heartbeat loop:

```cpp
leader_client_->HeartbeatAsync(&call_opts, &request, &response,
    [&](const absl::Status& s) { status = s; n.Notify(); });
n.WaitForNotification();

if (!status.ok()) {
    absl::SleepFor(absl::Seconds(1));
    if (!shutting_down_) {
        SetError(status);   // triggers LOG(QFATAL) → process killed
    }
}
```

A single non-OK RPC response kills the process immediately. No retries, no backoff,
no second chance.

### Bug 2: Heartbeat interval = timeout / 2

```cpp
const int64_t heartbeat_interval_ms =
    configs_.heartbeat_timeout_in_ms() > 0
        ? configs_.heartbeat_timeout_in_ms() / 2    // 900s / 2 = 450s = 7.5 minutes
        : absl::ToInt64Milliseconds(kDefaultHeartbeatTimeout) / 2;
```

With `heartbeat_timeout_seconds=900`, heartbeats are sent every **7.5 minutes**. Only
~2 fit in the timeout window — compared to typical distributed systems that heartbeat
every few seconds.

### Bug 3 (most likely root cause): Shared gRPC channel blocks heartbeats

All coordination RPCs share a single gRPC channel. `PollForError` is a **long-running
blocking call** that occupies a gRPC completion thread:

```cpp
leader_client_->PollForErrorAsync(..., [&](const absl::Status& s) {
    status = s; n.Notify();
});
n.WaitForNotification();  // blocks for the entire job lifetime
```

When the heartbeat thread calls `HeartbeatAsync`, its completion callback enters the
same gRPC completion queue. If the completion thread is servicing `PollForError`, the
heartbeat callback may never fire, causing the heartbeat thread to block indefinitely
on `n.WaitForNotification()`.

**This explains why tasks are alive but heartbeats stop.** Training continues on the
GPU while the heartbeat thread is stuck waiting for a callback that never comes.

### Bug 4: Server-side lock contention

On the coordinator, the `state_mu_` mutex serializes heartbeat processing with the
`CheckStaleness()` thread:

```cpp
void CoordinationService::CheckStaleness() {
    while (true) {
        absl::MutexLock l(state_mu_);
        check_staleness_thread_cv_.WaitWithTimeout(&state_mu_, absl::Seconds(1));
        CheckHeartbeatTimeout();
        CheckBarrierStatusWithRecoverableTasks();
        CheckBarrierTimeout();
    }
}
```

Incoming `RecordHeartbeat()` calls must acquire the same lock, creating contention
that can delay heartbeat recording.

### Design flaw 5: No heartbeat thread watchdog

No mechanism detects if a task's own heartbeat thread has stopped functioning.
Training runs indefinitely while the heartbeat thread is stuck — no warning until the
server-side timeout fires 900 seconds later.

### Design flaw 6: Misleading error message

The error says "The tasks have crashed" when they have only stopped sending
heartbeats. These are very different failure modes, but the message conflates them,
sending investigators down the wrong path.

### Reconstruction of the failure sequence

1. The heartbeat thread sends a successful heartbeat at time T.
2. It sleeps for 450 seconds.
3. At T+450, the next `HeartbeatAsync` RPC is submitted to the shared gRPC channel.
4. The gRPC completion thread is busy with `PollForError` or blocked on coordinator
   lock contention. The heartbeat callback never fires.
5. The heartbeat thread blocks on `n.WaitForNotification()` indefinitely.
6. Training continues on the GPU. All system metrics remain healthy.
7. At T+900, the coordinator's `CheckHeartbeatTimeout()` declares the task dead.
8. The error propagates via `PollForError` to all tasks — including the "dead" ones,
   which log their own death notification and terminate.

---

*This post-mortem was written with [Cursor](https://www.cursor.com/) +
[Claude-4.6-opus-high](https://www.anthropic.com/claude).*
