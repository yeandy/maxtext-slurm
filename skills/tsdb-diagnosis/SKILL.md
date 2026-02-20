---
name: tsdb-diagnosis
description: Diagnose training job incidents and check cluster health using the per-job Prometheus TSDB. Use when the user asks to diagnose a failure root cause, check GPU/network health, query Prometheus metrics, investigate a hang, or when the triage skill recommends deeper TSDB analysis.
---

# TSDB Diagnosis

Query the per-job Prometheus TSDB to diagnose incident root causes and assess cluster health. Works on any `RAY=1` job — finished or running.

**Relationship to other skills:**
- **job-log-triage** identifies *what* failed (from logs). This skill identifies *why* using time-series metrics.
- **performance-analysis** identifies *why training is slow* (from xplane/HLO traces). This skill identifies *why metrics are anomalous* at the system level.
- The triage skill's next-step templates often say "Query Prometheus TSDB" — this skill automates that.

## Prerequisites

This skill requires a Prometheus TSDB, which is only available for `RAY=1` jobs. If the job was launched with `RAY=0`, no TSDB exists — report this and suggest re-running with `RAY=1`.

## Workflow

0. **Triage first.** Before querying the TSDB, run the **job-log-triage** skill on the job (or confirm triage has already been done earlier in the conversation). Triage establishes critical context that shapes the entire diagnosis:
   - **Job state** — running, completed, failed, cancelled, or crashed (determines how to connect to Prometheus in step 2)
   - **Fresh start vs checkpoint restore** — affects expected baseline (restored jobs may leak RCCL resources; see "Checkpointing interference")
   - **Failure class** — hang, heartbeat-timeout, OOM, etc. (determines which playbook to run in step 5)
   - **Step range and timing** — which steps are comparable, where anomalies occurred
   - **Config parameters** — `PASSTHROUGH_ARGS`, `NNODES`, `checkpoint_period`, etc.

   Without triage, you risk misinterpreting metrics (e.g., querying post-hang idle metrics as if they were training metrics, or missing that a checkpoint restore caused resource leaks). For proactive health checks on a live job where no failure has occurred, triage is still useful to confirm the job is running and extract the head node hostname/port.

1. **Locate the TSDB.** Resolve the job directory (same rules as triage: given a log file, job dir, or Slurm ID). Verify `<job_dir>/prometheus/` exists **and contains data** — it should have subdirectories with ULID names (e.g., `01KHV6MFN61MJKZ3ZSYYRYGDGX`) and/or a `wal/` directory. An empty or near-empty `prometheus/` directory means Prometheus failed to start or never scraped — report "no TSDB data" and skip to log-only diagnosis.

2. **Connect to Prometheus.** First determine whether the job is still running, then choose the appropriate access method.

   **Determine job state** (from triage, or verify directly): The triage report from step 0 tells you whether the job is running or finished. If you need to verify independently:
   - `squeue -j <id> -h -o %T 2>/dev/null` — if it returns `RUNNING`, the job is live → **Case A**.
   - If `squeue` is unavailable or the job isn't found, check the log: a `JOB SUMMARY` block means the job is finished → **Case B**. No summary and no advancing steps → likely crashed → also **Case B** (the live Prometheus is gone).

   **Critical rule: never delete a TSDB lock file, and never run `prometheus.sh view` against a running job's TSDB.** The lock is held by the live Prometheus instance. Deleting it or starting a second instance against the same TSDB risks data corruption. For finished/crashed jobs, `prometheus.sh view` handles stale lock files automatically (Prometheus detects and replaces them).

   **Case A — Live job (still running):** Prometheus is already running on the job's head node (task 0). Find the head node hostname and port from the SSH tunnel command in the job log (`ssh -L ... <head_host>:<port>`). The port is usually 9190 but may differ if 9190 was occupied at job start (the startup script auto-increments: 9191, 9192, ...). Also check for `[Prometheus] WARNING: port 9190 was occupied; using <port> instead` in the log. Query at `http://<head_host>:<port>` — **not** `localhost:<port>`. The head node hostname comes from the log (e.g., `chi2816`); `localhost:9090` on the machine you are running from may be a completely different Prometheus (e.g., a cluster-level monitor). The SSH tunnel in the log is for the **user's laptop** to reach the head node through a jump host — it binds the port on the user's local machine, not on the head node or any intermediate host. Do **not** start a second Prometheus instance for a live job.

   **Case B — Finished job (completed, failed, or cancelled):** Start a read-only Prometheus against the persisted TSDB:
   ```bash
   utils/prometheus.sh view <job_dir>/prometheus -p <port> &
   ```
   Use port 9190 by default; if occupied, try 9191, 9192, etc. Wait for "Open http://localhost:" in stdout (typically <3 seconds). Kill the process when done (step 7).

   **Case C — Multi-job comparison:** At least one job must have a Prometheus TSDB. Start a read-only Prometheus for each finished job that has one, each on a different port:
   ```bash
   utils/prometheus.sh view <job_A_dir>/prometheus -p 9190 &
   utils/prometheus.sh view <job_B_dir>/prometheus -p 9191 &
   ```
   If one of the jobs is still live, query its existing Prometheus at `http://<head_host>:<port>` (Case A) alongside the read-only instances on localhost. Jobs without a TSDB (`RAY=0`) can only be compared using log-based data from the triage skill. Query each Prometheus on its own address/port and compare results side by side.

3. **Always query all metrics first.** Before any diagnostic work, discover what the TSDB contains. This tells you which metric families were collected, how many nodes are present, and what time range is covered — essential context for choosing the right queries and interpreting results.

   In the examples below, replace `<host>:<port>` with the address from step 2: `<head_host>:<port>` for live jobs (Case A), or `localhost:<port>` for read-only instances (Cases B/C).

   **For read-only instances (Case B/C), always start with `api/v1/status/tsdb`** to find the time range before any data queries. This endpoint doesn't need a timestamp and tells you the min/max times the TSDB covers. Without this, you won't know what `&time=` values to use for instant queries.

   ```bash
   # All available metric names — understand what the observability stack captured
   curl -s 'http://localhost:<port>/api/v1/label/__name__/values' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   names = sorted(data.get('data', []))
   print(f'Total metrics: {len(names)}')
   for prefix in ['hw_gpu_', 'hw_tcp_', 'hw_rdma_', 'hw_io_', 'hw_mem_', 'hw_oom_', 'hw_net_', 'hw_procs_', 'hw_dmesg_', 'ray_node_', 'tb_']:
       group = [n for n in names if n.startswith(prefix)]
       if group: print(f'  {prefix}*: {len(group)} metrics')
   other = [n for n in names if not any(n.startswith(p) for p in ['hw_', 'ray_', 'tb_', 'up', 'scrape_'])]
   if other: print(f'  other: {other}')
   "

   # All hosts in the TSDB
   curl -s 'http://localhost:<port>/api/v1/label/host/values' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   hosts = sorted(data.get('data', []))
   print(f'Hosts ({len(hosts)}): {', '.join(hosts)}')
   "

   # Time range of available data — use min/max timestamps from the TSDB.
   # For live jobs (Case A), omit &time to use current time.
   # For read-only instances (Case B/C), first find the TSDB time range:
   curl -s 'http://localhost:<port>/api/v1/status/tsdb' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   d = data.get('data', {})
   min_t = d.get('minTime', 0) / 1000
   max_t = d.get('maxTime', 0) / 1000
   print(f'TSDB range: {datetime.datetime.fromtimestamp(min_t)} to {datetime.datetime.fromtimestamp(max_t)}')
   print(f'  min_ts={min_t:.0f}  max_ts={max_t:.0f}')
   "
   # Then query 'up' at the last known timestamp to see which targets were scraped:
   curl -s 'http://localhost:<port>/api/v1/query?query=up&time=<max_ts>' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       ts = float(r['value'][0])
       print(f'{r[\"metric\"].get(\"job\",\"?\")} last_scrape={datetime.datetime.fromtimestamp(ts)}')
   "
   ```

   This step is not optional — always run it. The output tells you:
   - Whether GPU metrics (`hw_gpu_*`), network metrics (`hw_tcp_*`, `hw_rdma_*`), and training metrics (`tb_*`) are all present.
   - How many nodes were scraped (should match the job's `NNODES`).
   - The time range covered, confirming the TSDB has data for the period of interest.

   **Verify you are querying the correct TSDB.** After discovering metrics, cross-check a data point against the job log to confirm the database belongs to the expected job:

   ```bash
   # Query the last recorded step and loss (use a timestamp from the time range discovered above)
   curl -s 'http://localhost:<port>/api/v1/query?query=tb_step&time=<end_ts>' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       print(f'  step={r[\"value\"][1]}')
   "
   curl -s 'http://localhost:<port>/api/v1/query?query=tb_learning_loss&time=<end_ts>' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       print(f'  loss={r[\"value\"][1]}')
   "
   ```

   Compare these values against the corresponding `completed step:` line in the job log. The step number and loss should match exactly (loss may differ by rounding in the last decimal). If they don't match, you are querying the wrong Prometheus — re-check the address/port. This is especially important in multi-job comparisons (Case C) to avoid mixing up databases.

4. **Determine the time window.** The queries need a start/end time (Unix timestamps):
   - **From triage:** use the crash time or hang-start time identified in the triage report.
   - **From log timestamps:** parse the timestamp from the last `completed step:` line and the first training line.
   - **For health checks:** use the full job duration, or "last N minutes" for live jobs.
   - **Heartbeat timeout:** compute `crash_time - heartbeat_timeout_seconds` to get when heartbeats actually stopped.

   **Map training steps to wall-clock timestamps** using `tb_step`. This is essential for multi-job comparisons and for querying system metrics at a specific training step:

   ```bash
   # Find the wall-clock time range of the TSDB and map key steps to timestamps.
   # Use a wide range (e.g., 24h before the last known scrape) to cover the full job.
   curl -s 'http://localhost:<port>/api/v1/query_range?query=tb_step&start=<start>&end=<end>&step=60s' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   results = data.get('data', {}).get('result', [])
   if not results: print('No tb_step data'); sys.exit()
   vals = [(float(v[0]), float(v[1])) for v in results[0]['values'] if v[1] != 'NaN']
   print(f'Step range: {vals[0][1]:.0f} to {vals[-1][1]:.0f}')
   print(f'Time range: {datetime.datetime.fromtimestamp(vals[0][0])} to {datetime.datetime.fromtimestamp(vals[-1][0])}')
   # Find timestamps for specific steps — replace with your target steps
   for target in [200, 300, 500, 1000]:
       for ts, step in vals:
           if step >= target:
               print(f'  step {target}: ts={ts:.0f} ({datetime.datetime.fromtimestamp(ts)})')
               break
   "
   ```

   Use the resulting timestamps to query system metrics (`hw_*`, `ray_*`) at the wall-clock times corresponding to specific training steps.

5. **Run the appropriate playbook** (see Diagnostic Playbooks below). Each playbook specifies the PromQL queries, how to interpret results, and what to conclude.

6. **Report findings** in the structured format (see Output Format below).

7. **Cleanup.** Kill any read-only Prometheus instances you started in step 2:
   ```bash
   kill %1 2>/dev/null   # single instance
   kill %1 %2 2>/dev/null   # multi-job comparison
   ```
   Do **not** kill the live Prometheus of a running job.

## Querying Prometheus

All queries use the Prometheus HTTP API via `curl`. Parse responses with inline Python.

### Instant query (single point in time)

```bash
curl -s 'http://localhost:<port>/api/v1/query?query=<promql>&time=<unix_ts>'
```

**Always specify `&time=<unix_ts>` for read-only instances (Case B/C).** Without it, Prometheus defaults to "now" — which is past the end of the data for finished jobs, returning empty results. Use a timestamp from within the job's time range (discovered in step 3).

### Range query (time series)

```bash
curl -s 'http://localhost:<port>/api/v1/query_range?query=<promql>&start=<unix_ts>&end=<unix_ts>&step=<interval>'
```

Use `step=30s` for high resolution (short windows), `step=60s` for medium, `step=300s` for long durations.

### URL encoding for `rate()` and `increase()`

When using `rate()` or `increase()` in curl URLs, the square brackets must be URL-encoded:

```bash
# WRONG — brackets break the URL:
curl -s 'http://localhost:9190/api/v1/query?query=rate(hw_tcp_retransmits_total[5m])'

# CORRECT — URL-encode the brackets:
curl -s 'http://localhost:9190/api/v1/query?query=rate(hw_tcp_retransmits_total%5B5m%5D)'
```

`[` = `%5B`, `]` = `%5D`. This applies to all PromQL queries containing range selectors.

### Standard query + tabular output pattern

Use this pattern for all queries — it extracts the metric labels and values into a readable table:

```bash
curl -s 'http://localhost:<port>/api/v1/query?query=<promql>&time=<unix_ts>' | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data', {}).get('result', []):
    labels = r['metric']
    host = labels.get('host', labels.get('instance', '?'))
    gpu = labels.get('gpu', '')
    val = r['value'][1]
    print(f'  {host:<20} gpu={gpu:<4} {val}')
"
```

For range queries, use this to extract per-host min/max/avg:

```bash
curl -s 'http://localhost:<port>/api/v1/query_range?query=<promql>&start=<start>&end=<end>&step=30s' | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data', {}).get('result', []):
    labels = r['metric']
    host = labels.get('host', labels.get('instance', '?'))
    gpu = labels.get('gpu', '')
    vals = [float(v[1]) for v in r['values'] if v[1] != 'NaN']
    if vals:
        print(f'  {host:<20} gpu={gpu:<4} min={min(vals):.1f}  max={max(vals):.1f}  avg={sum(vals)/len(vals):.1f}')
"
```

### Filtering by host

Narrow queries to specific hosts (e.g., the accused nodes from triage):

```promql
hw_gpu_power_watts{host=~"node001|node002"}
ray_node_gpus_utilization{host="node003"}
```

### Rate queries for counters

Counters (names ending in `_total`) must use `rate()` or `increase()`:

```promql
rate(hw_tcp_retransmits_total[5m])
increase(hw_oom_kills_total[1h])
```

## Diagnostic Playbooks

Each playbook is triggered by a specific scenario (from triage output or user request). Run the listed queries, apply the interpretation rules, and report findings.

### Correlation is not causation

When an incident occurs, many metrics move at once. A single underlying event — a network blip, a thermal excursion, an I/O stall — ripples through the system and causes correlated changes across GPU power, clocks, utilization, throughput, step time, and more. Most of what you see in the TSDB are **symptoms**, not root causes. Do not report a correlated symptom as the diagnosis.

**The diagnostic principle: find the earliest anomaly.** The root cause is the metric that deviates *first*. Everything that follows is a consequence. When you spot any anomaly:

1. **Note the timestamp** of the anomaly.
2. **Query all other metric families** at the same timestamp and in the minutes preceding it — both **training metrics** (`tb_*`) and **system metrics** (`hw_*`, `ray_*`).
3. **Find which metric deviated first.** That is the root cause candidate. Everything that moved after it is a correlated effect.
4. **Verify the causal direction.** The root cause should logically explain the downstream symptoms.

**Root causes can be in either domain — training or system.** The TSDB contains both training-level metrics (loss, grad norms, LR, MoE load balance loss, throughput) and system-level metrics (GPU power/clocks, network, I/O) on the same timeline. Do not assume the root cause is always a system event. Examples:

- **System → training:** A network retransmit burst stalls a collective → GPU power drops → step time spikes → throughput drops. System event is the root cause.
- **Training → system:** An MoE model's routing changes dramatically → `tb_learning_moe_lb_loss` spikes → some experts are overloaded while others are idle → GPU utilization becomes uneven → TGS drops. The mathematical event (routing shift) is the root cause; the GPU utilization change is a passive symptom.
- **Training → system:** A grad norm explosion → NaN values → recompilation or fallback → throughput collapse. The training instability is the root cause.
- **Training → system:** Learning rate warmup ends → larger weight updates → checkpoint sizes grow → I/O pressure during saves → periodic step time spikes. The LR schedule change is the root cause.

Always query both domains. When system metrics (power, clocks, utilization) change, check whether a training metric (`tb_learning_*`, `tb_perf_*`) shifted first — the system may be passively responding to a change in the model's computational behavior.

Always report the full chain: root cause → intermediate effects → observed symptoms. This is the core value of having all metrics in a single time-aligned TSDB.

### Checkpointing interference

Checkpointing is one of the most disruptive periodic events in training. During a checkpoint save, the system behavior changes dramatically — and many metrics that look anomalous are actually normal checkpoint effects. Always check whether an anomaly coincides with a checkpoint step before diagnosing it as a problem.

**What happens during a checkpoint save:**
- GPU utilization drops (GPUs idle while waiting for D2H transfer and I/O to complete)
- GPU power drops (idle GPUs draw less power)
- Host memory spikes (model parameters copied from GPU to host RAM, especially on DP replica #0 nodes)
- I/O pressure rises (`hw_io_pressure_full_pct`, `hw_mem_dirty_bytes`) as parameters are written to storage
- Step time spikes (the checkpoint step takes much longer than a normal training step)
- `hw_procs_blocked` increases (processes waiting on I/O)
- Network traffic may drop (no collectives during the checkpoint window)

**How to identify checkpoint steps:** Parse `checkpoint_period` from the log (from `Config param checkpoint_period: N`). Checkpoint saves occur at steps that are multiples of this period. If `async_checkpointing=true` is in `PASSTHROUGH_ARGS`, the save happens in the background and may overlap with subsequent training steps, spreading the I/O impact over multiple steps.

**Rules:**
- When analyzing steady-state performance, **exclude steps around checkpoint boundaries** (the checkpoint step and 1-2 steps after for async checkpointing).
- When diagnosing an anomaly, **always check whether the timestamp falls on a checkpoint step** before investigating further. A step time spike at step 200 with `checkpoint_period=200` is expected, not an incident.
- When comparing metrics across jobs with different `checkpoint_period` values, normalize by excluding checkpoint steps from both.
- Memory spikes during checkpoint saves are not OOM precursors unless they push the host to its limit. DP replica #0 nodes use ~2x model size in host RAM during saves — this is expected.

**Single-replica checkpoint restore and RCCL resource leaks:**

When `enable_single_replica_ckpt_restoring=true` and the job restores from a checkpoint, the restore code path leaks RCCL-related resources (threads, handles, or sub-processes). This manifests as a **persistent increase in `hw_procs_running`** (~3–8 extra processes per host) that appears immediately after restore and may creep upward over the job's lifetime. The leaked processes create constant CPU contention — competing with data pipeline threads, RCCL coordination, and the Python runtime — causing a persistent throughput (TGS) drop of ~0.5–1%.

**How to detect:** When comparing a restored job against a fresh-start baseline with identical config:
- Check `hw_procs_running` — a uniform delta across all hosts (not a spike, not on specific nodes) after restore is the signature.
- The delta is present from the first training step after restore and does not recover.
- Training-level metrics (loss, grad norms, LR) will be identical between the two jobs — this is purely system-level contention.
- A fresh start with the same config but no actual restore (even if `enable_single_replica_ckpt_restoring=true`) does not trigger the leak — the restore code path must actually execute.

---

### Playbook 1: RCCL/NCCL Hang

**Trigger:** Triage reports `hang` or `cancelled` with underlying hang. User says "why did the job hang" or "confirm the RCCL hang."

**Goal:** Confirm the hang mechanism (RCCL busy-wait vs. dead node vs. network partition) and identify the trigger.

**Queries** (at the hang time window — from last successful training step to job end):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_gpus_utilization` | Per-host GPU utilization |
| 2 | `hw_gpu_power_watts` | Per-GPU power draw |
| 3 | `rate(hw_tcp_retransmits_total[5m])` | TCP retransmit rate per host |
| 4 | `rate(hw_rdma_tx_retx_pkts_total[5m])` | RDMA retransmit rate per device |
| 5 | `rate(hw_rdma_tx_ack_timeout_total[5m])` | RDMA ACK timeouts per device |
| 6 | `hw_rdma_port_state` | RDMA port state (1=ACTIVE) |
| 7 | `rate(hw_rdma_rx_cnp_pkts_total[5m])` | Congestion notification packets |

**Interpretation:**

| GPU util | Power | Network | Diagnosis |
|----------|-------|---------|-----------|
| ~100% all nodes | Idle-level (~300W MI300X, vs 600-700W+ active) | Clean | **Confirmed RCCL busy-wait hang.** All GPUs spinning in RCCL polling loop. No real computation. |
| ~100% all nodes | Idle-level | Retransmit spike before hang | **Network-triggered RCCL hang.** A transient network event caused a collective to stall, then all nodes entered busy-wait. |
| ~100% most nodes | Mixed (some idle, some active) | Clean | **Possible straggler.** One or more nodes may have fallen behind, causing others to wait. Check per-node power to identify the slow node. |
| 0% on some nodes | 0W on some nodes | N/A | **Node death.** Some nodes died — remaining nodes hung waiting for them. Check `hw_dmesg_gpu_errors_total` and `hw_gpu_ras_*` on the dead nodes. |
| ~100% all nodes | Active-level (600W+) | Clean | **Not a hang** — training was still running. Re-check the triage classification. |

**Key thresholds (MI300X):**
- Active training power: 600-750W per GPU
- RCCL busy-wait power: 250-350W per GPU
- Idle power: <100W per GPU

---

### Playbook 2: Heartbeat False-Positive

**Trigger:** Triage reports `heartbeat-timeout`. User wants TSDB confirmation that the accused tasks were healthy.

**Goal:** Prove or disprove that the accused tasks were alive and training when the heartbeat mechanism killed them.

**Setup:** From the log, extract:
- Crash time (the heartbeat error timestamp)
- Heartbeat timeout value (`jax_distributed_heartbeat_timeout_seconds`)
- Accused task IDs and their host mappings (from `NNODES`, `NODE_LIST`, task-to-host mapping in log)
- Compute heartbeat-stop time: `crash_time - heartbeat_timeout`

**Queries** (at the heartbeat-stop time — this is when the heartbeats actually failed):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_gpus_utilization{host=~"<accused_hosts>"}` | Were accused tasks training? |
| 2 | `ray_node_mem_used{host=~"<accused_hosts>"}` | Memory on accused hosts |
| 3 | `hw_io_pressure_full_pct{host=~"<accused_hosts>"}` | I/O pressure (checkpoint stall?) |
| 4 | `rate(hw_tcp_retransmits_total{host=~"<accused_hosts>"}[5m])` | Network health |
| 5 | `hw_tcp_listen_drops_total{host=~"<coordinator_host>"}` | gRPC coordinator overloaded? |
| 6 | `hw_mem_dirty_bytes{host=~"<accused_hosts>"}` | Dirty page pressure |

**Interpretation:**

| GPU util on accused | Network | I/O pressure | Diagnosis |
|---------------------|---------|--------------|-----------|
| High (active training) | Clean | Low | **Confirmed false positive.** Tasks were healthy. The heartbeat mechanism failed (known gRPC bug — see `docs/jax-heartbeat-false-positive-postmortem.md`). |
| High | Retransmit spike | Low | **Likely false positive** triggered by transient network issue blocking the heartbeat gRPC. Tasks were alive but heartbeat RPCs couldn't reach the coordinator. |
| High | Clean | High | **Likely false positive** triggered by checkpoint I/O pressure. Heavy writeback blocked the heartbeat thread. |
| Low/zero | Any | Any | **Possibly a true positive.** Tasks may have actually died. Check for OOM, GPU errors, or crashes on those hosts. |

**Report template for confirmed false positive:**
> The TSDB confirms this is a heartbeat false-positive kill. At the heartbeat-stop time (<time>), all accused hosts showed GPU utilization at <X>%, TCP retransmit rate near zero, and no I/O pressure. The tasks were alive and actively training when the heartbeat mechanism declared them dead. Root cause: shared gRPC channel bug documented in `docs/jax-heartbeat-false-positive-postmortem.md`.

---

### Playbook 3: Host OOM

**Trigger:** Triage reports `oom-host`. User wants to understand the memory trajectory.

**Goal:** Trace the memory growth pattern, identify what caused the OOM, and determine if it's reproducible.

**Queries** (over the full job duration, or last 30 minutes before the OOM):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_mem_used` | Per-host memory over time |
| 2 | `increase(hw_oom_kills_total[1h])` | OOM kill events |
| 3 | `hw_mem_dirty_bytes` | Dirty page accumulation |
| 4 | `hw_mem_writeback_bytes` | Writeback pressure |
| 5 | `hw_io_pressure_full_pct` | I/O stall percentage |
| 6 | `hw_procs_blocked` | Processes blocked on I/O |
| 7 | `hw_gpu_vram_used_bytes` | GPU VRAM (to correlate D2H transfers) |

**Interpretation:**
- **Gradual ramp** → memory leak (Python objects, data pipeline buffers, or XLA compilation cache growing unbounded).
- **Sudden spike** → checkpoint save (D2H copies all parameters to host RAM), data loading burst, or XLA recompilation allocating new buffers.
- **Periodic spikes that recover** → checkpoint saves. If the peak exceeds available RAM on one cycle, it's a checkpoint OOM.
- **Only DP replica #0 nodes OOM** → checkpoint save pattern. MaxText saves checkpoints from DP replica 0 only, which requires ~2x model size in host RAM (one copy in GPU VRAM, one on host for the save).
- **All nodes OOM simultaneously** → likely a data pipeline issue or XLA recompilation event.

---

### Playbook 4: Node Failure / Hardware Issue

**Trigger:** Triage reports `node-fail`, `signal-kill`, or `unknown-death`. User wants to identify hardware problems.

**Goal:** Identify hardware errors (ECC, PCIe, XGMI) or driver issues that caused the failure.

**Queries** (over full job duration — hardware errors can accumulate before causing a crash):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `increase(hw_gpu_ras_umc_ue_total[1h])` | HBM uncorrectable ECC errors (fatal) |
| 2 | `increase(hw_gpu_ras_umc_ce_total[1h])` | HBM correctable ECC errors (accumulating = concern) |
| 3 | `increase(hw_gpu_ras_xgmi_ue_total[1h])` | XGMI/WAFL link uncorrectable errors |
| 4 | `increase(hw_gpu_ras_xgmi_ce_total[1h])` | XGMI/WAFL link correctable errors |
| 5 | `increase(hw_gpu_ras_gfx_ue_total[1h])` | Compute engine errors |
| 6 | `increase(hw_gpu_pcie_fatal_total[1h])` | PCIe fatal errors |
| 7 | `increase(hw_gpu_pcie_nonfatal_total[1h])` | PCIe non-fatal errors |
| 8 | `increase(hw_dmesg_gpu_errors_total[1h])` | GPU/driver errors in kernel log |
| 9 | `hw_rdma_port_state` | RDMA port went down? |
| 10 | `hw_gpu_temperature_celsius` | Thermal excursion before crash? |

**Interpretation:**
- **Any uncorrectable error (UE) > 0** → hardware fault. That GPU/link is bad. The node should be drained for maintenance.
- **High correctable errors (CE)** → degrading hardware. Not immediately fatal but indicates the component is failing.
- **PCIe fatal** → PCIe link reset or card disconnect. Likely unrecoverable without node restart.
- **RDMA port state → 0** → network link went down. Check physical connectivity and switch.
- **dmesg GPU errors increasing** → driver-level GPU fault detected by the kernel.
- **Temperature spike before crash** → thermal throttling or shutdown. Check cooling.
- **Compare across nodes** — healthy nodes should have zero RAS/PCIe errors. Any node with non-zero values is the problem.

---

### Playbook 5: GPU Health Check

**Trigger:** User says "check GPU health", "are the GPUs OK", "check thermals", or proactive monitoring of a running job.

**Goal:** Assess GPU health across the cluster — temperatures, power, clocks, VRAM, error counters.

**Queries** (range over full job or recent window):

| # | PromQL | What it shows | Alert threshold |
|---|--------|---------------|-----------------|
| 1 | `hw_gpu_temperature_celsius` | Junction temperature | >90C = concern, >100C = throttling |
| 2 | `hw_gpu_power_watts` | Power draw | Variance >50W across GPUs = investigate |
| 3 | `hw_gpu_clock_mhz{type="sclk"}` | Core clock | Sudden drop = investigate |
| 4 | `hw_gpu_vram_used_bytes / hw_gpu_vram_total_bytes` | VRAM utilization | >95% = risk of GPU OOM |
| 5 | `hw_gpu_ras_umc_ce_total` | HBM correctable errors | Any >0 = accumulating hardware issue |
| 6 | `hw_gpu_ras_umc_ue_total` | HBM uncorrectable errors | Any >0 = bad GPU, drain node |
| 7 | `hw_gpu_ras_xgmi_ce_total` | XGMI correctable errors | Any >0 = inter-GPU link degrading |
| 8 | `hw_gpu_pcie_correctable_total` | PCIe correctable errors | Steady increase = link issue |
| 9 | `ray_node_gpus_utilization` | GPU utilization | <80% during training = underutilization |

**Report format for health check:**

```
GPU Health Summary (<N> nodes, <G> GPUs)

Temperature:  min <X>C  max <X>C  avg <X>C  (threshold: 90C)
Power:        min <X>W  max <X>W  avg <X>W  spread: <X>W
Core clock:   min <X>MHz  max <X>MHz  (nominal: <X>MHz)
VRAM:         <X>% used  (<X> GB / <X> GB)
Utilization:  min <X>%  max <X>%  avg <X>%

RAS Errors:   <N> correctable, <N> uncorrectable
PCIe Errors:  <N> correctable, <N> non-fatal, <N> fatal

Anomalous GPUs: <list of host:gpu with any non-zero errors or outlier metrics>
```

---

### Playbook 6: Network Health Check

**Trigger:** User says "check network", "is the network OK", "RDMA health", or investigating intermittent NCCL timeouts.

**Goal:** Assess network health — TCP retransmits, RDMA errors, congestion, port state.

**Queries** (range over full job or recent window):

| # | PromQL | What it shows | Alert threshold |
|---|--------|---------------|-----------------|
| 1 | `rate(hw_tcp_retransmits_total[5m])` | TCP retransmit rate | >10/s sustained = concern |
| 2 | `rate(hw_rdma_tx_retx_pkts_total[5m])` | RDMA retransmit rate | Any sustained > 0 = concern |
| 3 | `rate(hw_rdma_tx_ack_timeout_total[5m])` | RDMA ACK timeouts | Any > 0 = link or switch issue |
| 4 | `hw_rdma_port_state` | Port state (1=ACTIVE) | Any 0 = port down |
| 5 | `rate(hw_rdma_rx_cnp_pkts_total[5m])` | Congestion notifications | Sustained = network congestion |
| 6 | `rate(hw_rdma_req_tx_retry_excd_err_total[5m])` | Retry exhaustion | Any > 0 = packets lost |
| 7 | `rate(hw_rdma_req_rx_cqe_err_total[5m])` | CQE errors | Any > 0 = RDMA errors |
| 8 | `rate(hw_net_rx_errors_total[5m])` | NIC RX errors | Any > 0 = NIC issue |
| 9 | `rate(hw_net_tx_errors_total[5m])` | NIC TX errors | Any > 0 = NIC issue |
| 10 | `rate(hw_tcp_abort_on_timeout_total[5m])` | TCP connections aborted | Any > 0 = severe |

**Interpretation:**
- **TCP retransmits only, no RDMA errors** → IP/TCP path issue (NFS, coordinator gRPC), not the NCCL/RCCL data path.
- **RDMA retransmits + ACK timeouts on specific devices** → bad cable, bad port, or switch issue on that link. Identify the device and port labels.
- **Congestion notifications (CNP) across many nodes** → switch-level congestion. May need ECN tuning or traffic engineering.
- **Retry exhaustion** → packets permanently lost. Indicates a hard link failure, not transient congestion.
- **Port state 0** → RDMA link is down. Physical layer issue. Check cable and switch port.
- **Correlate with training events** — if retransmits spike at the same time as a step time increase or hang, the network event caused the training issue.

---

### Playbook 7: Training Stability

**Trigger:** User says "check training", "is training stable", "loss looks weird", "throughput dropping", or proactive monitoring.

**Goal:** Assess training health — loss convergence, gradient norms, throughput, step time consistency.

**Important:** Filter out synthetic anti-staleness fills. Only use data points where `tb_metrics_plugin_staleness_fill == 0`.

**Queries** (range over full job or recent window):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `tb_learning_loss and tb_metrics_plugin_staleness_fill == 0` | Training loss (real data only) |
| 2 | `tb_learning_grad_norm and tb_metrics_plugin_staleness_fill == 0` | Gradient norm |
| 3 | `tb_learning_raw_grad_norm and tb_metrics_plugin_staleness_fill == 0` | Pre-clipping gradient norm |
| 4 | `tb_perf_step_time_seconds and tb_metrics_plugin_staleness_fill == 0` | Step time |
| 5 | `tb_perf_per_device_tokens_per_sec and tb_metrics_plugin_staleness_fill == 0` | Throughput |
| 6 | `tb_learning_current_learning_rate and tb_metrics_plugin_staleness_fill == 0` | Learning rate |
| 7 | `tb_step` | Current step (for progress tracking) |
| 8 | `tb_learning_moe_lb_loss and tb_metrics_plugin_staleness_fill == 0` | MoE load balance loss (if MoE model) |

**Interpretation — training metrics:**
- **Loss divergence** (sudden increase or NaN) → learning rate too high, data issue, or numerical instability. Check if grad norm spiked at the same time.
- **Gradient norm spikes** → unstable training. If `raw_grad_norm >> grad_norm`, gradient clipping is active and may be too aggressive.
- **MoE load balance loss spike** → routing changed dramatically, causing expert load imbalance. This is a training-level root cause that will show up as uneven GPU utilization and TGS drop.
- **Learning rate anomaly** → verify the LR schedule matches expectations. A flat LR when warmup should be active (or vice versa) indicates a config issue.

**Interpretation — step time and throughput:**
- **Step time periodic spikes** → likely checkpoint saves. Correlate with `checkpoint_period` from the job config. Spikes of 10-30s every N steps are normal.
- **Throughput regression** → >10% drop from steady-state average warrants investigation. Run the contention checklist below.
- **Step time gradual increase** → resource contention worsening over time. Run the contention checklist below.

**Contention checklist.** When throughput drops or step time increases, systematically check all contention sources at the same timestamp. Any of these can silently degrade performance:

| Contention source | Metrics to check | What to look for |
|--------------------|-----------------|------------------|
| **CPU** | `hw_procs_running`, `hw_procs_blocked`, `hw_context_switches_total`, `ray_node_cpu_utilization`, `hw_gpu_user_processes` | High runnable count (oversubscribed), high blocked count (I/O wait), excessive context switching. **For multi-job comparison:** a constant but higher process count in one job creates a constant baseline CPU tax — extra processes compete with data pipeline threads, NCCL coordination, and Python runtime. This won't show as a spike but as a persistent throughput gap. Compare `hw_procs_running` across jobs, not just within a single job. **Known cause:** `enable_single_replica_ckpt_restoring=true` leaks RCCL resources on restore, adding ~3–8 procs/host (see "Checkpointing interference" section). |
| **Network (TCP)** | `rate(hw_tcp_retransmits_total[5m])`, `rate(hw_tcp_estab_resets_total[5m])`, `rate(hw_tcp_abort_on_timeout_total[5m])` | Retransmit bursts slow NFS and gRPC; resets and aborts indicate connection failures |
| **Network (RDMA)** | `rate(hw_rdma_tx_retx_pkts_total[5m])`, `rate(hw_rdma_tx_ack_timeout_total[5m])`, `rate(hw_rdma_rx_cnp_pkts_total[5m])` | RDMA retransmits slow NCCL/RCCL collectives; CNP indicates switch congestion |
| **Storage I/O** | `hw_io_pressure_full_pct`, `hw_io_pressure_some_pct`, `hw_mem_dirty_bytes`, `hw_mem_writeback_bytes` | I/O pressure stalls data loading and checkpointing; dirty page buildup indicates NFS/storage backlog |
| **Memory** | `ray_node_mem_used`, `hw_oom_kills_total`, `hw_procs_blocked` | Memory pressure causes swapping (blocked procs); approaching limit risks OOM |
| **GPU thermal** | `hw_gpu_temperature_celsius`, `hw_gpu_clock_mhz{type="sclk"}`, `hw_gpu_power_watts` | Temperature rise → clock throttle → power drop → throughput drop (all correlated symptoms of thermal issue) |
| **GPU hardware** | `hw_gpu_ras_*`, `hw_gpu_pcie_*_total`, `hw_dmesg_gpu_errors_total` | Accumulating errors degrade performance before causing a crash |
| **Training-level** | `tb_learning_moe_lb_loss`, `tb_learning_grad_norm`, `tb_learning_loss` | Model behavior changes (routing shifts, grad spikes, loss instability) that alter computational load |

Check all rows — contention sources often compound (e.g., I/O pressure from checkpointing + network congestion from RDMA traffic = amplified step time spike).

**Cross-domain correlation queries** (the power of a unified TSDB):

```promql
# Did a temperature spike cause a throughput drop?
# Query both in the same time window and compare timestamps
tb_perf_per_device_tokens_per_sec{host="node001"}
hw_gpu_temperature_celsius{host="node001"}

# Did network issues cause step time spikes?
tb_perf_step_time_seconds{host="node001"}
rate(hw_tcp_retransmits_total{host="node001"}[5m])

# Did I/O pressure from checkpointing cause a loss spike?
tb_learning_loss{host="node001"}
hw_io_pressure_full_pct{host="node001"}
```

## Multi-Job Comparison

When comparing metrics across two or more jobs (e.g., "why is job B slower than job A?"), follow these rules:

### 0. Triage all jobs first

Run the **job-log-triage** skill on **each** job being compared (per workflow step 0). For multi-job comparison, triage is especially critical because you need to know:
- Which jobs are running vs finished (determines Case A vs B for each)
- Which jobs are fresh starts vs checkpoint restores (restored jobs may have resource leaks — see "Checkpointing interference")
- The step range and checkpoint period for each job (needed to identify overlapping steps and exclude checkpoint boundaries)
- Any failures or hangs that affect which steps are comparable (e.g., don't compare post-hang idle metrics against active training)

### 1. Isolate config differences first

Before looking at metrics, compare the jobs' configurations. Parse `PASSTHROUGH_ARGS` from the header of each log file (first ~30 lines) and diff them. Common config parameters that affect performance:
- `per_device_batch_size`, `max_target_length`, `steps`
- `remat_policy`, `quantization`, `attention`
- `ici_*_parallelism`, `dcn_*_parallelism` (parallelism strategy)
- `enable_checkpointing`, `checkpoint_period`, `async_checkpointing`
- `load_balance_loss_weight` (MoE)
- `_env_*` flags (XLA flags, NCCL tuning)
- Number of nodes (`NNODES`), GPUs per node

If configs differ, the performance difference may be **expected** — the metric comparison should account for the config change, not treat it as an anomaly.

### 2. Align by training step, not wall clock

Jobs run at different times and speeds. Compare metrics at the **same training step range**, not the same wall-clock time. Use `tb_step` to map between step number and timestamp in each job's TSDB, then query system metrics at the corresponding wall-clock times.

### 3. Compare only overlapping steps

If job A ran steps 0-500 and job B ran steps 200-700, only compare metrics during the overlapping range (steps 200-500). Metrics outside the overlap are not comparable — warmup behavior (early steps) and long-run behavior (late steps) differ inherently.

**Checkpoint restore awareness:** If one job is a fresh start and the other restored from a checkpoint, the first few steps after restore may have different behavior (XLA recompilation, data pipeline warmup, potential resource leaks). Identify which job restored and what step it resumed from (from the triage report or log). Compare steady-state behavior after both jobs have fully warmed up — typically 5-10 steps after the later job's first step.

### 4. Compare runtime environment, not just config

Even with identical `PASSTHROUGH_ARGS`, the runtime environment can differ: different observability stacks (`RAY=1` vs lighter monitoring), different background processes, different node allocations. Check `hw_procs_running` across jobs — a higher process count means more CPU contention. A constant overhead (not a spike) creates a persistent throughput gap that is easy to misattribute to other causes.

### 5. Control for transient events

A throughput difference caused by a one-time network blip in job B is not a systematic issue. Use steady-state averages (skip warmup steps) and look at variance — a persistent gap indicates a real difference, while a spike indicates a transient event.

### 6. Report structure for comparison

```
## Multi-Job Comparison: <job_A> vs <job_B>

### Config differences
| Parameter | Job A | Job B |
|-----------|-------|-------|
| ... | ... | ... |

### Overlapping step range: <start> to <end>

### Metric comparison (steady-state averages over overlapping steps)
| Metric | Job A | Job B | Delta | Likely cause |
|--------|-------|-------|-------|--------------|
| ... | ... | ... | ... | ... |

### Root cause
<Traced from the earliest diverging metric, accounting for config differences>
```

## Common Pitfalls

Hard-won lessons from real diagnosis sessions. Avoid these mistakes:

1. **Empty results from read-only Prometheus.** If an instant query returns empty `result: []`, you almost certainly forgot the `&time=` parameter. Read-only Prometheus defaults to "now" which is past the end of the data. Use `api/v1/status/tsdb` to find the TSDB time range (step 3), then always include `&time=<max_ts>` for instant queries.

2. **Querying the wrong Prometheus.** `localhost:9090` on the head node may be a cluster-level Prometheus, not the job's Prometheus. The job's Prometheus runs on `<head_host>:9190` (or the auto-incremented port). Always verify the TSDB by cross-checking `tb_step`/`tb_learning_loss` against the job log. If the metrics don't match (wrong metric families, wrong step range), you're querying the wrong database.

3. **Mixing up databases in multi-job comparison.** When running multiple read-only Prometheus instances on different ports, it's easy to send a query to the wrong port. Label each port clearly (e.g., "9190 = job 7879, 9191 = job 7882") and verify each with the `tb_step` cross-check before starting diagnostic work.

4. **Diagnosing symptoms as root causes.** GPU power drops, clock drops, utilization drops, and throughput drops are usually symptoms, not causes. Always trace back to the *earliest* anomaly across all metric families — it may be a training-level event (MoE routing shift, grad spike) or a system event (network retransmit, I/O stall), but not the GPU metric itself.

5. **Ignoring checkpoint steps.** A 10x step time spike at step 200 with `checkpoint_period=200` is expected, not an incident. Always identify checkpoint steps before diagnosing step time anomalies.

6. **Comparing jobs without understanding their start conditions.** A job that restored from a checkpoint may have RCCL resource leaks, XLA recompilation overhead, or different data pipeline warmup compared to a fresh start. Always check whether a job is a fresh start or restore before comparing metrics.

7. **Assuming host memory contention affects GPU training.** For GPU training, the GPU compute path is largely independent of host memory usage. A spike in host memory (e.g., from checkpoint saves) does not directly slow GPU computation unless it triggers OOM or swapping. Focus on CPU contention (`hw_procs_running`), network contention, and GPU-level metrics.

## Metric Reference

Complete catalog of metrics available in the TSDB. All metrics have a `host` label.

### GPU metrics (`hw_gpu_*`) — from `gpu_metrics_plugin.sh`

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `hw_gpu_temperature_celsius` | `gpu`, `host` | gauge | Junction temperature |
| `hw_gpu_power_watts` | `gpu`, `host` | gauge | Power draw |
| `hw_gpu_clock_mhz` | `gpu`, `host`, `type` | gauge | Clock speed (`sclk`=core, `mclk`=memory) |
| `hw_gpu_vram_used_bytes` | `gpu`, `host` | gauge | VRAM used |
| `hw_gpu_vram_total_bytes` | `gpu`, `host` | gauge | VRAM total |
| `hw_gpu_ras_umc_{ue,ce}_total` | `gpu`, `host` | counter | HBM ECC errors |
| `hw_gpu_ras_xgmi_{ue,ce}_total` | `gpu`, `host` | counter | XGMI/WAFL link errors |
| `hw_gpu_ras_gfx_{ue,ce}_total` | `gpu`, `host` | counter | Compute engine errors |
| `hw_gpu_ras_mmhub_{ue,ce}_total` | `gpu`, `host` | counter | Memory hub errors |
| `hw_gpu_ras_sdma_{ue,ce}_total` | `gpu`, `host` | counter | SDMA engine errors |
| `hw_gpu_pcie_correctable_total` | `gpu`, `host` | counter | PCIe correctable errors |
| `hw_gpu_pcie_nonfatal_total` | `gpu`, `host` | counter | PCIe non-fatal errors |
| `hw_gpu_pcie_fatal_total` | `gpu`, `host` | counter | PCIe fatal errors |

### Host/network metrics (`hw_*`) — from `host_metrics_plugin.sh`

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `hw_net_{rx,tx}_bytes_total` | `device`, `host` | counter | Network bytes |
| `hw_net_{rx,tx}_errors_total` | `device`, `host` | counter | NIC errors |
| `hw_net_{rx,tx}_drop_total` | `device`, `host` | counter | NIC drops |
| `hw_tcp_retransmits_total` | `host` | counter | TCP retransmits |
| `hw_tcp_listen_overflows_total` | `host` | counter | Listen queue overflows |
| `hw_tcp_listen_drops_total` | `host` | counter | Listen queue drops |
| `hw_tcp_estab_resets_total` | `host` | counter | Established conn resets |
| `hw_tcp_abort_on_timeout_total` | `host` | counter | Conns aborted after timeout |
| `hw_rdma_{rx,tx}_bytes_total` | `device`, `port`, `host` | counter | RDMA bytes |
| `hw_rdma_{rx,tx}_pkts_total` | `device`, `port`, `host` | counter | RDMA packets |
| `hw_rdma_tx_retx_{bytes,pkts}_total` | `device`, `port`, `host` | counter | RDMA retransmits |
| `hw_rdma_tx_ack_timeout_total` | `device`, `port`, `host` | counter | RDMA ACK timeouts |
| `hw_rdma_{rx,tx}_cnp_pkts_total` | `device`, `port`, `host` | counter | Congestion notifications |
| `hw_rdma_req_rx_cqe_err_total` | `device`, `port`, `host` | counter | CQE errors |
| `hw_rdma_req_tx_retry_excd_err_total` | `device`, `port`, `host` | counter | Retry exhaustion |
| `hw_rdma_port_state` | `device`, `port`, `host` | gauge | 1=ACTIVE, 0=not |
| `hw_procs_running` | `host` | gauge | Runnable processes |
| `hw_procs_blocked` | `host` | gauge | Blocked on I/O |
| `hw_context_switches_total` | `host` | counter | Context switches |
| `hw_oom_kills_total` | `host` | counter | OOM killer invocations |
| `hw_mem_dirty_bytes` | `host` | gauge | Dirty pages |
| `hw_mem_writeback_bytes` | `host` | gauge | Writeback pages |
| `hw_io_pressure_some_pct` | `host` | gauge | I/O pressure (some, 10s) |
| `hw_io_pressure_full_pct` | `host` | gauge | I/O pressure (full, 10s) |
| `hw_io_pressure_{some,full}_avg300_pct` | `host` | gauge | I/O pressure (300s avg) |
| `hw_io_pressure_full_total_us` | `host` | counter | Cumulative I/O stall time |
| `hw_dmesg_gpu_errors_total` | `host` | counter | GPU/driver errors in dmesg |
| `hw_gpu_user_processes` | `host` | gauge | Processes with /dev/kfd open |
| `hw_scrape_duration_seconds` | `host` | gauge | Plugin scrape time (emitted by `metrics_exporter.sh`) |

### Ray metrics (`ray_node_*`) — from Ray built-in exporter

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `ray_node_gpus_utilization` | `host` (or `instance`) | gauge | Per-GPU utilization (%) |
| `ray_node_mem_used` | `host` | gauge | Host memory used (bytes) |
| `ray_node_mem_total` | `host` | gauge | Host memory total (bytes) |
| `ray_node_cpu_utilization` | `host` | gauge | CPU utilization (%) |
| `ray_node_disk_*` | `host` | various | Disk I/O metrics |

### Training metrics (`tb_*`) — from `tb_metrics_plugin.sh`

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `tb_step` | `host` | gauge | Current training step |
| `tb_learning_loss` | `host` | gauge | Training loss |
| `tb_learning_grad_norm` | `host` | gauge | Gradient norm (post-clipping) |
| `tb_learning_raw_grad_norm` | `host` | gauge | Raw gradient norm (pre-clipping) |
| `tb_learning_param_norm` | `host` | gauge | Parameter norm |
| `tb_learning_current_learning_rate` | `host` | gauge | Learning rate |
| `tb_learning_total_weights` | `host` | gauge | Total model parameters |
| `tb_perf_step_time_seconds` | `host` | gauge | Wall-clock time per step |
| `tb_perf_per_device_tflops` | `host` | gauge | Per-device TFLOP/s |
| `tb_perf_per_device_tokens_per_sec` | `host` | gauge | Per-device tokens/sec |
| `tb_learning_moe_lb_loss` | `host` | gauge | MoE load balance loss (only present for MoE models) |
| `tb_metrics_plugin_staleness_fill` | `host` | gauge | 0=real data, 1=synthetic fill |

## Output Format

```
## TSDB Diagnosis: <job_dir>

**Analysis type:** <hang | heartbeat | oom | hardware | gpu-health | network-health | training-stability>
**Time window:** <start_time> to <end_time> (<duration>)
**Nodes:** <N> (<host1, host2, ...>)

### Findings
<Plain-English summary: what the metrics show, what the root cause is>

### Metric evidence

<For each key metric queried, show a table with per-host/per-GPU values.
Use the actual query results — do not fabricate data.>

| Host | GPU | Value | Interpretation |
|------|-----|-------|----------------|
| ... | ... | ... | ... |

### Anomalies detected
<Any nodes/GPUs/metrics that deviate from cluster norm. "None" if all healthy.>

### Correlation with triage
<If triggered from a triage report, confirm or refute the failure hypothesis.
Example: "Triage classified this as an RCCL hang. TSDB confirms: all 192 GPUs
showed 100% utilization at 310W (idle-level) during the hang window, with zero
network errors. This is a confirmed RCCL busy-wait deadlock.">

### Recommended next steps
<Numbered list of specific actions based on the diagnosis>
```

## Integration with Triage

When the triage skill identifies a failure that benefits from TSDB analysis, it includes "Query Prometheus TSDB" in its recommended next steps. The handoff works as follows:

1. **Triage provides:** failure class, crash time, accused hosts/tasks, job directory.
2. **This skill takes over:** starts Prometheus against the persisted TSDB, runs the appropriate playbook, and reports metric-level evidence.
3. **Key handoff scenarios:**

| Triage class | Diagnosis playbook | What TSDB adds |
|--------------|-------------------|----------------|
| `hang` | Playbook 1 (RCCL Hang) | Confirms busy-wait signature, identifies network trigger |
| `heartbeat-timeout` | Playbook 2 (Heartbeat) | Proves false positive — tasks were healthy |
| `oom-host` | Playbook 3 (OOM) | Memory trajectory, checkpoint correlation |
| `node-fail` / `signal-kill` | Playbook 4 (Hardware) | RAS errors, PCIe faults, thermal excursion |
| `nccl-timeout` | Playbook 6 (Network) | Network health at failure time |
| `unknown-death` | Playbook 3 + 4 | OOM evidence or hardware faults |

For proactive use (no triage handoff), the user triggers directly with health-check requests.
