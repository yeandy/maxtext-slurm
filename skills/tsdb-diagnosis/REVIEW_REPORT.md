# TSDB Diagnosis SKILL.md — Review Report

**Review date:** 2025-02-20 (updated after refinement pass)  
**Reviewed by:** Automated consistency check + manual refinement from 7879-vs-7882 diagnosis session

---

## Summary

**Overall:** All checks pass. Refinements from real diagnosis sessions have been incorporated.

---

## 1. Metric name accuracy

| Status | Finding |
|--------|---------|
| ✅ Pass | All metric names referenced in playbooks and queries exist in the plugin source files |
| ✅ Pass | `hw_scrape_duration_seconds` correctly attributed to `metrics_exporter.sh` |
| ✅ Pass | `tb_learning_moe_lb_loss` added to both Playbook 7 queries and Metric Reference table |

**Verification:**
- **gpu_metrics_plugin.sh:** All `hw_gpu_*` metrics in SKILL.md match the plugin (temperature, power, clocks, VRAM, RAS blocks, PCIe AER).
- **host_metrics_plugin.sh:** All `hw_net_*`, `hw_tcp_*`, `hw_rdma_*`, `hw_procs_*`, `hw_oom_*`, `hw_mem_*`, `hw_io_pressure_*`, `hw_dmesg_gpu_errors_total`, `hw_gpu_user_processes` match.
- **tb_metrics_plugin.sh:** All `tb_*` metrics (learning_loss, grad_norm, step_time, moe_lb_loss, etc.) are either explicitly listed in the plugin header or produced dynamically from TensorBoard tags.

---

## 2. Prometheus.sh integration

| Status | Finding |
|--------|---------|
| ✅ Pass | View command syntax matches `prometheus.sh` |
| ✅ Pass | Option order `view <data-dir> -p <port>` is supported (prometheus.sh accepts options before or after positional args) |

**SKILL (step 2, Case B):** `utils/prometheus.sh view <job_dir>/prometheus -p <port> &`  
**prometheus.sh (line 10):** `utils/prometheus.sh view <data-dir> [-p PORT]`

---

## 3. Triage handoff

| Status | Finding |
|--------|---------|
| ✅ Pass | Every triage class that mentions Prometheus/TSDB has a corresponding diagnosis playbook |

**Mapping:**

| Triage class        | Prometheus/TSDB in triage?        | Diagnosis playbook          |
|---------------------|-----------------------------------|-----------------------------|
| `hang`              | Yes — "Query Prometheus TSDB at hang time" | Playbook 1 (RCCL Hang)     |
| `heartbeat-timeout` | Yes — "If TSDB is available..."  | Playbook 2 (Heartbeat)      |
| `unknown-death`     | Yes — "run with RAY=1 for TSDB diagnostics" | Playbook 3 + 4 (OOM + Hardware) |
| `oom-host`          | No explicit TSDB                 | Playbook 3 (OOM)            |
| `nccl-timeout`      | No explicit TSDB                 | Playbook 6 (Network)        |
| `node-fail`         | No explicit TSDB                 | Playbook 4 (Hardware)       |
| `signal-kill`       | No explicit TSDB                 | Playbook 4 (Hardware)       |

---

## 4. PromQL syntax

| Status | Finding |
|--------|---------|
| ✅ Pass | All PromQL queries use valid syntax |
| ✅ Pass | Counters use `rate()` or `increase()` where appropriate |
| ✅ Pass | Label selectors use valid `=` and `=~` syntax |

**Sample checks:**
- `rate(hw_tcp_retransmits_total[5m])` — counter with rate ✓
- `increase(hw_oom_kills_total[1h])` — counter with increase ✓
- `hw_gpu_vram_used_bytes / hw_gpu_vram_total_bytes` — valid division (common labels) ✓
- `tb_learning_loss and tb_metrics_plugin_staleness_fill == 0` — valid vector match on `host` ✓
- `tb_learning_moe_lb_loss and tb_metrics_plugin_staleness_fill == 0` — valid vector match ✓

---

## 5. Port handling

| Status | Finding |
|--------|---------|
| ✅ Pass | Default port 9190 matches `prometheus.sh` (avoids conflict with cluster Prometheus on 9090) |
| ✅ Pass | Auto-increment fallback (9191, 9192, ...) if port occupied |
| ✅ Pass | Skill documents port may differ from 9190 and to check the log |

**prometheus.sh (line 14):** `PROMETHEUS_PORT=${PROMETHEUS_PORT:-9190}`  
**prometheus.sh `start_prometheus()`:** Verifies bind succeeded; auto-increments port on failure up to 5 retries.  
**SKILL Case A:** Documents checking log for `[Prometheus] WARNING: port 9190 was occupied; using <port> instead`.  
**SKILL Case A:** Instructs querying `http://<head_host>:<port>`, not `localhost:<port>` (which may be a different Prometheus).

---

## 6. Read-only Prometheus querying (new)

| Status | Finding |
|--------|---------|
| ✅ Pass | Skill warns that `&time=` is required for instant queries on read-only instances |
| ✅ Pass | Step 3 uses `api/v1/status/tsdb` to discover TSDB time range before data queries |
| ✅ Pass | `up` query uses `&time=<max_ts>` from the TSDB status endpoint |
| ✅ Pass | URL encoding documented: `[` = `%5B`, `]` = `%5D` for `rate()`/`increase()` in curl |

**Rationale:** During diagnosis sessions, instant queries on read-only Prometheus returned empty results because the default evaluation time is "now", which is past the end of the persisted data. The `api/v1/status/tsdb` endpoint returns `minTime`/`maxTime` without needing a timestamp, enabling all subsequent queries to be properly time-scoped.

---

## 7. TSDB verification (new)

| Status | Finding |
|--------|---------|
| ✅ Pass | Step 3 includes concrete queries to verify `tb_step` and `tb_learning_loss` against job log |
| ✅ Pass | Step 1 requires verifying `prometheus/` contains data (ULID dirs or `wal/`), not just that the directory exists |
| ✅ Pass | Common Pitfalls #2 and #3 warn about querying wrong Prometheus and mixing up databases |

**Rationale:** During 7879-vs-7882 diagnosis, `localhost:9090` served a cluster-level Prometheus with different metric families (`gpu_*`, `rdma_stat_*`) instead of the job's Prometheus (`hw_*`, `tb_*`). Cross-checking `tb_step` against the job log catches this immediately.

---

## 8. Step-to-timestamp mapping (new)

| Status | Finding |
|--------|---------|
| ✅ Pass | Step 4 includes `tb_step` range query pattern for mapping training steps to wall-clock timestamps |

**Rationale:** Multi-job comparison requires aligning by training step, not wall clock. The mapping query lets you convert a step number to a Unix timestamp for querying system metrics at the corresponding time.

---

## 9. Multi-job comparison workflow (new)

| Status | Finding |
|--------|---------|
| ✅ Pass | Step 0 requires triaging all jobs before TSDB queries |
| ✅ Pass | Step 3 includes checkpoint restore awareness (fresh start vs restore differences) |
| ✅ Pass | Common Pitfall #6 warns about comparing jobs without understanding start conditions |

**Rationale:** During 7879-vs-7882 diagnosis, not knowing that 7882 restored from a checkpoint caused initial misinterpretation of the `hw_procs_running` delta as "stale processes" rather than the actual root cause (RCCL resource leak from `enable_single_replica_ckpt_restoring=true`).

---

## 10. Common Pitfalls section (new)

| Status | Finding |
|--------|---------|
| ✅ Pass | Seven pitfalls documented, all derived from real diagnosis sessions |

The pitfalls cover: empty query results, wrong Prometheus, database mix-ups, symptom-vs-cause confusion, checkpoint step misdiagnosis, fresh-vs-restore comparison, and host memory misconception.

---

## 11. TSDB lock file safety (existing, verified)

| Status | Finding |
|--------|---------|
| ✅ Pass | Critical rule documented: never delete a TSDB lock file, never run `prometheus.sh view` against a running job's TSDB |
| ✅ Pass | Job state determination (squeue + log check) precedes Prometheus access method selection |
| ✅ Pass | Cleanup step (7) explicitly states "do not kill the live Prometheus of a running job" |

---

## Conclusion

**No open issues.** All checks pass. The skill has been refined with practical lessons from the 7879-vs-7882 diagnosis session, covering read-only Prometheus querying pitfalls, TSDB verification, step-to-timestamp mapping, triage-first multi-job workflow, and the `tb_learning_moe_lb_loss` metric.
