---
name: performance-analysis
description: Analyze MaxText training job performance using tgs_tagger, TraceLens, and IRLens. Use when the user asks to analyze a training run, profile traces, HLO IR, TGS metrics, GPU utilization, or mentions tag_tgs, TraceLens, IRLens, xplane, or performance analysis.
---

# MaxText Performance Analysis

Post-training (or mid-training) analysis pipeline. Detects available artifacts in a job's output directory and runs the appropriate tools.

## Workflow (follow this order)

1. **Run `analyze_job.py` first** — before reading `analysis.json`. The dispatcher automatically detects staleness: it skips re-analysis when the job is finished and `analysis.json` is newer than the log file; otherwise it re-runs. Pass `-f` to force re-analysis regardless.
2. Read the generated `analysis.json` and any tool output.
3. Summarize key findings per the "Interpreting results" table below.
4. Ensure the web dashboard is running (see "Web dashboard" section).

```bash
python3 utils/analyze_job.py /outputs/<job>.log
python3 utils/analyze_job.py /outputs/<job_dir>/
python3 utils/analyze_job.py /outputs/local_2026*
```

The dispatcher detects which artifacts exist and runs only the relevant tools:
- **Log file with TGS data** -> `tgs_tagger.py`
- **`*.xplane.pb`** -> `TraceLens_generate_perf_report_jax`
- **`xla_dump/*.gpu_after_optimizations.txt`** -> `IRLens_analyze_hlo_ir.py`

For running jobs, pass `-f` to force re-analysis (bypasses staleness check). TGS metrics are computed but renames are always deferred until the job finishes:

```bash
python3 utils/analyze_job.py -f /outputs/<job>.log
```

## Job output layout

```
/outputs/<JOB_ID>-<JOB_NAME>[-TGS_<VALUE>]/
  log -> ../<log_file>                          # symlink to log file
  analysis.json                                 # structured metrics (written by analyze_job.py)
  xla_dump/                                     # if _env_ENABLE_XLA_DUMP=1
    module_NNNN.jit_train_step.*_gpu_after_optimizations.txt
    *-buffer-assignment.txt
  <run_name>/tensorboard/plugins/profile/<ts>/
    <hostname>.xplane.pb                        # if profiler=xplane
  tracelens/                                    # created by TraceLens
    <ts>/csvs/*.csv                               # per profiling window
    <ts>/*_report.xlsx
```

The `.log` file sits alongside the directory in `/outputs/`.

When `enable_checkpointing=true`, `OUTPUT_PATH` points to a shared model-based directory instead of the per-job directory. Profiler traces (`*.xplane.pb`) and TensorBoard events may end up outside the job directory. `analyze_job.py` parses `Config param tensorboard_dir` from the log to locate these external artifacts. The resolved paths are stored in `analysis.json` under `artifacts.tensorboard_dir` and `artifacts.profile_dir` (when external). The dashboard serves external profile files transparently.

## Individual tools

### tgs_tagger (TGS metrics)

Extracts steady-state Tokens/GPU/Second from the log, renames the log file and job directory.

```bash
utils/tag_tgs.sh <log_file_or_glob>
utils/tag_tgs.sh -f <log_file>       # force on running job
```

- Discards warmup steps 0-4, measures from steps 5-14.
- Needs `steps >= 15` for a full steady-state window.
- With `-f` on a running job: renames log file immediately, defers directory rename until the job finishes.

### TraceLens (runtime trace analysis)

GPU kernel utilization, GEMM performance, and communication patterns from xplane traces.

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
TraceLens_generate_perf_report_jax \
    --profile_path <xplane.pb> \
    --output_xlsx_path <job_dir>/tracelens/<ts>/report.xlsx \
    --output_csvs_dir <job_dir>/tracelens/<ts>/csvs
```

Find xplane files: `<job_dir>/**/tensorboard/plugins/profile/**/*.xplane.pb`

`analyze_job.py` selects **node 0's xplane** for analysis (parsed from `SLURM_JOB_NODELIST` in the log). In SPMD training all hosts execute the same program, so node 0 is representative. Distributed profiling may scatter host traces across multiple timestamp directories — filtering by node 0 naturally deduplicates to one trace per profiling step.

If TraceLens fails with protobuf/xprof errors (TF 2.19+), see [tracelens-patches.md](tracelens-patches.md) for required patches.

### IRLens (HLO static analysis)

Parses the XLA HLO dump for communication/computation ops.

```bash
utils/IRLens_analyze_hlo_ir.py <hlo_file>
utils/IRLens_analyze_hlo_ir.py <hlo_file> --op communication
utils/IRLens_analyze_hlo_ir.py <hlo_file> --op computation
utils/IRLens_analyze_hlo_ir.py <hlo_file> --name --topology --fusion-stats
```

Find the HLO file: `<job_dir>/xla_dump/module_*.jit_train_step.*_gpu_after_optimizations.txt` (use the highest module number if multiple exist).

## Running jobs

The dispatcher and tgs_tagger detect running jobs by checking for the `JOB SUMMARY` log marker and file modification time (15 min threshold).

- `analyze_job.py` never forwards `-f` to `tgs_tagger`, so **no renames happen** during analysis of running jobs. Renames happen automatically on the next analysis after the job finishes.
- Standalone `tgs_tagger -f` on a running job renames the log file (safe — fd follows inode) but defers the directory rename.
- `analyze_job.py -f` only bypasses the staleness check; it does not force renames.
- TraceLens needs a completed profiler trace; if `*.xplane.pb` doesn't exist yet, it's skipped.
- IRLens works on running jobs if `xla_dump/` is already populated (XLA dumps during compilation, before training steps).
- Re-running after the job finishes is handled automatically: the staleness check detects that `job_status` was `running` and triggers a full re-analysis, which renames the log/directory and picks up final artifacts.

## Interpreting results

After analysis, summarize key findings:

| Metric | Source | What to look for |
|--------|--------|------------------|
| **TGS** (steady-state) | tgs_tagger | Primary throughput metric |
| **MFU** | tgs_tagger log output | Model FLOPS utilization |
| **GPU compute %** | TraceLens `gpu_events_averages.csv` | Time on actual compute kernels |
| **Exposed comm %** | TraceLens | Communication NOT overlapped with compute (lower is better) |
| **Idle %** | TraceLens | GPU doing nothing (should be near 0) |
| **Comm ops** | IRLens `--op communication` | Total collectives per step |
| **Compute ops** | IRLens `--op computation` | Fusion kernels, GEMMs, attention calls |

Large per-GPU variance in compute % indicates load imbalance. High exposed comm % suggests opportunities for better overlap (latency hiding scheduler, pipelining flags in `train_env.sh`).

## Web dashboard (required post-analysis step)

**After every analysis run, always ensure the dashboard is running and give the user the URL.**

1. Check if port 8080 is already listening:
   ```bash
   ss -tlnp | grep 8080
   ```
2. If **not running**, install deps and start in background:
   ```bash
   pip install fastapi uvicorn   # one-time
   utils/perf_server.py --host 0.0.0.0 --port 8080 &
   ```
3. **Always** tell the user the dashboard URL: `http://<host>:8080`

`analyze_job.py` also prints a dashboard hint at the end of its output — if the server is running it shows the URL, otherwise it shows the start command.

Features: job listing with sortable metrics, per-step TGS/MFU/loss charts, HLO viewer with comm/compute filters, GPU utilization pie and per-GPU bar charts, file browser with Perfetto links for xplane traces, per-directory and full-job zip download, and side-by-side job comparison.
