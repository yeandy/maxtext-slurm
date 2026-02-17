# Performance

Profile training runs to find bottlenecks, analyze the results, then tune. All profiling and tuning settings live in `train_env.sh` and model config YAMLs. For system-level and training metrics (GPU thermals, power, VRAM, loss curves, throughput), see [Observability](observability.md).

## Profiling

### Runtime traces (xplane)

The [JAX profiler](https://docs.jax.dev/en/latest/profiling.html) captures GPU kernel timelines in xplane format. Enable it via CLI or in the model config:

```bash
submit.sh 70b -N 1 -- profiler=xplane skip_first_n_steps_for_profiler=3 profiler_steps=1
```

```yaml
profiler: "xplane"
upload_all_profiler_results: true
skip_first_n_steps_for_profiler: 3
profiler_steps: 1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profiler` | `""` (disabled) | Set to `"xplane"` to enable |
| `skip_first_n_steps_for_profiler` | `3` | Warmup steps to skip (avoids profiling compilation) |
| `profiler_steps` | `1` | Number of steps to profile |
| `upload_all_profiler_results` | `true` | Collect traces from all nodes (not just rank 0) |

Trace files are written to `base_output_directory` (i.e. `OUTPUT_PATH`).

### HLO IR dump

[XLA](https://openxla.org/)'s JIT compiler transforms JAX code into [HLO (High Level Operations)](https://openxla.org/xla/operation_semantics) IR before generating GPU kernels. HLO dumps capture the compiled computation graph — which collectives are fused, how loops are structured, and what kernels will execute each step — independent of actual execution timing.

Enable XLA dump for any run by passing `_env_ENABLE_XLA_DUMP=1`:

```bash
submit.sh           70b -N 1 -- steps=1 _env_ENABLE_XLA_DUMP=1
run_local.sh        70b      -- steps=1 _env_ENABLE_XLA_DUMP=1
in_container_run.sh 70b      -- steps=1 _env_ENABLE_XLA_DUMP=1
```

This appends the following XLA flags (configured in `train_env.sh`):

| Flag | Purpose |
|------|---------|
| `--xla_dump_hlo_as_text` | Emit human-readable `.txt` HLO files |
| `--xla_dump_hlo_module_re` | Regex filter for HLO modules (`^jit_train_step$` captures the main training step) |
| `--xla_dump_hlo_pipeline_re` | Regex filter for compiler passes (`(?i)gpu` captures GPU-specific passes) |
| `--xla_dump_to` | Output directory for dump files |

Dump files are written to `<OUTPUT_PATH>/xla_dump/`. Key files:

| File | Contents |
|------|----------|
| `*.before_optimizations.txt` | HLO before GPU compiler passes |
| `*.gpu_after_optimizations.txt` | Final optimized HLO that maps to GPU kernels |
| `*.gpu_after_optimizations-buffer-assignment.txt` | Memory buffer layout |
| `*.gpu_after_optimizations-memory-usage-report.txt` | Memory usage breakdown |
| `*.ir-with-opt.ll` | Optimized LLVM IR |
| `*.thunk_sequence.txt` | Runtime execution schedule |

## Analysis

### Visualize traces

Open trace files in [Perfetto](https://ui.perfetto.dev/) (recommended for large files) or `chrome://tracing`. Multi-node jobs produce one trace file per node. Each can be viewed individually, or merged into one file for side-by-side viewing:

```bash
utils/merge_xplane_traces.py /outputs/<job>/
utils/merge_xplane_traces.py node0.trace.json.gz node1.trace.json.gz -o combined.trace.json.gz
```

### TraceLens (runtime trace analysis)

[TraceLens](https://github.com/AMD-AGI/TraceLens) automates analysis of runtime trace files — GPU kernel utilization, GEMM performance, and communication patterns — from the xplane protobuf or JSON traces produced by the profiler.

```python
from TraceLens.TraceLens import JaxAnalyses

# GPU kernel utilization breakdown
averages, categorized, additional = JaxAnalyses.summarize_gpu_events("profile.xplane.pb")

# GEMM performance from runtime trace
gemms = JaxAnalyses.summarize_gpu_gemm_events_from_pb("profile.xplane.pb")

# Communication analysis (requires both trace and HLO buffer assignment)
comms = JaxAnalyses.summarize_gpu_communication_events("profile.xplane.pb", "buffer-assignment.txt")
```

Or generate a full performance report via CLI:

```bash
python generate_perf_report_jax.py --profile_path path/to/profile.xplane.pb --output_csvs_dir save/to/dir
```

See the [JAX analysis guide](https://github.com/AMD-AGI/TraceLens/blob/main/docs/jax_analyses.md) for the full TraceLens API.

### IRLens (HLO static analysis)

`IRLens_analyze_hlo_ir.py` extracts a hierarchical execution skeleton from an HLO dump — the control flow, communication ops, and computation ops with all plumbing stripped away:

```bash
# Show all ops (communication + computation)
utils/IRLens_analyze_hlo_ir.py xla_dump/module_0000.jit_train_step.gpu_after_optimizations.txt

# Communication ops only — all-gather, reduce-scatter, all-reduce patterns
utils/IRLens_analyze_hlo_ir.py <hlo_file> --op communication

# Computation ops only — fusion kernels, GEMM calls
utils/IRLens_analyze_hlo_ir.py <hlo_file> --op computation

# Full detail: variable names, communication topology, fusion breakdown
utils/IRLens_analyze_hlo_ir.py <hlo_file> --name --topology --fusion-stats
```

| Flag | Description |
|------|-------------|
| `--op {all,communication,computation}` | Filter by operation type (default: `all`) |
| `--name` | Show HLO result variable names and while-loop names |
| `--topology` | Show communication topology (replica groups, source-target pairs) |
| `--fusion-stats` | Break down fusion ops by subtype in the statistics summary |

**Example output** (communication only):

```
ENTRY %main:
  while i in range(8):
    all-gather-start | bf16[8,8192,8192] | jit(train_step)/... | layers.py:142
    reduce-scatter-start | bf16[8,8192,8192] | jit(train_step)/... | layers.py:218
    all-reduce-start | f32[1] | jit(train_step)/... | pipeline.py:95

# Total communication ops: 24
# 3 communication op categories:
#   all-gather-start                          8
#   all-reduce-start                          8
#   reduce-scatter-start                      8
```

Useful for verifying that sharding strategies (FSDP, tensor parallelism) produce the expected collective patterns and for spotting unexpected communication overhead.

## Tuning

All tunable settings live in `train_env.sh`. Per-run overrides can be passed via `_env_KEY=VALUE` after `--` without editing the file (see [Job Submission](job-submission.md)).

### XLA compiler flags

`train_env.sh` contains a curated set of `XLA_FLAGS` for performance tuning, organized by category. The entire block is commented out by default so the Docker image's built-in `XLA_FLAGS` are used as-is. To customize, uncomment the block and adjust individual flags:

| Category | Flags | What they control |
|----------|-------|-------------------|
| Core compiler | `xla_gpu_enable_cublaslt`, `xla_gpu_graph_level`, `xla_gpu_autotune_level` | GEMM library selection, graph capture, autotuning |
| GEMM / codegen | `xla_gpu_enable_triton_gemm`, `xla_gpu_enable_command_buffer` | Triton vs vendor GEMM kernels, command buffer use |
| Collective combining | `xla_gpu_all_gather_combine_threshold_bytes`, `xla_gpu_reduce_scatter_combine_threshold_bytes`, etc. | How aggressively XLA merges small collectives into larger ones (higher thresholds = more combining, but more memory) |
| Overlapping / pipelining | `xla_gpu_enable_latency_hiding_scheduler`, `xla_gpu_enable_pipelined_all_gather`, `xla_gpu_enable_pipelined_reduce_scatter`, `xla_gpu_enable_while_loop_double_buffering` | Overlap compute with communication to hide latency |

Some flags have model-specific trade-offs noted in the comments (e.g. double buffering may OOM on very large models even with adjusted combine thresholds). Start from the defaults, profile, then selectively enable flags based on what the traces and HLO analysis reveal.

### Environment variables

The rest of `train_env.sh` exports environment variables that control JAX, NCCL/RCCL, and vendor-specific libraries (memory, networking, InfiniBand, GPU compute, AMD/ROCm, Transformer Engine, GDR/DMA, compilation cache, etc.). The file is organized into commented sections with inline notes on trade-offs. Most defaults work well out of the box.

## Typical workflow

1. **Baseline run** — train with `steps=15` (minimum), `profiler=xplane`, and `_env_ENABLE_XLA_DUMP=1`.
2. **Tag TGS** — run `utils/tag_tgs.sh` to extract steady-state throughput (steps 5-14) and rename outputs.
3. **Analyze traces** — use TraceLens for GPU utilization breakdown (compute vs communication vs idle).
4. **Analyze HLO** — use IRLens to inspect the compiled computation graph (collective patterns, fusion structure).
5. **Tune** — adjust XLA flags and environment variables in `train_env.sh` based on findings.
