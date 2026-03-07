---
name: batch-sweep
description: "Four sweep operations: (1) Model perf sweep — find optimal batch size / TGS for a model. Use for: sweep batch size, tune TGS, benchmark throughput, find optimal config. (2) Node perf sweep — compare per-node GPU performance to find outliers. Use for: check nodes, node performance, find slow node, compare nodes. (3) Node network health sweep — detect inter-node network issues via multi-node bisection. Use for: network health, IB issues, RCCL problems, node pair testing, isolate network problem. (4) Commit validation sweep — test all model configs against a commit. Use for: regression test, validate commit, test all models, smoke test, CI."
---

# Sweep Operations

Four independent sweep operations for performance tuning and health validation. Each can be run standalone.

| Sweep | What it tests | Job type | When to use |
|-------|--------------|----------|-------------|
| [Model perf sweep](#model-perf-sweep) | Batch size / config for max TGS | Multi-node, target model | Tuning a new model config |
| [Node perf sweep](#node-perf-sweep) | Per-node GPU compute health | 1N per node, llama2-70b | Checking a set of nodes for outliers |
| [Node network health sweep](#node-network-health-sweep) | Inter-node IB/RCCL health | 2N/4N subsets, llama2-70b | Isolating network problems between nodes |
| [Commit validation sweep](#commit-validation-sweep) | All model configs runnable + TGS | Multi-node per model | Regression test after code/image changes |

## Common prerequisites

- host-cmd available (`python3 /maxtext-slurm/.host-cmd/host_cmd.py --ping`)
- A fixed nodelist (always use `-w <nodelist>` to avoid cross-node variance)

## Monitoring policy (applies to all sweeps)

Every submitted job must be actively monitored. Never submit-and-forget.

**Polling:** Check the log every 60–120s. If no new output, exponential backoff up to 5 min.

**Hang detection:** A job is hanging if no new log output for **10+ minutes** while `squeue` shows it `RUNNING`. Actions:

1. Verify the job is yours (`outputs/<jobid>-*` exists).
2. Cancel: `python3 /maxtext-slurm/.host-cmd/host_cmd.py "scancel <jobid>" --timeout 10`
3. Classify and decide:

| Hang phase | Likely cause | Retry? |
|------------|-------------|--------|
| Before `Memstats` (RCCL init) | Transient RCCL/IB | Yes, once |
| After `Memstats`, before `completed step: 0` | Silent OOM during XLA compile | No — treat as OOM |
| After `completed step: 0` | RCCL error mid-training | Yes, once |

**Failure detection:**

| Log pattern | Meaning | Action |
|-------------|---------|--------|
| `RESOURCE_EXHAUSTED: Out of memory` | OOM | No retry at this config. Record allocation size. |
| `HSA_STATUS_ERROR` / `rocdevice.cpp: Aborting` | GPU hardware error | Exclude that node, retry on remaining nodes. |
| `ECC` errors in preflight | Bad GPU memory | Exclude node for entire sweep. |
| One node `exit code 1`, others continue | Single-node crash | Exclude that node, retry. |
| All nodes fail consistently | Config issue | **Stop the sweep.** Fix config before continuing. |

**Bail-out rules (prevent infinite loops):**

- **Max 2 retries per job.** If a job fails/hangs 3 times, mark it as failed and move on.
- **Max 2 node exclusions per sweep.** If 3+ nodes are excluded, **stop the sweep** — the cluster is unhealthy. Report to user.
- **If the same error repeats on retry with different nodes**, it's a config issue, not a node issue. **Stop and report.**
- **Never retry an OOM** at the same batch size. The result is deterministic.

---

## Model perf sweep

Find the optimal `per_device_batch_size` (or other parameters) for maximum steady-state TGS on a fixed set of nodes.

**Additional prerequisite:** A working `.gpu.yml` config that runs at some batch size.

### Step 1: Gather inputs

Ask the user for:
1. **Model name** (must have a `.gpu.yml` in `configs/`)
2. **Node count and nodelist** (e.g. `-N 8 -w chi[2832,2863,2867-2868]`)
3. **Parameter to sweep** (default: `per_device_batch_size`)
4. **Any env overrides** (e.g. `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.93`)

### Step 2: Estimate memory budget

Read the model config and estimate:

```
pool = GPU_VRAM * XLA_PYTHON_CLIENT_MEM_FRACTION
param_memory ≈ (total_params * bytes_per_param) / total_sharding
available = pool - param_memory
```

For `bytes_per_param`, use `weight_dtype` from the config (bf16=2, fp32=4). For `total_sharding`, multiply all non-1 parallelism axes (both DCN and ICI; for MoE with `expert_shard_attention_option: "fsdp"`, expert axis also shards non-expert weights).

If a prior run exists, use `Memstats: After params initialized` from its log for exact param memory instead of estimating.

### Step 3: Choose sweep strategy

Use the **upper-bound probe** strategy — NOT binary search or linear scan:

1. **Estimate max batch** from memory budget:
   ```
   activation_per_batch ≈ seq_len × emb_dim × num_layers × 12 bytes
   max_batch ≈ available / activation_per_batch
   ```

2. **First probe at ~75% of estimated max** — if it works, probe higher; if OOM, we have an upper bound.

3. **Second probe at the upper bound** (e.g. 1.5× first) — establishes the ceiling.

4. **Narrow the range** — with a working lower bound and OOM upper bound, try the midpoint. One or two more runs to converge.

This finds the optimal in **3–4 runs** vs. 6–8 for binary search.

### Step 4: Submit jobs

Always use the **same nodelist** via `-w` to avoid hardware variance. Use CLI overrides to sweep without modifying the config file:

```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py \
  "cd <repo_path> && ./submit.sh <model> -N <nodes> -w <nodelist> -- per_device_batch_size=<N>" \
  --timeout 30
```

For env sweeps:
```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py \
  "cd <repo_path> && ./submit.sh <model> -N <nodes> -w <nodelist> -- _env_XLA_PYTHON_CLIENT_MEM_FRACTION=.93 per_device_batch_size=<N>" \
  --timeout 30
```

Submit one job at a time to the same nodes (they'll queue sequentially), or submit to different nodesets for parallel sweeps.

### Step 5: Monitor jobs

Apply the [Monitoring policy](#monitoring-policy-applies-to-all-sweeps) for hang detection, failure handling, and bail-out rules.

**Expected timeline for a 15-step run:**

| Phase | Expected duration | What to look for |
|-------|-------------------|------------------|
| Preflight + container pull | 1–3 min | `[INFO]` lines, `=== Pre-flight` |
| Model init + RCCL setup | 2–5 min | `Memstats: After params initialized` |
| XLA compilation (step 0) | 2–5 min | `completed step: 0` |
| Steady-state steps 1–14 | varies by model | `completed step: N` |

When excluding a node due to failures, the total node count may drop below what the model needs. If so, **stop the sweep** and inform the user.

### Step 6: Collect steady-state TGS

For each completed run, extract **steady-state TGS** from steps 5–14 (skip steps 0–4 for compilation warmup):

```
grep "^0: completed step: \(5\|6\|7\|8\|9\|10\|11\|12\|13\|14\)," outputs/<jobid>-*.log
```

Use node 0's values (all nodes report identical metrics). Compute the average `Tokens/s/device` across steps 5–14.

### Step 7: Interpret results and decide next probe

| Result | Interpretation | Next action |
|--------|---------------|-------------|
| OOM | Batch too large | Probe lower (midpoint between last-good and this) |
| Runs, TGS higher than previous | Better batch size | Probe higher if room remains |
| Runs, TGS lower or flat | Past the sweet spot | The previous batch size was optimal |
| Hang (retried, still hangs) | Treat as OOM | Probe lower |

**Key insight**: TGS does not always increase with batch size. At some point, communication overhead or memory pressure causes regression. The optimal is the batch with the **highest TGS**, not the largest batch that fits.

### Step 8: Record results

After finding the optimal, build a summary table:

```markdown
| Batch | Job | Avg TGS/device | Avg MFU | Status |
|-------|-----|----------------|---------|--------|
| 8     | 1234| 1,190          | 11.92%  | works  |
| 12    | 1235| 1,339          | 13.42%  | works  |
| 16    | 1236| 1,400          | 14.02%  | works ← optimal |
| 20    | 1237| 1,371          | 13.73%  | works (regression) |
| 24    | 1238| OOM            | —       | OOM    |
```

Update the config file with the optimal value and amend the commit if on a feature branch.

### Sweeping other parameters

The same strategy works for any parameter. Common sweeps:

| Parameter | Override syntax | Notes |
|-----------|----------------|-------|
| `per_device_batch_size` | `per_device_batch_size=N` | Most common sweep target |
| `max_target_length` | `max_target_length=N` | Memory scales linearly with seq len |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.93` | Higher gives more memory but may starve RCCL; XLA may inflate allocations |
| `remat_policy` | `remat_policy=full` vs `minimal_flash` | `full` uses less memory, more compute |
| `quantization` | `quantization=fp8` | Halves weight memory |

For multi-parameter sweeps (e.g. batch × mem_fraction), fix one and sweep the other. Document which combination won.

### Pitfalls

- **Different nodes → different results.** Always use `-w <nodelist>`. Node health, IB fabric quality, and thermal throttling vary.
- **XLA memory inflation.** Giving XLA more memory (`MEM_FRACTION`) doesn't always help — it may inflate allocations to fill the pool, eating the extra headroom.
- **Step 0 is not representative.** It includes XLA compilation overhead. Never use step 0 metrics for comparison.
- **MFU can regress at large batch.** Communication overhead grows with batch size for MoE models. The optimal batch is where TGS peaks, not where memory maxes out.
- **Hangs and failures are handled by the shared [Monitoring policy](#monitoring-policy-applies-to-all-sweeps).** Max 2 retries per job, max 2 node exclusions per sweep, never retry OOM at the same batch size.

---

## Node perf sweep

Compare per-node GPU compute performance and find outliers. Use when the user gives a list of nodes and wants to check for performance discrepancy.

### Workflow

1. **Submit 1N baseline jobs in parallel** to each node using `llama2-70b` (dense, fast to compile, stable TGS):

   ```bash
   for node in chi2832 chi2863 chi2867 chi2868; do
     python3 /maxtext-slurm/.host-cmd/host_cmd.py \
       "cd <repo_path> && ./submit.sh 70b:${node}: -N 1 -w ${node}" \
       --timeout 30
   done
   ```

   The `:${node}:` alias isolates each job's output directory. All jobs run in parallel since they target different nodes.

2. **Monitor all jobs.** Apply the [Monitoring policy](#monitoring-policy-applies-to-all-sweeps). Since these are 1N jobs, hangs/failures are node-specific — mark that node as unhealthy.

3. **Collect steady-state TGS** (steps 5–14) from each job:

   ```bash
   grep "^0: completed step: 10," outputs/*-JAX-llama2-70b.log
   ```

4. **Build a per-node comparison table:**

   ```markdown
   | Node | Job | TGS/device | vs. median | Status |
   |------|-----|------------|------------|--------|
   | chi2832 | 9701 | 1,842 | +0.1% | healthy |
   | chi2863 | 9702 | 1,838 | -0.1% | healthy |
   | chi2867 | 9703 | 1,840 | 0.0% | healthy |
   | chi2868 | 9704 | 1,695 | -7.9% | DEGRADED |
   ```

5. **Evaluate:**

   | TGS vs. median | Interpretation |
   |-----------------|---------------|
   | Within ±2% | Healthy |
   | 2–5% below | Marginal — monitor but usable |
   | 5%+ below | Degraded — exclude from multi-node jobs |
   | Failed/hung | Unhealthy — exclude and report to user |

6. **Report** the table and recommendation to the user. If this was done before a batch sweep, recommend the healthy subset as the nodelist.

---

## Node network health sweep

Detect inter-node network issues (bad IB links, RCCL problems between specific node pairs) by running multi-node jobs across subsets and bisecting to isolate the problem.

The node perf sweep (1N) only tests GPU compute. This sweep tests the **network fabric between nodes** using `llama2-70b` with 2N or 4N jobs.

Since this only tests network health (not steady-state TGS), use `-- steps=1` to minimize runtime. A single training step is enough to exercise RCCL collectives and expose network issues.

1. **Start with one job across all nodes** to get a baseline:

   ```bash
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:all-nodes: -N 8 -w chi[2832,2863,2867-2868,2870,2872,2880-2881] -- steps=1" \
     --timeout 30
   ```

2. **If baseline is healthy**, no network issues — done.

3. **If baseline is degraded** (TGS below expected), bisect with half-sized groups:

   ```bash
   # Group A: first half
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:group-a: -N 4 -w chi[2832,2863,2867-2868] -- steps=1" \
     --timeout 30
   # Group B: second half
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:group-b: -N 4 -w chi[2870,2872,2880-2881] -- steps=1" \
     --timeout 30
   ```

4. **Compare group TGS:**

   | Scenario | Interpretation | Next action |
   |----------|---------------|-------------|
   | Both groups healthy | Issue is cross-group (IB link between groups) | Try mixed groups to find the bad pair |
   | One group degraded | Problem node is in that group | Bisect again (2N) within the bad group |
   | Both degraded | Multiple issues or shared infrastructure | Report to user, likely needs HW team |

5. **Narrow with 2N jobs** to pinpoint the exact node:

   ```bash
   # If group A was bad (chi[2832,2863,2867-2868]):
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:pair-a1: -N 2 -w chi[2832,2863] -- steps=1" \
     --timeout 30
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:pair-a2: -N 2 -w chi[2867-2868] -- steps=1" \
     --timeout 30
   ```

6. **Build isolation table:**

   ```markdown
   | Group | Nodes | TGS/device | vs. expected | Status |
   |-------|-------|------------|-------------|--------|
   | all-8 | chi[2832,...,2881] | 1,320 | -5.2% | DEGRADED |
   | group-a | chi[2832,2863,2867-2868] | 1,385 | -0.4% | healthy |
   | group-b | chi[2870,2872,2880-2881] | 1,310 | -5.8% | DEGRADED |
   | pair-b1 | chi[2870,2872] | 1,388 | +0.0% | healthy |
   | pair-b2 | chi[2880-2881] | 1,295 | -6.9% | DEGRADED |
   ```

   In this example, the issue is between chi2880 and chi2881 (or one of them has a bad NIC). Cross-reference with the 1N per-node sweep — if both were healthy individually, the problem is the network link between them.

7. **Report** findings and recommend excluding the problematic node(s) or escalating to the infrastructure team for IB link repair.

---

## Commit validation sweep

Compare TGS of clean `main` branch vs. the current working tree (with unstaged/staged changes) across all model configs. Use to validate that code changes, config tweaks, or image updates don't regress performance or break models.

### Workflow

1. **Enumerate all model configs:**

   ```bash
   ls configs/*.gpu.yml | sed 's|configs/||;s|\.gpu\.yml||'
   ```

   Determine the required node count for each model from its config (`dcn_fsdp_parallelism` and other DCN axes).

2. **Run the baseline on clean `main`.**

   First, check the current branch and stash any uncommitted changes:
   ```bash
   git stash  # if there are uncommitted changes
   git checkout main
   ```

   Submit all models with `steps=15` and `dataset_type=synthetic`, on a fixed nodelist. Tag each job with `:main:` to identify baseline runs:

   ```bash
   for model in llama2-70b llama3.1-405b mixtral-8x22b grok-1 deepseek3-671b kimi-k2-1t; do
     python3 /maxtext-slurm/.host-cmd/host_cmd.py \
       "cd <repo_path> && ./submit.sh ${model}:main: -N <nodes> -w <nodelist> -- dataset_type=synthetic" \
       --timeout 30
   done
   ```

   Monitor all jobs. Collect steady-state TGS (steps 5–14) from each passing model. This is the **baseline**.

3. **Run the candidate on the current HEAD.**

   Switch back to the working branch and restore changes:
   ```bash
   git checkout <branch>
   git stash pop  # if changes were stashed
   ```

   Submit the same models, tagged with `:candidate:`:

   ```bash
   for model in llama2-70b llama3.1-405b mixtral-8x22b grok-1 deepseek3-671b kimi-k2-1t; do
     python3 /maxtext-slurm/.host-cmd/host_cmd.py \
       "cd <repo_path> && ./submit.sh ${model}:candidate: -N <nodes> -w <nodelist> -- dataset_type=synthetic" \
       --timeout 30
   done
   ```

   **Same nodelist as the baseline** — critical for valid comparison.

4. **Monitor all jobs.** Apply the [Monitoring policy](#monitoring-policy-applies-to-all-sweeps). For this sweep:
   - **OOM on candidate but not baseline** = regression. Flag as broken.
   - **OOM on both** = pre-existing issue, not a regression.
   - **Hang during RCCL init** = retry once. If persistent, flag as broken.
   - **Hang during compile** = possible XLA regression. Flag as broken.

5. **Build a comparison report:**

   ```markdown
   | Model | Baseline Job | Baseline TGS | Candidate Job | Candidate TGS | Delta | Status |
   |-------|-------------|-------------|--------------|--------------|-------|--------|
   | llama2-70b | 9801 | 1,840 | 9811 | 1,842 | +0.1% | PASS |
   | llama3.1-405b | 9802 | 980 | 9812 | 985 | +0.5% | PASS |
   | deepseek3-671b | 9803 | 1,416 | 9813 | 1,416 | 0.0% | PASS |
   | kimi-k2-1t | 9804 | 1,149 | 9814 | 1,149 | 0.0% | PASS |
   | grok-1 | 9805 | 1,200 | 9815 | FAIL (OOM) | — | REGRESSION |
   ```

6. **Evaluate:**

   | Outcome | Interpretation | Action |
   |---------|---------------|--------|
   | All PASS, TGS within ±2% | Candidate is safe | Report success |
   | All PASS, one model TGS regressed >5% | Performance regression | Flag model + investigate |
   | Candidate FAIL on model that passes on main | Breakage introduced | Flag broken models with failure details |
   | Both baseline and candidate FAIL | Pre-existing issue | Note but don't blame the candidate |
   | All candidate models FAIL | Fundamental issue | Stop — the change is broken |

7. **Report** the full comparison table to the user. Highlight regressions and failures clearly, distinguishing new breakage from pre-existing issues.
