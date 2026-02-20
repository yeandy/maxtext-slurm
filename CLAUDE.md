For performance analysis tasks, follow the instructions in `skills/performance-analysis/SKILL.md`.
For job triage tasks (failed, hanging, or running jobs), follow the instructions in `skills/job-log-triage/SKILL.md`.
For TSDB diagnosis tasks (metrics queries, GPU/network health, incident root cause), follow the instructions in `skills/tsdb-diagnosis/SKILL.md`.

**Multi-job performance comparisons** (e.g., "why is job B slower than job A?"): Start with `skills/tsdb-diagnosis/SKILL.md` (Multi-Job Comparison workflow) to check system-level metrics (process counts, network, I/O, GPU health) before running `skills/performance-analysis/SKILL.md`. TSDB surfaces root causes that TraceLens cannot see (CPU contention, RCCL resource leaks, network errors). Only fall back to TraceLens if the TSDB comparison is inconclusive.
