# AI Skills

Structured instructions for AI agents. Each skill encodes the methodology from very senior systems engineers — not just what commands to run, but how to interpret results, distinguish symptoms from root causes, and trace causal chains across the full stack. The agent reads the relevant skill on demand when the user's request matches its trigger keywords.

## Available skills

| Skill | Trigger |
|-------|---------|
| [performance-analysis](performance-analysis/SKILL.md) | "analyze job", "TGS", "TraceLens", "IRLens", profiling tasks |
| [job-log-triage](job-log-triage/SKILL.md) | "triage", "diagnose", "why did job fail", "is the job hanging", crash/hang/timeout/OOM/NCCL errors, job status |
| [tsdb-diagnosis](tsdb-diagnosis/SKILL.md) | "diagnose with TSDB", "check GPU health", "check network", "query prometheus", "metrics", incident root cause analysis |
| [coredump-debug](coredump-debug/SKILL.md) | "coredump", "core file", "SIGSEGV", "segfault", "crash dump", GDB backtrace analysis, crash root cause |
| [model-config-guide](model-config-guide/SKILL.md) | "add model", "create config", "model config", ".gpu.yml", parallelism, batch size, quantization, OOM tuning |
| [batch-sweep](batch-sweep/SKILL.md) | "sweep", "find optimal batch size", "tune TGS", "benchmark throughput", "maximize tokens per second" |
| [notifications](notifications/SKILL.md) | "notify me", "send TG message", "alert when done", "Telegram", cross-cutting notification for any skill |

## How agents discover skills

Both [Cursor](https://cursor.com/) and [Claude Code](https://docs.anthropic.com/en/docs/claude-code) read `CLAUDE.md` at the repo root, which references skill files by path.

For [Kubernetes](https://kubernetes.io/) job submission and direct-container runs, see [Kubernetes job submission](../docs/k8s-job-submission.md).
