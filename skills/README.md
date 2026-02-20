# AI Skills

Structured instructions for AI coding assistants. Each skill is a self-contained guide that the agent reads on demand when the user's request matches its trigger keywords.

## Available skills

| Skill | Trigger |
|-------|---------|
| [performance-analysis](performance-analysis/SKILL.md) | "analyze job", "TGS", "TraceLens", "IRLens", profiling tasks |
| [job-log-triage](job-log-triage/SKILL.md) | "triage", "diagnose", "why did job fail", "is the job hanging", crash/hang/timeout/OOM/NCCL errors, job status |
| [tsdb-diagnosis](tsdb-diagnosis/SKILL.md) | "diagnose with TSDB", "check GPU health", "check network", "query prometheus", "metrics", incident root cause analysis |

## How agents discover skills

Both [Cursor](https://cursor.com/) and [Claude Code](https://docs.anthropic.com/en/docs/claude-code) read `CLAUDE.md` at the repo root, which references skill files by path.
