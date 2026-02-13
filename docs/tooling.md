# Tooling

## Tail a job log

```
# utils/tail_job_log.sh [JOB_ID | FILE | DIRECTORY]
#
# Follow (tail -f) the most recent job log file (.log or legacy .out).
# Default output directory: $JOB_WORKSPACE if set, otherwise outputs/ next to the scripts.
#
# Examples:
#   utils/tail_job_log.sh                                     # latest log in outputs/
#   utils/tail_job_log.sh 12345                               # latest log for Slurm job 12345
#   utils/tail_job_log.sh outputs/12345.log                   # tail a specific file
#   utils/tail_job_log.sh outputs/12345-JAX-llama2-70b/       # job dir (follows log symlink)
#   utils/tail_job_log.sh logs/                               # latest log in logs/
```

## Inspect Prometheus metrics after a job ends

```
# utils/prometheus.sh <command> [options]
#
# Commands:
#   view <data-dir> [-p PORT]    Start Prometheus UI to browse persisted metrics
#   list [<workspace>]           List jobs with saved Prometheus data
#   install                      Download the Prometheus binary to /tmp/prometheus
#
# Examples:
#   utils/prometheus.sh list                                                      # List jobs with metrics data
#   utils/prometheus.sh list /shared/maxtext_jobs                                 # Custom workspace
#   utils/prometheus.sh view outputs/12345-JAX-llama2-70b/prometheus              # Open Prometheus UI on :9090
#   utils/prometheus.sh view outputs/12345-JAX-llama2-70b/prometheus -p 9091      # Custom port
```

This starts a read-only [Prometheus](https://prometheus.io/) instance (no scraping) serving the persisted TSDB data. Open `http://localhost:9090` (or your chosen port) to query and graph all metrics (GPU, host, network, training scalars, and [Ray](https://www.ray.io/)) that were collected during the run.

## Send a Telegram message

Standalone utility for sending messages to [Telegram](https://telegram.org/) via the Bot API. Handles URL-encoding for rich content (Markdown, code blocks, special characters). Credentials are auto-sourced from `~/.tg_env` after [one-time setup](notifications.md#one-time-setup).

```bash
utils/telegram_bot.sh send "Hello *world*"
echo "Deploy complete" | utils/telegram_bot.sh send
```

See [Notifications](notifications.md) for full usage, programmable patterns, and options.

## Monitor a Slurm job with Telegram alerts

Sends [Telegram](https://telegram.org/) push notifications for [Slurm](https://slurm.schedmd.com/) job state changes, hang detection, and periodic log updates. Credentials are auto-sourced from `~/.tg_env` after [one-time setup](notifications.md#one-time-setup).

```bash
utils/slurm_job_monitor.sh -j <slurm_job_id>
```

See [Notifications: slurm_job_monitor.sh](notifications.md#slurm_job_monitorsh) for monitor options, credential resolution, and tips.

## Parse MaxText log files (and clean up failed jobs) — load test jobs only

Only used for load test job TGS (tokens/GPU/sec) calculation.

```
# utils/tag_tgs.sh [path | JOB_ID] [-c/--cleanup]
#
# Examples:
#   # Process latest log file in default outputs directory
#   utils/tag_tgs.sh
#
#   # Process log file(s) for a Slurm job ID (looks in default outputs dir)
#   utils/tag_tgs.sh 12345
#
#   # Process latest log file in specific directory
#   utils/tag_tgs.sh logs
#
#   # Process specific file (and rename its job folder if it exists)
#   utils/tag_tgs.sh outputs/12345-run.log
#
#   # Job directory (follows log symlink — tab-completion friendly)
#   utils/tag_tgs.sh outputs/12345-JAX-llama2-70b/
#
#   # Process all log files in a directory with cleanup mode
#   utils/tag_tgs.sh -c outputs/*.log
#
#   # Process latest files from multiple directories
#   utils/tag_tgs.sh dir1 dir2 dir3
#
#   # Cleanup mode with short flag
#   utils/tag_tgs.sh -c
```

## Clean up orphaned artifacts

See [Job Submission: How artifacts work](job-submission.md#how-artifacts-work) for details on the artifact system.

```bash
utils/cleanup_artifacts.sh        # dry-run: list all artifacts and their status
utils/cleanup_artifacts.sh -c     # remove orphans (confirms one by one)
utils/cleanup_artifacts.sh -c -y  # remove orphans (skip confirmation)
```
