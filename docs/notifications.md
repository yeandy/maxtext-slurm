# Notifications

A programmable notification system built on [Telegram](https://telegram.org/). Two scripts, one credential store. Part of the [Observability](observability.md) stack but works independently — no `RAY=1` required.

| Script | Purpose |
|--------|---------|
| `utils/telegram_bot.sh` | General-purpose CLI for sending Telegram messages — use it in scripts, pipelines, or interactively |
| `utils/slurm_job_monitor.sh` | Automated [Slurm](https://slurm.schedmd.com/) job monitoring with push notifications (built on `telegram_bot.sh`) |

Both scripts auto-source credentials from `~/.tg_env` — set up once, use everywhere.

## One-time setup

Create a Telegram bot and save credentials:

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → get your **bot token**
2. Start a chat with your bot, then get your **chat ID** from `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Save credentials to `~/.tg_env`:
   ```bash
   echo 'export TG_BOT_TOKEN="your_token_here"' >> ~/.tg_env
   echo 'export TG_CHAT_ID="your_chat_id_here"' >> ~/.tg_env
   ```
4. Test: `utils/telegram_bot.sh send "Hello from $(hostname)"`

Tutorial: [Telegram Bot API — Getting Started](https://core.telegram.org/bots/tutorial)

## telegram_bot.sh

A subcommand-based CLI for interacting with the Telegram Bot API. Currently supports `send`, designed to be extended with future commands (e.g., `receive`).

### Sending messages

```bash
# Simple message (credentials from ~/.tg_env)
utils/telegram_bot.sh send "Checkpoint saved at step 10000"

# Pipe content
echo "Deploy complete on $(hostname)" | utils/telegram_bot.sh send

# Markdown formatting (default)
utils/telegram_bot.sh send "*Training complete* — final loss: 2.31"

# Plain text (no Markdown parsing) — safe for arbitrary content
PARSE_MODE="" utils/telegram_bot.sh send "file_names_with_underscores & special chars"

# Explicit credentials (override ~/.tg_env)
TG_BOT_TOKEN=tok TG_CHAT_ID=123 utils/telegram_bot.sh send "Hello"
```

### Usage patterns

Compose `telegram_bot.sh` into any workflow:

```bash
# Alert on failure
train.sh || utils/telegram_bot.sh send "Training failed on $(hostname)"

# Notify on completion
train.sh && utils/telegram_bot.sh send "Training done — check results"

# In a cron job
0 */6 * * * /path/to/telegram_bot.sh send "Disk usage: $(df -h / | tail -1)"

# Wrap any long-running command
some_command; utils/telegram_bot.sh send "Command finished (exit $?)"
```

### Credential resolution

Credentials are resolved in order:

1. `TG_BOT_TOKEN` / `TG_CHAT_ID` environment variables (inline or exported)
2. Auto-sourced from `~/.tg_env` if env vars are not set

### Optional environment variables

| Variable | Effect |
|----------|--------|
| `PARSE_MODE` | `Markdown` (default), `MarkdownV2`, `HTML`, or `""` for plain text |
| `DISABLE_NOTIFICATION` | `"true"` to send silently |
| `DISABLE_PREVIEW` | `"true"` to disable link previews |

## slurm_job_monitor.sh

Automated monitoring for Slurm jobs — sends push notifications for state changes, hang detection, and periodic log updates. Uses `telegram_bot.sh` under the hood.

```bash
utils/slurm_job_monitor.sh -j <slurm_job_id>
```

### What it monitors

| Notification | When |
|---|---|
| State changes | PENDING → RUNNING → COMPLETED / FAILED / CANCELLED / TIMEOUT |
| Hang alert | Log file stops updating for longer than the timeout (default 30m) |
| Resume alert | Log resumes after a hang was detected |
| Periodic updates | Last N log lines at configurable intervals (default 1h) |
| Signal handling | Graceful notification on Ctrl+C, `kill`, or SSH disconnect |

### Options

```bash
# Custom hang timeout (10 min) and update interval (30 min, last 20 lines)
utils/slurm_job_monitor.sh -j 12345 -t 600 -u 1800 -l 20

# Filter periodic updates to show only errors
utils/slurm_job_monitor.sh -j 12345 -g "ERROR|WARNING"

# Exclude noisy lines from updates
utils/slurm_job_monitor.sh -j 12345 -g "DEBUG|TRACE" -v
```

### Credential resolution

Credentials are resolved in order:

1. `TG_BOT_TOKEN` / `TG_CHAT_ID` environment variables
2. Auto-sourced from `~/.tg_env`
3. `-b` / `-c` flags (legacy, still supported — overrides env vars when both set)

### Tips

- Run in [`tmux`](https://github.com/tmux/tmux) so the monitor survives SSH disconnections.
- The monitor sends a notification when interrupted by signals (Ctrl+C, `kill`, SIGHUP) — only `kill -9` bypasses this.
- Jobs with no StdOut configured are still monitored for state changes; hang detection and log updates are disabled.

---

See also: [Observability](observability.md) for the full monitoring stack (dashboards, metrics, TSDB) | [Tooling](tooling.md) for the command reference overview.
