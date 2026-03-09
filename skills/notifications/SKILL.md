---
name: notifications
description: Send Telegram notifications to the user. Use when the user asks to be notified, messaged, or alerted about task completion, job status, or any result. This is a cross-cutting skill — other skills (batch-sweep, model-config, job-triage) can use it when the user explicitly requests notification.
---

# Notifications

Send Telegram messages to the user via `utils/telegram_bot.sh`. Use when the user says things like "send me a TG message", "notify me when done", or "alert me on Telegram".

## Sending flow

**NEVER read, print, or log the contents of `~/.tg_env`.** Only check if the file exists. The script auto-sources credentials internally.

Try in order. Stop at the first success.

### Step 1: Try from the container

```bash
test -f ~/.tg_env && echo "EXISTS" || echo "NOT FOUND"
```

If `~/.tg_env` exists, send directly:

```bash
utils/telegram_bot.sh send "Your message here"
```

For multi-line messages, pipe from stdin:

```bash
echo "Line 1
Line 2" | utils/telegram_bot.sh send
```

If this succeeds, done. If `~/.tg_env` doesn't exist, go to Step 2.

### Step 2: Try from the host via host-cmd

Check if host-cmd is available:

```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py --ping --timeout 5
```

If it does NOT respond `ALIVE`, go to Step 3.

If alive, check host credentials exist (never read or print the file contents):

```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 10 "test -f ~/.tg_env && echo EXISTS || echo NOT_FOUND"
```

If credentials exist, send using the **write-to-file pattern** (direct quoting through host-cmd breaks on special characters):

```bash
# Write message to a temp file on the host
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 10 "cat > /tmp/tg_msg.txt << 'EOF'
Your message here.
Multiple lines are fine.
Special chars & (parens) work.
EOF"

# Pipe the file into telegram_bot.sh
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 15 \
  "cat /tmp/tg_msg.txt | bash utils/telegram_bot.sh send"
```

If this succeeds, done. If credentials don't exist on the host either, go to Step 3.

### Step 3: Report failure and offer help

Tell the user what failed and give minimal next steps. Example:

> Could not send Telegram notification — no credentials found.
> `~/.tg_env` is missing in both the container and the host.
> Want me to help set up a Telegram bot? (takes ~2 minutes)

Or if host-cmd is unavailable:

> Could not send Telegram notification — `~/.tg_env` not found in the container, and host-cmd is not available.
> Want me to help set up Telegram credentials locally?

If the user says yes, walk them through `docs/notifications.md` setup:

1. Message @BotFather on Telegram → `/newbot` → get **bot token**
2. Start a chat with the bot, get **chat ID** from `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Save: `echo 'export TG_BOT_TOKEN="..."' >> ~/.tg_env && echo 'export TG_CHAT_ID="..."' >> ~/.tg_env`
4. Test: `utils/telegram_bot.sh send "Hello from $(hostname)"`

## Host-cmd pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `source: not found` | host-cmd uses `/bin/sh`, not bash | Don't use `source ~/.tg_env`; the script auto-sources it |
| `syntax error near unexpected token` | Special chars in inline message | Use the write-to-file pattern above |
| `command not found` on host | `utils/` path is relative to repo root | host-cmd cwd is already the repo root; `bash utils/telegram_bot.sh` works |
| Empty message error | Heredoc EOF marker was indented | Use unindented `EOF` marker |

## Message formatting

Keep messages concise. Telegram has a 4096-char limit per message (the script auto-splits longer messages). Use plain text structure:

```
<Task> complete

Key result 1
Key result 2
...

Summary line
```

Avoid Markdown special characters (`_`, `*`, `[`, `` ` ``) in dynamic content (model names, file paths) — they can break Telegram's Markdown parser. If the message contains such characters, set `PARSE_MODE=""` for plain text:

```bash
PARSE_MODE="" utils/telegram_bot.sh send "file_names_with_underscores"
```

Or via host-cmd:

```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 15 \
  "PARSE_MODE='' cat /tmp/tg_msg.txt | bash utils/telegram_bot.sh send"
```

## Integration with other skills

This skill is opt-in. Only use it when the user explicitly asks for notification. Typical integration points:

- **batch-sweep**: "notify me when the sweep is done" → send results table after Step 8
- **model-config**: "TG me when the test job finishes" → send job status after Step 7
- **job-triage**: "alert me if the job fails" → send failure summary

Do not proactively send notifications unless the user requested them.
