#!/bin/bash

# Telegram Bot CLI — a subcommand-based utility for interacting with the
# Telegram Bot API.
#
# Usage:
#   ./telegram_bot.sh <command> [args...]
#
# Commands:
#   send <message>    Send a message to a Telegram chat
#
# Credentials (in order of precedence):
#    1. Environment variables: TG_BOT_TOKEN, TG_CHAT_ID
#    2. Auto-sourced from ~/.tg_env if env vars are not set
#
# Examples:
#    # Credentials in ~/.tg_env — just use it:
#    ./telegram_bot.sh send "Hello world"
#
#    # Or pass explicitly:
#    TG_BOT_TOKEN=tok TG_CHAT_ID=123 ./telegram_bot.sh send "Hello world"
#    echo "Deploy *complete*" | TG_BOT_TOKEN=tok TG_CHAT_ID=123 ./telegram_bot.sh send

set -euo pipefail

readonly SCRIPT_NAME=$(basename "$0")

# ============================================================================
# COMMON HELPERS
# ============================================================================

readonly TG_ENV_FILE="$HOME/.tg_env"

# Source ~/.tg_env if credentials are missing and the file exists.
load_env() {
    if [[ -z "${TG_BOT_TOKEN:-}" || -z "${TG_CHAT_ID:-}" ]] && [[ -f "$TG_ENV_FILE" ]]; then
        echo "Sourcing credentials from $TG_ENV_FILE" >&2
        # shellcheck source=/dev/null
        source "$TG_ENV_FILE"
    fi
}

# Validate that required environment variables are set.
validate_env() {
    load_env

    if [[ -z "${TG_BOT_TOKEN:-}" ]]; then
        echo "Error: TG_BOT_TOKEN not set. Export it, or add it to $TG_ENV_FILE" >&2
        exit 1
    fi
    if [[ -z "${TG_CHAT_ID:-}" ]]; then
        echo "Error: TG_CHAT_ID not set. Export it, or add it to $TG_ENV_FILE" >&2
        exit 1
    fi
}

# ============================================================================
# COMMAND: send
# ============================================================================

# Send a message to Telegram.
#
# Message can come from:
#   1. Command-line arguments (joined with spaces)
#   2. Standard input (piped or redirected)
#
# Extra environment variables (optional):
#   PARSE_MODE             Markdown (default), MarkdownV2, HTML, or "" for plain text
#   DISABLE_NOTIFICATION   "true" to send silently
#   DISABLE_PREVIEW        "true" to disable link previews
cmd_send() {
    # --- read message ---
    local message=""

    if [[ $# -gt 0 ]]; then
        message="$*"
    elif [[ ! -t 0 ]]; then
        message=$(cat)
    else
        cat >&2 << EOF
Usage: $SCRIPT_NAME send <message>
    or: echo 'message' | $SCRIPT_NAME send
    or: $SCRIPT_NAME send <<< 'message'

Credentials (in order of precedence):
    1. TG_BOT_TOKEN / TG_CHAT_ID environment variables
    2. Auto-sourced from ~/.tg_env if env vars are not set

Optional environment variables:
    PARSE_MODE             Markdown (default), MarkdownV2, HTML, or "" for plain text
    DISABLE_NOTIFICATION   "true" to send silently
    DISABLE_PREVIEW        "true" to disable link previews
EOF
        exit 1
    fi

    if [[ -z "$message" ]]; then
        echo "Error: Message is empty" >&2
        exit 1
    fi

    validate_env

    # --- build request ---
    local parse_mode="${PARSE_MODE-Markdown}"
    local api_url="https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage"

    # Use --data-urlencode so curl properly percent-encodes every field.
    # This is critical for rich content: Markdown syntax (* _ ` [ ]),
    # code blocks (```), URLs containing & and =, etc.
    local curl_args=(
        -s
        -X POST
        "$api_url"
        --data-urlencode "chat_id=${TG_CHAT_ID}"
        --data-urlencode "text=${message}"
    )

    if [[ -n "$parse_mode" ]]; then
        curl_args+=(--data-urlencode "parse_mode=${parse_mode}")
    fi

    if [[ "${DISABLE_NOTIFICATION:-}" == "true" ]]; then
        curl_args+=(--data-urlencode "disable_notification=true")
    fi

    if [[ "${DISABLE_PREVIEW:-}" == "true" ]]; then
        curl_args+=(--data-urlencode "disable_web_page_preview=true")
    fi

    # --- send & check response ---
    local response
    response=$(curl "${curl_args[@]}")

    if echo "$response" | grep -q '"ok":true'; then
        exit 0
    else
        echo "Error: Telegram API request failed" >&2
        echo "$response" >&2
        exit 1
    fi
}

# ============================================================================
# MAIN DISPATCH
# ============================================================================

usage() {
    cat >&2 << EOF
Usage: $SCRIPT_NAME <command> [args...]

Commands:
    send <message>   Send a message to a Telegram chat

Credentials (in order of precedence):
    1. TG_BOT_TOKEN / TG_CHAT_ID environment variables
    2. Auto-sourced from ~/.tg_env if env vars are not set

Examples:
    $SCRIPT_NAME send "Hello world"
    TG_BOT_TOKEN=tok TG_CHAT_ID=123 $SCRIPT_NAME send "Hello world"
    echo "message" | $SCRIPT_NAME send

Setup: See docs/notifications.md for one-time Telegram bot creation and ~/.tg_env setup.
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    send)  cmd_send "$@" ;;
    -h|--help|help) usage ;;
    *)
        echo "Error: Unknown command: $COMMAND" >&2
        usage
        ;;
esac
