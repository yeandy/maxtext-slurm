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

# Perform a single sendMessage API call.
# Args: message, parse_mode (may be empty for plain text)
# Prints the API response (or curl error) to stdout.
_do_send() {
    local message="$1"
    local parse_mode="$2"
    local api_url="https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage"

    local curl_args=(
        -sS
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

    curl "${curl_args[@]}" 2>&1
}

# Split a message into Telegram-safe chunks (max 4096 chars each).
# Tracks Markdown ``` code blocks: closes them at chunk boundaries
# and reopens in the next chunk so formatting is preserved.
# Outputs NUL-delimited chunks for safe reading with `read -d ''`.
_split_message() {
    local message="$1"
    local max_len=4096
    # Reserve space for code fence close/open and [n/m] part prefix
    local overhead=40
    local effective=$((max_len - overhead))

    # Fast path: message already fits
    if [[ ${#message} -le $max_len ]]; then
        printf '%s\0' "$message"
        return
    fi

    local chunk=""
    local in_code_block=false

    while IFS= read -r line || [[ -n "$line" ]]; do
        local candidate
        if [[ -z "$chunk" ]]; then
            candidate="$line"
        else
            candidate="${chunk}"$'\n'"${line}"
        fi

        if [[ ${#candidate} -gt $effective && -n "$chunk" ]]; then
            # Adding this line would exceed the limit — flush current chunk
            if [[ "$in_code_block" == true ]]; then
                chunk+=$'\n```'
            fi
            printf '%s\0' "$chunk"

            # Start next chunk, reopening code block if we were inside one
            if [[ "$in_code_block" == true ]]; then
                chunk=$'```\n'"$line"
            else
                chunk="$line"
            fi
        else
            chunk="$candidate"
        fi

        # Track code block state: odd number of ``` on a line toggles the flag
        local tmp="$line"
        local fence_count=0
        while [[ "$tmp" == *'```'* ]]; do
            fence_count=$((fence_count + 1))
            tmp="${tmp#*\`\`\`}"
        done
        if [[ $((fence_count % 2)) -eq 1 ]]; then
            if [[ "$in_code_block" == true ]]; then
                in_code_block=false
            else
                in_code_block=true
            fi
        fi
    done <<< "$message"

    # Flush remaining content
    if [[ -n "$chunk" ]]; then
        printf '%s\0' "$chunk"
    fi
}

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

    # --- split into Telegram-safe chunks and send with plain-text fallback ---
    local parse_mode="${PARSE_MODE-Markdown}"

    local -a chunks=()
    while IFS= read -r -d '' chunk; do
        chunks+=("$chunk")
    done < <(_split_message "$message")

    local total=${#chunks[@]}

    if [[ $total -eq 0 ]]; then
        echo "Error: Internal error — message produced no chunks" >&2
        exit 1
    fi

    for i in "${!chunks[@]}"; do
        local part=$((i + 1))
        local chunk="${chunks[$i]}"

        # Add part indicator for multi-part messages
        if [[ $total -gt 1 ]]; then
            chunk="[${part}/${total}] ${chunk}"
        fi

        local response
        response=$(_do_send "$chunk" "$parse_mode")

        if echo "$response" | grep -q '"ok":true'; then
            continue
        fi

        # Retry this chunk as plain text
        if [[ -n "$parse_mode" ]]; then
            echo "Warning: parse_mode=${parse_mode} failed on part ${part}/${total}, retrying as plain text..." >&2
            response=$(_do_send "$chunk" "")
            if echo "$response" | grep -q '"ok":true'; then
                continue
            fi
        fi

        echo "Error: Telegram API request failed (part ${part}/${total})" >&2
        echo "$response" >&2
        exit 1
    done

    exit 0
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
