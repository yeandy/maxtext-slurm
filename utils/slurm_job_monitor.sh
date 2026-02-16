#!/bin/bash

# Hanging Job Monitor for SLURM (Messages get sent to Telegram)
# - Monitors SLURM job state transitions
# - Optionally monitors StdOut log mtime for "hang" detection (if log exists)
# - Sends Telegram notifications on "hang" (based on log mtime) or job state changes
# - Sends periodic log updates at configurable intervals (if log exists)

set -euo pipefail

# ============================================================================
# CONSTANTS AND DEFAULTS
# ============================================================================

readonly SCRIPT_NAME=$(basename "$0")
readonly SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly TELEGRAM_BOT="${SCRIPT_DIR}/telegram_bot.sh"
readonly DEFAULT_CHECK_INTERVAL=60
readonly DEFAULT_TIMEOUT=1800
readonly DEFAULT_UPDATE_INTERVAL=3600
readonly DEFAULT_UPDATE_LINES=5

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

HOSTNAME=$(hostname -s)
JOB_ID=""
JOB_NAME=""
TG_BOT_TOKEN="${TG_BOT_TOKEN:-}"
TG_CHAT_ID="${TG_CHAT_ID:-}"
CHECK_INTERVAL=$DEFAULT_CHECK_INTERVAL
TIMEOUT=$DEFAULT_TIMEOUT
UPDATE_INTERVAL=$DEFAULT_UPDATE_INTERVAL
UPDATE_LINES=$DEFAULT_UPDATE_LINES
GREP_PATTERN=""
INVERT_MATCH=false

# Derived values (calculated from user inputs)
TIMEOUT_CHECKS=0
UPDATE_CHECKS=0

# State tracking variables
PREV_STATE=""
PENDING_NOTIFIED=false
ALERT_SENT=false
NO_LOG_NOTIFIED=false
LOG_WAIT_NOTIFIED=false
INITIAL_NOTIFICATION_SENT=false

# Counter variables (drives all timing logic)
CHECK_COUNTER=0
LAST_PERIODIC_UPDATE_CHECK=0

# File paths and log availability
LOG_FILE=""
HAS_LOG_CONFIG=false  # Whether job has StdOut configured
LOG_EXISTS=false      # Whether log file actually exists on disk

# ============================================================================
# USAGE AND HELP
# ============================================================================

usage() {
    cat << EOF
Usage: $SCRIPT_NAME -j <job_id> [options]

Telegram credentials (in order of precedence):
    1. TG_BOT_TOKEN / TG_CHAT_ID environment variables
    2. Auto-sourced from ~/.tg_env if env vars are not set
    3. Flags: -b / -c (legacy, overrides env vars when both set)

    Recommended: save credentials to ~/.tg_env once, then just use -j:
        echo 'export TG_BOT_TOKEN="..."' >> ~/.tg_env
        echo 'export TG_CHAT_ID="..."'   >> ~/.tg_env
        $SCRIPT_NAME -j 123456

Options:
    -j, --job-id          SLURM job ID (required)
    -b, --bot-token       Telegram bot token (legacy flag, prefer TG_BOT_TOKEN env var)
    -c, --chat-id         Telegram chat ID (legacy flag, prefer TG_CHAT_ID env var)
    -i, --interval        Check interval in seconds (default: $DEFAULT_CHECK_INTERVAL)
    -t, --timeout         Timeout in seconds (default: $DEFAULT_TIMEOUT)
    -u, --update-interval Periodic log update interval in seconds (default: $DEFAULT_UPDATE_INTERVAL, 0 to disable)
    -l, --lines           Number of log lines to send in periodic updates (default: $DEFAULT_UPDATE_LINES)
    -g, --grep            Regex pattern to filter log lines (like grep)
    -v, --invert-match    Invert the grep pattern (like grep -v, discard matching lines)
    -h, --help            Show help

Example:
    $SCRIPT_NAME -j 123456
    $SCRIPT_NAME -j 123456 -t 1800 -u 3600 -l 20
    $SCRIPT_NAME -j 123456 -g "ERROR|WARNING"
    $SCRIPT_NAME -j 123456 -g "DEBUG|TRACE" -v
    $SCRIPT_NAME -b "123456:ABCdef..." -c "987654321" -j 123456  # legacy flag syntax

Signal Handling:
    The monitor gracefully handles these signals and sends a notification:
    - Ctrl+C (SIGINT)
    - kill <pid> (SIGTERM)
    - Ctrl+\ (SIGQUIT)
    - Terminal disconnect (SIGHUP)
    - tmux kill-session (SIGHUP)

    NOTE: 'kill -9 <pid>' (SIGKILL) cannot be caught - no notification will be sent.
          Use 'kill <pid>' (without -9) for graceful termination.

    Running in tmux protects against accidental SIGHUP from SSH disconnections.

Log Monitoring:
    - If job has no StdOut configured: Monitor continues, log features disabled
    - If job has StdOut but file doesn't exist yet: Monitor continues, waits for log
    - Hang detection and periodic updates only work when log file exists
EOF
    exit 1
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -j|--job-id) JOB_ID="$2"; shift 2 ;;
            -b|--bot-token) TG_BOT_TOKEN="$2"; shift 2 ;;
            -c|--chat-id) TG_CHAT_ID="$2"; shift 2 ;;
            -i|--interval) CHECK_INTERVAL="$2"; shift 2 ;;
            -t|--timeout) TIMEOUT="$2"; shift 2 ;;
            -u|--update-interval) UPDATE_INTERVAL="$2"; shift 2 ;;
            -l|--lines) UPDATE_LINES="$2"; shift 2 ;;
            -g|--grep) GREP_PATTERN="$2"; shift 2 ;;
            -v|--invert-match) INVERT_MATCH=true; shift ;;
            -h|--help) usage ;;
            *) echo "Error: Unknown option: $1"; usage ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$JOB_ID" ]]; then
        echo "Error: Missing required argument: -j/--job-id"
        usage
    fi

    # Auto-source ~/.tg_env if credentials are still missing after env + flags
    local tg_env_file="$HOME/.tg_env"
    if [[ (-z "$TG_BOT_TOKEN" || -z "$TG_CHAT_ID") && -f "$tg_env_file" ]]; then
        echo "Sourcing credentials from $tg_env_file"
        # shellcheck source=/dev/null
        source "$tg_env_file"
    fi

    if [[ -z "$TG_BOT_TOKEN" ]]; then
        echo "Error: TG_BOT_TOKEN not set. Add it to ~/.tg_env, export it, or use -b flag."
        usage
    fi
    if [[ -z "$TG_CHAT_ID" ]]; then
        echo "Error: TG_CHAT_ID not set. Add it to ~/.tg_env, export it, or use -c flag."
        usage
    fi

    # Convert seconds to checks (round up using ceiling division)
    TIMEOUT_CHECKS=$(( (TIMEOUT + CHECK_INTERVAL - 1) / CHECK_INTERVAL ))

    if [[ $UPDATE_INTERVAL -gt 0 ]]; then
        UPDATE_CHECKS=$(( (UPDATE_INTERVAL + CHECK_INTERVAL - 1) / CHECK_INTERVAL ))
    else
        UPDATE_CHECKS=0
    fi

    # If UPDATE_LINES <= 0, disable periodic updates
    if [[ $UPDATE_LINES -le 0 ]]; then
        UPDATE_CHECKS=0
    fi
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Get current timestamp
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Log message with timestamp
log_info() {
    echo "[$(timestamp)] $*"
}

# Format elapsed time into human-readable format
format_elapsed_time() {
    local seconds="$1"
    local days=$((seconds / 86400))
    local hours=$(((seconds % 86400) / 3600))
    local mins=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))

    if [[ $days -gt 0 ]]; then
        printf "%dd %dh %dm %ds" "$days" "$hours" "$mins" "$secs"
    elif [[ $hours -gt 0 ]]; then
        printf "%dh %dm %ds" "$hours" "$mins" "$secs"
    elif [[ $mins -gt 0 ]]; then
        printf "%dm %ds" "$mins" "$secs"
    else
        printf "%ds" "$secs"
    fi
}

# Get file modification time
get_mtime() {
    [[ ! -f "$1" ]] && echo "0" && return
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo "0"
}

# Sanitize text for embedding inside Markdown triple-backtick code blocks.
# Replaces backticks with single-quotes so they don't prematurely close the block.
sanitize_for_code_block() {
    tr '`' "'"
}

# ============================================================================
# SLURM JOB FUNCTIONS
# ============================================================================

# Get icon for SLURM state
state_icon() {
    local state="$1"
    case "$state" in
        PENDING)   echo "⏸️" ;;
        RUNNING)   echo "▶️" ;;
        COMPLETED) echo "✅" ;;
        FAILED)    echo "❌" ;;
        CANCELLED|CANCELED) echo "🛑" ;;
        TIMEOUT)   echo "⏱️" ;;
        PREEMPTED) echo "⚠️" ;;
        NODE_FAIL) echo "💥" ;;
        SUSPENDED) echo "⏸️" ;;
        NOTFOUND)  echo "❓" ;;
        UNKNOWN)   echo "❔" ;;
        *)         echo "🔄" ;;
    esac
}

# Get job information
# Returns: state|job_name|stdout_path
# If job not found, returns: NOTFOUND||
get_job_info() {
    local job_id="$1"
    local job_info
    job_info=$(scontrol show job "$job_id" 2>/dev/null || true)

    if [[ -z "$job_info" ]]; then
        echo "NOTFOUND||"
        return 0
    fi

    local state
    state=$(echo "$job_info" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p' | head -1)
    state="${state:-UNKNOWN}"

    local job_name
    job_name=$(echo "$job_info" | sed -n 's/.*JobName=\([^ ]*\).*/\1/p' | head -1)
    job_name="${job_name:-N/A}"

    local stdout_path
    stdout_path=$(echo "$job_info" | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' | head -1)

    # Replace %j with actual job ID if stdout_path exists
    if [[ -n "$stdout_path" ]]; then
        stdout_path="${stdout_path//%j/$job_id}"
    fi

    echo "${state}|${job_name}|${stdout_path}"
    return 0
}

# Check if state is terminal
is_terminal_state() {
    local state="$1"
    case "$state" in
        COMPLETED|FAILED|CANCELLED|CANCELED|TIMEOUT|NOTFOUND) return 0 ;;
        *) return 1 ;;
    esac
}

# ============================================================================
# LOG PROCESSING FUNCTIONS
# ============================================================================

# Filter log content using grep pattern
filter_log_lines() {
    local log_file="$1"
    local num_lines="$2"
    local grep_pattern="$3"
    local invert="$4"

    if [[ ! -f "$log_file" ]]; then
        echo "Unable to read log"
        return
    fi

    if [[ -z "$grep_pattern" ]]; then
        # No filter, just return last N lines
        tail -n "$num_lines" "$log_file" 2>/dev/null || echo "Unable to read log"
    else
        # Read more lines than needed to ensure we get enough after filtering
        local read_multiplier=10
        local lines_to_read=$((num_lines * read_multiplier))

        # Apply grep with or without invert
        local filtered_content
        if [[ "$invert" == "true" ]]; then
            filtered_content=$(tail -n "$lines_to_read" "$log_file" 2>/dev/null | grep -v -E "$grep_pattern" | tail -n "$num_lines" || echo "")
        else
            filtered_content=$(tail -n "$lines_to_read" "$log_file" 2>/dev/null | grep -E "$grep_pattern" | tail -n "$num_lines" || echo "")
        fi

        if [[ -z "$filtered_content" ]]; then
            if [[ "$invert" == "true" ]]; then
                echo "All lines filtered out by pattern (grep -v): $grep_pattern"
            else
                echo "No lines matching pattern (grep): $grep_pattern"
            fi
        else
            echo "$filtered_content"
        fi
    fi
}

# ============================================================================
# TELEGRAM FUNCTIONS
# ============================================================================

# Send message to Telegram (delegates to telegram_bot.sh)
send_telegram() {
    local message="$1"

    local output
    if output=$(TG_BOT_TOKEN="$TG_BOT_TOKEN" TG_CHAT_ID="$TG_CHAT_ID" "$TELEGRAM_BOT" send "$message" 2>&1); then
        log_info "✓ Message sent"
        return 0
    else
        log_info "✗ Message failed: $output"
        return 1
    fi
}

# Test Telegram connection
test_telegram_connection() {
    local job_id="$1"

    echo "Testing Telegram connection..."
    local test_message="✅ SLURM Job Monitor started \[${job_id}]"
    if send_telegram "$test_message"; then
        echo "✓ Telegram working"
        return 0
    else
        echo "✗ Telegram test failed"
        return 1
    fi
}

# ============================================================================
# TELEGRAM NOTIFICATION MESSAGES
# ============================================================================

send_initial_state_notification() {
    local job_id="$1"
    local state="$2"
    local icon
    icon=$(state_icon "$state")

    local log_info_msg=""
    if [[ "$HAS_LOG_CONFIG" == true ]]; then
        local timeout_seconds=$((CHECK_INTERVAL * TIMEOUT_CHECKS))
        local timeout_formatted=$(format_elapsed_time "$timeout_seconds")

        log_info_msg="
*Log File:* \`${LOG_FILE}\`
*Hang Timeout:* ${timeout_formatted}"

        if [[ $UPDATE_CHECKS -gt 0 ]]; then
            local update_seconds=$((CHECK_INTERVAL * UPDATE_CHECKS))
            local update_formatted=$(format_elapsed_time "$update_seconds")
            log_info_msg+="
*Update Interval:* ${update_formatted}"
        fi
    else
        log_info_msg="
*Log File:* None (log monitoring disabled)"
    fi

    local message="${icon} *SLURM Job Monitor Started* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Initial State:* ${state}
*Check Interval:* ${CHECK_INTERVAL}s${log_info_msg}
*Time:* $(timestamp)"

    send_telegram "$message"
}

send_pending_notification() {
    local job_id="$1"
    local message="⏸️ *SLURM Job Pending* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Status:* PENDING
*Time:* $(timestamp)

Job is queued and waiting for resources."
    send_telegram "$message"
}

send_state_change_notification() {
    local job_id="$1"
    local from_state="$2"
    local to_state="$3"

    local from_icon=$(state_icon "$from_state")
    local to_icon=$(state_icon "$to_state")

    local message="🔔 *SLURM Job State Changed* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*From:* ${from_icon} ${from_state}
*To:* ${to_icon} ${to_state}
*Time:* $(timestamp)"

    send_telegram "$message"
}

send_no_log_notification() {
    local job_id="$1"

    local message="ℹ️ *Log Monitoring Disabled* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Time:* $(timestamp)

Job has no StdOut configured. Monitoring will track job state only (no hang detection or log updates)."

    send_telegram "$message"
}

send_log_wait_notification() {
    local job_id="$1"
    local log_file="$2"

    local message="⏳ *Waiting for Log File* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Expected Log:* \`${log_file}\`
*Time:* $(timestamp)

Job is RUNNING but log file not created yet. Will enable log monitoring when file appears."

    send_telegram "$message"
}

send_log_appeared_notification() {
    local job_id="$1"
    local log_file="$2"

    local message="📄 *Log File Created* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Log File:* \`${log_file}\`
*Time:* $(timestamp)

Log file now exists. Hang detection and periodic updates enabled."

    send_telegram "$message"
}

send_hanging_alert() {
    local job_id="$1"
    local log_file="$2"
    local elapsed="$3"
    local state="$4"

    local formatted_elapsed=$(format_elapsed_time "$elapsed")
    local timeout_seconds=$((CHECK_INTERVAL * TIMEOUT_CHECKS))
    local formatted_timeout=$(format_elapsed_time "$timeout_seconds")

    # Always send last 10 lines without filter for hanging alerts
    local log_preview
    log_preview=$(tail -n 10 "$log_file" 2>/dev/null | head -c 400 | sanitize_for_code_block || echo "")

    # Preprocess log preview into formatted section
    local log_section
    if [[ -n "$log_preview" ]]; then
        log_section="*Last 10 lines:*
\`\`\`
${log_preview}
\`\`\`"
    else
        log_section="*Last 10 lines:* (log empty or unreadable)"
    fi

    local message="🚨 *SLURM Job Alert* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Status:* $(state_icon "$state") ${state}
*Log:* \`${log_file}\`
*Idle:* ${formatted_elapsed} (timeout: ${formatted_timeout})
*Time:* $(timestamp)

⚠️ Job running but log not updating

${log_section}"

    send_telegram "$message"
}

send_resumed_notification() {
    local job_id="$1"

    local message="▶️ *SLURM Job Resumed* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Status:* ▶️ RUNNING
*Time:* $(timestamp)

Job has resumed and log is being updated again."

    send_telegram "$message"
}

send_stopped_notification() {
    local job_id="$1"
    local job_state="$2"

    local icon=$(state_icon "$job_state")

    local message="${icon} *SLURM Job Finished* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Final State:* ${job_state}
*Time:* $(timestamp)

✅ Monitor stopped (job completed)"

    send_telegram "$message"
}

send_interruption_notification() {
    local signal="$1"
    local job_id="$2"
    local job_state="$3"

    local icon
    case "$signal" in
        SIGINT)  icon="🛑" ;;
        SIGTERM) icon="💥" ;;
        SIGQUIT) icon="🔥" ;;
        SIGHUP)  icon="🔌" ;;
        *)       icon="🛑" ;;
    esac

    local message="${icon} *SLURM Job Monitor Interrupted* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
*Signal:* ${signal}
*Job State:* $(state_icon "$job_state") ${job_state}
*Time:* $(timestamp)

⚠️ Monitor stopped by signal - job may still be running"

    send_telegram "$message"
}

send_periodic_log_update() {
    local job_id="$1"
    local log_file="$2"
    local num_lines="$3"
    local is_hanging="$4"
    local log_snapshot_content
    log_snapshot_content=$(printf '%s' "$5" | sanitize_for_code_block)
    local log_snapshot_mtime="$6"
    local current_state="$7"  # Add current state parameter

    # Calculate idle time using snapshot mtime
    local current_time=$(date +%s)
    local idle_info=""
    local status_icon=$(state_icon "$current_state")
    local status_line="*Status:* ${status_icon} ${current_state}"
    local message_icon="📋"

    if [[ "$log_snapshot_mtime" != "0" ]]; then
        local idle_time=$((current_time - log_snapshot_mtime))
        local formatted_idle=$(format_elapsed_time "$idle_time")
        local timeout_seconds=$((CHECK_INTERVAL * TIMEOUT_CHECKS))
        local formatted_timeout=$(format_elapsed_time "$timeout_seconds")

        # Use the is_hanging decision from hang detection
        if [[ "$is_hanging" == "true" ]]; then
            status_line="*Status:* ⚠️ HANGING (log not updating)"
            message_icon="🚨"
        fi

        idle_info="
*Log Idle:* ${formatted_idle} (timeout: ${formatted_timeout})"
    fi

    local filter_note=""
    if [[ -n "$GREP_PATTERN" ]]; then
        if [[ "$INVERT_MATCH" == "true" ]]; then
            filter_note="
*Filter (grep -v):* \`${GREP_PATTERN}\`"
        else
            filter_note="
*Filter (grep):* \`${GREP_PATTERN}\`"
        fi
    fi

    local message="${message_icon} *Periodic Log Update* \[${job_id}]

*Sender:* ${HOSTNAME}
*Job Name:* \`${JOB_NAME}\`
${status_line}
*Time:* $(timestamp)${idle_info}${filter_note}
*Lines:* Last ${num_lines}

\`\`\`
${log_snapshot_content}
\`\`\`"

    send_telegram "$message"
}

# ============================================================================
# SIGNAL HANDLING
# ============================================================================

# Signal handler function
cleanup_on_signal() {
    local signal="$1"

    log_info "Received signal: $signal"

    local job_info_result
    job_info_result=$(get_job_info "$JOB_ID")
    # Parse: state|job_name|stdout_path
    IFS='|' read -r current_state _ _ <<< "$job_info_result"

    # Send interruption notification
    send_interruption_notification "$signal" "$JOB_ID" "$current_state"

    log_info "Monitor interrupted by $signal"
    exit 130  # Standard exit code for SIGINT
}

# Setup signal traps
setup_signal_handlers() {
    # Trap common termination signals
    trap 'cleanup_on_signal SIGINT' SIGINT      # Ctrl+C
    trap 'cleanup_on_signal SIGTERM' SIGTERM    # kill (default)
    trap 'cleanup_on_signal SIGQUIT' SIGQUIT    # Ctrl+\
    trap 'cleanup_on_signal SIGHUP' SIGHUP      # Terminal hangup / tmux kill-session
}

# ============================================================================
# UNIFIED MONITORING LOGIC
# ============================================================================

# Single unified monitoring tick handler
handle_monitoring_tick() {
    local current_state="$1"

    # ========================================================================
    # 1. SEND INITIAL NOTIFICATION (first tick only)
    # ========================================================================

    if [[ "$INITIAL_NOTIFICATION_SENT" == false ]]; then
        send_initial_state_notification "$JOB_ID" "$current_state"
        INITIAL_NOTIFICATION_SENT=true

        # Send PENDING notification if applicable
        if [[ "$current_state" == "PENDING" ]]; then
            send_pending_notification "$JOB_ID"
            PENDING_NOTIFIED=true
        fi

        # Send no-log notification if no log configured
        if [[ "$HAS_LOG_CONFIG" == false ]]; then
            send_no_log_notification "$JOB_ID"
            NO_LOG_NOTIFIED=true
        fi
    fi

    # ========================================================================
    # 2. HANDLE STATE CHANGES
    # ========================================================================

    if [[ "$current_state" != "$PREV_STATE" ]]; then
        log_info "State change: $PREV_STATE -> $current_state"

        # Special notification for first PENDING state (if not already sent)
        if [[ "$current_state" == "PENDING" && "$PENDING_NOTIFIED" == false ]]; then
            send_pending_notification "$JOB_ID"
            PENDING_NOTIFIED=true
        fi

        # General state change notification (skip for first tick since we already sent initial notification)
        if [[ "$INITIAL_NOTIFICATION_SENT" == true && "$CHECK_COUNTER" -gt 1 ]]; then
            send_state_change_notification "$JOB_ID" "$PREV_STATE" "$current_state"
        fi

        PREV_STATE="$current_state"

        # If job just transitioned to RUNNING, check log status
        if [[ "$current_state" == "RUNNING" ]]; then
            if [[ "$HAS_LOG_CONFIG" == true && "$LOG_EXISTS" == false ]]; then
                # Job is running but log doesn't exist yet
                if [[ "$LOG_WAIT_NOTIFIED" == false ]]; then
                    send_log_wait_notification "$JOB_ID" "$LOG_FILE"
                    LOG_WAIT_NOTIFIED=true
                fi
            fi
        fi
    fi

    # ========================================================================
    # 3. CHECK IF LOG FILE HAS APPEARED
    # ========================================================================

    if [[ "$HAS_LOG_CONFIG" == true && "$LOG_EXISTS" == false ]]; then
        # Check if log file now exists
        if [[ -f "$LOG_FILE" ]]; then
            log_info "Log file appeared: $LOG_FILE"
            LOG_EXISTS=true
            send_log_appeared_notification "$JOB_ID" "$LOG_FILE"

            # Reset counter to trigger immediate periodic update on next tick
            LAST_PERIODIC_UPDATE_CHECK=$((CHECK_COUNTER - UPDATE_CHECKS))
        else
            # Log file still doesn't exist
            if [[ "$current_state" == "RUNNING" ]]; then
                log_info "Waiting for log file (job is RUNNING)"
            else
                log_info "Waiting for log file (job state: $current_state)"
            fi
            # Don't return - allow terminal state handling below
        fi
    fi

    # ========================================================================
    # 4. HANDLE TERMINAL STATES (with final periodic update if applicable)
    # ========================================================================

    if is_terminal_state "$current_state"; then
        # If we have a log and periodic updates are configured, send one final update
        if [[ "$HAS_LOG_CONFIG" == true && "$LOG_EXISTS" == true && $UPDATE_CHECKS -gt 0 ]]; then
            log_info "Sending final log update before exit"
            local log_snapshot_content
            log_snapshot_content=$(filter_log_lines "$LOG_FILE" "$UPDATE_LINES" "$GREP_PATTERN" "$INVERT_MATCH" | head -c 3800)
            local log_snapshot_mtime
            log_snapshot_mtime=$(get_mtime "$LOG_FILE")

            # Terminal state jobs are not hanging
            send_periodic_log_update "$JOB_ID" "$LOG_FILE" "$UPDATE_LINES" "false" "$log_snapshot_content" "$log_snapshot_mtime" "$current_state"
        fi

        # Now send stopped notification and exit
        log_info "Job in terminal state: $current_state"
        send_stopped_notification "$JOB_ID" "$current_state"
        log_info "Monitor stopped (job finished)"
        exit 0
    fi

    # ========================================================================
    # 5. LOG-BASED MONITORING (ONLY IF LOG EXISTS AND JOB NOT TERMINAL)
    # ========================================================================

    # Skip log-based features if no log configured or log doesn't exist
    if [[ "$HAS_LOG_CONFIG" == false || "$LOG_EXISTS" == false ]]; then
        log_info "Job state: $current_state (no log monitoring)"
        return
    fi

    # ========================================================================
    # 5A. DETERMINE IF PERIODIC UPDATE IS DUE
    # ========================================================================

    local periodic_update_due=false
    if [[ $UPDATE_CHECKS -gt 0 && "$current_state" == "RUNNING" ]]; then
        local checks_since_update=$((CHECK_COUNTER - LAST_PERIODIC_UPDATE_CHECK))
        if [[ $checks_since_update -ge $UPDATE_CHECKS ]]; then
            periodic_update_due=true
        fi
    fi

    # ========================================================================
    # 5B. CAPTURE LOG SNAPSHOT (ATOMICALLY: content + mtime)
    # ========================================================================
    # To avoid race conditions, we capture log content and mtime together.
    # This ensures consistency between hang detection and periodic updates.
    # Strategy: Read content FIRST, then immediately get mtime.
    # This way, the mtime is >= the time when content was read.

    local log_snapshot_content=""
    local log_snapshot_mtime=0

    # Capture log content if periodic update is due
    if [[ "$periodic_update_due" == true ]]; then
        log_snapshot_content=$(filter_log_lines "$LOG_FILE" "$UPDATE_LINES" "$GREP_PATTERN" "$INVERT_MATCH" | head -c 3800)
    fi
    # IMMEDIATELY get mtime after reading content (or just get it for hang detection)
    log_snapshot_mtime=$(get_mtime "$LOG_FILE")

    # ========================================================================
    # 5C. HANG DETECTION (only for RUNNING jobs)
    # ========================================================================

    local is_hanging=false

    if [[ "$current_state" != "RUNNING" ]]; then
        log_info "Job state: $current_state (hang detection paused)"
    else
        # Job is RUNNING, perform hang detection using the mtime we just captured
        if [[ "$log_snapshot_mtime" == "0" ]]; then
            log_info "Warning: Cannot read log file"
        else
            local current_time=$(date +%s)
            local elapsed=$((current_time - log_snapshot_mtime))
            local timeout_seconds=$((CHECK_INTERVAL * TIMEOUT_CHECKS))
            local formatted_elapsed=$(format_elapsed_time "$elapsed")
            local formatted_timeout=$(format_elapsed_time "$timeout_seconds")

            log_info "Log idle: ${formatted_elapsed} (timeout: ${formatted_timeout})"

            if [[ $elapsed -ge $timeout_seconds ]]; then
                # Log has been idle too long - JOB IS HANGING
                is_hanging=true

                if [[ "$ALERT_SENT" == false ]]; then
                    log_info "Timeout exceeded. Job state: $current_state"
                    echo ""
                    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    echo "!!! ALERT: Job ${JOB_ID} hanging !!!"
                    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    echo ""
                    send_hanging_alert "$JOB_ID" "$LOG_FILE" "$elapsed" "$current_state"
                    ALERT_SENT=true
                else
                    log_info "Still hanging (alert already sent)"
                fi
            else
                # Log is being updated - JOB IS NOT HANGING
                is_hanging=false

                if [[ "$ALERT_SENT" == true ]]; then
                    log_info "✓ Job resumed!"
                    send_resumed_notification "$JOB_ID"
                    ALERT_SENT=false
                fi
            fi
        fi
    fi

    # ========================================================================
    # 5D. PERIODIC LOG UPDATE (with snapshot and hanging state)
    # ========================================================================

    if [[ "$periodic_update_due" == true ]]; then
        log_info "Sending periodic log update (is_hanging=$is_hanging)"
        send_periodic_log_update "$JOB_ID" "$LOG_FILE" "$UPDATE_LINES" "$is_hanging" "$log_snapshot_content" "$log_snapshot_mtime" "$current_state"
        LAST_PERIODIC_UPDATE_CHECK=$CHECK_COUNTER
    fi
}

# ============================================================================
# UNIFIED COMMAND EXECUTION LOGIC
# ============================================================================

check_and_execute_remote_commands() {
    # Algorithm:
    # 1. Call Telegram getUpdates API with long polling (timeout=CHECK_INTERVAL)
    #    - offset=LAST_UPDATE_ID+1, allowed_updates=["message"]
    # 2. Parse response (grep for update_id and text fields)
    # 3. For each command: log it, call handle_command(cmd), update LAST_UPDATE_ID
    # 4. Return (continue to monitoring tick)

    # Global state needed: LAST_UPDATE_ID=0

    # Possible commands to support:
    # /status             - Show job state and log idle time
    # /log                - Send periodic_log_update with UPDATE_LINES
    # /log10|20|50|100    - Send N log lines
    # /cancelcancelcancel - scancel $JOB_ID
    # /help               - List available commands

    return  # Not implemented yet
}

# ============================================================================
# MAIN TICK LOOP
# ============================================================================

main_tick_loop() {
    while true; do
        local tick_start=$(date +%s)

        CHECK_COUNTER=$((CHECK_COUNTER + 1))

        check_and_execute_remote_commands

        local job_info_result
        job_info_result=$(get_job_info "$JOB_ID")
        # Parse: state|job_name|stdout_path
        IFS='|' read -r current_state _ _ <<< "$job_info_result"

        # Single unified handler for all monitoring logic
        handle_monitoring_tick "$current_state"

        # Dynamic sleep to maintain CHECK_INTERVAL rhythm
        local tick_end=$(date +%s)
        local elapsed=$((tick_end - tick_start))
        local remaining=$((CHECK_INTERVAL - elapsed))
        if [[ $remaining -gt 0 ]]; then
            sleep "$remaining"
        fi
    done
}

# ============================================================================
# INITIALIZATION
# ============================================================================

initialize_monitoring() {
    # Setup signal handlers first
    setup_signal_handlers

    # Print header
    echo "========================================"
    echo "SLURM Job Monitor (Telegram)"
    echo "========================================"
    echo "Job ID: $JOB_ID"
    echo "Monitor PID: $$"
    echo ""
    echo "⚠️  Signal Handling:"
    echo "   • Ctrl+C, kill $$, tmux kill-session → Sends notification ✓"
    echo "   • kill -9 $$ → NO notification (cannot be caught) ✗"
    echo "   • Running in tmux protects from SSH disconnect"
    echo "========================================"
    echo ""

    local job_info_result
    job_info_result=$(get_job_info "$JOB_ID")

    # Parse: state|job_name|stdout_path
    IFS='|' read -r PREV_STATE JOB_NAME LOG_FILE <<< "$job_info_result"

    # Check if job was found
    if [[ "$PREV_STATE" == "NOTFOUND" ]]; then
        echo "Error: Cannot find job $JOB_ID"
        exit 1
    fi

    # Determine if job has log configuration
    if [[ -z "$LOG_FILE" ]]; then
        HAS_LOG_CONFIG=false
        LOG_EXISTS=false
        echo "Job Name: $JOB_NAME"
        echo "Log: None (job has no StdOut configured)"
        echo "Log Monitoring: Disabled"
    else
        HAS_LOG_CONFIG=true
        echo "Job Name: $JOB_NAME"
        echo "Log: $LOG_FILE"

        # Check if log file already exists
        if [[ -f "$LOG_FILE" ]]; then
            LOG_EXISTS=true
            echo "Log Status: File exists"
        else
            LOG_EXISTS=false
            echo "Log Status: Waiting for file creation"
        fi

        # Calculate and display monitoring parameters
        local timeout_seconds=$((CHECK_INTERVAL * TIMEOUT_CHECKS))
        local timeout_formatted=$(format_elapsed_time "$timeout_seconds")
        echo "Hang Timeout: ${timeout_formatted} (${TIMEOUT_CHECKS} checks)"

        if [[ $UPDATE_CHECKS -gt 0 ]]; then
            local update_seconds=$((CHECK_INTERVAL * UPDATE_CHECKS))
            local update_formatted=$(format_elapsed_time "$update_seconds")
            echo "Periodic Updates: ${update_formatted} (${UPDATE_CHECKS} checks, ${UPDATE_LINES} lines)"
        else
            echo "Periodic Updates: Disabled"
        fi

        if [[ -n "$GREP_PATTERN" ]]; then
            if [[ "$INVERT_MATCH" == "true" ]]; then
                echo "Grep Filter (grep -v): $GREP_PATTERN"
            else
                echo "Grep Filter (grep): $GREP_PATTERN"
            fi
        fi
    fi

    echo "Check Interval: ${CHECK_INTERVAL}s"
    echo "========================================"
    echo ""

    # Test Telegram connection
    if ! test_telegram_connection "$JOB_ID"; then
        exit 1
    fi
    echo ""

    echo "Initial job state: $PREV_STATE"

    # Initialize counters
    CHECK_COUNTER=0
    # Set LAST_PERIODIC_UPDATE_CHECK to trigger immediate update on first tick if applicable
    if [[ $UPDATE_CHECKS -gt 0 ]]; then
        LAST_PERIODIC_UPDATE_CHECK=$((-UPDATE_CHECKS))
    else
        LAST_PERIODIC_UPDATE_CHECK=0
    fi

    # Note: All notifications are now handled in handle_monitoring_tick
    # This ensures consistent behavior whether job starts terminal or transitions to it

    echo "Monitoring started..."
}

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

main() {
    parse_arguments "$@"
    initialize_monitoring
    main_tick_loop
}

# Run main function with all arguments
main "$@"
