#!/usr/bin/env bash
#
# host_cmd_ctl.sh — manage the host-cmd server and policy.
# Run this ON THE HOST, not inside the container.
#
# Usage:
#   ./host_cmd_ctl.sh start
#   ./host_cmd_ctl.sh stop
#   ./host_cmd_ctl.sh restart
#   ./host_cmd_ctl.sh status
#   ./host_cmd_ctl.sh history            # list recent command results
#   ./host_cmd_ctl.sh history 10         # last 10 results
#   ./host_cmd_ctl.sh cleanup            # delete all results
#   ./host_cmd_ctl.sh cleanup 24         # delete results older than 24h
#   ./host_cmd_ctl.sh policy             # show current policy
#   ./host_cmd_ctl.sh deny  PATTERN      # add a deny pattern (restarts server)
#   ./host_cmd_ctl.sh allow PATTERN      # add an allow pattern (restarts server)
#   ./host_cmd_ctl.sh undeny  PATTERN    # remove a deny pattern (restarts server)
#   ./host_cmd_ctl.sh unallow PATTERN    # remove an allow pattern (restarts server)
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$DIR/daemon.pid"
LOG_FILE="$DIR/host_cmd_server.log"
SERVER="$DIR/host_cmd_server.py"
CLIENT="$DIR/host_cmd.py"

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

cmd_start() {
    if is_running; then
        echo "Already running (PID $(cat "$PID_FILE"))"
        return 0
    fi
    echo "Starting host-cmd server..."
    nohup python3 "$SERVER" > /dev/null 2>&1 &
    sleep 1
    if is_running; then
        echo "Started (PID $(cat "$PID_FILE")), log: $LOG_FILE"
    else
        echo "Failed to start. Check $LOG_FILE"
        return 1
    fi
}

cmd_stop() {
    if ! is_running; then
        echo "Not running"
        return 0
    fi
    local pid
    pid=$(cat "$PID_FILE")
    echo "Stopping (PID $pid)..."
    kill "$pid"
    for i in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Stopped"
            return 0
        fi
        sleep 0.5
    done
    echo "Force killing..."
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "Stopped"
}

cmd_status() {
    if is_running; then
        echo "Running (PID $(cat "$PID_FILE"))"
    else
        echo "Not running"
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
        return 1
    fi
}

cmd_history() {
    local limit="${1:-20}"
    local results_dir="$DIR/results"
    if [ ! -d "$results_dir" ] || [ -z "$(ls "$results_dir"/*.json 2>/dev/null)" ]; then
        echo "No results"
        return
    fi
    # Sort by mtime, newest first
    ls -t "$results_dir"/*.json | head -n "$limit" | while read -r f; do
        local ts cmd exit_code
        ts=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('started_at','?'))" 2>/dev/null)
        cmd=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('command','?')[:60])" 2>/dev/null)
        exit_code=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('exit_code','?'))" 2>/dev/null)
        local date_str
        date_str=$(python3 -c "import time; print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime($ts)))" 2>/dev/null || echo "?")
        printf "%-20s  exit=%-4s  %s\n" "$date_str" "$exit_code" "$cmd"
    done
}

cmd_cleanup() {
    python3 "$CLIENT" --cleanup "$@"
}

cmd_policy() {
    python3 "$CLIENT" --policy
}

cmd_deny() {
    [ -z "${1:-}" ] && { echo "Usage: $0 deny PATTERN"; exit 1; }
    python3 "$CLIENT" --deny "$1"
    echo "Restarting server to apply..."
    cmd_stop
    cmd_start
}

cmd_allow() {
    [ -z "${1:-}" ] && { echo "Usage: $0 allow PATTERN"; exit 1; }
    python3 "$CLIENT" --allow "$1"
    echo "Restarting server to apply..."
    cmd_stop
    cmd_start
}

cmd_undeny() {
    [ -z "${1:-}" ] && { echo "Usage: $0 undeny PATTERN"; exit 1; }
    python3 "$CLIENT" --undeny "$1"
    echo "Restarting server to apply..."
    cmd_stop
    cmd_start
}

cmd_unallow() {
    [ -z "${1:-}" ] && { echo "Usage: $0 unallow PATTERN"; exit 1; }
    python3 "$CLIENT" --unallow "$1"
    echo "Restarting server to apply..."
    cmd_stop
    cmd_start
}

case "${1:-}" in
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    restart)  cmd_stop; cmd_start ;;
    status)   cmd_status ;;
    history)  cmd_history "${2:-20}" ;;
    cleanup)  shift; cmd_cleanup "$@" ;;
    policy)   cmd_policy ;;
    deny)     cmd_deny "${2:-}" ;;
    allow)    cmd_allow "${2:-}" ;;
    undeny)   cmd_undeny "${2:-}" ;;
    unallow)  cmd_unallow "${2:-}" ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|history|cleanup|policy|deny|allow|undeny|unallow}"
        exit 1
        ;;
esac
