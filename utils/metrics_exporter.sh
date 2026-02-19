#!/usr/bin/env bash
# metrics_exporter.sh — Lightweight Prometheus metrics exporter with plugin discovery.
#
# Discovers *_metrics_plugin.sh scripts in the same directory, runs them every
# $POLL_INTERVAL seconds, concatenates their Prometheus-text output, and
# serves it over HTTP.
#
# Plugin contract:
#   - Filename matches *_metrics_plugin.sh in the same directory as this script.
#   - Receives the short hostname as $1.
#   - Outputs Prometheus exposition text to stdout.
#   - Logs errors/debug to stderr (captured to /tmp/metrics_exporter.log).
#   - (Optional) State persistence: a plugin may print "# STATE <value>" as
#     its LAST output line.  The exporter stores the value in memory and
#     exports _PLUGIN_STATE=<value> on the next invocation.  The STATE line
#     is a valid Prometheus comment (ignored by scrapers), so it passes
#     through harmlessly.  This lets short-lived plugins persist opaque
#     state across poll cycles without temp files.  State values are fully
#     opaque — the exporter never inspects or interprets them.
#   - Permanent skip (exit code 99):  A plugin that exits with code 99
#     tells the exporter to NEVER invoke it again for the lifetime of this
#     exporter process.  Use this for permanent exits: non-rank-0 nodes,
#     disabled features, etc.  The skip signal is an exit code — completely
#     separate from stdout and the state protocol — so normal plugins
#     (which exit 0) can never accidentally trigger it.  Resets on
#     exporter restart (new job).
#   - (Optional) Metric caching / idle replay:  When a state-using plugin
#     outputs ONLY a STATE line (no metric data), the exporter replays
#     the last cached metric output for that plugin.  This prevents
#     Prometheus staleness without requiring the plugin to re-do any
#     expensive work (file reads, parsing).  Replayed metrics keep their
#     original timestamps so Prometheus deduplicates identical samples;
#     the series stays present in the scrape, preventing staleness for
#     up to 5 minutes of idle.  Cache is populated whenever
#     the plugin produces metric lines and invalidated when:
#       • output is empty (plugin crash)
#       • STATE starts with __NONE__ (data source gone, e.g. file deleted)
#     Plugins that never use STATE are unaffected — they always produce
#     metrics, so replay never triggers.
#
# Normal (non-state) plugins just output Prometheus text and exit 0.
# They are completely unaffected by the state, skip, and caching mechanisms.
#
# Usage:
#   utils/metrics_exporter.sh [--port PORT] [--interval SECONDS]
#   EXPORTER_PORT=9400 utils/metrics_exporter.sh &

# NOTE: intentionally no "set -e" — plugins may fail partially and we want
# to keep running with whatever data we can collect.

EXPORTER_PORT="${EXPORTER_PORT:-9400}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
export POLL_INTERVAL  # plugins use this to estimate drip capacity
METRICS_FILE="/tmp/node_hw_metrics.prom"
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|-p)     EXPORTER_PORT="$2"; shift 2 ;;
        --interval|-i) POLL_INTERVAL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--port PORT] [--interval SECONDS]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Plugin state (persistent across poll cycles, keyed by plugin basename)
# ---------------------------------------------------------------------------
declare -A _PLUGIN_STATES  # opaque state values (never interpreted)
declare -A _PLUGIN_SKIP    # plugins that exited 99 (never invoke again)
declare -A _PLUGIN_METRICS # last metric output per plugin (for replay)

# ---------------------------------------------------------------------------
# Plugin discovery & metric collection
# ---------------------------------------------------------------------------
collect_metrics() {
    local start_ns end_ns duration_ms
    start_ns=$(date +%s%N 2>/dev/null || echo 0)

    local tmp
    tmp=$(mktemp /tmp/exporter_XXXXXX.prom 2>/dev/null) || tmp="/tmp/exporter_tmp.prom"

    for plugin in "$SCRIPT_DIR"/*_metrics_plugin.sh; do
        [[ -f "$plugin" ]] || continue
        local plugin_name raw_output plugin_rc
        plugin_name="$(basename "$plugin" .sh)"

        # Permanently skipped plugin — don't invoke at all.
        [[ "${_PLUGIN_SKIP[$plugin_name]:-}" == "1" ]] && continue

        # Export any saved state from previous cycle for this plugin.
        # Note: _PLUGIN_STATES (plural) is the internal associative array;
        # _PLUGIN_STATE (singular) is the scalar exported to child processes.
        export _PLUGIN_STATE="${_PLUGIN_STATES[$plugin_name]:-}"

        # Run plugin — capture output and exit code.
        # No "set -e", so non-zero exit codes don't abort the exporter.
        raw_output=$(bash "$plugin" "$HOSTNAME_SHORT" 2>>/tmp/metrics_exporter.log)
        plugin_rc=$?

        # Exit code 99: permanent skip — never invoke this plugin again.
        # This is a separate signal from stdout / state, so normal plugins
        # (which exit 0 or non-zero on error) can never trigger it.
        if [[ $plugin_rc -eq 99 ]]; then
            _PLUGIN_SKIP[$plugin_name]=1
            echo "[metrics_exporter] ${plugin_name} exited 99 — permanently skipped" \
                >>/tmp/metrics_exporter.log
        fi

        # Capture opaque state from last line (fully opaque — never interpreted).
        local last_line="${raw_output##*$'\n'}"
        if [[ "$last_line" == "# STATE "* ]]; then
            _PLUGIN_STATES[$plugin_name]="${last_line#\# STATE }"
        fi

        # Metric caching: cache only the metric portion (without STATE)
        # for clean replay.  STATE is already captured in _PLUGIN_STATES
        # above; including it in the cache would leave a stale STATE
        # comment in the prom file on replay.
        #
        # Replay is safe ONLY when the plugin explicitly signals "I'm
        # alive but have nothing new" — i.e. it outputs a STATE line
        # with no metric data.  Replay is suppressed when:
        #   • output is empty (plugin crash / failure)
        #   • STATE signals a reset (e.g. __NONE__ — the underlying
        #     data source is gone; cached metrics are stale)
        local metric_body=""
        if [[ -n "$raw_output" ]]; then
            if [[ "$last_line" == "# STATE "* && "$raw_output" == *$'\n'* ]]; then
                # Multi-line output ending with STATE — strip it.
                metric_body="${raw_output%$'\n'*}"
            elif [[ "$last_line" != "# STATE "* ]]; then
                # No STATE line at all — entire output is metrics.
                metric_body="$raw_output"
            fi
            # else: single-line STATE only — metric_body stays empty.
        fi

        # Invalidate cache when plugin signals a reset state.
        # __NONE__ means "data source gone" — stale cached metrics must
        # not be replayed.  This is the only STATE value the exporter
        # peeks at; all other values remain fully opaque.
        if [[ "$last_line" == "# STATE __NONE__"* ]]; then
            unset '_PLUGIN_METRICS['"$plugin_name"']'
        fi

        if [[ -n "$metric_body" ]] && \
           printf '%s' "$metric_body" | grep -qvE '^(#|$)'; then
            # Has actual metric data — cache (sans STATE) and write full output.
            _PLUGIN_METRICS[$plugin_name]="$metric_body"
            printf '%s\n' "$raw_output" >> "$tmp"
        elif [[ -n "$raw_output" ]] && \
             [[ "$last_line" == "# STATE "* ]] && \
             [[ -n "${_PLUGIN_METRICS[$plugin_name]:-}" ]]; then
            # Plugin is alive (produced STATE) but has no new metrics,
            # and the cache wasn't invalidated — replay cached metrics.
            # Replayed lines keep their timestamps so Prometheus
            # deduplicates identical samples (same ts+value = no new
            # storage).  Plugins that need to survive long idle periods
            # (e.g. checkpoint saves) handle their own anti-staleness
            # fills — the exporter just replays whatever they last emitted.
            printf '%s\n' "${_PLUGIN_METRICS[$plugin_name]}" >> "$tmp"
            printf '%s\n' "$last_line" >> "$tmp"
        else
            # No cache, no metrics, plugin failure, or cache invalidated.
            # Write whatever we got (STATE-only, empty, etc).
            [[ -n "$raw_output" ]] && printf '%s\n' "$raw_output" >> "$tmp"
        fi
    done

    # Scrape metadata — use nanosecond timestamps for sub-second precision
    end_ns=$(date +%s%N 2>/dev/null || echo 0)
    if [[ "$start_ns" != "0" && "$end_ns" != "0" ]]; then
        duration_ms=$(( (end_ns - start_ns) / 1000000 ))
    else
        duration_ms=0
    fi

    # Format as seconds with millisecond precision using pure bash (no python3 subprocess)
    local duration_sec
    duration_sec="$(( duration_ms / 1000 )).$(printf '%03d' "$(( duration_ms % 1000 ))")"

    cat >> "$tmp" <<EOF
# HELP hw_scrape_duration_seconds Time taken for the last poll cycle.
# TYPE hw_scrape_duration_seconds gauge
hw_scrape_duration_seconds{host="${HOSTNAME_SHORT}"} ${duration_sec}
EOF

    mv -f "$tmp" "$METRICS_FILE" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# HTTP server — Python http.server
# ---------------------------------------------------------------------------
start_http_server() {
    python3 -c "
import http.server, sys

_metrics_file = sys.argv[1]
_exporter_port = int(sys.argv[2])

class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        try:
            with open(_metrics_file, 'rb') as f:
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.wfile.write(b'# no data yet\n')

    def log_message(self, *args):
        pass  # suppress per-request logs

server = http.server.HTTPServer(('', _exporter_port), MetricsHandler)
server.serve_forever()
" "$METRICS_FILE" "$EXPORTER_PORT" &
    HTTP_PID=$!
    echo "[metrics_exporter] HTTP server PID=$HTTP_PID on port $EXPORTER_PORT"

    # Kill the HTTP server when the parent bash process exits, so it
    # doesn't get orphaned (reparented to PID 1) and hold the port.
    trap "kill $HTTP_PID 2>/dev/null; wait $HTTP_PID 2>/dev/null" EXIT INT TERM
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "[metrics_exporter] Starting on port ${EXPORTER_PORT} (poll every ${POLL_INTERVAL}s) on ${HOSTNAME_SHORT}"
    echo "[metrics_exporter] Plugin dir: ${SCRIPT_DIR}"

    for plugin in "$SCRIPT_DIR"/*_metrics_plugin.sh; do
        [[ -f "$plugin" ]] && echo "[metrics_exporter]   plugin: $(basename "$plugin")"
    done

    # Seed metrics file so the HTTP server has something to serve immediately
    echo "# exporter initializing" > "$METRICS_FILE"

    # Start HTTP server FIRST so Prometheus can connect while we collect
    start_http_server

    # Initial collection
    collect_metrics

    # Poll loop
    while true; do
        sleep "$POLL_INTERVAL"
        collect_metrics
    done
}

main "$@"
