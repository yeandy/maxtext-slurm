#!/bin/bash
# Prometheus helpers — install, start (live scraping), and view (post-hoc).
#
# As a library (sourced by ray_cluster.sh):
#   source utils/prometheus.sh
#   install_prometheus
#   start_prometheus          # live scrape during a job
#
# As a standalone script:
#   utils/prometheus.sh view <data-dir> [-p PORT]
#   utils/prometheus.sh install
#   utils/prometheus.sh list [<workspace>]

PROMETHEUS_PORT=${PROMETHEUS_PORT:-9190}
RAY_METRICS_PORT=${RAY_METRICS_PORT:-55080}
EXPORTER_PORT=${EXPORTER_PORT:-9400}

_PROM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================================
# Core functions (used both when sourced and when called directly)
# ============================================================================

install_prometheus() {
    command -v prometheus &>/dev/null && return 0
    [[ -x /tmp/prometheus/prometheus ]] && return 0

    echo "[Prometheus] Downloading..."
    local version="2.48.1"
    local arch=$(uname -m)
    [[ "$arch" == "x86_64" ]] && arch="amd64"
    [[ "$arch" == "aarch64" ]] && arch="arm64"

    mkdir -p /tmp/prometheus
    curl -sL "https://github.com/prometheus/prometheus/releases/download/v${version}/prometheus-${version}.linux-${arch}.tar.gz" \
        | tar -xz -C /tmp/prometheus --strip-components=1 2>/dev/null || {
        echo "[Prometheus] Download failed" >&2
        return 1
    }
    echo "[Prometheus] Installed to /tmp/prometheus"
}

# Resolve the prometheus binary path.
_prom_bin() {
    if command -v prometheus &>/dev/null; then
        echo "prometheus"
    elif [[ -x /tmp/prometheus/prometheus ]]; then
        echo "/tmp/prometheus/prometheus"
    else
        return 1
    fi
}

# Start Prometheus with live scraping (called inside a running job).
start_prometheus() {
    install_prometheus || return 0

    local prom_bin
    prom_bin=$(_prom_bin) || return 1

    # Build targets from pre-expanded node list (expanded on host before Docker)
    local ray_targets="'localhost:${RAY_METRICS_PORT}'"
    local hw_targets="'localhost:${EXPORTER_PORT}'"
    if [[ -n "${NODELIST_EXPANDED:-}" ]]; then
        ray_targets=""
        hw_targets=""
        IFS=',' read -ra nodes <<< "$NODELIST_EXPANDED"
        for node in "${nodes[@]}"; do
            [[ -n "$ray_targets" ]] && ray_targets+=", "
            ray_targets+="'${node}:${RAY_METRICS_PORT}'"
            [[ -n "$hw_targets" ]] && hw_targets+=", "
            hw_targets+="'${node}:${EXPORTER_PORT}'"
        done
    fi

    cat > /tmp/ray_prometheus.yml <<EOF
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'ray'
    static_configs:
      - targets: [${ray_targets}]
  - job_name: 'node_hw'
    scrape_interval: 10s
    scrape_timeout: 8s
    static_configs:
      - targets: [${hw_targets}]
EOF

    # Write TSDB directly to persistent storage.  Inside Docker, /outputs is
    # mounted from $JOB_WORKSPACE (or $SCRIPT_DIR/outputs) on the host.  This
    # way metrics survive even if the job is killed abruptly (preemption, OOM, etc.).
    PROMETHEUS_DATA_DIR="/outputs/${JOB_DIR:-unknown}/prometheus"
    mkdir -p "$PROMETHEUS_DATA_DIR"

    local prom_log
    prom_log=$(mktemp /tmp/prom_start_XXXXXX.log)

    local actual_port="$PROMETHEUS_PORT"
    local max_retries=5
    local prom_pid
    for ((attempt = 0; attempt <= max_retries; attempt++)); do
        "$prom_bin" --config.file=/tmp/ray_prometheus.yml \
            --web.listen-address=":${actual_port}" \
            --storage.tsdb.path="$PROMETHEUS_DATA_DIR" \
            --storage.tsdb.retention.time=30d >"$prom_log" 2>&1 &
        prom_pid=$!
        # Prometheus exits immediately on port-bind failure; wait then check.
        sleep 2
        if kill -0 "$prom_pid" 2>/dev/null; then
            break
        fi
        # Process exited — port likely occupied
        if [[ $attempt -lt $max_retries ]]; then
            actual_port=$((actual_port + 1))
            echo "[Prometheus] Port $((actual_port - 1)) occupied, trying ${actual_port}..."
        else
            echo "[Prometheus] FAILED: could not bind any port in range ${PROMETHEUS_PORT}-${actual_port}" >&2
            cat "$prom_log" >&2
            rm -f "$prom_log"
            # Clean up the TSDB directory so an empty dir doesn't mislead diagnosis tools.
            rm -rf "$PROMETHEUS_DATA_DIR"
            return 1
        fi
    done
    rm -f "$prom_log"

    if [[ "$actual_port" != "$PROMETHEUS_PORT" ]]; then
        echo "[Prometheus] WARNING: port ${PROMETHEUS_PORT} was occupied; using ${actual_port} instead"
        PROMETHEUS_PORT="$actual_port"
    fi
    echo "[Prometheus] Started on port ${actual_port} (pid ${prom_pid}; ray: ${ray_targets}; node_hw: ${hw_targets})"
    echo "[Prometheus] TSDB -> ${PROMETHEUS_DATA_DIR} (persistent)"
}

# Start a read-only Prometheus against persisted TSDB data.
view_prometheus() {
    local data_dir=""
    local port="$PROMETHEUS_PORT"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--port) port="$2"; shift 2 ;;
            -h|--help) _usage_view; return 0 ;;
            -*) echo "Unknown option: $1" >&2; _usage_view; return 1 ;;
            *)  data_dir="$1"; shift ;;
        esac
    done

    if [[ -z "$data_dir" ]]; then
        echo "Error: data directory is required" >&2
        _usage_view
        return 1
    fi

    if [[ ! -d "$data_dir" ]]; then
        echo "[Prometheus] Directory not found: $data_dir" >&2
        return 1
    fi

    install_prometheus || return 1

    local prom_bin
    prom_bin=$(_prom_bin) || return 1

    # Minimal config — no scraping, just serve the existing data.
    local cfg
    cfg=$(mktemp /tmp/prom_view_XXXXXX.yml)
    cat > "$cfg" <<'YAML'
global:
  scrape_interval: 999h
scrape_configs: []
YAML

    source "$(dirname "${BASH_SOURCE[0]}")/detect_ip.sh"
    local host_hostname
    host_hostname="$(whoami)@$(hostname -f 2>/dev/null || hostname)"
    local host_ip
    host_ip="$(whoami)@$(detect_ip)"

    echo "[Prometheus] Serving metrics from: $data_dir"
    echo "[Prometheus] Open http://localhost:${port}  (Ctrl-C to stop)"
    echo "[Prometheus] If running on a remote host, tunnel first (use hostname or IP, whichever works):"
    echo "  ssh -L ${port}:localhost:${port} ${host_hostname}"
    echo "  ssh -L ${port}:localhost:${port} ${host_ip}"
    "$prom_bin" --config.file="$cfg" \
        --web.listen-address=":${port}" \
        --storage.tsdb.path="$data_dir" \
        --storage.tsdb.retention.time=10y \
        --query.lookback-delta=15m
    rm -f "$cfg" 2>/dev/null
}

# List job directories that contain Prometheus data.
list_prometheus() {
    local search_dir="${1:-${JOB_WORKSPACE:-$_PROM_SCRIPT_DIR/outputs}}"
    if [[ ! -d "$search_dir" ]]; then
        echo "Directory not found: $search_dir" >&2
        return 1
    fi

    local found=0
    for d in "$search_dir"/*/prometheus; do
        [[ -d "$d" ]] || continue
        local job_dir
        job_dir=$(dirname "$d")
        local job_name
        job_name=$(basename "$job_dir")

        # Show human-readable size
        local size
        size=$(du -sh "$d" 2>/dev/null | cut -f1)

        # Show time range from TSDB blocks (first/last block dirs are timestamps)
        printf "  %-40s  %6s  %s\n" "$job_name" "$size" "$d"
        found=$((found + 1))
    done

    if [[ $found -eq 0 ]]; then
        echo "No Prometheus data found under: $search_dir" >&2
        return 1
    fi
    echo ""
    echo "View with:  $(basename "${BASH_SOURCE[0]}") view <path>"
}

# ============================================================================
# CLI usage
# ============================================================================

_usage_main() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  view <data-dir> [-p PORT]    Start Prometheus UI to browse persisted metrics
  list [<workspace>]           List jobs with saved Prometheus data
  install                      Download the Prometheus binary to /tmp/prometheus

Examples:
  $(basename "$0") list
  $(basename "$0") list /shared/maxtext_jobs
  $(basename "$0") view outputs/12345-JAX-llama2-70b/prometheus
  $(basename "$0") view ./outputs/12345-JAX-llama2-70b/prometheus -p 9091
EOF
}

_usage_view() {
    cat <<EOF
Usage: $(basename "$0") view <data-dir> [-p PORT]

Start a read-only Prometheus instance serving persisted TSDB data.

Options:
  -p, --port PORT   Port to listen on (default: $PROMETHEUS_PORT)
  -h, --help        Show this help
EOF
}

# ============================================================================
# Direct invocation dispatch
# ============================================================================

# Only run the CLI dispatcher when executed directly (not sourced).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -e
    cmd="${1:-}"
    shift 2>/dev/null || true

    case "$cmd" in
        view)    view_prometheus "$@" ;;
        list)    list_prometheus "$@" ;;
        install) install_prometheus ;;
        -h|--help|help|"") _usage_main ;;
        *)
            echo "Unknown command: $cmd" >&2
            _usage_main >&2
            exit 1
            ;;
    esac
fi
