#!/bin/bash
# Ray cluster management utilities

_RAY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RAY_LOG_TO_STDERR=0
export RAY_BACKEND_LOG_LEVEL=fatal
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# ---- Performance tuning: reduce Ray worker-thread overhead ----
# Training runs in a subprocess (no GIL sharing with Ray), but Ray's internal
# threads (metrics pusher, health checks, event stats) still consume CPU on the
# node.  Throttle reporting frequencies to minimise background overhead.
export RAY_metrics_report_interval_ms=${RAY_metrics_report_interval_ms:-300000}   # 5 min (default 10s)
export RAY_event_stats_print_interval_ms=${RAY_event_stats_print_interval_ms:-0}  # disable event stats
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-30000}            # 30s (default ~10s)
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-20}               # tolerate slower heartbeats

# RAY_PORT must be set by the caller (run_setup.sh, _job.sbatch, or
# _container.sh --env).  No default — prevents silent fallback to a
# potentially occupied port.
RAY_HEAD_IP="${JAX_COORDINATOR_IP:-localhost}"
RAY_METRICS_PORT=8080
PROMETHEUS_PORT=9090

# Persist Ray logs to the shared job output directory so they survive crashes.
# Inside Docker, /outputs is mounted from $JOB_WORKSPACE (or $SCRIPT_DIR/outputs) on the host.
# NOTE: We do NOT use --temp-dir for persistence because long paths exceed the
# 108-char Unix socket limit (sockaddr_un).  Instead, Ray uses the default
# /tmp/ray temp dir and we symlink its log directory to persistent storage.
RAY_LOG_DIR="/outputs/${JOB_DIR:-unknown}/ray_logs/$(hostname -s 2>/dev/null || hostname)"

# After ray start, redirect its log directory to persistent storage via symlink.
_persist_ray_logs() {
    local session_dir
    session_dir=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)
    if [[ -n "$session_dir" && -d "$session_dir/logs" ]]; then
        mkdir -p "$RAY_LOG_DIR"
        cp -a "$session_dir/logs/"* "$RAY_LOG_DIR/" 2>/dev/null || true
        rm -rf "$session_dir/logs"
        ln -sfn "$RAY_LOG_DIR" "$session_dir/logs"
        echo "[Ray] Logs -> ${RAY_LOG_DIR} (persistent, symlinked from $session_dir/logs)"
    fi
}

export RAY_PROMETHEUS_HOST="http://localhost:${PROMETHEUS_PORT}"

# Prometheus helpers (install_prometheus, start_prometheus, view_prometheus, …)
source "$(dirname "${BASH_SOURCE[0]}")/prometheus.sh"

# Node metrics exporter (GPU + host plugins)
_METRICS_EXPORTER_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/metrics_exporter.sh"

start_metrics_exporter() {
    if [[ -x "$_METRICS_EXPORTER_SCRIPT" ]]; then
        local log_file="/tmp/metrics_exporter.log"
        "$_METRICS_EXPORTER_SCRIPT" --port "${EXPORTER_PORT:-9400}" --interval 10 \
            >>"$log_file" 2>&1 &
        echo "[Metrics Exporter] Started on port ${EXPORTER_PORT:-9400} on $(hostname -s) (log: $log_file)"
    else
        echo "[Metrics Exporter] Script not found: $_METRICS_EXPORTER_SCRIPT" >&2
    fi
}

# ============================================================================
# Submission helpers (used by submit.sh)
# ============================================================================

# Build the sbatch --export flag for Ray jobs.
# Prints the flag string to stdout.
build_ray_export() {
    source "$(dirname "${BASH_SOURCE[0]}")/detect_ip.sh"
    local public_ip
    public_ip="$(detect_ip)"
    local login_hostname
    login_hostname=$(hostname -f 2>/dev/null || hostname)
    echo "--export=ALL,USE_RAY=true,LOGIN_NODE_HOSTNAME=${USER}@${login_hostname},LOGIN_NODE_IP=${USER}@${public_ip}"
}

# ============================================================================
# Ray cluster management
# ============================================================================

install_ray() {
    python3 -c "import ray" 2>/dev/null || pip3 install -q "ray[default]>=2.9.0"
    # py-spy enables stack traces & flame graphs from the Ray dashboard.
    # Install alongside Ray so the dependency is explicit.
    command -v py-spy &>/dev/null || pip3 install -q py-spy
}

start_ray_head() {
    echo "[Ray] Starting HEAD on $(hostname):${RAY_PORT}"
    if ! ray start --head --port=$RAY_PORT \
        --num-cpus=1 \
        --dashboard-host=0.0.0.0 --dashboard-port=8265 \
        --metrics-export-port=$RAY_METRICS_PORT \
        --disable-usage-stats 2>&1; then
        echo "[Ray] HEAD failed to start (port $RAY_PORT, dashboard 8265)" >&2
        return 1
    fi
    _persist_ray_logs

    for _ in {1..30}; do ray status &>/dev/null && return 0; sleep 2; done
    echo "[Ray] HEAD timeout" >&2
    return 1
}

start_ray_worker() {
    echo "[Ray] Starting WORKER -> ${RAY_HEAD_IP}:${RAY_PORT}"
    sleep 5
    for i in {1..18}; do
        ray start --address="${RAY_HEAD_IP}:${RAY_PORT}" \
            --num-cpus=1 \
            --no-monitor \
            --metrics-export-port=$RAY_METRICS_PORT \
            --disable-usage-stats &>/dev/null && {
            _persist_ray_logs
            return 0
        }
        echo "[Ray] Retry $i/18..."
        sleep 10
    done
    echo "[Ray] WORKER failed" >&2
    return 1
}

_install_pyspy_subprocess_wrapper() {
    # Wrap py-spy to redirect from the Ray worker PID to the training subprocess.
    #
    # The Ray Dashboard calls 'py-spy dump -p <worker_pid>', but the worker is
    # just sitting in p.wait() — the real training runs in a child process
    # (mfu_tracker.py).  Attaching py-spy to the worker's thread-heavy process
    # and traversing via --subprocesses is slow and unreliable (frequent 500s).
    #
    # This wrapper finds the worker's child PID and targets it directly, which
    # is faster and far more reliable.  Falls back to the original PID if no
    # child exists (training not yet started or already finished).
    local real_pyspy
    real_pyspy="$(command -v py-spy 2>/dev/null)" || return 0

    # Skip if already wrapped
    [[ -f "${real_pyspy}-real" ]] && return 0

    mv "$real_pyspy" "${real_pyspy}-real"
    cat > "$real_pyspy" <<'PYSPY_WRAPPER'
#!/bin/bash
REAL="$(dirname "$0")/$(basename "$0")-real"
if [[ "$1" == "dump" || "$1" == "record" ]]; then
    # Redirect to child process: the Ray Dashboard targets the Ray worker PID,
    # but the training code runs in a subprocess (mfu_tracker.py).  Targeting
    # the child directly is faster and far more reliable than traversing the
    # worker's thread-heavy process tree.
    ARGS=("$@")
    for ((i=0; i<${#ARGS[@]}; i++)); do
        if [[ "${ARGS[$i]}" == "-p" && -n "${ARGS[$((i+1))]}" ]]; then
            CHILD=$(pgrep -P "${ARGS[$((i+1))]}" 2>/dev/null | head -1)
            [[ -n "$CHILD" ]] && ARGS[$((i+1))]="$CHILD"
            break
        fi
    done
    exec "$REAL" "${ARGS[@]}" --subprocesses
else
    exec "$REAL" "$@"
fi
PYSPY_WRAPPER
    chmod +x "$real_pyspy"
    echo "[py-spy] Wrapper installed: dump/record will redirect to training subprocess"
}

start_ray_cluster() {
    install_ray || return 1
    _install_pyspy_subprocess_wrapper
    ray stop --force &>/dev/null || true
    pkill -f "prometheus" &>/dev/null || true
    pkill -f "metrics_exporter.sh" &>/dev/null || true
    pkill -f "node_hw_metrics.prom" &>/dev/null || true
    # Wait for ports to be released after SIGKILL
    sleep 2

    # Node metrics exporter runs on EVERY node (not just head)
    start_metrics_exporter

    if [[ ${NODE_RANK:-0} -eq 0 ]]; then
        start_ray_head || return 1
        start_prometheus
    else
        start_ray_worker
    fi
}

stop_ray_cluster() {
    ray stop --force &>/dev/null || true
    pkill -f "prometheus" &>/dev/null || true
    pkill -f "metrics_exporter.sh" &>/dev/null || true
    pkill -f "node_hw_metrics.prom" &>/dev/null || true
    pkill -f "tensorboard" &>/dev/null || true
}

# ============================================================================
# TensorBoard
# ============================================================================

start_tensorboard() {
    # OUTPUT_PATH is exported by _train_with_ray.sh via resolve_output_path().
    local output_dir="${OUTPUT_PATH:?OUTPUT_PATH not set}"
    mkdir -p "$output_dir"
    tensorboard --logdir="$output_dir" --port=6006 --bind_all &>/dev/null &
    echo "[TensorBoard] Started on port 6006 (logdir: $output_dir)"
}

# ============================================================================
# Info display
# ============================================================================

print_ray_info() {
    local host=$(hostname -s)
    local login_hostname="${LOGIN_NODE_HOSTNAME:-user-unknown@hostname-unknown}"
    local login_ip="${LOGIN_NODE_IP:-user-unknown@ip-unknown}"
    cat <<EOF
==============================================
SSH tunnel from your local machine (use hostname or IP, whichever works):
  ssh -L 8265:${host}:8265 -L 6006:${host}:6006 -L 9090:${host}:9090 ${login_hostname}
  ssh -L 8265:${host}:8265 -L 6006:${host}:6006 -L 9090:${host}:9090 ${login_ip}

Then open localhost:8265/6006/9090 in your browser:
  Ray Dashboard:  http://localhost:8265
  TensorBoard:    http://localhost:6006
  Prometheus:     http://localhost:${PROMETHEUS_PORT}

Prometheus TSDB and Ray logs write directly to persistent storage (survive job termination).
To inspect after the job ends:
  utils/prometheus.sh list
  utils/prometheus.sh view <JOB_WORKSPACE>/<job>/prometheus

Ray logs are persisted per node under:
  <JOB_WORKSPACE>/<job>/ray_logs/<hostname>/session_*/logs/

Tip: Port conflict when monitoring multiple jobs? Change local port: -L 18265:... -> localhost:18265

Debug:
  srun --jobid=\${SLURM_JOB_ID} --pty bash
  ray list actors / ray status
==============================================
EOF
}
