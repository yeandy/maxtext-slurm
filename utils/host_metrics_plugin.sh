#!/usr/bin/env bash
# host_metrics_plugin.sh — Prometheus metrics plugin: host network / RDMA / I/O.
#
# Collects per-interface network stats, TCP retransmits + listen queue
# health, RDMA port counters, scheduling pressure, OOM kills, and
# storage write pressure (dirty/writeback) — all from procfs/sysfs
# with zero external dependencies.
#
# Called by metrics_exporter.sh — outputs Prometheus text to stdout.
#
# Metrics (all prefixed hw_ for grouping in Prometheus UI):
#   hw_net_{rx,tx}_{bytes,errors,drop}_total{device,host}
#   hw_tcp_retransmits_total{host}
#   hw_tcp_listen_overflows_total{host}
#   hw_tcp_listen_drops_total{host}
#   hw_tcp_estab_resets_total{host}                  Established connections killed by RST
#   hw_tcp_abort_on_timeout_total{host}              Connections abandoned after retransmit exhaustion
#   hw_rdma_{rx,tx}_bytes_total{device,port,host}
#   hw_rdma_{rx,tx}_pkts_total{device,port,host}
#   hw_rdma_tx_retx_{bytes,pkts}_total{device,port,host}
#   hw_rdma_tx_ack_timeout_total{device,port,host}
#   hw_rdma_rx_ecn_pkts_total{device,port,host}
#   hw_rdma_{rx,tx}_cnp_pkts_total{device,port,host}
#   hw_rdma_req_rx_cqe_err_total{device,port,host}
#   hw_rdma_req_tx_retry_excd_err_total{device,port,host}
#   hw_rdma_port_state{device,port,host}            Port link state (1=ACTIVE, 0=not)
#   hw_procs_running{host}
#   hw_procs_blocked{host}
#   hw_context_switches_total{host}
#   hw_oom_kills_total{host}
#   hw_mem_dirty_bytes{host}
#   hw_mem_writeback_bytes{host}
#   hw_rdma_ccl_tx_retx_{pkts,bytes}_total{device,port,host}   Collective-traffic retransmissions
#   hw_rdma_ccl_tx_ack_timeout_total{device,port,host}         Collective-traffic ACK timeouts
#   hw_rdma_ccl_{tx,rx}_{pkts,bytes}_total{device,port,host}  Collective-traffic volume
#   hw_rdma_rx_dup_{response,request}_total{device,port,host} Duplicate packets (remote retx)
#   hw_io_pressure_{some,full}_pct{host}              I/O pressure (PSI 10s avg)
#   hw_io_pressure_{some,full}_avg300_pct{host}       I/O pressure (PSI 300s avg)
#   hw_io_pressure_full_total_us{host}                Cumulative I/O stall time (µs)
#   hw_dmesg_gpu_errors_total{host}                   GPU/driver errors in kernel ring buffer
#   hw_gpu_user_processes{host}                       Processes with /dev/kfd open (GPU users)

HOSTNAME_SHORT="${1:?Usage: host_metrics_plugin.sh <hostname>}"

python3 - "$HOSTNAME_SHORT" <<'PYEOF'
import os, re, sys
from glob import glob as globfn

hostname = sys.argv[1]
lines = []

def add(line):
    lines.append(line)

def read_file(path):
    """Read a file, return contents or empty string on failure."""
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ''

def read_int(path, default=0):
    """Read a single integer from a sysfs file."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return default

# =========================================================================
# Per-interface network stats  (/proc/net/dev)
# =========================================================================
# Format: iface: rx_bytes rx_packets rx_errs rx_drop ... tx_bytes tx_packets tx_errs tx_drop ...
#         indices:   0         1         2       3          8         9         10       11

SKIP_IFACES = {'lo', 'docker0'}

add('# HELP hw_net_rx_bytes_total Network bytes received per interface.')
add('# TYPE hw_net_rx_bytes_total counter')
add('# HELP hw_net_tx_bytes_total Network bytes transmitted per interface.')
add('# TYPE hw_net_tx_bytes_total counter')
add('# HELP hw_net_rx_errors_total Network receive errors per interface.')
add('# TYPE hw_net_rx_errors_total counter')
add('# HELP hw_net_tx_errors_total Network transmit errors per interface.')
add('# TYPE hw_net_tx_errors_total counter')
add('# HELP hw_net_rx_drop_total Network receive drops per interface.')
add('# TYPE hw_net_rx_drop_total counter')
add('# HELP hw_net_tx_drop_total Network transmit drops per interface.')
add('# TYPE hw_net_tx_drop_total counter')

try:
    for line in read_file('/proc/net/dev').splitlines():
        m = re.match(r'\s*(\S+):\s+(.*)', line)
        if not m:
            continue
        iface = m.group(1)
        if iface in SKIP_IFACES or iface.startswith('veth'):
            continue
        vals = m.group(2).split()
        if len(vals) < 16:
            continue
        lb = f'device="{iface}",host="{hostname}"'
        add(f'hw_net_rx_bytes_total{{{lb}}} {vals[0]}')
        add(f'hw_net_rx_errors_total{{{lb}}} {vals[2]}')
        add(f'hw_net_rx_drop_total{{{lb}}} {vals[3]}')
        add(f'hw_net_tx_bytes_total{{{lb}}} {vals[8]}')
        add(f'hw_net_tx_errors_total{{{lb}}} {vals[10]}')
        add(f'hw_net_tx_drop_total{{{lb}}} {vals[11]}')
except Exception as e:
    print(f'[host_plugin] /proc/net/dev: {e}', file=sys.stderr)

# =========================================================================
# TCP retransmits  (/proc/net/snmp)
# =========================================================================
add('# HELP hw_tcp_retransmits_total Cumulative TCP retransmitted segments.')
add('# TYPE hw_tcp_retransmits_total counter')

try:
    snmp = read_file('/proc/net/snmp')
    tcp_header = None
    for line in snmp.splitlines():
        if line.startswith('Tcp:'):
            parts = line.split()
            if parts[1] == 'RtoAlgorithm':
                tcp_header = parts[1:]  # field names
            elif tcp_header:
                vals = parts[1:]
                if len(vals) == len(tcp_header):
                    tcp_data = dict(zip(tcp_header, vals))
                    retrans = tcp_data.get('RetransSegs', '0')
                    add(f'hw_tcp_retransmits_total{{host="{hostname}"}} {retrans}')
except Exception as e:
    print(f'[host_plugin] /proc/net/snmp: {e}', file=sys.stderr)

# =========================================================================
# RDMA / InfiniBand port counters  (/sys/class/infiniband)
# =========================================================================
# Two sysfs layouts are supported:
#   Standard IB/RoCE:  .../ports/N/counters/port_rcv_data  (4-byte units)
#   Ionic (Pensando):  .../ports/N/hw_counters/rx_rdma_ucast_bytes (bytes)
# All byte metrics are normalised to actual bytes.

add('# HELP hw_rdma_rx_bytes_total RDMA unicast bytes received.')
add('# TYPE hw_rdma_rx_bytes_total counter')
add('# HELP hw_rdma_tx_bytes_total RDMA unicast bytes transmitted.')
add('# TYPE hw_rdma_tx_bytes_total counter')
add('# HELP hw_rdma_rx_pkts_total RDMA unicast packets received.')
add('# TYPE hw_rdma_rx_pkts_total counter')
add('# HELP hw_rdma_tx_pkts_total RDMA unicast packets transmitted.')
add('# TYPE hw_rdma_tx_pkts_total counter')
add('# HELP hw_rdma_tx_retx_bytes_total RDMA retransmitted bytes.')
add('# TYPE hw_rdma_tx_retx_bytes_total counter')
add('# HELP hw_rdma_tx_retx_pkts_total RDMA retransmitted packets.')
add('# TYPE hw_rdma_tx_retx_pkts_total counter')
add('# HELP hw_rdma_tx_ack_timeout_total RDMA ACK timeouts.')
add('# TYPE hw_rdma_tx_ack_timeout_total counter')
add('# HELP hw_rdma_rx_ecn_pkts_total RDMA packets received with ECN CE mark.')
add('# TYPE hw_rdma_rx_ecn_pkts_total counter')
add('# HELP hw_rdma_rx_cnp_pkts_total RDMA Congestion Notification Packets received.')
add('# TYPE hw_rdma_rx_cnp_pkts_total counter')
add('# HELP hw_rdma_tx_cnp_pkts_total RDMA Congestion Notification Packets sent.')
add('# TYPE hw_rdma_tx_cnp_pkts_total counter')
add('# HELP hw_rdma_req_rx_cqe_err_total RDMA requester CQE errors (bad completions).')
add('# TYPE hw_rdma_req_rx_cqe_err_total counter')
add('# HELP hw_rdma_req_tx_retry_excd_err_total RDMA requests where retries were exhausted.')
add('# TYPE hw_rdma_req_tx_retry_excd_err_total counter')
add('# HELP hw_rdma_ccl_tx_retx_pkts_total RDMA collective-traffic retransmitted packets.')
add('# TYPE hw_rdma_ccl_tx_retx_pkts_total counter')
add('# HELP hw_rdma_ccl_tx_retx_bytes_total RDMA collective-traffic retransmitted bytes.')
add('# TYPE hw_rdma_ccl_tx_retx_bytes_total counter')
add('# HELP hw_rdma_ccl_tx_ack_timeout_total RDMA collective-traffic ACK timeouts.')
add('# TYPE hw_rdma_ccl_tx_ack_timeout_total counter')
add('# HELP hw_rdma_ccl_tx_pkts_total RDMA collective-traffic packets sent.')
add('# TYPE hw_rdma_ccl_tx_pkts_total counter')
add('# HELP hw_rdma_ccl_tx_bytes_total RDMA collective-traffic bytes sent.')
add('# TYPE hw_rdma_ccl_tx_bytes_total counter')
add('# HELP hw_rdma_ccl_rx_pkts_total RDMA collective-traffic packets received.')
add('# TYPE hw_rdma_ccl_rx_pkts_total counter')
add('# HELP hw_rdma_ccl_rx_bytes_total RDMA collective-traffic bytes received.')
add('# TYPE hw_rdma_ccl_rx_bytes_total counter')
add('# HELP hw_rdma_rx_dup_response_total RDMA duplicate responses received (remote retransmitted).')
add('# TYPE hw_rdma_rx_dup_response_total counter')
add('# HELP hw_rdma_rx_dup_request_total RDMA duplicate requests received (remote retransmitted to us).')
add('# TYPE hw_rdma_rx_dup_request_total counter')

# ---------------------------------------------------------------------------
# Warm up ionic NIC firmware caches in parallel.
# The ionic driver queries the NIC firmware on the first sysfs counter read
# per device (~60ms round-trip).  With 8 NICs sequentially, that's ~500ms.
# Triggering one read per device in parallel brings this down to ~100ms
# (limited by the slowest single firmware query).  Subsequent reads within
# the same scrape cycle hit the driver cache and cost <0.1ms each.
# ---------------------------------------------------------------------------
try:
    from concurrent.futures import ThreadPoolExecutor
    hw_counter_dirs = []
    for dd in sorted(globfn('/sys/class/infiniband/*')):
        for pd in sorted(globfn(f'{dd}/ports/*')):
            hc = f'{pd}/hw_counters'
            if os.path.isdir(hc):
                hw_counter_dirs.append(hc)

    def _warmup(hc_dir):
        # Any read will prime the firmware cache for this device
        read_int(f'{hc_dir}/rx_rdma_ucast_bytes', 0)

    if hw_counter_dirs:
        with ThreadPoolExecutor(max_workers=len(hw_counter_dirs)) as pool:
            list(pool.map(_warmup, hw_counter_dirs))
except Exception:
    pass  # ThreadPoolExecutor not available — fall through to sequential reads

try:
    for dev_dir in sorted(globfn('/sys/class/infiniband/*')):
        device = os.path.basename(dev_dir)
        for port_dir in sorted(globfn(f'{dev_dir}/ports/*')):
            port = os.path.basename(port_dir)
            lb = f'device="{device}",port="{port}",host="{hostname}"'

            counters_dir = f'{port_dir}/counters'
            hw_counters_dir = f'{port_dir}/hw_counters'

            if os.path.isdir(counters_dir):
                # Standard IB/RoCE layout: values in 4-byte (lane) units.
                rcv = read_int(f'{counters_dir}/port_rcv_data')
                xmt = read_int(f'{counters_dir}/port_xmit_data')
                add(f'hw_rdma_rx_bytes_total{{{lb}}} {rcv * 4}')
                add(f'hw_rdma_tx_bytes_total{{{lb}}} {xmt * 4}')
                rcv_p = read_int(f'{counters_dir}/port_rcv_packets')
                xmt_p = read_int(f'{counters_dir}/port_xmit_packets')
                add(f'hw_rdma_rx_pkts_total{{{lb}}} {rcv_p}')
                add(f'hw_rdma_tx_pkts_total{{{lb}}} {xmt_p}')

            if os.path.isdir(hw_counters_dir):
                # Ionic / Pensando layout: byte counters are already in bytes.
                hw = hw_counters_dir
                # Traffic — only emit if the standard counters/ was absent,
                # to avoid double-counting on devices that have both.
                if not os.path.isdir(counters_dir):
                    for sysfs_name, metric in [
                        ('rx_rdma_ucast_bytes', 'hw_rdma_rx_bytes_total'),
                        ('tx_rdma_ucast_bytes', 'hw_rdma_tx_bytes_total'),
                        ('rx_rdma_ucast_pkts',  'hw_rdma_rx_pkts_total'),
                        ('tx_rdma_ucast_pkts',  'hw_rdma_tx_pkts_total'),
                    ]:
                        val = read_int(f'{hw}/{sysfs_name}')
                        add(f'{metric}{{{lb}}} {val}')

                # Retransmission (always useful regardless of counters/ presence)
                for sysfs_name, metric in [
                    ('tx_rdma_retx_bytes',     'hw_rdma_tx_retx_bytes_total'),
                    ('tx_rdma_retx_pkts',      'hw_rdma_tx_retx_pkts_total'),
                    ('tx_rdma_ack_timeout',    'hw_rdma_tx_ack_timeout_total'),
                ]:
                    path = f'{hw}/{sysfs_name}'
                    if os.path.exists(path):
                        add(f'{metric}{{{lb}}} {read_int(path)}')

                # Congestion (ECN / DCQCN)
                for sysfs_name, metric in [
                    ('rx_rdma_ecn_pkts',  'hw_rdma_rx_ecn_pkts_total'),
                    ('rx_rdma_cnp_pkts',  'hw_rdma_rx_cnp_pkts_total'),
                    ('tx_rdma_cnp_pkts',  'hw_rdma_tx_cnp_pkts_total'),
                ]:
                    path = f'{hw}/{sysfs_name}'
                    if os.path.exists(path):
                        add(f'{metric}{{{lb}}} {read_int(path)}')

                # Critical error counters
                for sysfs_name, metric in [
                    ('req_rx_cqe_err',        'hw_rdma_req_rx_cqe_err_total'),
                    ('req_tx_retry_excd_err', 'hw_rdma_req_tx_retry_excd_err_total'),
                ]:
                    path = f'{hw}/{sysfs_name}'
                    if os.path.exists(path):
                        add(f'{metric}{{{lb}}} {read_int(path)}')

                # ----- CCL (collective communication) specific counters -----
                # The ionic driver tracks collective-traffic retransmissions
                # separately from regular RDMA retransmissions.  These are the
                # most direct RDMA-level indicators of collective communication
                # degradation — the tx_retx counters here show packets that
                # were retransmitted specifically during allreduce/allgather
                # operations, isolating training-traffic retx from background
                # RDMA retx (e.g. checkpoint I/O via NFS-over-RDMA).
                for sysfs_name, metric in [
                    ('tx_rdma_ccl_cts_retx_pkts',  'hw_rdma_ccl_tx_retx_pkts_total'),
                    ('tx_rdma_ccl_cts_retx_bytes', 'hw_rdma_ccl_tx_retx_bytes_total'),
                    ('tx_rdma_ccl_cts_ack_timeout', 'hw_rdma_ccl_tx_ack_timeout_total'),
                ]:
                    path = f'{hw}/{sysfs_name}'
                    if os.path.exists(path):
                        add(f'{metric}{{{lb}}} {read_int(path)}')

                # CCL traffic volume (for computing retx ratio)
                for sysfs_name, metric in [
                    ('tx_rdma_ccl_cts_pkts',  'hw_rdma_ccl_tx_pkts_total'),
                    ('tx_rdma_ccl_cts_bytes', 'hw_rdma_ccl_tx_bytes_total'),
                    ('rx_rdma_ccl_cts_pkts',  'hw_rdma_ccl_rx_pkts_total'),
                    ('rx_rdma_ccl_cts_bytes', 'hw_rdma_ccl_rx_bytes_total'),
                ]:
                    path = f'{hw}/{sysfs_name}'
                    if os.path.exists(path):
                        add(f'{metric}{{{lb}}} {read_int(path)}')

                # ----- Duplicate packet counters -----
                # Duplicate responses mean our retransmissions crossed with
                # the original reply; duplicate requests mean the remote side
                # is retransmitting to us.  Both indicate network congestion
                # or packet loss on the path.
                for sysfs_name, metric in [
                    ('req_rx_dup_response',  'hw_rdma_rx_dup_response_total'),
                    ('resp_rx_dup_request',  'hw_rdma_rx_dup_request_total'),
                ]:
                    path = f'{hw}/{sysfs_name}'
                    if os.path.exists(path):
                        add(f'{metric}{{{lb}}} {read_int(path)}')

except Exception as e:
    print(f'[host_plugin] RDMA counters: {e}', file=sys.stderr)

# =========================================================================
# RDMA port link state  (/sys/class/infiniband/*/ports/*/state)
# =========================================================================
# Reports whether each RDMA port is ACTIVE (state 4) or not.  A link flap
# (ACTIVE → DOWN → ACTIVE) during training would cause RCCL collectives
# to hang or timeout.  Exporting as a gauge: 1 = ACTIVE, 0 = not active.

add('# HELP hw_rdma_port_state RDMA port link state (1=ACTIVE, 0=not active).')
add('# TYPE hw_rdma_port_state gauge')

try:
    for dev_dir in sorted(globfn('/sys/class/infiniband/*')):
        device = os.path.basename(dev_dir)
        for port_dir in sorted(globfn(f'{dev_dir}/ports/*')):
            port = os.path.basename(port_dir)
            state_file = f'{port_dir}/state'
            if os.path.exists(state_file):
                text = read_file(state_file).strip()
                # Format is "N: STATE_NAME", e.g. "4: ACTIVE"
                is_active = 1 if 'ACTIVE' in text else 0
                lb = f'device="{device}",port="{port}",host="{hostname}"'
                add(f'hw_rdma_port_state{{{lb}}} {is_active}')
except Exception as e:
    print(f'[host_plugin] RDMA port state: {e}', file=sys.stderr)

# =========================================================================
# Scheduling pressure  (/proc/stat)
# =========================================================================
# NOTE: host memory (ray_node_mem_*) and CPU utilization (ray_node_cpu_*)
# are already exported by Ray.  We add kernel-level scheduling counters
# that Ray does NOT expose — these are essential for diagnosing thread
# starvation (e.g. heartbeat thread blocked by GIL contention).

add('# HELP hw_procs_running Number of processes in runnable state.')
add('# TYPE hw_procs_running gauge')
add('# HELP hw_procs_blocked Number of processes blocked on I/O (D-state).')
add('# TYPE hw_procs_blocked gauge')
add('# HELP hw_context_switches_total Cumulative voluntary + involuntary context switches.')
add('# TYPE hw_context_switches_total counter')

try:
    for line in read_file('/proc/stat').splitlines():
        if line.startswith('procs_running '):
            add(f'hw_procs_running{{host="{hostname}"}} {line.split()[1]}')
        elif line.startswith('procs_blocked '):
            add(f'hw_procs_blocked{{host="{hostname}"}} {line.split()[1]}')
        elif line.startswith('ctxt '):
            add(f'hw_context_switches_total{{host="{hostname}"}} {line.split()[1]}')
except Exception as e:
    print(f'[host_plugin] /proc/stat: {e}', file=sys.stderr)

# =========================================================================
# OOM kills  (/proc/vmstat)
# =========================================================================
add('# HELP hw_oom_kills_total Cumulative OOM killer invocations.')
add('# TYPE hw_oom_kills_total counter')

try:
    for line in read_file('/proc/vmstat').splitlines():
        if line.startswith('oom_kill '):
            add(f'hw_oom_kills_total{{host="{hostname}"}} {line.split()[1]}')
            break
except Exception as e:
    print(f'[host_plugin] /proc/vmstat: {e}', file=sys.stderr)

# =========================================================================
# Storage write pressure  (/proc/meminfo — Dirty + Writeback)
# =========================================================================
# Dirty:     bytes queued in kernel page cache waiting for writeback.
# Writeback: bytes actively being written to backing storage.
#
# Together these show how much data the kernel is buffering for writes.
# During large checkpoint saves (hundreds of GB flushed to shared storage),
# these spike on the writing nodes and directly correlate with
# hw_procs_blocked — making them the most direct indicator of
# checkpoint I/O pressure that can stall heartbeats and NCCL collectives.
#
# Not exported by Ray (psutil.virtual_memory() omits these fields).

add('# HELP hw_mem_dirty_bytes Bytes in page cache queued for writeback.')
add('# TYPE hw_mem_dirty_bytes gauge')
add('# HELP hw_mem_writeback_bytes Bytes actively being written to storage.')
add('# TYPE hw_mem_writeback_bytes gauge')

try:
    for line in read_file('/proc/meminfo').splitlines():
        if line.startswith('Dirty:'):
            kb = int(line.split()[1])
            add(f'hw_mem_dirty_bytes{{host="{hostname}"}} {kb * 1024}')
        elif line.startswith('Writeback:'):
            kb = int(line.split()[1])
            add(f'hw_mem_writeback_bytes{{host="{hostname}"}} {kb * 1024}')
except Exception as e:
    print(f'[host_plugin] /proc/meminfo: {e}', file=sys.stderr)

# =========================================================================
# TCP extended stats  (/proc/net/netstat — TcpExt)
# =========================================================================
# ListenOverflows: SYN received when accept queue full (server-side)
# ListenDrops:     connections dropped because accept queue full
# EstabResets:     established connections killed by RST (local or remote)
# TCPAbortOnTimeout: connections our side abandoned after retransmit exhaustion
#
# ListenOverflows/Drops catch gRPC/coordination-service connection failures.
# EstabResets and AbortOnTimeout diagnose mid-session connection deaths:
#   - EstabResets spikes, AbortOnTimeout doesn't → remote sent RST
#   - AbortOnTimeout spikes → network path died, our side gave up
#   - Neither spikes → TCP was fine, problem was inside the process
# On dedicated nodes (no other tenants), these counters have high signal
# because all TCP connections belong to the job (Ray, JAX coordinator, NFS).

add('# HELP hw_tcp_listen_overflows_total TCP listen queue overflows.')
add('# TYPE hw_tcp_listen_overflows_total counter')
add('# HELP hw_tcp_listen_drops_total TCP connections dropped from listen queue.')
add('# TYPE hw_tcp_listen_drops_total counter')
add('# HELP hw_tcp_estab_resets_total Established TCP connections killed by RST.')
add('# TYPE hw_tcp_estab_resets_total counter')
add('# HELP hw_tcp_abort_on_timeout_total TCP connections aborted after retransmit timeout exhaustion.')
add('# TYPE hw_tcp_abort_on_timeout_total counter')

try:
    netstat = read_file('/proc/net/netstat')
    tcpext_header = None
    for line in netstat.splitlines():
        if line.startswith('TcpExt:'):
            parts = line.split()
            if tcpext_header is None:
                tcpext_header = parts[1:]  # field names
            else:
                vals = parts[1:]
                if len(vals) == len(tcpext_header):
                    ext_data = dict(zip(tcpext_header, vals))
                    lo = ext_data.get('ListenOverflows', '0')
                    ld = ext_data.get('ListenDrops', '0')
                    er = ext_data.get('EstabResets', '0')
                    at = ext_data.get('TCPAbortOnTimeout', '0')
                    add(f'hw_tcp_listen_overflows_total{{host="{hostname}"}} {lo}')
                    add(f'hw_tcp_listen_drops_total{{host="{hostname}"}} {ld}')
                    add(f'hw_tcp_estab_resets_total{{host="{hostname}"}} {er}')
                    add(f'hw_tcp_abort_on_timeout_total{{host="{hostname}"}} {at}')
except Exception as e:
    print(f'[host_plugin] /proc/net/netstat: {e}', file=sys.stderr)

# =========================================================================
# Kernel ring buffer GPU/driver error count  (/dev/kmsg)
# =========================================================================
# Counts err/warn lines matching amdgpu/drm/xgmi in the kernel ring buffer.
# This catches GPU faults, XGMI link resets, PCIe errors, and other
# kernel-level events that are invisible to sysfs counters and userspace.
# The counter is cumulative over the kernel ring buffer lifetime.
#
# Reads /dev/kmsg directly (not the syslog(2) syscall used by `dmesg`)
# because the syslog interface returns empty inside containers even with
# --privileged, due to kernel namespace isolation.  /dev/kmsg is the
# character device interface and always reflects the host kernel buffer.
#
# Requires /dev/kmsg to be readable (--privileged or CAP_SYSLOG).
# ~8ms for 10K messages on MI300X nodes.

add('# HELP hw_dmesg_gpu_errors_total GPU/driver error lines in kernel ring buffer.')
add('# TYPE hw_dmesg_gpu_errors_total counter')

try:
    fd = os.open('/dev/kmsg', os.O_RDONLY | os.O_NONBLOCK)
    try:
        count = 0
        while True:
            try:
                data = os.read(fd, 8192)
                if not data:
                    break
                line = data.decode('utf-8', errors='replace')
                # /dev/kmsg format: priority,sequence,timestamp,flags;message
                parts = line.split(';', 1)
                if len(parts) < 2:
                    continue
                hdr = parts[0].split(',')
                if not hdr:
                    continue
                try:
                    pri = int(hdr[0])
                except ValueError:
                    continue
                if pri > 4:  # only err(3) and warn(4) and below
                    continue
                msg = parts[1].lower()
                if ('amdgpu' in msg or 'drm' in msg or 'xgmi' in msg) and \
                   ('error' in msg or 'fault' in msg or 'fail' in msg or
                    'reset' in msg or 'timeout' in msg):
                    count += 1
            except BlockingIOError:
                break
            except OSError:
                break
    finally:
        os.close(fd)
    add(f'hw_dmesg_gpu_errors_total{{host="{hostname}"}} {count}')
except (OSError, PermissionError):
    pass  # /dev/kmsg not accessible — skip silently

# =========================================================================
# I/O pressure  (/proc/pressure/io — PSI: Pressure Stall Information)
# =========================================================================
# The kernel PSI subsystem reports the fraction of time tasks are stalled
# waiting for I/O.  "full" means ALL non-idle tasks are stalled (nothing
# is making forward progress), while "some" means at least one task is
# stalled.  We export the 10-second and 300-second averages.
#
# During checkpoint saves, DP replica 0 nodes generate heavy write I/O to
# the shared filesystem; PSI io.full spikes show exactly when this I/O
# saturates the storage path enough to stall training threads.

add('# HELP hw_io_pressure_some_pct Percentage of time at least one task stalled on I/O (10s avg).')
add('# TYPE hw_io_pressure_some_pct gauge')
add('# HELP hw_io_pressure_full_pct Percentage of time all tasks stalled on I/O (10s avg).')
add('# TYPE hw_io_pressure_full_pct gauge')
add('# HELP hw_io_pressure_some_avg300_pct I/O pressure some (300s avg).')
add('# TYPE hw_io_pressure_some_avg300_pct gauge')
add('# HELP hw_io_pressure_full_avg300_pct I/O pressure full (300s avg).')
add('# TYPE hw_io_pressure_full_avg300_pct gauge')
add('# HELP hw_io_pressure_full_total_us Total I/O pressure stall time in microseconds.')
add('# TYPE hw_io_pressure_full_total_us counter')

try:
    psi_io = read_file('/proc/pressure/io')
    for line in psi_io.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]  # 'some' or 'full'
        kvs = {}
        for p in parts[1:]:
            if '=' not in p:
                continue
            k, v = p.split('=', 1)
            kvs[k] = v
        if kind == 'some':
            add(f'hw_io_pressure_some_pct{{host="{hostname}"}} {kvs.get("avg10", "0")}')
            add(f'hw_io_pressure_some_avg300_pct{{host="{hostname}"}} {kvs.get("avg300", "0")}')
        elif kind == 'full':
            add(f'hw_io_pressure_full_pct{{host="{hostname}"}} {kvs.get("avg10", "0")}')
            add(f'hw_io_pressure_full_avg300_pct{{host="{hostname}"}} {kvs.get("avg300", "0")}')
            add(f'hw_io_pressure_full_total_us{{host="{hostname}"}} {kvs.get("total", "0")}')
except Exception as e:
    print(f'[host_plugin] /proc/pressure/io: {e}', file=sys.stderr)

# NOTE: Per-device disk I/O (/proc/diskstats) is NOT collected here because
# Ray already exports aggregate disk I/O via ray_node_disk_io_{read,write}
# (from the same psutil→diskstats source).  Checkpoints go to NFS which
# does not appear in diskstats anyway; use hw_io_pressure_* (PSI) above
# to detect NFS I/O stalls during checkpoint saves.

# =========================================================================
# GPU user process count  (/sys/class/kfd/kfd/proc/)
# =========================================================================
# The KFD (Kernel Fusion Driver) maintains a directory of PIDs that have
# opened /dev/kfd.  Reading this directory is O(1) — no /proc/*/fd scan
# needed.  The ROCm runtime opens /dev/kfd for GPU access, so this count
# is a reliable indicator of whether training processes are alive.
#
# This is the most direct liveness signal for the failure mode we observed:
# a silent _exit(1) from inside RCCL would cause this count to drop before
# the heartbeat timeout fires.

add('# HELP hw_gpu_user_processes Number of processes with /dev/kfd open (GPU users).')
add('# TYPE hw_gpu_user_processes gauge')

try:
    kfd_proc = '/sys/class/kfd/kfd/proc'
    if os.path.isdir(kfd_proc):
        pids = [d for d in os.listdir(kfd_proc) if d.isdigit()]
        add(f'hw_gpu_user_processes{{host="{hostname}"}} {len(pids)}')
    else:
        add(f'hw_gpu_user_processes{{host="{hostname}"}} 0')
except Exception as e:
    print(f'[host_plugin] GPU user processes: {e}', file=sys.stderr)

# Output
print('\n'.join(lines))
PYEOF
