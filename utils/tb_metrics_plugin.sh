#!/usr/bin/env bash

# tb_metrics_plugin.sh — Prometheus metrics plugin: TensorBoard scalar metrics.
#
# Reads TensorBoard event files (standard TFRecord/RecordIO format) and exports
# all per-step scalar metrics.  Framework-agnostic — works with any TensorBoard-
# writing framework (MaxText, PyTorch, TensorFlow, etc.).
#
# File discovery:
#   The plugin is designed to start BEFORE the training job creates its
#   TensorBoard event file.  Discovery must therefore distinguish "no
#   file yet" from "stale file from a previous job" and never lock onto
#   the latter.
#
#   Lifecycle:
#     1. Plugin starts → records filter_ts = now (birth timestamp).
#        Any event file with creation_ts < filter_ts is from a previous
#        job and excluded.  Glob finds nothing new → backoff.
#     2. Training starts → creates event file with creation_ts >= filter_ts.
#     3. Next discovery finds the file by creation_ts alone → locks on.
#     4. Locked: zero NFS stat calls per poll cycle; the tail read
#        (getsize + open + read) is the only NFS I/O.  The lock is
#        permanent — staleness (checkpoint saves, GC pauses, I/O stalls)
#        is transient and the file will become fresh when training
#        resumes.  The lock breaks only when the file is physically
#        deleted (OSError from the tail read).
#     5. File deleted → re-enter waiting with same filter_ts → re-discover.
#     6. If no file ever appears (TensorBoard disabled, etc.) → give up
#        after _GIVE_UP_SECONDS (30 min) → exit 99 → exporter never
#        invokes this plugin again, zero cost.
#     7. Non-rank-0 nodes: detected at the bash level before Python starts
#        → exit 99 immediately, zero cost for the entire job lifetime.
#
#   Signals used:
#     • hostname in filename — filter to this host only
#     • creation timestamp in filename (immutable) — must be >= filter_ts
#       (= birth_ts on first run) to be considered; this single check
#       filters out all files from previous jobs
#
#   NFS budget:
#     The recursive glob is expensive (~5 s on large output trees).
#     • Hot path (locked file): zero globs, zero stat calls; only the
#       tail read (one getsize + one read) per poll cycle.
#     • Discovery: one glob + ZERO stat calls (pure filename filtering
#       by hostname and creation timestamp — no getmtime).
#     • Waiting for file (first ~5 min): one glob per 120 s.
#     • Waiting after ~5 min: backoff escalates to one glob per 300 s.
#     • Non-rank-0 nodes: bash wrapper detects NODE_RANK, exits 99 —
#       exporter never invokes this plugin again.
#     • No event file (TB disabled, etc.): ~3 globs at 120 s, then ~5
#       globs at 300 s, gives up at 30 min → exit 99 — exporter stops
#       invoking.  ~8 globs total over 30 min.
#
# Read strategy:
#   Reads the last 1 MB of the event file (RecordIO tail read).  Each TFRecord
#   is self-delimiting ([8B len][4B CRC][data][4B CRC]), so a partial first
#   record at the seek boundary is safely skipped.  1 MB is generous enough
#   for heavy custom instrumentation (hundreds of tags, histograms, images)
#   while adding negligible cost (~2-5 ms sequential NFS read).
#
# Step-drip & Prometheus timestamps:
#   TensorBoard writers (e.g. tensorboardX) flush event files infrequently
#   (default 120 s).  With ~30 s step times, a single flush delivers ~4
#   steps at once.  Without mitigation the scraper would jump from step N
#   to step N+4 — producing a staircase / zigzag pattern in dashboards.
#
#   Two mechanisms eliminate this:
#     1. Step-drip: instead of jumping to the latest step, the plugin
#        emits one new step per poll cycle (~10 s).  Any backlog drains
#        during idle periods (e.g. checkpoint saves).  In the rare case
#        that training produces steps faster than the drip can emit,
#        the oldest pending steps eventually fall off the 1 MB read
#        window and are skipped (best-effort).
#     2. Prometheus timestamps: each metric sample carries the event's
#        wall_time as a Prometheus timestamp (milliseconds).  When a
#        new step is emitted, it's placed at its real recording time,
#        giving dashboards a smooth, correctly-timed curve.
#
#   Four rules govern timestamp handling:
#     1. Real events (staleness_fill=0) always retain their original
#        wall_time — no clamping, no modification.  This ensures the
#        timeline pace accurately reflects real data.
#     2. Fills (staleness_fill=1) must preserve timeline integrity.
#        They must never insert older samples after newer ones — the
#        fill-safe guard checks this before every fill and skips it
#        if the fill's timestamp would follow a pending real step.
#        Fill timestamps are clamped to guarantee monotonicity
#        (> _last_prom_ts_ms) and freshness (>= now - MAX_STALENESS)
#        so Prometheus always accepts them.
#     3. In best-effort scenarios, real data always takes precedence.
#        Staleness fill-up events may be dropped if the drip
#        mechanism cannot keep up.
#     4. If real data cannot be dripped timely, drop older data.
#        Before emitting, steps whose wall_time has fallen below
#        now - MAX_STALENESS_MS are dropped (Prometheus would reject
#        them anyway).  The drip advances to the earliest fresh
#        step, keeping dashboards current.
#
# Idle-poll optimisation (file-size gate + exporter metric cache):
#   When the event file hasn't grown since the last poll, no new data
#   has been flushed.  The plugin detects this via os.path.getsize()
#   (a single NFS stat, ~0.1 ms) and exits immediately with only a
#   STATE line — skipping the 1 MB tail read and all protobuf parsing.
#   The exporter's metric cache replays the last emitted metric lines,
#   keeping Prometheus series alive (preventing scrape-level staleness)
#   without any redundant work in the plugin.  Exception: when the
#   fill timer expires (query-time staleness forming), the gate is
#   bypassed so the fill logic can emit anti-staleness samples.
#
# Called by metrics_exporter.sh — outputs Prometheus text to stdout.
#
# Metrics (all prefixed tb_ for grouping in Prometheus UI):
#   tb_learning_loss{host}
#   tb_learning_grad_norm{host}
#   tb_learning_raw_grad_norm{host}
#   tb_learning_param_norm{host}
#   tb_learning_current_learning_rate{host}
#   tb_learning_total_weights{host}
#   tb_learning_moe_lb_loss{host}
#   tb_learning_mtp_loss{host}
#   tb_perf_step_time_seconds{host}
#   tb_perf_per_device_tflops{host}
#   tb_perf_per_device_tflops_per_sec{host}
#   tb_perf_per_device_tokens{host}
#   tb_perf_per_device_tokens_per_sec{host}
#   (plus any other scalar tags the framework writes)

HOSTNAME_SHORT="${1:?Usage: tb_metrics_plugin.sh <hostname>}"

# ---------------------------------------------------------------------------
# Fast-path exits (bash-level, BEFORE starting Python).
# Exit code 99 tells the exporter to never invoke this plugin again.
# ---------------------------------------------------------------------------

# Non-rank-0 nodes never write TensorBoard events — skip permanently.
# NODE_RANK: set by the orchestration layer and exported into Docker
# by _container.sh (--env NODE_RANK=...).
_node_rank="${NODE_RANK:-}"
if [[ -n "$_node_rank" && "$_node_rank" != "0" ]]; then
    exit 99
fi

python3 - "$HOSTNAME_SHORT" <<'PYEOF'
import glob, os, re, socket, struct, sys, time

HOST = sys.argv[1]

# ---------------------------------------------------------------------------
# 0. Early exit for non-rank-0 nodes.
# ---------------------------------------------------------------------------
# Only rank 0 writes TensorBoard events.  The bash wrapper above handles
# the common case (NODE_RANK set).  This is a fallback for any edge case
# where the bash check didn't fire (e.g. env var differences between bash
# and Python in non-standard setups).
_node_rank = os.environ.get('NODE_RANK', '')
if _node_rank and _node_rank != '0':
    sys.exit(99)

# ---------------------------------------------------------------------------
# 1. Discover the active TensorBoard event file for this host.
# ---------------------------------------------------------------------------

# Negative-cache backoff between globs when no fresh file is found.
# Short backoff: file expected soon after training starts (~120 s flush).
_NONE_BACKOFF_SECONDS = 120
# Long backoff: after waiting >300 s, escalate.  Keeps NFS quiet for
# non-rank-0 nodes (without NODE_RANK) and TB-disabled jobs.
_NONE_BACKOFF_STALE_SECONDS = 300

# Give-up threshold: if we remain in __NONE__ for this long, stop
# globbing permanently.  Covers non-rank-0 without NODE_RANK,
# rank 0 with TensorBoard disabled, etc.
_GIVE_UP_SECONDS = 1800  # 30 min

# Creation-timestamp floor: files with created_ts < _filter_ts are
# excluded from discovery (pure filename check, zero NFS stat calls).
# Set to birth_ts (= time of first invocation) on first run and never
# raised — once locked, the file stays locked until deleted.
# Persisted in STATE.
_filter_ts = None

# When we entered the __NONE__ state (for give-up timeout and backoff
# escalation).  Reset when a file is found.  Persisted in STATE.
_wait_start_ts = None

# Last step emitted to Prometheus — used by the step-drip logic (§4)
# to advance one step per poll cycle instead of jumping to the latest,
# smoothing out bursts from infrequent event-file flushes (default
# 120 s).  Persisted in the locked STATE.
_last_emitted_step = None

# Cached file size (bytes) of the locked event file.  If getsize()
# matches, no new data has been flushed — skip the tail read entirely
# and let the exporter replay cached metrics.  Persisted in STATE.
_last_file_size = None

# Server time (epoch seconds) of the last emission (real drip or fill).
# Used ONLY by the fill timer: if time.time() - _last_emit_ts >= interval,
# a staleness gap has formed and fills are needed.  Always set to
# time.time() on every emission.  Persisted in STATE field 5.
_last_emit_ts = None

# Monotonically increasing Prometheus custom timestamp (ms) of the
# last emitted sample (real drip or fill).  Used to compute fill
# timestamps — fills advance the Prometheus timeline from this point,
# keeping fill timestamps anchored to real wall_times rather than
# server time.  This minimises the gap between fills and
# pre-checkpoint steps.  Persisted in STATE field 6.
_last_prom_ts_ms = 0

# Fill interval: 150 s keeps anti-staleness samples within the 300 s
# Prometheus staleness window with safety margin.
_FILL_INTERVAL_S = 150

# Fill staleness clamp: Prometheus's TSDB compacts every ~2 h,
# advancing head.minValidTime to the latest block's maxt.  Fill
# timestamps older than minValidTime are rejected (ErrOutOfBounds).
# Clamping fills to now - MAX_STALENESS_MS guarantees they stay
# within the TSDB's acceptable range.  Real events are never clamped
# (Rule 1); Rule 4 drops stale steps instead of emitting them.
_MAX_STALENESS_MS = 3_600_000  # 1 hour

def _is_fresh(step_data):
    """True if the step's wall_time is within the TSDB staleness window.

    Used by Rule 4: if real data cannot be dripped timely, drop older
    data.  Steps whose wall_time has fallen below the Prometheus TSDB
    minValidTime (approximated by now - MAX_STALENESS_MS) would be
    rejected anyway — skip them and move to fresher data.
    """
    wt_ms = int(max(wt for wt, _ in step_data.values()) * 1000)
    return wt_ms >= int(time.time() * 1000) - _MAX_STALENESS_MS

def _fill_timer_expired():
    """True when enough time has elapsed since the last emission that a
    Prometheus query-time staleness gap is forming and a fill is needed.
    """
    return (_last_emit_ts is not None
            and (time.time() - _last_emit_ts) >= _FILL_INTERVAL_S)

# Sentinel: negative cache still valid — caller emits nothing.
_NEGATIVE_CACHED = object()

# Hostname matching: short hostname anchored on dot boundaries to
# prevent host001 matching host0012.
_short = socket.gethostname().split('.')[0]
_host_re = re.compile(rf'\.{re.escape(_short)}(\.|$)')

def _parse_embedded_ts(basename):
    """Extract the creation-epoch (seconds) from an event filename.

    Filename format: events.out.tfevents.<epoch_seconds>.<hostname>[suffix]
    Returns int epoch or None on parse failure.
    """
    parts = basename.split('.')
    # parts: ['events', 'out', 'tfevents', '<epoch>', '<hostname>', ...]
    if len(parts) < 5:
        return None
    try:
        return int(parts[3])
    except (ValueError, IndexError):
        return None

def _discover_event_file(filter_ts, tolerance=0):
    """Glob for event files created at or after filter_ts for this host.

    One pass: glob → hostname filter → creation-ts filter → return newest.
    No mtime checks — any file with created_ts >= filter_ts is from the
    current job (filter_ts = birth_ts on first run guarantees this).

    Args:
        filter_ts: exclude files with created_ts < filter_ts - tolerance.
        tolerance: seconds of slack (default 0 = strict).  Used only by
                   the last-chance recovery glob for backward compat with
                   pre-fix STATE that may carry filter_ts = created_ts + 1.

    NFS cost: one recursive glob + ZERO stat calls.

    Returns path or None.
    """
    output_path = os.environ.get('OUTPUT_PATH', '')
    base = output_path if output_path else '/outputs'
    pattern = f'{base}/**/events.out.tfevents.*'

    best = None  # (created_ts, path)
    for f in glob.glob(pattern, recursive=True):
        basename = os.path.basename(f)
        if not _host_re.search(basename):
            continue
        created_ts = _parse_embedded_ts(basename)
        if created_ts is None:
            continue
        # Pure filename check — zero NFS stat calls.
        if filter_ts is not None and created_ts < filter_ts - tolerance:
            continue
        if best is None or created_ts > best[0]:
            best = (created_ts, f)

    return best[1] if best else None

def find_event_file():
    """Resolve the active TensorBoard event file for this host.

    Returns:
        str:              path to the event file (lock on it)
        None:             discovery ran, found nothing (caller sets STATE)
        _NEGATIVE_CACHED: negative cache still valid — caller emits nothing

    STATE protocol (persisted via '# STATE ...' as last output line):
        '<path>|ft|step|fsize|emit_ts'          — locked (5-field; -1 = unset)
        '<path>|ft'                             — locked (2-field, step unknown)
        '__NONE__|ft|glob_ts|wait_start_ts'     — waiting

    Older formats (parsed for backward compat, never emitted):
        '<path>|ft|step|fsize|emit_ts|prom_ts'   — 6-field
        '<path>|ft|step|fsize'                   — 4-field
        '<path>|ft|step'                         — 3-field
    """
    global _filter_ts, _wait_start_ts, _last_emitted_step, _last_file_size
    global _last_emit_ts, _last_prom_ts_ms

    saved = os.environ.get('_PLUGIN_STATE', '').strip()

    # ------------------------------------------------------------------
    # First run — no STATE.  Record birth_ts as filter and discover.
    # The plugin always starts before training creates its event file,
    # so birth_ts is guaranteed to be <= the creation timestamp of any
    # event file from the current job.
    #
    # The event file's creation timestamp (embedded in its filename)
    # is always >= the exporter's birth time, because the exporter
    # starts during container init, before training creates the file.
    # So the strict filter (created_ts >= filter_ts) always works for
    # the current job.  No fallback is needed.
    # ------------------------------------------------------------------
    if not saved:
        _filter_ts = int(time.time())
        result = _discover_event_file(_filter_ts)
        if result is not None:
            return result
        _wait_start_ts = int(time.time())
        return None

    # ------------------------------------------------------------------
    # __NONE__ state: waiting for an event file to appear.
    # STATE = __NONE__|<filter_ts>|<glob_ts>|<wait_start_ts>
    # ------------------------------------------------------------------
    if saved.startswith('__NONE__|'):
        parts = saved.split('|')
        try:
            _filter_ts = int(parts[1])
        except (ValueError, IndexError):
            _filter_ts = int(time.time())
        try:
            glob_ts = int(parts[2])
        except (ValueError, IndexError):
            glob_ts = 0
        try:
            _wait_start_ts = int(parts[3])
        except (ValueError, IndexError):
            _wait_start_ts = _filter_ts

        now = time.time()

        # Give up: no event file is coming.
        if now - _wait_start_ts > _GIVE_UP_SECONDS:
            # Last-chance glob: if the most recent glob was BEFORE the
            # give-up deadline, run one more before exiting permanently.
            # tolerance=1 covers pre-fix artifacts whose stale lock-break
            # raised filter_ts to created_ts + 1.  Exactly one extra glob
            # over the plugin's lifetime — negligible NFS.
            if glob_ts < _wait_start_ts + _GIVE_UP_SECONDS:
                result = _discover_event_file(_filter_ts, tolerance=1)
                if result is not None:
                    _wait_start_ts = None
                    return result
                # Fall through: return None → caller prints __NONE__ STATE
                # with updated glob_ts so the NEXT cycle hits the branch
                # below and exits 99.
                return None
            # Permanent exit: no event file after 30 min + last-chance glob.
            # Exit 99 → exporter will never invoke us again.
            sys.exit(99)

        # Backoff escalation: 120 s for first ~5 min, 300 s after.
        waited = now - _wait_start_ts
        backoff = (_NONE_BACKOFF_STALE_SECONDS
                   if waited > _NONE_BACKOFF_STALE_SECONDS
                   else _NONE_BACKOFF_SECONDS)
        if now - glob_ts < backoff:
            return _NEGATIVE_CACHED

        # Discovery (one glob, zero stats).
        result = _discover_event_file(_filter_ts)
        if result is not None:
            _wait_start_ts = None
            return result
        return None

    # ------------------------------------------------------------------
    # Locked state: reuse file unconditionally.
    # STATE = <path>|ft|step|fsize|emit_ts|prom_ts   (6-field, current)
    #       | <path>|ft|step|fsize|emit_ts           (5-field, compat)
    #       | <path>|ft|step|fsize                   (4-field, compat)
    #       | <path>|ft|step                         (3-field, compat)
    #       | <path>|ft                              (2-field, legacy)
    # -1 sentinel = "not set" for fsize, emit_ts, step.
    # ------------------------------------------------------------------
    # Once locked, the file is from the current job (it passed the
    # creation-timestamp filter during discovery).  Staleness is
    # transient (checkpoint saves, GC pauses, I/O stalls) — the file
    # will become fresh again when training resumes.  Breaking the lock
    # and raising filter_ts on stale mtime would permanently exclude
    # the current file (created_ts < raised filter_ts).
    #
    # The lock breaks only when the file is physically deleted (OSError
    # from read_tail_records in the main block).
    #
    # rsplit from the right — path may contain '|' (unlikely but safe).
    # Try widest format first, narrow on ValueError.

    # 6-field: path|filter_ts|step|file_size|emit_ts|prom_ts_ms
    # prom_ts_ms = monotonically increasing Prometheus custom ts (ms).
    # Also accepts legacy 6-field states where the 6th field was
    # fill_ts (epoch seconds) or prom_ts_ms — both are valid
    # starting points for the Prometheus timeline tracker.
    parts = saved.rsplit('|', 5)
    if len(parts) >= 6:
        try:
            ft, ls, fs, ets, ptm = (int(parts[-5]), int(parts[-4]),
                                    int(parts[-3]), int(parts[-2]),
                                    int(parts[-1]))
            _filter_ts = ft
            _last_emitted_step = ls if ls >= 0 else None
            _last_file_size = fs if fs >= 0 else None
            _last_emit_ts = ets if ets >= 0 else None
            _last_prom_ts_ms = ptm if ptm > 0 else 0
            return '|'.join(parts[:-5])
        except ValueError:
            pass

    # 5-field: path|filter_ts|step|file_size|emit_ts
    parts = saved.rsplit('|', 4)
    if len(parts) >= 5:
        try:
            ft, ls, fs, ets = (int(parts[-4]), int(parts[-3]),
                               int(parts[-2]), int(parts[-1]))
            _filter_ts = ft
            _last_emitted_step = ls if ls >= 0 else None
            _last_file_size = fs if fs >= 0 else None
            _last_emit_ts = ets if ets >= 0 else None
            return '|'.join(parts[:-4])
        except ValueError:
            pass

    # 4-field: path|filter_ts|last_step|file_size
    parts = saved.rsplit('|', 3)
    if len(parts) >= 4:
        try:
            ft, ls, fs = int(parts[-3]), int(parts[-2]), int(parts[-1])
            _filter_ts = ft
            _last_emitted_step = ls if ls >= 0 else None
            _last_file_size = fs
            return '|'.join(parts[:-3])
        except ValueError:
            pass

    # 3-field: path|filter_ts|last_step
    parts = saved.rsplit('|', 2)
    if len(parts) >= 3:
        try:
            ft, ls = int(parts[-2]), int(parts[-1])
            _filter_ts = ft
            _last_emitted_step = ls if ls >= 0 else None
            return '|'.join(parts[:-2])
        except ValueError:
            pass

    # 2-field (legacy): path|filter_ts
    parts = saved.rsplit('|', 1)
    path = parts[0]
    try:
        _filter_ts = int(parts[1])
    except (ValueError, IndexError):
        _filter_ts = int(time.time())

    return path

# ---------------------------------------------------------------------------
# 2. Read the tail of the event file (RecordIO format).
# ---------------------------------------------------------------------------

_TAIL_BYTES = 1_048_576  # 1 MB — covers heavy custom instrumentation + non-scalar data
_MAX_RECORD_LEN = 100_000  # sanity cap for a single TFRecord

def read_tail_records(path):
    """Read TFRecord records from the last _TAIL_BYTES of *path*.

    After seeking, we may land inside a record.  We scan forward byte-by-byte
    (up to 256 bytes) looking for a position where at least two consecutive
    records parse with plausible lengths.  Once synced, all remaining records
    in the buffer are returned.
    """
    file_size = os.path.getsize(path)
    offset = max(0, file_size - _TAIL_BYTES)
    with open(path, 'rb') as f:
        f.seek(offset)
        buf = f.read()

    if offset == 0:
        # Reading from start of file — already aligned.
        return _parse_records(buf, 0)

    # Scan for the first valid record boundary.
    scan_limit = min(256, len(buf))
    for start in range(scan_limit):
        recs = _parse_records(buf, start)
        if len(recs) >= 2:
            return recs
    return []


def _parse_records(buf, start):
    """Try to parse consecutive TFRecord records from buf[start:]."""
    records = []
    pos = start
    while pos + 12 <= len(buf):
        length = struct.unpack('<Q', buf[pos:pos + 8])[0]
        end = pos + 12 + length + 4
        if end > len(buf):
            break  # truncated record at buffer tail — stop
        if length > _MAX_RECORD_LEN:
            # Large record (tensor summary, graph proto, etc.) — skip it
            # but keep parsing.  Only scalar events matter downstream.
            pos = end
            continue
        records.append(buf[pos + 12:pos + 12 + length])
        pos = end
    return records

# ---------------------------------------------------------------------------
# 3. Parse Event protobufs and collect scalars (all steps).
# ---------------------------------------------------------------------------

def parse_all_scalars(records, from_step=None):
    """Parse TFRecord data blobs into {step: {tag: (wall_time, value)}}.

    Returns data for steps found in *records* so the caller can drip
    them out one step per poll cycle.

    Optimisation: records in a TensorBoard event file are chronological
    (steps increase monotonically).  When *from_step* is set, iteration
    starts from the **tail** of the record list and stops as soon as a
    step < from_step is encountered.  This avoids deserializing the
    bulk of the 1 MB buffer on every poll — only records at or after
    *from_step* are touched.

    The current step (= from_step) is always included so the caller can
    detect whether any steps beyond it exist (needed for the drip vs
    caught-up decision in the main block).

    Typical cost (steady-state, caught up):
        ~15 deserializations (one step's worth of tags).
    Cost after a flush that delivers N new steps:
        ~15 × (N + 1) deserializations.
    First run (from_step=None): all records are parsed (one-time cost).
    """
    try:
        from tensorboardX.proto.event_pb2 import Event
    except ImportError:
        print('[tb_metrics_plugin] tensorboardX not installed — cannot parse events',
              file=sys.stderr)
        return {}

    by_step = {}  # step -> {tag -> (wall_time, value)}
    for data in reversed(records):
        try:
            event = Event()
            event.ParseFromString(data)
        except Exception:
            continue  # skip partial/corrupt records (e.g. first record after seek)
        if not event.HasField('summary'):
            continue
        step = event.step
        # Early exit: events are chronological — everything before this
        # record is at an even older step.  We use strict '<' so that
        # from_step itself is included (needed for re-emit on idle).
        if from_step is not None and step < from_step:
            break
        for v in event.summary.value:
            tag = v.tag
            # Skip config text_summary entries (logged once at step 0).
            if tag.endswith('/text_summary'):
                continue
            # Only process scalar values.  Non-scalars (histograms, images,
            # audio, tensors) have simple_value defaulting to 0.0, which
            # would pollute Prometheus with false zero-valued metrics.
            try:
                if v.WhichOneof('value') != 'simple_value':
                    continue
            except ValueError:
                # Proto has no 'value' oneof (older tensorboardX version) —
                # fall back to accepting any non-zero simple_value, plus
                # zero only if no other oneof field is set.
                if v.simple_value == 0.0 and (v.image.ByteSize() > 0
                        or v.histo.ByteSize() > 0
                        or v.audio.ByteSize() > 0):
                    continue
            if step not in by_step:
                by_step[step] = {}
            by_step[step][tag] = (event.wall_time, v.simple_value)
    return by_step

# ---------------------------------------------------------------------------
# 4. Output Prometheus text format.
# ---------------------------------------------------------------------------

def _build_prom_lines(step_data, step, timestamp_ms, is_fill):
    """Build Prometheus exposition lines for a single step's scalars.

    Args:
        step_data: {tag: (wall_time, value)} for a single step.
        step: the training step number.
        timestamp_ms: Prometheus timestamp in milliseconds.
        is_fill: True for anti-staleness fill (staleness_fill=1),
                 False for real data (staleness_fill=0).
    Returns:
        list of str (Prometheus text lines).
    """
    lines = [
        f'# HELP tb_step Current training step.',
        f'# TYPE tb_step gauge',
        f'tb_step{{host="{HOST}"}} {step} {timestamp_ms}',
    ]

    emitted_help = set()
    for tag in sorted(step_data):
        _, value = step_data[tag]
        metric_name = 'tb_' + re.sub(r'[^a-zA-Z0-9_]', '_', tag)
        if metric_name not in emitted_help:
            lines.append(f'# HELP {metric_name} TensorBoard scalar: {tag}')
            lines.append(f'# TYPE {metric_name} gauge')
            emitted_help.add(metric_name)
        lines.append(f'{metric_name}{{host="{HOST}"}} {value} {timestamp_ms}')

    fill_val = 1 if is_fill else 0
    lines.append(f'# HELP tb_metrics_plugin_staleness_fill 1 during anti-staleness fills, 0 during real data.')
    lines.append(f'# TYPE tb_metrics_plugin_staleness_fill gauge')
    lines.append(f'tb_metrics_plugin_staleness_fill{{host="{HOST}"}} {fill_val} {timestamp_ms}')

    return lines


def _clamp_fill_ts(intended_ms):
    """Clamp a fill timestamp to stay within TSDB bounds.

    Only used for fills (staleness_fill=1).  Real events always
    retain their original wall_time (Rule 1).

    Ensures the fill timestamp is:
      1. >= intended_ms (the fill's natural anchor point)
      2. >  _last_prom_ts_ms (monotonic — prevents out-of-order)
      3. >= now - MAX_STALENESS_MS (prevents too-old rejection after
         TSDB compaction advances minValidTime)
    """
    now_ms = int(time.time() * 1000)
    return max(intended_ms, _last_prom_ts_ms + 1, now_ms - _MAX_STALENESS_MS)


def emit_prometheus(step_data, step):
    """Print Prometheus exposition text for a single step's scalars.

    Uses the event's original wall_time as the Prometheus custom
    timestamp — never modified (Rule 1).  If the wall_time is too
    old for Prometheus (below minValidTime), the sample is rejected
    by the TSDB; Rule 4 drops stale steps before they reach
    emission, so this only occurs in edge cases.

    Returns the wall_time_ms so the caller can update
    ``_last_prom_ts_ms`` (high-water mark only — never goes back).
    """
    if not step_data:
        return 0
    wall_time_ms = int(max(wt for wt, _ in step_data.values()) * 1000)
    sys.stdout.write('\n'.join(
        _build_prom_lines(step_data, step, wall_time_ms, is_fill=False)
    ) + '\n')
    return wall_time_ms


def emit_prometheus_fill(step_data, step, fill_time_ms=None):
    """Emit a fill: same data as emit_prometheus but with a synthetic
    timestamp and staleness_fill=1.

    Fill timestamps are clamped (Rule 2) to guarantee monotonicity
    and freshness — Prometheus must always accept fills so the series
    stays present and avoids staleness marking.

    Returns the timestamp_ms actually sent so the caller can update
    ``_last_prom_ts_ms``.
    """
    if not step_data:
        return 0
    if fill_time_ms is None:
        fill_time_ms = int(time.time() * 1000)
    timestamp_ms = _clamp_fill_ts(fill_time_ms)
    sys.stdout.write('\n'.join(
        _build_prom_lines(step_data, step, timestamp_ms, is_fill=True)
    ) + '\n')
    return timestamp_ms

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _locked_state(event_file, step, file_size=None, emit_ts=None,
                   prom_ts_ms=0):
    """Build STATE string for the locked state.

    Emits the 6-field format when *step* is set, using -1 as
    sentinel for ``None`` values.

    *emit_ts* — server time at last emission (for fill timer).
    *prom_ts_ms* — last Prometheus custom timestamp in ms
        (monotonically increasing; used to compute fill timestamps).
    """
    if step is not None:
        fs = file_size if file_size is not None else -1
        ets = int(emit_ts) if emit_ts is not None else -1
        ptm = int(prom_ts_ms) if prom_ts_ms else 0
        return f'# STATE {event_file}|{_filter_ts}|{step}|{fs}|{ets}|{ptm}'
    return f'# STATE {event_file}|{_filter_ts}'

_DIAG_ENABLED = os.environ.get('TB_PLUGIN_DIAG', '') == '1'
_DIAG_FILE = (os.environ.get('OUTPUT_PATH', '/outputs')
              + '/.tb_plugin_diag.log') if _DIAG_ENABLED else None

def _diag(msg):
    """Write diagnostic to NFS for remote debugging.

    No-op unless TB_PLUGIN_DIAG=1 is set in the environment.
    To enable: export TB_PLUGIN_DIAG=1 before starting the exporter.
    """
    if not _DIAG_ENABLED:
        return
    try:
        with open(_DIAG_FILE, 'a') as f:
            f.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
    except Exception:
        pass

_diag(f'plugin invoked, _PLUGIN_STATE present={bool(os.environ.get("_PLUGIN_STATE","").strip())}')

try:
    result = find_event_file()

    if result is _NEGATIVE_CACHED:
        _diag('NEGATIVE_CACHED')
        sys.exit(0)

    if result is None:
        _diag('NONE - no file found')
        print(f'# STATE __NONE__|{_filter_ts}|{int(time.time())}|{_wait_start_ts}')
        sys.exit(0)

    event_file = result
    _diag(f'event_file={event_file}, step={_last_emitted_step}, fsize={_last_file_size}, emit_ts={_last_emit_ts}, prom_ts={_last_prom_ts_ms}')

    # Seed _last_prom_ts_ms from _last_emit_ts when upgrading from an
    # older state format that didn't track the Prometheus timeline.
    # Without this, fills would compute next_fill_ts from epoch 0.
    if _last_prom_ts_ms == 0 and _last_emit_ts is not None:
        _last_prom_ts_ms = int(_last_emit_ts * 1000)

    # ---------------------------------------------------------------
    # File-size gate: if the event file hasn't grown since the last
    # poll, no new data has been flushed.  Skip the 1 MB tail read
    # and all protobuf parsing — output only STATE and let the
    # exporter replay the cached metric lines (prevents Prometheus
    # staleness without any plugin I/O).
    # ---------------------------------------------------------------
    try:
        current_size = os.path.getsize(event_file)
    except OSError:
        # File disappeared — re-enter waiting state.
        _wait_start_ts = int(time.time())
        print(f'# STATE __NONE__|{_filter_ts}|{int(time.time())}|{_wait_start_ts}')
        sys.exit(0)

    # --- File-size gate ---------------------------------------------------
    # When the event file hasn't grown, no new data has been flushed.
    # Normally, the exporter replays cached metrics and we exit early.
    # Exception: if the fill timer has expired (query-time staleness gap
    # forming), bypass the gate so the fill logic can run below.
    if (_last_file_size is not None
            and current_size == _last_file_size
            and _last_emitted_step is not None
            and not _fill_timer_expired()):
        # File unchanged, no fill needed.  Emit STATE only; exporter
        # replays the cached metric lines (prevents scrape-level
        # staleness).
        print(_locked_state(event_file, _last_emitted_step,
                            current_size, _last_emit_ts,
                            prom_ts_ms=_last_prom_ts_ms))
        sys.exit(0)

    try:
        records = read_tail_records(event_file)
    except OSError:
        # Locked file disappeared (job cleanup, NFS stale handle, etc.).
        # Re-enter waiting state with same filter_ts — a new file from
        # the current job (if any) will still pass the birth_ts filter.
        _wait_start_ts = int(time.time())
        print(f'# STATE __NONE__|{_filter_ts}|{int(time.time())}|{_wait_start_ts}')
        sys.exit(0)
    if not records:
        # File exists but no parseable records yet (e.g. just created,
        # first auto-flush hasn't happened).  Lock onto it.
        # Preserve emit_ts so the fill timer isn't accidentally reset.
        print(_locked_state(event_file, _last_emitted_step,
                            current_size, _last_emit_ts,
                            prom_ts_ms=_last_prom_ts_ms))
        sys.exit(0)

    by_step = parse_all_scalars(records, from_step=_last_emitted_step)
    if not by_step:
        # No scalars at all (first run with only text_summary, or
        # tensorboardX missing).  Lock without cursor.
        print(_locked_state(event_file, _last_emitted_step,
                            current_size, _last_emit_ts,
                            prom_ts_ms=_last_prom_ts_ms))
        sys.exit(0)

    sorted_steps = sorted(by_step.keys())

    # ---------------------------------------------------------------
    # Step-drip: emit one new step per poll cycle.
    #
    # When the event file flushes, several steps arrive at once.
    # Instead of jumping to the latest, we emit them one-by-one
    # (one per poll, ~10 s apart) so the curve is smooth in
    # dashboards.  Any backlog drains during the next idle period
    # (e.g. checkpoint).  If training is permanently faster than the
    # drip, the oldest pending steps may fall off the 1 MB read
    # window and be skipped (best-effort).
    #
    # First poll (or old state without drip): estimate drip capacity
    # from step_time and poll_interval.  If the backlog can be
    # drained before the next TB flush, start from the earliest
    # step (preserving initial metrics).  Otherwise, skip enough
    # early steps to avoid falling permanently behind.
    #
    # When caught up (no new steps after parse — file grew with
    # only non-scalar events): skip metric output, let exporter
    # replay cached metrics.  Exception: on the first poll with
    # file-size tracking (3→4 state upgrade), always emit to
    # seed the exporter's metric cache.
    # ---------------------------------------------------------------

    if _last_emitted_step is not None:
        # from_step is inclusive, so sorted_steps contains
        # _last_emitted_step plus any new steps.
        new_steps = [s for s in sorted_steps if s > _last_emitted_step]
        if new_steps:
            # --- Decide: fill or drip? --------------------------------
            #
            # Three rules govern the decision:
            #   1. Real events (staleness_fill=0) always use their own
            #      wall_time.  No timestamp clamping, ever.
            #   2. Fills (staleness_fill=1) must not land AFTER any
            #      pending real step's wall_time on the Prometheus
            #      timeline — that would shadow the real data.
            #   3. When the drip can't keep up (best-effort), real data
            #      has higher priority; fills are skipped.
            #
            # Implementation: attempt a fill if the emit timer says a
            # staleness gap has formed.  But only emit the fill if its
            # timestamp is strictly before the first pending real step's
            # wall_time (Rule 2).  Otherwise, fall through to drip
            # (Rule 3).  The drip always uses raw wall_times (Rule 1).

            do_fill = False
            if (_fill_timer_expired()
                    and _last_emitted_step in by_step):
                # Fill timestamp anchored to the Prometheus timeline,
                # not server time — keeps fills close to real
                # wall_times and minimises post-checkpoint data loss.
                next_fill_ts = (_last_prom_ts_ms / 1000.0
                                + _FILL_INTERVAL_S)
                # Rule 2: fill must land before the earliest pending
                # real step on the Prometheus timeline.
                first_real_wt = max(
                    wt for wt, _ in by_step[new_steps[0]].values())
                if next_fill_ts < first_real_wt:
                    do_fill = True

            if do_fill:
                fill_ms = int(next_fill_ts * 1000)
                actual_fill_ms = emit_prometheus_fill(
                    by_step[_last_emitted_step],
                    _last_emitted_step,
                    fill_time_ms=fill_ms)
                _last_prom_ts_ms = max(_last_prom_ts_ms, actual_fill_ms)
                # emit_size=None so file-size gate won't fire on next
                # poll — we need to keep reading for more fills or the
                # eventual real drip.
                print(_locked_state(event_file, _last_emitted_step,
                                    None, time.time(),
                                    prom_ts_ms=_last_prom_ts_ms))
            else:
                # Drip one real step (Rule 1: original wall_time).
                #
                # Rule 4: drop steps whose wall_time is stale — they
                # cannot be dripped timely and Prometheus would reject
                # them (below minValidTime after TSDB compaction).
                fresh = [s for s in new_steps if _is_fresh(by_step[s])]
                if fresh:
                    emit_step = fresh[0]
                    new_steps = fresh
                else:
                    # All pending steps are stale — emit the latest
                    # (least stale, best chance of Prometheus acceptance).
                    emit_step = new_steps[-1]
                    new_steps = [emit_step]

                # Gap handling: if the from_step reverse-iteration
                # optimisation broke due to a non-monotonic event
                # (e.g. step-0 config written mid-stream), fall back
                # to a full parse.  Only relevant when freshness
                # filtering didn't skip anything (otherwise the gap
                # is intentional from Rule 4).
                if (len(fresh) == len([s for s in sorted_steps
                                       if s > _last_emitted_step])
                        and emit_step > _last_emitted_step + 1):
                    by_step_full = parse_all_scalars(records,
                                                     from_step=None)
                    if by_step_full:
                        ss_full = sorted(by_step_full.keys())
                        ns_full = [s for s in ss_full
                                   if s > _last_emitted_step]
                        if ns_full and ns_full[0] <= _last_emitted_step + 1:
                            by_step = by_step_full
                            sorted_steps = ss_full
                            new_steps = [s for s in ns_full
                                         if _is_fresh(by_step[s])] or ns_full[-1:]
                            emit_step = new_steps[0]
                # Cache file size ONLY when this is the last pending
                # step.  If more remain, omit size so the gate won't
                # fire — the drip must keep reading to advance.
                emit_size = (current_size
                             if len(new_steps) == 1 else None)
                actual_ts_ms = emit_prometheus(by_step[emit_step],
                                               emit_step)
                _last_prom_ts_ms = max(_last_prom_ts_ms, actual_ts_ms)
                print(_locked_state(event_file, emit_step, emit_size,
                                    time.time(),
                                    prom_ts_ms=_last_prom_ts_ms))
        else:
            # No new steps beyond _last_emitted_step.
            # Check if a pure fill is needed (idle gap forming).
            if (_fill_timer_expired()
                    and _last_emitted_step in by_step):
                # Pure fill: no pending real step, so Rule 2 is
                # trivially satisfied (nothing to conflict with).
                next_fill_ts = (_last_prom_ts_ms / 1000.0
                                + _FILL_INTERVAL_S)
                fill_ms = int(next_fill_ts * 1000)
                actual_fill_ms = emit_prometheus_fill(
                    by_step[_last_emitted_step],
                    _last_emitted_step,
                    fill_time_ms=fill_ms)
                _last_prom_ts_ms = max(_last_prom_ts_ms, actual_fill_ms)
                # Re-arm gate so we don't re-read the file every
                # poll between fill bursts.
                print(_locked_state(event_file, _last_emitted_step,
                                    current_size, time.time(),
                                    prom_ts_ms=_last_prom_ts_ms))
            elif _last_file_size is not None:
                # Normal steady state: exporter has cached metrics.
                # Skip metric output — exporter replays cache.
                print(_locked_state(event_file, _last_emitted_step,
                                    current_size, _last_emit_ts,
                                    prom_ts_ms=_last_prom_ts_ms))
                sys.exit(0)
            else:
                # First poll with file-size tracking (state upgrade
                # from 3→4→5-field).  Must emit once to seed the
                # exporter's metric cache; subsequent idle polls use
                # file-size gate + replay.  Re-emitting the current
                # step is a no-op for Prometheus (same wall_time +
                # value = deduplicated).
                emit_step = _last_emitted_step
                emit_size = current_size
                ats = emit_prometheus(by_step[emit_step], emit_step)
                _last_prom_ts_ms = max(_last_prom_ts_ms, ats)
                print(_locked_state(event_file, emit_step, emit_size,
                                    time.time(),
                                    prom_ts_ms=_last_prom_ts_ms))
    else:
        # First run (or upgraded from old state).
        #
        # Rule 4: drop stale steps, then estimate drip capacity on
        # the remaining fresh steps.
        fresh = [s for s in sorted_steps if _is_fresh(by_step[s])]
        candidates = fresh if fresh else sorted_steps[-1:]

        # Estimate how many steps we can drip before the next TB flush
        # to decide whether to start from the beginning or jump ahead.
        emit_step = candidates[-1]  # default: jump to latest
        if len(candidates) >= 2:
            wt_vals = [wt for s in candidates
                       for wt, _ in by_step[s].values()]
            step_range = candidates[-1] - candidates[0]
            if step_range > 0 and wt_vals:
                step_time = (max(wt_vals) - min(wt_vals)) / step_range
                flush_budget = len(candidates) * step_time
                poll_interval = float(
                    os.environ.get('POLL_INTERVAL', 10))
                if poll_interval > 0 and flush_budget > 0:
                    drip_capacity = int(flush_budget / poll_interval)
                    skip = max(0, len(candidates)
                               - max(1, drip_capacity))
                    emit_step = candidates[skip]
        emit_size = (current_size
                     if emit_step == sorted_steps[-1] else None)
        ats = emit_prometheus(by_step[emit_step], emit_step)
        _last_prom_ts_ms = max(_last_prom_ts_ms, ats)
        print(_locked_state(event_file, emit_step, emit_size,
                            time.time(),
                            prom_ts_ms=_last_prom_ts_ms))
except Exception as exc:
    import traceback
    _diag(f'EXCEPTION: {exc}\n{traceback.format_exc()}')
    print(f'[tb_metrics_plugin] ERROR: {exc}', file=sys.stderr)
    _saved = os.environ.get('_PLUGIN_STATE', '').strip()
    if _saved:
        print(f'# STATE {_saved}')

PYEOF
