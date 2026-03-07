#!/usr/bin/env python3
"""
host-cmd server — runs on the HOST outside the container.

Watches .host-cmd/queue/ for job files, executes commands on the host,
writes results to .host-cmd/results/.  The directory where this script
lives is the shared directory — no config needed.

Usage:
    python3 host_cmd_server.py                        # foreground
    python3 host_cmd_server.py --poll-interval 0.5    # faster polling
    ./host_cmd_ctl.sh start                           # managed start
"""

import argparse
import asyncio
import fcntl
import json
import logging
import logging.handlers
import os
import re
import signal
import sys
import time
from pathlib import Path

HOST_CMD_DIR = Path(__file__).resolve().parent
SHARED_ROOT = HOST_CMD_DIR.parent
QUEUE_DIR = HOST_CMD_DIR / "queue"
RUNNING_DIR = HOST_CMD_DIR / "running"
RESULTS_DIR = HOST_CMD_DIR / "results"
PID_FILE = HOST_CMD_DIR / "daemon.pid"
LOCK_FILE = HOST_CMD_DIR / "daemon.lock"
POLICY_FILE = HOST_CMD_DIR / "policy.json"
POLICY_DEFAULT = HOST_CMD_DIR / "policy.json.default"

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_FILE = HOST_CMD_DIR / "host_cmd_server.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 3

_log = logging.getLogger("host-cmd")
_log.setLevel(logging.INFO)
_formatter = logging.Formatter(LOG_FMT)
_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
)
_file_handler.setFormatter(_formatter)
_log.addHandler(_file_handler)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_log.addHandler(_console_handler)

DEFAULT_TIMEOUT = 600
MAX_CONCURRENT = 16
PING_COMMAND = "__ping__"
STALE_CMD_S = 60  # discard queued commands older than this


# ---------------------------------------------------------------------------
# Policy — allow / deny patterns
# ---------------------------------------------------------------------------
#
# Evaluation order:
#   1. max_command_length — reject if too long
#   2. deny_patterns  — reject if any pattern matches
#   3. allow_patterns — if non-empty, reject unless at least one matches
#
# Loaded once at startup, immutable after that.

_policy: dict = {}


def load_policy():
    """Load policy once into _policy. Called at server startup only."""
    global _policy
    path = POLICY_FILE if POLICY_FILE.exists() else POLICY_DEFAULT
    if path.exists():
        try:
            _policy = json.loads(path.read_text())
            _log.info("Loaded policy from %s", path)
            deny = _policy.get("deny_patterns", [])
            allow = _policy.get("allow_patterns", [])
            max_len = _policy.get("max_command_length", "unlimited")
            _log.info("  max_command_length: %s", max_len)
            _log.info("  deny_patterns (%d):", len(deny))
            for p in deny:
                _log.info("    - %s", p)
            _log.info("  allow_patterns (%d):%s", len(allow),
                       " (all non-denied allowed)" if not allow else "")
            for p in allow:
                _log.info("    + %s", p)
            return
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("Failed to load %s: %s", path, exc)
    _policy = {}
    _log.warning("No policy file found — all commands allowed")


def check_policy(cmd: str) -> str | None:
    """Return a rejection reason if *cmd* violates policy, else None."""
    policy = _policy

    max_len = policy.get("max_command_length", 8192)
    if len(cmd) > max_len:
        return f"Command too long ({len(cmd)} > {max_len} chars)"

    for pattern in policy.get("deny_patterns", []):
        try:
            if re.search(pattern, cmd):
                return f"Denied by pattern: {pattern!r}"
        except re.error:
            _log.warning("Bad regex in deny_patterns: %s", pattern)

    allow_patterns = policy.get("allow_patterns", [])
    if allow_patterns:
        for pattern in allow_patterns:
            try:
                if re.search(pattern, cmd):
                    return None
            except re.error:
                _log.warning("Bad regex in allow_patterns: %s", pattern)
        return "Not matched by any allow_patterns"

    return None


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def _write_result(job_id: str, result: dict):
    """Atomic write of result file."""
    tmp_result = RESULTS_DIR / f".{job_id}.tmp"
    result_file = RESULTS_DIR / f"{job_id}.json"
    tmp_result.write_text(json.dumps(result, indent=2))
    tmp_result.rename(result_file)


async def execute_job(job_file: Path, sem: asyncio.Semaphore):
    """Pick up a .cmd file, execute the command, write a .result file."""
    job_id = job_file.stem
    async with sem:
        try:
            raw = job_file.read_text()
            job = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            _log.error("Bad job file %s: %s", job_file, exc)
            job_file.unlink(missing_ok=True)
            return

        cmd = job.get("command", "")
        submitted_at = job.get("submitted_at", 0)
        age = time.time() - submitted_at

        if cmd != PING_COMMAND and age > STALE_CMD_S:
            _log.warning("STALE  [%s] %.0fs old, discarding: %s",
                         job_id[:8], age, cmd[:80])
            _write_result(job_id, {
                "id": job_id,
                "command": cmd,
                "exit_code": -5,
                "stdout": "",
                "stderr": f"Command discarded: submitted {age:.0f}s ago (limit {STALE_CMD_S}s)",
                "timed_out": False,
                "started_at": time.time(),
                "elapsed_s": 0,
                "stale": True,
            })
            job_file.unlink(missing_ok=True)
            return

        if cmd == PING_COMMAND:
            _write_result(job_id, {
                "id": job_id,
                "command": PING_COMMAND,
                "exit_code": 0,
                "stdout": "pong",
                "stderr": "",
                "timed_out": False,
                "started_at": time.time(),
                "elapsed_s": 0,
                "pong": True,
            })
            job_file.unlink(missing_ok=True)
            return

        timeout = job.get("timeout", DEFAULT_TIMEOUT)
        env_override = job.get("env")
        relative_cwd = job.get("cwd")

        rejection = check_policy(cmd)
        if rejection:
            _log.warning("REJECT [%s] %s — %s", job_id[:8], cmd[:80], rejection)
            _write_result(job_id, {
                "id": job_id,
                "command": cmd,
                "exit_code": -3,
                "stdout": "",
                "stderr": f"POLICY VIOLATION: {rejection}",
                "timed_out": False,
                "started_at": time.time(),
                "elapsed_s": 0,
                "rejected": True,
            })
            job_file.unlink(missing_ok=True)
            return

        running_file = RUNNING_DIR / job_file.name
        try:
            job_file.rename(running_file)
        except FileNotFoundError:
            return

        if relative_cwd:
            work_dir = str(SHARED_ROOT / relative_cwd)
        else:
            work_dir = str(SHARED_ROOT)

        _log.info("START  [%s] cwd=%s %s", job_id[:8], work_dir, cmd[:120])
        started = time.time()

        env = os.environ.copy()
        if env_override and isinstance(env_override, dict):
            env.update(env_override)

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=work_dir,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            exit_code = proc.returncode
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout, stderr = "", f"Command timed out after {timeout}s"
            exit_code = -1
            timed_out = True
        except Exception as exc:
            stdout, stderr = "", str(exc)
            exit_code = -2
            timed_out = False

        elapsed = round(time.time() - started, 3)
        _log.info("DONE   [%s] exit=%s elapsed=%.1fs", job_id[:8], exit_code, elapsed)

        _write_result(job_id, {
            "id": job_id,
            "command": cmd,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "started_at": started,
            "elapsed_s": elapsed,
        })

        running_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def recover_orphaned_running():
    orphans = list(RUNNING_DIR.glob("*.cmd"))
    for f in orphans:
        _log.warning("Removing orphaned running job: %s", f.name)
        f.unlink(missing_ok=True)
    if orphans:
        _log.info("Cleaned up %d orphaned running jobs", len(orphans))


# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------

async def poll_loop(interval: float):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    known: set[str] = set()
    _log.info("Server started — watching %s (poll every %.1fs)", QUEUE_DIR, interval)

    while True:
        try:
            cmd_files = sorted(QUEUE_DIR.glob("*.cmd"))
        except OSError:
            cmd_files = []

        for f in cmd_files:
            if f.name not in known:
                known.add(f.name)
                asyncio.create_task(execute_job(f, sem))

        if len(known) > 1000:
            existing = {f.name for f in QUEUE_DIR.glob("*.cmd")}
            existing |= {f.name for f in RUNNING_DIR.glob("*.cmd")}
            known &= existing

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Singleton lock — one server per directory
# ---------------------------------------------------------------------------

_lock_fd = None


def acquire_lock() -> bool:
    global _lock_fd
    LOCK_FILE.touch(exist_ok=True)
    _lock_fd = open(LOCK_FILE, "r+")
    try:
        fcntl.lockf(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _lock_fd.close()
        _lock_fd = None
        return False
    _lock_fd.seek(0)
    _lock_fd.truncate()
    _lock_fd.write(f"{os.getpid()}\n")
    _lock_fd.flush()
    return True


def read_existing_server_info() -> str:
    try:
        content = LOCK_FILE.read_text().strip()
        return f"PID {content}" if content else "unknown PID"
    except (OSError, ValueError):
        return "unknown"


def cleanup(_sig=None, _frame=None):
    PID_FILE.unlink(missing_ok=True)
    if _lock_fd:
        _lock_fd.close()
    LOCK_FILE.unlink(missing_ok=True)
    _log.info("Server stopped.")
    sys.exit(0)


def _is_in_container() -> bool:
    try:
        return "container" in Path("/proc/1/cgroup").read_text()
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="host-cmd server")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between queue polls (default: 0.5)",
    )
    args = parser.parse_args()

    if Path("/.dockerenv").exists() or _is_in_container():
        _log.error(
            "Refusing to start inside a container. "
            "The server must run on the host. Exiting."
        )
        sys.exit(1)

    for d in (QUEUE_DIR, RUNNING_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    recover_orphaned_running()
    load_policy()

    if not acquire_lock():
        info = read_existing_server_info()
        _log.error(
            "Another server is already running (%s). Exiting.\n"
            "If the other server is dead, its lock will have been "
            "auto-released. If you see this after a crash, check "
            "with host_cmd_ctl.sh status or remove %s manually.",
            info,
            LOCK_FILE,
        )
        sys.exit(1)

    PID_FILE.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    _log.info("Working dir: %s", HOST_CMD_DIR)
    try:
        asyncio.run(poll_loop(args.poll_interval))
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
