#!/usr/bin/env python3
"""
host-cmd client — run commands on the host from inside a container.

No config needed. This script and the server share the same directory
via a shared filesystem (e.g. NFS).

Usage as a CLI:
    python3 host_cmd.py "squeue"
    python3 host_cmd.py --async "srun ..."
    python3 host_cmd.py --result <job-id>
    python3 host_cmd.py --status
    python3 host_cmd.py --ping
    python3 host_cmd.py --policy
    python3 host_cmd.py --deny '\\bsudo\\b'
    python3 host_cmd.py --cleanup

Usage as a library:
    from host_cmd import HostBridge
    br = HostBridge()
    br.ping()  # True / False
    result = br.run("squeue")
    print(result["stdout"])
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
PING_COMMAND = "__ping__"


class ServerNotRunningError(RuntimeError):
    """Raised when the host-cmd server doesn't respond to ping."""


class HostBridge:
    def __init__(self, host_cmd_dir: str | Path | None = None):
        self.dir = Path(host_cmd_dir).resolve() if host_cmd_dir else _SELF_DIR
        self.queue_dir = self.dir / "queue"
        self.results_dir = self.dir / "results"
        self.shared_root = self.dir.parent

    # ------------------------------------------------------------------
    # Ping
    # ------------------------------------------------------------------

    def ping(self, timeout: float = 3.0) -> bool:
        """Send a ping through the queue and wait for a pong."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        job_file = self.queue_dir / f"{job_id}.cmd"
        job_file.write_text(json.dumps({
            "id": job_id,
            "command": PING_COMMAND,
            "timeout": int(timeout),
            "submitted_at": time.time(),
        }))

        result_file = self.results_dir / f"{job_id}.json"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if result_file.exists():
                try:
                    data = json.loads(result_file.read_text())
                    result_file.unlink(missing_ok=True)
                    if data.get("pong"):
                        return True
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(0.1)

        job_file.unlink(missing_ok=True)
        return False

    def _require_server(self):
        """Ping the server before every command."""
        if not self.ping():
            raise ServerNotRunningError(
                "host-cmd server is not responding. "
                "Start it on the host with:\n"
                "  cd .host-cmd && ./host_cmd_ctl.sh start"
            )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def submit(
        self,
        command: str,
        timeout: int = 300,
        env: dict | None = None,
        cwd: str | None = None,
    ) -> str:
        """Submit a command for async execution. Returns job ID."""
        self._require_server()
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        job: dict = {
            "id": job_id,
            "command": command,
            "timeout": timeout,
            "submitted_at": time.time(),
        }
        if env:
            job["env"] = env
        if cwd:
            job["cwd"] = cwd
        job_file = self.queue_dir / f"{job_id}.cmd"
        job_file.write_text(json.dumps(job))
        return job_id

    def poll(self, job_id: str, timeout: float = 120, interval: float = 0.3) -> dict | None:
        """Poll for a result. Returns result dict or None on timeout.

        Re-pings every 5s while the .cmd file is still queued (not picked up).
        If the server died after submit, fails fast instead of blocking.
        """
        result_file = self.results_dir / f"{job_id}.json"
        cmd_file = self.queue_dir / f"{job_id}.cmd"
        deadline = time.time() + timeout
        last_ping = time.time()
        while time.time() < deadline:
            if result_file.exists():
                try:
                    return json.loads(result_file.read_text())
                except json.JSONDecodeError:
                    time.sleep(0.1)
                    continue
            if cmd_file.exists() and (time.time() - last_ping) > 5:
                if not self.ping(timeout=2.0):
                    cmd_file.unlink(missing_ok=True)
                    return {
                        "id": job_id,
                        "exit_code": -4,
                        "stdout": "",
                        "stderr": "Server died — command was never picked up",
                    }
                last_ping = time.time()
            time.sleep(interval)
        return None

    def run(
        self,
        command: str,
        timeout: int = 300,
        poll_timeout: float = 0,
        cwd: str | None = None,
    ) -> dict:
        """Submit and wait for result (synchronous)."""
        effective_poll = poll_timeout if poll_timeout > 0 else timeout + 10
        job_id = self.submit(command, timeout=timeout, cwd=cwd)
        result = self.poll(job_id, timeout=effective_poll)
        if result is None:
            return {
                "id": job_id,
                "error": f"Timed out waiting for result after {effective_poll}s",
                "command": command,
            }
        return result

    def result(self, job_id: str) -> dict | None:
        """Check if a result exists (non-blocking)."""
        result_file = self.results_dir / f"{job_id}.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        return None

    def status(self) -> dict:
        """Check server status via ping."""
        alive = self.ping(timeout=3.0)
        pid_file = self.dir / "daemon.pid"
        pending = list(self.queue_dir.glob("*.cmd")) if self.queue_dir.exists() else []
        running_dir = self.dir / "running"
        running = list(running_dir.glob("*.cmd")) if running_dir.exists() else []
        completed = list(self.results_dir.glob("*.json")) if self.results_dir.exists() else []
        return {
            "dir": str(self.dir),
            "server_alive": alive,
            "pid": pid_file.read_text().strip() if pid_file.exists() else None,
            "pending": len(pending),
            "running": len(running),
            "completed": len(completed),
        }

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def _load_policy(self) -> dict:
        pf = self.dir / "policy.json"
        if pf.exists():
            return json.loads(pf.read_text())
        return {"deny_patterns": [], "allow_patterns": [], "max_command_length": 8192}

    def _save_policy(self, policy: dict):
        pf = self.dir / "policy.json"
        pf.write_text(json.dumps(policy, indent=2) + "\n")

    def policy_list(self) -> dict:
        pf = self.dir / "policy.json"
        if pf.exists():
            return {"source": str(pf), **json.loads(pf.read_text())}
        return {"source": "none"}

    def policy_add(self, pattern: str, kind: str = "deny"):
        policy = self._load_policy()
        key = f"{kind}_patterns"
        patterns = policy.get(key, [])
        if pattern not in patterns:
            patterns.append(pattern)
        policy[key] = patterns
        self._save_policy(policy)

    def policy_remove(self, pattern: str, kind: str = "deny"):
        policy = self._load_policy()
        key = f"{kind}_patterns"
        patterns = policy.get(key, [])
        patterns = [p for p in patterns if p != pattern]
        policy[key] = patterns
        self._save_policy(policy)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, max_age_hours: float | None = None):
        removed = 0
        if not self.results_dir.exists():
            return 0
        cutoff = time.time() - (max_age_hours * 3600) if max_age_hours else None
        for pattern in ("*.json", ".*.tmp"):
            for f in self.results_dir.glob(pattern):
                try:
                    if cutoff is None or f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
                        removed += 1
                except OSError:
                    pass
        return removed


def main():
    parser = argparse.ArgumentParser(description="host-cmd client")
    parser.add_argument("command", nargs="*", help="Command to execute")
    parser.add_argument("--async", dest="async_mode", action="store_true",
                        help="Async mode: print job ID and return immediately")
    parser.add_argument("--result", metavar="JOB_ID",
                        help="Retrieve result for a job ID")
    parser.add_argument("--status", action="store_true",
                        help="Check server status")
    parser.add_argument("--ping", dest="ping_mode", action="store_true",
                        help="Ping the server and report liveness")
    parser.add_argument("--cleanup", nargs="?", type=float, const=0, default=None,
                        metavar="HOURS",
                        help="Delete results older than HOURS (default: delete all)")
    parser.add_argument("--policy", action="store_true",
                        help="Show current policy")
    parser.add_argument("--deny", metavar="PATTERN",
                        help="Add a deny pattern")
    parser.add_argument("--allow", metavar="PATTERN",
                        help="Add an allow pattern")
    parser.add_argument("--undeny", metavar="PATTERN",
                        help="Remove a deny pattern")
    parser.add_argument("--unallow", metavar="PATTERN",
                        help="Remove an allow pattern")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Command timeout in seconds (default: 300)")
    parser.add_argument("--cwd", default=None,
                        help="Working dir relative to shared root (e.g. 'scripts/')")
    parser.add_argument("--json", dest="json_output", action="store_true",
                        help="Output raw JSON")
    args = parser.parse_args()

    br = HostBridge()

    if args.ping_mode:
        alive = br.ping()
        print("ALIVE" if alive else "NOT RESPONDING")
        sys.exit(0 if alive else 1)

    if args.cleanup is not None:
        hours = args.cleanup if args.cleanup > 0 else None
        removed = br.cleanup(max_age_hours=hours)
        label = f"older than {hours}h" if hours else "all"
        print(f"Removed {removed} result files ({label})")
        return

    if args.deny:
        br.policy_add(args.deny, kind="deny")
        print(f"Added deny pattern: {args.deny}")
        return

    if args.allow:
        br.policy_add(args.allow, kind="allow")
        print(f"Added allow pattern: {args.allow}")
        return

    if args.undeny:
        br.policy_remove(args.undeny, kind="deny")
        print(f"Removed deny pattern: {args.undeny}")
        return

    if args.unallow:
        br.policy_remove(args.unallow, kind="allow")
        print(f"Removed allow pattern: {args.unallow}")
        return

    if args.policy:
        p = br.policy_list()
        source = p.pop("source")
        print(f"Source: {source}")
        deny = p.get("deny_patterns", [])
        allow = p.get("allow_patterns", [])
        max_len = p.get("max_command_length", "default")
        print(f"Max command length: {max_len}")
        print(f"\nDeny patterns ({len(deny)}):")
        for pat in deny:
            print(f"  - {pat}")
        print(f"\nAllow patterns ({len(allow)}):")
        if allow:
            for pat in allow:
                print(f"  + {pat}")
        else:
            print("  (empty — all non-denied commands allowed)")
        return

    if args.status:
        s = br.status()
        if args.json_output:
            print(json.dumps(s, indent=2))
        else:
            alive_str = "ALIVE" if s["server_alive"] else "NOT RUNNING"
            print(f"Server: {alive_str}  (PID: {s['pid'] or '?'})")
            print(f"Pending: {s['pending']}  Running: {s['running']}  Completed: {s['completed']}")
        return

    if args.result:
        r = br.result(args.result)
        if r:
            if args.json_output:
                print(json.dumps(r, indent=2))
            else:
                print(f"exit_code: {r.get('exit_code')}")
                if r.get("stdout"):
                    print(r["stdout"], end="")
                if r.get("stderr"):
                    print(r["stderr"], end="", file=sys.stderr)
        else:
            print(f"No result yet for {args.result}")
            sys.exit(1)
        return

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmd = " ".join(args.command)

    try:
        if args.async_mode:
            job_id = br.submit(cmd, timeout=args.timeout, cwd=args.cwd)
            print(job_id)
            return

        result = br.run(cmd, timeout=args.timeout, cwd=args.cwd)
    except ServerNotRunningError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        if result.get("stdout"):
            print(result["stdout"], end="")
        if result.get("stderr"):
            print(result["stderr"], end="", file=sys.stderr)
        sys.exit(result.get("exit_code", 1))


if __name__ == "__main__":
    main()
