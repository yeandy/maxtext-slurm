#!/usr/bin/env python3
"""Parse MaxText training logs and rename them with TGS (Tokens/GPU/Second) metrics.

Extracts Tokens/s/device values from .log (or legacy .out) log files,
computes statistics, and renames files (and their corresponding job folders)
to include TGS in the filename.

Usage:
    tgs_tagger.py [OPTIONS] [PATH ...]

See --help for full details.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── ANSI colors ──────────────────────────────────────────────────────────────

RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ── Regex patterns ───────────────────────────────────────────────────────────

RE_TOKENS = re.compile(r"Tokens/s/device:\s*([0-9]+(?:\.[0-9]+)?)")
RE_STEP_TGS = re.compile(
    r"completed step:\s*(\d+),.*?Tokens/s/device:\s*([0-9]+(?:\.[0-9]+)?)"
)
RE_QUANT = re.compile(r"Config param quantization:\s*(.*)")
RE_PDBS = re.compile(r"Config param per_device_batch_size:\s*([0-9]+(?:\.[0-9]+)?)")
RE_TGS_TAG = re.compile(r"(.*[-_]TGS_)[0-9]+(\.[0-9]+)?$")
RE_JOB_ID = re.compile(r"^([^-]+)-")
RE_JOB_SUMMARY = re.compile(r"^=+ JOB SUMMARY =+", re.MULTILINE)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _report_na(label: str, msg: str, need: int | str = 0) -> None:
    """Print the standard NA output block used during file discovery errors."""
    print(f"Parsed {label}")
    print(f"  TGS=NA, std=NA, n=0/{need}")
    print(f"  {RED}ERROR: {msg}{RESET}")
    print()


def _find_job_folder(directory: Path, job_id: str) -> Path | None:
    """Return the first directory in *directory* whose name starts with ``job_id-``."""
    for entry in sorted(directory.iterdir()):
        if entry.is_dir() and entry.name.startswith(f"{job_id}-"):
            return entry
    return None


def _cleanup_prompt(file: Path, cleanup: bool) -> None:
    """In cleanup mode, prompt to delete *file* and its job folder."""
    if not cleanup:
        return

    directory = file.parent
    job_id_m = RE_JOB_ID.match(file.name)

    if job_id_m:
        job_id = job_id_m.group(1)
        job_folder = _find_job_folder(directory, job_id)

        if job_folder:
            print(f"  Found job folder: {job_folder.name}")
            ans = input("  Delete this file and its job folder? [y/N] ")
            if ans.strip().lower() == "y":
                file.unlink()
                try:
                    shutil.rmtree(job_folder)
                except PermissionError:
                    # Fall back to sudo rm like the bash version
                    subprocess.run(
                        ["sudo", "rm", "-rf", str(job_folder)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                print("  Deleted file and job folder")
            return

    # No job folder found (or no job_id in name)
    ans = input("  Delete this file? [y/N] ")
    if ans.strip().lower() == "y":
        file.unlink()


def _is_job_running(text: str, file: Path) -> bool:
    """Detect whether the job that produced this log is still running.

    Detection logic:
      1. JOB SUMMARY block present → job finished.  Written by an EXIT trap
         that fires on normal exit, error exit, and SIGTERM (scancel).
         Only SIGKILL (OOM killer, kill -9) bypasses the trap.
      2. No JOB SUMMARY + file modified within the last 900 s (15 min) →
         likely still running.  The generous window accommodates long
         checkpoint writes (~10 min for 70b+ models) plus step time.
      3. No JOB SUMMARY + file is stale → hard-killed or orphaned; treat
         as finished so it can still be renamed.
    """
    # JOB SUMMARY is the definitive "done" signal.
    if RE_JOB_SUMMARY.search(text):
        return False

    # No summary — check how recently the file was written to.
    # 900 s (15 min) accommodates long checkpoint writes (70b+ models
    # can take ~10 min) plus step time and I/O variance.
    try:
        age = time.time() - file.stat().st_mtime
        if age < 900:
            return True
    except OSError:
        pass

    return False


# ── Constants ────────────────────────────────────────────────────────────────

# Steps 0‑4 are unstable (XLA compilation, profiler warm-up, etc.)
WARMUP_STEPS = 5
# Steps 5‑14 form the steady-state measurement window (10 steps)
STEADY_STATE_STEPS = 10
# Recommended minimum: WARMUP_STEPS + STEADY_STATE_STEPS = 15
RECOMMENDED_MIN_STEPS = WARMUP_STEPS + STEADY_STATE_STEPS
# Minimum steady-state data points required before renaming
MIN_STEADY_FOR_RENAME = 5

# ── Extraction & stats ───────────────────────────────────────────────────────


def _extract_quantization(text: str) -> str:
    """Extract and sanitize quantization value from log text."""
    m = RE_QUANT.search(text)
    if not m:
        return ""
    raw = m.group(1).strip()
    # Handle enum format: QuantizationType.FP8 → FP8
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    # Keep only filename-safe characters (alphanumeric, hyphen, underscore)
    raw = re.sub(r"[^A-Za-z0-9_-]", "", raw)
    # Filter out non-quantization defaults (class names with no actual quant type)
    if not raw or raw in ("NONE", "IntQuantization"):
        return ""
    return raw


def _extract_per_device_batch_size(text: str) -> str:
    """Extract per_device_batch_size from log text."""
    m = RE_PDBS.search(text)
    return m.group(1) if m else ""


def _compute_stats(values: list[float]) -> tuple[float, float, int]:
    """Compute TGS stats on the provided values.

    Returns (mean, std, n).
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(values) / n
    var = max(sum(v * v for v in values) / n - mean * mean, 0.0)
    sd = math.sqrt(var)
    return mean, sd, n


def _select_steady_state(all_values: list[float], num_nodes: int = 1) -> list[float]:
    """Return the steady-state measurement window from all step values.

    Multi-node jobs log one value per node per step, so each logical step
    occupies ``num_nodes`` consecutive entries.  Skips the first
    ``WARMUP_STEPS * num_nodes`` values (steps 0‑4) and takes up to
    ``STEADY_STATE_STEPS * num_nodes`` values after that (steps 5‑14).
    """
    skip = WARMUP_STEPS * num_nodes
    take = STEADY_STATE_STEPS * num_nodes
    after_warmup = all_values[skip:]
    return after_warmup[:take]


# ── Core: process one file ───────────────────────────────────────────────────


_LOG_SUFFIXES = {".log", ".out"}


def process_file(file: Path, cleanup: bool, force: bool = False) -> int:
    """Process a single .log/.out file. Returns an exit-status int."""

    print(f"Parsed {file}")

    def fail(msg: str, need: int | str = 0, rc: int = 1) -> int:
        print(f"  TGS=NA, std=NA, n=0/{need}")
        print(f"  {RED}ERROR: {msg}{RESET}")
        print()
        return rc

    # ── validation ──
    if not file.exists():
        return fail("file does not exist")
    if not file.is_file():
        return fail("not a regular file")
    if file.suffix not in _LOG_SUFFIXES:
        return fail("not a .log/.out file (skipping)")

    # ── read file ──
    text = file.read_text(errors="replace")
    lines = text.splitlines()

    # ── num_nodes from header (KEY=VALUE format) ──
    num_nodes = None
    for line in lines[:20]:
        m = re.match(r"^(?:NUM_NODES|NNODES|SLURM_JOB_NUM_NODES)\s*=\s*(\d+)", line)  # NNODES/SLURM_* for old logs
        if m:
            num_nodes = int(m.group(1))
            break
    if num_nodes is None:
        return fail("NUM_NODES=<int> not found in log header", rc=3)

    # ── check if job is still running (used by rename and cleanup guards) ──
    running = _is_job_running(text, file)

    # ── extract all Tokens/s/device values and step numbers ──
    matches = [(int(m.group(1)), float(m.group(2))) for m in RE_STEP_TGS.finditer(text)]
    if not matches:
        # Fallback: extract TGS values without step numbers
        all_values = [float(m.group(1)) for m in RE_TOKENS.finditer(text)]
        all_step_nums = list(range(len(all_values)))
    else:
        all_step_nums = [s for s, _ in matches]
        all_values = [v for _, v in matches]
    total_steps = len(all_values)

    if total_steps == 0:
        print(f"  TGS=NA, std=NA, n=0/{RECOMMENDED_MIN_STEPS}")
        print(f"  {RED}ERROR: no Tokens/s/device lines found{RESET}")
        if running and not force:
            print(f"  {YELLOW}(job appears to still be running, use -f to override){RESET}")
        else:
            _cleanup_prompt(file, cleanup)
        print()
        return 2

    # ── deduplicate step numbers to get unique logical steps ──
    # Multi-node jobs log one value per node per step.
    unique_steps = sorted(set(all_step_nums))

    # ── partition into warmup / steady / tail windows ──
    # Scale windows by num_nodes (each logical step has num_nodes entries).
    warmup_n = WARMUP_STEPS * num_nodes
    steady_n = STEADY_STATE_STEPS * num_nodes
    warmup = all_values[:warmup_n]
    steady = _select_steady_state(all_values, num_nodes)
    tail_start = warmup_n + steady_n
    tail = all_values[tail_start:] if total_steps > tail_start else []

    # ── compute stats for each window ──
    w_mean, w_sd, w_n = _compute_stats(warmup)
    s_mean, s_sd, s_n = _compute_stats(steady)
    a_mean, a_sd, a_n = _compute_stats(all_values)

    # ── derive actual step ranges from the data ──
    def _step_range(start_idx: int, end_idx: int) -> str:
        """Return 'step_first-step_last' from unique_steps by position."""
        if start_idx >= len(unique_steps):
            return "?"
        first = unique_steps[start_idx]
        last = unique_steps[min(end_idx, len(unique_steps) - 1)]
        return f"{first}-{last}" if first != last else str(first)

    warmup_range = _step_range(0, WARMUP_STEPS - 1)
    s_logical = s_n // num_nodes
    steady_range = _step_range(WARMUP_STEPS, WARMUP_STEPS + s_logical - 1) if s_n > 0 else "—"

    # ── print TGS: all first, then breakdown ──
    total_logical = total_steps // num_nodes
    all_range = _step_range(0, len(unique_steps) - 1)
    print(
        f"  all     TGS={a_mean:.3f}, std={a_sd:.3f}, "
        f"n={a_n} ({total_logical} steps x {num_nodes}N, steps {all_range})"
    )
    if w_n > 0:
        print(
            f"    warmup  TGS={w_mean:.3f}, std={w_sd:.3f}, "
            f"n={w_n} (steps {warmup_range} x {num_nodes}N)"
        )
    if s_n > 0:
        print(
            f"    steady  TGS={s_mean:.3f}, std={s_sd:.3f}, "
            f"n={s_n}/{STEADY_STATE_STEPS * num_nodes} "
            f"(steps {steady_range} x {num_nodes}N)"
        )
    else:
        print(f"    steady  TGS=NA, n=0/{STEADY_STATE_STEPS * num_nodes}")
    if tail:
        t_mean, t_sd, t_n = _compute_stats(tail)
        tail_range = _step_range(
            WARMUP_STEPS + STEADY_STATE_STEPS,
            WARMUP_STEPS + STEADY_STATE_STEPS + t_n // num_nodes - 1,
        )
        print(
            f"    tail    TGS={t_mean:.3f}, std={t_sd:.3f}, "
            f"n={t_n} (steps {tail_range} x {num_nodes}N)"
        )

    # ── no steady-state data → error out ──
    if s_n == 0:
        print(
            f"  {RED}ERROR: only {total_logical} logical step(s) logged, need "
            f">{WARMUP_STEPS} to have any steady-state data{RESET}"
        )
        print(
            f"  {YELLOW}HINT: set steps >= {RECOMMENDED_MIN_STEPS} "
            f"(first {WARMUP_STEPS} are warmup, "
            f"next {STEADY_STATE_STEPS} are measured){RESET}"
        )
        if running and not force:
            print(f"  {YELLOW}(job appears to still be running, use -f to override){RESET}")
        else:
            _cleanup_prompt(file, cleanup)
        print()
        return 2

    # Official value for renaming
    avg_tgs = f"{s_mean:.3f}"
    print(f"  {GREEN}{BOLD}>> Using steady TGS={avg_tgs} for benchmarking result{RESET}")

    # ── warnings ──
    if total_logical < RECOMMENDED_MIN_STEPS:
        print(
            f"  {YELLOW}WARNING: only {total_logical} logical steps "
            f"({s_n // num_nodes} steady-state); recommend steps >= "
            f"{RECOMMENDED_MIN_STEPS} for reliable TGS{RESET}"
        )

    threshold = 0.95 * s_mean
    below95 = sum(1 for v in steady if v < threshold)
    if below95 > 0:
        print(
            f"  {YELLOW}WARNING: {below95} of {s_n} steady-state steps "
            f"are below 95% ({threshold:.3f}) of steady TGS ({avg_tgs}){RESET}"
        )

    # ── check minimum data for renaming ──
    s_logical = s_n // num_nodes if num_nodes else s_n
    if s_logical < MIN_STEADY_FOR_RENAME:
        print(
            f"  {YELLOW}WARNING: insufficient steady-state data for renaming "
            f"({s_logical}/{MIN_STEADY_FOR_RENAME} logical steps required){RESET}"
        )
        print()
        return 0

    # ── skip rename if job is still running ──
    if running:
        if not force:
            print(
                f"  {YELLOW}WARNING: job appears to still be running, "
                f"skipping rename (use -f to override){RESET}"
            )
            print()
            return 0
        print(
            f"  {YELLOW}WARNING: job appears to still be running, "
            f"forcing rename (-f){RESET}"
        )

    # ── build new name ──
    directory = file.parent
    fname_noext = file.stem

    tgs_m = RE_TGS_TAG.match(fname_noext)
    if tgs_m:
        # Already has TGS tag — update the value
        new_noext = f"{tgs_m.group(1)}{avg_tgs}"
    else:
        quant = _extract_quantization(text)
        pdbs = _extract_per_device_batch_size(text)
        q_part = f"{quant}-" if quant else ""
        pdbs_part = f"pdbs{pdbs}-" if pdbs else ""
        new_noext = f"{fname_noext}-{q_part}{num_nodes}N-{pdbs_part}TGS_{avg_tgs}"

    new_path = directory / f"{new_noext}{file.suffix}"

    # ── rename file (skip no-op) ──
    renamed_file = file.resolve() != new_path.resolve()
    if renamed_file:
        file.rename(new_path)
        print(f"  renamed to {new_path}")

    # ── rename job folder ──
    # Skip directory rename while the job is running — the training process
    # writes checkpoints, profiles, and TensorBoard events by path.
    # Renaming the directory would break those writes.  The log file rename
    # above is safe (processes write via fd, which follows the inode).
    # Run tgs_tagger again after the job finishes to rename the directory.
    final_folder: Path | None = None
    job_id_m = RE_JOB_ID.match(file.name)
    if job_id_m:
        job_folder = _find_job_folder(directory, job_id_m.group(1))
        if job_folder:
            new_folder = directory / new_noext
            if job_folder.resolve() != new_folder.resolve():
                if running:
                    print(
                        f"  {YELLOW}skipping directory rename while job is running "
                        f"(re-run after job completes){RESET}"
                    )
                else:
                    job_folder.rename(new_folder)
                    print(f"  renamed job folder to {new_folder}")
                    final_folder = new_folder

    # ── update log symlink inside job folder ──
    # When both the log file and directory were renamed, update the symlink
    # so it points to the new log filename.
    if renamed_file and final_folder and final_folder.is_dir():
        log_link = final_folder / "log"
        if log_link.is_symlink():
            log_link.unlink()
            log_link.symlink_to(f"../{new_path.name}")
            print(f"  updated log symlink → ../{new_path.name}")
    elif renamed_file and job_id_m:
        # Directory wasn't renamed (running job) — update the symlink in the
        # existing folder so it tracks the renamed log file.
        existing = _find_job_folder(directory, job_id_m.group(1))
        if existing and existing.is_dir():
            log_link = existing / "log"
            if log_link.is_symlink():
                log_link.unlink()
                log_link.symlink_to(f"../{new_path.name}")
                print(f"  updated log symlink → ../{new_path.name}")

    print()
    return 0


# ── File discovery ───────────────────────────────────────────────────────────


def _find_log_files(directory: Path, pattern: str = "*") -> list[Path]:
    """Return all .log and .out files in *directory* matching *pattern*."""
    candidates = []
    for suffix in _LOG_SUFFIXES:
        candidates.extend(f for f in directory.glob(f"{pattern}{suffix}") if f.is_file())
    return candidates


def _latest_log_file(directory: Path) -> Path | None:
    """Return the most recently modified .log/.out file in *directory*."""
    candidates = _find_log_files(directory)
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


def resolve_files(args: list[str], default_dir: str) -> list[Path]:
    """Resolve CLI arguments to a list of .log/.out file paths.

    Handles: single directory (latest only), job IDs, file paths,
    directories (all .log/.out), and glob patterns.
    """

    # No arguments → default directory, latest file only
    if not args:
        args = [default_dir]

    # Single directory argument → job dir log symlink, or latest log file
    if len(args) == 1 and Path(args[0]).is_dir():
        d = Path(args[0])
        # Job directory with a log symlink (e.g. tab-completed job dir)
        log_link = d / "log"
        if log_link.is_file():
            return [log_link.resolve()]
        latest = _latest_log_file(d)
        if latest is None:
            _report_na(str(d), "no log files found")
            return []
        return [latest]

    # Multiple arguments: job IDs, files, directories, globs
    files: list[Path] = []
    for arg in args:
        p = Path(arg)

        # Pure integer → job ID
        if arg.isdigit():
            job_dir = Path(default_dir)
            matches = sorted(_find_log_files(job_dir, f"{arg}-*"))
            if matches:
                files.extend(matches)
            else:
                _report_na(
                    f"job ID {arg}",
                    f"no log files found for job ID {arg} in {job_dir}",
                )
            continue

        # Existing regular file
        if p.is_file():
            if p.suffix in _LOG_SUFFIXES:
                files.append(p)
            continue

        # Existing directory → job dir log symlink, or all first-level log files
        if p.is_dir():
            log_link = p / "log"
            if log_link.is_file():
                files.append(log_link.resolve())
            else:
                files.extend(sorted(_find_log_files(p)))
            continue

        # Try glob expansion
        expanded = sorted(glob.glob(arg))
        if not expanded:
            if any(c in arg for c in "*?["):
                _report_na(arg, "pattern matched no files or path not found")
            else:
                _report_na(arg, "file does not exist")
        else:
            files.extend(
                Path(f) for f in expanded
                if Path(f).is_file() and Path(f).suffix in _LOG_SUFFIXES
            )

    return files


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgs_tagger",
        description=(
            "Parse training log files (.log or legacy .out) and rename them "
            "with TGS (Tokens/GPU/Second) metrics. Also renames corresponding "
            "job folders."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
arguments:
  PATH can be one or more of:
    - Job ID:  pure integer → searches default outputs dir for <id>-*.log/out
    - File:    specific .log/.out file to process
    - Job dir: directory with a log symlink → follows symlink (tab-completion friendly)
    - Dir:     single dir → latest .log/.out; multiple dirs → all log files
    - Glob:    pattern expanding to .log/.out files (e.g. outputs/*.log)

  Default: outputs directory (from JOB_WORKSPACE or relative to script)

file requirements:
  - .log or .out extension
  - NUM_NODES=<int> in the log header
  - Must contain "Tokens/s/device:" lines

steady-state measurement:
  Steps 0-4 are discarded (XLA compilation / profiler warmup).
  Steps 5-14 (up to 10 data points) are used for TGS calculation.
  Recommended: set training steps >= 15 for reliable metrics.

exit status:
  0  Success
  1  File not found / not a regular file / not .log/.out / no files to process
  2  No Tokens/s/device lines found (or no steady-state data)
  3  NUM_NODES not found in log header

examples:
  %(prog)s                                    # latest log in default outputs dir
  %(prog)s 12345                              # process files for job 12345
  %(prog)s outputs/12345-run.log              # specific file
  %(prog)s outputs/12345-JAX-llama2-70b/      # job dir (follows log symlink)
  %(prog)s -f outputs/12345-run.log           # force rename/cleanup if job looks active
  %(prog)s -c outputs/*.log                   # cleanup mode on all log files
""",
    )
    parser.add_argument(
        "-c",
        "--cleanup",
        action="store_true",
        help="prompt to delete files with no Tokens/s/device data (and their job folders)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force rename/cleanup even if the job appears to still be running",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="files, directories, job IDs, or glob patterns to process",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default outputs directory: JOB_WORKSPACE or ../outputs relative to this script
    script_dir = Path(__file__).resolve().parent
    default_dir = os.environ.get("JOB_WORKSPACE", str(script_dir.parent / "outputs"))

    files = resolve_files(args.paths, default_dir)
    if not files:
        return 1

    last_rc = 0
    for f in files:
        rc = process_file(f, cleanup=args.cleanup, force=args.force)
        if rc != 0:
            last_rc = rc

    return last_rc


if __name__ == "__main__":
    sys.exit(main())
