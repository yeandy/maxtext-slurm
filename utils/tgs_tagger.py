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
from pathlib import Path

# ── ANSI colors ──────────────────────────────────────────────────────────────

RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# ── Regex patterns ───────────────────────────────────────────────────────────

RE_TOKENS = re.compile(r"Tokens/s/device:\s*([0-9]+(?:\.[0-9]+)?)")
RE_QUANT = re.compile(r"Config param quantization:\s*(.*)")
RE_PDBS = re.compile(r"Config param per_device_batch_size:\s*([0-9]+(?:\.[0-9]+)?)")
RE_TGS_TAG = re.compile(r"(.*[-_]TGS_)[0-9]+(\.[0-9]+)?$")
RE_JOB_ID = re.compile(r"^([^-]+)-")

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
    # Strip quotes and special chars, sanitize for filenames
    for ch in "\"'`":
        raw = raw.replace(ch, "")
    raw = raw.replace("/", "-")
    return "" if raw in ("NONE", "") else raw


def _extract_per_device_batch_size(text: str) -> str:
    """Extract per_device_batch_size from log text."""
    m = RE_PDBS.search(text)
    return m.group(1) if m else ""


def _compute_stats(
    values: list[float], need: int
) -> tuple[float, float, int, int, float]:
    """Compute TGS stats on the last min(need, len(values)) entries.

    Returns (mean, std, n_used, below_95_count, threshold_95).
    """
    use = min(len(values), need)
    tail = values[-use:]
    mean = sum(tail) / use
    var = max(sum(v * v for v in tail) / use - mean * mean, 0.0)
    sd = math.sqrt(var)
    threshold = 0.95 * mean
    below = sum(1 for v in tail if v < threshold)
    return mean, sd, use, below, threshold


# ── Core: process one file ───────────────────────────────────────────────────


_LOG_SUFFIXES = {".log", ".out"}


def process_file(file: Path, cleanup: bool) -> int:
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
        m = re.match(r"^(?:NNODES|SLURM_JOB_NUM_NODES)\s*=\s*(\d+)", line)
        if m:
            num_nodes = int(m.group(1))
            break
    if num_nodes is None:
        return fail("NNODES=<int> not found in log header", rc=3)
    need = num_nodes * 10
    min_for_rename = num_nodes * 5

    # ── extract all Tokens/s/device values ──
    all_values = [float(m.group(1)) for m in RE_TOKENS.finditer(text)]
    total_available = len(all_values)

    if total_available == 0:
        print(f"  TGS=NA, std=NA, n=0/{need}")
        print(f"  {RED}ERROR: no Tokens/s/device lines found{RESET}")
        _cleanup_prompt(file, cleanup)
        print()
        return 2

    # ── compute stats ──
    mean, sd, n_used, below95, threshold = _compute_stats(all_values, need)
    avg_tgs = f"{mean:.3f}"

    print(f"  TGS={avg_tgs}, std={sd:.3f}, n={n_used}/{need}")
    if below95 > 0:
        print(
            f"  {RED}ERROR: {below95} of {n_used} are below 95% "
            f"({threshold:.3f}) of TGS ({avg_tgs}){RESET}"
        )

    # ── check minimum data for renaming ──
    if total_available < min_for_rename:
        print(
            f"  {YELLOW}WARNING: insufficient data for renaming "
            f"({total_available}/{min_for_rename} required){RESET}"
        )
        print()
        return 0

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
    if file.resolve() != new_path.resolve():
        file.rename(new_path)
        print(f"  renamed to {new_path}")

    # ── rename job folder ──
    job_id_m = RE_JOB_ID.match(file.name)
    if job_id_m:
        job_folder = _find_job_folder(directory, job_id_m.group(1))
        if job_folder:
            new_folder = directory / new_noext
            if job_folder.resolve() != new_folder.resolve():
                job_folder.rename(new_folder)
                print(f"  renamed job folder to {new_folder}")

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

        # Pure integer → Slurm job ID
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
  - Line 2 must be an integer N (sample size = N * 10)
  - Must contain "Tokens/s/device:" lines

exit status:
  0  Success
  1  File not found / not a regular file / not .log/.out / no files to process
  2  No Tokens/s/device lines found
  3  Line 2 is not an integer
  4  Failed to compute statistics

examples:
  %(prog)s                                    # latest log in default outputs dir
  %(prog)s 12345                              # process files for Slurm job 12345
  %(prog)s logs                               # latest log in logs/
  %(prog)s outputs/12345-run.log              # specific file
  %(prog)s outputs/12345-JAX-llama2-70b/      # job dir (follows log symlink)
  %(prog)s -c outputs/*.log                   # cleanup mode on all log files
  %(prog)s dir1 dir2 dir3                     # all log files from each directory
""",
    )
    parser.add_argument(
        "-c",
        "--cleanup",
        action="store_true",
        help="prompt to delete files with no Tokens/s/device data (and their job folders)",
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
        rc = process_file(f, cleanup=args.cleanup)
        if rc != 0:
            last_rc = rc

    return last_rc


if __name__ == "__main__":
    sys.exit(main())
