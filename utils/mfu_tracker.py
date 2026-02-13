#!/usr/bin/env python3
"""On-the-fly MFU (Model FLOPs Utilization) tracking for MaxText training.

Usage:
    # Diagnostic (print GPU detection + peak TFLOPS):
    python3 mfu_tracker.py

    # Training mode (direct or via Ray subprocess):
    python3 -u mfu_tracker.py <config.yml> [key=value ...]

Override auto-detection:
    export HARDWARE_PEAK_TFLOPS=5000
"""

import io
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Peak TFLOPS table  (gpu x dtype, dense, NO sparsity)
# ---------------------------------------------------------------------------

_GPU_PEAK_TFLOPS = {
    # AMD Instinct -- CDNA 4
    "MI355X": {"bf16": 2500, "fp16": 2500, "fp8": 5000, "fp32": 157},
    "MI350X": {"bf16": 2250, "fp16": 2250, "fp8": 4500, "fp32": 143},
    # AMD Instinct -- CDNA 3
    "MI325X": {"bf16": 1307, "fp16": 1307, "fp8": 2614, "fp32": 163},
    "MI300X": {"bf16": 1307, "fp16": 1307, "fp8": 2614, "fp32": 163},
    "MI300A": {"bf16":  981, "fp16":  981, "fp8": 1963, "fp32": 122},
    # AMD Instinct -- CDNA 2
    "MI250X": {"bf16": 383, "fp16": 383, "fp32": 47},
    "MI250":  {"bf16": 362, "fp16": 362, "fp32": 45},
    "MI210":  {"bf16": 181, "fp16": 181, "fp32": 22},
    # NVIDIA Blackwell
    "B300": {"bf16": 2250, "fp16": 2250, "fp8": 4500, "fp32": 75},
    "B200": {"bf16": 2250, "fp16": 2250, "fp8": 4500, "fp32": 75},
    # NVIDIA Hopper
    "H200": {"bf16": 989, "fp16": 989, "fp8": 1979, "fp32": 67},
    "H100": {"bf16": 989, "fp16": 989, "fp8": 1979, "fp32": 67},
    "H800": {"bf16": 989, "fp16": 989, "fp8": 1979, "fp32": 67},
    # NVIDIA Ada / Ampere
    "L40S": {"bf16": 362, "fp16": 362, "fp8": 733, "fp32": 91},
    "A100": {"bf16": 312, "fp16": 312, "fp32": 19},
    "A800": {"bf16": 312, "fp16": 312, "fp32": 19},
    "A10G": {"bf16":  70, "fp16":  70, "fp32": 35},
}

# gfx architecture → representative GPU (fallback when product name is generic)
_GFX_TO_GPU = {
    "gfx950": "MI355X",
    "gfx942": "MI300X",
    "gfx941": "MI300A",
    "gfx940": "MI300A",
    "gfx90a": "MI250X",
    "gfx908": "MI210",
}

_DEFAULT_DTYPE = "bf16"
_TFLOPS_RE = re.compile(r"TFLOP/s/device:\s+([\d.]+)")

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _run_cmd(cmd, timeout=10):
    """Run a command, return stdout or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _match_known_gpu(text):
    """Find a known GPU model in text (case-insensitive, longest match first)."""
    upper = text.upper()
    for name in sorted(_GPU_PEAK_TFLOPS, key=len, reverse=True):
        if name in upper:
            return name
    return None


def detect_gpu():
    """Auto-detect GPU model.  Returns e.g. 'MI355X', 'H100', or None.

    AMD: rocminfo (Marketing Name → gfx ID fallback) → amd-smi
    NVIDIA: nvidia-smi
    """
    # AMD: rocminfo -- single source with both product name and gfx ID
    out = _run_cmd(["rocminfo"])
    if out:
        gfx_fallback = None
        for line in out.splitlines():
            stripped = line.strip()
            if stripped.startswith("Marketing Name:"):
                name = _match_known_gpu(stripped.split(":", 1)[1])
                if name:
                    return name
            elif stripped.startswith("Name:") and not gfx_fallback:
                gfx = stripped.split(":", 1)[1].strip().lower()
                if gfx in _GFX_TO_GPU:
                    gfx_fallback = _GFX_TO_GPU[gfx]
        if gfx_fallback:
            return gfx_fallback

    # AMD: amd-smi (may have better product names on some systems)
    out = _run_cmd(["amd-smi", "static", "--gpu", "0", "--asic", "--json"])
    if out:
        name = _match_known_gpu(out)
        if name:
            return name

    # NVIDIA
    out = _run_cmd(["nvidia-smi", "--query-gpu=name",
                    "--format=csv,noheader", "--id=0"])
    if out:
        return _match_known_gpu(out)

    return None

# ---------------------------------------------------------------------------
# Dtype detection
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "fp8": "fp8", "nanoo_fp8": "fp8", "fp8_full": "fp8",
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "fp16": "fp16",
    "float32": "fp32", "fp32": "fp32",
}


def _normalize_dtype(raw):
    """Normalise a MaxText dtype / quantization value to a lookup key."""
    return _DTYPE_MAP.get(raw.strip().strip("\"'").lower())


def resolve_compute_dtype(argv):
    """Determine compute dtype from MaxText training args.

    Resolution order: CLI overrides > YAML config > default (bf16).
    FP8 quantization takes priority over dtype (matmuls run in FP8).
    """
    cli_quant = cli_dtype = None
    for arg in (argv or []):
        if "=" not in arg:
            continue
        key, _, val = arg.partition("=")
        k = key.strip().lower()
        if k == "quantization" and val.strip():
            cli_quant = _normalize_dtype(val)
        elif k == "dtype":
            cli_dtype = _normalize_dtype(val)

    if cli_quant == "fp8":
        return "fp8"
    if cli_dtype:
        return cli_dtype

    # Parse YAML config (first positional arg that is a file)
    for arg in (argv or []):
        if arg.startswith("-") or "=" in arg:
            continue
        if os.path.isfile(arg):
            quant, dtype = _parse_yaml_dtype(arg)
            if quant == "fp8":
                return "fp8"
            if dtype:
                return dtype
            break

    return _DEFAULT_DTYPE


def _parse_yaml_dtype(path):
    """Extract quantization and dtype from a MaxText YAML config."""
    quant = dtype = None
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s or s[0] == "#" or ":" not in s:
                    continue
                key, _, val = s.partition(":")
                key = key.strip().lower()
                val = val.split("#")[0].strip()
                if key == "quantization" and val:
                    quant = _normalize_dtype(val)
                elif key == "dtype" and val:
                    dtype = _normalize_dtype(val)
    except OSError:
        pass
    return quant, dtype

# ---------------------------------------------------------------------------
# Peak TFLOPS resolution
# ---------------------------------------------------------------------------

def detect_peak_tflops(argv=None):
    """Return (peak_tflops, gpu_name, compute_dtype, source)."""
    compute_dtype = resolve_compute_dtype(argv)

    # Manual override
    env_val = os.environ.get("HARDWARE_PEAK_TFLOPS", "").strip()
    if env_val and env_val not in ("0", "auto"):
        try:
            return float(env_val), "manual", compute_dtype, "env"
        except ValueError:
            pass

    # Auto-detect
    gpu = detect_gpu()
    if gpu and gpu in _GPU_PEAK_TFLOPS:
        dtype_map = _GPU_PEAK_TFLOPS[gpu]
        if compute_dtype in dtype_map:
            return float(dtype_map[compute_dtype]), gpu, compute_dtype, "auto"
        if _DEFAULT_DTYPE in dtype_map:
            return float(dtype_map[_DEFAULT_DTYPE]), gpu, _DEFAULT_DTYPE, "auto(dtype_fallback)"

    return 0.0, gpu or "unknown", compute_dtype, "none"

# ---------------------------------------------------------------------------
# Stream interceptor
# ---------------------------------------------------------------------------

class _MFUStream(io.TextIOBase):
    """Wraps a text stream to append ', MFU: X.XX%' after TFLOP/s/device."""

    def __init__(self, wrapped, peak_tflops):
        self._wrapped = wrapped
        self._peak = peak_tflops

    def __getattr__(self, name):
        # Delegate any attribute not explicitly defined (reconfigure, buffer,
        # name, mode, newlines, etc.) to the wrapped stream.
        return getattr(self._wrapped, name)

    def write(self, text):
        if "TFLOP/s/device" in text:
            m = _TFLOPS_RE.search(text)
            if m:
                mfu = float(m.group(1)) / self._peak * 100.0
                pos = m.end()
                text = f"{text[:pos]}, MFU: {mfu:.2f}%{text[pos:]}"
        return self._wrapped.write(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup(argv, tag="[MFU]"):
    """One-shot MFU tracker setup.  Call once before training starts.

    Detects GPU + dtype, wraps stdout/stderr to append MFU% to log lines.
    Prints a status message.  Returns (peak_tflops, gpu, dtype, source).
    """
    peak, gpu, dtype, source = detect_peak_tflops(argv)
    if peak > 0:
        sys.stdout = _MFUStream(sys.stdout, peak)
        sys.stderr = _MFUStream(sys.stderr, peak)
        print(f"{tag} MFU tracking enabled (gpu={gpu}, dtype={dtype}, "
              f"peak={peak:.0f} TFLOP/s, source={source})", flush=True)
    else:
        print(f"{tag} MFU tracking disabled (gpu={gpu} not in lookup table; "
              f"set HARDWARE_PEAK_TFLOPS=<value> to override)", flush=True)
    return peak, gpu, dtype, source

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_gpu_info():
    """Print GPU detection results and peak TFLOPS table (diagnostic mode)."""
    gpu = detect_gpu()
    print(f"Detected GPU: {gpu or 'unknown'}")
    print()

    if gpu and gpu in _GPU_PEAK_TFLOPS:
        dtype_map = _GPU_PEAK_TFLOPS[gpu]
        print(f"Peak TFLOPS for {gpu} (dense, no sparsity):")
        for dtype in ("fp8", "bf16", "fp16", "fp32"):
            if dtype in dtype_map:
                print(f"  {dtype:>4s}:  {dtype_map[dtype]:>6,} TFLOP/s")
    else:
        print(f"GPU '{gpu or 'unknown'}' not found in lookup table.")
        print("Set HARDWARE_PEAK_TFLOPS=<value> to override.")
        print()
        print("Known GPUs:")
        for name in _GPU_PEAK_TFLOPS:
            bf16 = _GPU_PEAK_TFLOPS[name].get("bf16", "—")
            print(f"  {name:<8s}  bf16={bf16} TFLOP/s")


def main():
    """Run MaxText training with MFU tracking, or print GPU info if no args."""
    argv = sys.argv[1:]
    if not argv:
        _print_gpu_info()
        return

    setup(argv)
    from MaxText import train as maxtext_train
    maxtext_train.main(["maxtext_train"] + argv)


if __name__ == "__main__":
    main()
