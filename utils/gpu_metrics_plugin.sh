#!/usr/bin/env bash
# gpu_metrics_plugin.sh — Prometheus metrics plugin: AMD GPU hardware.
#
# Collects per-GPU temperature, power draw, clock speeds, and VRAM usage
# via sysfs (zero subprocess overhead — plain file reads from
# /sys/class/hwmon and /sys/class/drm).
#
# Always collects XGMI RAS error counters via sysfs (zero overhead).
#
# Called by metrics_exporter.sh — outputs Prometheus text to stdout.
#
# Metrics (all prefixed hw_ for grouping in Prometheus UI):
#   hw_gpu_temperature_celsius{gpu,host}           Junction temperature (°C)
#   hw_gpu_power_watts{gpu,host}                   Current power draw (W)
#   hw_gpu_clock_mhz{gpu,host,type=sclk|mclk}     Core / memory clock (MHz)
#   hw_gpu_vram_used_bytes{gpu,host}                VRAM currently used (bytes)
#   hw_gpu_vram_total_bytes{gpu,host}               VRAM total capacity (bytes)
#   hw_gpu_ras_umc_{ue,ce}_total{gpu,host}         HBM memory ECC errors
#   hw_gpu_ras_xgmi_{ue,ce}_total{gpu,host}        XGMI/WAFL link errors
#   hw_gpu_ras_gfx_{ue,ce}_total{gpu,host}         Compute engine errors
#   hw_gpu_ras_mmhub_{ue,ce}_total{gpu,host}       Memory hub errors
#   hw_gpu_ras_sdma_{ue,ce}_total{gpu,host}        SDMA engine errors
#   hw_gpu_pcie_correctable_total{gpu,host}          PCIe correctable AER errors
#   hw_gpu_pcie_nonfatal_total{gpu,host}             PCIe non-fatal AER errors
#   hw_gpu_pcie_fatal_total{gpu,host}                PCIe fatal AER errors

HOSTNAME_SHORT="${1:?Usage: gpu_metrics_plugin.sh <hostname>}"

python3 - "$HOSTNAME_SHORT" <<'PYEOF'
import os, re, sys
from pathlib import Path

hostname = sys.argv[1]
lines = []

def add(line):
    lines.append(line)

def read_int(path):
    """Read a single integer from a sysfs file, return None on failure."""
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None

# =========================================================================
# Discover AMD GPUs via sysfs hwmon (no subprocess, no driver ioctls)
# =========================================================================
# Each amdgpu device exposes a hwmon directory with temperature, power,
# and clock files.  We sort by PCI bus address to assign GPU indices 0..N
# matching the order ROCm uses.

def discover_amd_gpus():
    """Return list of (gpu_index, hwmon_path) sorted by PCI bus address."""
    hwmon_root = Path('/sys/class/hwmon')
    if not hwmon_root.exists():
        return []

    gpus = []  # (pci_addr, hwmon_path)
    for hwdir in hwmon_root.iterdir():
        name_file = hwdir / 'name'
        if not name_file.exists():
            continue
        try:
            name = name_file.read_text().strip()
        except Exception:
            continue
        if name != 'amdgpu':
            continue

        # Resolve PCI bus address from the device symlink
        dev_path = (hwdir / 'device').resolve()
        pci_match = re.findall(r'[0-9a-f]+:[0-9a-f]+:[0-9a-f]+\.[0-9a-f]+', str(dev_path))
        pci_addr = pci_match[-1] if pci_match else str(hwdir)
        gpus.append((pci_addr, str(hwdir)))

    # Sort by PCI address → GPU index 0, 1, 2, ...
    gpus.sort(key=lambda x: x[0])
    return [(idx, path) for idx, (_, path) in enumerate(gpus)]

# =========================================================================
# GPU: temperature, power, clocks, VRAM  (sysfs reads — no subprocess)
# =========================================================================
add('# HELP hw_gpu_temperature_celsius GPU temperature in Celsius.')
add('# TYPE hw_gpu_temperature_celsius gauge')
add('# HELP hw_gpu_power_watts GPU power draw in Watts.')
add('# TYPE hw_gpu_power_watts gauge')
add('# HELP hw_gpu_clock_mhz GPU clock speed in MHz.')
add('# TYPE hw_gpu_clock_mhz gauge')
add('# HELP hw_gpu_vram_used_bytes GPU VRAM currently used in bytes.')
add('# TYPE hw_gpu_vram_used_bytes gauge')
add('# HELP hw_gpu_vram_total_bytes GPU VRAM total capacity in bytes.')
add('# TYPE hw_gpu_vram_total_bytes gauge')
# NOTE: GPU utilization is NOT collected here because the sysfs
# gpu_busy_percent / mem_busy_percent files return 0 on MI355 OAM
# with current ROCm drivers.  Use ray_node_gpus_utilization from
# Ray's Prometheus exporter (port 8080) instead — it works and
# provides per-GPU utilization with a GpuIndex label.

amd_gpus = discover_amd_gpus()

if amd_gpus:
    for gpu_id, hwpath in amd_gpus:
        hw = Path(hwpath)
        lb = f'gpu="{gpu_id}",host="{hostname}"'

        # Temperature: prefer junction (temp2), fall back to mem (temp3),
        # then try temp1.  Values are in millidegrees C.
        for temp_file in ('temp2_input', 'temp3_input', 'temp1_input'):
            val = read_int(hw / temp_file)
            if val is not None and val > 0:
                add(f'hw_gpu_temperature_celsius{{{lb}}} {val / 1000.0:.1f}')
                break

        # Power: power1_input is in microwatts.
        pval = read_int(hw / 'power1_input')
        if pval is not None:
            add(f'hw_gpu_power_watts{{{lb}}} {pval / 1e6:.1f}')

        # Clocks: freq1_input (sclk) and freq2_input (mclk) are in Hz.
        sclk = read_int(hw / 'freq1_input')
        if sclk is not None:
            add(f'hw_gpu_clock_mhz{{{lb},type="sclk"}} {sclk / 1e6:.0f}')
        mclk = read_int(hw / 'freq2_input')
        if mclk is not None:
            add(f'hw_gpu_clock_mhz{{{lb},type="mclk"}} {mclk / 1e6:.0f}')

        # VRAM: mem_info_vram_used / _total are in the device directory (bytes).
        dev = (hw / 'device').resolve()
        vram_used = read_int(dev / 'mem_info_vram_used')
        if vram_used is not None:
            add(f'hw_gpu_vram_used_bytes{{{lb}}} {vram_used}')
        vram_total = read_int(dev / 'mem_info_vram_total')
        if vram_total is not None:
            add(f'hw_gpu_vram_total_bytes{{{lb}}} {vram_total}')

else:
    print('[gpu_plugin] No amdgpu hwmon devices found in /sys/class/hwmon',
          file=sys.stderr)

# =========================================================================
# GPU RAS error counters  (sysfs — always on, zero overhead)
# =========================================================================
# The amdgpu driver exposes per-block RAS counters via sysfs:
#   aca_umc        — HBM memory ECC  (equivalent to check_ecc.sh / rocm-smi)
#   aca_xgmi_wafl  — XGMI/WAFL inter-GPU link errors
#   aca_gfx        — Compute engine (shader) errors
#   aca_mmhub      — Memory hub / VRAM controller errors
#   aca_sdma       — SDMA (DMA copy engine) errors
#
# Each block reports ue (uncorrectable) and ce (correctable) counts.
# A non-zero ue in any block means the GPU has a hardware fault and the
# node should be drained.

RAS_BLOCKS = [
    ('aca_umc',       'umc'),
    ('aca_xgmi_wafl', 'xgmi'),
    ('aca_gfx',       'gfx'),
    ('aca_mmhub',     'mmhub'),
    ('aca_sdma',      'sdma'),
]

for _, short in RAS_BLOCKS:
    add(f'# HELP hw_gpu_ras_{short}_ue_total GPU {short.upper()} uncorrectable RAS errors.')
    add(f'# TYPE hw_gpu_ras_{short}_ue_total counter')
    add(f'# HELP hw_gpu_ras_{short}_ce_total GPU {short.upper()} correctable RAS errors.')
    add(f'# TYPE hw_gpu_ras_{short}_ce_total counter')

# Build PCI → GPU index mapping once (reuse discover_amd_gpus result).
pci_to_gpu = {}
for idx, hwpath in amd_gpus:
    hw_pci = re.findall(r'[0-9a-f]+:[0-9a-f]+:[0-9a-f]+\.[0-9a-f]+',
                        str(Path(hwpath, 'device').resolve()))
    if hw_pci:
        pci_to_gpu[hw_pci[-1]] = idx

try:
    for dev in sorted(Path('/sys/bus/pci/drivers/amdgpu').iterdir()):
        if not dev.name.startswith('0000:'):
            continue
        gpu_id = pci_to_gpu.get(dev.name)
        if gpu_id is None:
            continue

        lb = f'gpu="{gpu_id}",host="{hostname}"'
        ras_dir = dev / 'ras'
        for sysfs_name, short in RAS_BLOCKS:
            ras_file = ras_dir / sysfs_name
            if not ras_file.exists():
                continue
            text = ras_file.read_text()
            for line in text.splitlines():
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if key == 'ue':
                        add(f'hw_gpu_ras_{short}_ue_total{{{lb}}} {val}')
                    elif key == 'ce':
                        add(f'hw_gpu_ras_{short}_ce_total{{{lb}}} {val}')
except Exception as e:
    print(f'[gpu_plugin] GPU RAS: {e}', file=sys.stderr)

# =========================================================================
# PCIe AER error counters  (sysfs — Advanced Error Reporting)
# =========================================================================
# PCIe AER errors are a major source of GPU disconnects and hangs.  The
# kernel exposes per-device totals via:
#   aer_dev_correctable  — recoverable (retried) errors
#   aer_dev_nonfatal     — uncorrectable but not fatal (device usable)
#   aer_dev_fatal        — uncorrectable fatal (device unusable)
#
# Each file contains named counters and a TOTAL_ERR_* line.  We export
# the totals.  A non-zero fatal count means the GPU's PCIe link failed.
# A spike in correctable errors often precedes a fatal event.

add('# HELP hw_gpu_pcie_correctable_total PCIe correctable AER errors.')
add('# TYPE hw_gpu_pcie_correctable_total counter')
add('# HELP hw_gpu_pcie_nonfatal_total PCIe non-fatal uncorrectable AER errors.')
add('# TYPE hw_gpu_pcie_nonfatal_total counter')
add('# HELP hw_gpu_pcie_fatal_total PCIe fatal uncorrectable AER errors.')
add('# TYPE hw_gpu_pcie_fatal_total counter')

def parse_aer_total(path, prefix='TOTAL_ERR'):
    """Parse a PCIe AER sysfs file and return the total error count."""
    try:
        text = Path(path).read_text()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].startswith(prefix):
                return int(parts[1])
    except Exception:
        pass
    return None

try:
    for dev in sorted(Path('/sys/bus/pci/drivers/amdgpu').iterdir()):
        if not dev.name.startswith('0000:'):
            continue
        gpu_id = pci_to_gpu.get(dev.name)
        if gpu_id is None:
            continue
        lb = f'gpu="{gpu_id}",host="{hostname}"'

        for sysfs_name, metric, prefix in [
            ('aer_dev_correctable', 'hw_gpu_pcie_correctable_total', 'TOTAL_ERR_COR'),
            ('aer_dev_nonfatal',    'hw_gpu_pcie_nonfatal_total',    'TOTAL_ERR_NONFATAL'),
            ('aer_dev_fatal',       'hw_gpu_pcie_fatal_total',       'TOTAL_ERR_FATAL'),
        ]:
            aer_file = dev / sysfs_name
            if aer_file.exists():
                val = parse_aer_total(str(aer_file), prefix)
                if val is not None:
                    add(f'{metric}{{{lb}}} {val}')
except Exception as e:
    print(f'[gpu_plugin] PCIe AER: {e}', file=sys.stderr)

# Output
print('\n'.join(lines))
PYEOF
