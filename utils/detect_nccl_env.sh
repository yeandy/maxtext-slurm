#!/bin/bash
#
# Auto-detect NCCL network environment variables (IB HCA, QoS, socket interface).
#
# Sourced by train_env.sh (script mode) and _container.sh (interactive mode).
# Each variable is only set when not already present, so manual overrides
# and _env_KEY=VALUE passthrough args take precedence.
#
# Usage:
#   source utils/detect_nccl_env.sh

_DETECT_NCCL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NCCL_IB_HCA: enumerate InfiniBand HCA devices
if [[ -z "${NCCL_IB_HCA:-}" && -d /sys/class/infiniband ]]; then
    NCCL_IB_HCA=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    [[ -n "$NCCL_IB_HCA" ]] && export NCCL_IB_HCA
fi

# NCCL_IB_TC / NCCL_IB_FIFO_TC: Pensando AINIC QoS auto-detection
if [[ -z "${NCCL_IB_TC:-}" && -z "${NCCL_IB_FIFO_TC:-}" ]]; then
    source "$_DETECT_NCCL_DIR/detect_ainic_nccl_ib_tc.sh"
    if is_pensando; then
        _tc=$(detect_pensando_tc)
        NCCL_IB_TC=$(echo "$_tc" | awk '{print $1}')
        NCCL_IB_FIFO_TC=$(echo "$_tc" | awk '{print $2}')
        echo "[INFO] $(hostname -s): Pensando AINIC detected: NCCL_IB_TC=$NCCL_IB_TC NCCL_IB_FIFO_TC=$NCCL_IB_FIFO_TC"
        [[ -n "$NCCL_IB_TC" ]] && export NCCL_IB_TC
        [[ -n "$NCCL_IB_FIFO_TC" ]] && export NCCL_IB_FIFO_TC
        unset _tc
    else
        echo "[INFO] $(hostname -s): Not a Pensando AINIC cluster, no NCCL_IB_TC/NCCL_IB_FIFO_TC override needed"
    fi
fi

# NCCL_SOCKET_IFNAME: network interface for NCCL socket communication
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    source "$_DETECT_NCCL_DIR/choose_nccl_socket_ifname.sh"
    if nccl_nic=$(choose_nccl_socket_ifname); then
        export NCCL_SOCKET_IFNAME="${nccl_nic}"
        echo "NCCL INFO $(hostname -s): NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
        if command -v ethtool &>/dev/null && ethtool -i "$NCCL_SOCKET_IFNAME" &>/dev/null; then
            echo "NIC_DRIVER_CHECK $(hostname -s) iface=$NCCL_SOCKET_IFNAME $(ethtool -i "$NCCL_SOCKET_IFNAME" | awk -F': *' '/^(driver|version|firmware-version):/{printf "%s=%s ", $1, $2}')"
        fi
    else
        if [[ "${NNODES:-1}" -gt 1 ]]; then
            echo "NCCL FATAL $(hostname -s): Failed to auto-detect NCCL_SOCKET_IFNAME; ABORTING..." >&2
            unset _DETECT_NCCL_DIR
            return 1
        else
            echo "NCCL WARN $(hostname -s): Could not auto-detect NCCL_SOCKET_IFNAME; leaving it unset" >&2
        fi
    fi
fi

unset _DETECT_NCCL_DIR
