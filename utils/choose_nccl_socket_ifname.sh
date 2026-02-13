#!/usr/bin/env bash

# Return a "good" interface name for NCCL_SOCKET_IFNAME on stdout.
#
# This script selects a network interface suitable for NCCL/RCCL socket communication
# in a deterministic way that produces the SAME result on all nodes in a cluster.
#
# Logic:
#   1. If NCCL_SOCKET_IFNAME is already set to a valid value, reuse it.
#   2. Query all global IPv4 interfaces and sort them deterministically.
#   3. Prefer interfaces in this order:
#      - 10.x.x.x (private class A)
#      - 172.16.x.x - 172.31.x.x (private class B)
#      - 192.168.x.x (private class C)
#      - Any other non-loopback, non-virtual interface
#   4. Within each tier, pick the interface with the lowest IP address.
#
# This ensures all nodes in a multi-node job select the same interface name,
# which is critical for NCCL/RCCL to establish consistent connections.

choose_nccl_socket_ifname() {
  # --- 1. Check if NCCL_SOCKET_IFNAME is already set reasonably -----------
  # If the environment already has a valid interface name, just use it.
  # This allows manual overrides to take precedence.
  if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    case "$NCCL_SOCKET_IFNAME" in
      lo|docker*|cni*|virbr*|veth*|br-* )
        # These are loopback, container, or virtual bridge interfaces
        # that should never be used for NCCL - ignore them
        ;;
      * )
        # Looks reasonable, use it
        echo "$NCCL_SOCKET_IFNAME"
        return 0
        ;;
    esac
  fi

  local nic=""

  # --- Helper function: check if interface name is unsuitable --------------
  # Returns 0 (true) if the NIC should be rejected, 1 (false) if it's acceptable.
  _nccl_is_bad_nic() {
    case "$1" in
      ""|lo|docker*|cni*|virbr*|veth*|br-* )
        return 0 ;;  # bad - loopback, empty, or virtual interface
      * )
        return 1 ;;  # good - potential candidate
    esac
  }

  # --- Helper function: explain why a NIC matches a priority tier ----------
  _explain_priority_match() {
    local ip="$1"
    if [[ "$ip" =~ ^10\. ]]; then
      echo "matches priority 1 (10.x.x.x private network)"
    elif [[ "$ip" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]]; then
      echo "matches priority 2 (172.16-31.x.x private network)"
    elif [[ "$ip" =~ ^192\.168\. ]]; then
      echo "matches priority 3 (192.168.x.x private network)"
    else
      echo "public/other IP (priority 4)"
    fi
  }

  # --- 2. Build list of candidate interfaces --------------------------------
  # Get all IPv4 addresses with global scope (not link-local, not loopback).
  # Format: "<interface_name> <ip_address/prefix>"
  #
  # Sort by interface name first for deterministic ordering across nodes.
  # This is critical: without sorting, 'ip' command output order can vary
  # between nodes or between reboots based on interface initialization order.
  local candidates
  candidates=$(ip -o -4 addr show scope global 2>/dev/null | awk '{print $2, $4}' | sort -k1,1)

  # --- 3. Select interface by priority --------------------------------------

  # Priority 1: Prefer 10.x.x.x addresses (private class A network)
  # These are commonly used for high-performance interconnects.
  #
  # Strategy: Print IP and interface name, sort by IP (using version sort for
  # proper numeric ordering), take the first (lowest IP), then extract the interface name.
  # This ensures if multiple 10.x interfaces exist, we pick the same one on all nodes.
  nic=$(echo "$candidates" | awk '$2 ~ /^10\./ {print $2, $1}' | sort -V -k1,1 | head -n1 | awk '{print $2}')

  # Priority 2: Try 172.16.x.x - 172.31.x.x (private class B network)
  # This regex matches 172.16-31.x.x ranges only.
  if [[ -z "$nic" ]]; then
    nic=$(echo "$candidates" | awk '$2 ~ /^172\.(1[6-9]|2[0-9]|3[0-1])\./ {print $2, $1}' | sort -V -k1,1 | head -n1 | awk '{print $2}')
  fi

  # Priority 3: Try 192.168.x.x (private class C network)
  if [[ -z "$nic" ]]; then
    nic=$(echo "$candidates" | awk '$2 ~ /^192\.168\./ {print $2, $1}' | sort -V -k1,1 | head -n1 | awk '{print $2}')
  fi

  # Priority 4: Fall back to first non-bad global interface
  # If no private IP ranges found, iterate through sorted candidates
  # and pick the first one that passes validation.
  if [[ -z "$nic" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      local dev=$(echo "$line" | awk '{print $1}')
      if ! _nccl_is_bad_nic "$dev"; then
        nic="$dev"
        break
      fi
    done < <(printf '%s\n' "$candidates")
  fi

  # --- 4. Return the selected interface or fail -----------------------------
  # Final validation: make sure we got something and it's not on the bad list
  if [[ -n "$nic" ]] && ! _nccl_is_bad_nic "$nic"; then
    echo "$nic"
    return 0
  fi

  # If we reach here, we couldn't find any suitable interface.
  # Print diagnostics to help debug the failure.
  echo "[ERROR] Failed to select NCCL socket interface" >&2
  echo "" >&2

  if [[ -z "$candidates" ]]; then
    echo "[CAUSE] No global IPv4 interfaces found" >&2
    echo "[DEBUG] Output of 'ip -o -4 addr show scope global':" >&2
    ip -o -4 addr show scope global 2>&1 | sed 's/^/  /' >&2
  else
    echo "[DEBUG] Selection process for each interface:" >&2
    echo "" >&2

    local has_private=0
    local has_public_suitable=0
    local public_interface=""

    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      local iface_dev=$(echo "$line" | awk '{print $1}')
      local iface_ip=$(echo "$line" | awk '{print $2}')
      local ip_only=$(echo "$iface_ip" | cut -d'/' -f1)

      echo "  Interface: $iface_dev ($iface_ip)" >&2

      # Check if it's a bad NIC
      if _nccl_is_bad_nic "$iface_dev"; then
        echo "    ✗ REJECTED: Interface name matches exclusion pattern" >&2
        echo "      (docker*, cni*, virbr*, veth*, br-*, lo)" >&2
      else
        # Check priority tier
        if [[ "$ip_only" =~ ^10\. ]]; then
          has_private=1
          echo "    ✓ Priority 1: 10.x.x.x private network - SHOULD BE SELECTED" >&2
        elif [[ "$ip_only" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]]; then
          has_private=1
          echo "    ✓ Priority 2: 172.16-31.x.x private network - SHOULD BE SELECTED" >&2
        elif [[ "$ip_only" =~ ^192\.168\. ]]; then
          has_private=1
          echo "    ✓ Priority 3: 192.168.x.x private network - SHOULD BE SELECTED" >&2
        else
          has_public_suitable=1
          public_interface="$iface_dev"
          echo "    ~ Priority 4: Public/other IP" >&2
          echo "      WARNING: Public IPs are last resort and may cause issues" >&2
          echo "      in multi-node clusters if other nodes have private networks" >&2
        fi
      fi
      echo "" >&2
    done < <(printf '%s\n' "$candidates")

    echo "[ANALYSIS]" >&2
    if [[ $has_private -eq 0 ]] && [[ $has_public_suitable -eq 1 ]]; then
      echo "  This node has NO private network interfaces (10.x, 172.16-31.x, 192.168.x)" >&2
      echo "  Only public IP interface available: $public_interface" >&2
      echo "" >&2
      echo "[CAUSE] Network configuration mismatch in multi-node cluster" >&2
      echo "  Other nodes likely selected a private network interface," >&2
      echo "  but this node only has public IPs. This breaks NCCL's requirement" >&2
      echo "  that all nodes use the same interface name." >&2
      echo "" >&2
      echo "[SOLUTION] Either:" >&2
      echo "  1. Set NCCL_SOCKET_IFNAME=$public_interface on ALL nodes (if reachable)" >&2
      echo "  2. Configure a private network interface on this node" >&2
      echo "  3. Use a different network topology where all nodes have consistent IPs" >&2
    elif [[ $has_private -eq 1 ]]; then
      echo "  Private network interfaces exist but were not selected" >&2
      echo "" >&2
      echo "[CAUSE] Selection logic bug - private interfaces should have been chosen" >&2
      echo "[DEBUG] Final nic value: '$nic'" >&2
    else
      echo "  All interfaces were rejected (only virtual/loopback interfaces found)" >&2
      echo "" >&2
      echo "[CAUSE] No suitable physical network interfaces available" >&2
    fi
  fi

  echo "" >&2

  return 1
}
