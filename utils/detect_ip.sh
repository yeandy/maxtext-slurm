#!/bin/bash
# detect_ip.sh — Detect the public (or local) IP address of this host.
#
# Usage:
#   source utils/detect_ip.sh
#   ip="$(detect_ip)"            # best-effort IP string, never empty
#
# Strategy:
#   1. curl external services (ifconfig.me, icanhazip.com)
#   2. Validate the response looks like an IP (not a proxy error page)
#   3. Fall back to hostname -I (local/private IP)
#   4. Last resort: "ip-unknown"

detect_ip() {
    local ip
    ip=$(curl -s --connect-timeout 2 ifconfig.me 2>/dev/null || \
         curl -s --connect-timeout 2 icanhazip.com 2>/dev/null || \
         echo "")
    # Validate: curl can succeed (exit 0) yet return a proxy error page
    # instead of an IP.  Accept only IPv4/IPv6-shaped strings.
    if ! [[ "$ip" =~ ^[0-9a-fA-F.:]+$ ]]; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        [[ -z "$ip" ]] && ip="ip-unknown"
    fi
    echo "$ip"
}
