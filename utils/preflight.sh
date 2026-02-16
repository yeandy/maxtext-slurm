#!/bin/bash

# preflight.sh — Per-host pre-flight checks (run via srun).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=$(hostname -s)

# ---- Container + GPU cleanup ----
bash "$SCRIPT_DIR/utils/release_gpu.sh"

# ---- Disk cleanup ----
disk_before=$(df -BG / 2>/dev/null | awk 'NR==2 {print $4}')
# Stale temp files from previous jobs (older than 1 day, includes core dumps)
find /tmp -maxdepth 1 -user "$(id -u)" -mtime +1 -exec rm -rf {} + 2>/dev/null || true
find /var/tmp -maxdepth 1 -user "$(id -u)" -mtime +1 -exec rm -rf {} + 2>/dev/null || true
# Stopped containers and dangling (untagged) images — preserves cached tagged images
source "$SCRIPT_DIR/utils/docker_utils.sh"
if _DOCKER_BIN="$(get_docker_bin 2>/dev/null)"; then
    read -ra _DOCKER_CMD <<< "$_DOCKER_BIN"
    "${_DOCKER_CMD[@]}" container prune -f 2>/dev/null || true
    "${_DOCKER_CMD[@]}" image prune -f 2>/dev/null || true
fi
disk_after=$(df -BG / 2>/dev/null | awk 'NR==2 {print $4}')
echo "$HOST | disk free: $disk_before -> $disk_after (cleanup)"

# ---- NUMA auto-balancing ----
numa_before=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo "N/A")
# Uncomment next 3 lines to disable NUMA auto-balancing:
#if [ "$numa_before" = "1" ]; then
#  echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null 2>&1
#fi
numa_after=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo "N/A")
echo "$HOST | numa_balancing: before=$numa_before after=$numa_after"

# ---- Leaked semaphores (safe: --exclusive gives us sole access to the node) ----
sem_before=$(ls -1 /dev/shm/sem.* 2>/dev/null | wc -l)
rm -f /dev/shm/sem.* 2>/dev/null
sem_after=$(ls -1 /dev/shm/sem.* 2>/dev/null | wc -l)
echo "$HOST | semaphores: before=$sem_before after=$sem_after deleted=$((sem_before - sem_after))"

# ---- Transparent Huge Pages ----
thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)
thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)
echo "$HOST | THP before | enabled: $thp_enabled | defrag: $thp_defrag"
{ echo never > /sys/kernel/mm/transparent_hugepage/enabled && \
  echo never > /sys/kernel/mm/transparent_hugepage/defrag; } 2>/dev/null || \
{ echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled \
                         /sys/kernel/mm/transparent_hugepage/defrag > /dev/null 2>&1; }
thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)
thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)
echo "$HOST | THP after  | enabled: $thp_enabled | defrag: $thp_defrag"
