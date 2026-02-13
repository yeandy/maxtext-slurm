#!/usr/bin/env bash

###############################################################################
# Public: get_docker_bin
# Detects a usable container runtime.
#
# Returns a string that may contain multiple words (e.g. "sudo docker").
# Callers MUST split into an array before invoking:
#
#   DOCKER_BIN="$(get_docker_bin)" || exit 1
#   read -ra DOCKER_CMD <<< "$DOCKER_BIN"
#   "${DOCKER_CMD[@]}" ps -q
#
# Or for simple unquoted usage (relies on word splitting — fragile):
#   $DOCKER_BIN ps -q
###############################################################################

get_docker_bin() {
    # 1. Prefer podman if installed AND usable
    if command -v podman >/dev/null 2>&1; then
        local runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

        if [ -d "$runtime_dir" ] && podman info >/dev/null 2>&1; then
            echo "podman"
            return 0
        fi
    fi

    # 2. Fallback to docker
    if command -v docker >/dev/null 2>&1; then
        if docker info >/dev/null 2>&1; then
            echo "docker"
        else
            echo "sudo docker"
        fi
        return 0
    fi

    echo "ERROR: Neither usable podman nor docker found in PATH" >&2
    return 1
}

###############################################################################
# Internal: _docker_free_gb
# Get free disk space (GB) on Docker's storage filesystem.
#
# Args: docker_bin_string (may be "sudo docker")
###############################################################################

_docker_free_gb() {
    local -a cmd
    read -ra cmd <<< "$1"
    local root

    # Docker uses DockerRootDir; podman uses Store.GraphRoot
    root="$("${cmd[@]}" info --format '{{.DockerRootDir}}' 2>/dev/null \
        || "${cmd[@]}" info --format '{{.Store.GraphRoot}}' 2>/dev/null \
        || echo /var/lib/containers/storage)"
    df -BG "$root" 2>/dev/null | awk 'NR==2 {sub("G","",$4); print $4}'
}

###############################################################################
# Public: docker_smart_prune
# Delete oldest images until required free space is reached.
#
# Usage:
#   docker_smart_prune "$docker_bin" 50
###############################################################################

docker_smart_prune() {
    local -a cmd
    read -ra cmd <<< "$1"
    local need_gb="${2:-50}"
    local free

    free="$(_docker_free_gb "$1")"
    free="${free:-0}"

    if ! [[ "$free" =~ ^[0-9]+$ ]]; then
        echo "[WARN] Could not determine Docker free space."
        return 0
    fi

    if [ "$free" -ge "$need_gb" ]; then
        echo "[INFO] Enough free space: ${free}G >= ${need_gb}G"
        return 0
    fi

    echo "[INFO] Free space low (${free}G). Pruning oldest images until ${need_gb}G..."

    local img
    for img in $("${cmd[@]}" images --format '{{.CreatedAt}} {{.ID}}' | sort | awk '{print $NF}'); do
        free="$(_docker_free_gb "$1")"
        [ "$free" -ge "$need_gb" ] && break

        echo "[INFO] Removing image $img ..."
        "${cmd[@]}" rmi -f "$img" >/dev/null 2>&1 || true
    done

    free="$(_docker_free_gb "$1")"
    echo "[INFO] Free space after prune: ${free}G"
}
