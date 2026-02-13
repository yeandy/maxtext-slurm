#!/usr/bin/env bash

# rsync_gitignore.sh — rsync a directory while respecting its .gitignore

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <src_dir> <dst_dir>"
  exit 1
fi

SRC="${1%/}/"   # ensure trailing slash so rsync copies contents
DST="${2%/}/"

if [[ ! -d "$SRC" ]]; then
  echo "Error: source directory '$SRC' does not exist."
  exit 1
fi

# Use git to list ignored files if inside a repo, otherwise fall back to the
# .gitignore filter flag built into rsync.

if git -C "$SRC" rev-parse --is-inside-work-tree &>/dev/null; then
  # Best approach: let git tell us exactly what is tracked/untracked-but-not-ignored
  # We rsync only the files git knows about (tracked + untracked-not-ignored).
  rsync -av \
    --files-from=<(git -C "$SRC" ls-files --cached --others --exclude-standard) \
    "$SRC" "$DST"
else
  # Fallback: use rsync's --filter flag to read .gitignore rules directly.
  # The ':- .gitignore' syntax tells rsync to read dir-merge .gitignore files.
  rsync -av \
    --filter=':- .gitignore' \
    "$SRC" "$DST"
fi
