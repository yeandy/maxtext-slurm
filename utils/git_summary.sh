#!/bin/bash

echo "=== GIT SUMMARY BEGIN ==="

# Keep provenance logging non-fatal: missing git or non-repo cwd should not
# break callers or job startup.
if ! command -v git >/dev/null 2>&1; then
  echo "[WARN] git is not available in PATH; skipping git summary."
  echo "=== GIT SUMMARY END ==="
  exit 0
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[WARN] current directory is not a git worktree: $(pwd)"
  echo "=== GIT SUMMARY END ==="
  exit 0
fi

safe_git() {
  git "$@" || true
}

# Branch & status
echo "[BRANCH]"
safe_git status --branch --short

# Last commit info (one-line)
echo
echo "[LAST COMMIT]"
safe_git --no-pager log -1 --pretty=format:"%h %s (%ad) <%an>"

# Last commit diff (compact)
echo
echo "[LAST COMMIT DIFF --stat]"
safe_git --no-pager show --stat --oneline

# Working tree changes (staged + unstaged)
echo
echo "[CHANGES --name-status]"
safe_git --no-pager diff --cached --name-status
safe_git --no-pager diff --name-status

# Full diff of working tree (compact)
echo
echo "[DIFF]"
safe_git --no-pager diff

# Untracked files
echo
echo "[UNTRACKED]"
safe_git ls-files --others --exclude-standard

echo "=== GIT SUMMARY END ==="
