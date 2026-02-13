#!/bin/bash

echo "=== GIT SUMMARY BEGIN ==="

# Branch & status
echo "[BRANCH]"
git status --branch --short

# Last commit info (one-line)
echo
echo "[LAST COMMIT]"
git --no-pager log -1 --pretty=format:"%h %s (%ad) <%an>"

# Last commit diff (compact)
echo
echo "[LAST COMMIT DIFF --stat]"
git --no-pager show --stat --oneline

# Working tree changes (staged + unstaged)
echo
echo "[CHANGES --name-status]"
git --no-pager diff --cached --name-status
git --no-pager diff --name-status

# Full diff of working tree (compact)
echo
echo "[DIFF]"
git --no-pager diff

# Untracked files
echo
echo "[UNTRACKED]"
git ls-files --others --exclude-standard

echo "=== GIT SUMMARY END ==="
