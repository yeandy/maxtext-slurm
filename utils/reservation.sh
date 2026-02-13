#!/bin/bash

# Usage: source utils/reservation.sh
#        RESERVATION_NAME="$(resolve_reservation "$USER")"

# Detect a GNU-compatible date command that supports -d.
_detect_date_cmd() {
  local cmd="date"
  if ! date -d '1970-01-01 00:00:00' +%s >/dev/null 2>&1; then
    if command -v gdate >/dev/null 2>&1; then
      cmd="gdate"
    else
      echo "WARNING: Your 'date' doesn't support -d. Reservation time parsing may fail; falling back to first active match." >&2
    fi
  fi
  echo "$cmd"
}

# Pick the newest currently-active reservation for a given user.
# Args: $1 = username, $2 = date command (default: auto-detect)
_get_reservation_for_user() {
  local uname="${1}"
  local datecmd="${2:-$(_detect_date_cmd)}"
  local now_epoch
  now_epoch="$("$datecmd" +%s)"
  # -o puts each reservation on one line; easier to parse
  scontrol show reservation -o 2>/dev/null | \
  awk -v user="$uname" -v now="$now_epoch" -v datecmd="$datecmd" '
    function to_epoch(ts,   cmd, epoch_str) {
      # Convert ISO-ish "YYYY-MM-DDTHH:MM:SS" to epoch using external date
      gsub(/T/, " ", ts)
      if (ts == "" || ts == "Unknown") return 0
      cmd = datecmd " -d \"" ts "\" +%s"
      epoch_str = ""
      cmd | getline epoch_str
      close(cmd)
      if (epoch_str ~ /^[0-9]+$/) return epoch_str + 0
      return 0
    }
    {
      # Extract fields
      name=""; users=""; start_s=""; end_s=""
      if (match($0, /ReservationName=([^ ]+)/, m)) name=m[1]
      if (match($0, /Users=([^ ]+)/, mu))          users=mu[1]
      if (match($0, /StartTime=([^ ]+)/, ms))      start_s=ms[1]
      if (match($0, /EndTime=([^ ]+)/, me))        end_s=me[1]
      # Exact user match in comma-separated list
      n = split(users, arr, ",")
      ok=0
      for (i=1; i<=n; i++) if (arr[i] == user) { ok=1; break }
      if (!ok) next
      start = to_epoch(start_s)
      end   = to_epoch(end_s)
      if (start==0 || end==0) {
        # If we cannot parse times, treat as active but give start=1 so it sorts low.
        start=1; end=now+1
      }
      if (start <= now && now <= end) {
        # Print start epoch and name; we will sort by start desc to pick the newest
        printf("%d\t%s\n", start, name)
      }
    }
  ' | sort -nr | awk 'NR==1 { print $2 }'
}

# Resolve the reservation for a user and print its name (empty if none found).
# Also prints informational messages to stderr.
# Args: $1 = username
resolve_reservation() {
  local uname="${1}"
  local datecmd
  datecmd="$(_detect_date_cmd)"
  local reservation
  reservation="$(_get_reservation_for_user "$uname" "$datecmd")"
  if [[ -n "$reservation" ]]; then
    echo "Using reservation for user '${uname}': ${reservation}" >&2
  else
    echo "No active reservation found for user '${uname}'. Submitting without --reservation." >&2
  fi
  echo "$reservation"
}
