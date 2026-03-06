# .host-cmd

Run commands on the host from inside a Docker container, via a shared directory.

> **WARNING**: This executes arbitrary commands on the host machine with the
> privileges of whoever starts the server. Review the policy carefully
> before starting. Only run the server on machines you trust all container
> users to have host-level access to. Do NOT run this on production systems
> without proper deny/allow policies in place.

## Quick Start

### 1. On the HOST — start the server

```bash
cd /path/to/.host-cmd
./host_cmd_ctl.sh start
```

### 2. From the CONTAINER — run commands

```bash
python3 .host-cmd/host_cmd.py "hostname"
python3 .host-cmd/host_cmd.py --ping
python3 .host-cmd/host_cmd.py --status
```

Or as a library:

```python
from host_cmd import HostBridge
br = HostBridge()
result = br.run("hostname")
print(result["stdout"])
```

## Server Management

All commands run on the host via `host_cmd_ctl.sh`:

```bash
./host_cmd_ctl.sh start       # start server
./host_cmd_ctl.sh stop        # stop server
./host_cmd_ctl.sh restart     # restart (e.g. after policy change)
./host_cmd_ctl.sh status      # check if running
./host_cmd_ctl.sh history     # list recent commands and exit codes
./host_cmd_ctl.sh cleanup     # delete all result files
./host_cmd_ctl.sh cleanup 24  # delete results older than 24 hours
```

## Policy

Commands are filtered by deny/allow regex patterns. The server loads
the policy once at startup (immutable at runtime).

- `policy.json.default` — shipped template with sensible defaults (tracked in git)
- `policy.json` — user customization (not tracked in git)

If `policy.json` exists, it is used. Otherwise `policy.json.default` is
used. The user's file fully replaces the default (no merging).

Manage via `host_cmd_ctl.sh` (auto-restarts the server to apply):

```bash
./host_cmd_ctl.sh policy              # show current rules
./host_cmd_ctl.sh deny '\bwhoami\b'   # add a deny pattern
./host_cmd_ctl.sh allow '^squeue'     # add an allow pattern
./host_cmd_ctl.sh undeny '\bwhoami\b' # remove a deny pattern
./host_cmd_ctl.sh unallow '^squeue'   # remove an allow pattern
```

Evaluation order: deny first, then allow. If `allow_patterns` is empty,
all non-denied commands are permitted. If non-empty, only matching
commands pass.

## Architecture

Multiple containers can share one server. Each command gets a unique ID,
so there are no collisions. Different users with separate shared
directories run their own independent servers.

```
                  shared directory (.host-cmd/)
                 +---------------------------+
Container A ---> |                           |
Container B ---> |  queue/*.cmd  -------->   | ---> host_cmd_server.py
Container C ---> |  results/*.json  <-----   |      (on the host)
                 +---------------------------+
```
