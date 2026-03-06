# .host-cmd

Run commands on the host from inside a Docker container, via a shared directory.

> **WARNING**: This executes arbitrary commands on the host machine with the
> privileges of whoever starts the server. Review `policy.json` carefully
> before starting. Only run the server on machines you trust all container
> users to have host-level access to. Do NOT run this on production systems
> without proper deny/allow policies in place.

## Quick Start

### 1. On the HOST — start the server

```bash
cd /path/to/.host-cmd
./host_cmd_ctl.sh start       # start
./host_cmd_ctl.sh stop        # stop
./host_cmd_ctl.sh restart     # restart
./host_cmd_ctl.sh status      # check if running
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
