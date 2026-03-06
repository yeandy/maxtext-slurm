You are running inside a Docker container. You cannot run host commands (like `squeue`, `sbatch`, `sinfo`, `scancel`, `scontrol`, `ssh`) directly. To execute commands on the host, use `.host-cmd/host_cmd.py`. Example: `python3 /maxtext-slurm/.host-cmd/host_cmd.py "squeue -u $USER"`. You can also use it for ssh, e.g. `python3 /maxtext-slurm/.host-cmd/host_cmd.py "ssh node01 rocm-smi"`. All commands must be non-interactive — always pass a command to ssh, never open a bare shell. Use `--ping` to check if the server is running. If it's not running, tell the user to start it on the host with: `cd .host-cmd && ./host_cmd_ctl.sh start`

**CAUTION**: This might be a shared cluster. NEVER `scancel` other users' jobs. Job owner fields are unreliable (e.g. all jobs show as `root`). To tell if a job is yours: check if a directory for that job ID exists under `outputs/`. If it does, the job is yours and you can cancel it. If not, it belongs to someone else — do not touch it.

For performance analysis tasks, follow the instructions in `skills/performance-analysis/SKILL.md`.
For job triage tasks (failed, hanging, or running jobs), follow the instructions in `skills/job-log-triage/SKILL.md`.
For TSDB diagnosis tasks (metrics queries, GPU/network health, incident root cause), follow the instructions in `skills/tsdb-diagnosis/SKILL.md`.
For coredump debugging (GDB analysis, source code identification, crash root cause from core files), follow the instructions in `skills/coredump-debug/SKILL.md`.
For model config tasks (adding a model, creating .gpu.yml configs, parallelism, batch size, quantization), follow the instructions in `skills/model-config-guide/SKILL.md`.

**Multi-job performance comparisons** (e.g., "why is job B slower than job A?"): Start with `skills/tsdb-diagnosis/SKILL.md` (Multi-Job Comparison workflow) to check system-level metrics (process counts, network, I/O, GPU health) before running `skills/performance-analysis/SKILL.md`. TSDB surfaces root causes that TraceLens cannot see (CPU contention, RCCL resource leaks, network errors). Only fall back to TraceLens if the TSDB comparison is inconclusive.
