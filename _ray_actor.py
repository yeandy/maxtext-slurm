#!/usr/bin/env python3
"""Ray actor wrapper for MaxText training.

Training runs in a subprocess (via subprocess.Popen) so that the training
process has its own Python interpreter with zero Ray threads.  This
eliminates GIL contention between Ray internals and the training loop.

The Ray actor handles:
  - Launching the subprocess on the correct node (NodeAffinity)
  - Streaming logs (subprocess inherits actor's stdout/stderr fds)
  - Collecting the exit code

Stack traces & flame graphs are available via the Ray Dashboard (port 8265).
py-spy is wrapped to target the training subprocess directly (see
ray_cluster.sh).
"""

import os
import signal
import socket
import subprocess
import sys
import traceback

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


# ---------------------------------------------------------------------------
# Ray actor  (thin launcher — no training code runs here)
# ---------------------------------------------------------------------------

@ray.remote
class MaxTextTrainerActor:
    """Ray actor that launches training in a subprocess.

    Training runs in a separate Python process with no Ray threads,
    eliminating GIL contention.  The actor handles log routing and
    result collection.
    """

    def __init__(self):
        self.hostname = socket.gethostname()
        self.node_rank = int(os.environ.get("NODE_RANK", 0))
        self.tag = f"[Node {self.node_rank} @ {self.hostname}]"

    def run_training(self, argv: list, env_vars: dict) -> int:
        """Launch training in a subprocess and wait for result.

        Uses subprocess.Popen with env=env_vars to give the training process
        a clean environment (exactly what _train.sh exported) with no Ray
        thread contamination.  stdout/stderr are inherited from the actor
        worker, so output flows through Ray's log streaming automatically.
        """
        # Resolve the mfu_tracker.py script path (same entry point as non-Ray mode)
        script_dir = env_vars.get(
            "MAXTEXT_SLURM_DIR",
            os.path.dirname(os.path.abspath(__file__)),
        )
        mfu_script = os.path.join(script_dir, "utils", "mfu_tracker.py")

        cmd = [sys.executable, "-u", mfu_script] + list(argv)

        # Ensure PYTHONUNBUFFERED is set for real-time log streaming
        launch_env = dict(env_vars)
        launch_env["PYTHONUNBUFFERED"] = "1"

        print(f"{self.tag} Launching training subprocess: {' '.join(cmd[:3])} ...",
              flush=True)

        p = subprocess.Popen(
            cmd,
            env=launch_env,
            cwd=env_vars.get("PWD") or None,
        )
        print(f"{self.tag} Training subprocess started (pid={p.pid})",
              flush=True)

        p.wait()  # block until training finishes

        # ---- report result ----
        if p.returncode == 0:
            return 0
        elif p.returncode < 0:
            sig_num = -p.returncode
            try:
                sig_name = signal.Signals(sig_num).name
            except (ValueError, AttributeError):
                sig_name = f"signal {sig_num}"
            print(f"{self.tag} Training subprocess killed by {sig_name} "
                  f"(signal {sig_num})", flush=True)
        else:
            print(f"{self.tag} Training subprocess exited with code "
                  f"{p.returncode}", flush=True)
        return p.returncode


# ---------------------------------------------------------------------------
# Driver  (one per node — connects to Ray, creates actor, waits)
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: _ray_actor.py <config.yml> [key=value ...]")
        sys.exit(1)

    train_argv = sys.argv[1:]
    node_rank = int(os.environ.get("NODE_RANK", 0))
    captured_env = dict(os.environ)

    # Ensure MAXTEXT_SLURM_DIR is in the captured env so the actor can
    # reliably resolve mfu_tracker.py.  (submit.sh exports this on the host,
    # but it is not passed as a Docker --env flag.)
    captured_env.setdefault(
        "MAXTEXT_SLURM_DIR",
        os.path.dirname(os.path.abspath(sys.argv[0])),
    )

    ray.init(address="auto", namespace="maxtext", log_to_driver=True)

    # Pin actor to the local node; num_cpus=0 since the actor is just a
    # thin launcher (training runs in a subprocess, not in this process).
    local_node_id = ray.get_runtime_context().get_node_id()
    actor = MaxTextTrainerActor.options(
        name=f"maxtext_trainer_{node_rank}",
        num_gpus=0,
        num_cpus=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=local_node_id,
            soft=False,
        ),
    ).remote()

    try:
        exit_code = ray.get(actor.run_training.remote(train_argv, captured_env))
    except Exception as e:
        print(f"[Node {node_rank}] Actor failed: {e}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        ray.kill(actor, no_restart=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
