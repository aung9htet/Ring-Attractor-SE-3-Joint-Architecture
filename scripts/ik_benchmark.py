#!/usr/bin/env python3
"""How fast the network with the inverse-kinematics layer runs, with nothing else on the machine.

    python3 -B scripts/ik_benchmark.py --out docs/ik/results/data/speed.json

Each model is built and run in closed loop for 10 s of simulated time (the hand on its way to a
goal near the start pose).  A value of 1 means the network keeps up with real time.  Run it
alone: other jobs on the machine change the result.
"""

import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_se3 import kinematics  # noqa: E402
from ring_se3.chains import MODEL_NAMES  # noqa: E402
from ring_se3.ik import experiments  # noqa: E402
from ring_se3.ik.design import TASKS, PoseIkDesign  # noqa: E402
from ring_se3.ik.session import PoseIkSession  # noqa: E402
from ring_so3.experiments import write_json  # noqa: E402
from ring_so3.kinematics import lift_pitch_yaw  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="*", default=["tiago4", "generic4", "generic7", "tiago7"], choices=MODEL_NAMES)
    parser.add_argument("--task", default=None, choices=TASKS, help="default: pose from six joints on, else position")
    parser.add_argument("--threads", nargs="*", type=int, default=[12])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import nest

    rows = []
    for name in args.models:
        for threads in args.threads:
            started = time.time()
            design = PoseIkDesign(task=args.task) if args.task else None
            chain, network = experiments.build_ik(nest, name, design, threads=threads)
            build_s = time.time() - started
            start = np.full(chain.n, 0.5) if chain.joints[0].circular else chain.limits().mean(axis=1)
            session = PoseIkSession(network, start)
            session.warm_up()
            there = chain.clip(start + 0.5 * np.resize([1.0, -1.0], chain.n))
            session.give(lift_pitch_yaw(chain.rotation(there)), kinematics.position(chain, there))
            ticks = int(round(args.seconds / session.dt))
            started = time.time()
            for _ in range(ticks):
                session.tick()
            wall = time.time() - started
            rows.append({"model": name, "joints": chain.n, "task": network.ik_design.task, "threads": threads, "neurons": network.neuron_count(),
                         "ik_neurons": network.ik_neuron_count(), "synapses": int(nest.GetKernelStatus("num_connections")), "build_s": build_s,
                         "simulated_s": ticks * session.dt, "wall_s": wall, "speed": ticks * session.dt / wall, "ms_per_tick": 1000.0 * wall / ticks})
            print("%-9s %-11s %2d threads: %6d neurons, built in %.1f s, %.2f x real time (%.0f ms per 50 ms tick)"
                  % (name, rows[-1]["task"], threads, rows[-1]["neurons"], build_s, rows[-1]["speed"], rows[-1]["ms_per_tick"]), flush=True)
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            cpu = next((line.split(":", 1)[1].strip() for line in stream if line.startswith("model name")), "")
    except OSError:
        pass
    write_json(args.out, {"experiment": "pose_ik_speed", "cpu": cpu or platform.processor(), "cores": os.cpu_count(), "rows": rows})
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
