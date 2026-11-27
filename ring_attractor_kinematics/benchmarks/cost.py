"""B7: what each network costs — neurons, synapses, spikes per second and simulated seconds per wall second.

Measured on the built networks (forward network alone, and the pose IK network of each method) of TIAGo4 (position),
TIAGo7 and generic7 (pose), holding a pose for 2 s after a 1 s warm-up, 8 NEST threads, nothing else running: the old
method (``pose_ik``), the new method (``lead``) and, for pose tasks, the observer method (``observer``).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import nest
import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.geometry.rotations import lift_pitch_yaw

from .. import results
from ..network.factory import build_forward, build_pose_ik

ARMS = ("tiago4", "tiago7", "generic7")
WARM_S = 1.0
MEASURE_S = 2.0


def measure(network: Any, q: np.ndarray, step: Any) -> Dict[str, Any]:
    recorder = nest.Create("spike_recorder")
    nest.Connect(nest.GetNodes({"model": "iaf_psc_alpha"}) + nest.GetNodes({"model": "iaf_psc_alpha_multisynapse"}), recorder)
    for _ in range(int(WARM_S / 0.05)):
        step(q)
    nest.SetStatus(recorder, {"n_events": 0})
    started = time.time()
    for _ in range(int(MEASURE_S / 0.05)):
        step(q)
    wall = time.time() - started
    return {"neurons": network.neuron_count(), "synapses": int(nest.GetKernelStatus("num_connections")),
            "spikes_per_s": float(nest.GetStatus(recorder, "n_events")[0]) / MEASURE_S, "speed": MEASURE_S / wall}


def run_command(args: Any) -> None:
    rows = []
    for arm in ([args.arm] if args.arm else ARMS):
        chain = get_chain(arm)
        _, forward = build_forward(nest, arm, threads=args.threads, seed=args.seed, level=True)
        row = {"arm": arm, "joints": chain.n, "forward": measure(forward, chain.start_pose(), forward.step)}
        _, network = build_pose_ik(nest, arm, threads=args.threads, seed=args.seed)
        q = chain.start_pose()
        lpy, position = np.array(lift_pitch_yaw(chain.rotation(q))), kinematics.position(chain, q)
        row["pose_ik"] = measure(network, q, lambda angles: network.step_ik(angles, lpy, position))
        row["task"] = network.ik_design.task
        methods = ("lead", "observer") if network.ik_design.orientation else ("lead",)
        for method in methods:
            _, other = build_pose_ik(nest, arm, method=method, threads=args.threads, seed=args.seed)
            row[method] = measure(other, q, lambda angles: other.step_ik(angles, lpy, position))
        rows.append(row)
        print("cost %s: forward %d neurons, IK %d neurons, %.2f× real time" % (arm, row["forward"]["neurons"], row["pose_ik"]["neurons"],
                                                                             row["pose_ik"]["speed"]), flush=True)
    results.write_raw("cost", "all", {"benchmark": "cost", "threads": args.threads, "rows": rows})


def summarise(name: str = "cost") -> Dict[str, Any]:
    return results.read_raw("cost")[0]
