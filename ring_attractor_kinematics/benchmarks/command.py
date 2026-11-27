"""``python -m ring_attractor_kinematics benchmark NAME`` and ``python -m ring_attractor_kinematics summarise NAME``.

Every benchmark writes its runs to ``results/ring_attractor_kinematics/NAME/`` and its summary to
``results/ring_attractor_kinematics/NAME/summary.json``.

==================== ===================================================== ==========================================
name                 question                                               module
==================== ===================================================== ==========================================
screening            which orientation source is worth a spiking build?     benchmarks/screening.py
                     (exact loop, every source, swept lag and cue noise)
held_pose            the forward network at rest: 30 held poses on 12 arms  benchmarks/forward.py
held_pose_neuron_    the same poses under diffuse neuron loss                benchmarks/forward.py
loss                 (``held_pose --condition loss_sweep``)
moving_estimate      the forward network while the arm moves (speed         benchmarks/estimation.py
                     replay, shared common path)
reach_kinematic      closed-loop reaching on the kinematic plant             benchmarks/reach.py
reach_pole           goals near the lift-pitch-yaw pole                     benchmarks/reach.py
reach_gain           loop gain κ = 1.5, 2                                   benchmarks/reach.py
reach_disturbance    a push of one joint; noise current                     benchmarks/reach.py
reach_gazebo         closed-loop reaching with the TIAGo in Gazebo          benchmarks/gazebo_reach.py
joint_loss           one joint disabled 2 s into each reach (lock, motor,   benchmarks/reach.py
                     allocation; diffuse loss)
joint_loss_gazebo    the same with the TIAGo in Gazebo                      benchmarks/gazebo_reach.py
joint_map            every joint disabled in turn, exact loop (no NEST)     benchmarks/joint_map.py
cost                 neurons, synapses, spikes, speed                       benchmarks/cost.py
structure            neurons, wiring, divisors from the code (no NEST)      benchmarks/structure.py
twins                recompute the exact-loop twins of every stored trial   benchmarks/twins.py
==================== ===================================================== ==========================================

``--variant`` picks the model variant: none (the default, the hand's late matrix), ``predictive_encoding`` (joint
angles written ahead by the commanded velocity) or ``observer`` (two-chart orientation observer).

Every long run shows a progress bar (reaching_benchmark/progress.py); ``python -m reaching_benchmark status`` shows every
run at once.
"""

from __future__ import annotations

import argparse
import importlib

from .. import results

#: name -> (module that runs it, module that summarises it)
P = "ring_attractor_kinematics.benchmarks."
MODULES = {"screening": (P + "screening", P + "screening"),
           "moving_estimate": (P + "estimation", P + "estimation_summary"),
           "held_pose": (P + "forward", P + "forward_summary"),
           "held_pose_neuron_loss": (P + "forward", P + "forward_summary"),
           "cost": (P + "cost", P + "cost"),
           "structure": (P + "structure", P + "structure"),
           "reach_gazebo": (P + "gazebo_reach", P + "reach_summary"),
           "joint_loss": (P + "reach", P + "joint_loss_summary"),
           "joint_loss_gazebo": (P + "gazebo_reach", P + "joint_loss_summary"),
           "joint_map": (P + "joint_map", P + "joint_map"),
           "twins": (P + "twins", None),
           **{name: (P + "reach", P + "reach_summary") for name in ("reach_kinematic", "reach_pole", "reach_gain", "reach_disturbance")}}
#: the closed-loop benchmarks whose trials carry exact-loop twins
CLOSED_LOOP = ("reach_kinematic", "reach_pole", "reach_gain", "reach_disturbance", "reach_gazebo")
#: ``--variant`` → the code's network method alias
VARIANTS = {"predictive_encoding": "lead", "observer": "observer"}
#: the code's network method alias → the variant name used in run file names
VARIANT_NAMES = {value: key for key, value in VARIANTS.items()}


def add_arguments(parser: argparse.ArgumentParser, command: str) -> None:
    parser.add_argument("name", choices=sorted(MODULES))
    if command == "summarise":
        return
    parser.add_argument("--pairs", type=int, default=None, help="trials per arm (default: the benchmark's own)")
    parser.add_argument("--arm", default=None, help="one arm only (default: the benchmark's arms)")
    parser.add_argument("--variant", default=None, choices=sorted(VARIANTS), help="model variant (default: the hand's late matrix)")
    parser.add_argument("--condition", default=None, help="one condition only (benchmark-specific, e.g. kappa=1.5)")
    parser.add_argument("--processes", type=int, default=20, help="worker processes for the exact-loop benchmarks")
    parser.add_argument("--threads", type=int, default=8, help="NEST threads")
    parser.add_argument("--seed", type=int, default=1, help="NEST kernel seed")
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th trial from the i-th on")
    parser.add_argument("--max-minutes", type=float, default=None, help="stop before the next trial after this long")
    parser.add_argument("--onset", type=float, default=None, help="joint_loss: when the joint is disabled, s (default 2)")
    parser.add_argument("--loss", type=float, default=0.1, help="diffuse neuron loss: the fraction of neurons silenced (default 0.1)")
    parser.add_argument("--loss-class", default="all", help="diffuse neuron loss: all, or a comma list of rings, value, sheets")
    parser.add_argument("--lesion-seed", type=int, default=1, help="diffuse neuron loss: which neurons are silenced (seed)")


def main(args: argparse.Namespace, command: str) -> int:
    runner, summariser = MODULES[args.name]
    if command == "summarise":
        path = results.write_summary(args.name, importlib.import_module(summariser).summarise(args.name))
        print("wrote", path)
        return 0
    args.method = VARIANTS.get(args.variant) if args.variant else None
    importlib.import_module(runner).run_command(args)
    if args.name == "twins":
        names = [n for n in (args.condition.split(",") if args.condition else CLOSED_LOOP) if results.read_raw(n)]
    elif args.name == "held_pose" and args.condition in ("loss", "loss_sweep"):
        names = ["held_pose_neuron_loss"]
    else:
        names = [args.name]
    for name in names:
        path = results.write_summary(name, importlib.import_module(MODULES[name][1]).summarise(name))
        print("wrote", path)
    return 0
