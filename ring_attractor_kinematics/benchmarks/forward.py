"""Fig. 2a–b: the forward network at rest — held-pose orientation and position error on all twelve arms.

Each arm's forward network (with level cells on its sheets, as the IK uses it) is moved by the moving robot
to 30 held poses (``POSE_SEED``); the readout is averaged over the last 0.5 s of a 1 s hold and compared with
the exact kinematics.

**Diffuse neuron loss** (``--condition loss_sweep``, TIAGo7 unless ``--arm``). The same 30 held poses are measured with
a fraction of the neurons of every population of some classes silenced from the build on
(:meth:`~ring_attractor_kinematics.network.forward.ForwardNetwork.lesion_fraction`). The sweep runs every class together at 5, 10, 20
and 30%, and each class alone (joint rings, value populations, sheets) at 10%, each with lesion seeds 1–3: 21 runs. The
intact reference is the arm's own file. ``--condition loss --loss F --loss-class C --lesion-seed S`` runs one of them.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import nest

from reaching_benchmark.plants.light_robot import MovingRobot
from reaching_benchmark.geometry.chains import MODEL_NAMES
from reaching_benchmark.progress import Progress, say
from reaching_benchmark.targets import POSE_SEED, test_poses

from .. import results
from ..control.session import ForwardSession
from ..network.factory import build_forward
from ..network.forward import LESION_CLASSES

POSES = 30
SWEEP_FRACTIONS = (0.05, 0.1, 0.2, 0.3)
SWEEP_CLASS_FRACTION = 0.1
SWEEP_LESION_SEEDS = (1, 2, 3)
#: (fraction, classes, lesion seed)
Loss = Tuple[float, str, int]


def loss_sweep() -> List[Loss]:
    every = [(fraction, "all", seed) for fraction in SWEEP_FRACTIONS for seed in SWEEP_LESION_SEEDS]
    return every + [(SWEEP_CLASS_FRACTION, name, seed) for name in LESION_CLASSES for seed in SWEEP_LESION_SEEDS]


def loss_part(arm: str, loss: Loss) -> str:
    fraction, classes, seed = loss
    return "%s_loss%g_%s_l%d" % (arm, fraction, classes.replace(",", "+"), seed)


def measure(arm: str, args: Any, label: str, loss: Optional[Loss] = None) -> Dict[str, Any]:
    """The held poses of one arm (``loss``: with that diffuse lesion), as a raw document."""

    started = time.time()
    chain, network = build_forward(nest, arm, threads=args.threads, seed=args.seed, level=True)
    lesion = network.lesion_fraction(loss[0], loss[2], loss[1].split(",")) if loss is not None else None
    session = ForwardSession(network, MovingRobot(chain, chain.start_pose(), max_speed=1.0))
    session.warm_up()
    poses = test_poses(chain, args.pairs or POSES, POSE_SEED)
    progress = Progress(label, len(poses), unit="pose")
    rows = []
    for index, q in enumerate(poses):
        rows.append(session.go_to(q))
        progress.advance(1, "pose %d held" % index)
    progress.close()
    document = {"benchmark": "held_pose_neuron_loss" if lesion is not None else "held_pose", "arm": arm, "joints": chain.n, "radius_m": network.space.radius,
                "neurons": network.neuron_count(), "synapses": int(nest.GetKernelStatus("num_connections")),
                "simulated_s": network.t_ms / 1000.0, "wall_s": time.time() - started, "planned": len(poses), "rows": rows}
    if lesion is not None:
        document["lesion"] = lesion
    return document


def run_command(args: Any) -> None:
    if args.name == "held_pose_neuron_loss" and not args.condition:
        args.condition = "loss_sweep"
    if args.condition in ("loss", "loss_sweep"):
        arm = args.arm or "tiago7"
        runs = loss_sweep() if args.condition == "loss_sweep" else [(float(args.loss), str(args.loss_class), int(args.lesion_seed))]
        for number, loss in enumerate(runs, 1):
            part = loss_part(arm, loss)
            if results.read_raw_part("held_pose_neuron_loss", part):
                say("held_pose_neuron_loss %s: measured already" % part)
                continue
            results.write_raw("held_pose_neuron_loss", part, measure(arm, args, "held_pose_neuron_loss %s (%d/%d)" % (part, number, len(runs)), loss))
        return
    arms = [args.arm] if args.arm else MODEL_NAMES
    shard = tuple(int(v) for v in args.shard.split("/"))
    for position, arm in enumerate(arms):
        if position % shard[1] != shard[0] or results.read_raw_part("held_pose", arm):
            continue                                       # another shard's arm, or measured already (resume)
        document = measure(arm, args, "held_pose %s" % arm)
        results.write_raw("held_pose", arm, document)
        say("held_pose %s: %d poses, %.0f s" % (arm, len(document["rows"]), document["wall_s"]))
