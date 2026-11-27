"""B1 (spiking part): what the forward network actually shows while the arm moves.

The TIAGo7 forward network is driven along joint paths (the arm moved from outside, no control):

* ``speed0.25``, ``speed0.5``, ``speed1``: six random goals, minimum-jerk moves with that peak joint speed (rad/s);
* ``pole``: goals whose hand pitch alternates between beyond +80° and beyond −80° (through the chart's pole).

Every path is run twice on the forward network: with each joint ring written at the joint's angle (the old method:
cues for M0 and M2), and with every angle written ahead by the documented lead ``0.14 s + 0.01 s`` per later joint
(M1).  Each tick's decoded matrix of the hand and readout-ring angles are stored; ``estimation_summary`` replays them
as the cue of every orientation source, so the new method is judged on the network's real output.  A third run builds
the observer method's network (the observer seeded while the arm is held at the start, then fed the path's joint
velocities) and stores the matrix its gates pass to the turn: the built observer's own estimate.

**Speed replay** (``--condition common``). The paths above differ in their goals as well as their speed: each path draws
its own goals. ``common0.25``, ``common0.5`` and ``common1`` take the start and six goals of ``speed0.5`` and move
through them at each peak speed, so only the speed changes. They record the forward network, plain and written ahead,
without the observer. ``common0.5`` is ``speed0.5`` again, which checks that the recording reproduces.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import nest
import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.geometry.rotations import lift_pitch_yaw, wrap
from reaching_benchmark.progress import Progress, say
from reaching_benchmark.targets import MOTION_SEED

from .. import results
from ..control.session import lead_times
from ..network.factory import build_forward, build_pose_ik
from ..rules import observer_rule

ARM = "tiago7"
PATHS = ("speed0.25", "speed0.5", "speed1", "pole")
#: the speed replay: the goals of COMMON_GOALS at every speed
COMMON = ("common0.25", "common0.5", "common1")
COMMON_GOALS = "speed0.5"
WARM_TICKS = 30


def make_path(name: str, goals: int = 6) -> np.ndarray:
    chain = get_chain(ARM)
    common = name in COMMON
    rng = np.random.default_rng([MOTION_SEED, PATHS.index(COMMON_GOALS if common else name)])
    start = chain.random_pose(rng)
    if name == "pole":
        targets = observer_rule.pitch_sweep(chain, start, rng, goals)
        speed = 0.5
    else:
        targets = [chain.random_pose(rng) for _ in range(goals)]
        speed = float(name[len("common" if common else "speed"):])
    return observer_rule.smooth_path(chain, start, targets, speed)


def record(path: np.ndarray, lead: bool, threads: int, seed: int, label: str = "estimation") -> Dict[str, Any]:
    chain, network = build_forward(nest, ARM, threads=threads, seed=seed, level=True)
    dt = network.options.tick_ms / 1000.0
    ahead = lead_times(chain.n)
    for _ in range(WARM_TICKS):
        network.step(path[0])
    rows: Dict[str, List[Any]] = {"tool": [], "angles": [], "ring_spikes": [], "position": []}
    started = time.time()
    progress = Progress("%s %s" % (label, "written ahead" if lead else "plain"), len(path), unit="tick")
    for t in range(len(path)):
        velocity = wrap(path[t] - path[max(t - 1, 0)]) / dt
        written = chain.clip(path[t] + ahead * velocity) if lead else path[t]
        tick = network.step(written)
        rows["tool"].append(tick.tool)
        rows["angles"].append(tick.angles)
        rows["ring_spikes"].append(tick.ring_spikes)
        rows["position"].append(tick.position)
        progress.advance(1, "t %.1f/%.1f s" % ((t + 1) * dt, len(path) * dt))
    progress.close()
    return {"lead": bool(lead), "neurons": network.neuron_count(), "wall_s": time.time() - started,
            **{key: np.array(value).tolist() for key, value in rows.items()}}


def record_observer(path: np.ndarray, threads: int, seed: int, hold_ticks: int = 40) -> Dict[str, Any]:
    """The observer method's network along a path: seeded while held at the start, then moved with the path's velocities."""

    chain, network = build_pose_ik(nest, ARM, method="observer", threads=threads, seed=seed)
    dt = network.options.tick_ms / 1000.0

    def step(q: np.ndarray, velocity: np.ndarray, seed_now: bool) -> Any:
        return network.step_ik(q, lift_pitch_yaw(chain.rotation(q)), kinematics.position(chain, q), velocity=velocity, seed=seed_now)

    for t in range(hold_ticks):
        step(path[0], np.zeros(chain.n), t < hold_ticks - 10)
    rows: Dict[str, List[Any]] = {"hand": [], "selected": [], "fallback": [], "angles": [], "tool": []}
    started = time.time()
    for t in range(len(path)):
        tick = step(path[t], wrap(path[t] - path[max(t - 1, 0)]) / dt, False)
        rows["hand"].append(tick.observer.hand)
        rows["selected"].append(tick.observer.selected)
        rows["fallback"].append(tick.observer.fallback)
        rows["angles"].append(tick.observer.angles)
        rows["tool"].append(tick.forward.tool)
    return {"neurons": network.neuron_count(), "wall_s": time.time() - started, **{key: np.array(value).tolist() for key, value in rows.items()}}


def run_command(args: Any) -> None:
    if args.condition == "common" or args.condition in COMMON:
        for name in (COMMON if args.condition == "common" else (args.condition,)):
            if results.read_raw_part("moving_estimate", name):
                say("moving_estimate %s: recorded already" % name)
                continue
            path = make_path(name)
            document = {"benchmark": "moving_estimate", "arm": ARM, "path": name, "goals_of": COMMON_GOALS, "q": path.tolist(), "dt": 0.05,
                        "plain": record(path, False, args.threads, args.seed, "moving_estimate " + name),
                        "lead": record(path, True, args.threads, args.seed, "moving_estimate " + name)}
            results.write_raw("moving_estimate", name, document)
            say("moving_estimate %s: %d ticks" % (name, len(path)))
        return
    for name in ([args.condition] if args.condition else PATHS):
        earlier = results.read_raw_part("moving_estimate", name)
        if earlier and "observer" in earlier and not args.condition:
            continue                                       # recorded already (resume); --condition NAME records it again
        path = make_path(name)
        if earlier and not args.condition:                 # the forward runs are there: add the observer run
            document = earlier
        else:
            document = {"benchmark": "moving_estimate", "arm": ARM, "path": name, "q": path.tolist(), "dt": 0.05,
                        "plain": record(path, False, args.threads, args.seed), "lead": record(path, True, args.threads, args.seed)}
        document["observer"] = record_observer(path, args.threads, args.seed)
        results.write_raw("moving_estimate", name, document)
        say("moving_estimate %s: %d ticks" % (name, len(path)))
