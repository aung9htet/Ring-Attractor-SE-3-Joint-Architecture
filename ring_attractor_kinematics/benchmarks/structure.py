"""``python -m ring_attractor_kinematics benchmark structure``: what the networks are made of, counted from the code (no NEST, seconds).

For TIAGo 4-DoF (position), TIAGo 7-DoF and generic 7-DoF (pose), under both methods: neurons by technique, the forward
network alone, the wiring (products and sums of the forward and inverse layers), each joint's divisor ``N_k`` and cap,
and the sampled relaxation ``ω_k = ‖h_k‖² / N_k`` of the built cascade against the exact coordinate step.  The model
explanation quotes these numbers; nothing here is a measurement of spikes.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain, workspace
from reaching_benchmark.targets import STUDY_SEED

from .. import results
from ..calibration.artifacts import Artifact
from ..config import IkDesign, default_task
from ..network.budget import TECHNIQUES, forward_budget, neuron_budget, observer_budget
from ..network.forward_wiring import forward_wiring
from ..network.ik_wiring import Half, Mul, Relay, ik_wiring
from ..rules.cascade_rule import levers, normalisers, share_caps

ARMS = ("tiago4", "tiago7", "generic7")
#: held-out poses the relaxation ω_k is sampled over (a seed apart from the one the divisors are averaged over)
OMEGA_SAMPLES = 4000


def relaxation(chain: Any, design: IkDesign) -> Dict[str, Any]:
    """``ω_k = (α + β|m_k|²) / N_k`` over held-out poses: below 1 the built step is short of the exact coordinate step,
    above 2 a stage can make the residual grow (in the ideal, uncapped rule)."""

    norms = normalisers(design, levers(chain))
    rng = np.random.default_rng([STUDY_SEED, chain.n, 5])
    omega = np.array([(design.turn_weight + design.move_weight * np.sum(kinematics.moves(chain, chain.random_pose(rng)) ** 2, axis=1)) / norms
                      for _ in range(OMEGA_SAMPLES)])
    return {"largest": float(omega.max()), "largest_by_joint": omega.max(axis=0).tolist(), "median_by_joint": np.median(omega, axis=0).tolist(),
            "share_above_2": float(np.mean(omega > 2.0)), "stages_above_2": int(np.sum(omega.max(axis=0) > 2.0))}


def ring_bump() -> Dict[str, Any]:
    """The held bump of a joint ring, from the stored calibration's measured profile: how many neurons fire, the rate of the
    11 neurons the encoder drives, and the rates of the flanks the ring itself recruits."""

    profile = Artifact.load("value_populations").arrays["ring_profile"]
    offsets = (np.arange(profile.size) + profile.size // 2) % profile.size - profile.size // 2
    active = profile > 0.05 * profile.max()
    flank = profile[active & (np.abs(offsets) > 5)]
    return {"active": int(active.sum()), "driven_hz": float(profile[np.abs(offsets) <= 5].mean()),
            "flank_min_hz": float(flank.min()), "flank_max_hz": float(flank.max()), "total_hz": float(profile.sum())}


def describe(arm: str) -> Dict[str, Any]:
    chain = get_chain(arm)
    design = IkDesign(task=default_task(chain.n))
    forward = forward_wiring(chain)
    wiring = ik_wiring(chain, design, 200, forward)
    part_of = wiring.part_of()
    parts = {}
    for name in ("goal", "turn", "offset", "levers", "direction", "share", "rest"):
        steps = [step for step in wiring.steps if part_of.get(step.node, ("", -1))[0] == name]
        parts[name] = {"products": sum(isinstance(step, Mul) for step in steps), "sums": sum(isinstance(step, Relay) for step in steps),
                       "halves": sum(isinstance(step, Half) for step in steps)}
    return {"arm": arm, "joints": chain.n, "task": design.task, "reach_m": workspace(chain).radius,
            "budget": {method: neuron_budget(chain, design, method)
                       for method in (("matrix", "lead", "observer") if design.orientation else ("matrix", "lead"))},
            "forward_neurons": forward_budget(chain), "observer": observer_budget(chain),
            "forward_wiring": forward.summary(), "ik_wiring": wiring.summary(),
            "parts": parts,
            "divisors": normalisers(design, levers(chain)).tolist(), "caps_rad": share_caps(chain, design).tolist(),
            "relaxation": relaxation(chain, design)}


def run_command(args: Any) -> None:
    results.write_raw("structure", "all", {"benchmark": "structure", "techniques": list(TECHNIQUES),
                                           "arms": [describe(arm) for arm in ARMS], "ring_bump": ring_bump()})


def summarise(name: str = "structure") -> Dict[str, Any]:
    return results.read_raw("structure")[0]
