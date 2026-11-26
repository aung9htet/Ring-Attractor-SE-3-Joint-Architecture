"""The non-neural rules the exact loop runs beside the model, from their own packages: DLS IK (:mod:`dls_ik`) and the
Jacobian transpose (:mod:`jacobian_transpose_ik`), plus this model's cascade in exact arithmetic.

They serve as exact-loop twins of every trial (benchmarks/twins.py), in the joint map and the reachability check, and as
the simulator's reference controllers (:func:`exact_targets`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from dls_ik import rule as dls_rule
from jacobian_transpose_ik import rule as transpose_rule
from reaching_benchmark.closed_loop import rule_targets
from reaching_benchmark.geometry.chains import Chain

from .cascade_rule import levers, pose_cascade

#: ``rule(axes, move, turn, offset, design, lever, caps, room) -> steps``, as :func:`cascade_rule.simulate` calls them
RULES: Dict[str, Callable[..., np.ndarray]] = {"transpose": transpose_rule.transpose, "dls": dls_rule.dls}
#: loop gain of each method relative to κ
GAINS = {"transpose": transpose_rule.GAIN, "dls": dls_rule.GAIN, "cascade": 1.0}
#: the methods that read the exact joint angles
EXACT_METHODS = ("cascade", "dls", "transpose")


def cascade_steps(chain: Chain) -> Callable[..., np.ndarray]:
    """The cascade rule in exact arithmetic, in the shape :class:`reaching_benchmark.closed_loop.RuleLoop` calls."""

    lever = levers(chain)

    def steps(axes, move, turn_, offset_, design, caps, room=None) -> np.ndarray:
        shares, _ = pose_cascade(axes, move, turn_, offset_, design, lever, caps, room)
        return shares
    return steps


def steps_of(method: str, chain: Chain) -> Callable[..., np.ndarray]:
    if method == "cascade":
        return cascade_steps(chain)
    return {"dls": dls_rule.steps, "transpose": transpose_rule.steps}[method]


def exact_targets(method: str) -> Callable[[Any], np.ndarray]:
    """Joint targets of a method that reads the exact joint angles, for a session's ``method`` (the simulator)."""

    def targets(session: Any) -> np.ndarray:
        return rule_targets(session, steps_of(method, session.chain), GAINS[method])
    return targets
