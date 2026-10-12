"""Standard inverse kinematics for a pose goal, in the shape of the ring IK's loop.

Every method answers the same question each tick: given what each joint does to the hand per
radian (it turns it about ``a_k`` and moves its end point by ``m_k`` reaches) and the turn ``e``
and offset ``d`` that are still to make, how far should each joint move?  With the task's
weights the stacked error is ``E = (√α e, √β d)`` and the Jacobian's column k is
``J_k = (√α a_k, √β m_k)``; α is 1 or 0 (orientation asked for or not) and β the balance squared
(or 0).  The answer is a vector of shares, as :func:`ring_se3.ik.rules.pose_cascade` gives, so the
methods run in the same loop (:func:`ring_se3.ik.rules.simulate`, argument ``rule``), with the
same caps and joint limits.

* ``transpose``: ``t = J^T E``.  The gradient of the error; every joint gets the whole of it.
* ``dls``: damped least squares, ``t = J^T (J J^T + λ² I)^-1 E``: the smallest joint motion that
  makes turn and move, kept finite near poses where the joints do not span every direction.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ring_so3.ik import rules as so3

from ..ik.design import PoseIkDesign

METHODS = ("transpose", "dls")
LABELS = {"transpose": "Jacobian transpose", "dls": "damped least squares", "cascade": "exact cascade", "ring": "ring IK (spikes)"}
DAMPING = 0.1
#: loop gain of each method, 1/s (the transpose makes turn and move several times over and needs a lower one)
GAINS = {"transpose": 0.5, "dls": 1.0, "cascade": 1.0}


def stacked(axes: np.ndarray, move: np.ndarray, turn: np.ndarray, offset: np.ndarray, design: PoseIkDesign):
    """``J`` (6 x n) and ``E`` (6,) with the task's weights."""

    ra, rb = np.sqrt(design.turn_weight), np.sqrt(design.move_weight)
    jacobian = np.concatenate([ra * np.asarray(axes, dtype=float).T, rb * np.asarray(move, dtype=float).T])
    return jacobian, np.concatenate([ra * np.asarray(turn, dtype=float), rb * np.asarray(offset, dtype=float)])


def _faded(shares: np.ndarray, caps: np.ndarray, room: Optional[np.ndarray]) -> np.ndarray:
    if room is None:
        return shares
    return np.array([so3.faded(shares[k], caps[k], room[k, 0], room[k, 1], "scale") for k in range(shares.size)])


def transpose(axes, move, turn, offset, design, lever, caps, room=None) -> np.ndarray:
    jacobian, error = stacked(axes, move, turn, offset, design)
    return _faded(np.clip(jacobian.T @ error, -np.asarray(caps), np.asarray(caps)), np.asarray(caps), room)


def dls(axes, move, turn, offset, design, lever, caps, room=None, damping: float = DAMPING) -> np.ndarray:
    jacobian, error = stacked(axes, move, turn, offset, design)
    caps = np.asarray(caps, dtype=float)
    toward = jacobian.T @ error
    weights = np.ones(caps.size) if room is None else np.array([room[k, 1] if toward[k] > 0 else room[k, 0] for k in range(caps.size)])
    weighted = jacobian * weights[None, :]                            # a joint at its limit is left out of the solution
    shares = weighted.T @ np.linalg.solve(weighted @ weighted.T + damping ** 2 * np.eye(6), error)
    scale = float(np.max(np.abs(shares) / caps)) if shares.size else 0.0
    shares = shares / scale if scale > 1.0 else shares                # the whole step is scaled: the direction is kept
    return _faded(shares, caps, room)


RULES: Dict[str, Callable[..., np.ndarray]] = {"transpose": transpose, "dls": dls}


__all__ = ["DAMPING", "GAINS", "LABELS", "METHODS", "RULES", "dls", "stacked", "transpose"]
