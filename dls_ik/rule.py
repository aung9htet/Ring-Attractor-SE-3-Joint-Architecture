"""Damped least squares (Wampler 1986): every tick, the smallest joint step that makes the task's turn and move.

``Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ E`` with the task's weighted Jacobian and error
(:func:`reaching_benchmark.limits.weighted_jacobian`), λ = 0.1. A joint at a limit it would move toward is left out of
the solution (its column is weighed by its room). If any joint's step is larger than its cap, the whole step is scaled
down together, so its direction is kept. It reads the measured joint angles and the true geometry every tick: it has no
internal model, no delay and nothing learned.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from reaching_benchmark.limits import fade_steps, weighted_jacobian
from reaching_benchmark.task import IkDesign

DAMPING = 0.1
#: the step is sent as a target this many times ahead of the joint, over κ (as every controller's step)
GAIN = 1.0


def dls(axes, move, turn_, offset_, design: IkDesign, lever: Any, caps, room=None, damping: float = DAMPING) -> np.ndarray:
    """The joint steps (n,), rad. ``lever`` is not used (the signature of every rule of the exact loop)."""

    jacobian, error = weighted_jacobian(axes, move, turn_, offset_, design)
    caps = np.asarray(caps, dtype=float)
    toward = jacobian.T @ error
    weights = np.ones(caps.size) if room is None else np.array([room[k, 1] if toward[k] > 0 else room[k, 0] for k in range(caps.size)])
    weighted = jacobian * weights[None, :]                    # a joint at its limit is left out of the solution
    shares = weighted.T @ np.linalg.solve(weighted @ weighted.T + damping ** 2 * np.eye(6), error)
    scale = float(np.max(np.abs(shares) / caps)) if shares.size else 0.0
    shares = shares / scale if scale > 1.0 else shares        # the whole step is scaled: the direction is kept
    return fade_steps(shares, caps, room)


def steps(axes, move, turn_, offset_, design: IkDesign, caps, room: Optional[np.ndarray] = None) -> np.ndarray:
    """:func:`dls` in the shape :class:`reaching_benchmark.closed_loop.RuleLoop` calls."""

    return dls(axes, move, turn_, offset_, design, None, caps, room)
