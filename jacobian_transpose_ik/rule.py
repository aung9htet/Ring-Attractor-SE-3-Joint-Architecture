"""The Jacobian transpose: ``Δq = Jᵀ E``, the gradient of the task's weighted error, each joint clipped to its cap and
faded near its limits. It makes the turn and the move several times over per radian, so it runs at half the loop gain.
Used as an exact-loop reference beside the ring-attractor kinematics model's trials and in the simulator; it is not a
benchmark of its own.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from reaching_benchmark.limits import fade_steps, weighted_jacobian
from reaching_benchmark.task import IkDesign

#: loop gain relative to κ
GAIN = 0.5


def transpose(axes, move, turn_, offset_, design: IkDesign, lever: Any, caps, room=None) -> np.ndarray:
    jacobian, error = weighted_jacobian(axes, move, turn_, offset_, design)
    return fade_steps(np.clip(jacobian.T @ error, -np.asarray(caps), np.asarray(caps)), np.asarray(caps), room)


def steps(axes, move, turn_, offset_, design: IkDesign, caps, room: Optional[np.ndarray] = None) -> np.ndarray:
    return transpose(axes, move, turn_, offset_, design, None, caps, room)
