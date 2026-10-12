"""A method without neurons as the thing a robot follows: joint targets from the exact joint angles."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ring_so3.ik import rules as so3

from ..ik import rules
from . import ik as standard

#: the methods that read the exact joint angles (everything but the ring IK)
EXACT_METHODS = ("cascade", "dls", "transpose")


def exact_targets(method: str) -> Callable[[Any], np.ndarray]:
    """Joint targets of a method that reads the exact joint angles; the session's written goal is what it is given."""

    def targets(session: Any) -> np.ndarray:
        chain, design, q = session.chain, session.design, session.q
        turn = (rules.bounded(so3.turn(chain.rotation(q), session.written_lpy, design.comparison), 0.5)
                if design.orientation else np.zeros(3))
        away = (rules.bounded(rules.offset(chain, q, session.written_position), 1.0 / design.position_scale)
                if design.position else np.zeros(3))
        caps = so3.share_caps(chain, design, session.ring_size)
        room = so3.limit_room(chain, q, design, session.ring_size)
        axes, move, lever = so3.joint_axes(chain, q), rules.moves(chain, q), rules.levers(chain)
        if method == "cascade":
            shares, _ = rules.pose_cascade(axes, move, turn, away, design, lever, caps, room)
        else:
            shares = standard.RULES[method](axes, move, turn, away, design, lever, caps, room)
        session.method_shares = np.asarray(shares, dtype=float)
        return q + session.method_shares * standard.GAINS.get(method, design.kappa) / design.kappa
    return targets


__all__ = ["EXACT_METHODS", "exact_targets"]
