"""The geometry the architecture schematic is drawn to: one TIAGo7 start pose, a nearby goal, and the cascade's shares there.

Computed with the RAK model's own constants (divisors N_k, caps); no neural simulation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain, workspace
from reaching_benchmark.geometry.rotations import lift_pitch_yaw, rotation_angle, turn_between
from reaching_benchmark.task import IkDesign
from ring_attractor_kinematics.neurons.ring_layout import RingMap
from ring_attractor_kinematics.rules.cascade_rule import levers, normalisers, share_caps

#: neurons per joint ring
RING_N = 200

#: the snapshot: a start pose and the pose the goal is taken from
Q_START = (0.9, 0.6, -1.57, 1.0, 0.0, 0.5, 0.0)
Q_GOAL = (0.98, 0.5, -1.52, 1.08, 0.12, 0.6, 0.05)




def snapshot() -> Dict[str, Any]:
    """The cascade at the snapshot pose, with TIAGo7's constants: shares (of the cap), leads, residuals."""

    chain, design = get_chain("tiago7"), IkDesign(task="pose")
    space = workspace(chain)
    q, goal_q = np.array(Q_START), np.array(Q_GOAL)
    points = kinematics.joint_origins(chain, q)
    p, p_goal = points[-1], kinematics.position(chain, goal_q)
    r_tool, g_tool = chain.rotation(q), chain.rotation(goal_q)
    axes, moves = kinematics.joint_axes(chain, q), kinematics.moves(chain, q)
    divisors = normalisers(design, levers(chain))
    caps = share_caps(chain, design, RING_N)
    rho = np.array([m.radians_per_index for m in (RingMap.for_joint(j, RING_N) for j in chain.joints)])
    w2 = design.balance ** 2
    turn, offset = turn_between(r_tool, g_tool), (p_goal - p) / space.radius
    shares: List[float] = []
    leads: List[float] = []
    residual: List[float] = []
    for k in range(chain.n):
        residual.append(float(np.sqrt(turn @ turn + w2 * offset @ offset)))
        share = float(np.clip((axes[k] @ turn + w2 * moves[k] @ offset) / divisors[k], -caps[k], caps[k]))
        shares.append(share / caps[k])
        leads.append(share / rho[k])
        turn, offset = turn - axes[k] * share, offset - moves[k] * share
    residual.append(float(np.sqrt(turn @ turn + w2 * offset @ offset)))
    as_list = lambda a: np.asarray(a).tolist()  # noqa: E731
    return {"origins": as_list(points[:-1]), "p": as_list(p), "p_goal": as_list(p_goal),
            "rotations": [as_list(r) for r in chain.link_rotations(q)], "r_tool": as_list(r_tool), "g_tool": as_list(g_tool),
            "s": as_list(space.scaled(p)), "s_goal": as_list(space.scaled(p_goal)), "d": as_list((p_goal - p) / space.radius),
            "shares": shares, "leads": leads, "residual": residual, "turn_deg": math.degrees(rotation_angle(r_tool, g_tool)),
            "positions": [RingMap.for_joint(j, RING_N).position(a) for j, a in zip(chain.joints, q)],
            "hand_lpy": lift_pitch_yaw(r_tool), "goal_lpy": lift_pitch_yaw(g_tool)}
