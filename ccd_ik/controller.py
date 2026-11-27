"""CCD IK as a controller of the shared closed loop: every tick, CCD from the measured joint angles to the written goal,
and a step toward its solution with the same caps and limit fading as every controller (the whole step scaled down
together when one joint's is over its cap, as DLS IK's)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from reaching_benchmark.closed_loop import ClosedLoop
from reaching_benchmark.geometry.chains import Chain
from reaching_benchmark.geometry.rotations import wrap
from reaching_benchmark.limits import fade_steps, goal_matrix, limit_room, step_caps
from reaching_benchmark.task import IkDesign

from . import rule

#: what the controller reads, is given and learns (stored in every run file)
DESCRIPTION: Dict[str, Any] = {
    "name": "CCD IK",
    "reads": "the measured joint angles, every 50 ms tick",
    "given": "the true geometry: forward kinematics, joint axes and origins",
    "learns": "nothing",
    "internal_model": "none: CCD is solved afresh from the measured state every tick (reactive)",
    "rule": "cyclic coordinate descent, wrist to base, closed-form combined position-orientation angle per joint "
            "(Wang & Chen 1991), joint limits clamped, up to %d sweeps per tick" % rule.SWEEPS,
    "gain": rule.GAIN,
}


class CcdLoop(ClosedLoop):
    def __init__(self, chain: Chain, design: IkDesign, robot: Any) -> None:
        super().__init__(chain, design, robot)
        self.circular = chain.circular_mask()
        self.method_shares: np.ndarray = np.zeros(chain.n)
        self.sweeps = 0

    def targets(self) -> np.ndarray:
        q = self.q
        solution, self.sweeps = rule.solve(self.chain, q, goal_matrix(self.written_lpy), self.written_position, self.design,
                                           radius=self.radius)
        step = solution - q
        step = np.where(self.circular, wrap(step), step)
        caps = step_caps(self.chain, self.design, self.ring_size)
        scale = float(np.max(np.abs(step) / caps)) if step.size else 0.0
        step = step / scale if scale > 1.0 else step                 # the whole step is scaled: the direction is kept
        self.method_shares = fade_steps(step, caps, limit_room(self.chain, q, self.design, self.ring_size))
        return q + self.method_shares * rule.GAIN / self.design.kappa

    def record_tick(self, rows: Dict[str, List[Any]], tick: Any, rotation: np.ndarray, position: np.ndarray) -> None:
        rows.setdefault("shares", []).append(self.method_shares.copy())
        rows.setdefault("sweeps", []).append(self.sweeps)


def make_loop(chain: Chain, design: IkDesign, robot: Any) -> CcdLoop:
    return CcdLoop(chain, design, robot)
