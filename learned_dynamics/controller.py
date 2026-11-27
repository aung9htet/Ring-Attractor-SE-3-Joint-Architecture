"""The learned baselines as controllers of the shared closed loop, and the random-action controller of phase 1.

Both send a joint step ``u`` (within the caps) as the target ``q + u / κ``, exactly as DLS IK's step, and can record every
transition ``(s, u, s')`` for training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from reaching_benchmark.closed_loop import ClosedLoop
from reaching_benchmark.geometry.chains import Chain
from reaching_benchmark.limits import goal_matrix, step_caps
from reaching_benchmark.task import IkDesign

from .data import Transitions


class LearningLoop(ClosedLoop):
    """A controller whose state is the measured ``(q, q̇)``; records transitions to ``record`` when one is given."""

    def __init__(self, chain: Chain, design: IkDesign, robot: Any, record: Optional[Transitions] = None) -> None:
        super().__init__(chain, design, robot)
        self.caps = step_caps(chain, design, self.ring_size)
        self.record = record
        self.episode = 0
        self._pending: Optional[tuple] = None
        self.method_shares: Optional[np.ndarray] = None

    def state(self) -> np.ndarray:
        return np.concatenate([self.q, getattr(self.robot, "measured_velocity", np.zeros(self.chain.n))])

    def action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def targets(self) -> np.ndarray:
        state = self.state()
        u = np.clip(np.asarray(self.action(state), dtype=float), -self.caps, self.caps)
        self._pending = (state, u)
        self.method_shares = u
        return self.q + u / self.design.kappa

    def advance(self, ahead: bool = True) -> Any:
        if self._pending is not None and self.record is not None:
            self.record.add(self._pending[0], self._pending[1], self.state(), self.episode)
        self._pending = None
        return None

    def record_tick(self, rows: Dict[str, List[Any]], tick: Any, rotation: np.ndarray, position: np.ndarray) -> None:
        rows.setdefault("shares", []).append(np.array(self.method_shares, dtype=float).copy())


class RandomActionLoop(LearningLoop):
    """Phase 1: every joint's step drawn uniformly within its cap, every tick."""

    def __init__(self, chain: Chain, design: IkDesign, robot: Any, record: Transitions, seed: int) -> None:
        super().__init__(chain, design, robot, record)
        self.rng = np.random.default_rng(int(seed))

    def action(self, state: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-self.caps, self.caps)


class MpcLoop(LearningLoop):
    """The learned model's MPC: every tick, the planner's first step toward the written goal."""

    def __init__(self, chain: Chain, design: IkDesign, robot: Any, planner: Any, record: Optional[Transitions] = None) -> None:
        super().__init__(chain, design, robot, record)
        self.planner = planner

    def action(self, state: np.ndarray) -> np.ndarray:
        return self.planner.plan(state, goal_matrix(self.written_lpy), self.written_position)

    def reach(self, *args: Any, **kwargs: Any):
        self.planner.reset()
        return super().reach(*args, **kwargs)
