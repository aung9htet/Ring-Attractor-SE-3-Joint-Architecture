"""Forward kinematics and the task's cost in PyTorch, for a whole batch of predicted joint angles at once (the planners
score thousands of predicted trajectories per tick). The same arithmetic as :mod:`reaching_benchmark.geometry`."""

from __future__ import annotations

import numpy as np
import torch

from reaching_benchmark.geometry.chains import Chain, link_offsets, workspace
from reaching_benchmark.task import IkDesign

AXIS = {"x": 0, "y": 1, "z": 2}


class TorchChain:
    """``rotation(q)`` (B, 3, 3) and ``position(q)`` (B, 3) of the hand for joint angles ``q`` (B, n)."""

    def __init__(self, chain: Chain, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.chain = chain
        self.device, self.dtype = device, dtype
        self.fixed = [torch.tensor(np.array(j.fixed, dtype=float), device=device, dtype=dtype) for j in chain.joints]
        self.axes = [AXIS[j.axis] for j in chain.joints]
        self.offsets = [torch.tensor(o, device=device, dtype=dtype) for o in link_offsets(chain)]
        self.tool = torch.tensor(np.array(chain.tool_fixed, dtype=float), device=device, dtype=dtype)
        self.radius = float(workspace(chain).radius)

    def _rot(self, axis: int, angle: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(angle), torch.sin(angle)
        one, zero = torch.ones_like(angle), torch.zeros_like(angle)
        if axis == 0:
            rows = [one, zero, zero, zero, c, -s, zero, s, c]
        elif axis == 1:
            rows = [c, zero, s, zero, one, zero, -s, zero, c]
        else:
            rows = [c, -s, zero, s, c, zero, zero, zero, one]
        return torch.stack(rows, dim=-1).reshape(angle.shape + (3, 3))

    def pose(self, q: torch.Tensor):
        """``(R_hand, p_hand)`` for ``q`` of shape (..., n)."""

        batch = q.shape[:-1]
        rotation = torch.eye(3, device=self.device, dtype=self.dtype).expand(batch + (3, 3))
        origin = torch.zeros(batch + (3,), device=self.device, dtype=self.dtype)
        for k in range(self.chain.n):
            origin = origin + torch.einsum("...ij,j->...i", rotation, self.offsets[k])
            rotation = rotation @ self.fixed[k] @ self._rot(self.axes[k], q[..., k])
        origin = origin + torch.einsum("...ij,j->...i", rotation, self.offsets[-1])
        return rotation @ self.tool, origin


def pose_cost(chain: TorchChain, q: torch.Tensor, goal_rotation: torch.Tensor, goal_position: torch.Tensor,
              design: IkDesign) -> torch.Tensor:
    """The task's error, weighed as every controller's: ``θ² + w²·(‖p* − p‖ / reach)²`` (θ the angle between the hand's
    and the goal's orientation; only what the task asks), for ``q`` of shape (..., n)."""

    rotation, position = chain.pose(q)
    cost = torch.zeros(q.shape[:-1], device=q.device, dtype=q.dtype)
    if design.orientation:
        trace = torch.einsum("...ij,ij->...", rotation, goal_rotation)          # tr(Rᵀ R*)
        theta = torch.arccos(torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7))
        cost = cost + design.turn_weight * theta ** 2
    if design.position:
        distance = torch.linalg.norm(goal_position - position, dim=-1) / chain.radius
        cost = cost + design.move_weight * distance ** 2
    return cost
