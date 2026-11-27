"""Model-predictive control: every tick, plan ``horizon`` joint steps with the learned model, carry out the first.

* :class:`RandomShooting` (Nagabandi et al. 2018): ``samples`` action sequences drawn uniformly within the caps, rolled
  out with the deterministic model; the sequence with the lowest summed cost wins.
* :class:`CrossEntropy` (Chua et al. 2018, PETS): CEM over action sequences (truncated Gaussian within the caps; the
  ``elites`` best of ``population`` refit the mean and variance, smoothed by ``alpha``, for ``iterations`` rounds), each
  sequence scored by ``particles`` rollouts under trajectory sampling TS∞ (each particle keeps one randomly chosen
  ensemble member for the whole horizon, and samples that member's Gaussian at every step). The mean is shifted by one
  step and reused as the next tick's start.

The cost of a step is :func:`learned_dynamics.kinematics.pose_cost` of the predicted joint angles against the written goal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from reaching_benchmark.task import IkDesign

from .kinematics import TorchChain, pose_cost
from .model import DynamicsModel


class Planner:
    def __init__(self, model: DynamicsModel, chain: TorchChain, design: IkDesign, caps: np.ndarray, horizon: int, seed: int) -> None:
        self.model, self.chain, self.design, self.horizon = model, chain, design, int(horizon)
        self.device = chain.device
        self.caps = torch.tensor(caps, device=self.device, dtype=torch.float32)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.n = int(caps.size)
        limits = torch.tensor(chain.chain.limits(), device=self.device, dtype=torch.float32)
        self.lower, self.upper = limits[:, 0], limits[:, 1]

    def reset(self) -> None:
        """A new trial: forget the warm start."""

    def plan(self, state: np.ndarray, goal_rotation: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _goal(self, goal_rotation: np.ndarray, goal_position: np.ndarray):
        return (torch.tensor(goal_rotation, device=self.device, dtype=torch.float32),
                torch.tensor(goal_position, device=self.device, dtype=torch.float32))

    def _limits(self, q: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(q, self.lower), self.upper)

    def _horizon_cost(self, angles: list, goal_r: torch.Tensor, goal_p: torch.Tensor) -> torch.Tensor:
        """The summed cost of every rollout: the predicted joint angles of all steps are scored in one batch, then added
        step by step (the same sum, in the same order, as scoring each step on its own)."""

        costs = pose_cost(self.chain, torch.stack(angles), goal_r, goal_p, self.design)
        total = torch.zeros_like(costs[0])
        for t in range(costs.shape[0]):
            total = total + costs[t]
        return total


class RandomShooting(Planner):
    def __init__(self, *args, samples: int = 1000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.samples = int(samples)

    @torch.no_grad()
    def plan(self, state: np.ndarray, goal_rotation: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
        goal_r, goal_p = self._goal(goal_rotation, goal_position)
        actions = (torch.rand((self.samples, self.horizon, self.n), device=self.device, generator=self.generator) * 2 - 1) * self.caps
        s = torch.tensor(state, device=self.device, dtype=torch.float32).expand(1, self.samples, -1).contiguous()
        angles = []
        for t in range(self.horizon):
            s = self.model.step(s, actions[None, :, t], sample=False)
            s = torch.cat([self._limits(s[..., : self.n]), s[..., self.n:]], dim=-1)
            angles.append(s[0, :, : self.n])
        cost = self._horizon_cost(angles, goal_r, goal_p)
        return actions[int(torch.argmin(cost)), 0].cpu().numpy().astype(float)


class CrossEntropy(Planner):
    def __init__(self, *args, population: int = 400, elites: int = 40, iterations: int = 5, alpha: float = 0.1,
                 particles: int = 20, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.population, self.elites, self.iterations = int(population), int(elites), int(iterations)
        self.alpha, self.particles = float(alpha), int(particles)
        self.mean: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.mean = None

    @torch.no_grad()
    def _score(self, state: torch.Tensor, actions: torch.Tensor, goal_r: torch.Tensor, goal_p: torch.Tensor) -> torch.Tensor:
        """Mean summed cost of every sequence over its particles. TS∞ as in PETS's code: the population × particles rows
        are split among the members by a random permutation (an equal number each), and each row keeps its member for the
        whole horizon."""

        members = self.model.members
        rows = self.population * self.particles
        if rows % members:
            raise ValueError("population × particles must divide among the %d members" % members)
        index = torch.randperm(rows, device=self.device, generator=self.generator).reshape(members, rows // members)
        sequence = index // self.particles                               # which action sequence each row follows
        s = state.expand(members, rows // members, -1).contiguous()
        angles = []
        for t in range(self.horizon):
            s = self.model.step(s, actions[sequence, t], generator=self.generator, sample=True)
            s = torch.cat([self._limits(s[..., : self.n]), s[..., self.n:]], dim=-1)
            angles.append(s[..., : self.n])
        cost = self._horizon_cost(angles, goal_r, goal_p)
        total = torch.zeros(rows, device=self.device)
        total[index.reshape(-1)] = cost.reshape(-1)
        return total.reshape(self.population, self.particles).mean(1)

    @torch.no_grad()
    def plan(self, state: np.ndarray, goal_rotation: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
        goal_r, goal_p = self._goal(goal_rotation, goal_position)
        s = torch.tensor(state, device=self.device, dtype=torch.float32)
        bound = self.caps.expand(self.horizon, -1)
        mean = torch.zeros((self.horizon, self.n), device=self.device) if self.mean is None else self.mean
        var = (2 * bound) ** 2 / 16.0
        for _ in range(self.iterations):
            room = torch.minimum(bound - mean, bound + mean)                 # PETS: the spread never reaches past the caps
            std = torch.sqrt(torch.minimum((room / 2.0) ** 2, var))
            noise = torch.fmod(torch.randn((self.population, self.horizon, self.n), device=self.device, generator=self.generator), 2.0)
            actions = (mean + noise * std).clamp(-bound, bound)
            costs = self._score(s, actions, goal_r, goal_p)
            elite = actions[torch.topk(-costs, self.elites).indices]
            mean = self.alpha * mean + (1 - self.alpha) * elite.mean(0)
            var = self.alpha * var + (1 - self.alpha) * elite.var(0, unbiased=False)
        self.mean = torch.cat([mean[1:], torch.zeros_like(mean[:1])], dim=0)
        return mean[0].cpu().numpy().astype(float)
