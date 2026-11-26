"""The learned dynamics: ``Δs = f(s, u)`` as one deterministic network (Nagabandi et al. 2018) or an ensemble of
probabilistic networks (Chua et al. 2018, PETS), with inputs and outputs normalised by the training data.

All members are evaluated together (an ensemble linear layer: weights ``(E, in, out)``), so a batch of thousands of
predicted trajectories is one matrix product per layer on the GPU.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class EnsembleLinear(nn.Module):
    def __init__(self, members: int, inputs: int, outputs: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(members, 1, outputs))
        bound = 1.0 / math.sqrt(inputs)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (E, B, inputs)
        return torch.baddbmm(self.bias, x, self.weight)


ACTIVATIONS = {"relu": nn.ReLU, "swish": nn.SiLU}


class DynamicsModel(nn.Module):
    """``members`` networks with ``layers`` hidden layers of ``hidden`` units. ``probabilistic``: each outputs a mean and a
    log variance (held within learned bounds, as in PETS); otherwise a mean only (trained by mean squared error)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int, layers: int, activation: str, members: int = 1,
                 probabilistic: bool = False) -> None:
        super().__init__()
        self.state_dim, self.action_dim, self.members, self.probabilistic = state_dim, action_dim, members, probabilistic
        sizes = [state_dim + action_dim] + [hidden] * layers
        blocks = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            blocks += [EnsembleLinear(members, a, b), ACTIVATIONS[activation]()]
        self.body = nn.Sequential(*blocks)
        self.head = EnsembleLinear(members, hidden, state_dim * (2 if probabilistic else 1))
        if probabilistic:
            self.max_logvar = nn.Parameter(torch.full((1, 1, state_dim), 0.5))
            self.min_logvar = nn.Parameter(torch.full((1, 1, state_dim), -10.0))
        for name in ("in_mean", "in_std", "out_mean", "out_std"):
            size = state_dim + action_dim if name.startswith("in") else state_dim
            self.register_buffer(name, torch.zeros(size) if name.endswith("mean") else torch.ones(size))

    def fit_normaliser(self, inputs: torch.Tensor, deltas: torch.Tensor) -> None:
        self.in_mean.copy_(inputs.mean(0))
        self.in_std.copy_(inputs.std(0).clamp_min(1e-6))
        self.out_mean.copy_(deltas.mean(0))
        self.out_std.copy_(deltas.std(0).clamp_min(1e-6))

    def raw(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Normalised mean and log variance (``None`` if deterministic) for normalised ``inputs`` (E, B, in)."""

        out = self.head(self.body(inputs))
        if not self.probabilistic:
            return out, None
        mean, logvar = out[..., : self.state_dim], out[..., self.state_dim:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def step(self, state: torch.Tensor, action: torch.Tensor, generator: Optional[torch.Generator] = None,
             sample: bool = True) -> torch.Tensor:
        """Next state for (E, B, ·) states and actions: each member predicts its own rows; a probabilistic member samples
        its Gaussian (``sample``) or gives its mean."""

        inputs = (torch.cat([state, action], dim=-1) - self.in_mean) / self.in_std
        mean, logvar = self.raw(inputs)
        if logvar is not None and sample:
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            mean = mean + noise * torch.exp(0.5 * logvar)
        return state + mean * self.out_std + self.out_mean

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Normalised ``inputs`` and target deltas (E, B, ·): Gaussian NLL with the log-variance bounds' penalty, or MSE."""

        mean, logvar = self.raw(inputs)
        if logvar is None:
            return ((mean - targets) ** 2).mean()
        inverse = torch.exp(-logvar)
        nll = (((mean - targets) ** 2) * inverse + logvar).mean()
        return nll + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
