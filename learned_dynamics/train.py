"""Training a dynamics model on every transition so far: each member sees its own bootstrap resample (PETS) or the data as
it is (one member), minibatches in a seeded order, Adam with weight decay, a fixed number of epochs; 10% of the
transitions are held out to report the fit."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .data import Transitions
from .model import DynamicsModel


def fit(model: DynamicsModel, data: Transitions, epochs: int, batch: int, lr: float, weight_decay: float, seed: int,
        device: torch.device, holdout: float = 0.1) -> Dict[str, Any]:
    states, actions, next_states, _ = data.arrays()
    inputs = torch.tensor(np.concatenate([states, actions], axis=1), device=device)
    deltas = torch.tensor(next_states - states, device=device)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(inputs))
    held = order[: int(round(holdout * len(order)))]
    train = order[len(held):]
    model.fit_normaliser(inputs[train], deltas[train])
    normal_in = (inputs - model.in_mean) / model.in_std
    normal_out = (deltas - model.out_mean) / model.out_std
    members = model.members
    picks = np.stack([rng.choice(train, size=len(train), replace=True) if members > 1 else train for _ in range(members)])
    optimiser = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    history = []
    model.train()
    for epoch in range(int(epochs)):
        shuffle = np.stack([rng.permutation(picks.shape[1]) for _ in range(members)])
        total, count = 0.0, 0
        for start in range(0, picks.shape[1], int(batch)):
            index = torch.tensor(np.take_along_axis(picks, shuffle[:, start: start + int(batch)], axis=1), device=device)
            loss = model.loss(normal_in[index], normal_out[index])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * index.shape[1]
            count += index.shape[1]
        history.append(total / max(count, 1))
    model.eval()
    with torch.no_grad():
        held_index = torch.tensor(held, device=device)
        mean, _ = model.raw(normal_in[held_index].unsqueeze(0).expand(members, -1, -1))
        held_mse = float(((mean.mean(0) - normal_out[held_index]) ** 2).mean()) if len(held) else float("nan")
    return {"epochs": int(epochs), "transitions": int(len(inputs)), "train_loss_last": history[-1] if history else None,
            "train_loss": history, "holdout_mse_normalised": held_mse}
