"""The transitions a learned model is trained on: ``(s, u, s')`` with ``s = (q, q̇)``, one row per 50 ms tick, each with
the episode it came from. Stored as ``results/<model>/training/transitions.npz``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Transitions:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    episodes: List[int] = field(default_factory=list)

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, episode: int) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.episodes.append(int(episode))

    def __len__(self) -> int:
        return len(self.states)

    def arrays(self):
        return (np.array(self.states, dtype=np.float32), np.array(self.actions, dtype=np.float32),
                np.array(self.next_states, dtype=np.float32), np.array(self.episodes, dtype=np.int64))

    def episode_count(self) -> int:
        return len(set(self.episodes))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        states, actions, next_states, episodes = self.arrays()
        temporary = path + ".part.npz"
        np.savez_compressed(temporary, states=states, actions=actions, next_states=next_states, episodes=episodes)
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, path: str) -> "Transitions":
        out = cls()
        if not os.path.exists(path):
            return out
        data = np.load(path)
        out.states, out.actions = list(data["states"]), list(data["actions"])
        out.next_states, out.episodes = list(data["next_states"]), [int(e) for e in data["episodes"]]
        return out
