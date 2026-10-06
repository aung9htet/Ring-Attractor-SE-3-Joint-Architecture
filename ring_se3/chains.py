"""The chains of this repo: those of :mod:`ring_so3.chains`, with an end point that has a position worth decoding.

``generic2`` .. ``generic7`` are taken as they are (the end effector sits one link length
beyond the last joint).  The shortened TIAGo arms ``tiago2`` .. ``tiago6`` end, in
:mod:`ring_so3`, at the origin of their last joint, which that joint does not move; here
their end point is the origin of the *next* joint of the real arm, so every joint of the
chain moves it (``tiago4`` ends at the wrist, for example).  Their orientation is unchanged.
``tiago7`` is the whole arm, ending at ``arm_tool_link``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np

from ring_so3.chains import MODEL_NAMES, get_chain as _so3_chain
from ring_so3.kinematics import Chain


@dataclass(frozen=True)
class Workspace:
    """A ball that holds every position the end point can take: positions are given to the network as
    ``(p - centre) / radius``, three numbers in [-1, 1]."""

    centre: np.ndarray
    radius: float

    def scaled(self, position: np.ndarray) -> np.ndarray:
        return (np.asarray(position, dtype=float) - self.centre) / self.radius

    def metres(self, scaled: np.ndarray) -> np.ndarray:
        return np.asarray(scaled, dtype=float) * self.radius + self.centre


def get_chain(name: str) -> Chain:
    chain = _so3_chain(name)
    if name.startswith("tiago") and chain.n < 7:
        following = _so3_chain("tiago7").joints[chain.n]
        chain = replace(chain, tool_offset=tuple(following.offset),
                        description="TIAGo arm, arm_1 .. arm_%d, ending at the origin of %s" % (chain.n, following.name))
    return chain


def link_offsets(chain: Chain) -> List[np.ndarray]:
    """``o_1 .. o_(n+1)``: the origin of joint k in link k-1, and last the end point in link n."""

    return [np.array(j.offset, dtype=float) for j in chain.joints] + [np.array(chain.tool_offset, dtype=float)]


def lever_lengths(chain: Chain) -> np.ndarray:
    """``L_k``, k = 1 .. n: the longest the lever from joint k's origin to the end point can be."""

    lengths = np.array([np.linalg.norm(o) for o in link_offsets(chain)])
    return np.array([lengths[k + 1:].sum() for k in range(chain.n)])


def workspace(chain: Chain) -> Workspace:
    """Centre: the origin of the first joint (it does not move).  Radius: the summed lengths of the links after it."""

    offsets = link_offsets(chain)
    return Workspace(offsets[0].copy(), float(lever_lengths(chain)[0]))


def registry() -> Dict[str, Chain]:
    return {name: get_chain(name) for name in MODEL_NAMES}


__all__ = ["MODEL_NAMES", "Workspace", "get_chain", "lever_lengths", "link_offsets", "registry", "workspace"]
