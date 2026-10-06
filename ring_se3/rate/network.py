"""The position and velocity populations as firing rates, without spikes (NumPy only).

The SO(3) part is :class:`ring_so3.rate.network.RateNetwork`; on top of it every product and
every sum population of a :class:`~ring_se3.wiring.PoseWiring` is replaced by its rates for a
constant input, read with readouts fitted to those rates.  It gives the position the
*position cells* hold and the velocity the *velocity populations* hold.  The attractor sheets have
no copy without spikes: what limits them (a packet settling on its cells, drifting, being
moved by single spikes) is not in a rate picture, so their size was chosen from
measurements in NEST (docs/tests.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from ring_so3.design import Design
from ring_so3.rate.network import RateNetwork

from ..wiring import Mul, PoseWiring, Relay


@dataclass
class PoseRateReadout:
    position: np.ndarray          # m, base frame
    velocity: np.ndarray          # m/s (NaN without the velocity path)
    tool: np.ndarray              # raw 3 x 3 orientation readout


class PoseRateNetwork:
    def __init__(self, wiring: PoseWiring, design: Optional[Design] = None) -> None:
        self.wiring = wiring
        self.design = design or Design()
        self.so3 = RateNetwork(wiring.so3, self.design)
        self.decoders: Dict[str, np.ndarray] = self.so3.decoders

    def _read(self, kind: str, x: np.ndarray) -> float:
        return float((self.design.tuning(kind).rates(x)[0] @ self.decoders[kind])[0])

    def decode(self, q: Sequence[float], qdot: Optional[Sequence[float]] = None) -> PoseRateReadout:
        so3 = self.so3.decode(q)

        def mul(a: float, b: float, _step: Mul) -> float:
            return self._read("product", np.array([[a, b]]))

        def relay(x: float, _step: Relay) -> float:
            return self._read("scalar", np.array([[x]]))

        values = self.wiring.evaluate(q, qdot, so3_values=so3.values, mul=mul, relay=relay)
        velocity = self.wiring.velocity_m_s(values) if self.wiring.velocity_nodes else np.full(3, np.nan)
        return PoseRateReadout(self.wiring.position_m(values), velocity, so3.tool)


__all__ = ["PoseRateNetwork", "PoseRateReadout"]
