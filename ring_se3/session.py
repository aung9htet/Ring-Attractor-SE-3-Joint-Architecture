"""A pose network and the light robot stepped together, and the measurement of one held pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from ring_so3.kinematics import Chain
from ring_so3.session import HOLD_TICKS, MAX_MOVE_TICKS, MEASURE_TICKS, Measurement, Session, pose_row as so3_pose_row

from . import kinematics


@dataclass
class PoseMeasurement:
    so3: Measurement
    #: position (m) from the sheets' spikes of the measured ticks, and the difference of the two copies of each coordinate (m)
    position: np.ndarray
    disagreement: np.ndarray
    #: position (m) held by the position cells, mean over the measured ticks
    cells_position: np.ndarray
    #: scaled place of each sheet's packet over the measured ticks, (3, 2), and its spikes in them
    places: np.ndarray
    sheet_spikes: np.ndarray


class PoseSession(Session):
    """As :class:`ring_so3.session.Session`, and the joint velocities are handed to the network as well."""

    def tick(self):
        self.robot.step(self.dt_s)
        return self.network.step(self.written(), self.robot.velocity)

    def rest(self, ticks: int) -> List[Any]:
        return [self.network.step(self.robot.q, np.zeros(self.robot.chain.n)) for _ in range(int(ticks))]

    def go_to(self, q: Sequence[float], hold_ticks: int = HOLD_TICKS, measure_ticks: int = MEASURE_TICKS,      # type: ignore[override]
              max_move_ticks: int = MAX_MOVE_TICKS) -> PoseMeasurement:
        self.robot.set_target(q)
        moved = 0
        while not self.robot.arrived() and moved < max_move_ticks:
            self.tick()
            moved += 1
        arrived = self.robot.arrived()
        ticks = [self.tick() for _ in range(int(hold_ticks))]
        return measure(self.network, self.robot.q.copy(), ticks[-int(measure_ticks):], moved, arrived)


def measure(network: Any, q: np.ndarray, measured: List[Any], moved: int = 0, arrived: bool = True) -> PoseMeasurement:
    n = len(q)
    held = np.array([np.angle(np.mean(np.exp(1j * np.array([t.held[k] for t in measured])))) for k in range(n)])
    so3 = Measurement(
        q=q, move_ticks=int(moved),
        tool=np.mean([t.tool for t in measured], axis=0), links=np.mean([t.links for t in measured], axis=0),
        angles=measured[-1].angles.copy(), ring_spikes=measured[-1].ring_spikes.copy(), held=held,
        ring_error=np.nanmax(np.abs([t.ring_error for t in measured]), axis=0), arrived=bool(arrived),
    )
    counts = np.sum([t.sheet_counts for t in measured], axis=0)
    position, disagreement = network.position_from_counts(counts)
    places = np.array([network.mapping.centre(c) for c in counts])
    return PoseMeasurement(so3, position, disagreement, np.mean([t.cells_position for t in measured], axis=0),
                           places, counts.sum(axis=1))


def pose_row(chain: Chain, network: Any, m: PoseMeasurement) -> Dict[str, Any]:
    """Compare one held pose's readout with the exact kinematics: the SO(3) row plus the position."""

    from .sheet import PLANES

    row = so3_pose_row(chain, network, m.so3)
    true = kinematics.position(chain, m.so3.q)
    radius = network.space.radius
    scaled = network.space.scaled(true)
    row.update({
        "true_position": [float(v) for v in true],
        "position": [float(v) for v in m.position],
        "position_error_m": kinematics.position_error(m.position, true),
        "cells_position": [float(v) for v in m.cells_position],
        "cells_error_m": kinematics.position_error(m.cells_position, true),
        #: exact position of the angles read back from the joint rings' activity
        "held_position_error_m": (kinematics.position_error(kinematics.position(chain, m.so3.held), true)
                                  if np.all(np.isfinite(m.so3.held)) else float("nan")),
        "disagreement_m": [float(v) for v in m.disagreement],
        #: each sheet's own two coordinates against the true ones, m
        "sheet_error_m": [float(np.linalg.norm(m.places[i] - scaled[[a, b]]) * radius) for i, (_, a, b) in enumerate(PLANES)],
        "sheet_spikes": [float(v) for v in m.sheet_spikes],
        "radius_m": float(radius),
    })
    return row


__all__ = ["PoseMeasurement", "PoseSession", "measure", "pose_row"]
