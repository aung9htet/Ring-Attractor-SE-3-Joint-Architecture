"""The attractor sheet: where its neurons sit, how they are connected, how a position is read from it (NumPy only).

A sheet is a square of ``size x size`` neurons laid over one coordinate plane of the
workspace.  Neuron ``(i, j)`` stands for the place ``(s_i, s_j)`` in scaled coordinates
(``(p - centre) / radius``); the sheet reaches a little beyond ``[-1, 1]`` so that a packet
at the edge of the workspace still fits.

* **Recurrent weights**: the SO-2 ring's profile (:func:`ring_so3.nest.ring.ring_weight`:
  neighbours excite, a wider surround inhibits), over the distance in the plane instead of
  along the ring.  One packet of activity forms and stays where it is.
* **Direction labels**: every neuron carries one of four labels (+a, -a, +b, -b, in a 2 x 2
  tiling) and the centre of its outgoing weights is moved ``shift`` cells that way.  With all
  four groups active the shifts cancel.  Velocity cells silence, for a few milliseconds per
  spike, the group that points against the motion; the packet then moves, faster the more
  often they fire.
* **Anchor**: neuron at place ``c`` gets the current ``offset - gain/2 · |c - x|²`` (in cells),
  largest at the anchored place ``x``.  Written out it is ``c·x - |x|²/2 - |c|²/2``: a
  weighted sum of ``x`` and ``x²`` per coordinate plus a fixed bias per neuron, so it needs no
  place code on the way in.
* **Readout**: the centre of the spikes of the last ``window`` ticks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.nest.ring import MAX_DISTANCE, ring_weight

#: the three sheets and the two coordinates each one holds
PLANES: Tuple[Tuple[str, int, int], ...] = (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2))
#: direction of each label in the sheet's own two coordinates
DIRECTIONS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


@dataclass(frozen=True)
class SheetDesign:
    size: int = 48
    #: half-width of the sheet in scaled coordinates (the workspace is [-1, 1])
    extent: float = 1.2
    #: distance units of the ring's weight profile per cell: sets the packet's width (about 6 cells across at 1.5)
    cell_units: float = 1.5
    #: cells by which a neuron's outgoing weights are moved along its label
    shift: float = 3.0
    #: anchor: pA per cell² of distance from the anchored place, and the current at that place
    anchor_gain: float = 8.0
    anchor_offset: float = 32.0
    #: velocity cells per direction, their rate at the largest speed (Hz), the weight (pA) and time constant (ms)
    #: of their synapses on the sheet
    velocity_cells: int = 8
    velocity_rate_hz: float = 12.0
    velocity_weight: float = 300.0
    velocity_tau_ms: float = 3.0
    #: share of their full drive that all velocity cells get while the arm moves.  The four directions then
    #: cancel, and a velocity raises one side and lowers the other; without it a slow motion would not move
    #: the packet at all.  While no joint moves the cells get nothing and are silent
    velocity_rest: float = 0.19
    #: largest drive (share of the full drive) a velocity adds on one side; beyond it the packet's speed is erratic
    velocity_span: float = 0.3
    #: ticks of spikes one reading of the packet's place is taken from
    window: int = 2

    def describe(self) -> Dict[str, float]:
        return asdict(self)


class SheetMap:
    def __init__(self, design: Optional[SheetDesign] = None) -> None:
        self.design = design or SheetDesign()
        n = int(self.design.size)
        self.n = n
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        #: cell coordinates (i, j) of every neuron, row-major
        self.cells = np.stack([ii, jj], axis=-1).reshape(-1, 2).astype(float)
        self.labels = ((ii % 2) * 2 + (jj % 2)).reshape(-1)
        self.directions = DIRECTIONS[self.labels]

    @property
    def count(self) -> int:
        return self.n * self.n

    @property
    def cell_size(self) -> float:
        """Scaled units per cell."""

        return 2.0 * self.design.extent / self.n

    def to_cells(self, scaled: np.ndarray) -> np.ndarray:
        return (np.asarray(scaled, dtype=float) + self.design.extent) / self.cell_size - 0.5

    def to_scaled(self, cells: np.ndarray) -> np.ndarray:
        return (np.asarray(cells, dtype=float) + 0.5) * self.cell_size - self.design.extent

    @property
    def places(self) -> np.ndarray:
        """Scaled coordinates of every neuron, (count, 2)."""

        return self.to_scaled(self.cells)

    def weight_matrix(self) -> np.ndarray:
        """``(pre, post)`` recurrent weights in pA."""

        centres = self.cells + self.design.shift * self.directions
        distance = np.linalg.norm(self.cells[None, :, :] - centres[:, None, :], axis=2) * self.design.cell_units
        return ring_weight(np.minimum(distance, MAX_DISTANCE))

    def anchor_gain_per_unit(self) -> float:
        """The anchor's gain in pA per scaled unit²."""

        return self.design.anchor_gain / self.cell_size ** 2

    def anchor_bias(self) -> np.ndarray:
        """The fixed part of the anchor current of every neuron: ``offset - gain/2 · |c|²``."""

        return self.design.anchor_offset - 0.5 * self.anchor_gain_per_unit() * np.sum(self.places ** 2, axis=1)

    def anchor_current(self, scaled: Sequence[float]) -> np.ndarray:
        """The whole anchor current for an anchored place (what the populations deliver on average)."""

        x = np.asarray(scaled, dtype=float).reshape(2)
        return self.design.anchor_offset - 0.5 * self.anchor_gain_per_unit() * np.sum((self.places - x) ** 2, axis=1)

    def centre(self, counts: np.ndarray) -> np.ndarray:
        """Scaled place of the centre of the spikes; NaN when there are none."""

        c = np.asarray(counts, dtype=float).reshape(-1)
        total = c.sum()
        if total <= 0:
            return np.array([np.nan, np.nan])
        return self.to_scaled((c[:, None] * self.cells).sum(axis=0) / total)

    def group(self, label: int) -> np.ndarray:
        return np.flatnonzero(self.labels == int(label))


def against(axis_in_plane: int, sign: int) -> int:
    """Label of the group that points against a motion along the plane's coordinate ``axis_in_plane`` (0 or 1) with ``sign``."""

    return 2 * int(axis_in_plane) + (1 if sign > 0 else 0)


def combine(estimates: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled position (3,) as the mean of the two sheets that hold each coordinate, and the two copies' difference (3,)."""

    copies: List[List[float]] = [[], [], []]
    for name, a, b in PLANES:
        if name in estimates:
            copies[a].append(float(estimates[name][0]))
            copies[b].append(float(estimates[name][1]))
    position = np.array([np.nanmean(c) if len(c) and not np.all(np.isnan(c)) else np.nan for c in copies])
    difference = np.array([c[0] - c[1] if len(c) == 2 else np.nan for c in copies])
    return position, difference


__all__ = ["DIRECTIONS", "PLANES", "SheetDesign", "SheetMap", "against", "combine"]
