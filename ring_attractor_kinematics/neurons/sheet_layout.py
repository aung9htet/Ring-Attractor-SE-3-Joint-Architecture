"""The 2-D attractor sheet: where its neurons sit, how they are connected, how a place is read (NumPy only).

A sheet is ``size × size`` neurons over one coordinate plane of the workspace (xy, xz or yz).
Neuron ``(i, j)`` stands for the place ``(s_i, s_j)`` in scaled coordinates ``(p − centre) / reach``;
the sheet reaches a little beyond [−1, 1] so a packet at the workspace's edge still fits.

* **Recurrent weights** (an Amari field): the ring attractor's profile over the distance in the plane.
  One packet of activity forms and stays where it is.
* **Label shift**: every neuron carries one of four labels (±a, ±b, in a 2 × 2 tiling) and the centre
  of its outgoing weights is moved ``shift`` cells that way; with all four groups active the shifts cancel.
* **Anchor**: neuron at place ``c`` gets ``offset − gain/2 · |c − x|²``, largest at the anchored place ``x``.
  Written out, ``c·x − |x|²/2 − |c|²/2``: a weighted sum of ``x`` and ``x²`` per coordinate (from the gates)
  plus a fixed bias per neuron (from the bias cells), so it needs no place code on the way in.
* **Readout**: the centre of the spikes of the last ``window`` ticks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SheetDesign
from .ring_layout import MAX_DISTANCE, ring_weight

#: the three sheets and the two coordinates each one holds
PLANES: Tuple[Tuple[str, int, int], ...] = (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2))
#: direction of each label in the sheet's own two coordinates
DIRECTIONS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


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
        """The fixed part of every neuron's anchor current: ``offset − gain/2 · |c|²``."""

        return self.design.anchor_offset - 0.5 * self.anchor_gain_per_unit() * np.sum(self.places ** 2, axis=1)

    def anchor_current(self, scaled: Sequence[float]) -> np.ndarray:
        """The whole anchor current for an anchored place (what gates and bias cells deliver on average)."""

        x = np.asarray(scaled, dtype=float).reshape(2)
        return self.design.anchor_offset - 0.5 * self.anchor_gain_per_unit() * np.sum((self.places - x) ** 2, axis=1)

    def centre(self, counts: np.ndarray) -> np.ndarray:
        """Scaled place of the centre of the spikes; NaN when there are none."""

        c = np.asarray(counts, dtype=float).reshape(-1)
        total = c.sum()
        if total <= 0:
            return np.array([np.nan, np.nan])
        return self.to_scaled((c[:, None] * self.cells).sum(axis=0) / total)


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


def sheet_position(mapping: SheetMap, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled position and disagreement from spike counts of the three sheets, (3, neurons)."""

    return combine({name: mapping.centre(counts[i]) for i, (name, _, _) in enumerate(PLANES)})


def sheets_of(coordinate: int) -> List[Tuple[int, int]]:
    """The sheets that hold a coordinate (0: x, 1: y, 2: z): ``(index of the sheet, which of its two coordinates)``."""

    return [(index, plane[1:].index(int(coordinate))) for index, plane in enumerate(PLANES) if int(coordinate) in plane[1:]]


def pooled_centre(mapping: SheetMap, counts: np.ndarray, coordinate: int) -> float:
    """Centre along one coordinate of the spikes of the two sheets that hold it (``counts``: (3, neurons))."""

    total, moment = 0.0, 0.0
    for index, axis_in_plane in sheets_of(coordinate):
        c = np.asarray(counts[index], dtype=float)
        total += float(c.sum())
        moment += float(c @ mapping.places[:, axis_in_plane])
    return moment / total if total > 0 else float("nan")
