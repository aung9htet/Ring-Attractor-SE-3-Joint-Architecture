"""Ramp read-outs: one coordinate of the packets on two sheets, as a number populations can add.

Every coordinate is on two sheets.  The neurons of both, each at its scaled place ``c``, send their spikes
to one pair population::

    ramp  = Σ_j r_j c_j,a / (S_0 · RAMP_SPAN)          a number that rises along the sheets
    level = (Σ_j r_j / S_0 − 1) / LEVEL_SPAN           how strong the two packets are

(``S_0``: the spikes per second of two packets as they usually are).  The ramp alone would grow with the
packets' strength as well as with their place; the read-out is fitted to both values at once and gives the
place ``Σ r c / Σ r`` whatever the strength.  This is the warm/cold comparator of the SO-2 repo in a plane:
the goal sheets' read-outs are *cold* (the set point), the forward sheets' *warm*, and the offset is their difference.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..neurons.sheet_layout import SheetMap, sheets_of
from ..neurons.tuning import Tuning
from .populations import Population, add_bias, connect_weights, create_population
from .sheet import Sheet

RAMP_SPAN = 1.5
LEVEL_SPAN = 0.7


def ramp_inputs(mapping: SheetMap, axis_in_plane: int, total_rate_hz: float) -> np.ndarray:
    """What one spike per second of every neuron of one sheet adds to the pair, (2, sheet neurons)."""

    places = mapping.places[:, int(axis_in_plane)]
    return np.stack([places / (float(total_rate_hz) * RAMP_SPAN), np.full(mapping.count, 1.0 / (float(total_rate_hz) * LEVEL_SPAN))])


def create_ramp_readout(backend: Any, tuning: Tuning, sheets: Sequence[Sheet], coordinate: int, total_rate_hz: float, name: str,
                        seed: int = 0, record: bool = False) -> Population:
    """The population that reads coordinate ``coordinate`` from the two of ``sheets`` (xy, xz, yz) that hold it."""

    population = create_population(backend, tuning, name, seed=seed, record=record)
    for index, axis_in_plane in sheets_of(coordinate):
        sheet = sheets[index]
        weights = tuning.input_weights(np.eye(2)) @ ramp_inputs(sheet.mapping, axis_in_plane, total_rate_hz) / population.charge
        connect_weights(backend, sheet.neurons, population.neurons, weights)
    add_bias(backend, population, tuning.input_weights(np.array([[0.0], [-1.0 / LEVEL_SPAN]]))[:, 0])     # the "− 1" of the level
    return population
