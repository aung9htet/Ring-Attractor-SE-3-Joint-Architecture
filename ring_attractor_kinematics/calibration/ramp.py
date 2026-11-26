"""The ramp read-out's decoder of a packet's place (``artifacts/ramp_readout``).

Fitted on three real sheets (with level cells), anchored through real gate populations at places on a grid,
to the centre of the spikes of the two packets that hold each coordinate in the same half second; at every
place the packets are also made weaker and stronger on purpose (±25 pA into every sheet neuron).  A second
grid, between the first one's places and with another kernel seed, checks the decoder where it was not fitted.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..config import LevelDesign, SheetDesign
from ..neurons.lif import RING_NEURON
from ..neurons.sheet_layout import PLANES, SheetMap, pooled_centre
from ..neurons.tuning import RAMP_NEURONS, RAMP_TAU_MS, Tuning, Tunings, solve_decoders
from ..spiking.populations import connect_weights, create_kernel, create_population, decoded_weights, set_value
from ..spiking.ramp_readout import LEVEL_SPAN, RAMP_SPAN, create_ramp_readout
from ..spiking.sheet import build_sheet, connect_anchor, connect_anchor_bias, connect_level_cells, create_bias_cells
from .artifacts import Artifact

FORMAT = 1
REGULARISATION = 0.01
TICK_MS = 50.0
GRID = 9
CHECK_GRID = 6
STEP_UNITS = 0.06
HOLD_TICKS = 16
MEASURE_TICKS = 10
PLACE_RADIUS = 1.05
STRENGTH_OFFSETS_PA = (0.0, -25.0, 25.0)


def grid_places(points: int = GRID, radius: float = PLACE_RADIUS, shift: float = 0.0) -> np.ndarray:
    """Places on a cubic grid inside a ball, (k, 3), in an order in which the anchor never has far to go."""

    axis = np.linspace(-radius, radius, int(points)) + float(shift)
    rows = []
    for i, x in enumerate(axis):
        for j, y in enumerate(axis if i % 2 == 0 else axis[::-1]):
            column = axis if (i * len(axis) + j) % 2 == 0 else axis[::-1]
            rows += [(x, y, z) for z in column if x * x + y * y + z * z <= radius * radius]
    return np.array(rows)


def measure(backend: Any, tunings: Tunings, decoders: Dict[str, np.ndarray], bias_rate_hz: float, mapping: SheetMap, places: np.ndarray,
            tuning: Tuning, total_rate_hz: float, level: LevelDesign, threads: int, seed: int,
            offsets_pa: Sequence[float] = (0.0,)) -> Dict[str, np.ndarray]:
    """Anchored sheets rest at each place, once per offset; per place and offset: the read-outs' rates (3, neurons),
    the pooled centre of each coordinate (3,), and each sheet's spikes per second (3,)."""

    create_kernel(backend, threads, seed)
    sources = [create_population(backend, tunings.entry, "source%d" % i, seed=50 + i) for i in range(3)]
    gates = [create_population(backend, tunings.entry, "gate%d" % i, seed=60 + i) for i in range(3)]
    for source, gate in zip(sources, gates):
        connect_weights(backend, source.neurons, gate.neurons, decoded_weights(decoders["source"], np.ones((1, 1)), gate))
    bias_cells = create_bias_cells(backend, tunings.model)
    sheets = []
    for name, first, second in PLANES:
        sheet = build_sheet(backend, name, mapping)
        connect_anchor(backend, sheet, [gates[first].neurons, gates[second].neurons], decoders["gate"])
        connect_anchor_bias(backend, sheet, bias_cells, bias_rate_hz)
        connect_level_cells(backend, sheet, tunings, decoders["bounded"], level, seed=90 + len(sheets))
        sheets.append(sheet)
    readouts = [create_ramp_readout(backend, tuning, sheets, i, total_rate_hz, "ramp%d" % i, seed=70 + i, record=True) for i in range(3)]

    def tick(x: np.ndarray) -> np.ndarray:
        for source, value in zip(sources, x):
            set_value(backend, source, [float(value)])
        backend.Simulate(TICK_MS)
        return np.array([sheet.read(backend) for sheet in sheets])

    x = np.zeros(3)
    for _ in range(20):
        tick(x)
    rates, centres, totals, where = [], [], [], []
    seconds = MEASURE_TICKS * TICK_MS / 1000.0
    resting = float(RING_NEURON["I_e"])
    for target in np.asarray(places, dtype=float):
        steps = int(np.ceil(np.linalg.norm(target - x) / STEP_UNITS))
        for i in range(1, steps + 1):
            tick(x + (target - x) * i / steps)
        x = target.copy()
        for offset in offsets_pa:
            for sheet in sheets:
                backend.SetStatus(sheet.neurons, {"I_e": resting + float(offset)})
            for _ in range(HOLD_TICKS - MEASURE_TICKS):
                tick(x)
            for readout in readouts:
                readout.counts(backend)
            counts = np.array([tick(x) for _ in range(MEASURE_TICKS)]).sum(axis=0)
            rates.append([readout.counts(backend) / seconds for readout in readouts])
            centres.append([pooled_centre(mapping, counts, i) for i in range(3)])
            totals.append(counts.sum(axis=1) / seconds)
            where.append(x.copy())
        for sheet in sheets:
            backend.SetStatus(sheet.neurons, {"I_e": resting})
    return {"places": np.array(where), "rates": np.array(rates), "centres": np.array(centres), "totals": np.array(totals)}


def samples(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Rates and targets of all three coordinates in one list (they share one decoder); places without spikes left out."""

    alive = np.isfinite(data["centres"])
    return data["rates"][alive], data["centres"][alive]


def run(backend: Any, tunings: Tunings, values: Artifact, bounded: Artifact, anchor: Artifact, sheet: SheetDesign = SheetDesign(),
        level: Optional[LevelDesign] = None, threads: int = 8, seed: int = 1, log: Callable[[str], None] = print) -> Artifact:
    level = level or LevelDesign()
    started = time.time()
    mapping = SheetMap(sheet)
    tuning = tunings.ramp(RAMP_NEURONS, RAMP_TAU_MS)
    decoders = {"source": values.arrays["decoders_source"], "gate": anchor.arrays["decoders_gate"], "bounded": bounded.arrays["decoders_bounded"]}
    bias_rate = float(anchor.meta["fit"]["bias_rate_hz"])
    common = dict(level=level, threads=threads)
    first = measure(backend, tunings, decoders, bias_rate, mapping, grid_places(4), tuning, 2.0 * level.set_rate_hz, seed=seed, **common)
    total_rate = 2.0 * float(np.median(first["totals"]))
    log("two packets send %.0f spikes a second in the median (%.0f s)" % (total_rate, time.time() - started))
    data = measure(backend, tunings, decoders, bias_rate, mapping, grid_places(GRID), tuning, total_rate, seed=seed,
                   offsets_pa=STRENGTH_OFFSETS_PA, **common)
    rates, targets = samples(data)
    d_ramp = solve_decoders(rates, targets[:, None], REGULARISATION)
    fit: Dict[str, Any] = {"places": int(len(data["places"])), "samples": int(len(targets)), "total_rate_hz": total_rate,
                           "ramp_rms": float(np.sqrt(np.mean((rates @ d_ramp[:, 0] - targets) ** 2)))}
    check = measure(backend, tunings, decoders, bias_rate, mapping, grid_places(CHECK_GRID, PLACE_RADIUS - 0.07, 0.031), tuning, total_rate,
                    seed=seed + 1, offsets_pa=(0.0, -15.0), **common)
    rates, targets = samples(check)
    error = rates @ d_ramp[:, 0] - targets
    fit.update(check_places=int(len(check["places"])), check_rms=float(np.sqrt(np.mean(error ** 2))),
               check_largest=float(np.max(np.abs(error))), check_bias=float(np.mean(error)))
    log("ramp read-out: fit rms %.4f; where it was not fitted rms %.4f, largest %.4f of the reach (%.0f s)" % (
        fit["ramp_rms"], fit["check_rms"], fit["check_largest"], time.time() - started))
    settings = {"format": FORMAT, "seed": int(seed), "threads": int(threads), "neurons": RAMP_NEURONS, "tau_ms": RAMP_TAU_MS,
                "ramp_span": RAMP_SPAN, "level_span": LEVEL_SPAN, "total_rate_hz": total_rate, "grid": GRID, "check_grid": CHECK_GRID,
                "regularisation": REGULARISATION, "level": level.describe(), "pose": "sheet_anchor", "seconds": round(time.time() - started, 1)}
    return Artifact("ramp_readout", {"decoders_ramp": d_ramp}, {"settings": settings, "fit": fit})
