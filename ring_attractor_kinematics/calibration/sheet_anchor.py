"""What the sheets' anchor needs (``artifacts/sheet_anchor``).

* ``gate``: decoders of a sum population that give its value ``x`` and ``x²`` (the anchor current of a sheet
  neuron is a weighted sum of both);
* ``bias_rate_hz``: the rate of one bias cell as NEST runs it (its spikes fall on the 0.1 ms grid).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ..config import SheetDesign
from ..neurons.tuning import Tunings, solve_decoders
from ..spiking.populations import create_kernel
from ..spiking.sheet import BIAS_CELLS, create_bias_cells
from .artifacts import Artifact
from .value_populations import measure_driven

FORMAT = 1


def measure_bias_rate(backend: Any, tunings: Tunings, threads: int, seed: int, seconds: float = 10.0) -> float:
    create_kernel(backend, threads, seed)
    cells = create_bias_cells(backend, tunings.model)
    recorder = backend.Create("spike_recorder")
    backend.Connect(cells, recorder)
    backend.Simulate(1000.0)
    backend.SetStatus(recorder, {"n_events": 0})
    backend.Simulate(float(seconds) * 1000.0)
    return float(backend.GetStatus(recorder, "n_events")[0]) / (BIAS_CELLS * float(seconds))


def run(backend: Any, tunings: Tunings, values: Artifact, sheet: SheetDesign = SheetDesign(), threads: int = 8, seed: int = 1,
        log: Callable[[str], None] = print) -> Artifact:
    started = time.time()
    settings = values.meta["settings"]
    x = np.linspace(-1.1, 1.1, int(settings["scalar_points"]))
    held = np.clip(x, -1.0, 1.0)
    rates = measure_driven(backend, tunings, "scalar", x[:, None], values.arrays["decoders_source"], threads, seed)
    d_gate = solve_decoders(rates, np.stack([held, held ** 2], axis=1), float(settings["regularisation"]))
    fit = {"gate_rms": float(np.sqrt(np.mean((rates @ d_gate[:, :1] - held[:, None]) ** 2))),
           "gate_square_rms": float(np.sqrt(np.mean((rates @ d_gate[:, 1:] - (held ** 2)[:, None]) ** 2))),
           "bias_rate_hz": measure_bias_rate(backend, tunings, threads, seed)}
    log("gate: value rms %.4f, square rms %.4f; bias cells %.3f Hz (%.0f s)" % (fit["gate_rms"], fit["gate_square_rms"],
                                                                              fit["bias_rate_hz"], time.time() - started))
    meta = {"sheet": sheet.describe(), "fit": fit,
            "settings": {"format": FORMAT, "seed": int(seed), "threads": int(threads), "base": "value_populations",
                         "seconds": round(time.time() - started, 1)}}
    return Artifact("sheet_anchor", {"decoders_gate": d_gate}, meta)
