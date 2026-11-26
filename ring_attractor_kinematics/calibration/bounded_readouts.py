"""The two read-outs the IK needs beyond the forward network's (``artifacts/bounded_readouts``).

Both read the sum population, driven over twice the range a population holds (a comparison can ask for more
than it holds, and then the read-out has to stay at the bound):

* ``half``: the positive part of the value, held at 1 above 1 — a joint's share is held by two of them;
* ``bounded``: the value, held at −1 and +1 outside — comparisons, rests and level cells.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import numpy as np

from ..neurons.tuning import Tunings, solve_decoders
from .artifacts import Artifact
from .value_populations import measure_driven

FORMAT = 1
#: fitted for inputs up to this size (twice what a population holds, and a little more)
WIDEST = 2.2


def run(backend: Any, tunings: Tunings, values: Artifact, threads: int = 8, seed: int = 1, points: int = 133,
        log: Callable[[str], None] = print) -> Artifact:
    started = time.time()
    # the whole range, and the small values once more: near nothing is where a share has to be right
    x = np.concatenate([np.linspace(-WIDEST, WIDEST, int(points)), np.linspace(-0.3, 0.3, int(points) // 2)])
    inputs = np.stack([np.clip(x, -1.1, 1.1), x - np.clip(x, -1.1, 1.1)], axis=1)      # a larger value is the sum of two sources
    rates = measure_driven(backend, tunings, "scalar", inputs, values.arrays["decoders_source"], threads, seed)
    regularisation = float(values.meta["settings"]["regularisation"])
    arrays: Dict[str, np.ndarray] = {}
    fit: Dict[str, Any] = {}
    small = np.abs(x) <= 0.3
    for name, target in (("half", np.clip(x, 0.0, 1.0)[:, None]), ("bounded", np.clip(x, -1.0, 1.0)[:, None])):
        decoders = solve_decoders(rates, target, regularisation)
        arrays["decoders_" + name] = decoders
        fit[name + "_rms"] = float(np.sqrt(np.mean((rates @ decoders - target) ** 2)))
        fit[name + "_small_rms"] = float(np.sqrt(np.mean((rates[small] @ decoders - target[small]) ** 2)))
        log("%s: fit rms %.4f (%.0f s)" % (name, fit[name + "_rms"], time.time() - started))
    settings = {"format": FORMAT, "seed": int(seed), "points": int(points), "widest": WIDEST, "seconds": round(time.time() - started, 1)}
    return Artifact("bounded_readouts", arrays, {"settings": settings, "fit": fit})
