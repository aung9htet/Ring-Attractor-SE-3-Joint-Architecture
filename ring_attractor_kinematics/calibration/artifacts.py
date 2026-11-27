"""The stored calibrations: decoders measured once in NEST, kept in ``artifacts/`` as ``<name>.npz`` + ``<name>.json``.

Four files, each written by one module of this package:

* ``value_populations`` (:mod:`.value_populations`): decoders of the source, sum, product, norm and cos/sin
  populations, and the joint ring's bump profile;
* ``bounded_readouts`` (:mod:`.bounded_readouts`): a sum read out as its positive part (share halves) and as
  itself held at ±1 (comparisons, rests, level cells);
* ``sheet_anchor`` (:mod:`.sheet_anchor`): the gates' decoders of ``x`` and ``x²``, and the bias cells' rate;
* ``ramp_readout`` (:mod:`.ramp`): the ramp read-out's decoder of a packet's place.

Two more serve the observer method only: ``shift_cells`` (:mod:`.shift_cells`: the shift cells' settings and
speed curve) and ``observer`` (:mod:`.observer`: the secant and drive decoders, and the cue's delay).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from ..neurons.ring_layout import RingMap

ARTIFACTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts"))
NAMES = ("value_populations", "bounded_readouts", "sheet_anchor", "ramp_readout")
#: what only the observer method needs
OBSERVER_NAMES = ("shift_cells", "observer")


@dataclass
class Artifact:
    name: str
    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str = ARTIFACTS) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, self.name)
        np.savez_compressed(path + ".npz", **self.arrays)
        with open(path + ".json", "w", encoding="utf-8") as stream:
            json.dump(self.meta, stream, indent=1, sort_keys=True)
        return path + ".npz"

    @classmethod
    def load(cls, name: str, directory: str = ARTIFACTS) -> "Artifact":
        path = os.path.join(directory, name)
        if not os.path.exists(path + ".npz"):
            raise FileNotFoundError("no calibration %s.npz in %s: run  python -m ring_attractor_kinematics calibrate %s" % (name, directory, name))
        data = np.load(path + ".npz")
        with open(path + ".json", encoding="utf-8") as stream:
            meta = json.load(stream)
        return cls(name, {key: data[key] for key in data.files}, meta)


class Calibrations:
    """All four calibrations, with the decoders under the names the network uses."""

    def __init__(self, artifacts: Dict[str, Artifact]) -> None:
        self.artifacts = artifacts
        values = artifacts["value_populations"]
        self.decoders: Dict[str, np.ndarray] = {key[len("decoders_"):]: array for key, array in values.arrays.items()
                                                if key.startswith("decoders_")}
        for name in ("bounded_readouts", "sheet_anchor", "ramp_readout", "observer"):
            if name in artifacts:
                self.decoders.update({key[len("decoders_"):]: array for key, array in artifacts[name].arrays.items()
                                      if key.startswith("decoders_")})
        self.ring_profile: np.ndarray = values.arrays["ring_profile"]
        self.settings: Dict[str, Any] = values.meta["settings"]

    @classmethod
    def load(cls, directory: str = ARTIFACTS) -> "Calibrations":
        found = {name: Artifact.load(name, directory) for name in NAMES + OBSERVER_NAMES
                 if os.path.exists(os.path.join(directory, name + ".npz"))}
        if "value_populations" not in found:
            Artifact.load("value_populations", directory)          # raises with the command that writes it
        return cls(found)

    @property
    def ring_size(self) -> int:
        return int(self.settings["ring_size"])

    @property
    def cos_sin_tau_ms(self) -> float:
        return float(self.settings["feature_tau"])

    @property
    def bias_rate_hz(self) -> float:
        return float(self.artifacts["sheet_anchor"].meta["fit"]["bias_rate_hz"])

    @property
    def ramp_total_rate_hz(self) -> float:
        return float(self.artifacts["ramp_readout"].meta["settings"]["total_rate_hz"])

    @property
    def ramp_neurons(self) -> int:
        return int(self.artifacts["ramp_readout"].meta["settings"]["neurons"])

    @property
    def ramp_tau_ms(self) -> float:
        return float(self.artifacts["ramp_readout"].meta["settings"]["tau_ms"])

    @property
    def observer_delay_ms(self) -> float:
        """How much later the cue shows the hand than the observer's cos/sin show its estimate (ms)."""

        return float(self.artifacts["observer"].meta["fit"]["delay_ms"])

    @property
    def observer_feature_scale(self) -> float:
        """Factor that makes the cos/sin sums of a free observer bump (shift cells attached) a pair of length 1."""

        return float(self.artifacts["observer"].meta["fit"]["feature_scale"])

    @property
    def shift_settings(self) -> Dict[str, Any]:
        return dict(self.artifacts["shift_cells"].meta["settings"]["shift"])

    def has(self, name: str) -> bool:
        return name in self.artifacts

    def feature_scale(self, mapping: RingMap) -> float:
        """Factor that makes the cos/sin sums of a held bump a pair of length 1 on this ring."""

        size = mapping.size
        if self.ring_profile.size != size:
            raise ValueError("calibrated for rings of %d neurons, asked for %d" % (self.ring_profile.size, size))
        offsets = (np.arange(size) + size // 2) % size - size // 2
        return 1.0 / float(abs(np.sum(self.ring_profile * np.exp(1j * offsets * mapping.radians_per_index))))

    def fits(self) -> Dict[str, Any]:
        return {name: artifact.meta.get("fit", {}) for name, artifact in self.artifacts.items()}
