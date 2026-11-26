"""Every design constant of the network and of the controller, in one place.

* :class:`Sizes` — neurons per population;
* :class:`SheetDesign`, :class:`LevelDesign` — the 2-D attractor sheets and their level cells;
* :class:`GainPairSettings` — the gain pairs that write each joint's goal ring;
* :class:`NetworkOptions` — how a network is run (threads, kernel seed, tick);
* :class:`IkDesign` — the reaching task (defined in reaching_benchmark/task.py, re-exported here);
* :class:`ObserverDesign` — the observer method's orientation observer: correction, selection, fallback;
* :data:`METHODS` — where the turn's orientation comes from: the hand's late matrix (*matrix*, the old method), the
  same matrix with the joint angles written ahead by the forward network's lag (*lead*, the new method), or the
  two-chart orientation observer (*observer*); see network/pose_ik_network.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from reaching_benchmark.task import NORMALISERS, POSITION_GOALS, TASKS, IkDesign, default_task  # noqa: F401  (the shared task)

#: what the forward network is shown (see network/pose_ik_network.py)
METHODS = ("matrix", "lead", "observer")


@dataclass(frozen=True)
class Sizes:
    """Neurons per population."""

    ring: int = 200             # joint ring, goal ring (ring attractors) and written goal ring
    feature: int = 120          # cos and sin of one joint
    product: int = 200          # one product
    entry: int = 60             # one sum: a link-orientation entry, a position cell, a gate, a comparison, ...
    output_ring: int = 100      # readout rings: lift, pitch, yaw
    norm: int = 120             # cos(pitch), for the pitch readout ring


@dataclass(frozen=True)
class SheetDesign:
    """A ``size × size`` attractor sheet over one coordinate plane of the workspace."""

    size: int = 48
    #: half-width of the sheet in scaled coordinates (the workspace is [−1, 1])
    extent: float = 1.2
    #: distance units of the ring's weight profile per cell: sets the packet's width (about 6 cells across)
    cell_units: float = 1.5
    #: label shift: every neuron's outgoing weights are moved this many cells along one of four labels; the four
    #: shifts cancel.  Kept as calibrated (the stored readouts were measured on sheets with it)
    shift: float = 3.0
    #: anchor: pA per cell² of distance from the anchored place, and the current at that place
    anchor_gain: float = 8.0
    anchor_offset: float = 32.0
    #: ticks of spikes one reading of the packet's place is taken from
    window: int = 2

    def describe(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LevelDesign:
    """The *level cells* of a sheet: a set-point loop that holds its packet's strength."""

    #: the spikes per second a packet should send (the set point)
    set_rate_hz: float = 500.0
    #: the cells hold (rate / set_rate − 1) / span
    span: float = 0.8
    #: the current (pA) full cells send to every sheet neuron, against the error
    gain_pa: float = 200.0
    #: time constant (ms) of the synapses from the sheet onto the cells (slow on purpose: a fast loop swings)
    tau_ms: float = 100.0

    def describe(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GainPairSettings:
    """One joint's goal ring, written beside its joint ring by a gain pair (no attractor of its own)."""

    #: joint-ring neuron i → gain neuron i of both sides: below what makes a gain neuron fire on its own
    gate_weight: float = 137000.0
    #: a constant current into both sides: with the gate, both fire at a resting rate under the joint's bump
    rest_pa: float = 304000.0
    #: current (pA) at a full half of the share: added to the side it points to, taken from the other
    share_drive_pa: float = 228000.0
    #: ring indices by which each side writes beside the joint's bump: the lead at a full share
    shift: int = 8
    write_weight: float = 84000.0           # gain neuron → goal ring neuron
    ring_bias_pa: float = 0.0               # constant current into the goal ring's neurons

    def describe(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkOptions:
    threads: int = 8
    seed: int = 1                    # NEST kernel seed (the encoders' Poisson generators)
    tick_ms: float = 50.0
    readout_tau_ms: float = 50.0     # smoothing of the readout cells
    record: bool = False             # spike recorders on every value population (diagnostics)
    noise_pa: float = 0.0            # s.d. of a noise current into every value population, pA
    #: give the forward sheets level cells (the inverse kinematics needs them; the forward network alone runs without)
    level: bool = False
    level_design: LevelDesign = field(default_factory=LevelDesign)
    sheet: SheetDesign = field(default_factory=SheetDesign)

    def describe(self) -> Dict[str, Any]:
        return {"threads": self.threads, "seed": self.seed, "tick_ms": self.tick_ms, "readout_tau_ms": self.readout_tau_ms,
                "noise_pa": self.noise_pa, "level": self.level, "level_design": self.level_design.describe(),
                "sheet": self.sheet.describe()}


@dataclass(frozen=True)
class ObserverDesign:
    """The observer method: observer rings in two charts move at the chart rates of the commanded angular velocity
    and are corrected toward the late hand matrix; a selector picks a chart, a confidence falls back to the matrix."""

    #: the correction moves a bump at (innovation / correction_s): slow enough that the delayed comparison is stable
    correction_s: float = 0.3
    #: a chart's cue corrects it only while that chart's pitch, as the cue shows it, is within this (deg)
    cue_limit_deg: float = 60.0
    #: the sum populations hold a rate as rate / rate_full (rad/s)
    rate_full: float = 3.0
    #: a chart's fitness = its cos pitch − confidence_weight · Σ |innovation| (rad); the selector moves to the other
    #: chart when its fitness is larger by selector_margin; the selected chart's fitness is the confidence, and
    #: below fallback_confidence the turn takes the late matrix
    selector_margin: float = 0.25
    confidence_weight: float = 1.5
    fallback_confidence: float = 0.3

    def describe(self) -> Dict[str, Any]:
        return asdict(self)
