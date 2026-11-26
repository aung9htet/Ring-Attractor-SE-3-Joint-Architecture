"""The exact rule in the shape of the spiking pose IK network, for trying the simulator where NEST is not installed."""

from __future__ import annotations

from typing import Optional

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import Chain, workspace
from reaching_benchmark.geometry.rotations import lift_pitch_yaw, nearest_rotation

from ..config import IkDesign, SheetDesign, default_task
from ..network.forward import ForwardTick
from ..network.pose_ik_network import IkTick
from ..neurons.ring_layout import RingMap
from ..neurons.sheet_layout import PLANES, SheetMap, combine
from ..rules.cascade_rule import TURN_BOUND, bounded, levers, limit_room, pose_cascade, share_caps, turn

TICK_MS = 50.0


class StubRing:
    def __init__(self, mapping: RingMap) -> None:
        self.mapping = mapping
        self.counts = np.zeros(mapping.size)
        self.target_index = 0.0

    def write(self, angle: float) -> None:
        self.target_index = self.mapping.position(angle)
        size = self.mapping.size
        offsets = (np.arange(size) - self.target_index + size / 2.0) % size - size / 2.0
        self.counts = np.round(3.0 * np.exp(-0.5 * (offsets / 4.0) ** 2))


class StubOptions:
    tick_ms = TICK_MS


class StubCalibrations:
    def __init__(self, ring_size: int) -> None:
        self.ring_size = int(ring_size)


class StubNetwork:
    """Exact kinematics with a little noise, and the exact cascade, returning the network's tick records."""

    def __init__(self, chain: Chain, ik_design: Optional[IkDesign] = None, noise_deg: float = 0.5, noise_position: float = 0.006,
                 seed: int = 0, ring_size: int = 200) -> None:
        self.chain = chain
        self.ik_design = ik_design or IkDesign(task=default_task(chain.n))
        self.options = StubOptions()
        self.calibrations = StubCalibrations(ring_size)
        self.space = workspace(chain)
        self.mapping = SheetMap(SheetDesign())
        self.rng = np.random.default_rng(seed)
        self.noise = np.radians(noise_deg)
        self.noise_position = float(noise_position)
        self.t_ms = 0.0
        self.lever = levers(chain)
        self.caps = share_caps(chain, self.ik_design, ring_size)
        self.rings = [StubRing(RingMap.for_joint(joint, ring_size)) for joint in chain.joints]
        self.goal_rings = [StubRing(RingMap.circle(ring_size)) for _ in range(3)]
        self.goal_writing_rings = [StubRing(RingMap.for_joint(joint, ring_size)) for joint in chain.joints]
        self.on_sheets = self.ik_design.position and self.ik_design.position_goal == "sheets"
        self.goal_anchor_on = self.on_sheets
        self._readout = np.zeros((3, 100))
        self._goal_scaled = self.space.scaled(kinematics.position(chain, chain.start_pose()))

    def set_goal_anchor(self, on: bool) -> None:
        self.goal_anchor_on = bool(on) and self.on_sheets

    def _packets(self, scaled: np.ndarray) -> np.ndarray:
        width = 2.0 * self.mapping.cell_size
        places = [scaled[[a, b]] for _, a, b in PLANES]
        return np.array([np.round(3.0 * np.exp(-np.sum((self.mapping.places - p) ** 2, axis=1) / (2 * width ** 2))) for p in places])

    def step(self, q: np.ndarray) -> ForwardTick:
        angles = np.asarray(q, dtype=float)
        for ring, angle in zip(self.rings, angles):
            ring.write(float(angle))
        self.t_ms += TICK_MS
        links = np.array(self.chain.link_rotations(angles)) + self.rng.normal(0.0, self.noise / 2.0, (self.chain.n, 3, 3))
        tool = self.chain.rotation(angles) + self.rng.normal(0.0, self.noise / 2.0, (3, 3))
        lpy = np.array(lift_pitch_yaw(nearest_rotation(tool)))
        theta = 2.0 * np.pi * np.arange(100) / 100.0
        self._readout = np.round(4.0 * np.maximum(np.cos(theta[None, :] - lpy[:, None]) - 0.25, 0.0))
        true = self.space.scaled(kinematics.position(self.chain, angles))
        scaled_now = true + self.rng.normal(0.0, self.noise_position, 3)
        counts = self._packets(scaled_now)
        scaled, disagreement = combine({name: self.mapping.centre(counts[i]) for i, (name, _, _) in enumerate(PLANES)})
        cells = true + self.rng.normal(0.0, self.noise_position / 2.0, 3)
        places = np.array([scaled_now[[a, b]] for _, a, b in PLANES])
        return ForwardTick(self.t_ms, angles.copy(), np.zeros(self.chain.n), links, tool, lpy, self._readout.sum(axis=1), cells, places,
                           counts, scaled, disagreement, self.space.metres(scaled), self.space.metres(cells))

    def step_ik(self, q: np.ndarray, goal_lpy: np.ndarray, goal_position: np.ndarray) -> IkTick:
        design = self.ik_design
        tick = self.step(q)
        angles = np.asarray(q, dtype=float)
        goal = np.asarray(goal_lpy, dtype=float)
        if self.goal_anchor_on or not self.on_sheets:
            self._goal_scaled = self.space.scaled(goal_position)
        seen = tick.position if self.on_sheets else tick.cells_position
        turn_ = bounded(turn(nearest_rotation(tick.tool), goal), TURN_BOUND) if design.orientation else np.zeros(3)
        away = (bounded((self.space.metres(self._goal_scaled) - seen) / self.space.radius, 1.0 / design.position_scale)
                if design.position else np.zeros(3))
        room = limit_room(self.chain, angles, design, self.calibrations.ring_size)
        shares, rests = pose_cascade(kinematics.joint_axes(self.chain, angles), kinematics.moves(self.chain, angles), turn_, away, design,
                                     self.lever, self.caps, room, noise=0.003, rng=self.rng)
        for ring, angle in zip(self.goal_rings, goal):
            ring.write(float(angle))
            if not design.orientation:
                ring.counts = np.zeros_like(ring.counts)
        joint_goal, lead = angles + shares, np.zeros(self.chain.n)
        for k, ring in enumerate(self.goal_writing_rings):
            ring.write(float(joint_goal[k]))
            lead[k] = shares[k] / ring.mapping.radians_per_index
        goal_places, goal_counts, held = np.full((3, 2), np.nan), np.zeros((3, self.mapping.count)), np.full(3, np.nan)
        if self.on_sheets:
            goal_places = np.array([self._goal_scaled[[a, b]] for _, a, b in PLANES])
            goal_counts = self._packets(self._goal_scaled)
            held = self.space.metres(combine({name: self.mapping.centre(goal_counts[i]) for i, (name, _, _) in enumerate(PLANES)})[0])
        ramps = (held, seen) if self.on_sheets else (np.full(3, np.nan), np.full(3, np.nan))
        return IkTick(tick, turn_, away, shares, rests[1:, :3], rests[1:, 3:], goal.copy() if design.orientation else np.full(3, np.nan),
                      goal_places, goal_counts, held, ramps[0], ramps[1], joint_goal, lead)

    def ring_counts(self) -> np.ndarray:
        return np.array([ring.counts for ring in self.rings])

    def readout_ring_counts(self) -> np.ndarray:
        return self._readout

    def goal_counts(self) -> np.ndarray:
        return np.array([ring.counts for ring in self.goal_rings])

    def joint_goal_counts(self) -> np.ndarray:
        return np.array([ring.counts for ring in self.goal_writing_rings])

    def neuron_count(self) -> int:
        return 0

    def ik_neuron_count(self) -> int:
        return 0
