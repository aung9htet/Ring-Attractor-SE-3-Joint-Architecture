"""The pose inverse-kinematics layer in NEST, with few points (the full measurements: scripts/run_ik_*.py)."""

import os
import sys
import unittest

import numpy as np

try:
    import nest
except ImportError:          # pragma: no cover
    nest = None

from ring_so3.design import Design
from ring_se3.ik import experiments
from ring_se3.ik.design import PoseIkDesign
from ring_se3.ik.pairs import DEVELOPMENT_SEED

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))


@unittest.skipIf(nest is None, "NEST is not installed")
class SheetReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ring_so3.nest import calibrate as so3_calibrate
        from ring_so3.nest import ik_calibrate as so3_ik_calibrate
        from ring_se3.nest import calibrate, ik_calibrate
        from ring_se3.sheet import SheetDesign, SheetMap

        cls.design = Design()
        cls.mapping = SheetMap(SheetDesign())
        cls.base = so3_calibrate.load(cls.design)
        cls.pose = calibrate.load(cls.design)
        cls.bounded = so3_ik_calibrate.load(cls.design).decoders["bounded"]
        cls.ramp = ik_calibrate.load(cls.design)
        cls.places = np.array([[0.5, 0.2, -0.3], [-0.6, -0.3, 0.4], [0.62, 0.62, 0.1], [0.0, 0.8, -0.5]])

    def measure(self, level, offsets=(0.0, -25.0)):
        from ring_se3.nest import ik_calibrate
        from ring_se3.nest.sheet import LevelDesign

        return ik_calibrate.measure(nest, self.design, self.base, self.pose, self.mapping, self.places, self.ramp.tuning(self.design),
                                    self.ramp.total_rate_hz, threads=4, offsets_pa=offsets,
                                    level=LevelDesign() if level else None, level_decoders=self.bounded)

    def test_level_cells_keep_every_packet_at_its_strength(self):
        from ring_se3.nest.sheet import LevelDesign

        held = self.measure(level=True)["totals"]
        self.assertGreater(held.min(), 0.7 * LevelDesign().set_rate_hz)               # also with 25 pA taken from every neuron
        self.assertLess(held.max(), 1.3 * LevelDesign().set_rate_hz)
        free = self.measure(level=False)["totals"]
        self.assertGreater(free.std(), 2.0 * held.std())                             # without them a packet's strength depends on its place

    def test_ramp_readout_gives_the_place_of_the_packets(self):
        data = self.measure(level=True)
        read = data["rates"] @ self.ramp.decoders["ramp"][:, 0]                       # (measurements, 3 coordinates)
        self.assertLess(np.sqrt(np.mean((read - data["centres"]) ** 2)), 0.012)
        self.assertLess(np.max(np.abs(read - data["centres"])), 0.04)
        self.assertLess(np.max(np.abs(data["centres"] - data["places"])), 0.04)       # and the packets are where they were anchored

    def test_stored_ramp_calibration(self):
        from ring_se3.nest import ik_calibrate

        self.assertEqual(self.ramp.decoders["ramp"].shape, (ik_calibrate.RAMP_NEURONS, 1))
        self.assertLess(self.ramp.fit["check_rms"], 0.01)
        self.assertEqual(ik_calibrate.sheets_of(0), [(0, 0), (1, 0)])                 # x is on the sheets xy and xz
        self.assertEqual(ik_calibrate.sheets_of(2), [(1, 1), (2, 1)])                 # z on xz and yz
        self.assertEqual(len(ik_calibrate.grid_places(3)), 7)


@unittest.skipIf(nest is None, "NEST is not installed")
class PoseIkNetworkTests(unittest.TestCase):
    def test_offset_and_shares_open_loop(self):
        document = experiments.open_loop_run(nest, "tiago3", 6, PoseIkDesign(task="position", goal_ring="none"), DEVELOPMENT_SEED, log=None, threads=4)
        summary = document["summary"]
        self.assertLess(summary["offset_error_median"], 0.03)
        self.assertLess(summary["ramp_hand_error_median"], 0.03)
        self.assertLess(summary["ramp_goal_error_median"], 0.03)
        self.assertGreater(summary["sheet_spikes_smallest"], 10.0)                    # no packet near going out
        if summary["share_right_sign"] is not None:
            self.assertGreaterEqual(summary["share_right_sign"], 0.8)
        self.assertLess(summary["share_rms_rad"], 0.05)

    def test_neuron_count_is_what_the_drawing_counts(self):
        from draw_ik_architecture import neurons

        chain, network = experiments.build_ik(nest, "tiago2", PoseIkDesign(task="position"), threads=2)
        forward, layer = neurons(chain, network.ik_wiring)
        self.assertEqual(network.ik_neuron_count(), layer)
        self.assertEqual(network.neuron_count(), forward + layer)
        self.assertEqual(len(network.level_cells), 3)
        self.assertEqual(len(network.goal_sheets), 3)

    def test_positions_compared_as_numbers_need_no_goal_sheet(self):
        chain, network = experiments.build_ik(nest, "tiago2", PoseIkDesign(task="position", position_goal="cells"), threads=2)
        self.assertEqual(network.goal_sheets, [])
        self.assertEqual(network.ramp_readouts, {})
        q = chain.limits().mean(axis=1)
        from ring_se3 import kinematics

        goal = kinematics.position(chain, q + 0.2)
        for _ in range(24):
            tick = network.step_ik(q, np.zeros(3), goal)
        true = (goal - kinematics.position(chain, q)) / network.space.radius
        np.testing.assert_allclose(tick.offset, np.clip(true, -0.25, 0.25), atol=0.04)
        self.assertTrue(np.all(np.isnan(tick.ramp_goal)))

    def test_three_joints_reach_a_position_goal(self):
        document = experiments.reach_run(nest, "tiago3", 100, PoseIkDesign(task="position"), DEVELOPMENT_SEED, spread=0.4, shard=(0, 50), log=None, threads=4)
        rows = document["rows"]
        self.assertEqual(len(rows), 2)
        for row in rows:
            if row["exact_reached"]:
                self.assertLess(row["final_error_reach"], 0.06)
            self.assertLess(row["final_error_reach"], row["start_error_reach"])
            self.assertLess(row["largest_lead"], 12.0)

    def test_goal_sheets_remember_the_goal_without_their_anchor(self):
        from ring_se3 import kinematics

        chain, network = experiments.build_ik(nest, "tiago2", PoseIkDesign(task="position", goal_ring="none"), threads=2)
        q = chain.limits().mean(axis=1)
        goal = kinematics.position(chain, q + 0.3)
        for _ in range(24):
            network.step_ik(q, np.zeros(3), goal)
        network.set_goal_anchor(False)
        elsewhere = kinematics.position(chain, q - 0.3)
        for _ in range(40):                                                          # 2 s with another goal handed over, and no anchor
            tick = network.step_ik(q, np.zeros(3), elsewhere)
        self.assertFalse(network.goal_anchor_on)
        self.assertLess(np.linalg.norm(tick.goal_position - goal) / network.space.radius, 0.08)      # still the old goal, give or take a cell
        network.set_goal_anchor(True)
        for _ in range(24):
            tick = network.step_ik(q, np.zeros(3), elsewhere)
        self.assertLess(np.linalg.norm(tick.goal_position - elsewhere) / network.space.radius, 0.04)


if __name__ == "__main__":
    unittest.main()
