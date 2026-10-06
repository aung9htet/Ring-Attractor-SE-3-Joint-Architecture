"""The new parts in NEST, with few points: a sheet on its own, the anchor, the velocity cells, a 2-joint pose end to end."""

import unittest

import numpy as np

try:
    import nest
except ImportError:              # pragma: no cover
    nest = None

from ring_so3.design import Design


@unittest.skipIf(nest is None, "NEST is not installed")
class SheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ring_so3.nest import calibrate as so3_calibrate
        from ring_se3.nest import calibrate
        from ring_se3.sheet import SheetDesign, SheetMap

        cls.design = Design()
        cls.mapping = SheetMap(SheetDesign())
        cls.base = so3_calibrate.load(cls.design)
        cls.pose = calibrate.load(cls.design)

    def test_a_packet_stays_alive_and_near_where_it_was_written(self):
        from ring_se3.nest import rigs

        row = rigs.measure_hold(nest, self.mapping, [0.3, -0.4], seconds=3.0)
        self.assertEqual(row["alive"], 1.0)
        self.assertLess(row["drift_cells"], 1.5)
        self.assertGreater(row["spikes_per_tick"], 5)

    def test_anchor_through_gate_populations_puts_the_packet_within_a_cell(self):
        from ring_se3.nest import rigs

        rows = rigs.measure_anchor(nest, self.design, self.base, self.pose, self.mapping, np.array([[0.5, 0.2], [-0.6, -0.3], [0.0, 0.8]]))
        self.assertLess(max(r["error_cells"] for r in rows), 1.0)

    def test_velocity_moves_a_free_packet_the_right_way(self):
        from ring_se3.nest import rigs

        top = float(self.pose.fit["velocity_value_at_full_speed"])
        for value in ([0.5 * top, 0.0], [0.0, -0.5 * top]):
            row = rigs.measure_velocity_chain(nest, self.design, self.base, self.pose, self.mapping, value)
            velocity, wanted = np.array(row["velocity"]), np.array(row["wanted"])
            along = float(velocity @ wanted / np.linalg.norm(wanted))
            self.assertGreater(along, 0.3 * np.linalg.norm(wanted))        # crude: see docs/results.md on how proportional it is
            self.assertLess(along, 3.0 * np.linalg.norm(wanted))

    def test_stored_calibration_matches_the_design(self):
        self.assertEqual(self.pose.sheet["size"], self.mapping.design.size)
        self.assertEqual(self.pose.decoders["gate"].shape, (self.design.sizes.entry, 2))
        self.assertEqual(self.pose.decoders["velocity"].shape, (self.design.sizes.entry, 1))
        self.assertTrue(np.all(np.diff(self.pose.curve_drive) > 0))


@unittest.skipIf(nest is None, "NEST is not installed")
class PoseNetworkTests(unittest.TestCase):
    def test_two_joint_pose_end_to_end_and_the_switches(self):
        from ring_se3 import experiments, kinematics
        from ring_se3.session import pose_row

        chain, design, network, session = experiments.build(nest, "generic2", threads=4)
        self.assertEqual(len(network.sheets), 3)
        self.assertGreater(network.neuron_count(), network.so3_neuron_count() + 3 * network.mapping.count)
        session.warm_up()
        rng = np.random.default_rng(11)
        for _ in range(3):
            row = pose_row(chain, network, session.go_to(chain.random_pose(rng)))
            self.assertLess(row["position_error_m"] / row["radius_m"], 0.05)
            self.assertLess(row["cells_error_m"] / row["radius_m"], 0.05)
            self.assertLess(row["error_deg"], 5.0)
        # anchor off with the arm at rest: the packets stay alive and near; anchor on: back within a few ticks
        true = kinematics.position(chain, session.robot.q)
        network.set_anchor(False)
        self.assertFalse(network.anchor_on)
        ticks = session.rest(20)
        self.assertTrue(np.all(ticks[-1].sheet_counts.sum(axis=1) > 0))
        self.assertLess(np.linalg.norm(ticks[-1].position - true) / network.space.radius, 0.15)
        network.set_anchor(True)
        ticks = session.rest(10)
        self.assertLess(np.linalg.norm(ticks[-1].position - true) / network.space.radius, 0.06)
        # the velocity cells give the end point's velocity for joint velocities handed in at rest
        qdot = np.array([0.4, -0.3])
        ticks = [network.step(session.robot.q, qdot) for _ in range(20)]
        decoded = np.mean([network.velocity_m_s(t) for t in ticks[10:]], axis=0)
        exact = kinematics.velocity(chain, session.robot.q, qdot)
        self.assertLess(np.linalg.norm(decoded - exact), 0.35 * np.linalg.norm(exact) + 0.01)

    def test_network_without_the_velocity_path(self):
        from ring_se3 import experiments
        from ring_se3.session import pose_row

        chain, design, network, session = experiments.build(nest, "tiago2", threads=4, velocity=False)
        self.assertEqual(len(network.muls), 0)
        self.assertEqual(len(network.velocity_cells), 0)
        session.warm_up()
        row = pose_row(chain, network, session.go_to(chain.limits().mean(axis=1) + 0.3))
        self.assertLess(row["position_error_m"] / row["radius_m"], 0.06)


if __name__ == "__main__":
    unittest.main()
