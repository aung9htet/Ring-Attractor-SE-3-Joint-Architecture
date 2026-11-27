"""The spiking networks in NEST (skipped where NEST is not installed): sizes, and that the old method still works."""

import importlib.util
import unittest

import numpy as np

from reaching_benchmark.targets import TEST_SEED, pose_pairs
from reaching_benchmark.geometry.rotations import lift_pitch_yaw, orientation_error_deg
from ring_attractor_kinematics.network.budget import neuron_budget

HAVE_NEST = importlib.util.find_spec("nest") is not None
#: neurons and synapses of the pose IK networks as built before the restructuring (the golden master)
SIZES = {"tiago4": (35916, 31297304), "tiago7": (69036, 38472696)}


@unittest.skipUnless(HAVE_NEST, "NEST is not installed")
class NetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nest = importlib.import_module("nest")
        cls.factory = importlib.import_module("ring_attractor_kinematics.network.factory")

    def test_sizes_are_unchanged(self):
        for name, (neurons, synapses) in SIZES.items():
            _, network = self.factory.build_pose_ik(self.nest, name, threads=4)
            self.assertEqual(network.neuron_count(), neurons, name)
            self.assertEqual(self.nest.GetKernelStatus("num_connections"), synapses, name)

    def test_new_method_matches_its_budget(self):
        chain, network = self.factory.build_pose_ik(self.nest, "tiago4", threads=4, method="lead")
        self.assertEqual(network.neuron_count(), neuron_budget(chain, network.ik_design, "lead")["total"])

    def test_forward_network_reads_a_held_pose(self):
        chain, network = self.factory.build_forward(self.nest, "tiago4", threads=4, level=True)
        q = chain.limits().mean(axis=1)
        ticks = [network.step(q) for _ in range(30)][-10:]
        self.assertLess(orientation_error_deg(np.mean([t.tool for t in ticks], axis=0), chain.rotation(q)), 5.0)

    def test_shares_point_the_right_way(self):
        chain, network = self.factory.build_pose_ik(self.nest, "tiago4", threads=4)
        pair = pose_pairs(chain, 3, TEST_SEED)[2]
        q = pair.start
        for _ in range(20):
            network.step_ik(q, np.array(lift_pitch_yaw(chain.rotation(q))), chain.frames(q)[-1][0])
        ticks = [network.step_ik(q, pair.goal, pair.goal_position) for _ in range(24)][-8:]
        shares = np.mean([t.shares for t in ticks], axis=0)
        _, _, exact, _ = network.exact(q, pair.goal, pair.goal_position)
        large = np.abs(exact) > 0.05
        self.assertTrue(np.all(np.sign(shares[large]) == np.sign(exact[large])))


if __name__ == "__main__":
    unittest.main()
