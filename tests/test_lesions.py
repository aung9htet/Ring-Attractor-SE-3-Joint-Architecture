"""Lesions in NEST (skipped where NEST is not installed): one joint's allocation or motor neurons silenced and given back,
diffuse neuron loss, and a short reach with a joint disabled part-way."""

import importlib.util
import unittest

import numpy as np

from reaching_benchmark.targets import TEST_SEED, pose_pairs
from ring_attractor_kinematics.config import IkDesign
from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.rotations import lift_pitch_yaw

HAVE_NEST = importlib.util.find_spec("nest") is not None


@unittest.skipUnless(HAVE_NEST, "NEST is not installed")
class JointLesionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nest = importlib.import_module("nest")
        factory = importlib.import_module("ring_attractor_kinematics.network.factory")
        cls.chain, cls.network = factory.build_pose_ik(cls.nest, "tiago4", threads=4)
        cls.pair = pose_pairs(cls.chain, 3, TEST_SEED)[2]
        q = cls.pair.start
        for _ in range(20):
            cls.network.step_ik(q, np.array(lift_pitch_yaw(cls.chain.rotation(q))), kinematics.position(cls.chain, q))

    def ticks(self, count):
        """``count`` ticks with the arm held at the pair's start and the pair's goal written."""

        return [self.network.step_ik(self.pair.start, self.pair.goal, self.pair.goal_position) for _ in range(count)]

    @staticmethod
    def handed_on(ticks, k):
        """How much of the offset joint k took: what it was handed less what it passed on (reaches, mean over ticks)."""

        before = [t.rests_position[k - 1] if k > 0 else t.offset for t in ticks]
        return float(np.linalg.norm(np.mean(before, axis=0) - np.mean([t.rests_position[k] for t in ticks], axis=0)))

    def test_allocation_lesion_hands_the_share_on_and_is_undone(self):
        _, _, exact, _ = self.network.exact(self.pair.start, self.pair.goal, self.pair.goal_position)
        k = int(np.argmax(np.abs(exact[:-1])))                         # a joint with a large share and one after it
        intact = self.ticks(16)[-6:]
        share, took = np.mean([t.shares[k] for t in intact]), self.handed_on(intact, k)
        lesion = self.network.lesion_joint(k, "allocation")
        self.assertEqual(lesion.neurons, 120)
        lesioned = self.ticks(10)[-6:]
        self.assertLess(abs(np.mean([t.shares[k] for t in lesioned])), 0.3 * abs(share))
        self.assertLess(self.handed_on(lesioned, k), 0.5 * took)       # the later joints are handed what it took
        self.network.restore(lesion)
        restored = np.mean([t.shares[k] for t in self.ticks(16)[-6:]])
        self.assertEqual(np.sign(restored), np.sign(share))
        self.assertGreater(abs(restored), 0.5 * abs(share))

    def test_motor_lesion_leaves_the_target_missing_and_is_undone(self):
        k = 1
        self.assertTrue(np.isfinite(self.ticks(8)[-1].joint_goal[k]))
        lesion = self.network.lesion_joint(k, "motor")
        self.assertEqual(lesion.neurons, 600)
        self.assertEqual(lesion.describe()["populations"], ["gain_pair2_left", "gain_pair2_right", "goal_ring2"])
        self.assertTrue(np.isnan(self.ticks(4)[-1].joint_goal[k]))
        self.network.restore(lesion)
        self.assertTrue(np.isfinite(self.ticks(8)[-1].joint_goal[k]))


@unittest.skipUnless(HAVE_NEST, "NEST is not installed")
class DiffuseLossTest(unittest.TestCase):
    def test_an_exact_fraction_of_every_population_is_silenced(self):
        nest = importlib.import_module("nest")
        factory = importlib.import_module("ring_attractor_kinematics.network.factory")
        chain, network = factory.build_forward(nest, "tiago4", threads=4, level=True)
        groups = network.lesion_classes()
        expected = sum(int(round(0.1 * size)) for name in groups for _, _, size in groups[name])
        lesion = network.lesion_fraction(0.1, 1, ["all"])
        self.assertEqual(lesion["silenced"], expected)
        self.assertEqual(lesion["classes"], ["rings", "value", "sheets"])
        thresholds = np.array(nest.GetStatus(groups["rings"][0][1], "V_th"))
        self.assertEqual(int(np.sum(thresholds > 1e6)), int(round(0.1 * groups["rings"][0][2])))
        q = chain.limits().mean(axis=1)
        tick = [network.step(q) for _ in range(4)][-1]
        self.assertTrue(np.all(np.isfinite(tick.tool)))
        with self.assertRaises(ValueError):
            network.lesion_fraction(0.1, 1, ["dendrites"])


@unittest.skipUnless(HAVE_NEST, "NEST is not installed")
class DisabledReachTest(unittest.TestCase):
    def test_a_short_reach_with_a_joint_disabled_is_undone_after(self):
        nest = importlib.import_module("nest")
        factory = importlib.import_module("ring_attractor_kinematics.network.factory")
        session_module = importlib.import_module("ring_attractor_kinematics.control.session")
        design = IkDesign(task="position", trial_s=1.5)
        chain, network = factory.build_pose_ik(nest, "tiago4", design, threads=4)
        session = session_module.PoseIkSession(network)
        session.warm_up(10)
        pair = pose_pairs(chain, 3, TEST_SEED)[2]
        session.place(pair.start)
        trial = session.reach(pair.goal, pair.goal_position, disable=session_module.Disable(0.5, 1, "allocation_lock"))
        after = trial.t > 0.5 + 1e-9
        self.assertTrue(np.all(trial.q[after, 1] == trial.disabled["q_onset"][1]))
        self.assertEqual(trial.disabled["lesion"]["neurons"], 120)
        self.assertFalse(session.robot.locked)
        for _, nodes, _ in network.joint_lesion_targets(1, "allocation"):
            self.assertTrue(np.all(np.array(nest.GetStatus(nodes, "V_th")) < 1e6))
        fields = session_module.joint_loss_fields(trial, 1, 0.5)
        self.assertEqual(fields["disabled_drift_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
