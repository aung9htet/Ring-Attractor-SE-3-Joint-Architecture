"""The learned baselines on the kinematic plant, small and on the CPU (they need PyTorch: the baselines image):
forward kinematics in PyTorch is the NumPy one, a model learns the plant's response from random actions, and both planners
bring the hand closer to a goal with it.

    docker/run.bash nest --image baselines -- 'python3 -B -m unittest tests.test_learned'
"""

import dataclasses
import importlib.util
import unittest

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.limits import step_caps
from reaching_benchmark.plants.light_robot import LightRobot
from reaching_benchmark.targets import TRAIN_SEED, pose_pairs
from reaching_benchmark.task import IkDesign

HAVE_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed (use the baselines image)")
class LearnedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from learned_dynamics.setup import setup
        cls.torch = torch
        cls.device, _ = setup(1, "cpu")
        cls.chain = get_chain("tiago7")

    def test_forward_kinematics_is_the_numpy_one(self):
        from learned_dynamics.kinematics import TorchChain
        chain = TorchChain(self.chain, self.device, self.torch.float64)
        rng = np.random.default_rng(1)
        q = np.array([self.chain.random_pose(rng) for _ in range(16)])
        rotation, position = chain.pose(self.torch.tensor(q, dtype=self.torch.float64))
        for k in range(len(q)):
            np.testing.assert_allclose(rotation[k].numpy(), self.chain.rotation(q[k]), atol=1e-12)
            np.testing.assert_allclose(position[k].numpy(), kinematics.position(self.chain, q[k]), atol=1e-12)

    def test_cost_is_nothing_at_the_goal(self):
        from learned_dynamics.kinematics import TorchChain, pose_cost
        chain = TorchChain(self.chain, self.device, self.torch.float64)
        q = self.chain.start_pose()
        goal_r = self.torch.tensor(self.chain.rotation(q), dtype=self.torch.float64)
        goal_p = self.torch.tensor(kinematics.position(self.chain, q), dtype=self.torch.float64)
        cost = pose_cost(chain, self.torch.tensor(q[None, :], dtype=self.torch.float64), goal_r, goal_p, IkDesign())
        self.assertLess(float(cost[0]), 1e-6)

    def collect(self):
        from learned_dynamics.controller import RandomActionLoop
        from learned_dynamics.data import Transitions
        data = Transitions()
        design = dataclasses.replace(IkDesign(), trial_s=2.0)
        loop = RandomActionLoop(self.chain, design, LightRobot(self.chain), data, seed=1)
        for number, pair in enumerate(pose_pairs(self.chain, 6, TRAIN_SEED)):
            loop.episode = number
            loop.place(pair.start)
            loop.reach(pair.goal, pair.goal_position, stop_when_settled=False)
        return data

    def model(self, members, probabilistic, data):
        from learned_dynamics.model import DynamicsModel
        from learned_dynamics.train import fit
        model = DynamicsModel(2 * self.chain.n, self.chain.n, 64, 2, "relu" if members == 1 else "swish", members, probabilistic)
        stats = fit(model, data, 40, 128, 3e-3, 0.0, 1, self.device)
        return model, stats

    def test_a_model_learns_the_plant_and_both_planners_use_it(self):
        from learned_dynamics.controller import MpcLoop
        from learned_dynamics.kinematics import TorchChain
        from learned_dynamics.planner import CrossEntropy, RandomShooting
        data = self.collect()
        self.assertGreater(len(data), 200)
        design = IkDesign()
        caps = step_caps(self.chain, design)
        pair = pose_pairs(self.chain, 40, TRAIN_SEED)[30]
        for members, probabilistic, make in ((1, False, lambda m: RandomShooting(m, TorchChain(self.chain, self.device), design, caps, 8, 1, samples=300)),
                                             (5, True, lambda m: CrossEntropy(m, TorchChain(self.chain, self.device), design, caps, 8, 1,
                                                                              population=60, elites=10, iterations=3, alpha=0.1, particles=5))):
            model, stats = self.model(members, probabilistic, data)
            self.assertLess(stats["holdout_mse_normalised"], 0.5, stats)
            loop = MpcLoop(self.chain, dataclasses.replace(design, trial_s=4.0), LightRobot(self.chain), make(model))
            loop.place(pair.start)
            trial = loop.reach(pair.goal, pair.goal_position, stop_when_settled=False)
            start = design.turn_weight * np.radians(trial.start_error_deg) ** 2 + design.move_weight * trial.start_error_reach ** 2
            end = design.turn_weight * np.radians(trial.final_error_deg) ** 2 + design.move_weight * trial.final_error_reach ** 2
            self.assertLess(end, 0.5 * start, "members %d: cost %.4f -> %.4f" % (members, start, end))


    def test_the_protocol_end_to_end_on_the_kinematic_plant(self):
        import os
        import tempfile
        from learned_dynamics import protocol
        from pets.config import CONFIG
        from reaching_benchmark import results
        small = dataclasses.replace(CONFIG, hidden=32, layers=2, epochs=3, horizon=4, random_episodes=2, random_episode_s=1.0,
                                    iterations=1, episodes_per_iteration=1,
                                    planner_settings={"population": 20, "elites": 4, "iterations": 2, "alpha": 0.1, "particles": 5})
        saved = results.ROOT
        with tempfile.TemporaryDirectory() as root:
            results.ROOT = root
            try:
                protocol.collect(small, "light", 1)
                protocol.train(small, "light", 1, 1, "cpu")
                protocol.reach(small, "light", 1, 2, "cpu", None)
                record = protocol.read_record(small)
                document = results.read_run("pets", "reach_kinematic", "tiago7_seed1")
                self.assertTrue(os.path.exists(protocol.paths(small)["model"]))
            finally:
                results.ROOT = saved
        self.assertEqual(len(record["iterations"]), 2)                     # the fit on phase 1, then one round
        self.assertEqual(len(document["rows"]), 2)
        self.assertEqual(document["controller"]["name"], "PETS")
        self.assertIn("device", document["run"])


if __name__ == "__main__":
    unittest.main()
