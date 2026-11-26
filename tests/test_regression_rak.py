"""The ring-attractor kinematics model still makes the stored trials, number for number.

Replays the first stored trial of two runs with the code as it is now and compares it with the stored row (the files
round to 5 decimals):

* ``reach_kinematic/tiago7_seed1``: pair 0 on the kinematic plant;
* ``joint_loss/tiago7_seed1_lock``: pair 0 with its joint jammed 2 s into the reach.

Each takes a few minutes in NEST (8 threads, as the stored runs), so it runs only when asked::

    docker/run.bash nest -- 'RAK_REGRESSION=1 python3 -B -m unittest tests.test_regression_rak'
"""

import importlib
import os
import unittest

import numpy as np

from reaching_benchmark import results
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.targets import TEST_SEED, disabled_joints, pose_pairs
from reaching_benchmark.task import IkDesign, default_task

MODEL = "ring_attractor_kinematics"


@unittest.skipUnless(os.environ.get("RAK_REGRESSION"), "set RAK_REGRESSION=1 to replay stored trials (minutes in NEST)")
class StoredTrialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.nest = importlib.import_module("nest")
        except ImportError:
            raise unittest.SkipTest("NEST is not installed")
        cls.factory = importlib.import_module("ring_attractor_kinematics.network.factory")
        cls.session = importlib.import_module("ring_attractor_kinematics.control.session")

    def replay(self, benchmark, run, condition=None):
        stored = results.read_run(MODEL, benchmark, run)
        self.assertTrue(stored, "no stored run %s/%s" % (benchmark, run))
        row = next(r for r in stored["rows"] if r["index"] == 0)
        chain = get_chain("tiago7")
        design = IkDesign(task=default_task(chain.n), kappa=1.0)
        self.assertEqual(design.describe(), stored["design"])
        _, network = self.factory.build_pose_ik(self.nest, "tiago7", design, method="matrix", threads=8, seed=1)
        session = self.session.PoseIkSession(network)
        session.warm_up()
        pair = pose_pairs(chain, 30, TEST_SEED)[0]
        session.place(pair.start)
        disable = None
        if condition is not None:
            joint = int(disabled_joints(chain, 30)[0])
            disable = self.session.Disable(2.0, joint, condition)
        trial = session.reach(pair.goal, pair.goal_position, disable=disable)
        new = self.session.trial_row(trial, design)
        for key in ("final_q", "error_deg", "error_reach", "q", "shares"):
            np.testing.assert_allclose(np.asarray(new[key], dtype=float), np.asarray(row[key], dtype=float), rtol=0, atol=2e-5,
                                       err_msg="%s/%s %s" % (benchmark, run, key))
        self.assertEqual(bool(new["reached"]), bool(row["reached"]))
        self.assertAlmostEqual(new["duration_s"], row["duration_s"], places=4)

    def test_kinematic_reach_seed1_pair0(self):
        self.replay("reach_kinematic", "tiago7_seed1")

    def test_joint_loss_lock_seed1_pair0(self):
        self.replay("joint_loss", "tiago7_seed1_lock", condition="lock")


if __name__ == "__main__":
    unittest.main()
