"""A joint stuck part-way through a reach, for any controller (``--stuck``): the plant holds it, the row records it."""

import unittest

import numpy as np

from reaching_benchmark.benchmark import STUCK_ONSET_S, test_pairs
from reaching_benchmark.closed_loop import StuckJoint, stuck_fields
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.plants.light_robot import LightRobot
from reaching_benchmark.targets import disabled_joints
from reaching_benchmark.task import IkDesign, default_task


class StuckJointTest(unittest.TestCase):
    def test_dls_with_a_stuck_joint(self):
        from dls_ik.controller import make_loop

        chain = get_chain("tiago7")
        design = IkDesign(task=default_task(chain.n), kappa=1.0, trial_s=6.0)
        pair = test_pairs(chain, 1)[0]
        joint = int(disabled_joints(chain, 1)[0])
        loop = make_loop(chain, design, LightRobot(chain))
        loop.place(pair.start)
        trial = loop.reach(pair.goal, pair.goal_position, stop_when_settled=False, disable=StuckJoint(STUCK_ONSET_S, joint))
        fields = stuck_fields(trial, joint, STUCK_ONSET_S)
        self.assertEqual(fields["disabled_joint"], joint)
        self.assertEqual(fields["condition"], "lock")
        self.assertEqual(fields["disabled_drift_rad"], 0.0)
        self.assertGreater(fields["remaining_excursion_rad"], 0.0)
        index, q_onset = trial.onset(STUCK_ONSET_S)
        np.testing.assert_allclose(trial.q[index:, joint], q_onset[joint])
        self.assertFalse(loop.robot.locked, "the lock is undone when the trial ends")


if __name__ == "__main__":
    unittest.main()
