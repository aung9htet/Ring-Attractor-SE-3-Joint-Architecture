"""The joint-loss study without NEST: the joint assignment, the jammed robot, the exact rule with a lost joint, the
reachability check, the per-trial record and the summary."""

import importlib
import importlib.util
import os
import tempfile
import types
import unittest

import numpy as np

from reaching_benchmark import results as shared_results
from ring_attractor_kinematics import results
from ring_attractor_kinematics.benchmarks import joint_loss_summary
from reaching_benchmark.targets import STUDY_SEED, TEST_SEED, PosePair, disabled_joints, pose_pairs
from ring_attractor_kinematics.benchmarks.twins import LOSS_TWINS, joint_loss_twins, reachable_after
from ring_attractor_kinematics.config import IkDesign
from reaching_benchmark.plants.light_robot import LightRobot
from ring_attractor_kinematics.control.session import Disable, Trial, joint_loss_fields
from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.geometry.rotations import lift_pitch_yaw
from ring_attractor_kinematics.rules.cascade_rule import levers, limit_room, pose_cascade, share_caps, simulate, turn


class AssignmentTest(unittest.TestCase):
    def test_every_joint_is_disabled_equally_often(self):
        chain = get_chain("tiago7")
        joints = disabled_joints(chain, 30)
        counts = np.bincount(joints, minlength=chain.n)
        self.assertEqual(counts.sum(), 30)
        self.assertTrue(np.all((counts >= 4) & (counts <= 5)), counts)

    def test_a_pair_keeps_its_joint_whatever_the_count(self):
        chain = get_chain("tiago7")
        self.assertTrue(np.array_equal(disabled_joints(chain, 2), disabled_joints(chain, 30)[:2]))

    def test_unknown_conditions_are_refused(self):
        with self.assertRaises(ValueError):
            Disable(2.0, 0, "paralysis")
        self.assertEqual(Disable(2.0, 0, "allocation_lock").site, "allocation")
        self.assertTrue(Disable(2.0, 0, "allocation_lock").locks)
        self.assertIsNone(Disable(2.0, 0, "lock").site)


class RobotLockTest(unittest.TestCase):
    def test_a_jammed_joint_ignores_its_target_but_moves_by_hand(self):
        chain = get_chain("tiago7")
        robot = LightRobot(chain, chain.limits().mean(axis=1))
        start = robot.q.copy()
        robot.lock(3)
        robot.follow(start + 0.1, 1.0, 0.05)
        self.assertEqual(robot.q[3], start[3])
        self.assertEqual(robot.velocity[3], 0.0)
        self.assertGreater(robot.q[2], start[2])
        robot.follow(start + 0.2, 0.05, 0.05, exact=True)               # placing the arm by hand
        self.assertAlmostEqual(robot.q[3], start[3] + 0.2)
        robot.unlock()
        before = robot.q.copy()
        robot.follow(before + 0.1, 1.0, 0.05)
        self.assertGreater(robot.q[3], before[3])


class FakeLink:
    """A stand-in for the ROS link: the arm goes exactly where it is sent at each tick."""

    physics_step_s = 0.001

    def __init__(self, q):
        self.positions = list(q) + [0.0, 0.0]
        self.sent = []

    def joint_state(self):
        return list(self.positions), [], 0.0

    def send(self, positions, time_s):
        self.sent.append(list(positions))
        self.positions = list(positions)

    def step(self, iterations):
        return 0.0

    def base_pose(self):
        return None


HAVE_ROS = importlib.util.find_spec("rospy") is not None and importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAVE_ROS, "ROS is not installed")
class GazeboLockTest(unittest.TestCase):
    def test_a_jammed_joint_is_held_where_it_jammed(self):
        gazebo_robot = importlib.import_module("reaching_benchmark.plants.gazebo_robot")
        chain = get_chain("tiago7")
        start = chain.limits().mean(axis=1)
        robot = gazebo_robot.GazeboRobot(chain, FakeLink(start))
        robot.lock(3)
        robot.follow(start + 0.1, 1.0, 0.05)
        self.assertEqual(robot.target[3], start[3])                    # the controller is told to hold it
        self.assertAlmostEqual(robot.target[2], start[2] + 0.1)
        self.assertEqual(robot.describe()["locked"], [3])
        robot.follow(start + 0.2, 0.05, 0.05, exact=True)             # placing by hand still moves it
        self.assertAlmostEqual(robot.q[3], start[3] + 0.2)
        robot.unlock()
        robot.follow(start + 0.3, 1.0, 0.05)
        self.assertAlmostEqual(robot.target[3], start[3] + 0.3)


class GazeboPlanTest(unittest.TestCase):
    def test_gazebo_joint_loss_runs(self):
        try:
            reach = importlib.import_module("ring_attractor_kinematics.benchmarks.reach")
        except ImportError:
            self.skipTest("NEST is not installed")
        args = types.SimpleNamespace(pairs=None, arm=None, method=None, condition=None, seed=1, onset=None, loss=0.1,
                                     loss_class="all", lesion_seed=1)
        runs = reach.plan("joint_loss_gazebo", args)
        self.assertEqual([r["condition"] for r in runs], ["lock", "motor", "allocation"])
        self.assertTrue(all(r["robot"] == "gazebo" and r["arm"] == "tiago7" and r["onset_s"] == 2.0 for r in runs))
        self.assertEqual(reach.part_name(runs[0], "pose", (0, 1)), "tiago7_seed1_lock")


class ExactRuleTest(unittest.TestCase):
    def setUp(self):
        self.chain = get_chain("tiago7")
        self.design = IkDesign(task="pose")
        self.pair = pose_pairs(self.chain, 1, STUDY_SEED)[0]

    def test_an_informed_cascade_hands_the_joint_s_share_on(self):
        chain, design, q = self.chain, self.design, self.pair.start
        axes, move = kinematics.joint_axes(chain, q), kinematics.moves(chain, q)
        error = turn(chain.rotation(q), self.pair.goal)
        offset = (self.pair.goal_position - kinematics.position(chain, q)) / 0.80574
        room = limit_room(chain, q, design)
        room[2] = 0.0
        shares, rests = pose_cascade(axes, move, error, offset, design, levers(chain), share_caps(chain, design), room)
        self.assertEqual(shares[2], 0.0)
        self.assertTrue(np.array_equal(rests[3], rests[2]))             # what joint 3 leaves is what joint 4 is handed

    def test_a_locked_joint_stays_where_it_was_at_onset(self):
        for informed in (False, True):
            trace = simulate(self.chain, self.pair.start, self.pair.goal, self.pair.goal_position, self.design, delay_s=0.3,
                             noise=0.01, rng=np.random.default_rng(0), lock=(1.0, 2), informed=informed, stop_when_settled=False)
            after = trace.t > 1.0 + 1e-9
            self.assertTrue(np.all(trace.q[after, 2] == trace.q[np.flatnonzero(after)[0] - 1, 2]))
            self.assertEqual(bool(np.all(trace.shares[after, 2] == 0.0)), informed)

    def test_twins_and_reachability(self):
        chain, design = self.chain, self.design
        start = self.pair.start
        goal_pose = start + np.array([0.15, -0.1, 0.12, 0.1, 0.0, -0.1, 0.1])
        goal_pose[4] = start[4]                                         # a goal joint 5 does not have to move for
        goal_pose = chain.clip(goal_pose)
        pair = PosePair(0, start, goal_pose, np.array(lift_pitch_yaw(chain.rotation(goal_pose))), kinematics.position(chain, goal_pose))
        self.assertTrue(reachable_after(chain, start, 4, pair, design)["reachable"])
        out = joint_loss_twins(chain, pair, design, 0.0, 4, ("cascade", "cascade_informed"))
        self.assertTrue(out["cascade"]["reached"])
        self.assertIn("final_error_deg", out["cascade_informed"])


class TrialFieldsTest(unittest.TestCase):
    def trial(self, disabled=None):
        t = 0.05 * np.arange(1, 41)
        q = np.outer(t, np.arange(1, 4, dtype=float))                   # three joints moving at 1, 2, 3 rad/s
        if disabled is not None:
            q[disabled["onset_index"]:, disabled["joint"]] = disabled["q_onset"][disabled["joint"]]
        zeros = np.zeros(t.size)
        return Trial(t, q, 10.0 - t, 0.1 - 0.001 * t, np.zeros((t.size, 6)), 10.0, 0.1, 1.0, 0.1, None, False, disabled=disabled,
                     extras={"shares": np.ones((t.size, 3)), "lead": np.zeros((t.size, 3)), "left": zeros})

    def test_an_intact_trial_records_where_the_joint_would_have_been(self):
        trial = self.trial()
        fields = joint_loss_fields(trial, 1, 1.0)
        self.assertTrue(np.allclose(fields["q_onset"], [1.0, 2.0, 3.0]))
        self.assertAlmostEqual(fields["disabled_drift_rad"], 2.0 * (2.0 - 1.0))
        self.assertAlmostEqual(fields["remaining_excursion_rad"], (1.0 + 3.0) * 1.0)
        self.assertAlmostEqual(fields["error_at_onset_deg"], 9.0)

    def test_a_disabled_trial_uses_its_own_onset(self):
        disabled = {"joint": 2, "condition": "lock", "onset_s": 1.0, "onset_index": 20, "q_onset": np.array([1.0, 2.0, 3.0]), "lesion": None}
        fields = joint_loss_fields(self.trial(disabled), 2, 1.0)
        self.assertEqual(fields["disabled_drift_rad"], 0.0)
        self.assertEqual(fields["disable_condition"], "lock")


class SummaryTest(unittest.TestCase):
    """The summary on made-up raw files: an intact run and a lock run of three pairs."""

    def row(self, pair, joint, reached, final_q):
        t = 0.05 * np.arange(1, 101, 4)
        return {"index": pair.index, "start": pair.start.tolist(), "goal_lpy": pair.goal.tolist(), "goal_position": pair.goal_position.tolist(),
                "reached": reached, "final_error_deg": 1.0 if reached else 9.0, "final_error_reach": 0.01 if reached else 0.08,
                "time_to_reach_s": 4.0 if reached else None, "remaining_excursion_rad": 1.0, "disabled_drift_rad": 0.0,
                "disabled_share_median_rad": 0.01, "disabled_joint": joint, "q_onset": pair.start.tolist(), "final_q": list(final_q),
                "t": t.tolist(), "q": np.linspace(pair.start, final_q, t.size).tolist(), "error_deg": np.linspace(40.0, 1.0, t.size).tolist(), "error_reach": np.linspace(0.3, 0.01, t.size).tolist(),
                "twins": {name: {"reached": True} for name in LOSS_TWINS}, "reachability": {"reachable": True}}

    def test_paired_summary_and_absorption(self):
        chain = get_chain("tiago7")
        pairs = pose_pairs(chain, 3, TEST_SEED)
        joints = disabled_joints(chain, 3)
        design = IkDesign(task="pose").describe()
        documents = {
            "tiago7_seed1_intact": [self.row(p, int(j), True, p.goal_pose) for p, j in zip(pairs, joints)],
            "tiago7_seed1_lock": [self.row(p, int(j), r, p.goal_pose) for p, j, r in zip(pairs, joints, (True, False, True))]}
        saved = shared_results.ROOT
        with tempfile.TemporaryDirectory() as root:
            shared_results.ROOT = root
            try:
                for part, rows in documents.items():
                    condition = part.split("_")[2]
                    results.write_raw("joint_loss", part, {"run": {"arm": "tiago7", "kappa": 1.0, "seed": 1, "condition": condition, "onset_s": 2.0},
                                                           "design": design, "neurons": 69036, "simulated_s": 10.0, "wall_s": 100.0, "rows": rows})
                summary = joint_loss_summary.summarise("joint_loss")
                written = results.write_summary("joint_loss", summary)          # what the benchmark command does next
                self.assertTrue(os.path.exists(written))
            finally:
                shared_results.ROOT = saved
        lock = summary["groups"]["lock_seed1"]
        self.assertEqual(lock["success"]["reached"], 2)
        self.assertEqual(lock["against_intact"]["reached"], [3, 2])
        self.assertEqual(lock["against_intact"]["reference"], "joint_loss intact")
        absorption = lock["against_intact"]["absorption"]
        self.assertLessEqual(absorption["absorbed"], absorption["would_miss"])
        self.assertEqual(len(lock["course_deg"]["median"]), len(joint_loss_summary.GRID))
        self.assertNotIn("against_intact", summary["groups"]["intact_seed1"])

    def test_gazebo_pairs_with_the_stored_gazebo_run(self):
        chain = get_chain("tiago7")
        pairs = pose_pairs(chain, 3, TEST_SEED)
        joints = disabled_joints(chain, 3)
        design = IkDesign(task="pose").describe()
        saved = shared_results.ROOT
        with tempfile.TemporaryDirectory() as root:
            shared_results.ROOT = root
            try:
                intact = [self.row(p, int(j), True, p.goal_pose) for p, j in zip(pairs, joints)]
                for row in intact:
                    row.pop("q_onset")                                  # the stored runs have none: read off the path
                results.write_raw("reach_gazebo", "tiago7_seed1", {
                    "run": {"arm": "tiago7", "kappa": 1.0, "seed": 1, "robot": "gazebo", "method": "ring"}, "design": design, "rows": intact})
                results.write_raw("reach_gazebo", "tiago7_seed1_dls", {
                    "run": {"arm": "tiago7", "kappa": 1.0, "seed": 1, "robot": "gazebo", "method": "dls"}, "design": design, "rows": intact})
                results.write_raw("joint_loss_gazebo", "tiago7_seed1_lock", {
                    "run": {"arm": "tiago7", "kappa": 1.0, "seed": 1, "condition": "lock", "onset_s": 2.0, "robot": "gazebo"},
                    "design": design, "neurons": 69036, "simulated_s": 10.0, "wall_s": 100.0,
                    "rows": [self.row(p, int(j), r, p.goal_pose) for p, j, r in zip(pairs, joints, (True, False, True))]})
                summary = joint_loss_summary.summarise("joint_loss_gazebo")
            finally:
                shared_results.ROOT = saved
        against = summary["groups"]["lock_seed1"]["against_intact"]
        self.assertEqual(against["reference"], "reach_gazebo")
        self.assertEqual(against["pairs"], 3)                           # the DLS file is not taken as intact


if __name__ == "__main__":
    unittest.main()
