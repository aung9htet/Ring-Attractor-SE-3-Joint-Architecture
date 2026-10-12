"""The pose IK simulator's loop and server with the stand-in network (no NEST)."""

import json
import unittest
import urllib.request

import numpy as np

from ring_se3 import kinematics
from ring_se3.chains import get_chain, workspace
from ring_se3.ik.design import PoseIkDesign
from ring_se3.sim.ik_server import METHODS, PoseIkSimServer, PoseIkSimState, PoseIkSimulator, PoseIkStubNetwork
from ring_so3.kinematics import lift_pitch_yaw
from ring_so3.robots import GazeboRobot

try:                                    # "discover -s tests" imports the test files as top-level modules
    from test_robots import FakeLink
except ImportError:
    from tests.test_robots import FakeLink


def make(chain, design):
    return PoseIkStubNetwork(chain, design)


def run(simulator, ticks):
    for _ in range(ticks):
        snap = simulator.tick()
    return snap


class LoopTests(unittest.TestCase):
    def test_the_task_fits_the_chain(self):
        self.assertEqual(PoseIkSimulator(make, "tiago4", realtime=False).state.config["task"], "position")
        config = PoseIkSimulator(make, "generic7", realtime=False).state.config
        self.assertEqual(config["task"], "pose")
        self.assertTrue(config["on_sheets"])
        self.assertEqual(config["planes"], ["xy", "xz", "yz"])
        self.assertEqual([m["name"] for m in config["methods"]], list(METHODS))

    def test_a_position_goal_is_reached_and_the_run_is_kept(self):
        simulator = PoseIkSimulator(make, "tiago4", realtime=False)
        run(simulator, 5)
        chain = simulator.chain
        goal_pose = simulator.session.q + 0.3
        goal = kinematics.position(chain, goal_pose)
        simulator.state.commands.put({"command": "goal", "position": goal.tolist(), "commit": True})
        snap = run(simulator, 400)
        self.assertEqual(snap["mode"], "ik")
        self.assertLess(snap["ik"]["goal_error_mm"], 1000 * 0.04 * workspace(chain).radius)
        self.assertEqual(len(snap["runs"]), 1)
        self.assertIsNotNone(snap["runs"][0]["reached_s"])
        self.assertEqual(len(snap["sheets"]), 3)
        self.assertGreater(len(snap["goal_sheets"][0]), 3)                    # the goal is a packet on the goal sheets
        self.assertEqual(len(snap["ik"]["shares"]), 4)
        np.testing.assert_allclose(snap["ik"]["goal_position"], goal, atol=1e-3)

    def test_a_pose_goal_and_the_same_goal_with_another_method(self):
        simulator = PoseIkSimulator(make, "generic7", realtime=False)
        run(simulator, 5)
        chain = simulator.chain
        goal_pose = simulator.session.q + 0.25
        matrix, position = chain.rotation(goal_pose), kinematics.position(chain, goal_pose)
        simulator.state.commands.put({"command": "goal", "matrix": matrix.reshape(9).tolist(), "position": position.tolist()})
        snap = run(simulator, 400)
        self.assertLess(snap["ik"]["goal_error"], 5.0)
        self.assertLess(snap["ik"]["goal_error_mm"], 1000 * 0.04 * workspace(chain).radius)
        np.testing.assert_allclose(snap["ik"]["goal"], lift_pitch_yaw(matrix), atol=1e-3)
        simulator.state.commands.put({"command": "replay"})
        snap = run(simulator, 600)
        self.assertEqual(snap["method"], "cascade")
        self.assertEqual([r["method"] for r in snap["runs"]], ["ring", "cascade"])
        self.assertEqual(snap["runs"][0]["start_mm"], snap["runs"][1]["start_mm"])      # the arm was put back to the same start
        self.assertLess(snap["ik"]["goal_error_mm"], 1000 * 0.04 * workspace(chain).radius)

    def test_a_goal_outside_the_workspace_is_taken_to_its_edge(self):
        simulator = PoseIkSimulator(make, "tiago4", realtime=False)
        run(simulator, 2)
        simulator.state.commands.put({"command": "goal", "position": [5.0, 0.0, 0.0]})
        snap = run(simulator, 2)
        space = workspace(simulator.chain)
        self.assertAlmostEqual(np.linalg.norm(space.scaled(snap["ik"]["goal_position"])), 1.0, places=3)

    def test_task_and_comparison_can_be_changed(self):
        simulator = PoseIkSimulator(make, "generic7", realtime=False)
        run(simulator, 3)
        pose = simulator.session.q.copy()
        simulator.state.commands.put({"command": "task", "task": "orientation"})
        snap = run(simulator, 2)
        self.assertEqual(snap["ik"]["task"], "orientation")
        self.assertEqual(simulator.state.config["task"], "orientation")
        np.testing.assert_allclose(snap["q"], pose, atol=1e-3)                          # the arm stays where it was
        simulator.state.commands.put({"command": "task", "task": "pose"})
        simulator.state.commands.put({"command": "position_goal", "on": "cells"})
        snap = run(simulator, 2)
        self.assertFalse(simulator.state.config["on_sheets"])
        self.assertEqual(snap["goal_sheets"], [[], [], []])
        simulator.state.commands.put({"command": "chain", "chain": "tiago4"})
        snap = run(simulator, 2)
        self.assertEqual((snap["chain"], snap["ik"]["task"]), ("tiago4", "position"))

    def test_sliders_take_the_arm_over_and_the_goal_stays_on_the_hand(self):
        simulator = PoseIkSimulator(make, "tiago4", realtime=False)
        target = get_chain("tiago4").limits().mean(axis=1) + 0.2
        simulator.state.commands.put({"command": "target", "q": target.tolist()})
        snap = run(simulator, 80)
        self.assertEqual(snap["mode"], "manual")
        np.testing.assert_allclose(snap["q"], target, atol=1e-3)
        self.assertLess(snap["ik"]["goal_error_mm"], 1.0)

    def test_gazebo_is_offered_only_when_it_is_there(self):
        simulator = PoseIkSimulator(make, "tiago7", realtime=False)
        config = simulator.config()
        self.assertEqual(config["robots_there"], ["light"])
        self.assertEqual([r["name"] for r in config["robots"]], ["light", "gazebo"])
        simulator.handle({"command": "robot", "robot": "gazebo"})           # ignored
        snap = simulator.tick()
        self.assertEqual(snap["robot"], "light")
        self.assertIsNone(snap["gazebo"])

    def test_with_gazebo_the_arm_is_gazebos_and_a_change_of_robot_keeps_the_pose(self):
        chain = get_chain("tiago7")
        link = FakeLink(chain.limits().mean(axis=1) + 0.1)
        simulator = PoseIkSimulator(make, "tiago7", realtime=False, make_robot=lambda name, c: GazeboRobot(c, link),
                                    robots=("light", "gazebo"), robot="gazebo")
        snap = run(simulator, 3)
        self.assertEqual(snap["robot"], "gazebo")
        self.assertEqual(len(snap["gazebo"]["lag"]), 7)
        np.testing.assert_allclose(snap["q"], link.q, atol=1e-4)                        # the arm is where Gazebo has it
        goal_pose = simulator.session.q + 0.2
        simulator.state.commands.put({"command": "goal", "matrix": chain.rotation(goal_pose).reshape(9).tolist(),
                                      "position": kinematics.position(chain, goal_pose).tolist()})
        snap = run(simulator, 400)
        self.assertLess(snap["ik"]["goal_error"], 5.0)
        self.assertLess(snap["ik"]["goal_error_mm"], 1000 * 0.04 * workspace(chain).radius)
        self.assertEqual(snap["runs"][-1]["robot"], "gazebo")
        pose = simulator.session.q.copy()
        simulator.state.commands.put({"command": "robot", "robot": "light"})
        snap = run(simulator, 2)
        self.assertEqual(snap["robot"], "light")
        self.assertEqual(simulator.state.config["robot"], "light")
        np.testing.assert_allclose(snap["q"], pose, atol=1e-3)                          # the light robot takes over where the arm is
        simulator.state.commands.put({"command": "robot", "robot": "gazebo"})
        simulator.state.commands.put({"command": "chain", "chain": "generic4"})         # Gazebo has the TIAGo arm only
        snap = run(simulator, 2)
        self.assertEqual((snap["chain"], snap["robot"]), ("generic4", "light"))

    def test_the_goal_sheets_keep_the_goal_without_their_anchor(self):
        network = PoseIkStubNetwork(get_chain("tiago4"), PoseIkDesign(task="position"))
        q = network.chain.limits().mean(axis=1)
        here = kinematics.position(network.chain, q)
        network.step_ik(q, np.zeros(3), here)
        network.set_goal_anchor(False)
        tick = network.step_ik(q, np.zeros(3), here + [0.1, 0.0, 0.0])
        np.testing.assert_allclose(tick.goal_position, here, atol=0.01)                 # still the goal that was given before


class HttpTests(unittest.TestCase):
    def test_page_state_and_commands_over_http(self):
        state = PoseIkSimState()
        server = PoseIkSimServer(state, "127.0.0.1", 0)
        server.start()
        try:
            simulator = PoseIkSimulator(make, "tiago4", state, realtime=False)
            simulator.tick()
            base = "http://127.0.0.1:%d" % server.port
            page = urllib.request.urlopen(base + "/").read().decode("utf-8")
            self.assertIn("Pose inverse kinematics", page)
            self.assertIn("goal sheets", page)
            snap = json.loads(urllib.request.urlopen(base + "/state").read())
            self.assertIn("goal_position", snap["ik"])
            request = urllib.request.Request(base + "/api/random_goal", data=b'{"near": true}', headers={"Content-Type": "application/json"})
            self.assertTrue(json.loads(urllib.request.urlopen(request).read())["ok"])
            self.assertEqual(simulator.tick()["mode"], "ik")
            self.assertIn("gmm", simulator.state.history[-1])
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(urllib.request.Request(base + "/api/nonsense", data=b"{}"))
            for piece in ("/api/robot", "/gazebo.jpg", "TIAGo in Gazebo"):
                self.assertIn(piece, page)
            with self.assertRaises(urllib.error.HTTPError):                              # no Gazebo: no picture
                urllib.request.urlopen(base + "/gazebo.jpg")
            state.image = lambda: b"\xff\xd8 a picture"
            answer = urllib.request.urlopen(base + "/gazebo.jpg")
            self.assertEqual(answer.headers["Content-Type"], "image/jpeg")
            self.assertEqual(answer.read(), b"\xff\xd8 a picture")
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
