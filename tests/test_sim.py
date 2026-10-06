"""The simulator's server and loop with the stand-in network (no NEST)."""

import json
import unittest
import urllib.request

import numpy as np

from ring_se3 import kinematics
from ring_se3.chains import get_chain
from ring_se3.sim.server import PoseSimServer, PoseSimState, PoseSimulator, PoseStubNetwork


def make(chain):
    return PoseStubNetwork(chain)


class LoopTests(unittest.TestCase):
    def test_snapshot_holds_the_true_and_the_decoded_position(self):
        simulator = PoseSimulator(make, "tiago4", realtime=False, mode="goals", speed=1.0)
        config = simulator.state.config
        self.assertEqual(config["chain"], "tiago4")
        self.assertEqual(config["sheet"]["size"], 48)
        self.assertEqual(config["planes"], ["xy", "xz", "yz"])
        for _ in range(60):
            snap = simulator.tick()
        true = kinematics.position(simulator.chain, simulator.robot.q)
        np.testing.assert_allclose(snap["pos_true"], true, atol=1e-3)
        self.assertLess(snap["pos_error_mm"], 40.0)
        self.assertEqual(len(snap["sheets"]), 3)
        self.assertGreater(len(snap["sheets"][0]), 3)
        self.assertTrue(snap["anchor"] and snap["pi"])
        self.assertEqual(len(simulator.state.history), 60)
        self.assertIn("perr", simulator.state.history[-1])

    def test_switches(self):
        simulator = PoseSimulator(make, "generic3", realtime=False, mode="goals", speed=1.0)
        for _ in range(10):
            simulator.tick()
        simulator.state.commands.put({"command": "anchor", "on": False})
        simulator.state.commands.put({"command": "pi", "on": False})
        snap = simulator.tick()
        self.assertFalse(snap["anchor"] or snap["pi"])
        held = snap["pos"]
        for _ in range(30):
            snap = simulator.tick()
        np.testing.assert_allclose(snap["pos"], held, atol=0.03)            # nothing moves the packets: they stay
        self.assertGreater(snap["pos_error_mm"], 20.0)                      # while the arm has moved on
        simulator.state.commands.put({"command": "anchor", "on": True})
        for _ in range(3):
            snap = simulator.tick()
        self.assertLess(snap["pos_error_mm"], 40.0)

    def test_chain_can_be_changed(self):
        simulator = PoseSimulator(make, "generic2", realtime=False)
        simulator.state.commands.put({"command": "chain", "chain": "tiago7"})
        snap = simulator.tick()
        self.assertEqual(snap["chain"], "tiago7")
        self.assertEqual(len(snap["q"]), 7)
        self.assertEqual(get_chain("tiago7").n, 7)


class HttpTests(unittest.TestCase):
    def test_page_state_and_commands_over_http(self):
        state = PoseSimState()
        server = PoseSimServer(state, "127.0.0.1", 0)
        server.start()
        try:
            simulator = PoseSimulator(make, "generic3", state, realtime=False)
            simulator.tick()
            base = "http://127.0.0.1:%d" % server.port
            page = urllib.request.urlopen(base + "/").read().decode("utf-8")
            self.assertIn("Pose from ring attractors", page)
            self.assertIn("attractor sheets", page)
            snap = json.loads(urllib.request.urlopen(base + "/state").read())
            self.assertIn("pos_true", snap)
            request = urllib.request.Request(base + "/api/anchor", data=b'{"on": false}', headers={"Content-Type": "application/json"})
            self.assertTrue(json.loads(urllib.request.urlopen(request).read())["ok"])
            self.assertFalse(simulator.tick()["anchor"])
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(urllib.request.Request(base + "/api/nonsense", data=b"{}"))
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
