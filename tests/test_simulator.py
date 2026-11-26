"""The simulator with the stub network: commands, ticks and the HTTP side."""

import json
import unittest
import urllib.request

from ring_attractor_kinematics.simulator.loop import Simulator
from ring_attractor_kinematics.simulator.server import SimServer, SimState
from ring_attractor_kinematics.simulator.stub import StubNetwork


class SimulatorTest(unittest.TestCase):
    def test_a_random_goal_is_reached(self):
        simulator = Simulator(StubNetwork, "tiago4", realtime=False)
        simulator.state.commands.put({"command": "random_goal", "near": True})
        for _ in range(400):
            snapshot = simulator.tick()
        self.assertEqual(snapshot["mode"], "ik")
        self.assertIsNotNone(snapshot["runs"][-1]["reached_s"])

    def test_methods_and_tasks_switch(self):
        simulator = Simulator(StubNetwork, "tiago7", realtime=False)
        for command in ({"command": "method", "method": "dls"}, {"command": "task", "task": "orientation"},
                        {"command": "position_goal", "on": "cells"}, {"command": "chain", "chain": "generic3"}):
            simulator.state.commands.put(command)
            simulator.tick()
        self.assertEqual(simulator.chain.name, "generic3")
        self.assertEqual(simulator.method, "dls")

    def test_http(self):
        state = SimState()
        server = SimServer(state, "127.0.0.1", 0)
        server.start()
        try:
            simulator = Simulator(StubNetwork, "tiago4", state, realtime=False)
            simulator.tick()
            base = "http://127.0.0.1:%d" % server.port
            page = urllib.request.urlopen(base + "/").read().decode("utf-8")
            self.assertIn("<title>", page)
            config = json.loads(urllib.request.urlopen(base + "/config").read())
            self.assertEqual(config["chain"], "tiago4")
            request = urllib.request.Request(base + "/api/stop", data=b"{}", method="POST")
            self.assertTrue(json.loads(urllib.request.urlopen(request).read())["ok"])
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
