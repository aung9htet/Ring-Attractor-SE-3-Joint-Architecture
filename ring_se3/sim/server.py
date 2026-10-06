"""The simulator: the spiking pose network live behind one browser page.

The server of :mod:`ring_so3.sim.server` with the position added: the true and the decoded end
point, the three sheets, and switches for the anchor and for path integration.

* ``GET /``, ``/config``, ``/state``, ``/history``, ``/events`` as there;
* ``POST /api/target|mode|speed|chain|random|lead|quit`` as there;
* ``POST /api/anchor`` ``{"on": true|false}``   the position computed from the rings holds the packets (or not);
* ``POST /api/pi``     ``{"on": true|false}``   the end point's velocity moves the packets (or not).

:class:`PoseStubNetwork` stands in for the spiking network in the tests and in
``scripts/simulator.py --stub``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from ring_so3.kinematics import Chain
from ring_so3.sim import server as so3_server
from ring_so3.sim.robot import GoalSequence, SimRobot

from .. import kinematics
from ..chains import MODEL_NAMES, get_chain, registry, workspace
from ..session import PoseSession
from ..sheet import PLANES, SheetDesign, SheetMap, combine
from .page import PAGE_HTML

COMMANDS = ("target", "mode", "speed", "chain", "random", "lead", "anchor", "pi", "quit")


class PoseStubNetwork(so3_server.StubNetwork):
    """Exact kinematics with a little noise in the shape of the pose network (no NEST): anchored, the position is
    the true one; not anchored, it is carried by the velocity (15% too fast) or stays where it was."""

    def __init__(self, chain: Chain, noise_deg: float = 0.5, noise_position: float = 0.006, seed: int = 0) -> None:
        super().__init__(chain, noise_deg, seed)
        from ..nest.network import PoseTick        # needs no NEST to import

        self._pose_tick = PoseTick
        self.space = workspace(chain)
        self.mapping = SheetMap(SheetDesign())
        self.noise_position = float(noise_position)
        self.anchor_on, self.path_integration_on = True, True
        self._scaled = self.space.scaled(kinematics.position(chain, np.zeros(chain.n)))

    def set_anchor(self, on: bool) -> None:
        self.anchor_on = bool(on)

    def set_path_integration(self, on: bool) -> None:
        self.path_integration_on = bool(on)

    def step(self, q, qdot=None):                  # type: ignore[override]
        tick = super().step(q)
        angles = np.asarray(q, dtype=float)
        rates = np.zeros(self.chain.n) if qdot is None else np.asarray(qdot, dtype=float)
        true = self.space.scaled(kinematics.position(self.chain, angles))
        velocity = kinematics.velocity(self.chain, angles, rates) / self.space.radius
        if self.anchor_on:
            self._scaled = true + self.rng.normal(0.0, self.noise_position, 3)
        elif self.path_integration_on:
            self._scaled = self._scaled + 1.15 * velocity * self.options.tick_ms / 1000.0
        places = np.array([self._scaled[[a, b]] for _, a, b in PLANES])
        width = 2.0 * self.mapping.cell_size
        counts = np.array([np.round(3.0 * np.exp(-np.sum((self.mapping.places - place) ** 2, axis=1) / (2 * width ** 2))) for place in places])
        scaled, disagreement = combine({name: self.mapping.centre(counts[i]) for i, (name, _, _) in enumerate(PLANES)})
        cells = true + self.rng.normal(0.0, self.noise_position / 2.0, 3)
        return self._pose_tick(tick, cells, velocity / 1.5, places, counts, scaled, disagreement,
                               self.space.metres(scaled), self.space.metres(cells))

    def so3_neuron_count(self) -> int:
        return 0


class PoseSimState(so3_server.SimState):
    def publish(self, kind: str, payload: Dict[str, Any]) -> None:
        super().publish(kind, payload)
        if kind != "config" and self.history:
            with self._lock:
                self.history[-1].update({"pt": payload.get("pos_true"), "pd": payload.get("pos"), "perr": payload.get("pos_error_mm")})


class _Handler(so3_server._Handler):
    server_version = "RingSE3Sim/1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?")[0] == "/":
            self._send(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        state = self.server.state          # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length else {}
        except ValueError:
            self._json({"error": "the body is not JSON"}, 400)
            return
        name = self.path.split("?")[0]
        if not name.startswith("/api/") or name[len("/api/"):] not in COMMANDS:
            self._json({"error": "unknown command"}, 404)
            return
        state.commands.put(dict(payload if isinstance(payload, dict) else {}, command=name[len("/api/"):]))
        self._json({"ok": True})


class PoseSimServer(so3_server.SimServer):
    def __init__(self, state: PoseSimState, host: str = "0.0.0.0", port: int = 8765) -> None:
        import threading
        from http.server import ThreadingHTTPServer

        self.httpd = ThreadingHTTPServer((host, int(port)), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.state = state                         # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)


class PoseSimulator(so3_server.Simulator):
    def __init__(self, make_network: Callable[[Chain], Any], chain_name: str = "tiago7", state: Optional[PoseSimState] = None,
                 speed: float = 0.5, mode: str = "manual", seed: int = 0, realtime: bool = True, lead: bool = False) -> None:
        super().__init__(make_network, chain_name, state or PoseSimState(), speed, mode, seed, realtime, lead)

    def load(self, chain_name: str) -> None:
        self.chain = get_chain(chain_name)
        started = time.time()
        self.network = self.make_network(self.chain)
        self.build_s = time.time() - started
        start = np.zeros(self.chain.n) if self.chain.joints[0].circular else self.chain.limits().mean(axis=1)
        self.robot = SimRobot(self.chain, start, max_speed=self.speed)
        self.session = PoseSession(self.network, self.robot, lead=self.lead)
        self.goals = GoalSequence(self.chain, self.seed, dwell_s=1.0)
        self.dt_s = self.session.dt_s
        self.state.publish("config", self.config())

    def config(self) -> Dict[str, Any]:
        out = super().config()
        space, mapping = self.network.space, self.network.mapping
        out.update({
            "chains": MODEL_NAMES, "stub": isinstance(self.network, PoseStubNetwork),
            "radius": float(space.radius), "centre": [float(v) for v in space.centre], "reach": float(space.radius + np.linalg.norm(space.centre)),
            "sheet": {"size": int(mapping.n), "extent": float(mapping.design.extent)}, "planes": [name for name, _, _ in PLANES],
            "so3_neurons": int(self.network.so3_neuron_count()),
        })
        return out

    def handle(self, command: Dict[str, Any]) -> None:
        name = command.get("command")
        if name == "anchor":
            self.network.set_anchor(bool(command.get("on")))
        elif name == "pi":
            self.network.set_path_integration(bool(command.get("on")))
        elif name == "chain":
            if command.get("chain") in registry():
                self.load(command["chain"])
        else:
            super().handle(command)

    def snapshot(self, tick: Any) -> Dict[str, Any]:
        out = super().snapshot(tick)
        network, mapping, space = self.network, self.network.mapping, self.network.space
        true = kinematics.position(self.chain, self.robot.q)
        scaled = space.scaled(true)
        r4 = lambda values: [None if not np.isfinite(v) else round(float(v), 4) for v in values]  # noqa: E731
        error = kinematics.position_error(tick.position, true)
        sheets, packets = [], []
        for i in range(len(PLANES)):
            counts = tick.sheet_counts[i]
            active = np.flatnonzero(counts)
            sheets.append([[int(k), int(counts[k])] for k in active])
            packets.append(r4(mapping.to_cells(tick.places[i])) if np.all(np.isfinite(tick.places[i])) else None)
        out.update({
            "pos_true": r4(true), "pos": r4(tick.position), "pos_cells": r4(tick.cells_position),
            "pos_error_mm": None if not np.isfinite(error) else round(1000.0 * error, 1),
            "cells_error_mm": round(1000.0 * kinematics.position_error(tick.cells_position, true), 1),
            "anchor": bool(network.anchor_on), "pi": bool(network.path_integration_on),
            "sheets": sheets, "sheet_packet": packets,
            "sheet_true": [r4(mapping.to_cells(scaled[[a, b]])) for _, a, b in PLANES],
            "disagreement_mm": r4(1000.0 * space.radius * tick.disagreement),
        })
        return out


__all__ = ["COMMANDS", "PoseSimServer", "PoseSimState", "PoseSimulator", "PoseStubNetwork"]
