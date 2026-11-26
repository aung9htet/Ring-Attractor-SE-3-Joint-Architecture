"""The simulator's HTTP side: the page, a server-sent event stream of ticks, and commands as ``POST /api/<name>``.

``GET /`` the page (static/index.html); ``/config``, ``/state``, ``/history`` JSON; ``/events`` the stream;
``/gazebo.jpg`` the latest picture of Gazebo's camera.  Commands go into a queue the main loop reads every tick.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
COMMANDS = ("goal", "random_goal", "method", "robot", "task", "position_goal", "goal_anchor", "replay", "target", "stop", "speed",
            "chain", "quit")


class SimState:
    """What the HTTP threads and the main loop share."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: List["queue.Queue[str]"] = []
        self.commands: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.config: Dict[str, Any] = {}
        self.snapshot: Dict[str, Any] = {}
        self.history: "deque[Dict[str, Any]]" = deque(maxlen=600)          # 30 s of 50 ms ticks
        #: the latest camera picture of Gazebo as JPEG bytes (None: no Gazebo, or no picture)
        self.image: Optional[Callable[[], Optional[bytes]]] = None

    def subscribe(self) -> "queue.Queue[str]":
        subscriber: "queue.Queue[str]" = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: "queue.Queue[str]") -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, kind: str, payload: Dict[str, Any]) -> None:
        text = "event: %s\ndata: %s\n\n" % (kind, json.dumps(payload, separators=(",", ":")))
        with self._lock:
            if kind == "config":
                self.config = payload
                self.history.clear()
            else:
                self.snapshot = payload
                ik = payload.get("ik") or {}
                self.history.append({"t": payload.get("t"), "error": payload.get("error"), "tr": payload.get("lpy_true"),
                                     "ring": payload.get("lpy_ring"), "goal": ik.get("written"), "gpos": ik.get("written_position"),
                                     "pt": payload.get("pos_true"), "gdeg": ik.get("goal_error"), "gmm": ik.get("goal_error_mm")})
            for subscriber in list(self._subscribers):
                try:
                    subscriber.put_nowait(text)
                except queue.Full:                      # a page that stopped reading: drop its oldest message
                    try:
                        subscriber.get_nowait()
                        subscriber.put_nowait(text)
                    except (queue.Empty, queue.Full):
                        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "RingAttractorKinematicsSimulator/1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def _send(self, body: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json", code)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        state: SimState = self.server.state          # type: ignore[attr-defined]
        path = self.path.split("?")[0]
        if path == "/":
            with open(PAGE, "rb") as stream:
                self._send(stream.read(), "text/html; charset=utf-8")
        elif path == "/config":
            self._json(state.config)
        elif path == "/state":
            self._json(state.snapshot)
        elif path == "/history":
            self._json(list(state.history))
        elif path == "/gazebo.jpg":
            picture = state.image() if state.image is not None else None
            if picture is None:
                self._json({"error": "no picture"}, 404)
            else:
                self._send(picture, "image/jpeg")
        elif path == "/events":
            self._stream(state)
        else:
            self._json({"error": "not found"}, 404)

    def _stream(self, state: SimState) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        subscriber = state.subscribe()
        try:
            if state.config:
                self.wfile.write(("event: config\ndata: %s\n\n" % json.dumps(state.config)).encode("utf-8"))
                self.wfile.flush()
            while True:
                try:
                    self.wfile.write(subscriber.get(timeout=10.0).encode("utf-8"))
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            state.unsubscribe(subscriber)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        state: SimState = self.server.state          # type: ignore[attr-defined]
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


class SimServer:
    def __init__(self, state: SimState, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.httpd.daemon_threads = True
        self.httpd.state = state                         # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
