"""Progress of a long run, written to stderr.

There are three modes:

* ``bar`` redraws one line in place, for a terminal;
* ``lines`` prints one line per finished item plus a heartbeat every minute, for logs;
* ``off`` keeps the old behaviour: one plain line per finished item on stdout.

``REACHING_PROGRESS`` picks the mode. ``docker/run.bash`` sets it to ``bar`` when its own stderr is a terminal and to
``lines`` otherwise; inside the container stderr is never a terminal. Standard library only, so ``python3 -m ring_attractor_kinematics
status`` also runs on the host.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional, TextIO

MODES = ("bar", "lines", "off")
#: characters of the bar itself
WIDTH = 24
#: the bar is redrawn at most this often (s)
REDRAW_S = 0.5
#: ``lines``: a heartbeat line at least this often while an item runs (s)
HEARTBEAT_S = 60.0


def mode_from_environment(stream: Any = None) -> str:
    mode = os.environ.get("REACHING_PROGRESS", "").strip().lower()
    if mode in MODES:
        return mode
    isatty = getattr(stream if stream is not None else sys.stderr, "isatty", None)
    return "bar" if callable(isatty) and isatty() else "lines"


def clock(seconds: Optional[float]) -> str:
    """``h:mm:ss``; ``--:--:--`` when unknown."""

    if seconds is None or not seconds >= 0.0 or seconds == float("inf"):
        return "--:--:--"
    s = int(round(seconds))
    return "%d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60)


def bar(fraction: float, width: int = WIDTH) -> str:
    """``[#####-------]``: ASCII, so it prints whatever the container's encoding."""

    fraction = min(max(float(fraction), 0.0), 1.0)
    full = int(round(fraction * width))
    return "[" + "#" * full + "-" * (width - full) + "]"


def columns() -> int:
    try:
        return max(40, int(os.environ.get("COLUMNS", "")))
    except ValueError:
        return 120


class Progress:
    """``total`` items (trials, poses, ticks) of which ``done`` were finished before, in ``wall_before`` seconds (a run
    that resumes). Construct it once the slow set-up (the network build) is over, so the time per item excludes it.

    Use :meth:`start` when an item begins, :meth:`update` while it runs, and :meth:`finish` when it ends.
    :meth:`write` prints a line without breaking the bar, and :meth:`close` ends the bar's line.
    """

    def __init__(self, label: str, total: int, done: int = 0, wall_before: float = 0.0, reached: int = 0, unit: str = "trial",
                 stream: Optional[TextIO] = None, mode: Optional[str] = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.mode = mode if mode in MODES else mode_from_environment(self.stream)
        self.label, self.unit = str(label), str(unit)
        self.total, self.done, self.reached = int(total), int(done), int(reached)
        self.done_before, self.wall_before = int(done), float(wall_before)
        self.started = time.time()
        self.item: Optional[str] = None
        self.detail = ""
        self._drawn = 0.0
        self._said, self._said_done = time.time(), int(done)
        self._shown = False

    # -- numbers -------------------------------------------------------------
    def seconds_per_item(self) -> Optional[float]:
        here = self.done - self.done_before
        if here > 0:
            return (time.time() - self.started) / here
        if self.done_before > 0 and self.wall_before > 0.0:
            return self.wall_before / self.done_before
        return None

    def eta(self) -> Optional[float]:
        per = self.seconds_per_item()
        return None if per is None else per * max(self.total - self.done, 0)

    def status(self) -> str:
        """The bar, the count, what is running now and the times, as one line."""

        fraction = self.done / self.total if self.total else 1.0
        text = "%s %s %d/%d %3.0f%%" % (self.label, bar(fraction), self.done, self.total, 100.0 * fraction)
        if self.unit == "trial":
            text += " reached %d" % self.reached
        if self.item is not None:
            text += " | %s %s%s" % (self.unit, self.item, ": " + self.detail if self.detail else "")
        elif self.detail:
            text += " | " + self.detail
        return text + " | %s elapsed, ETA %s" % (clock(time.time() - self.started), clock(self.eta()))

    # -- events --------------------------------------------------------------
    def start(self, item: Any, detail: str = "") -> None:
        self.item, self.detail = str(item), str(detail)
        self._draw(force=True)

    def update(self, detail: str) -> None:
        self.detail = str(detail)
        now = time.time()
        if self.mode == "bar" and now - self._drawn >= REDRAW_S:
            self._draw(force=True)
        elif self.mode == "lines" and now - self._said >= HEARTBEAT_S:
            self._line(self.status())

    def advance(self, count: int = 1, detail: str = "") -> None:
        """``count`` more items done, without a line for each (the ticks of a recording, the cases of a sweep)."""

        self.done += int(count)
        self.update(detail)

    def finish(self, reached: Optional[bool] = None, line: str = "") -> None:
        """One item finished (``reached``: its success, for trials); ``line`` says how it went."""

        self.done += 1
        if reached:
            self.reached += 1
        self.item, self.detail = None, ""
        if self.mode == "off":
            if line:
                print(line, flush=True)
            return
        if self.mode == "lines":
            self._line((line + " | " if line else "") + self.status())
            return
        if line:
            self.write(line)            # redraws the bar below the line
        else:
            self._draw(force=True)

    def write(self, line: str) -> None:
        """A line of its own (a log message), above the bar."""

        if self.mode == "off":
            print(line, flush=True)
            return
        if self.mode == "bar" and self._shown:
            self.stream.write("\r\033[K")
        self.stream.write(str(line) + "\n")
        self.stream.flush()
        self._shown = False
        if self.mode == "bar":
            self._draw(force=True)

    def close(self) -> None:
        """End the bar's line; for logs, a last status line unless the last one already said it all."""

        if self.mode == "bar" and self._shown:
            self.stream.write("\n")
            self.stream.flush()
            self._shown = False
        elif self.mode == "lines" and self._said_done != self.done:
            self._line(self.status())

    # -- drawing -------------------------------------------------------------
    def _draw(self, force: bool = False) -> None:
        if self.mode != "bar":
            return
        now = time.time()
        if not force and now - self._drawn < REDRAW_S:
            return
        text = self.status()[: columns() - 1]
        self.stream.write("\r\033[K" + text)
        self.stream.flush()
        self._drawn, self._shown = now, True

    def _line(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()
        self._said, self._said_done = time.time(), self.done


def say(text: str) -> None:
    """A line of its own, outside any bar: on stderr, or on stdout with ``REACHING_PROGRESS=off``."""

    if mode_from_environment() == "off":
        print(text, flush=True)
    else:
        sys.stderr.write(str(text) + "\n")
        sys.stderr.flush()


def trial_detail(t_s: float, limit_s: float, error_deg: Optional[float] = None, error_mm: Optional[float] = None) -> str:
    """``t 7.3/25 s, 3.1 deg 21 mm``: where a running trial is."""

    text = "t %4.1f/%g s" % (float(t_s), float(limit_s))
    if error_deg is not None:
        text += ", %.1f deg" % float(error_deg)
    if error_mm is not None:
        text += " %.0f mm" % float(error_mm)
    return text


def demo(items: int = 3, ticks: int = 20, pause_s: float = 0.05) -> None:
    """Draw a short fake run (checks the bar through ``docker/run.bash``; writes no results)."""

    progress = Progress("demo", items)
    for item in range(items):
        progress.start(item)
        for tick in range(ticks):
            progress.update(trial_detail(0.05 * (tick + 1), 0.05 * ticks, 10.0 / (tick + 1), 100.0 / (tick + 1)))
            time.sleep(pause_s)
        progress.finish(item % 2 == 0, "demo item %d: done" % item)
    progress.close()


if __name__ == "__main__":
    demo()
