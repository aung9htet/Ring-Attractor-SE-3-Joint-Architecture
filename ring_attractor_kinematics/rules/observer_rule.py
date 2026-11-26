"""Where the turn's orientation comes from: the old method, the proposed observer and its controls (no neurons).

The forward network shows the arm ``τ`` late and a little noisy.  Every source below turns that delayed
view (and the joint velocities the controller itself commands, which it knows without delay) into the
orientation the turn ``e = ½ Σ r_j × g_j`` is computed from.  They plug into
:func:`ring_attractor_kinematics.rules.cascade_rule.simulate` (argument ``orientation``) and into the open-loop estimation
benchmark, so they are compared on exactly the same trials.

=====  ==============================  =========================================================================
code   source                          what it is
=====  ==============================  =========================================================================
M0     :class:`MatrixSource`           the old method: the decoded matrix of the hand, ``τ`` late, noisy
M1     ``MatrixSource(lead=True)``     the old method with joint-angle lead: angles written ``τ`` ahead
M2     :class:`ReadoutRingSource`      the passive readout rings: lift, pitch, yaw as bumps (silent where a
                                       ring's drive is below 0.25), read over a 200 ms window
M3     ``ChartObserver(predict=False,  recurrent observer rings in one chart, cue-driven only (memory and
       charts=1)``                     filtering)
M4     ``ChartObserver(charts=1)``     M3 with prediction: ``ω = Σ a_k q̇_k`` moves the bumps, the delayed cue
                                       corrects against the estimate of ``τ`` ago
M5     ``ChartObserver(charts=2)``     the full proposal: two charts (``U₀ = I``, ``U₁ = R_y(π/2)``) with
                                       disjoint poles, hysteresis selection, confidence, and fallback to M0
MQ     :class:`MatrixObserver`         the same predictor-corrector on SO(3) itself (no chart, no pole)
MB     :class:`ObserverAsBuilt`        the observer method exactly as the spiking network computes it (M5 with the
                                       changes its circuit forces: the stored past is a plain delay, cues are
                                       not normalised, the rates stop at the shift cells' top speed, selection and
                                       confidence as the selector and confidence cells compute them)
=====  ==============================  =========================================================================

Chart rates are bounded as a spiking population would bound them: ``1/cos θ`` is never more than
:data:`RATE_BOUND` (the decoders are fitted for ``|θ| ≤ 60°``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import Chain
from reaching_benchmark.geometry.rotations import from_lift_pitch_yaw, lift_pitch_yaw, nearest_rotation, rodrigues, rot, rotation_angle, wrap

from ..config import ObserverDesign

#: 1/cos θ is bounded here: a chart is trusted for |pitch| ≤ 60°
RATE_BOUND = 2.0
CHART_LIMIT_RAD = float(np.radians(60.0))
#: a readout ring is silent when the pair that drives it is shorter than this
RING_THRESHOLD = 0.25
#: the readout rings are read over 4 ticks: on average 1.5 ticks (75 ms) older than the matrix cells
RING_WINDOW_LAG_S = 0.075
CHARTS = (np.eye(3), rot("y", np.pi / 2.0))


@dataclass(frozen=True)
class SourceSettings:
    """What every source sees: the forward network's lag and noise."""

    #: per-axis s.d. of the decoded matrix's error, deg (a held pose reads to about 1–1.5°)
    cue_noise_deg: float = 0.8
    #: per-angle s.d. of a readout ring's angle, deg
    ring_noise_deg: float = 1.0
    #: correction time constant of the observers (s): the cue pulls the estimate with gain dt/τ_c per tick
    correction_s: float = 0.2
    #: hysteresis of the chart selector: switch only when the other chart's |pitch| is smaller by this (rad)
    hysteresis_rad: float = float(np.radians(10.0))
    #: fallback: below this confidence (cos of the selected chart's pitch, times cue agreement) use the matrix
    fallback_confidence: float = 0.3
    seed: int = 0


def euler_rate_map(lpy: Sequence[float]) -> np.ndarray:
    """``E`` with ``ω = E · (lifṫ, pitcḣ, yaẇ)`` for ``R = Rz(yaw) Ry(pitch) Rx(lift)``, ω in the base frame."""

    _, pitch, yaw = [float(v) for v in lpy]
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, -sy, 0.0], [sy * cp, cy, 0.0], [-sp, 0.0, 1.0]])


def chart_rates(lpy: Sequence[float], omega: np.ndarray, bound: float = RATE_BOUND) -> np.ndarray:
    """``(lifṫ, pitcḣ, yaẇ) = E⁻¹ ω`` for a base-frame ``ω``, with ``1/cos pitch`` held at ``bound``::

        lifṫ  = (cos yaw ω_x + sin yaw ω_y) / cos pitch
        pitcḣ = −sin yaw ω_x + cos yaw ω_y
        yaẇ   = ω_z + sin pitch · lifṫ
    """

    _, pitch, yaw = [float(v) for v in lpy]
    cy, sy = np.cos(yaw), np.sin(yaw)
    secant = min(1.0 / max(abs(np.cos(pitch)), 1e-9), float(bound))
    w = np.asarray(omega, dtype=float)
    lift_rate = (cy * w[0] + sy * w[1]) * secant
    return np.array([lift_rate, -sy * w[0] + cy * w[1], w[2] + np.sin(pitch) * lift_rate])


def noisy(rotation: np.ndarray, sd_deg: float, rng: np.random.Generator) -> np.ndarray:
    if sd_deg <= 0.0:
        return rotation
    return rodrigues(rng.normal(0.0, np.radians(sd_deg), 3)) @ rotation


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """The rotation vector of a rotation matrix (inverse of Rodrigues)."""

    r = np.asarray(rotation, dtype=float)
    angle = float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * np.sin(angle))
    return angle * axis


class Source:
    """Base: ``observe(seen_q, seen_qdot, q, qdot, dt) -> (pose the forward network shows, orientation for the turn)``."""

    code = ""

    def __init__(self, chain: Chain, delay_s: float, settings: Optional[SourceSettings] = None) -> None:
        self.chain = chain
        self.delay_s = float(delay_s)
        self.settings = settings or SourceSettings()
        self.rng = np.random.default_rng(self.settings.seed)
        self.cue_on = True
        #: replay: the network's decoded matrix of this tick (and its readout-ring angles), in place of the noise model
        self.replay: Optional[np.ndarray] = None
        self.replay_angles: Optional[np.ndarray] = None

    def reset(self, q: Sequence[float]) -> None:
        self.rng = np.random.default_rng(self.settings.seed)

    def cue(self, seen_q: np.ndarray) -> Optional[np.ndarray]:
        """The decoded matrix of the hand (``None`` while the cue is dropped)."""

        if not self.cue_on:
            return None
        if self.replay is not None:
            return nearest_rotation(self.replay)
        return noisy(self.chain.rotation(seen_q), self.settings.cue_noise_deg, self.rng)

    def observe(self, seen_q: np.ndarray, seen_qdot: np.ndarray, q: np.ndarray, qdot: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class MatrixSource(Source):
    """M0 (``lead=False``): the decoded matrix.  M1 (``lead=True``): joint angles written ``τ`` ahead on the joint
    rings, so the whole forward network (matrix, axes, position) shows ``q(t−τ) + τ·q̇(t−τ)``."""

    def __init__(self, chain: Chain, delay_s: float, settings: Optional[SourceSettings] = None, lead: bool = False) -> None:
        super().__init__(chain, delay_s, settings)
        self.lead = bool(lead)
        self.code = "M1" if lead else "M0"
        self.last: Optional[np.ndarray] = None

    def observe(self, seen_q, seen_qdot, q, qdot, dt):
        shown = seen_q
        if self.lead:
            ahead = seen_q + self.delay_s * seen_qdot
            shown = np.where(self.chain.circular_mask(), wrap(ahead), self.chain.clip(ahead))
        cue = self.cue(shown)
        self.last = cue if cue is not None else (self.last if self.last is not None else self.chain.rotation(shown))
        return shown, self.last


class ReadoutRingSource(Source):
    """M2: lift, pitch, yaw read from the readout rings, the matrix rebuilt from them.  The lift and yaw rings are
    driven by pairs of length cos(pitch): above |pitch| ≈ 75.5° they fall silent and hold their last angle."""

    code = "M2"

    def reset(self, q):
        super().reset(q)
        self.angles = np.array(lift_pitch_yaw(self.chain.rotation(q)))

    def observe(self, seen_q, seen_qdot, q, qdot, dt):
        if self.replay_angles is not None:
            if self.cue_on:
                held = np.isfinite(self.replay_angles)
                self.angles[held] = self.replay_angles[held]
            return seen_q, from_lift_pitch_yaw(*self.angles)
        older = seen_q - RING_WINDOW_LAG_S * seen_qdot
        rotation = self.chain.rotation(older)
        if self.cue_on:
            true = np.array(lift_pitch_yaw(rotation))
            noise = self.rng.normal(0.0, np.radians(self.settings.ring_noise_deg), 3)
            drive = abs(np.cos(true[1]))
            if drive >= RING_THRESHOLD:
                self.angles[0], self.angles[2] = true[0] + noise[0], true[2] + noise[2]
            self.angles[1] = true[1] + noise[1]
        return seen_q, from_lift_pitch_yaw(*self.angles)


class ChartObserver(Source):
    """M3 / M4 / M5: observer rings that hold lift, pitch, yaw of the hand in one or two charts.

    Each chart ``c`` holds the angles of ``R_c = U_cᵀ R``.  Prediction moves them at the chart rates of
    ``U_cᵀ ω`` (``ω = Σ a_k q̇_k`` from the commanded joint velocities, through the axes of the delayed view);
    the cue (the delayed matrix, gated off where that chart's |pitch| > 60°) corrects them against what the chart
    held ``τ`` ago.  With two charts the selector takes the one whose pitch is nearer 0 (with hysteresis); a low
    confidence falls back to the matrix."""

    def __init__(self, chain: Chain, delay_s: float, settings: Optional[SourceSettings] = None, charts: int = 1,
                 predict: bool = True, fallback: bool = True) -> None:
        super().__init__(chain, delay_s, settings)
        self.charts = CHARTS[: int(charts)]
        self.predict = bool(predict)
        self.fallback = bool(fallback) and int(charts) > 1
        self.code = "M5" if charts > 1 else ("M4" if predict else "M3")
        self.selected = 0
        self.confidence = 1.0

    def reset(self, q):
        super().reset(q)
        rotation = self.chain.rotation(q)
        self.state = [np.array(lift_pitch_yaw(u.T @ rotation)) for u in self.charts]
        #: the last ``steps`` estimates (after correction): history[c][0] is the estimate of the time the cue shows
        self.steps = int(round(self.delay_s / 0.05))
        self.history: List[List[np.ndarray]] = [[s.copy() for _ in range(self.steps)] for s in self.state]
        self.selected = int(np.argmax([abs(np.cos(s[1])) for s in self.state]))
        self.last_cue = rotation

    def observe(self, seen_q, seen_qdot, q, qdot, dt):
        gain = min(1.0, float(dt) / self.settings.correction_s)
        omega = kinematics.joint_axes(self.chain, seen_q).T @ np.asarray(qdot, dtype=float) if self.predict else np.zeros(3)
        cue = self.cue(seen_q)
        agreement = 1.0
        for c, u in enumerate(self.charts):
            state = self.state[c]
            if self.predict:
                state = state + dt * chart_rates(state, u.T @ omega)
            if cue is not None:
                seen = np.array(lift_pitch_yaw(u.T @ cue))
                if abs(seen[1]) <= CHART_LIMIT_RAD or len(self.charts) == 1:
                    reference = self.history[c][0] if (self.predict and self.steps) else state
                    innovation = np.array([wrap(seen[0] - reference[0]), seen[1] - reference[1], wrap(seen[2] - reference[2])])
                    if abs(seen[1]) > CHART_LIMIT_RAD:              # one chart near its pole: lift and yaw mean little
                        innovation[[0, 2]] *= max(abs(np.cos(seen[1])), 0.0)
                    state = state + gain * innovation
                    if self.predict:                                # the stored past moves with the correction, so the
                        self.history[c] = [h + gain * innovation for h in self.history[c]]      # next cue does not recount it
                    if c == self.selected:
                        agreement = float(np.exp(-np.linalg.norm(innovation) / np.radians(20.0)))
            state = np.array([wrap(state[0]), np.clip(state[1], -np.pi / 2, np.pi / 2), wrap(state[2])])
            self.state[c] = state
            if self.steps:
                self.history[c] = self.history[c][1:] + [state.copy()]
        if len(self.charts) > 1:
            other = 1 - self.selected
            if abs(self.state[other][1]) > CHART_LIMIT_RAD:        # the chart away from its range is re-seeded from the
                held = self.charts[self.selected] @ from_lift_pitch_yaw(*self.state[self.selected])      # observer matrix
                self.state[other] = np.array(lift_pitch_yaw(self.charts[other].T @ held))
                self.history[other] = [self.state[other].copy() for _ in range(self.steps)]
            pitches = [abs(s[1]) for s in self.state]
            if pitches[other] + self.settings.hysteresis_rad < pitches[self.selected]:
                self.selected = other
        self.confidence = float(abs(np.cos(self.state[self.selected][1]))) * agreement
        if cue is not None:
            self.last_cue = cue
        if self.fallback and self.confidence < self.settings.fallback_confidence:
            return seen_q, self.last_cue
        return seen_q, self.charts[self.selected] @ from_lift_pitch_yaw(*self.state[self.selected])


class MatrixObserver(Source):
    """MQ: the predictor-corrector on SO(3) itself: ``R̂ ← exp([ω]× dt) R̂``, then ``R̂ ← R̂ exp(g · log(R̂(t−τ)ᵀ cue))``."""

    code = "MQ"

    def reset(self, q):
        super().reset(q)
        self.estimate = self.chain.rotation(q)
        self.steps = int(round(self.delay_s / 0.05))
        self.history = [self.estimate.copy() for _ in range(self.steps)]

    def observe(self, seen_q, seen_qdot, q, qdot, dt):
        gain = min(1.0, float(dt) / self.settings.correction_s)
        omega = kinematics.joint_axes(self.chain, seen_q).T @ np.asarray(qdot, dtype=float)
        estimate = rodrigues(omega * dt) @ self.estimate
        cue = self.cue(seen_q)
        if cue is not None:
            reference = self.history[0] if self.steps else estimate
            correction = rodrigues(gain * rotation_log(reference.T @ cue))
            estimate = estimate @ correction
            self.history = [h @ correction for h in self.history]      # the stored past moves with the correction
        u, _, vt = np.linalg.svd(estimate)
        self.estimate = u @ vt
        if self.steps:
            self.history = self.history[1:] + [self.estimate.copy()]
        return seen_q, self.estimate


class ObserverAsBuilt(Source):
    """MB: the observer method as built (see :class:`ring_attractor_kinematics.config.ObserverDesign`), one tick at a time.

    Per chart ``c`` (``R_c = U_cᵀ R``): the bumps move at ``chart_rates(state, U_cᵀ ω)`` plus the innovation over
    ``correction_s``, at most ``top_rad_s``.  The innovation compares the cue with the chart's estimate of ``τ``
    ago, through sines of angle differences, the lift and yaw ones scaled by the cue's ``cos pitch`` (the cue's
    pairs come straight from matrix entries); a chart whose cue pitch is beyond ``cue_limit_deg`` gets none.  Each
    chart's *fitness* is its ``cos pitch`` less ``confidence_weight`` times its summed |innovation|: away from its
    pole *and* agreeing with the cue.  The selector moves to the other chart when its fitness is larger by
    ``selector_margin``; the selected chart's fitness is the confidence, and below ``fallback_confidence`` the late
    matrix goes to the turn."""

    code = "MB"
    #: the shift cells' measured top speed (artifacts/shift_cells.json, fit.top_rad_s)
    top_rad_s = 2.34

    def __init__(self, chain: Chain, delay_s: float, settings: Optional[SourceSettings] = None,
                 design: Optional[ObserverDesign] = None) -> None:
        super().__init__(chain, delay_s, settings)
        self.design = design or ObserverDesign()
        self.selected = 0
        self.confidence = 1.0
        self.fallen_back = False

    def reset(self, q):
        super().reset(q)
        rotation = self.chain.rotation(q)
        self.state = [np.array(lift_pitch_yaw(u.T @ rotation)) for u in CHARTS]
        self.steps = int(round(self.delay_s / 0.05))
        self.history = [[s.copy() for _ in range(max(self.steps, 1))] for s in self.state]
        self.selected = int(np.argmax([np.cos(s[1]) for s in self.state]))
        self.last_cue = rotation

    def innovation(self, chart: int, cue: np.ndarray) -> np.ndarray:
        r = CHARTS[chart].T @ cue
        cos_pitch = float(np.hypot(r[0, 0], r[1, 0]))
        if cos_pitch < np.cos(np.radians(self.design.cue_limit_deg)):
            return np.zeros(3)
        lift, pitch, yaw = np.arctan2(r[2, 1], r[2, 2]), np.arctan2(-r[2, 0], cos_pitch), np.arctan2(r[1, 0], r[0, 0])
        past = self.history[chart][0]
        return np.array([cos_pitch * np.sin(lift - past[0]), np.sin(pitch - past[1]), cos_pitch * np.sin(yaw - past[2])])

    def observe(self, seen_q, seen_qdot, q, qdot, dt):
        d = self.design
        omega = kinematics.joint_axes(self.chain, seen_q).T @ np.asarray(qdot, dtype=float)
        cue = self.cue(seen_q)
        innovations = []
        for c, u in enumerate(CHARTS):
            state = self.state[c]
            rates = chart_rates_with_secant(state, u.T @ omega, 1.0 / max(float(np.cos(state[1])), 0.5))
            innovation = self.innovation(c, cue) if cue is not None else np.zeros(3)
            total = np.clip(rates + innovation / d.correction_s, -self.top_rad_s, self.top_rad_s)
            self.state[c] = wrap(state + float(dt) * total)
            self.history[c] = self.history[c][1:] + [self.state[c].copy()]
            innovations.append(innovation)
        fitness = [max(float(np.cos(s[1])), 0.0) - d.confidence_weight * float(np.sum(np.abs(i))) for s, i in zip(self.state, innovations)]
        other = 1 - self.selected
        if fitness[other] - fitness[self.selected] > d.selector_margin:
            self.selected = other
        self.confidence = fitness[self.selected]
        if cue is not None:
            self.last_cue = cue
        self.fallen_back = self.confidence < d.fallback_confidence
        if self.fallen_back:
            return seen_q, self.last_cue
        return seen_q, CHARTS[self.selected] @ from_lift_pitch_yaw(*self.state[self.selected])


def chart_rates_with_secant(lpy: Sequence[float], omega: np.ndarray, secant: float) -> np.ndarray:
    """:func:`chart_rates` with ``1/cos pitch`` replaced by ``secant`` (the circuit decodes 1/max(cos pitch, 0.5))."""

    _, pitch, yaw = [float(v) for v in lpy]
    cy, sy = np.cos(yaw), np.sin(yaw)
    w = np.asarray(omega, dtype=float)
    lift_rate = (cy * w[0] + sy * w[1]) * secant
    return np.array([lift_rate, -sy * w[0] + cy * w[1], w[2] + np.sin(pitch) * lift_rate])


CODES = ("M0", "M1", "M2", "M3", "M4", "M5", "MQ", "MB")
LABELS: Dict[str, str] = {"M0": "matrix (old)", "M1": "matrix + joint lead", "M2": "readout rings", "M3": "observer, cue only",
                          "M4": "observer + prediction", "M5": "two-chart observer (proposal)", "MQ": "SO(3) observer",
                   "MB": "two-chart observer (as built)"}


def make_source(code: str, chain: Chain, delay_s: float, settings: Optional[SourceSettings] = None) -> Source:
    if code == "M0":
        return MatrixSource(chain, delay_s, settings)
    if code == "M1":
        return MatrixSource(chain, delay_s, settings, lead=True)
    if code == "M2":
        return ReadoutRingSource(chain, delay_s, settings)
    if code == "M3":
        return ChartObserver(chain, delay_s, settings, charts=1, predict=False)
    if code == "M4":
        return ChartObserver(chain, delay_s, settings, charts=1, predict=True)
    if code == "M5":
        return ChartObserver(chain, delay_s, settings, charts=2, predict=True)
    if code == "MQ":
        return MatrixObserver(chain, delay_s, settings)
    if code == "MB":
        return ObserverAsBuilt(chain, delay_s, settings)
    raise ValueError("unknown source %r; known: %s" % (code, ", ".join(CODES)))


def open_loop(source: Source, chain: Chain, path: np.ndarray, dt: float = 0.05, cue_drop: Optional[Tuple[int, int]] = None,
              cues: Optional[np.ndarray] = None, ring_angles: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Run a source along a joint path (ticks, n) with the arm moved from outside.  ``cue_drop``: ticks [a, b) with no
    cue.  ``cues`` (ticks, 3, 3) and ``ring_angles`` (ticks, 3): the network's recorded output, replayed as the cue."""

    steps = int(round(source.delay_s / dt))
    source.reset(path[0])
    errors, pitches, chart, estimates = [], [], [], []
    for t in range(len(path)):
        seen_q = path[max(t - steps, 0)]
        seen_qdot = (path[max(t - steps, 0)] - path[max(t - steps - 1, 0)]) / dt
        qdot = (path[t] - path[max(t - 1, 0)]) / dt
        source.cue_on = not (cue_drop is not None and cue_drop[0] <= t < cue_drop[1])
        if cues is not None:
            source.replay = cues[t]
        if ring_angles is not None:
            source.replay_angles = ring_angles[t]
        _, estimate = source.observe(seen_q, wrap(seen_qdot), path[t], wrap(qdot), dt)
        rotation = chain.rotation(path[t])
        estimates.append(estimate)
        errors.append(float(np.degrees(rotation_angle(estimate, rotation))))
        pitches.append(float(np.degrees(lift_pitch_yaw(rotation)[1])))
        chart.append(float(getattr(source, "selected", 0)))
    return {"error_deg": np.array(errors), "pitch_deg": np.array(pitches), "chart": np.array(chart), "estimates": np.array(estimates)}


def smooth_path(chain: Chain, start: np.ndarray, goals: Sequence[np.ndarray], speed: float, dt: float = 0.05,
                hold_ticks: int = 10) -> np.ndarray:
    """A joint path that moves from goal to goal with every joint at up to ``speed`` rad/s (minimum-jerk in time)."""

    out = [np.asarray(start, dtype=float)]
    current = out[0]
    for goal in goals:
        difference = np.where(chain.circular_mask(), wrap(goal - current), goal - current)
        duration = max(float(np.max(np.abs(difference))) / float(speed) * 1.875, dt)      # peak speed of minimum jerk: 1.875 × mean
        ticks = int(np.ceil(duration / dt))
        for i in range(1, ticks + 1):
            s = i / ticks
            out.append(current + difference * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))
        current = out[-1]
        out.extend([current.copy() for _ in range(int(hold_ticks))])
    return np.array(out)


def pitch_sweep(chain: Chain, start: np.ndarray, rng: np.random.Generator, count: int = 6, tries: int = 50000) -> List[np.ndarray]:
    """Joint goals whose hand pitch alternates between near +90° and near −90° (through the chart pole)."""

    goals: List[np.ndarray] = []
    sign = 1.0
    for _ in range(int(tries)):
        if len(goals) >= int(count):
            break
        q = chain.random_pose(rng)
        if sign * np.degrees(lift_pitch_yaw(chain.rotation(q))[1]) > 80.0:
            goals.append(q)
            sign = -sign
    return goals


def summarise(values: Sequence[float]) -> Dict[str, Any]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return {"n": 0, "median": None, "p90": None}
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
