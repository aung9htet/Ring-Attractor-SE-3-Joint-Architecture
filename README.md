# Ring-Attractor-SE-3-Joint-Architecture

The pose of a robot arm's end point, position and orientation, read from ring attractors
that hold the joint angles. The orientation comes from the network of the SO-3 repo; this
repo adds the position, held as packets of activity on three attractor sheets in the manner
of RatSLAM's pose cells. Spiking neurons in NEST, 2 to 7 joints, the TIAGo arm among them,
with a simulator in which you move the arm and watch the decoded pose next to the true one.

Start here. Details: [`docs/architecture.md`](docs/architecture.md) (how it works),
[`docs/tests.md`](docs/tests.md) (what was tested and the pass marks),
[`docs/results.md`](docs/results.md) (what came out),
[`docs/future-ik.md`](docs/future-ik.md) (ideas for inverse kinematics; not built).
The mathematics and derivations of every part: [`TECHNICAL.md`](TECHNICAL.md).

Related repos: `Ring-Attractor-SO-2-Joint-Architecture` (one ring per joint, the joint
controller, the paper draft) and `Ring-Attractor-SO-3-Joint-Architecture` (orientation from
the rings). This repo runs on its own: it carries a copy of the SO-3 package (`ring_so3/`).

## 1. What this is and what was measured

![the pose network](docs/figures/architecture.png)

- **In:** one ring attractor per joint with a bump at the joint angle, and the joint
  velocities.
- **Orientation:** the SO(3) network, unchanged.
- **Position:** every link's orientation times its fixed offset, summed (no product), held
  by three *position cells*; these anchor one packet on each of three sheets of 48 x 48
  spiking neurons (planes xy, xz, yz). The position is read from the packets.
- **Velocity:** joint axis times lever times joint velocity gives the end point's velocity;
  *velocity cells* move the packets with it when the anchor is off.
- **Weights:** fitted once per kind of population to its measured response. No pose of any
  chain is used for fitting.

### What was tested, and the result

Position error is the distance between the decoded and the true end point, as a share of
the arm's reach (0.806 m for the 7-joint TIAGo arm), given as median / 90th percentile /
largest. Everything runs in NEST with a light robot (joint angles that move toward a target
at a speed limit). Full account: [`docs/results.md`](docs/results.md).

| Test | What was done | Result |
| --- | --- | --- |
| Maths | position as a weighted sum of link orientations, velocity from joint velocities, for 12 chains | equal to the exact kinematics |
| Parts | an anchored sheet; a free packet; a packet moved by a known velocity | an anchored packet sits within 0.56 cells of its place (mark 1 cell). A free packet moves at 0.66 to 1.71 times the speed asked for (mark: within 10%: **missed**) |
| **Held poses** | 300 random poses for each of 12 models (2 to 7 joints): move there, rest 1 s | position off by 0.97% to 1.37% of the reach in the median, largest 90th percentile 2.34%; the 7-joint TIAGo arm 11.1 mm / 18.6 mm / 32.8 mm. All 12 models within the mark set beforehand (median at most 2%, 90th percentile at most 4%). Orientation as in the SO-3 repo (largest median 1.12°) |
| Sheets against position cells | the same poses, position read before and after the sheets | the position cells alone are slightly closer (7 joints: 1.28% against 1.37%): the sheets hold the position, they do not sharpen it |
| **Moving arm** | the arm keeps going from goal to goal at 0.25, 0.5 and 1.0 rad/s | 0.21 to 0.23 s behind at 0.5 rad/s (mark 0.5 s); 5.25% to 7.40% of the reach off while moving. With the anchor given the position 0.2 s ahead: 2.28% (7 joints) |
| **Without the anchor, arm at rest** | anchor off for 2, 5, 10 s | in 29 of 30 gaps all three packets stay alive (in one, a sheet lost its packet); they move 4.59% of the reach in the median in 10 s (mark 2%: **missed**). Anchor on again: readout back in 0.10 s in the median, later than 0.5 s in 15 of 190 gaps (**missed** narrowly) |
| **Without the anchor, arm moving** | path integration alone for 2 to 10 s | **does not work well**: a clear gain in one condition (3 joints, 0.25 rad/s, 2 s: error 0.52 of the distance travelled against 1.21 without); little gain or harm in the others |
| Robustness | 100 poses, 4 and 7 joints | 20% of the sheets' neurons off: 1.93% (7 joints, as built 1.32%). Not robust to a noise current of 150 pA (6.31%) or to losing 5% of the SO(3) product neurons (7.21%) |

![position error against the number of joints](docs/results/figures/error_by_joints.png)

## 2. Setup

Everything runs in the Docker image of the SO-2 repo (NEST, Python 3.8, NumPy, SciPy,
Matplotlib, ffmpeg; pinned in `docker/image.env`). Nothing is installed on the host.

**Every command on this page can be pasted into a terminal at the root of this repo as it
stands.** Each one starts a fresh container (no network), runs, and removes the container.

- `docker/run.bash nest --out /tmp/ring-se3 -- '...'` runs what is between the quotes in
  the container. Results that a command writes to `/out/...` appear on the host in
  `/tmp/ring-se3/`.
- `docker/run.bash sim -- '...'` is the same with port 8765 published on localhost (the
  simulator).
- The repo is mounted read-write at `/work`. NEST uses `--threads` threads per run.

## 3. Run it

### 3.1 Calibration

Both calibration files of the default design are checked in (`artifacts/`), so nothing has
to be done. They are written by

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/calibrate.py --out /out/so3.npz'   # the SO(3) readouts, about a minute
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/calibrate_pose.py --out /out/pose.npz'   # anchor readout and packet speed curve, about two minutes
```

Without `--out` they write into `artifacts/` and replace the files there. Do that only for
a design that has none yet (`--sheet 32`, `--seed 2`): the recorded results belong to the
files that are checked in.

### 3.2 Held poses of one model

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/run_static.py --chain tiago4 --poses 20 --out /out/tiago4.json'
```

The arm moves to each random pose and rests 1 s; position and orientation read over the
last 0.5 s are compared with the exact kinematics. Models: `generic2` .. `generic7`,
`tiago2` .. `tiago7`. The summary is printed every 50 poses; every pose is in the JSON
file ([`docs/results/data/README.md`](docs/results/data/README.md) explains the fields).
Without `--poses` it runs the 300 poses of the test.

### 3.3 A moving arm

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/run_tracking.py --chain tiago4 --speed 0.5 --goals 5 --out /out/tracking.json'
```

The arm keeps moving from one random goal to the next; every 50 ms tick is logged.

### 3.4 Without the anchor

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/run_dropout.py --chain generic3 --speed 0 --gaps 5 --repeats 2 --out /out/rest.json'
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/run_dropout.py --chain generic3 --speed 0.25 --gaps 2 --repeats 2 --out /out/moving.json'
```

The anchor is switched off for the given number of seconds and on again. `--speed 0`: the
arm rests during the gap (does the packet stay?). Otherwise it keeps moving (does path
integration carry the packet?).

### 3.5 Options

Of `run_static.py`, `run_tracking.py` and `run_dropout.py`:

| Option | Effect |
| --- | --- |
| `--no-path-integration` | the velocity cells are silent: nothing moves the packets but the anchor |
| `--anchor-lead S` | the sheets are anchored at position + velocity x S seconds |
| `--no-velocity` | build no velocity path at all (a smaller network, anchored sheets only) |
| `--sheet N` | sheets of N x N neurons (needs its pose calibration) |
| `--design-seed N` | other thresholds and maximum rates in the populations (needs both calibrations) |
| `--noise PA` | a noise current into every neuron between the rings and the sheets |
| `--silence-sheets 0.1` | switch off this share of the sheets' neurons (`run_static.py`) |
| `--silence 0.05` | switch off this share of the SO(3) product neurons (`run_static.py`) |
| `--lead` | write each joint's angle ahead by the SO(3) readout's delay (`run_tracking.py`) |

### 3.6 The parts, the shift-register experiment, speed

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/run_parts.py --quick --out /out/parts.json'
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/run_trail.py --chain generic3 --mode trail --repeats 3 --out /out/trail.json'
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/rate_study.py --out /out/rate.json'
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/benchmark.py --models generic3 --threads 8 --seconds 10 --out /out/speed.json'
```

`run_parts.py` measures a sheet and the velocity path on their own (test 3; without
`--quick` the full set). `run_trail.py`: for 2 s the network gets neither the anchor nor the
joint velocities, and the velocity cells are given the velocity that the last 0.5 s of
decoded positions show (`--mode none` and `--mode velocity` are the two comparisons).
`rate_study.py` needs no NEST.

### 3.7 Everything, and the report

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/run_experiments.py --list'
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/run_experiments.py --jobs 3 --threads 8 --only static --models tiago7 generic7 generic6'
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/report.py'
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/check_doc_numbers.py'
```

`run_experiments.py --list` prints every run of the results; a run whose output exists is
skipped. All of it is several hours on 24 hardware threads; start it in parts (`--only`,
`--models`), as above. `report.py` writes [`docs/results/tables.md`](docs/results/tables.md)
and `docs/results/figures/`. The last command fails if a document quotes a number that is
not in the tables.

## 4. The simulator

```bash
docker/run.bash sim -- 'python3 -B scripts/simulator.py --chain tiago4'
```

Then open <http://localhost:8765/>. The spiking network runs live, one 50 ms tick at a
time, in step with a light robot (joint angles that move toward their targets at a speed
limit; no dynamics).

- **Sliders**: one per joint, within its limits. **Keep moving to random goals**: the arm
  goes from goal to goal by itself. **One random pose**, the **joint speed limit**, a
  **chain** selector.
- **Anchor** and **Path integration** switches. Anchor off, arm at rest: the packets stay.
  Anchor off, arm moving: the packets are carried by the velocity, roughly; with path
  integration off as well they stay where they were. Anchor on: they jump back.
- **The arm in 3D** with the true end point (dot) and the one read from the sheets
  (circle), and the true and decoded frames at every link; drag to turn the view. Below it
  the position error and the orientation error, now and over the last 10 s.
- **Position** x, y, z over the last 30 s, true and read from the sheets.
- **The three sheets**: spikes of the last tick, the true position (cross) and the centre
  of the packet (circle).
- **Lift, pitch, yaw** and the **spikes** of the joint rings and output rings, as in the
  SO-3 simulator.

The status line says how fast the network runs. A 2- or 3-joint network runs faster than
real time, the 7-joint one at less than half of it; the page then shows the arm in slow
motion. Options: `--port`, `--speed`, `--threads`, `--mode manual`, and `--stub`, which
replaces the network by exact kinematics with a little noise, to try the page where NEST is
not installed (`python3 scripts/simulator.py --stub` needs only NumPy). If your browser or
proxy does not pass event streams, open `http://localhost:8765/?poll`.

## 5. Record a session

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/record_session.py --chain tiago4 --seconds 20 --gap 8 5 --out /out/tiago4_gap'
```

Runs the goal mode without a browser and writes `tiago4_gap.mp4` (the arm with both end
points, the three sheets, position and position error over time), `.png` (the curves of the
whole session) and `.json` (every tick). `--gap START LENGTH` switches the anchor off for
LENGTH seconds, START seconds into the session. The recordings of the results page are in
`docs/results/videos/`.

## 6. Tests

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B -m unittest discover -s tests'   # about a minute
```

Covers the maths, the sheet, the copy without spikes, every new part in NEST, a 2-joint
pose end to end with both switches, the simulator's interface, the report, and that the
documents quote only numbers from the generated tables. What each of the ten tests of the
results asks, and its pass mark: [`docs/tests.md`](docs/tests.md).

## 7. From Python

```python
import nest
import numpy as np
from ring_se3 import experiments, kinematics

chain, design, network, session = experiments.build(nest, "tiago4", threads=8)   # built once, then kept
session.warm_up()                                       # 1 s: the rings and the packets form
measurement = session.go_to(chain.limits().mean(axis=1) + 0.3)       # move there, rest 1 s
print(measurement.position)                             # end point read from the sheets, m, base frame
print(kinematics.position(chain, session.robot.q))      # the exact one
tick = session.tick()                                   # one more 50 ms tick
print(tick.position, tick.cells_position)               # from the sheets (last two ticks), from the position cells
print(tick.tool_rotation())                             # orientation, as in the SO-3 repo
network.set_anchor(False)                               # the packets are on their own now
```

The same lines are in `scripts/example.py`; run them with

```bash
docker/run.bash nest -- 'python3 -B scripts/example.py'
```

`ring_se3/rate/network.py` is the position and velocity part as firing rates, with no NEST.

## 8. Add a chain

One entry in [`ring_so3/chains.py`](ring_so3/chains.py), as in the SO-3 repo: for each joint
its axis, the fixed rotation and offset of its frame in the previous link, its limits. The
offsets are what the position is made of. Everything else follows: the wiring, the size of
the workspace, the network, the simulator. `tests/test_kinematics.py` checks every chain in
`MODEL_NAMES`.

## 9. Status and limits

Date 2026-10-05, branch `fk-pose`.

- **Done:** position and velocity of the end point for any chain of turning joints, three
  attractor sheets with anchor and path integration, 12 models measured, simulator,
  recorded sessions, documents.
- **What works:** the position of a held pose (about 1% of the reach), the orientation
  beside it, a packet that stays alive without input, re-anchoring, following a moving arm
  with a delay that can be removed.
- **A packet can go out** when the anchor is off: seen on one sheet in 1 of 30 gaps with the
  arm at rest and in 31 of 160 with the arm moving. The other two sheets still give the
  position, and the packet returns with the anchor.
- **What does not work well:** path integration. A packet's speed for a given velocity
  varies by tens of percent from run to run, so the packets cannot be carried far without
  the anchor ([`docs/results.md`](docs/results.md), sections 2 and 8). What would have to
  change is a sheet whose packet has many more neurons or fires faster, so that a push is
  not a matter of a few spikes.
- **Three pass marks missed:** packet speed proportional to the velocity; a packet at rest
  within 2% of the reach for 10 s; readout back within 0.5 s in every gap
  ([`docs/results/tables.md`](docs/results/tables.md), section 12).
- **Kinematics only.** No Gazebo, no real robot, no dynamics.
- **Slow for long arms:** the 7-joint network runs at 0.44 of real time on 8 threads.
- **Inherited from the SO(3) network:** its delay of about 0.2 s, and its weakness to
  losing product neurons.
- **The resting drive of the velocity cells and the two switches are set from outside**
  ([`docs/architecture.md`](docs/architecture.md), section 8).
- **Next:** inverse kinematics toward a pose goal ([`docs/future-ik.md`](docs/future-ik.md)).

## 10. Where things are

Paths are relative to the repo root.

| Topic | Code | Document | Data |
| --- | --- | --- | --- |
| Chains, workspace, exact position and velocity | `ring_se3/chains.py`, `ring_se3/kinematics.py` | [`docs/architecture.md`](docs/architecture.md) section 2 | |
| Position and velocity as sums and products | `ring_se3/wiring.py` | section 2 | |
| The sheet: weights, anchor, labels, readout | `ring_se3/sheet.py`, `ring_se3/nest/sheet.py` | section 4 | |
| The pose network | `ring_se3/nest/network.py` | sections 3, 7 | |
| Calibration | `ring_se3/nest/calibrate.py`, `scripts/calibrate_pose.py`, `scripts/calibrate.py` | section 6 | `artifacts/` |
| Copy without spikes | `ring_se3/rate/network.py`, `scripts/rate_study.py` | [`docs/tests.md`](docs/tests.md) test 2 | `docs/results/data/rate_study.json` |
| Parts on their own | `ring_se3/nest/rigs.py`, `scripts/run_parts.py` | test 3 | `docs/results/data/parts.json` |
| Held poses | `ring_se3/session.py`, `ring_se3/experiments.py`, `scripts/run_static.py` | tests 4, 5, 6, 10 | `docs/results/data/static/` |
| Moving arm | `scripts/run_tracking.py` | test 7 | `docs/results/data/tracking/`, `tracking_variants/` |
| Without the anchor | `scripts/run_dropout.py` | test 8 | `docs/results/data/dropout/` |
| Shift-register experiment | `scripts/run_trail.py` | [`docs/architecture.md`](docs/architecture.md) section 5 | `docs/results/data/trail/` |
| Variants and robustness | `scripts/run_static.py` with options | test 9 | `docs/results/data/robust/` |
| All runs, speed | `scripts/run_experiments.py`, `scripts/benchmark.py` | | `docs/results/data/speed.json` |
| Tables and figures | `scripts/report.py`, `scripts/check_doc_numbers.py` | [`docs/results.md`](docs/results.md) | [`docs/results/tables.md`](docs/results/tables.md), `docs/results/figures/` |
| Simulator and recordings | `ring_se3/sim/`, `scripts/simulator.py`, `scripts/record_session.py` | this file, sections 4 and 5 | `docs/results/videos/` |
| Architecture figure | `scripts/draw_architecture.py` | | `docs/figures/architecture.png` |
| The SO(3) network (copied) | `ring_so3/` | [`docs/architecture.md`](docs/architecture.md) section 9 | |
| Ideas for inverse kinematics | | [`docs/future-ik.md`](docs/future-ik.md) | |
