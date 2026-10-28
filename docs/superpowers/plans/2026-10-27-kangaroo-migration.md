# Kangaroo Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the SE(3) pose network PAL's Kangaroo (4-DoF arms, MuJoCo, ROS 2 Humble) everywhere it has the TIAGo. That covers the decoded pose, position IK on the light robot and on MuJoCo, and the IK simulator page. Both arms are measured.

**Architecture:** The network is untouched. SO-3's staged Kangaroo code (chains, `KangarooRobot`, `Ros2Link`, the docker `kangaroo` mode) is copied into this repo's `ring_so3/` copy. SE-3 adds:
- the Kangaroo chains to its registry;
- a TF reading of the tool pose for the kinematics check;
- Kangaroo as a robot in `run_ik_gazebo.py` and the IK simulator.

Measurements reuse the existing scripts, and one small report script turns their JSON into figures and tables.

**Tech Stack:** Python 3.10 in the image, code that also runs on 3.8. NumPy, NEST `HEAD@41892a5`, rclpy, tf2_ros, `mujoco_ros2_control`, Matplotlib, unittest, Docker image `ring-so3-kangaroo:humble`.

**Spec:** `docs/superpowers/specs/2026-10-27-kangaroo-migration-design.md`. Reference: SO-3's plan, `../Ring-Attractor-SO-3-Joint-Architecture/docs/superpowers/plans/2026-10-26-kangaroo-migration.md`.

## Global Constraints

- Branch `kangaroo-migration`. Never push. Commit only when the owner asks. Every "commit" step below is **stage + `git diff --cached --stat` + show the diff**, and the owner commits.
- `SO3=../Ring-Attractor-SO-3-Joint-Architecture`. "Copy from SO-3" means the **staged** version: `git -C $SO3 show :<path> > <path>`. Record the sha256 of every copied file. They go in `docs/kangaroo/README.md` (Task 8).
- No change to `ring_so3/nest/*`, `ring_se3/nest/*`, `*/wiring.py`, `*/ik/wiring.py`, `ring_so3/neurons.py`, `ring_se3/sheet.py`, `ring_so3/sim/server.py`.
- Never write into `artifacts/`, `docs/results/` or `docs/ik/results/`. New data goes under `docs/kangaroo/results/data/`, and notes go in `docs/experiments/2026-10-27_kangaroo_<name>.md`.
- `MODEL_NAMES` (12 models) stays unchanged. Kangaroo chains are `KANGAROO_MODELS = ["kangaroo_left4", "kangaroo_right4"]`.
- Kangaroo task is `position` only.
- Every result file records seed and threads. Light-robot and MuJoCo runs of the same experiment use `--threads 8`.
- Tests use unittest: `docker/run.bash kangaroo run -- 'python3 -B -m unittest <module> -v'`. Results are reported verbatim, including skips.
- Style: explanatory module docstrings, short functions, numpy. No options nobody uses (ponytail).

## Agents (owner's process, as SO-3)

For each task:
1. The **test writer** gets the task's *Interfaces* and test step only. They write the tests first, and the tests must fail.
2. The **code writer** gets *Interfaces* and the code step only, never the tests. They implement.
3. The orchestrator runs the tests.
4. The **semantic checker** and the **authenticator** review the diff:
   - semantic checker: frames, angle conventions, units, limits, the same robot calls as `GazeboRobot`, spec goals;
   - authenticator: copied files equal SO-3's staged files, numbers equal the URDF in the image, topics and services exist in the running sim, every reported number traces to a file.

Failures go back to the code writer as behaviour, never as test code. Copy tasks (Task 1) have no test writer for the copied tests; the authenticator checks the copy.

## Review Focus

1. **No simulation running** when a Kangaroo script starts: a clear `RuntimeError` within about 30 s, not a hang. SO-3's `test_no_sim_fails_fast` is copied in Task 1.
2. **Run interrupted** (Ctrl-C or an exception): MuJoCo must be left unpaused, so `close()` runs in `finally` in both scripts. Tested in Task 4 (`test_check_closes_the_link_on_error`) and by SO-3's `test_close_unpauses`.
3. **TF not there or stale** for a pose in the kinematics check: that row gets `None` errors and the run goes on. Tested in Task 4 (`test_a_pose_without_tf_is_recorded_not_fatal`).
4. **Pose task asked for a 4-joint Kangaroo chain** (`--task pose`): refused before any ROS or NEST import. Tested in Task 4 (`test_pose_task_is_refused_for_kangaroo`).
5. **Wrong arm or wrong robot**: a left chain on the right link, a TIAGo chain on Kangaroo, or Kangaroo offered for a TIAGo chain in the simulator. These are refused or ignored by SO-3's `test_wrong_side_is_refused` (Task 1) and `test_kangaroo_is_offered_only_for_its_chains` (Task 5).

---

### Task 1: Carry over SO-3's Kangaroo base

**Files:**
- Copy from SO-3 (staged):
  - `ring_so3/chains.py`
  - `ring_so3/robots.py`
  - `ring_so3/kangaroo/__init__.py`
  - `ring_so3/kangaroo/link.py`
  - `docker/kangaroo/Dockerfile`
  - `docker/kangaroo/kangaroo_sim.launch.py`
  - `docker/image.env`
  - `tests/test_kangaroo_chains.py`
  - `tests/test_kangaroo_sim.py`
- Modify `docker/run.bash`: SO-3's staged version with this repo's one difference kept, `mktemp /tmp/ring-se3-xauth.XXXXXX`.
- Adapt `tests/test_kangaroo_robot.py`: SO-3's copy with only `test_the_closed_loop_runs_on_it` changed (step 1).

Before copying, confirm that this repo's `ring_so3/chains.py`, `ring_so3/robots.py`, `docker/image.env` and `docker/run.bash` equal `git -C $SO3 show 9e3ed24:<path>`, apart from the xauth line. If they don't, stop and report.

**Interfaces:**
- Produces (exactly as SO-3):
  - `ring_so3.chains`: `KANGAROO_MODELS`, `KANGAROO_ARM`, `KANGAROO_MOUNT`, `KANGAROO_TOOL`, `KANGAROO_BASE_LINK = "torso_link"`, `KANGAROO_TOOL_LINK = "arm_%s_tool_link"`, `kangaroo_chain(side)`; `get_chain("kangaroo_left4")` works.
  - `ring_so3.robots.KangarooRobot(chain, link)`.
  - `ring_so3.kangaroo.link.Ros2Link(side, timeout_s=30.0)`, with `joint_state()`, `sim_time_s()`, `send()`, `step()`, `close()`, `side`, `PHYSICS_STEP_S = 0.0005`, `stale_max_s` and `node`.
  - `docker/run.bash kangaroo up|exec|down|run`.

- [ ] **Step 1: Copy the three test files.** In `tests/test_kangaroo_robot.py`, replace the SO-3 imports `IkDesign`, `IkSession`, `StubIkNetwork` and `lift_pitch_yaw` with this repo's, and rewrite the last test:

```python
from ring_se3 import kinematics
from ring_se3.ik.design import PoseIkDesign
from ring_se3.ik.session import PoseIkSession
from ring_se3.sim.ik_server import PoseIkStubNetwork

    def test_the_closed_loop_runs_on_it(self):
        design = PoseIkDesign(task="position")
        session = PoseIkSession(PoseIkStubNetwork(self.chain, design), robot=self.robot)
        session.warm_up(5)
        goal_q = self.chain.limits().mean(axis=1) + 0.25
        trial = session.reach(lift_pitch_yaw(self.chain.rotation(goal_q)), kinematics.position(self.chain, goal_q))
        self.assertTrue(trial.reached(design))
        self.assertLess(trial.final_error_reach, 0.04)        # share of the workspace radius, the IK's own mark
```

  (`lift_pitch_yaw` stays imported from `ring_so3.kinematics`.)
- [ ] **Step 2: Run** `docker run --rm -v "$PWD:/work" -w /work ring-so3-kangaroo:humble bash -c 'source /opt/ros/humble/setup.bash; PYTHONPATH=/work python3 -B -m unittest tests.test_kangaroo_chains tests.test_kangaroo_robot -v'`. Expected: FAIL on `ImportError` or `unknown chain 'kangaroo_left4'`.
- [ ] **Step 3: Copy the code and docker files** listed above. Apply the one xauth-name change in `run.bash`.
- [ ] **Step 4: Run** `docker/run.bash kangaroo run -- 'python3 -B -m unittest tests.test_kangaroo_chains tests.test_kangaroo_robot -v'`. Expected: PASS. Then run the whole suite: `docker/run.bash kangaroo run -- 'python3 -B -m unittest discover -s tests'`. Every test that passed before still passes; TIAGo URDF tests skip in this image. `test_no_sim_fails_fast` passes in `kangaroo run`.
- [ ] **Step 5: Live check.** Run `docker/run.bash kangaroo up k1`, then `docker/run.bash kangaroo exec k1 -- 'python3 -B -m unittest tests.test_kangaroo_sim -v'`, which should PASS (live tests), then `kangaroo down k1`.
- [ ] **Step 6: Authenticator.** `sha256sum` of each copied file equals SO-3's staged blob (`git -C $SO3 show :<path> | sha256sum`). The only differences allowed are the xauth line and the closed-loop test. Then stage and show the diff.

### Task 2: Kangaroo chains in the SE(3) registry

**Files:**
- Modify: `ring_se3/chains.py` (import and export `KANGAROO_MODELS`; `registry()` also returns the Kangaroo chains; module docstring adds one line on them).
- Test: `tests/test_kangaroo_position.py`.

**Interfaces:**
- Consumes: Task 1's `ring_so3.chains` Kangaroo names.
- Produces: `ring_se3.chains.KANGAROO_MODELS` and `ring_se3.chains.registry()` containing `kangaroo_left4` and `kangaroo_right4`. `get_chain("kangaroo_<s>4")` keeps SO-3's tool offset.

- [ ] **Step 1: Write the failing tests.** This module is independent of `ring_so3.kinematics`. Compose 4×4 transforms from the URDF in the image as `tests/test_kangaroo_chains.py` does; import its helper if it is a plain function, else repeat the small composition. Skip when the URDF is absent.
  - `test_end_point_is_the_tool_link_of_the_urdf`: 25 random poses inside the limits, both sides, seed 20261027. `ring_se3.kinematics.position(chain, q)` equals the URDF translation `torso_link → arm_<s>_tool_link`, `atol=1e-5`.
  - `test_workspace_holds_every_reachable_tool_position`: 5000 random poses per side. `np.abs(workspace(chain).scaled(position)).max() <= 1.0`, and `np.linalg.norm(scaled) <= 1.0` for each.
  - `test_registry_has_the_kangaroo_chains_and_model_names_is_unchanged`: `{"kangaroo_left4", "kangaroo_right4"} <= set(registry())`, `len(MODEL_NAMES) == 12`, and no `MODEL_NAMES` entry starts with `kangaroo`.
- [ ] **Step 2: Run** `docker/run.bash kangaroo run -- 'python3 -B -m unittest tests.test_kangaroo_position -v'`. Expected: the registry test FAILS. The other two may already pass (spec goal 2: "tests only" is expected). Report which.
- [ ] **Step 3: Implement.** `registry()` returns `{name: get_chain(name) for name in MODEL_NAMES + KANGAROO_MODELS}`. Make no other change unless a test fails.
- [ ] **Step 4: Run** the module, then the full suite. Expected: PASS, and no regressions.
- [ ] **Step 5:** Semantic check (end point = tool link, workspace centre = joint 1 origin = mount xyz), then stage and show the diff.

### Task 3: Calibration re-run (experiment)

- [ ] **Step 1: Run** `docker/run.bash kangaroo run --out docs/kangaroo/results/data/calibration -- 'python3 -B scripts/calibrate_pose.py --out /out/pose_N200_f120_p200_e60_seed1_fractional_s48_u1p5_k3.npz && python3 -B scripts/calibrate_ramp.py --out /out/ramp_N200_f120_p200_e60_seed1_fractional_s48_u1p5_k3.npz'`. Both read the checked-in `artifacts/` as their inputs.
- [ ] **Step 2: Compare** per array with `artifacts/`: shape, largest absolute and relative difference, identical? The script is `docs/kangaroo/results/data/calibration/compare.py`, adapted from SO-3's file of the same name, and writes `comparison.json` there.
- [ ] **Step 3: Write the note** `docs/experiments/2026-10-27_kangaroo_calibration_rerun.md`:
  - hypothesis, command, commit, seed, threads, result;
  - SO-3's result for the two shared files, cited by path (they are byte-identical here, sha256 `33b2cdd5…` and `4820071d…`).

  If either array differs beyond 1e-4 relative, report it to the owner and go on: the experiments load the checked-in files, as in SO-3. Then stage.

### Task 4: Kangaroo behind `run_ik_gazebo.py` (closed loop and kinematics check)

**Files:**
- Create `ring_se3/kangaroo/__init__.py`: a one-line docstring, as `ring_se3/gazebo/__init__.py`.
- Create `ring_se3/kangaroo/link.py`.
- Modify `ring_se3/gazebo/link.py`, function `kinematics_check`.
- Modify `scripts/run_ik_gazebo.py`.
- Test: `tests/test_kangaroo_check.py`.

**Interfaces:**
- Consumes: `Ros2Link` (Task 1) and its `node`.
- Produces:
  - `class PoseRos2Link(Ros2Link)` in `ring_se3/kangaroo/link.py`. It adds `tool_pose() -> Optional[Tuple[np.ndarray, np.ndarray]]`: the position (m) and the 3×3 rotation of `arm_<side>_tool_link` in `torso_link`, read with `tf2_ros.Buffer` and a `TransformListener` on the link's node. It waits up to 2 s for a transform stamped at or after the newest joint state and returns `None` otherwise. The rotation comes from `ring_se3.gazebo.link.quaternion_matrix`.
  - `kinematics_check(link, robot, chain, poses, log=print, end_point=end_point, base_link=TIAGO_BASE_LINK, end_link=None)`. `end_point(link, chain)` gives the seen pose, and `end_link` defaults to `arm_link(chain.n)`. For TIAGo the output is byte-for-byte unchanged.
  - `run_ik_gazebo.py` accepts `--chain kangaroo_left4|kangaroo_right4`. A Kangaroo chain uses `link = PoseRos2Link(side)` and `KangarooRobot(chain, link)`. `--check` passes `end_point=lambda link, chain: link.tool_pose()`, `base_link=KANGAROO_BASE_LINK`, `end_link=KANGAROO_TOOL_LINK % side`. `link.close()` runs in `finally` for both the check and the reaches. A Kangaroo chain with `--task pose` gets `parser.error(...)` before any ROS or NEST import. Printed lines say "MuJoCo" for Kangaroo, and the docstring gets the Kangaroo commands, as SO-3's.

- [ ] **Step 1: Write the failing tests** in `tests/test_kangaroo_check.py` (host, no ROS; a stand-in link with `tool_pose`, `close` and a call log):
  - `test_check_compares_the_tf_tool_pose_with_the_chain`: the stand-in robot holds the poses, and the stand-in `tool_pose` returns the exact chain pose plus 1 mm in x. Every row has `assertAlmostEqual(row["position_error_m"], 0.001, places=9)` and `orientation_error_deg < 1e-9`; `base_link == "torso_link"`, `end_link == "arm_left_tool_link"`.
  - `test_a_pose_without_tf_is_recorded_not_fatal`: `tool_pose` returns `None` for pose 1 of 3. Row 1 has `position_error_m is None`, rows 0 and 2 are numbers, and `largest_position_error_m` comes from those two.
  - `test_pose_task_is_refused_for_kangaroo`: `run_ik_gazebo.main(["--chain", "kangaroo_left4", "--task", "pose", "--out", tmp])` raises `SystemExit` with a nonzero code, and `sys.modules` gains neither `rclpy` nor `nest` from the call.
  - `test_check_closes_the_link_on_error`: patch `ring_se3.kangaroo.link.PoseRos2Link` with a stand-in whose robot `reset` raises `RuntimeError`. `main(["--chain", "kangaroo_left4", "--check", "2", "--out", tmp])` raises, and the stand-in's `close` was called once.
  - `test_tiago_check_output_is_unchanged`: with a stand-in `link_pose` link on `tiago4`, the new `kinematics_check` returns the same dict as the code at `origin/ik-pose`. Inline the expected dict for two fixed poses, computed once from the old function before the change.
- [ ] **Step 2: Run** `docker/run.bash kangaroo run -- 'python3 -B -m unittest tests.test_kangaroo_check -v'`. Expected: FAIL (no `ring_se3.kangaroo`).
- [ ] **Step 3: Implement** the Interfaces.
- [ ] **Step 4: Run** the module and the full suite. Expected: PASS.
- [ ] **Step 5: Live check.** Run `kangaroo up k1`, then `kangaroo exec k1 -- 'python3 -B scripts/run_ik_gazebo.py --chain kangaroo_left4 --check 2 --out /tmp/c.json'`. Expected: 2 rows, position error < 1 mm, orientation < 0.1°. Then `... --pairs 2 --threads 8 --out /tmp/r.json`, which finishes 2 trials, and the JSON has `robot.joint_state_stale_max_s`.

  **Goal 3:** in the same session, reset to 3 start poses and log the ticks each reset took. If any reset uses the full 2,000 ticks, report it to the owner before Task 7. Don't add `PoseGazeboRobot`'s stillness test on your own.

  Then run `kangaroo down k1`.
- [ ] **Step 6:** Semantic check (frames `torso_link`, metres, degrees, `finally` placement), then the authenticator (TF frame names exist in the sim: `ros2 run tf2_ros tf2_echo torso_link arm_left_tool_link`). Stage and show the diff.

### Task 5: IK simulator and forward simulator offer Kangaroo

**Files:**
- Modify:
  - `ring_se3/sim/ik_server.py`: SO-3's `ROBOT_CHAINS` and `_usable` pattern, chains `MODEL_NAMES + KANGAROO_MODELS`, `robot_chains` in place of `gazebo_chains`, snapshot block for any non-light robot with `getattr(link, "overshoot_s", 0.0)`;
  - `ring_se3/sim/ik_page.py`: robot labels and an option per `robot_chains`; for `kangaroo` the note reads "MuJoCo shows the robot in its own window", with no picture;
  - `scripts/ik_simulator.py`: `--kangaroo`, `--robot` choice `kangaroo`, one `Ros2Link` per side on demand, every link closed in `finally`; SO-3's diff is the model;
  - `ring_se3/sim/server.py` and `scripts/simulator.py`: chain list `MODEL_NAMES + KANGAROO_MODELS`.
- Test: `tests/test_ik_sim.py` (one new test), `tests/test_sim.py` (one new assertion).

**Interfaces:**
- Consumes: Task 1 (`KangarooRobot`, `Ros2Link`) and Task 2 (`registry()` with the Kangaroo chains).
- Produces: `ik_server.ROBOT_CHAINS = {"gazebo": "tiago", "kangaroo": "kangaroo_"}`; config key `robot_chains: {robot: [chain names]}` for the offered non-light robots; `ROBOT_LABELS["kangaroo"] = "Kangaroo in MuJoCo"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_kangaroo_is_offered_only_for_its_chains` (`test_ik_sim.py`): `PoseIkSimulator(make, "kangaroo_left4", realtime=False, make_robot=<stand-in KangarooRobot on FakeKangarooLink>, robots=("light", "kangaroo"), robot="kangaroo")`.
    - Config `robot_chains["kangaroo"] == ["kangaroo_left4", "kangaroo_right4"]`.
    - A position goal 0.3 rad away is reached with snapshot `robot == "kangaroo"`.
    - After `{"command": "chain", "chain": "tiago4"}` the robot is `light`, and `{"command": "robot", "robot": "kangaroo"}` is ignored.
  - In `test_sim.py`'s config test: `"kangaroo_left4" in config["chains"]`.
  - The existing `test_gazebo_is_offered_only_when_it_is_there` changes only where it read `gazebo_chains`.
- [ ] **Step 2: Run** `docker/run.bash kangaroo run -- 'python3 -B -m unittest tests.test_ik_sim tests.test_sim -v'`. Expected: the new test FAILS.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Manual check.**
  1. `kangaroo up k1`, then `kangaroo exec k1 -- 'python3 -B scripts/ik_simulator.py --chain kangaroo_left4 --kangaroo --threads 8'`.
  2. Open `http://localhost:8765/` and press "Random goal nearby": MuJoCo's left arm moves and the goal is reached.
  3. Switch the chain to `kangaroo_right4`: the right arm moves.
  4. Ctrl-C: the MuJoCo clock runs again.

  Write what was seen in `docs/experiments/2026-10-27_kangaroo_plant_checks.md`, together with Task 4's reset ticks. Then semantic check, stage and show the diff.

### Task 6: Light-robot measurements (experiment, long; run in the background)

All per side `s ∈ {left, right}`, in `docker/run.bash kangaroo run --out docs/kangaroo/results/data -- '<cmd>'`, `--threads 8`. Commands go in `docs/kangaroo/results/data/logs/light_<s>.sh`, and output goes to `light_<s>.log`.

- [ ] `python3 -B scripts/run_static.py --chain kangaroo_<s>4 --threads 8 --out /out/static/kangaroo_<s>4.json`
- [ ] `python3 -B scripts/run_tracking.py --chain kangaroo_<s>4 --speed {0.25,0.5,1.0} --threads 8 --out /out/tracking/kangaroo_<s>4_<speed>.json`
- [ ] `python3 -B scripts/run_dropout.py --chain kangaroo_<s>4 --speed 0 --threads 8 --out /out/dropout/kangaroo_<s>4_rest.json`, and `--speed 0.25 ... _move025.json`
- [ ] `python3 -B scripts/run_ik_reach.py --chain kangaroo_<s>4 --task position --pairs 30 --threads 8 --out /out/reach/light_ring_<s>.json`
- [ ] A run cut short is finished with `--only` or `--max-minutes` into a second file, and both files are named in the note. Check that every JSON records seed and threads.

### Task 7: MuJoCo measurements (experiment, hours)

A fresh sim per arm: `kangaroo up km_<s>`, then `kangaroo exec km_<s> -- '<cmd>'` (log in `logs/mujoco_<s>.{sh,log}`), then `down`.

- [ ] `python3 -B scripts/run_ik_gazebo.py --chain kangaroo_<s>4 --check 5 --out docs/kangaroo/results/data/check/kangaroo_<s>4.json`
- [ ] `python3 -B scripts/run_ik_gazebo.py --chain kangaroo_<s>4 --pairs 30 --method ring --max-minutes 1000 --threads 8 --out docs/kangaroo/results/data/reach/mujoco_ring_<s>.json`
- [ ] The same with `--method dls --out .../reach/mujoco_dls_<s>.json`.
- [ ] Write the note `docs/experiments/2026-10-27_kangaroo_measurements.md`:
  - hypothesis: held-pose position error median ≤ 2% and p90 ≤ 4% of the reach, the marks set for the other chains; MuJoCo reaches close to light-robot reaches;
  - every command, the commit and the image digest;
  - results quoted from `tables.md` (Task 8);
  - misses classified as at a limit, blocked or slow;
  - negative results.

### Task 8: Figures, tables, docs

**Files:**
- Create: `scripts/kangaroo_report.py`, `docs/kangaroo/README.md`.
- Modify: `README.md` (a Kangaroo section beside section 11, a line in the repo map).
- Test: `tests/test_kangaroo_report.py`.

**Interfaces:**
- Consumes: the JSON layouts written by `run_static`, `run_tracking`, `run_dropout`, `run_ik_reach` and `run_ik_gazebo` (Tasks 4, 6, 7), and `comparison.json` (Task 3).
- Produces: `python3 -B scripts/kangaroo_report.py [--data docs/kangaroo/results/data] [--out docs/kangaroo/results]`. It reuses `scripts/report.py` and `scripts/ik_report.py` readers and summaries by import, and adds only what they lack. A missing input is listed in the manifest, not treated as an error. It writes:
  - `figures/static_error.{png,pdf}`: position error (% of reach and mm) and orientation error per arm, with the 2% and 4% marks;
  - `figures/tracking.{png,pdf}`: delay and error at the three speeds per arm;
  - `figures/dropout.{png,pdf}`: packet drift over the gap, rest and moving, per arm;
  - `figures/reach_error_over_time.{png,pdf}`: distance to goal (mm) over time, median and IQR, for light-ring, MuJoCo-ring and MuJoCo-dls, per arm;
  - `tables.md`: every number the notes quote. That includes the reached mark in mm (0.04 × workspace radius), the kinematics-check maxima and the calibration comparison;
  - `figures/manifest.json`: inputs (path, size, sha256), commit, date, and the seeds and threads read from the inputs.

- [ ] **Step 1: Write the failing test** `test_report_from_minimal_data`. Synthetic files go in a temp dir: two static rows per arm with known position errors, one reach row, and no dropout file. Expected:
  - the figure files and `tables.md` exist;
  - `tables.md` contains the hand-computed medians;
  - the manifest lists each given input with its sha256 and lists the dropout files as missing.
- [ ] **Step 2: Run** it: FAIL. **Step 3:** Implement. **Step 4:** PASS, then run the script on the real data.
- [ ] **Step 5: Authenticator** recomputes every number in `tables.md` from the JSON. `docs/kangaroo/README.md` covers:
  - what was migrated, with the sha256 of the copied SO-3 files (Task 1);
  - the adapted test;
  - the spec's "my calls";
  - a results summary pointing at `tables.md` and the three notes;
  - open items.

  README section: the commands of Tasks 4 to 7, in the manner of section 11.2.
- [ ] **Step 6: Whole-branch review.** Semantic checker and authenticator on the full diff, plus `ponytail-review` for over-engineering. Final full test run (`kangaroo run` plus the live `test_kangaroo_sim`), reported verbatim. Stage everything and show `git diff --cached --stat` for the owner to commit.
