# Kangaroo migration of the SE(3) pose network: design

Date 2026-10-27, branch `kangaroo-migration` (local, from `origin/ik-pose`, commit 3c99537).

## Intent

The lab has PAL's Kangaroo and no TIAGo. Everything this repo does with the TIAGo (decoded pose, position on
the sheets, closed-loop IK, a physical robot behind the loop) has to be possible on the Kangaroo. The SO-2 and
SO-3 repos have already made this move (`../Ring-Attractor-SO-2-Joint-Architecture` and
`../Ring-Attractor-SO-3-Joint-Architecture`, both on branch `kangaroo-migration`). This repo follows the
**same approach**, which the owner asked for: the network stays as it is, the robot side gains Kangaroo in
MuJoCo on ROS 2 Humble, both arms are measured, and the IK simulator page can drive the simulated arm. The
outcome is for a research publication, so every number and figure has to come from a recorded run.

The reference is SO-3's spec and plan:
`../Ring-Attractor-SO-3-Joint-Architecture/docs/superpowers/specs/2026-10-26-kangaroo-migration-design.md`
and `.../plans/2026-10-26-kangaroo-migration.md`. SO-3's Kangaroo work is staged in its working tree, not
committed. `docs/kangaroo/results/data/code_at_runs.patch` there is the code its runs used.

## What is already true (checked 2026-10-27)

- This repo's `ring_so3/` is a copy of SO-3 at 9e3ed24, the commit SO-3's migration started from. Every
  file is identical except `ring_so3/sim/page.py` and `ring_so3/sim/server.py`, which SO-3's migration does
  not touch. So SO-3's changes to `ring_so3/chains.py` and `ring_so3/robots.py`, and its new
  `ring_so3/kangaroo/`, apply to this repo unchanged.
- The image `ring-so3-kangaroo:humble` (ROS 2 Humble, MuJoCo, PAL `kangaroo-simulation`, NEST
  `HEAD@41892a5`, matplotlib) is built on this machine. Its recipe is SO-3's `docker/kangaroo/`.
- `artifacts/calibration_N200_f120_p200_e60_seed1_fractional.npz` and
  `artifacts/ik_calibration_N200_f120_p200_e60_seed1_fractional.npz` are byte-identical to SO-3's. SO-3
  has already re-run both in the image and recorded the result (`docs/experiments/2026-10-26_kangaroo_calibration_rerun.md`
  there: deterministic in the image, but different from `artifacts/` beyond 1e-4; the experiments load
  the checked-in files).
- PAL ships the Kangaroo ready to run with **4 joints per arm** (owner: "whatever PAL provides officially").

## Goals

1. **Carry over SO-3's Kangaroo code.** `ring_so3/chains.py` (`kangaroo_left4`, `kangaroo_right4`,
   `KANGAROO_*`, `_matrix_tuple` tolerance), `ring_so3/robots.py` (`GazeboRobot.chain_prefix`,
   `KangarooRobot`), `ring_so3/kangaroo/` (`Ros2Link`, lock step through MuJoCo's pause/step services),
   `docker/kangaroo/`, and the `kangaroo up|exec|down|run` mode in `docker/run.bash` and `docker/image.env`.
   Bring their tests too (`tests/test_kangaroo_chains.py`, `test_kangaroo_robot.py`, `test_kangaroo_sim.py`).
   They must pass here as they do in SO-3. Copy SO-3's **staged** tree and record the sha256 of every copied
   file in `docs/kangaroo/README.md`. The one allowed deviation: `test_kangaroo_robot.py`'s closed-loop test
   imports `StubIkNetwork` from `ring_so3.sim.server`, which this repo's copy lacks. Here it uses this repo's
   own stub network and IK session with the position task, the loop the Kangaroo runs here use, and
   `ring_so3/sim/server.py` stays untouched.
2. **Position for the Kangaroo chains** (`ring_se3/chains.py`). `kangaroo_<side>4` is the whole arm, so
   its end point is `arm_<side>_tool_link`, which is the chain's tool offset as it stands. The `tiago`
   shortening rule must not touch it. `workspace()` and `lever_lengths()` must give a workspace that
   holds every reachable tool position. Kangaroo chains are registered next to `MODEL_NAMES` the way SO-3
   does it (`KANGAROO_MODELS`); `MODEL_NAMES`, the 12 models of the existing tests, stays unchanged.
   Checked: the shortening rule already skips names that don't start with `tiago`, and the present
   workspace (centre at joint 1's origin, radius the summed link lengths, about 0.87 m) holds every tool
   position. So this goal is expected to need tests only, no new code.
3. **The pose robot on Kangaroo.** `ring_se3/gazebo/robot.py`'s `PoseGazeboRobot` exists for a TIAGo
   wrist that reports false velocities. The Kangaroo uses SO-3's `KangarooRobot`, unless a measured
   start-pose reset shows the same problem.
4. **Forward-kinematics check against the robot.** The `--check N` of `scripts/run_ik_gazebo.py`
   (Gazebo's end point against the chain's, at held poses) works for Kangaroo chains. On Kangaroo it reads
   **position and orientation** of `torso_link → arm_<side>_tool_link` from TF (robot_state_publisher in
   the sim) and writes the same JSON layout as `gazebo_kinematics_*.json`.
5. **Closed loop on MuJoCo.** `scripts/run_ik_gazebo.py` accepts `kangaroo_left4` and `kangaroo_right4`
   (task `position`, ring and `dls`), runs them on `KangarooRobot(chain, Ros2Link(side))`, and closes the link
   in `finally`.
6. **IK simulator.** `scripts/ik_simulator.py`, `ring_se3/sim/ik_server.py` and `ik_page.py` offer
   "Kangaroo in MuJoCo" (a `--kangaroo` flag) only for Kangaroo chains, as SO-3 did. The chain choice switches
   between the left and right arm. The forward simulator (`scripts/simulator.py`) lists the Kangaroo chains.
7. **Calibration re-run.** In the Kangaroo image, re-run this repo's own calibrations that SO-3 does not
   cover: `pose_N200_f120_p200_e60_seed1_fractional_s48_u1p5_k3` and
   `ramp_N200_f120_p200_e60_seed1_fractional_s48_u1p5_k3`. Write them to a new place and compare them with
   `artifacts/` per array. `calibrate_pose.py` and `calibrate_ramp.py` take their inputs from the
   checked-in `artifacts/`, so each stage is re-run on those (no new option). Cite SO-3's result for the two
   shared files. The comparison is the finding; the
   checked-in files stay untouched and the experiments load them.
8. **Measurements, both arms**, with the existing scripts:
   - held poses (`run_static.py`, 300 poses), light robot;
   - tracking (`run_tracking.py`) at 0.25, 0.5 and 1.0 rad/s, light robot;
   - anchor dropout (`run_dropout.py`) with the arm at rest and moving at 0.25 rad/s, light robot;
   - position-IK reaches (`run_ik_reach.py --task position`, 30 test pairs), light robot;
   - position-IK reaches on MuJoCo, ring and `dls`, the same 30 pairs;
   - the forward-kinematics check of goal 4 at 5 poses per arm.
9. **Figures and record**, as in SO-3. Data and figures (PNG and PDF) go under `docs/kangaroo/results/`. Each
   figure set gets a `manifest.json` listing inputs with sha256, commit, seeds and threads. Each experiment
   gets a note in `docs/experiments/2026-10-27_kangaroo_<name>.md` (hypothesis, config, seed, command,
   commit, results, interpretation, negative results). One `tables.md` holds every number the notes quote.
10. **Docs.** A Kangaroo section in `README.md` with the commands, beside the Gazebo section 11.
    `docs/kangaroo/README.md` gives what was migrated, the decisions, a results summary and the open items.

## Non-goals

- No change to the network, its wiring, populations or calibration code (`ring_so3/nest/*`, `ring_se3/nest/*`,
  `*/wiring.py`, `*/ik/wiring.py`, `ring_so3/neurons.py`, `ring_se3/sheet.py`).
- No pose IK on Kangaroo. Four joints cannot set position and orientation together, so the task is
  `position` only. Pose IK stays on `generic7` and `tiago7`.
- No 5- or 7-joint Kangaroo arm, no torso, legs or base motion.
- No re-run of the TIAGo or generic results. No variants (sheet sizes, noise, seeds 2-3, motor gain,
  pushes) on Kangaroo.
- No Gazebo camera panel for Kangaroo. MuJoCo shows the robot in its own window.
- No new UI framework.

## Anti-goals (a failure even if it runs)

- Kangaroo numbers typed in by hand rather than taken from SO-3's URDF-checked table, or SO-3 code
  "improved" on the way over so that the two repos drift apart.
- A result, figure or number that no recorded run produced, or a number tuned to look better without
  saying so.
- `artifacts/`, `docs/results/`, `docs/ik/results/` or any recorded data overwritten.
- The TIAGo/Gazebo path or any existing test broken.
- Helpers, options or abstractions that no goal above needs.

## Constraints

- **Where:** this local repo, branch `kangaroo-migration`. Nothing pushed. No commit unless the owner asks:
  changes are staged and the diff shown.
- **Image:** reuse `ring-so3-kangaroo:humble` as it is (`KANGAROO_IMAGE` in `docker/image.env`). Copy
  `docker/kangaroo/` so that the repo can rebuild it on its own. Add to the image only what this repo needs
  and lacks, after checking for it.
- **Python:** the image runs 3.10, the repo targets 3.8. Code must run on both.
- **Hardware:** RTX 4070 laptop, 22 cores, 30 GB RAM. `kangaroo up` needs a desktop session (DISPLAY) for
  MuJoCo's viewer.
- **Plant facts from SO-3** (`docs/experiments/2026-10-26_kangaroo_plant_checks.md` there):
  - lock step through `/mujoco_ros2_control_node/set_pause` and `step_simulation`, physics step 0.0005 s;
  - `/joint_states` arrives at 50 Hz, so staleness is recorded;
  - `KangarooRobot.tolerance` is 0.005 rad;
  - joint 2's upper limit is SO-2's measured value per side.
- **Seeds and threads** are recorded in every output. Results depend slightly on the thread count, so
  light and MuJoCo runs of the same experiment use the same thread count.
- **Code shape:** the style of `ring_se3/` (explanatory module docstrings, short functions, numpy).
  Minimal, with no options nobody uses. Copied SO-3 files stay as identical to SO-3 as possible.

## Process (owner's choice: same as SO-3)

- Skills: brainstorming (this), `mattpocock-skills:tdd`, ponytail for minimal code.
- Separate agents, per task:
  - **test writer:** writes the tests from the design and plan only, before any code (they must fail);
    covers behaviour with independent oracles, for the main functionality only;
  - **code writer:** gets the contract only, never the tests;
  - **semantic checker:** does the code do what this design says (frames, angle conventions, units,
    limits, the same robot calls as `GazeboRobot`)?
  - **authenticator:** is the code true to its sources (Kangaroo numbers equal SO-3's URDF table and the
    URDF in the image, topics and services exist in the running sim, every figure and number traces to a
    recorded run)?
- Tests that need NEST or the sim run in the container. Results are reported verbatim, including what
  could not run.

## Decided by the owner

- Kangaroo gets everything the TIAGo does in this repo, the same way SO-2 and SO-3 did it.
- PAL's official 4-joint arm.
- The SO-3 approach and process, including the four agents.

## My calls (not discussed; change them if wrong)

- **Kangaroo next to TIAGo, not instead of it.** This is what SO-3 did, and it is what "same approach"
  means. The TIAGo results stay as the record of what was measured.
- **Chain names** `kangaroo_left4` and `kangaroo_right4`, as SO-3. No shorter Kangaroo prefixes, which no
  experiment needs.
- **Pairs and poses:** 30 IK test pairs per arm on both robots (as SO-3 and the TIAGo Gazebo runs). The
  tracking and dropout runs use the scripts' default goal counts and gaps.
- **Dropout:** only the conditions the main results table uses (rest, and moving at 0.25 rad/s with path
  integration). The `no_pi` and 0.10/0.50 variants are left out.
- **Misses:** the pair set is not filtered. A trial that does not reach counts as a miss and is classified
  (at a joint limit / blocked / slow), as SO-3 does.
- **Report:** a small `scripts/kangaroo_report.py` that imports what `scripts/report.py` and
  `scripts/ik_report.py` already have, rather than more branches in those two.
- **Results** go under `docs/kangaroo/results/` and notes under `docs/experiments/`.
- **IK settings untouched.** The "reached" mark stays relative (0.04 of the workspace radius, about 3.5 cm on
  Kangaroo). Tables give it in mm as well. Both arms get nearly the same test pairs (the pair seed depends on
  joint count only); they differ only through joint 2's upper limit.
- **One copied test adapted** (goal 1), noted in `docs/kangaroo/README.md`.

## Left to the builder

- Whether `ring_se3/gazebo/` is extended or a sibling `ring_se3/kangaroo/` holds the TF-based check of
  goal 4, whichever is the smaller diff.
- Whether `PoseGazeboRobot`'s stillness test is needed on Kangaroo (goal 3). Decide from one measured reset.
- How the IK simulator's robot ↔ chain table generalises `startswith("tiago")`. Prefer SO-3's
  `ROBOT_CHAINS` pattern.
- Run order and sharding of the long runs (the MuJoCo reaches take hours). Each run must be resumable
  (`--only`, `--max-minutes`).
