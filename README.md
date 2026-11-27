# Ring-attractor kinematics: pose reaching in spiking neurons, and the baselines it is compared with

This repository holds a spiking neural network that takes a robot arm's hand to a goal **pose** (position and
orientation together), the controllers it is compared with, the shared benchmark they are all measured on, and every
result. Everything runs in Docker; every result in `results/` can be produced again with the commands below.

**The models**, one package each:

| Package | Model | What it is |
|---|---|---|
| `ring_attractor_kinematics/` | **RAK model** (ring-attractor kinematics) | The spiking network (NEST, leaky integrate-and-fire neurons). Joint angles live as bumps on ring attractors; hand position as packets on three 2-D attractor sheets; populations compose the hand's pose from the joints and compare it with the goal; a *cascade*, inspired by cyclic coordinate descent, hands the remaining error down the arm joint by joint (below). Variants: the default (the turn is read from the hand's late decoded matrix), *predictive encoding* (joint angles written ahead by the commanded velocity) and *observer* (a two-chart orientation observer). |
| `dls_ik/` | **DLS IK** | Standard inverse kinematics by damped least squares: every tick, the smallest joint step that makes the remaining turn and move, from the measured joint angles and the true geometry. Reactive, exact, no neurons, no internal model. |
| `ccd_ik/` | **CCD IK** (Wang & Chen 1991; Kenwright 2012) | Standard inverse kinematics by cyclic coordinate descent: every tick, from the measured joint angles, each joint in turn from the wrist to the base is turned by the closed-form angle that brings the hand closest to the goal, with the hand's pose recomputed after every joint, until the hand is at the goal; the arm then steps toward that solution. Reactive, exact, no neurons, no internal model. |
| `mpc_nn_dynamics/` | **NN dynamics + MPC** (Nagabandi et al., ICRA 2018) | A neural-network model of how the arm responds to joint steps, learned from data, with model-predictive control by random shooting. |
| `pets/` | **PETS** (Chua et al., NeurIPS 2018) | An ensemble of probabilistic neural-network models of the arm's response, learned from data, with model-predictive control by the cross-entropy method over trajectories sampled from the ensemble. |
| `jacobian_transpose_ik/` | Jacobian transpose | The gradient of the task error; used only as an exact-loop reference beside the RAK model's trials and in the simulator (no benchmark of its own). |

Shared by all: `reaching_benchmark/` (the task, the arms, the plants, the closed loop, the runner, the results files,
statistics, progress bars, status and comparison) and, for the two learned models, `learned_dynamics/`.

**What "cascade" means, and how it relates to CCD.** The RAK model decides how far each joint moves with a *cascade*:
every 50 ms it goes through the joints from base to wrist; joint 1 takes the part of the remaining hand error (turn and
offset) that its own axis can make, capped at its step limit; that part is subtracted from the error; joint 2 takes its
share of what joint 1 left; and so on to the wrist. Spiking populations compute this. The idea comes from cyclic
coordinate descent (CCD), the classical inverse-kinematics method that also moves one joint at a time, each correcting
what the others left. The cascade is a linearised, single-sweep form of it run continuously in closed loop: each joint's
share comes from its first-order effect (axis and lever) rather than an exact rotation with the pose recomputed, one sweep
is made per tick rather than sweeps to convergence, every joint moves at once while the loop keeps running, and the sweep
goes from base to wrist. Standard CCD IK (`ccd_ik/`) is therefore one of the baselines. The exact cascade without neurons
(`ring_attractor_kinematics/rules/cascade_rule.py`) is kept as the RAK model's exact-loop twin of every trial, not as a
benchmark of its own.

## Contents

1. [Layout](#layout)
2. [Install](#install)
3. [Tests](#tests)
4. [The reaching task](#the-reaching-task) (what every controller is measured on)
5. [Results: how each one was produced](#results-how-each-one-was-produced)
6. [Running the baselines (to fill Fig. 3b)](#running-the-baselines)
7. [The learned baselines: protocol and settings](#the-learned-baselines-protocol-and-settings)
8. [What each controller reads, is given and learns](#what-each-controller-reads-is-given-and-learns)
9. [Gazebo](#gazebo)
10. [Result files](#result-files)
11. [Watching runs](#watching-runs)
12. [The simulator](#the-simulator)
13. [Code conventions](#code-conventions)
14. [Licences and references](#licences-and-references)

## Layout

```
reaching_benchmark/          shared by every model; never imports a model
  geometry/                  rotations, arms (chains), forward kinematics                        NumPy only
  task.py                    IkDesign: the task, goal travel, step limits, when a trial ends and counts as reached
  targets.py                 start poses and goals, and their seeds (TEST_SEED: the reported pairs; TRAIN_SEED: training)
  limits.py                  step caps, limit room, goal travel, the turn and offset errors, the weighted Jacobian
  plants/light_robot.py      the kinematic plant (joints move exactly as told)
  plants/gazebo_robot.py     the TIAGo arm in Gazebo, in lock step                               (ROS)
  closed_loop.py             the closed loop every controller runs in; Trial; trial_row; RuleLoop
  benchmark.py               run_reach: the shared runner (resume, checkpoint, progress bar)
  summary.py, compare.py     the shared reaching summary; every model side by side
  results.py                 where results live and how run files are named
  statistics.py              Wilson intervals, McNemar, paired bootstrap, censoring
  progress.py, status.py     progress bars; python -m reaching_benchmark status
ring_attractor_kinematics/   the RAK model
  config.py                  every design constant of the network (sizes, sheets, gain pairs, observer)
  neurons/, spiking/         the LIF neuron, tuning, ring and sheet layout; populations, rings, sheets in NEST
  network/                   wiring (no NEST), the forward network, the pose-IK network, budgets
  rules/                     cascade_rule (the exact rule and loop), observer_rule, references (DLS, transpose, cascade)
  control/session.py         the network as a controller of the shared closed loop; the forward session
  calibration/, artifacts/   the decoders measured in NEST, and the stored calibrations
  benchmarks/                one module per benchmark, and their summaries
  simulator/                 the browser simulator
dls_ik/                      rule.py (damped least squares), controller.py, __main__.py
ccd_ik/                      rule.py (cyclic coordinate descent), controller.py, __main__.py
jacobian_transpose_ik/       rule.py
learned_dynamics/            setup (device, determinism), kinematics (PyTorch), data, model, train, planner, controller,
                             protocol (the shared protocol and command line of the two learned baselines)
mpc_nn_dynamics/, pets/      config.py (each model's settings), __main__.py
tests/                       unittest; tests that need NEST, ROS or PyTorch skip themselves where it is missing
docker/                      run.bash (every container), image.env (pinned images), Dockerfile.baselines
gazebo/                      the Gazebo world
results/                     every run and summary, one folder per model and benchmark (see Result files)
```

## Install

Everything runs in one of two Docker images, both recorded in `docker/image.env`:

* **`MODEL_IMAGE`** (pinned by digest, pulled from Docker Hub): Ubuntu 20.04, ROS Noetic, Gazebo 11 with the TIAGo,
  NEST, Python 3.8, NumPy, SciPy, Matplotlib. Every model runs here except the two learned baselines.
* **`BASELINE_IMAGE`**: `MODEL_IMAGE` plus PyTorch 2.4.1 (CUDA 12.1 wheel), built locally from
  `docker/Dockerfile.baselines`. The learned baselines run here; with `--cuda` they use the NVIDIA GPU.

```bash
docker pull "$(. docker/image.env; echo "$MODEL_IMAGE")"
docker/run.bash build-baselines          # once, needs the network (about 25 GB image); then put the printed id in
                                         # docker/image.env as BASELINE_IMAGE
```

The GPU needs the NVIDIA driver (525 or later) and the NVIDIA Container Toolkit on the host. Check it:

```bash
docker/run.bash nest --image baselines --cuda -- 'python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```

What runs where: the RAK model runs in NEST on the CPU (NEST has no GPU back end for this network); Gazebo's physics runs
on the CPU; DLS IK and CCD IK need little; only the two learned baselines' training and planning
run on the GPU.

## Tests

```bash
docker/run.bash nest -- 'python3 -B -m unittest discover -s tests'                                 # ~2.5 min
docker/run.bash nest --image baselines --cuda -- 'python3 -B -m unittest tests.test_learned'       # ~15 s
docker/run.bash nest -- 'RAK_REGRESSION=1 python3 -B -m unittest tests.test_regression_rak'        # ~4 min
```

The last one replays two stored RAK trials (a kinematic reach and a joint-loss trial) with the code as it is and checks
they come out number for number: the guard against any change to the model's behaviour.

## The reaching task

Every controller is measured on the same task (`reaching_benchmark/task.py`, `reaching_benchmark/closed_loop.py`):

* **Arms.** *TIAGo arm, 7 DoF* (`tiago7`): the arm of PAL Robotics' TIAGo, from the shoulder (`arm_1`) to the tool flange
  (`arm_tool_link`), with the joint limits of its robot description. *TIAGo arm, 4 DoF* (`tiago4`): its first four joints
  (shoulder and elbow), asked for position only. *Synthetic arm, 2–7 DoF* (`generic2`…`generic7`): 0.2 m links, joint axes
  alternating x, y, z, no joint limits. Every joint is revolute (one degree of freedom).
* **Plants.** The *kinematic plant* (`light`): every joint moves toward its target as if to arrive after 1/κ (κ = 1 s⁻¹),
  no dynamics. *Gazebo*: the TIAGo arm with its trajectory controller and physics, advanced in lock step with the
  controller, 50 ms per tick.
* **Pairs.** 30 start–goal pairs per arm from `TEST_SEED` (`reaching_benchmark/targets.py`): each goal is the pose of
  another joint configuration within 0.8 rad of the start joint by joint, so every goal can be reached.
* **The written goal.** The goal a controller is given travels toward the final goal at 1 rad/s (lift, pitch, yaw) and
  0.3 reach/s (position), the same for every controller.
* **Step limits.** A joint may be told to move at most 8 *steps* per tick; a step is 1/200 of its range on a circular
  joint, 1/140 on a bounded one (`reaching_benchmark/limits.py`); a step toward a limit fades over the last 5 steps.
* **Errors.** Position error is the Euclidean distance from the hand to the goal; orientation error is the angle of the
  one rotation that turns the hand onto the goal (combining lift, pitch and yaw).
* **End and success.** A trial ends when the written goal has arrived and the hand moved less than 0.5° and 0.4% of
  reach in 2 s, or at 25 s. It counts as **reached** when, at its last sample, orientation is within 5° and position
  within 4% of reach (32.2 mm on the 7-DoF TIAGo arm), only what the task asks.

## Results: how each one was produced

Times are wall times on 8 NEST threads on a 24-core machine; keep 8 threads (the thread count changes NEST's random
streams). Each command writes the run files and the benchmark's `summary.json` in `results/<model>/<benchmark>/`;
`summarise NAME` recomputes a summary from its run files. Run every command from the repository root.

### RAK model (`results/ring_attractor_kinematics/`)

**Calibration** (stored in `ring_attractor_kinematics/artifacts/`; optional, ~75 min):

```bash
docker/run.bash nest -- 'python3 -B -m ring_attractor_kinematics calibrate all'
```

| Benchmark folder | Question | Command (inside `docker/run.bash nest -- '…'`) | Runs | Wall |
|---|---|---|---|---|
| `screening/` | Which orientation source is worth a spiking build? Exact loop, every source, swept lag, cue noise and gain | `python3 -B -m ring_attractor_kinematics benchmark screening` | 1 (8,640 cases) | 4 min |
| `held_pose/` | The forward network at rest: 30 held poses on 12 arms | `… benchmark held_pose` | 12 | ~1 h |
| `held_pose_neuron_loss/` | The same poses under diffuse neuron loss: every class at 5–30%, each class alone at 10%, lesion seeds 1–3 | `… benchmark held_pose_neuron_loss` | 21 | ~2 h |
| `moving_estimate/` | The forward network while the arm moves: four paths, three methods; then one shared path at three speeds | `… benchmark moving_estimate`, then `… benchmark moving_estimate --condition common` | 7 | ~1.5 h |
| `reach_kinematic/` | Closed-loop reaching on the kinematic plant: 30 pairs each on the 7-DoF TIAGo arm, the synthetic 7-DoF arm and the 4-DoF TIAGo arm (position) | `… benchmark reach_kinematic` (seed 1), `… --variant predictive_encoding`, `… --variant observer`; seed repeats `… --arm tiago7 --seed 2` and `--seed 3`, each also with `--variant predictive_encoding` | 12 | 1–4.5 h each |
| `reach_pole/` | Goals near the lift-pitch-yaw pole (20 pairs, \|pitch\| > 75°) | `… benchmark reach_pole`, `… --variant observer` | 4 | 2–3 h each |
| `reach_gain/` | Loop gain κ = 1.5 and 2 | `… benchmark reach_gain --condition kappa=1.5` (and `kappa=2`), each with the three variants | 6 | 3.5–6 h each |
| `reach_disturbance/` | A 0.3 rad push of one joint at 2 s; a 100 pA noise current (20 pairs) | `… benchmark reach_disturbance`, `… --variant observer` | 4 | ~2.7 h each |
| `reach_gazebo/` | Reaching with the TIAGo arm in Gazebo (30 pairs) | see [Gazebo](#gazebo): `… benchmark reach_gazebo`, with `--variant predictive_encoding` and `--variant observer` | 3 | 2–6 h each |
| `joint_loss/` | One joint disabled 2 s into each reach: actuator lock, motor-neuron loss, allocation-neuron loss; seeds 1–3; and 10% diffuse loss | `… benchmark joint_loss --condition intact` (seed 1), then for seeds 1–3: `--condition lock`, `motor`, `allocation` (`--seed 2`, `--seed 3`); `--condition loss --loss 0.1` | 11 | ~1 h each (1.5 h for loss) |
| `joint_map/` | Every joint disabled in turn, in the exact loop (no NEST; 20 processes) | `… benchmark joint_map` | 1 (840 cases) | ~10 min |
| `cost/` | Neurons, synapses, spikes, speed (run with nothing else running) | `… benchmark cost` | 1 | ~5 min |
| `structure/` | Neuron and wiring counts from the code (no NEST) | `… benchmark structure` | 1 | ~10 s |

The intact reference of `joint_loss` at seeds 2 and 3 is the `reach_kinematic` run of the same seed. `twins`
(`… benchmark twins`) recomputes the exact-loop twins stored beside every closed-loop trial after a change to a rule.

Options of `benchmark`: `--variant` (above), `--arm`, `--seed` (NEST kernel seed), `--condition`, `--pairs`,
`--shard i/n` (every n-th pair from the i-th on, so several jobs can share a run), `--max-minutes M` (stop before the
next trial after M minutes), `--threads` (keep 8), and for `joint_loss` `--onset`, `--loss`, `--loss-class`,
`--lesion-seed`. Running the same command again resumes a run that stopped.

### DLS IK, CCD IK, NN dynamics + MPC, PETS

See [Running the baselines](#running-the-baselines). Their results appear in `results/dls_ik/reach_gazebo/`,
`results/ccd_ik/reach_gazebo/`, `results/mpc_nn_dynamics/` and `results/pets/` once run;
`results/comparison/reach_gazebo.json` puts every model side by side.

## Running the baselines

These fill the baseline rows of the paper's Fig. 3b: every model on the same 30 pairs of the 7-DoF TIAGo arm in Gazebo.
Gazebo takes one run at a time. Estimated times are for this machine.

**1. DLS IK and CCD IK** (model image):

```bash
docker/run.bash gazebo up rak-gazebo
docker/run.bash gazebo exec rak-gazebo -- 'python3 -B -m dls_ik reach --plant gazebo'    # ~0.5 h
docker/run.bash gazebo exec rak-gazebo -- 'python3 -B -m ccd_ik reach --plant gazebo'    # ~0.5–1 h
docker/run.bash gazebo down rak-gazebo
```

**2. The learned baselines** (baselines image, GPU):

```bash
docker/run.bash gazebo up rak-gazebo-gpu --image baselines --cuda
docker/run.bash gazebo exec rak-gazebo-gpu -- 'python3 -B -m mpc_nn_dynamics all --plant gazebo'   # ~2–3 h
docker/run.bash gazebo exec rak-gazebo-gpu -- 'python3 -B -m pets all --plant gazebo'              # ~4–5 h
docker/run.bash gazebo down rak-gazebo-gpu
```

`all` runs the three steps in order; each can be run on its own (`collect`, `train`, `reach`) and each resumes where it
stopped. Planning takes about 20 ms per tick for NN dynamics + MPC and about 270 ms for PETS on an RTX 4070 Ti.

**3. Side by side:**

```bash
docker/run.bash nest -- 'python3 -B -m reaching_benchmark compare reach_gazebo'
```

This writes `results/comparison/reach_gazebo.json` and prints a table: targets reached with Wilson 95% intervals, median
final errors, median time to enter tolerance for good (missed targets at 25 s), median path ratio, and for every
controller paired tests against the RAK model on the same targets (exact McNemar on reached; paired bootstrap of the
differences in final error and time).

Every baseline also runs on the kinematic plant (`--plant light`, results in `reach_kinematic/`), which needs no Gazebo.

## The learned baselines: protocol and settings

Both learned baselines follow one protocol (`learned_dynamics/protocol.py`), so that they differ only where their papers
differ, the model and the planner:

* **State** `s = (q, q̇)`: the measured joint angles and velocities. **Action** `u`: a joint step within the same caps as
  every controller, sent as the target `q + u/κ`, exactly as DLS IK's step.
* **What is learned**: how the arm responds, `Δs = f(s, u)`. The geometry is supplied: the planner scores predicted joint
  angles by forward kinematics against the written goal, with the task's weighting, cost = θ² + w²·(distance/reach)²
  (w = 3). As the RAK model and DLS IK are given the geometry, nothing about the arm's kinematics is learned.
* **Data.** Training starts and goals come from `TRAIN_SEED`; the 30 reported pairs are never used in training.
  *Phase 1*: 20 episodes of 10 s with a random step every tick (uniform within the caps). *Phase 2*: 10 iterations of
  model-based reinforcement learning as in both papers: 3 episodes of MPC with the current model on training goals, the
  transitions added to the data, the model refitted on everything. The transitions are kept in
  `results/<model>/training/transitions.npz`, the final model in `model.pt`, and every iteration's data, fit and wall
  time in `training.json`.
* **Evaluation.** The final model's MPC on the 30 reported pairs, through the shared runner, without further learning.
* **Reproducibility.** Fixed seeds for NumPy and PyTorch, deterministic algorithms, deterministic cuBLAS; the device,
  PyTorch and CUDA versions are written into every file. On the same device and versions the planners give the same
  actions run after run; Gazebo's physics is the one source of run-to-run variation.

| | NN dynamics + MPC (`mpc_nn_dynamics/config.py`) | PETS (`pets/config.py`) |
|---|---|---|
| Model | 1 deterministic network, 2 hidden layers × 500 ReLU | ensemble of 5 probabilistic networks, 4 hidden layers × 200 swish, Gaussian outputs with learned log-variance bounds |
| Loss | mean squared error of the normalised state change | Gaussian negative log-likelihood; each member on its own bootstrap |
| Training | Adam, learning rate 1e-3, batch 512, 60 epochs per fit | Adam, learning rate 1e-3, weight decay 5e-5, batch 256, 25 epochs per fit |
| Planner | random shooting: 1,000 sequences, horizon 10 ticks | CEM: population 400, 40 elites, 5 iterations, smoothing 0.1, horizon 25 ticks; 20 particles per sequence under TS∞ |
| After each tick | the first step of the best sequence | the first step of the mean; the mean shifted and reused next tick |

## What each controller reads, is given and learns

| Controller | Reads every 50 ms | Given | Learns | Internal model |
|---|---|---|---|---|
| RAK model | measured joint angles | the arm's geometry, wired into the network; calibrated decoders | nothing during reaching (decoders calibrated offline) | its own population estimate of the hand, which lags the arm by about 0.2 s; the *predictive encoding* variant runs it ahead of the measured state |
| DLS IK | measured joint angles | the arm's geometry (Jacobian) | nothing | none: every step recomputed from the measured state (reactive) |
| CCD IK | measured joint angles | the arm's geometry (forward kinematics, joint axes and origins) | nothing | none: CCD solved afresh from the measured state every tick (reactive) |
| NN dynamics + MPC | measured joint angles and velocities | the arm's geometry (for the cost) | the arm's response to joint steps | a learned model, planned 10 ticks ahead (predictive) |
| PETS | measured joint angles and velocities | the arm's geometry (for the cost) | the arm's response to joint steps, with its uncertainty | a learned ensemble, planned 25 ticks ahead (predictive) |

DLS IK and CCD IK are therefore the accuracy references of reactive controllers (CCD IK the coordinate-wise method the
cascade draws on), not like-for-like comparisons with the RAK model; the learned MPC baselines are the predictive,
model-based comparison.

## Gazebo

Start a Gazebo with the TIAGo in the background, run beside it, and stop it when done:

```bash
docker/run.bash gazebo up rak-gazebo                     # add --image baselines --cuda for the learned baselines
docker/run.bash gazebo exec rak-gazebo -- 'python3 -B -m ring_attractor_kinematics benchmark reach_gazebo'
docker/run.bash gazebo down rak-gazebo
```

Physics is paused; every 50 ms tick sends the joint targets to the arm's trajectory controller and then lets exactly one
tick of physics pass, so a controller of any speed sees the same arm. The mobile base is not anchored and is put back at
the origin before each trial. Add `--gpu` to `up` only for the camera picture in the simulator.

**Joint loss in Gazebo** (`benchmark joint_loss_gazebo`): the joint-loss study with the TIAGo arm's trajectory controller
and dynamics. Under `lock` the jammed joint's trajectory target stays at its angle at onset. Its intact reference is the
stored `reach_gazebo` run. Not run yet.

## Result files

```
results/<model>/<benchmark>/<run>.json     every trial of one run, written after every trial
results/<model>/<benchmark>/summary.json   computed from the run files of that folder only
results/<model>/training/                  the learned baselines' data, model and training record
results/comparison/<benchmark>.json        every model side by side (python -m reaching_benchmark compare)
```

A run file is named `<arm>[_<task>][_kappa<κ>][_seed<s>][_<variant>][_<condition>][_shard<i>of<n>]`, each part only when it
is not the default: the task only when not pose, κ only when not 1, the seed only for controllers with something random
in them (the RAK model's NEST kernel seed; the learned baselines' seed), the variant (`predictive_encoding`, `observer`),
the condition (`push`, `noise`, `lock`, `motor`, `allocation`, `intact`, `loss0.1_all_l1`, …) and the shard
only when a run was split. Examples: `reach_kinematic/tiago7_seed2_predictive_encoding.json`,
`reach_gain/tiago7_kappa1.5_seed1.json`, `joint_loss/tiago7_seed3_motor.json`, `dls_ik/reach_gazebo/tiago7.json`.

A reaching run file holds `run` (what was run), `design` (the whole task record), `controller` (what it reads, is given
and learns; baselines), `rows` (one per trial: start and final errors, reached, time to reach, path ratio, final joint
angles, and time courses every 4th tick) and wall and simulated times. Floats are rounded to 5 decimals in run files and
6 in summaries; NaN is written as null.

## Watching runs

Every run shows a progress bar on stderr (count, targets reached, the running trial's time and errors, an ETA); it
redraws in place in a terminal and prints one line per trial when redirected. `REACHING_PROGRESS=bar|lines|off` overrides.
To see every run of every model at once, on the host (standard library only):

```bash
python3 -m reaching_benchmark status --watch 60
python3 -m reaching_benchmark status joint_loss dls_ik/reach_gazebo
```

A run file not written for much longer than a trial (about 8 min for the RAK model on the 7-DoF TIAGo arm) means its
run has stopped; the same command resumes it. In the background with a log:

```bash
nohup docker/run.bash nest -- 'python3 -B -m ring_attractor_kinematics benchmark joint_loss --condition lock' > lock.log 2>&1 &
```

## The simulator

Set a pose goal in the browser and watch an arm reach it (http://localhost:8765/), with the RAK model, its cascade in
exact arithmetic, DLS IK or the Jacobian transpose:

```bash
docker/run.bash sim -- 'python3 -B -m ring_attractor_kinematics simulator --chain tiago7'
docker/run.bash sim -- 'python3 -B -m ring_attractor_kinematics simulator --chain tiago4 --stub'    # no NEST
docker/run.bash gazebo exec rak-gazebo -- 'python3 -B -m ring_attractor_kinematics simulator --chain tiago7 --gazebo'
```

## Code conventions

* Imports only at the top of a module. NEST, ROS and PyTorch are imported only by the modules that need them; each command
  line loads its command's module by name, so one command's imports never slow down or break another.
* `reaching_benchmark/` never imports a model. Model packages import `reaching_benchmark/` and not each other, except
  that the RAK model uses DLS IK and the Jacobian transpose as exact-loop references (`rules/references.py`).
* `reaching_benchmark/geometry/`, `ring_attractor_kinematics/neurons/`, `network/*_wiring.py` and `rules/` run without NEST.

## Licences and references

* Nagabandi, A., Kahn, G., Fearing, R. S. & Levine, S. Neural network dynamics for model-based deep reinforcement learning
  with model-free fine-tuning. *IEEE International Conference on Robotics and Automation* (2018).
* Chua, K., Calandra, R., McAllister, R. & Levine, S. Deep reinforcement learning in a handful of trials using
  probabilistic dynamics models. *Advances in Neural Information Processing Systems 31* (2018).
* Wampler, C. W. Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods.
  *IEEE Transactions on Systems, Man, and Cybernetics* 16, 93–101 (1986).
* Wang, L.-C. T. & Chen, C. C. A combined optimization method for solving the inverse kinematics problems of mechanical
  manipulators. *IEEE Transactions on Robotics and Automation* 7, 489–499 (1991).
* Kenwright, B. Inverse kinematics: cyclic coordinate descent (CCD). *Journal of Graphics Tools* 16, 177–217 (2012).

The learned baselines are reimplemented here in PyTorch from these papers; their original code is not used.
