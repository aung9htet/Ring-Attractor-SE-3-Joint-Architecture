# Analysis: the paper's figures from `results/`

This folder turns the runs in `results/` into every figure of the paper and the numbers behind each panel. It never
runs a controller or a robot. The one exception is `activity`, which replays one stored joint path through the forward
network in NEST to record the spikes drawn in Fig. 2.

## Order of work

1. **Run the models** as in the main [README](../README.md) ("Results: how each one was produced" and "Running the
   baselines"). This fills `results/<model>/<benchmark>/`. The repository already holds every run, so this step is
   needed only to reproduce them.
2. **Run the analysis below**, inside the model image, from the repository root. Outputs go to `analysis/`.

Everything at once (under a minute):

```bash
docker/run.bash nest -- 'python3 -B -m analysis all'
python3 -B -m analysis.architecture.render_figure1_b_final   # on the host, if the image has no Chrome (PDF of Fig. 1)
```

The same, step by step:

```bash
docker/run.bash nest -- 'python3 -B -m analysis derive'       # shared data: seconds, NumPy only
docker/run.bash nest -- 'python3 -B -m analysis activity'     # Fig. 2a-c spikes: seconds, NEST
docker/run.bash nest -- 'python3 -B -m analysis figure all'   # every data figure: about 20 s
docker/run.bash nest -- 'python3 -B -m analysis architecture' # Fig. 1: hand paths + circuit + motifs (SVG; PDF where Chrome exists)
docker/run.bash nest -- 'python3 -B -m analysis list'         # every figure, its file and what it needs first
```

Each figure can also be drawn on its own, `python3 -B -m analysis figure NAME [NAME …]`, once the step in its "Run
first" column has run.

## Outputs

| Folder | What |
|---|---|
| `analysis/figures/` | every figure as PDF (vector), SVG and PNG (300 dpi) |
| `analysis/source_data/` | data shared by several figures, written by `derive` and `activity` (below) |
| `analysis/source_data/plotexports/` | the numbers behind every plotted panel, one CSV or JSON per panel group; `inputs_sha256.json` (the SHA-256 of every input read) and `selection_rules.json` (which trials each figure shows and why) |

## Shared data (`derive`, `activity`)

| Command | Writes (`analysis/source_data/`) | From |
|---|---|---|
| `derive` | `common_path_traces.npz`, `common_path_metrics.json`: the hand-orientation estimate error at every 50 ms tick of the shared joint path at 0.25, 0.5 and 1 rad/s, without and with predictive encoding | `results/ring_attractor_kinematics/moving_estimate/common{0.25,0.5,1}.json` |
| | `sensitivity_metrics.json`: held-pose error medians of each of the 21 neuron-silencing masks, beside the intact arm | `held_pose_neuron_loss/tiago7_loss*.json`, `held_pose/tiago7.json` |
| | `closed_loop_loss.json`: the 30 TIAGo7 reaches with 10% of neurons silenced, beside the same reaches intact | `joint_loss/tiago7_seed1_loss0.1_all_l1.json`, `reach_kinematic/tiago7_seed1.json` |
| `activity` | `illustrative_activity.npz` (+ metadata): joint-ring and position-sheet spike counts, decoded ring centres and sheet-decoded hand position over a forward replay (kernel seed 11, 8 threads) of the joint path of the RAK model's kinematic reach to target pair 0 | `reach_kinematic/tiago7_seed1.json`, row `index` 0 |

`activity` is deterministic: with the stored calibrations, seed 11 and 8 threads it reproduces the paper's recording
spike for spike (3,928 spikes on the xz sheet).

## Every figure

Folders below are under `results/`. "RAK" means `results/ring_attractor_kinematics/`. Every closed-loop trial is
scored at the trial's last sample: θ ≤ 5° and ‖Δp‖ ≤ 5% of reach (40.3 mm on the TIAGo 7-DoF arm).

| Paper | Name (`figure NAME`) | Output | Run first | Reads | Panel data (`plotexports/`) |
|---|---|---|---|---|---|
| Fig. 1 | `fig1`, then `architecture` | `fig1_model` (a alone: `fig1a_hand_paths`) | — | a: target pair 0 in every controller's `reach_gazebo/` (the RAK model without and with predictive encoding, DLS IK, CCD IK, NN dynamics + MPC, PETS); b–e: none (the circuit and its motifs, drawn from the arm's geometry and the cascade's constants at one pose pair, `analysis/architecture/`) | `fig1_hand_paths_target0.csv`; `../figure1_snapshot.json`, `../figure1_b_manifest.json` |
| Fig. 2 | `fig2` | `fig2_representation` | `derive`, `activity` | `illustrative_activity.npz`; RAK `held_pose/summary.json`; `common_path_traces.npz`; RAK `reach_kinematic/tiago7_seed1.json` (goal of target 0) | `fig2_*.csv` |
| Fig. 3 | `fig3` | `fig3_reaching` | — | RAK `reach_gazebo/` (without and with predictive encoding); `dls_ik/`, `ccd_ik/`, `mpc_nn_dynamics/`, `pets/` `reach_gazebo/`; RAK `reach_kinematic/` (Supp. Table S3) | `fig3_*.csv`, `supp_rak_kinematic_plant.csv` (Table S3) |
| Fig. 4 | `fig4` | `fig4_prediction` | `derive` | `common_path_traces.npz`; RAK `moving_estimate/summary.json`; every controller's `reach_gazebo/` (d, e); RAK `reach_kinematic/`, `reach_gain/`, `reach_gazebo/` (Supp. Table S4) | `fig4_*.csv`, `tableS4_predictive_encoding_paired.csv` (Table S4) |
| Fig. 5 | `fig5` | `fig5_ablation` | `derive` | RAK `joint_loss/` (seeds 1–3, every condition); every controller's `reach_gazebo/` and `joint_loss_gazebo/` lock run (panel f; drawn as "not run yet" until `docker/run_tiago_stuck.bash` has run); RAK `reach_kinematic/tiago7_seed{2,3}.json` (intact reference); `sensitivity_metrics.json`; RAK `held_pose/summary.json` | `fig5_*.csv`, `ablation_counts.json` |
| Supp. events of one reach | `supp_reach_events` | `supp_reach_events` | — | RAK `reach_gazebo/tiago7_seed1.json` (target 0) | `supp_reach_events_*.csv` |
| Supp. supporting controls | `supp_supporting` | `supp_supporting` | — | RAK `reach_disturbance/`, `reach_gain/` | `supp_gain_*.csv`, `supp_push_recovery.csv` |
| Supp. joint loss by joint and exact map | `supp_joint_loss` | `supp_joint_loss` | — | RAK `joint_loss/`, `joint_map/all.json` | `supp_joint_loss_*.csv`, `supp_joint_map.csv` |
| Supp. held-pose silencing | `supp_neuron_loss` | `supp_neuron_loss` | `derive` | `sensitivity_metrics.json`; RAK `held_pose/summary.json` | — |
| Supp. closed-loop silencing | `supp_closed_loop_loss` | `supp_closed_loop_loss` | `derive` | `closed_loop_loss.json` | `supp_closed_loop_loss.csv` |
| Supp. forward representation | `supp_forward` | `supp_forward` | — | RAK `held_pose/summary.json`, `moving_estimate/summary.json` | — |
| Supp. orientation sources | `supp_observer` | `supp_observer` | — | RAK `moving_estimate/summary.json` | — |
| Supp. chart poles | `supp_observer_pole` | `supp_observer_pole` | — | RAK `moving_estimate/summary.json` (`pole` path) | — |
| Supp. exact-loop screening | `supp_screening` | `supp_screening` | — | RAK `screening/summary.json` | — |

Examples:

```bash
docker/run.bash nest -- 'python3 -B -m analysis figure fig3'
docker/run.bash nest -- 'python3 -B -m analysis derive && python3 -B -m analysis figure fig4 fig5'
```

## Code

| File | What |
|---|---|
| `__main__.py` | the command line above |
| `common.py` | where inputs and outputs live; reading with hashes |
| `derive.py` | the shared data (no simulation) |
| `activity.py` | the NEST replay for Fig. 2a–c |
| `figures.py` | Figs 1–5 and the supplementary figures drawn from runs; `FIGURE_MAKERS` maps each name to its function |
| `diagnostics/` | the four diagnostic supplementary figures (forward representation, orientation sources, chart poles, screening) |
| `architecture/` | Fig. 1: `snapshot.py` (the geometry), `figure1_art.py` and `figure1_population_strip.py` (glyphs), `draw_figure1_b_final.py` (composes a with the circuit b–d, its blue predictive-encoding marks, the Fig. 5 ablation badges and dashed outlines, and the motif numbers of e, as one SVG), `render_figure1_b_final.py` (PDF and PNG, with Chrome) |

The analysis reads only `results/` and the repository's own packages (`reaching_benchmark`,
`ring_attractor_kinematics`), so after a model is rerun, rerunning the analysis redraws every figure from the new runs.
