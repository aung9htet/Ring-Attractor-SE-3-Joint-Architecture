# The data behind the inverse-kinematics results

Everything here was written by the scripts named below, in the container.
`scripts/ik_report.py` reads it and writes [`../tables.md`](../tables.md) and
[`../figures/`](../figures/). To redo it: top-level [`README.md`](../../../../README.md),
section 11.3.

| File | Test | Written by | Holds |
| --- | --- | --- | --- |
| `study.json` | 2 | `scripts/ik_study.py` | the rule with exact kinematics: one row per arm and condition, 60 pairs each |
| `sheets.json` | 3c | `scripts/run_ik_sheets.py` | three sheets alone at 80 random places, without and with level cells; the ramp readouts on them; level cells with other gains and synapses on the first 30 places |
| `parts_<arm>_<task>.json` | 3a, 3b | `scripts/run_ik_parts.py` | the arm held at a pose with a goal part of the way to another pose: turn, offset and shares as the network holds them, against the exact ones |
| `reach_<arm>_<task>.json` | 4, 5, 9 | `scripts/run_ik_reach.py` | closed-loop trials on the test pairs: one row per trial, with its time courses and the exact rule on the same pair |
| `reach_<arm>_<task>_cells.json`, `_all_at_once.json`, `reach_tiago7_orientation.json` | 6, 7 | `scripts/run_ik_reach.py` with `--position-goal cells`, `--all-at-once`, `--task orientation` | the same pairs with a variant of the layer |
| `reach_tiago4_position_motor070.json`, `_motor130.json`, `_push.json`, `_goal_change.json`, `_anywhere.json` | 8 | `scripts/run_ik_reach.py` with `--motor-scale`, `--push`, `--goal-change`, `--spread 0` | the first 20 pairs with a disturbance; `_anywhere`: goals from the whole joint ranges |
| `gazebo_kinematics_<arm>.json` | 10a | `scripts/run_ik_gazebo.py --check 5` | the Gazebo arm held at poses: its end point from Gazebo's link states against the chain's kinematics |
| `gazebo_<arm>_<task>.json`, `gazebo_tiago7_pose_dls.json` | 10b | `scripts/run_ik_gazebo.py` | closed-loop trials with the TIAGo arm in Gazebo as the robot, in the format of a reach file |
| `speed.json` | | `scripts/ik_benchmark.py` | simulated time per wall-clock time of the networks with the IK layer, with nothing else on the machine |
| `logs/` | | `scripts/run_ik_experiments.py` | the output of every run (not kept in git) |

Lengths are in metres and angles in radians unless a name says otherwise. `_reach` in a name:
a length as a share of the arm's reach (`radius_m`). `_deg`: degrees.

## A reach file

Top level: the chain, the design of the populations (`design`), of the layer (`ik_design`),
of the sheets (`sheet`) and of the level cells (`level`); `radius_m` and `centre_m` of the
workspace; the network's size (`neurons`, `ik_neurons`, `ik_wiring`, `synapses`); what each
share is divided by (`normalisers`) and its largest value (`caps`); `pair_seed`, `pairs`,
`spread` (how far a goal's joint pose is from the start, rad per joint; null: anywhere);
what was done to the loop (`motor_scale`, `push`, `goal_change_s`, `silence`); `simulated_s`,
`wall_s`, `speed`; the fit errors of the calibrations; `skipped` (pairs a time guard left
out); `summary`; and `rows`.

One row is one trial.

| Field | Meaning |
| --- | --- |
| `index`, `start`, `goal_lpy`, `goal_position` | which pair; start joint angles; goal lift, pitch, yaw; goal position |
| `start_error_deg`, `start_error_reach` | how far the hand is from the goal at the start |
| `final_error_deg`, `final_error_reach` | the same at the end: true pose against the goal |
| `reached` | within the marks of the task at the end |
| `time_to_reach_s` | from when on the hand stayed within the marks (null: not within them at the end) |
| `path_reach`, `path_ratio` | length of the end point's path, and that over the straight line |
| `path_deg`, `turn_ratio` | how far the hand turned in all, and that over the shortest turn |
| `settled_s`, `duration_s` | when the trial ended because the hand rested (null: after 25 s); its length |
| `at_limit` | a joint ended inside the 5 ring indices before a limit |
| `final_q` | joint angles at the end |
| `decoded_error_median_deg`, `decoded_error_median_reach`, `decoded_error_final_reach` | how far the forward network's pose is from the true one during the trial, and its position at the end |
| `goal_error_final_reach` | how far the goal sheets' packets are from the goal that is written, at the end |
| `largest_lead` | the largest lead of a joint's goal ring during the trial, ring indices |
| `t`, `error_deg`, `error_reach`, `left`, `shares`, `lead`, `q` | time courses, every fourth tick: time; distance from the goal; size of what no joint took; each joint's decoded share; each goal ring's lead; joint angles |
| `exact_final_error_deg`, `exact_final_error_reach`, `exact_reached`, `exact_time_to_reach_s`, `exact_path_ratio`, `exact_at_limit` | the exact rule on the same pair |
| `pushed_joint`, `start_off_rad` | the joint that was pushed, if any; how far the arm was from the start pose when the trial began |

`summary` holds what the tables quote: shares of trials reached, medians and 90th
percentiles of the final errors, of the time and of the path.

A run in Gazebo has three more things: at the top level `base_at_start` (x, y and yaw of the
robot's base); in a row `base` (the same at the end of the trial), `lag_median_rad` (how far
the joint that is furthest behind is from the target it was sent, median over the ticks) and
`lag_final_rad` (the same for every joint at the end). Its "exact rule" columns are the exact
rule on the light robot.

## A parts file

One row is one held pose with one goal.

| Field | Meaning |
| --- | --- |
| `q`, `goal_lpy`, `goal_position`, `fraction` | where the arm is held; the goal that is written; how far along the way from the pair's start to its goal that is |
| `turn`, `true_turn`, `wired_turn` | the turn as the network holds it (mean of 8 ticks), the exact one, and the one the wiring gives with exact values |
| `offset`, `true_offset`, `wired_offset` | the same for the offset, in reaches |
| `turn_direction_error_deg`, `offset_direction_error_deg` | angle between held and exact |
| `shares`, `wired_shares`, `share_sd` | each joint's share as held, as the exact cascade gives it, and its scatter over the 8 ticks |
| `left`, `wired_left` | what no joint took |
| `ramp_goal_error`, `ramp_hand_error` | the positions the ramp readouts give, against the goal that is written and the true position of the hand |
| `goal_sheet_error`, `hand_sheet_error`, `cells_error` | the same from the centre of the goal sheets' spikes, of the forward sheets' spikes, and from the position cells |
| `sheet_spikes`, `goal_sheet_spikes` | spikes per tick of each sheet |
| `room` | per joint, how much of a share toward its lower and its upper limit is let through (1: all) |

## `study.json`

`base`: the conditions every row shares. One row is one arm under one condition: `model`,
`group` (which table of section 1 it belongs to), `label`, the settings that differ
(`rule`, `normaliser`, `cascade`, `balance`, `kappa`, `spread`, `conditions`), and the
outcome over its `trials` (`reached_pct`, `final_median_reach`, `final_median_deg`,
`time_to_reach_s`, `path_over_straight`, `missed`, `missed_at_limit`).

## `sheets.json`

`places`: the 80 places, scaled. `without`, `with`: per place and sheet the spikes per second
(`totals`), per place and coordinate the centre of the packets' spikes (`centres`) and what
the ramp readout gives (`read`), and the summary the tables quote. `scan`: the same summary
for level cells with other gains and synapses.
