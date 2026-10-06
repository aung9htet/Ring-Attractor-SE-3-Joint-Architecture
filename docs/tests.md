# The tests: what each one asks, and its pass mark

The marks were fixed in the plan before any test pose was run. Results:
[`results.md`](results.md); every table: [`results/tables.md`](results/tables.md); how to run
things: the top-level [`README.md`](../README.md), section 3.

## Terms

- **Position error**: the distance between the position read from the network and the true
  position of the end point. Given as a share of the arm's **reach** (the radius of its
  workspace: the summed length of the links after the first joint; 0.806 m for the 7-joint
  TIAGo arm, 0.4 to 1.4 m for the generic chains) and in mm.
- **"Similar"**: median position error at most 2% of the reach and 90th percentile at most
  4%. For the 7-joint TIAGo arm that is 16 mm and 32 mm.
- **Orientation error**: as in the SO-3 repo, the angle of the single rotation between the
  decoded and the true orientation, with the marks of that repo (median at most 5°, 90th
  percentile at most 10°).
- **A held pose**: the arm moves to the pose at up to 1 rad/s per joint and rests 1 s; the
  position is the centre of the sheets' spikes of the last 0.5 s.
- **Anchor**: the position computed from the rings holds the packets. **Path integration**:
  the end point's velocity moves them. Both are on unless a test says otherwise.

## The tests

| Test | Question | What is done | Pass mark | Script |
| --- | --- | --- | --- | --- |
| 1 | Is the position the weighted sum claimed, and is the velocity formula right? | Exact maths for all 12 models: position against the frames of the SO-3 code, velocity against finite differences, the wiring against both | position equal to 1e-9, velocity to 1e-6 | `tests/test_kinematics.py` |
| 2 | What should the position and velocity populations reach? | The copy without spikes on 200 poses per model (not the test poses) | none: sets the expectation | `scripts/rate_study.py` |
| 3 | Does each new part do its job? | In NEST: a packet left alone for 10 s; a sheet anchored through real gate populations at random places; a free packet moved by a known velocity through the real velocity populations and cells; the velocity populations of a whole network | anchored packet within 1 cell; packet speed proportional to the velocity to within 10% | `scripts/run_parts.py`, `tests/test_nest_parts.py` |
| 4 | Is the decoded position similar to the true one? | 300 random held poses per model, 12 models | "similar", for every model | `scripts/run_static.py` |
| 5 | What do the sheets add or cost? | The same poses: position from the position cells alone, from the sheets, and the exact position of the angles the joint rings hold | none: paired comparison reported | the same data |
| 6 | Do the sheets agree? | The same poses: the two copies of each coordinate | none: reported | the same data |
| 7 | Does the position follow a moving arm? | The arm keeps moving from goal to goal (1 s rest at each) at 0.25, 0.5 and 1.0 rad/s; anchor only against anchor and path integration; the anchor given the position ahead | delay at most 0.5 s at 0.5 rad/s; back at the held-pose level 1 s after a stop | `scripts/run_tracking.py` |
| 8 | Is the packet a persistent state, and does path integration carry it? | The anchor is switched off for 2, 5 and 10 s, five times each: (a) arm at rest, (b) arm moving, with and without path integration, (c) anchor on again | (a) the packet moves at most 2% of the reach in 10 s; (c) the readout is back within 0.5 s; (b) reported | `scripts/run_dropout.py` |
| 9 | What does it take to break it? | 100 poses, two models (generic4, tiago7): 5 / 10 / 20% of the sheets' neurons switched off; 5% of the SO(3) product neurons off; noise current; sheets of 32 and 64; other neurons (2 more seeds); no velocity path | none: reported | `scripts/run_static.py` with options |
| 10 | Is the orientation unchanged in the larger network? | The orientation error on the poses of test 4, beside the SO-3 repo's | within the SO(3) marks | the same data |

Beside these: the shift-register experiment (`scripts/run_trail.py`; what happens when both
the anchor and the joint velocities are taken away), reported without a mark.

`scripts/run_experiments.py` starts tests 4 to 10 (`--list` prints the runs);
`scripts/report.py` turns the data into the tables and figures.

## How the marks are judged

- **Test 7, delay**: the shift in time for which the position read from the sheets matches
  the true position best while the arm moves.
- **Test 7, after a stop**: the median of the single readings 1 s after each stop is at
  most the 90th percentile of the same model's held poses.
- **Test 8c, "back"**: the readout is back when it is again at the level it has with the
  anchor on in the same run (the 90th percentile of the single-tick errors before the gaps)
  and stays there for five ticks. A fixed level would not do for a moving arm, whose
  readout is late by the delay of test 7 all the time.

## Pass marks that were missed

Three, all on the sheets without their anchor ([`results/tables.md`](results/tables.md),
section 12; discussed in [`results.md`](results.md), sections 2, 7 and 8):

- **Test 3, packet speed proportional to the velocity to within 10%.** Missed by far: the
  speed is right in the median and off by a third in the typical run.
- **Test 8a, a packet at rest moves at most 2% of the reach in 10 s.** Missed: about one
  cell, which is 5% of the reach. The packets all stay alive.
- **Test 8c, readout back within 0.5 s after the anchor returns.** Met in the median by a
  wide margin, missed in 15 of 190 gaps.

Nothing was changed after the test runs to meet them.

## What is and is not a test pose

- Readout weights are fitted on grids of values (`scripts/calibrate.py`,
  `scripts/calibrate_pose.py`). No pose of any chain is used.
- The numbers of the sheet design were set on sheets alone and on a 3-joint chain with
  poses drawn from other seeds (5 and 3 in throw-away scripts), and then frozen
  (`SheetDesign` in `ring_se3/sheet.py`).
- The 300 test poses of a model (seed 20261005, the same poses as in the SO-3 repo for the
  same model) were first run after that, with one exception that is said here: the first 50
  test poses of `tiago7` were run once during development to see that a 7-joint network
  works at all (position error 1.38% of the reach in the median). No setting was chosen on
  that result; what was changed afterwards (how the anchor is switched, the velocity cells)
  was decided on the tests of the anchor switch and of path integration.
- Sessions of a moving arm use goals from seed 20261016, sessions with gaps from 20261017,
  the shift-register experiment from 20261018, the parts test from 4711. Development used
  other goals (seeds 20261006 to 20261008, on the 3-joint generic chain), so that the
  sessions reported were not seen before.

## Unit tests

`python3 -B -m unittest discover -s tests` in the container (about a minute):

| File | Covers |
| --- | --- |
| `tests/test_kinematics.py` | test 1; every value a population holds stays in its range |
| `tests/test_sheet.py` | the sheet without NEST: places, labels, weights, anchor current, readout, inverse of the speed curve |
| `tests/test_rate.py` | the copy without spikes of the position and velocity populations |
| `tests/test_nest_parts.py` | test 3 with few points; a 2-joint pose end to end with both switches; a network without the velocity path |
| `tests/test_sim.py` | the simulator's loop and server with a stand-in network |
| `tests/test_report.py` | the report's statistics, the delay estimate on a log with a known delay, the gap summary, and that the documents quote only numbers from the generated tables |
