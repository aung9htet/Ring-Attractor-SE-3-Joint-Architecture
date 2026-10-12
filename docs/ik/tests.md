# Inverse kinematics for a pose: the tests and their marks

The marks were fixed on 2026-10-09, before any test pair was run in closed loop; test 10 and
its marks on 2026-10-10, before any run in Gazebo. What came out: [`results.md`](results.md); every table: [`results/tables.md`](results/tables.md); what
is built: [`architecture.md`](architecture.md).

## Terms

- **Reach**: the radius of the arm's workspace, the summed length of the links after the
  first joint (0.806 m for the 7-joint TIAGo arm). Position errors are given as a share of it.
- **Task**: what the hand is asked for. `pose` (orientation and position) for the 7-joint
  arms; `position` for the 4-joint arms, which have too few joints for a pose.
- **A goal is reached** when, at the end of a trial, the true position of the hand is within
  4% of the reach of the goal and, where the task has an orientation, the true orientation
  within 5° of it. 4% is the mark the forward network's position has for its 90th percentile;
  5° is the mark of the orientation IK in the SO-3 repo.
- **Test goals** are the poses of joint poses of the same arm, so each one can be reached.
  The goal's joint pose is the start pose with every joint moved by up to 0.8 rad, inside its
  limits. Test 2 says why: with exact kinematics the rule reaches 95% to 100% of such goals,
  and 37% to 77% of goals drawn from the whole joint ranges (damped least squares: 53% to
  80%). It follows the error downhill and does not plan.

Seeds: the test pairs (`TEST_SEED`), the development pairs (`DEVELOPMENT_SEED`) and the pairs
of the exact-rule study (`STUDY_SEED`) are three different streams
(`ring_se3/ik/pairs.py`).

## The tests

| Test | What it answers | Pass mark | Run with |
| --- | --- | --- | --- |
| 1. Maths | what a joint does to the hand, the cascade adds up to turn and offset, the wiring computes what the rule computes | equal to 1e-9 | `tests/test_ik_rules.py`, `tests/test_ik_wiring.py` |
| 2. Exact rule | what a share is divided by; cascade against all joints at once; the balance between turning and moving; goals near and anywhere; two standard methods | none: sets the design | `scripts/ik_study.py` |
| 3a. Comparisons in spikes | decoded turn and offset against the exact ones, arm held by hand | direction of the turn within 10° (median); the offset within 2% of the reach (median) | `scripts/run_ik_parts.py` |
| 3b. Shares in spikes | decoded shares against the exact cascade | right sign in 95% of the shares above 0.05 rad | `scripts/run_ik_parts.py` |
| 3c. The sheets | does a packet keep its strength with level cells; does the ramp readout give the packets' place, on places it was not fitted on | none: reported (both were measured while they were built) | `scripts/run_ik_sheets.py` |
| 4. Reaching | 40 start and goal pairs for tiago4 and generic4 (position), 30 for generic7 and tiago7 (pose) | reached: 85% of the goals, for each arm; 90% of the goals the exact rule reaches; final error of reached goals 3% of the reach and 3° (medians) | `scripts/run_ik_experiments.py --batch reach_position reach_pose` |
| 5. Trajectory | the path of the end point against the straight line; time until the hand is within the marks for good | path at most 1.5 times the straight line; within the marks after at most 10 s (medians) | the same data |
| 6. Sheets or numbers | positions compared on the sheets, against the goal's three numbers compared with the position cells; `tiago4` on its 40 pairs, `tiago7` on its first 20 | none: reported | `--batch variants` |
| 7. The cascade | every joint gets the whole error (`tiago4`, 40 pairs); orientation alone, the SO-3 cascade in this network (`tiago7`, 20 pairs) | none: reported | `--batch variants` |
| 8. Disturbances | `tiago4`, the first 20 pairs each: the robot moves 30% less or more than told; a joint pushed 0.3 rad; the goal changed on the way; goals anywhere in the joint ranges | none: reported | `--batch disturbances` |
| 9. How trials end | with a joint at a limit, or free; every missed goal listed | none: reported | the data of test 4 |
| 10a. Gazebo's arm against the chain | the TIAGo arm in Gazebo held at 5 poses: its end point from Gazebo's own link states against the chain's kinematics at the joint angles Gazebo reports | within 1 mm | `scripts/run_ik_gazebo.py --check 5` |
| 10b. Gazebo as the robot | the Gazebo arm follows the goal rings through its own controller, with gravity and lag. `tiago7`, pose, the first 30 test pairs, with the ring IK and with damped least squares; `tiago4`, position, the first 20, ring IK | ring IK on `tiago7`: at least 80% of the goals it reaches on the light robot. The rest reported: how far a joint is behind its target, trials in which the arm is stopped by its own body, how far the base moved | `scripts/run_ik_gazebo.py` |

## How a trial runs

1. The arm is moved by hand to the start pose. The goal that is written into the network
   follows the hand, and the trial starts only when it has arrived.
2. The goal is given. What is written travels to it: lift, pitch and yaw on the goal rings at
   up to 1 rad/s, the position on the goal sheets along a straight line at up to 0.3 reaches
   per second.
3. The trial ends when the written goal has arrived and the hand has turned less than 0.5°
   and moved less than 0.4% of the reach within 2 s, or after 25 s.
4. Recorded: true distance and turn from the goal over time, path of the end point, decoded
   shares, lead of every goal ring, how far the forward sheets are off, whether a joint ended
   within 5 ring indices of a limit.
5. The exact rule runs on the same pair (0.3 s delay, share noise 0.01 rad, errors bounded as
   the populations bound them), so that a goal the network misses can be told from a goal
   the rule misses.

In Gazebo (test 10b) the same, with three differences: the arm is moved to the start pose
through its controller; the base, which is not anchored, is put back at the origin before
every trial; and physics advances 50 ms for every tick of the network, so a network that is
slower than real time does not let the arm run away. A joint is told the angle its goal ring
holds, as on the light robot.

## What was set on what

- **Balance 3, loop gain 1 per second, shares divided by the mean over the joint ranges**:
  by the exact rule on the study pairs (test 2). No spiking run entered.
- **Level cells** (set point 500 spikes per second, gain 200 pA, synapse 100 ms) and the
  **ramp readout** (160 neurons, fitted on a grid of 257 places): on three sheets alone. No
  pose of any chain.
- **Goal rings of the joints, share halves, the hold at a joint limit**: taken from the SO-3
  repo as they are.
- **Development pairs**: 3 + 6 trials of tiago4, 2 of generic7 and 3 of tiago7, and the
  comparisons on 12 held poses of tiago4, to see that the loop closes. They ran with a balance
  of 2: the default in the code had not yet been set to the value the study chose.
- **One exception among the test pairs, said here.** The first runs of test 4 on the two
  7-joint arms were started with that same wrong default. It was noticed after three trials
  in all; they were thrown away, the default was set to 3 (the study's value, chosen before),
  and the runs were started again. No setting was chosen on those three trials.
- **The other level cells of test 3c** (four other gains and synapses) were run after the
  built ones were fixed, to check the model of `TECHNICAL.md`, section 21. Nothing was
  changed by them.
- **Gazebo** (test 10): the loop gain is that of the light robot, 1 per second. Should it
  have to be set again because of the lag of the arm's controller, that is done on
  development pairs and said here.
- The marks of tests 3a, 3b, 4 and 5 were written down before the test pairs ran, those of
  test 10 before any run in Gazebo. Nothing was changed afterwards to meet them.

## Unit tests

`python3 -B -m unittest discover -s tests` in the container:

| File | Covers |
| --- | --- |
| `tests/test_ik_rules.py` | test 1: the moves against finite differences, the lever constants, the cascade's sums, the exact share leaves the least, orientation alone equals the SO-3 cascade, joint limits, the closed loop with exact kinematics, the standard methods, the pairs |
| `tests/test_ik_wiring.py` | test 1: the wiring against the rule for 12 chains, three tasks, both position goals, with and without cascade; every held value in its range; the parts of the 7-joint arm |
| `tests/test_ik_nest.py` | tests 3a to 3c with few points; a network built without goal sheets; a 3-joint arm reaching position goals; the goal sheets keeping a goal without their anchor; the neuron count against the drawing's |
| `tests/test_ik_sim.py` | the IK simulator's loop and server with a stand-in network; the choice of robot; the route of Gazebo's camera picture |
| `tests/test_robots.py` | the light robot; the Gazebo robot, the pose loop on it and the check of test 10a, all on a stand-in for ROS |
