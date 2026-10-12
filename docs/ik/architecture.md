# Inverse kinematics for a pose: what is built

The hand gets a pose goal: an orientation on three goal rings (lift, pitch, yaw) and a
position as a packet on three goal sheets (planes xy, xz, yz). The network works out, all the
time, how the hand still has to turn and where it still has to go, hands every joint a share
of both, joint after joint down the arm, and each joint follows its own goal ring. The loop
closes through the robot and the forward network: joint rings, orientation, position cells,
forward sheets.

The mathematics, and how each part runs as spikes: [`TECHNICAL.md`](../../TECHNICAL.md), parts
II and III. What was measured: [`results.md`](results.md). The tests and their marks:
[`tests.md`](tests.md). How to start the demonstration: [`README.md`](README.md).

Everything here sits on top of the pose network ([`../architecture.md`](../architecture.md)).
The orientation half is the layer of the SO-3 repo, taken over as it is.

## 1. The picture

![The pose inverse-kinematics layer of the 4-joint TIAGo arm: goal rings and goal sheets, the two comparisons, the cascade of shares, and per joint a gain pair and a goal ring](figures/architecture.png)

The 4-joint arm with a pose goal, drawn from the wiring the network is built from, with the
neurons of every part. Read it from the top left.

- The goal's three angles sit on three rings; cos and sin of each give the goal orientation.
  Compared with the decoded orientation this is the **turn** `e`: about which axis, and how
  far, the hand still has to turn.
- The goal's position is three numbers, which anchor a packet on each of three goal sheets,
  as the position cells anchor the forward sheets. The **ramp readouts** turn the packets of
  both sets of sheets back into numbers, and their difference is the **offset** `d`: where
  the hand still has to go. This is the ramp comparator of the SO-2 repo, in a plane: the goal
  sheets are its "cold" side, the forward sheets its "warm" side.
- Along the upper row turn and offset are handed down the arm. Each joint takes its
  **share** (middle row) from what the joint knows about itself (green row): it turns the hand
  about its axis `a_k` and moves the end point along `c_k = a_k x lever_k`.
- The share moves the two sides of the joint's gain pair apart, which writes the joint's
  goal ring beside its state ring; the robot moves every joint toward its goal ring.

The same for seven joints: [`figures/architecture_tiago7.png`](figures/architecture_tiago7.png).
With a position goal only, which is what the 4-joint arms are asked for:
[`figures/architecture_position.png`](figures/architecture_position.png).

```bash
docker/run.bash nest --out /tmp/ring-se3 -- 'python3 -B scripts/draw_ik_architecture.py --chain generic7 --out /out/ik_generic7.png'
```

draws it for any chain (`--task position`, `--position-goal cells` and `--all-at-once` for the
variants).

The same as text, with the signals named:

```
 goal rings            goal gates x, y, z          pose network (unchanged)
 lift pitch yaw             │ anchor               joint rings ─► cos/sin ─► link orientations ─► decoded R
   │    │    │              ▼                           │              │            │
   ▼    ▼    ▼         3 goal sheets                    │              │ axis a_k   └─► position cells ─► gates
  cos / sin of each    (a packet each)                  │              │ lever_k                             │ anchor
   │                        │ ramp readout ("cold")     │              │                                     ▼
   ▼                        ▼                           │              │                             3 forward sheets
  goal matrix G        goal position ───────────────────┼──────────────┼───► offset d = goal − hand ◄── ramp readout ("warm")
   │                                                    │              │            │
   └────────────────────────────────────────────────────┼──► turn e = ½ Σ r_i × g_i │
                                                        │              │     │      │
                                                        │              ▼     ▼      ▼
                                                        │   joint 1: c1 = a1 × lever1;  t1 = (a1·e + w² ρ1 c1·d) / N1
                                                        │            left: e2 = e − a1 t1,  d2 = d − ρ1 c1 t1
                                                        │   joint 2: the same with e2, d2;  ...
                                                        │                    │ two halves per share (+ and −)
                                                        ▼                    ▼
                                          state ring of joint k ──gates──► gain pair of joint k ──writes──► goal ring of joint k
                                                        ▲                                                         │
                                                        │ encoder                                                 │ read as an angle
                                                    joint angle q_k ◄──── the joint moves toward its goal ring ───┘
```

`w` is the balance (3: a reach of offset counts as 3 rad of turn), `ρ_k` the largest length
of joint k's lever over the reach, and `N_k` what the share is divided by, a constant of the
chain (section 4).

## 2. The parts

| Part | Per | Neurons | Holds | File |
| --- | --- | ---: | --- | --- |
| Goal ring with encoder | goal angle (3) | 200 | the goal lift, pitch or yaw as a bump | `ring_so3/nest/ring.py` |
| cos / sin population | goal ring (3) | 120 | cos and sin of the goal angle | `ring_se3/nest/ik_network.py` |
| Goal matrix | once | 14 x 200 + 4 x 60 | the nine entries of the goal orientation | `ring_so3/ik/wiring.py`, `_goal_steps` |
| Turn | once | 18 x 200 + 3 x 60 | twice the turn that is still to make, three numbers | `ring_se3/ik/wiring.py` |
| Goal gate | coordinate (3) | 2 x 60 | the goal's x, y or z as a number: a population set from outside, and the gate it drives | `ring_se3/nest/ik_network.py`, `_build_position_goal` |
| Goal sheet | plane (3) | 48 x 48 | the goal position as a packet; anchored by two goal gates and 16 bias cells | `ring_se3/nest/sheet.py` |
| Level cells | sheet (6: 3 goal, 3 forward) | 60 | how far the packet's strength is from its set point; they send every neuron of the sheet the same current against it | `ring_se3/nest/sheet.py`, `connect_level` |
| Ramp readout | coordinate and side (6) | 160 | one coordinate of the two packets that carry it, as a number | `ring_se3/nest/ik_calibrate.py` |
| Offset | once | 3 x 60 | four times the offset (goal ramp minus hand ramp), three numbers | `ring_se3/ik/wiring.py` |
| Lever | joint from the second on | 3 x 60 | the lever from the joint to the end point over its largest length | `ring_se3/ik/wiring.py`, `_levers` |
| Direction | joint | up to 6 x 200 + 3 x 60 | `c_k = a_k x lever_k`: along what the joint moves the end point | `ring_se3/ik/wiring.py` |
| Share | joint | up to 6 x 200 + 2 x 60 | the joint's part of turn and offset, as a + half and a − half | `ring_se3/ik/wiring.py` |
| What is left | joint | up to 6 x 200 + 6 x 60 | turn and offset less what the joints so far took | `ring_se3/ik/wiring.py` |
| Hold at a limit | joint with limits | synapses only | the joint's own ring inhibits the half that points beyond a limit | `ring_se3/nest/ik_network.py`, `_hold_at_limit` |
| Gain pair | joint | 2 x 200 | left and right side, gated by the state bump, moved apart by the share | `ring_so3/nest/goal_ring.py`, `build_written_ring` |
| Goal ring of a joint | joint | 200 | where the joint should be next | `ring_so3/nest/goal_ring.py` |
| Readout cells | once | 6 to 12, and 2 to 8 per joint | comparisons, shares, what is left, the ramp readouts, as numbers for the tests and the page | `ring_se3/nest/ik_network.py` |

- 200-neuron populations multiply two values, 60-neuron populations hold a sum (SO-3
  reference, sections 5 to 8).
- Joint 1 turns about a fixed axis: its part of the turn, what it leaves of the turn, and its
  direction `c_1` need no product.
- A joint whose lever lies along its own axis moves the end point nowhere (the last joint of
  `tiago7`): it gets no lever and no direction populations, and its share is its part of the
  turn alone.
- Neurons: 35,916 for `tiago4` with a position goal (pose network 14,968, this layer
  20,948); 71,976 for `generic7` and 69,036 for `tiago7` with a pose goal (pose network
  24,208). The table of every part: [`results/tables.md`](results/tables.md), section 8.
- The pose network is built without its velocity path (nothing integrates here: the sheets
  are anchored all the time) and with level cells on its three sheets
  (`ik_options` in `ring_se3/nest/ik_network.py`).
- **The robot** is not part of the network. Two answer the same few calls
  (`ring_so3/robots.py`): the *light robot*, joint angles that move exactly as told, and the
  *TIAGo arm in Gazebo*, with its own trajectory controller and gravity
  (`ring_so3/gazebo/link.py`, `docker/run.bash gazebo`; the TIAGo chains only).

## 3. One tick (50 ms)

1. The goal that is written moves toward the goal that was given: lift, pitch and yaw at up
   to 1 rad/s (lift and yaw the short way round), the position on a straight line at up to
   0.3 reaches per second.
2. Every joint moves at 1 per second times the distance from its angle to the angle its goal
   ring held at the end of the last tick (the light robot: no dynamics). In Gazebo the same
   angle goes to the arm's controller as a target to be reached after one second, and
   physics then advances by exactly these 50 ms.
3. The joint angles are written on the joint rings, the goal angles on the goal rings, the
   goal position into the goal gates; NEST simulates 50 ms.
4. Read: the decoded pose, turn and offset, each joint's share and what is left, the place
   of the goal packets, and the centre of each joint's goal ring (summed over the last 3 ticks).

`ring_se3/ik/session.py` is this loop. `PoseIkSession.reach(goal_lpy, goal_position)` runs one
trial, `PoseIkSession.place(q)` moves the arm by hand (the goal then stays on the hand, so
nothing is asked of the arm).

## 4. Three things that are new against the orientation IK

**The ramp readout: a sheet's place as a number.** The cascade works with numbers; the
position is on sheets. One population per coordinate reads the two sheets that hold it. Every
sheet neuron sends it two things: its spike weighted with its place (a number that rises
along the sheet: the *ramp*), and its spike as such (the packets' strength: the *level*). The
population's readout is fitted to give ramp over level, the centre of the spikes. A sheet has
two coordinates and each coordinate is on two sheets: this is how a two-dimensional packet
ends up as three numbers, each read from all the spikes that carry it. Fitted on three real
sheets, on a grid of places, not on any pose (`scripts/calibrate_ramp.py`).

**Level cells: a packet of one strength.** The anchor's current is a small difference of
large parts, and its height is off by some tens of pA from place to place. Measured: a
packet then sends anything from no spikes to more than 600 a second, and at a few places it
goes out. A readout that weighs spikes cannot work on that. An error of the height is the
same current at every neuron of the sheet, so 60 *level cells* per sheet hold how far the
packet's strength is from a set point and send every neuron the same current against it.
This is a homeostatic loop on the sheet itself. With it no packet is weak, and the ramp
readout is right to 0.59% of the reach, root mean square ([`results.md`](results.md),
section 2).

**The share is divided by a constant.** The share that removes most of what is left divides
by `1 + w² |m_k|²`, where `m_k` is how far joint k moves the end point per radian; this
changes with the pose, and nothing in the tool box divides. The network divides by
`N_k = 1 + w² x (the mean of |m_k|² over the joint's poses)`, a number worked out once from
the chain's geometry. With exact kinematics this reaches 95% to 100% of the goals, as the
exact division does; on no arm more than two goals of 60 fewer (test 2).

## 5. The simulator

`scripts/ik_simulator.py` serves one page (`ring_se3/sim/ik_page.py`); the loop behind it is
`ring_se3/sim/ik_server.py`. [`README.md`](README.md) walks through it.

- **The goal** is drawn in the 3D view: a dot with three arrows (drag the dot to move it in
  the plane of the screen, an arrow tip to move it along one axis) and a ring around each of
  its axes (drag a ring to turn it). Six sliders do the same.
- **Task**: pose, position only, orientation only. **Position compared on**: the sheets, or
  as numbers. Both rebuild the network, since the cascade is wired in.
- **Method**: ring IK (spikes); the exact cascade; damped least squares; Jacobian transpose.
- **Robot**: the light robot, or the TIAGo in Gazebo when the simulator runs beside one. A
  Gazebo panel shows the camera picture and how far the joints are behind their targets.
- **Goal anchor** off: the goal gates fall silent and the goal sheets keep the goal by
  themselves.
- **Runs**: every goal starts a run; "same start and goal, next method" puts the arm back
  and repeats the goal with another method. The distance from the goal of the last six runs
  is drawn in two charts (position, orientation), with a table.
- The three planes with the goal's packets and the hand's packets on top of each other, each
  joint's share and lead, and the spikes of every ring.

## 6. Choices, and where each was decided

| Choice | Decided by |
| --- | --- |
| Turn and offset enter at the base joint and are passed down the arm; all joints move together | the exact rule: with every joint getting the whole error, 78% to 98% of the goals are reached where the cascade reaches 95% to 100% ([`results/tables.md`](results/tables.md), section 1) |
| The position goal is a packet on three goal sheets, and positions are compared through the sheets | you (2026-10-09); a second version that compares numbers is built for comparison (test 6) |
| Each coordinate is read from both sheets that carry it, by one population | a packet that is weak then counts little; measured on sheets alone |
| Level cells on all six sheets | measured: without them 33 of 240 packets are weak, two of those out ([`results.md`](results.md), section 2) |
| The share is divided by a constant, the mean over the joint ranges | the exact rule (test 2): as good as the exact division, clearly better than its largest value |
| Balance 3 | the exact rule (test 2): 2 is slower on every arm; with 4 and 5 the 7-joint TIAGo arm reaches fewer goals |
| Loop gain 1 per second | the exact rule (test 2): 0.75 is slower; 1.5 is faster, but the end point's path is about a quarter longer; with 1 a delay of 0.55 s costs no goals |
| A share is two halves, each capped at a lead of 8 ring indices; near a limit it is held down by the joint's own ring | taken from the SO-3 repo |
| The goal position travels on a straight line at 0.3 reaches per second | an anchor that travels carries its packet, one that jumps silences it; and the comparison never sees the whole distance at once |
| The 4-joint arms are asked for the position alone | a pose has six freedoms |
| A trial ends when the hand rests | the decoded shares are never exactly quiet |
| No velocity path, no path integration | the sheets are anchored all the time; the forward results say path integration is crude |
| Gazebo is stepped with the network, 50 ms of physics per tick | the 7-joint network runs at about a fifth of real time; taken from the SO-3 repo, which took it from the SO-2 repo |

## 7. What is not built

- The v4 controller of the SO-2 repo under each goal ring. A joint is told the angle its goal
  ring holds, by the light robot's rule or by the Gazebo arm's own controller.
- A way out of poses where the hand cannot move along the offset (a stretched arm), and a
  pull of the joints toward the middle of their ranges. Both show in the results.
- Goals far from the hand: the rule follows the error downhill and misses many of them
  (test 2). A planner in front of it (via-points) is not built.
- A second sweep of the cascade per tick. With exact kinematics it brings the hand within
  the marks after 3.0 to 4.3 s instead of 4.1 to 5.5 s; it would double the products.
