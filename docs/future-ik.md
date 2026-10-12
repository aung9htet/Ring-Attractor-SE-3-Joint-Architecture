# Inverse kinematics toward a pose goal: the ideas, and which are built

This page collects ways to go from "the end point should be at this position with this
orientation" to "these joints should move", using what the forward network of this repo
already holds. It was written before any of it was built (2026-10-05) and is kept as it
was, with a line under each idea that says what became of it (2026-10-09).

**Idea 1 is built**, in a changed form: [`ik/architecture.md`](ik/architecture.md), measured
in [`ik/results.md`](ik/results.md). Idea 2 is built in part. Ideas 3 to 7 are not built.

The forward network is described in [`architecture.md`](architecture.md), what it achieves
in [`results.md`](results.md). The orientation half of the problem is on the same page of
the SO-3 repo (`docs/future-ik.md` there); this page adds the position and puts the two
together.

## What is there to build on

- **The position of the end point** twice over: as three numbers (the position cells) and
  as packets on three sheets, where a place is a group of neurons.
- **The orientation of every link**, and with it **each joint's turning axis** `a_k` (a
  column of link k's matrix), as in the SO-3 repo.
- **Each joint's lever** `l_k`, from the joint's origin to the end point: a population per
  joint, built for the velocity.
- **The cross products `a_k × l_k`** inside the velocity path. `a_k × l_k` is the direction
  and the amount by which joint k moves the end point: the k-th column of the position
  Jacobian. Today the network multiplies by the joint velocity first and takes the cross
  product second; with the order swapped the column itself is a population, at the same
  cost.
- **The velocity of the end point** from the joint velocities, to about a tenth of its size.
- **A population that multiplies**, with one readout fitted once.
- **Per joint, the controller of the SO-2 repo**: a goal ring whose bump the joint follows.
  Moving a goal bump is how a joint is told to move.

## Idea 1: "error times Jacobian column" for each joint

> **Built (2026-10-09), with three changes.**
> (1) The error is handed down the arm, joint after joint, each joint taking its share of
> what the joints before it left (the cascade of the SO-3 repo, over turn and offset). With
> every joint getting the whole error, as written below, the exact rule reaches 78% to 98%
> of the goals where the cascade reaches 95% to 100%.
> (2) The position goal is a packet on a second set of three sheets. Both sets are read by
> *ramp readouts*, and the sheets have *level cells* that hold a packet's strength; without
> those a packet cannot be read as a number.
> (3) A joint's share is divided by a constant of the chain, so that turn and offset are in
> balance. The free number `w` came out as 3 radians per reach.
> Of the weak spots listed below, the second and third show in the results as expected
> (the gain had to stay at 1 per second; the hand ends about as far from the goal as the
> position is read). The fourth was not measured for the inverse kinematics.

The transpose-of-the-Jacobian rule. Every part is a sum or a product.

1. **Goal.** A position goal is three numbers held by three goal populations, or a goal
   packet on a second set of sheets read like the first. An orientation goal is three goal
   rings turned into a matrix by a small copy of the SO(3) network (SO-3 page, idea 1).
2. **Position error** `e_p = goal - p`: three sum populations. No product.
3. **Orientation error** `e_R = ½ Σ_i (r_i × g_i)` over the columns of the current and the
   goal matrix: 18 products (SO-3 page).
4. **Each joint's share.** `d_k = (a_k × l_k) · e_p / reach + w · a_k · e_R`. The first
   term is three products per joint on top of the Jacobian column, the second three more.
   `w` weighs a radian of orientation error against a reach of position error; it is the
   one free number of the scheme.
5. **Into the joint controller.** `d_k` shifts joint k's goal bump, forward when positive,
   back when negative. The joint's own loop follows.

Why it should work: with small steps the rule can only reduce `|e_p|² + w·(orientation
error)`, it never divides, and only the *direction* of the columns has to be right, which
the forward network gives to about a degree and a tenth.

Cost for 7 joints, on top of this repo's network: 21 products for the position part, 18 +
21 for the orientation part.

Known weak spots, to be measured:

- **It is slow or stuck near singular poses** (arm stretched out, or two axes in line), and
  it can stop at joint limits. Remedy to try: a small random or mid-range drive on the
  redundant direction (idea 4).
- **The forward readout is late**, by about a quarter of a second for the position
  ([`results.md`](results.md), section 6). The outer loop has to be slower than that, or
  use the write-ahead options.
- **The goal can only be reached as well as the position is read**: about 1% of the reach,
  a centimetre for the TIAGo arm.
- **Losing product neurons** shifts the read-out pose (SO-3 results), and with it the pose
  that is taken for the goal.

## Idea 2: let the sheet plan the move

> **Built in part.** The goal that is written on the goal sheets travels to a new goal on a
> straight line at 0.3 reaches per second, so the arm is led and not asked to jump. But the
> travelling is done by whoever writes the goal: the packet is not pulled by two anchors on
> one sheet, and the sheet does not plan.

A packet pulled by an anchor goes to it in a straight line, at a speed the bowl sets
(architecture, section 4). So:

1. Take the forward anchor off the sheets (or weaken it) and switch on an anchor at the
   goal. The packets leave the current position and travel to the goal: an imagined move in
   a straight line.
2. The distance between the packet and the forward position, which the position cells
   still hold, is the error of idea 1, but for a goal that moves ahead of the arm at a
   bounded distance. The arm is led along the straight line and not asked to jump.
3. When the arm lags too far behind, the pull on the packet is reduced: the two anchors
   (goal and forward position) both act on the sheet, and their gains set how far the
   packet runs ahead.

This uses the attractor as a path generator and needs no new mechanism: two anchors with
different gains on the same sheets. To measure first: how straight and how even the
packet's path is under an anchor that is far away (the bowl is steep there; the gain may
have to be capped).

## Idea 3: path integration run backwards

> Not built. The network with the inverse-kinematics layer has no velocity path at all.

Path integration answers "the joints move like this; where does the packet go?". Planning
asks the reverse: "the packet should go there; which joints do that?" The same velocity
populations can be probed with one joint velocity at a time, without moving the arm: joint
k's test velocity gives a push on the packet, and the joints whose push points toward the
goal are the ones to move. This is idea 1 computed by trying instead of by a dot product.
It needs the sheets' response to a velocity to be even, and it is not yet (tables,
section 0: packet speed within a factor of about 1.5 of the asked speed). Idea 1 does not
depend on that.

## Idea 4: redundancy and limits

> Not built. It shows most where goals are far: with goals anywhere in the joint ranges
> the exact rule misses a quarter to two thirds of them ([`ik/results.md`](ik/results.md),
> sections 3 and 11).

A 7-joint arm has one more joint than a pose needs. Rule 1 picks one solution without
asking. Useful additions, each a weak extra drive on the goal rings:

- toward the middle of each joint's range (the bounded rings of the SO-2 repo already keep
  a bump from leaving the range);
- toward the last pose that worked for a similar goal (idea 5);
- away from poses in which a lever or an axis lines up with another (measured from the
  populations that exist: the levers and the axes).

## Idea 5: a learned map of poses, after RatSLAM's experience map

RatSLAM keeps a graph of places it has been, with the moves between them. The arm's
version: while the arm moves, store pairs of (pose: packets and orientation rings) and
(joint rings). Later a pose goal recalls the joint angles of the nearest stored pose as a
first guess, and idea 1 corrects the rest. A learned association from the sheets to the
joint goal rings (place cells to joint bumps) is the neural form of it; it also answers
"which of the many solutions" by "the one used before".

## Idea 6: places the end point must not enter

A region of a sheet that is inhibited cannot hold a packet. With idea 2 the packet then
goes round it. Three planes cannot describe every obstacle in space (a place is only
excluded if it is excluded in a plane), so this is for walls and table tops, not for a
general scene.

## Idea 7: a sequence of goals

The shift register of Lin, Yiu and Leibold (2026) keeps a sequence readable for a while.
Used the other way round it replays one: a list of goal positions handed to idea 2 one
after another gives a path through via-points.

## A first experiment

> Steps 1 and 2 were done in another form: 40 pairs per 4-joint arm for the position, 30
> per 7-joint arm for the pose, goals near the start pose, reached within 4% of the reach
> and 5° ([`ik/tests.md`](ik/tests.md)). Step 3 in part (idea 2). Step 4 is open.

1. The transpose rule (idea 1) for the position alone, with the kinematic robot of this
   repo and exact integration of the joint commands: 100 random start and goal pairs per
   model; reported: share of goals reached to within 2% of the reach, time to get there,
   the path's detour, against the same rule with exact kinematics.
2. Add the orientation term and repeat with full pose goals; find `w`.
3. Replace the direct goal by the travelling packet of idea 2 and compare paths.
4. Only then the joint controllers of the SO-2 repo in place of the kinematic robot.
