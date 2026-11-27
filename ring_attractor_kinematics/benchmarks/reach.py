"""Closed-loop reaching with the spiking network (B2–B6, and the joint-loss study), each trial paired with exact-loop twins.

Every spiking trial (the network, old method: the turn from the hand's matrix) is run next to twins on the
same start, goal and seed in the exact loop (:func:`ring_attractor_kinematics.rules.cascade_rule.simulate`, delay 0.3 s, share
noise 0.01 rad): the exact cascade, damped least squares, the Jacobian transpose, and the cascade with each
orientation source of :mod:`ring_attractor_kinematics.rules.observer_rule`.

=========== ============================================ ================================================
name        trials                                        conditions
=========== ============================================ ================================================
reach       30 pairs: TIAGo7 pose, generic7 pose,         κ = 1; kernel seeds 1–3 on TIAGo7
            TIAGo4 position (parity control)
pole        20 pairs whose goal has |pitch| > 75°         κ = 1
gain        30 pairs, TIAGo7 pose                         κ = 1.5, 2 (κ = 1 is ``reach``)
robust.     20 pairs, TIAGo7 pose                         a 0.3 rad push of one joint at 2 s; 100 pA noise current
gazebo      30 pairs, TIAGo7 pose, TIAGo in Gazebo        κ = 1
joint_loss  30 pairs, TIAGo7 pose (the ``reach`` pairs)   one joint disabled 2 s into the reach (see below)
gazebo_     the same 30 pairs, TIAGo in Gazebo            the same, on the arm's trajectory controller
joint_loss
=========== ============================================ ================================================

**Joint loss.** Each pair has one joint to disable, from :func:`~reaching_benchmark.targets.disabled_joints` (every
joint 4–5 times in 30 pairs). Every condition and seed disables the same joint for a given pair. The conditions, each a
run of its own (``--condition``):

* ``intact``: nothing disabled. This is the paired reference; its rows still record where the pair's joint was at onset;
* ``lock`` (actuator lock): the robot's joint jams at onset. The network is untouched and keeps handing it a share;
* ``motor`` (motor-neuron loss): the joint's gain pair and written goal ring are silenced at onset. The joint holds,
  and the cascade still hands it a share;
* ``allocation`` (allocation-neuron loss): the joint's two share halves are silenced at onset. The joint holds, and
  every later joint is handed the whole remainder;
* ``allocation_lock``: ``allocation`` with the actuator jammed as well (not part of the default set);
* ``loss`` (diffuse neuron loss): a fraction of the neurons of every population of some classes is silenced for the
  whole run (``--loss``, ``--loss-class``, ``--lesion-seed``). No joint is disabled.

In Gazebo (``gazebo_joint_loss``) the jammed joint's trajectory target stays at the angle it had at onset, so the
controller brakes it there (:meth:`~reaching_benchmark.plants.gazebo_robot.GazeboRobot.lock`). The lesions act on the network
exactly as on the light robot. Without ``--condition`` it runs ``lock``, ``motor`` and ``allocation``; the intact
reference is then the stored ``gazebo`` run, or ``--condition intact`` for one run with this code.

The lock and the lesions are undone when each trial ends. A row also records the joint-loss twins
(:func:`~ring_attractor_kinematics.benchmarks.twins.joint_loss_twins`) and whether the goal was still reachable with the joint frozen
where this trial froze it (:func:`~ring_attractor_kinematics.benchmarks.twins.reachable_after`). The raw file's ``lesion`` entry says
exactly which populations and how many neurons each condition silences.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import nest
import numpy as np

from reaching_benchmark.results import run_name
from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import Chain, get_chain
from reaching_benchmark.geometry.rotations import rotation_angle, wrap
from reaching_benchmark.progress import Progress, say, trial_detail
from reaching_benchmark.targets import TEST_SEED, PosePair, disabled_joints, pole_pairs, pose_pairs

from .. import results
from ..config import IkDesign, default_task
from ..control.session import DISABLE_CONDITIONS, Disable, PoseIkSession, joint_loss_fields, trial_row
from ..network.factory import build_pose_ik
from ..rules.references import exact_targets
from ..rules.cascade_rule import goal_matrix
from .command import VARIANT_NAMES
from .twins import joint_loss_twins, reachable_after, twins

PUSH = (2.0, 0.3)               # time (s), angle (rad)
NOISE_PA = 100.0
#: the joint-loss study: when the joint is disabled (s), and the conditions a run without ``--condition`` makes
JOINT_LOSS_ONSET_S = 2.0
JOINT_LOSS_CONDITIONS = ("intact", "lock", "motor", "allocation")
GAZEBO_JOINT_LOSS_CONDITIONS = ("lock", "motor", "allocation")
#: the benchmarks that disable a joint: on the light robot, and on the TIAGo in Gazebo
JOINT_LOSS_BENCHMARKS = ("joint_loss", "joint_loss_gazebo")


def plan(name: str, args: Any) -> List[Dict[str, Any]]:
    """The runs of one benchmark: arm, pairs, κ, kernel seed, options.  ``--variant predictive_encoding`` runs the new
    method's network, ``--variant observer`` the observer method's (pose tasks only: on a position task it would change
    nothing)."""

    runs = _plan(name, args)
    if args.method in ("lead", "observer"):
        if args.method == "observer":
            runs = [run for run in runs if default_task(get_chain(run["arm"]).n) != "position"]
        for run in runs:
            run["network_method"] = args.method
    return runs


def _plan(name: str, args: Any) -> List[Dict[str, Any]]:
    pairs = args.pairs
    if name == "reach_kinematic":
        arms = [args.arm] if args.arm else ["tiago7", "generic7", "tiago4"]
        return [{"arm": arm, "pairs": pairs or 30, "kappa": 1.0, "seed": args.seed, "kind": "near"} for arm in arms]
    if name == "reach_pole":
        arms = [args.arm] if args.arm else ["tiago7", "generic7"]
        return [{"arm": arm, "pairs": pairs or 20, "kappa": 1.0, "seed": args.seed, "kind": "pole"} for arm in arms]
    if name == "reach_gain":
        kappas = [float(args.condition.split("=")[1])] if args.condition else [1.5, 2.0]
        return [{"arm": args.arm or "tiago7", "pairs": pairs or 30, "kappa": k, "seed": args.seed, "kind": "near"} for k in kappas]
    if name == "reach_disturbance":
        conditions = [args.condition] if args.condition else ["push", "noise"]
        return [{"arm": args.arm or "tiago7", "pairs": pairs or 20, "kappa": 1.0, "seed": args.seed, "kind": "near", "condition": c}
                for c in conditions]
    if name == "reach_gazebo":
        return [{"arm": "tiago7", "pairs": pairs or 30, "kappa": 1.0, "seed": args.seed, "kind": "near", "robot": "gazebo",
                 "method": "ring"}]                                            # lead, observer: below
    if name == "joint_loss":
        return [joint_loss_run(args, condition) for condition in (args.condition.split(",") if args.condition else JOINT_LOSS_CONDITIONS)]
    if name == "joint_loss_gazebo":
        conditions = args.condition.split(",") if args.condition else GAZEBO_JOINT_LOSS_CONDITIONS
        return [dict(joint_loss_run(args, condition), arm="tiago7", robot="gazebo") for condition in conditions]
    raise ValueError(name)


def joint_loss_run(args: Any, condition: str) -> Dict[str, Any]:
    run = {"arm": args.arm or "tiago7", "pairs": args.pairs or 30, "kappa": 1.0, "seed": args.seed, "kind": "near",
           "condition": condition, "onset_s": float(getattr(args, "onset", None) or JOINT_LOSS_ONSET_S)}
    if condition == "loss":
        fraction, classes, lesion_seed = float(args.loss), str(args.loss_class), int(args.lesion_seed)
        run.update(loss=fraction, loss_class=classes, lesion_seed=lesion_seed,
                   condition="loss%g_%s_l%d" % (fraction, classes.replace(",", "+"), lesion_seed))
    elif condition != "intact" and condition not in DISABLE_CONDITIONS:
        raise ValueError("joint_loss conditions: intact, %s, loss" % ", ".join(DISABLE_CONDITIONS))
    return run


def part_name(run: Dict[str, Any], task: str, shard: Tuple[int, int]) -> str:
    """The run file of one run (:func:`reaching_benchmark.results.run_name`): arm, task, κ, seed, then the variant
    (``predictive_encoding``, ``observer``) and the condition when not the default."""

    variant = VARIANT_NAMES.get(run.get("network_method", "matrix"))
    return run_name(run["arm"], run["seed"], task=task, kappa=run["kappa"], variant=variant, condition=run.get("condition"),
                    shard=shard)


def pairs_for(chain: Chain, run: Dict[str, Any]) -> List[PosePair]:
    return pole_pairs(chain, run["pairs"]) if run["kind"] == "pole" else pose_pairs(chain, run["pairs"], TEST_SEED)


def condition_record(network: Any, run: Dict[str, Any], loss: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What a joint-loss run disables and how (the raw file's ``lesion`` entry); ``None`` for the other benchmarks."""

    condition = run.get("condition")
    if loss is not None:
        return {"condition": "loss", **loss}
    if condition == "intact":
        return {"condition": "intact", "disabled": "nothing: the paired reference"}
    if condition not in DISABLE_CONDITIONS:
        return None
    disable = Disable(run["onset_s"], 0, condition)
    record: Dict[str, Any] = {"condition": condition, "onset_s": run["onset_s"],
                              "joint": "one per pair: tasks.disabled_joints (row field disabled_joint, counted from 0)",
                              "until": "the end of each trial; undone before the next placement"}
    if disable.locks:
        record["robot"] = ("actuator jammed (LightRobot.lock): the joint's commanded motion is ignored and its angle stays where it "
                           "was at onset; the network still sees that true angle")
    if disable.site is not None:
        chain = network.chain
        per_joint = {}
        for k in range(chain.n):
            targets = network.joint_lesion_targets(k, disable.site)
            per_joint[str(k + 1)] = {"populations": [name for name, _, _ in targets], "neurons": int(sum(size for _, _, size in targets))}
        record["network"] = {"site": disable.site, "per_joint": per_joint, "network_neurons": network.neuron_count(),
                             "how": "threshold of every neuron raised to 1e12 mV at onset (no input makes it fire); each neuron's "
                                    "threshold and potential at onset written back when the trial ends",
                             "takes_effect": "at once for the share halves; the written goal ring's 3-tick window still holds older "
                                             "spikes, so the target goes missing (motor) or settles on the joint (allocation) within 0.15 s"}
    return record


def tick_progress(progress: Progress, session: PoseIkSession, pair: PosePair, design: IkDesign) -> Callable[[Any], None]:
    """``on_tick`` for :meth:`PoseIkSession.reach`: trial time and the hand's errors on the progress bar."""

    chain, goal = session.chain, goal_matrix(pair.goal)
    ticks = [0]

    def on_tick(_: Any) -> None:
        ticks[0] += 1
        q = session.q
        degrees = float(np.degrees(rotation_angle(chain.rotation(q), goal))) if design.orientation else None
        mm = 1000.0 * float(np.linalg.norm(kinematics.position(chain, q) - pair.goal_position)) if design.position else None
        progress.update(trial_detail(ticks[0] * session.dt, design.trial_s, degrees, mm))
    return on_tick


def run_one(backend: Any, name: str, run: Dict[str, Any], shard: Tuple[int, int], max_minutes: Optional[float], threads: int,
            make_robot: Optional[Callable[[Chain], Any]] = None, log: Optional[Callable[[str], None]] = None) -> None:
    """One run, checkpointed to results/ring_attractor_kinematics/<name>/<part>.json after every trial.  ``make_robot``: another robot (Gazebo).
    ``log``: where each trial's line also goes (it is always shown on the progress bar's stream).

    Resumes: pairs already in that file are kept and not run again (a run stopped part-way goes on where it was)."""

    started = time.time()
    chain = get_chain(run["arm"])
    design = IkDesign(task=default_task(chain.n), kappa=run["kappa"])
    part = part_name(run, design.task, shard)
    earlier = results.read_raw_part(name, part)
    rows: List[Dict[str, Any]] = list(earlier.get("rows", []))
    done = {row["index"] for row in rows}
    every = pairs_for(chain, run)
    mine = [pair for pair in every if pair.index % shard[1] == shard[0]]
    chosen = [pair for pair in mine if pair.index not in done]
    if not chosen:
        say("%s %s: all %d pairs done" % (name, part, len(rows)))
        return
    if rows:
        say("%s %s: resuming after %d pairs" % (name, part, len(rows)))
    options: Dict[str, Any] = {"threads": threads, "seed": run["seed"]}
    if run.get("condition") == "noise":
        options["noise_pa"] = NOISE_PA
    _, network = build_pose_ik(backend, run["arm"], design, method=run.get("network_method", "matrix"), **options)
    loss = network.lesion_fraction(run["loss"], run["lesion_seed"], run["loss_class"].split(",")) if run.get("loss") else None
    lesion = condition_record(network, run, loss)
    session = PoseIkSession(network, robot=make_robot(chain) if make_robot is not None else None)
    if run.get("method", "ring") != "ring":
        session.method = exact_targets(run["method"])
    session.warm_up()
    pushed = np.random.default_rng([TEST_SEED, 77]).integers(chain.n, size=len(every))
    joint_loss = name in JOINT_LOSS_BENCHMARKS
    disabled = disabled_joints(chain, len(every)) if joint_loss else None
    condition = run.get("condition")
    progress = Progress("%s %s" % (name, part), len(mine), len(rows), earlier.get("wall_s", 0.0), sum(bool(r["reached"]) for r in rows))
    skipped: List[int] = []
    for position, pair in enumerate(chosen):
        if max_minutes is not None and (time.time() - started) / 60.0 > max_minutes:
            skipped = [p.index for p in chosen[position:]]
            break
        if hasattr(session.robot, "link"):
            session.robot.link.put_base_back()
        session.place(pair.start)
        push = (PUSH[0], int(pushed[pair.index]), PUSH[1]) if condition == "push" else None
        joint = int(disabled[pair.index]) if disabled is not None else None
        disable = Disable(run["onset_s"], joint, condition) if joint_loss and condition in DISABLE_CONDITIONS else None
        progress.start("%d" % pair.index + (" (joint %d %s)" % (joint + 1, condition) if disable is not None else ""))
        trial = session.reach(pair.goal, pair.goal_position, push=push, disable=disable,
                              on_tick=tick_progress(progress, session, pair, design))
        row = trial_row(trial, design)
        row.update(index=pair.index, start=pair.start.tolist(), goal_lpy=pair.goal.tolist(), goal_position=pair.goal_position.tolist(),
                   start_off_rad=float(np.max(np.abs(wrap(trial.q[0] - pair.start)))), pushed_joint=push[1] if push else None)
        if joint_loss:
            fields = joint_loss_fields(trial, joint, run["onset_s"])
            row.update(fields, condition=condition, twins=joint_loss_twins(chain, pair, design, run["onset_s"], joint),
                       reachability=reachable_after(chain, fields["q_onset"], joint, pair, design))
            reference = "exact cascade %s, informed %s; goal %s with joint %d frozen" % (
                "reached" if row["twins"]["cascade_lock"]["reached"] else "missed",
                "reached" if row["twins"]["cascade_informed"]["reached"] else "missed",
                "reachable" if row["reachability"]["reachable"] else "not reachable", joint + 1)
        else:
            row.update(twins=twins(chain, pair, design, push))
            reference = "exact cascade %s" % ("reached" if row["twins"]["cascade"]["reached"] else "missed")
        rows.append(row)
        rows.sort(key=lambda r: r["index"])
        line = "%s %s pair %d: %.0f° %.0f%% → %.1f° %.1f%% (%s) in %.1f s; %s | %d done, %.0f min" % (
            name, part, pair.index, row["start_error_deg"], 100 * row["start_error_reach"], row["final_error_deg"],
            100 * row["final_error_reach"], "reached" if row["reached"] else "missed", row["duration_s"], reference, len(rows),
            (time.time() - started) / 60.0)
        progress.finish(bool(row["reached"]), line)
        if log is not None:
            log(line)
        document = {"benchmark": name, "run": run, "design": design.describe(), "shard": list(shard), "planned": len(mine),
                    "neurons": network.neuron_count(), "synapses": int(backend.GetKernelStatus("num_connections")),
                    "simulated_s": earlier.get("simulated_s", 0.0) + network.t_ms / 1000.0,
                    "wall_s": earlier.get("wall_s", 0.0) + time.time() - started, "robot": session.robot.describe(), "rows": rows,
                    "skipped": skipped + [p.index for p in chosen[position + 1:]]}
        if lesion is not None:
            document["lesion"] = lesion
        if disabled is not None:
            document["disabled_joints"] = [int(j) for j in disabled]
        results.write_raw(name, part, document)
    progress.close()


def run_command(args: Any) -> None:
    shard = tuple(int(v) for v in args.shard.split("/"))
    for run in plan(args.name, args):
        run_one(nest, args.name, run, (shard[0], shard[1]), args.max_minutes, args.threads)
