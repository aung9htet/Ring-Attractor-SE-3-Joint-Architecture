#!/usr/bin/env python3
"""Who fires, and how often, in the pose inverse-kinematics network; and the sheets' level cells at work.

    python3 -B scripts/ik_spike_census.py                       # tiago4 with a position goal: docs/ik/spikes/
    python3 -B scripts/ik_spike_census.py --chain tiago7 --only census --out /out/spikes_tiago7

Writes ``census.json`` and ``census.md`` (per kind of population: how many there are, how many of
their neurons fire, how fast, how many spikes a 50 ms tick carries; once while the hand is asked
to move and once at its goal), ``figures.json`` (the numbers quoted with the figure) and
``docs/ik/figures/spikes_sheets.png`` (a packet whose anchor is suddenly 30 pA too low, without and
with level cells, and what the ramp readout makes of it).  TECHNICAL.md, part III, quotes these.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.design import Design  # noqa: E402
from ring_so3.kinematics import lift_pitch_yaw  # noqa: E402
from ring_se3 import kinematics  # noqa: E402
from ring_se3.ik import experiments  # noqa: E402
from ring_se3.ik.design import PoseIkDesign, TASKS  # noqa: E402
from ring_se3.ik.pairs import DEVELOPMENT_SEED, pairs  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TICK_S = 0.05
#: ticks a readout is averaged over where a figure shows what it gives (the network's own readout cells smooth with 50 ms)
SMOOTH_TICKS = 5
#: after the step, the anchored place travels as a goal does: this far per second (scaled units: reaches), for this many
#: ticks, along this direction
FOLLOW_SPEED = 0.3
FOLLOW_TICKS = 30
FOLLOW_DIRECTION = (-1.0, 1.0, -0.5)
INK, GREY, LIGHT, BLUE, ORANGE = "#1c2330", "#8a94a3", "#d9dee6", "#2a78d6", "#eb6834"


def events(nest, population):
    """Spike times (ms) per neuron of a population that was built with ``record=True``, since the last call."""

    data = nest.GetStatus(population.recorder, "events")[0]
    senders, times = np.asarray(data["senders"], dtype=int) - population.first_id, np.asarray(data["times"], dtype=float)
    nest.SetStatus(population.recorder, {"n_events": 0})
    return [times[senders == i] for i in range(population.n)]


def kind_of(node):
    """The row of the census a node of the wirings belongs to."""

    head = node[0]
    return {"mul": "product, pose network", "pos": "sum: position cell", "prod": "product, orientation of the links",
            "entry": "sum: entry of a link's orientation", "g": "goal orientation (products and sums)", "ikmul": "product, comparison and cascade",
            "turn": "sum: the turn e", "offset": "sum: the offset d", "lever": "sum: lever of a joint", "jac": "sum: direction c of a joint",
            "share": "share half", "rest": "sum: what is left of the turn", "restp": "sum: what is left of the offset"}.get(head, head)


class Census:
    def __init__(self, nest, network):
        self.nest, self.network = nest, network
        n = network
        self.recorded = []                              # (kind, population)
        for population in n.features:
            self.recorded.append(("cos / sin of a joint", population))
        for node, population in list(n.products.items()) + list(n.entries.items()) + list(n.muls.items()) + list(n.relays.items()) + list(n.ik_populations.items()):
            self.recorded.append((kind_of(node), population))
        self.recorded += [("gate of the forward sheets", p) for p in n.gates] + [("level cells of a sheet", p) for p in n.level_cells + n.goal_level_cells]
        self.recorded += [("goal gate", p) for p in n.goal_gates] + [("cos / sin of a goal angle", p) for p in n.goal_features]
        self.recorded += [("ramp readout", p) for p in n.ramp_readouts.values()]
        if hasattr(n, "pitch_norm"):
            self.recorded.append(("length of a pair (cos of pitch)", n.pitch_norm))
        self.bias_recorder = nest.Create("spike_recorder")
        nest.Connect(n.bias_cells, self.bias_recorder)
        if n.goal_bias_cells is not None:
            nest.Connect(n.goal_bias_cells, self.bias_recorder)

    def run(self, session, ticks):
        """Hold the pose for ``ticks`` ticks with the session's written goal; returns the rows of the census."""

        nest, n = self.nest, self.network
        for _, population in self.recorded:
            events(nest, population)
        nest.SetStatus(self.bias_recorder, {"n_events": 0})
        for goal in n.joint_goal_rings:
            goal.gain_counts(nest)                          # their counters run since they were last read
        placed = {"joint ring": [], "forward sheet": [], "goal sheet": [], "output ring": [], "goal ring": [], "goal ring of a joint": [], "gain pair, one side": []}
        shares, leads, offsets = [], [], []
        for _ in range(int(ticks)):
            tick = n.step_ik(session.q, session.written_lpy, session.written_position)
            placed["joint ring"].append(n.ring_counts())
            placed["forward sheet"].append(tick.tick.sheet_counts)
            if n.goal_sheets:
                placed["goal sheet"].append(tick.goal_sheet_counts)
            placed["output ring"].append(n.output_counts())
            if n.goal_rings:
                placed["goal ring"].append(n.goal_counts())
            placed["goal ring of a joint"].append(n.joint_goal_counts())
            placed["gain pair, one side"].append(np.array([goal.gain_counts(nest) for goal in n.joint_goal_rings]).reshape(-1))
            shares.append(tick.shares), leads.append(tick.lead), offsets.append(tick.offset)
        seconds = ticks * TICK_S
        rows = []
        for kind, stack in placed.items():
            if not stack:
                continue
            counts = np.sum(stack, axis=0)                              # (populations, neurons), or (populations,) for the gain pairs
            if kind == "gain pair, one side":
                rows.append({"kind": kind, "populations": int(counts.size), "neurons": int(n.rings[0].size), "fire": None, "rate_hz": None,
                             "spikes_per_tick": float(counts.mean() / ticks), "gaps_vary": None})
                continue
            fire = counts / seconds >= 1.0
            rows.append({"kind": kind, "populations": int(counts.shape[0]), "neurons": int(counts.shape[1]),
                         "fire": float(fire.sum(axis=1).mean()), "rate_hz": float(counts[fire].mean() / seconds) if fire.any() else 0.0,
                         "spikes_per_tick": float(counts.sum(axis=1).mean() / ticks), "gaps_vary": None})
        by_kind = {}
        for kind, population in self.recorded:
            by_kind.setdefault(kind, []).append(events(nest, population))
        for kind, populations in by_kind.items():
            fire, rates, spikes, spread = [], [], [], []
            for trains in populations:
                counts = np.array([len(train) for train in trains])
                active = counts / seconds >= 1.0
                fire.append(active.sum())
                rates += list(counts[active] / seconds)
                spikes.append(counts.sum() / ticks)
                for train in trains:
                    if len(train) >= 5:
                        gaps = np.diff(train)
                        spread.append(np.std(gaps) / np.mean(gaps))
            rows.append({"kind": kind, "populations": len(populations), "neurons": len(populations[0]), "fire": float(np.mean(fire)),
                         "rate_hz": float(np.mean(rates)) if rates else 0.0, "spikes_per_tick": float(np.mean(spikes)),
                         "gaps_vary": float(np.median(spread)) if spread else None})
        cells = n.bias_cells.tolist() + (n.goal_bias_cells.tolist() if n.goal_bias_cells is not None else [])
        rows.append({"kind": "bias cell of the anchors", "populations": 2 if n.goal_bias_cells is not None else 1, "neurons": 16, "fire": 16.0,
                     "rate_hz": float(nest.GetStatus(self.bias_recorder, "n_events")[0]) / (len(cells) * seconds),
                     "spikes_per_tick": float(nest.GetStatus(self.bias_recorder, "n_events")[0]) / (ticks * (2 if n.goal_bias_cells is not None else 1)),
                     "gaps_vary": None})
        total = sum(row["spikes_per_tick"] * row["populations"] for row in rows)
        return {"rows": sorted(rows, key=lambda r: r["kind"]), "spikes_per_tick": total, "spikes_per_second": total / TICK_S,
                "shares_rad": np.mean(shares, axis=0).tolist(), "lead_indices": np.nanmean(leads, axis=0).tolist(),
                "offset_reach": np.mean(offsets, axis=0).tolist()}


def census(nest, chain_name, task, threads, seconds=2.0):
    design = PoseIkDesign(task=task or experiments.default_task(experiments.get_chain(chain_name)))
    chain, network = experiments.build_ik(nest, chain_name, design, threads=threads, record=True)
    from ring_se3.ik.session import PoseIkSession

    session = PoseIkSession(network)
    session.warm_up()
    pair = pairs(chain, 1, DEVELOPMENT_SEED, experiments.SPREAD)[0]
    session.place(pair.start)
    counter = Census(nest, network)
    ticks = int(round(seconds / TICK_S))
    at_goal = counter.run(session, ticks)                                # the goal is on the hand: nothing is asked
    part = pair.start + 0.3 * (pair.goal_pose - pair.start)              # a goal part of the way to another pose, written at once
    session.give(lift_pitch_yaw(chain.rotation(part)), kinematics.position(chain, part))
    for _ in range(40):                                                  # the written goal travels there; the arm is held
        session._travel()
        network.step_ik(session.q, session.written_lpy, session.written_position)
    asked = counter.run(session, ticks)
    return {"chain": chain.name, "task": design.task, "neurons": network.neuron_count(), "ik_neurons": network.ik_neuron_count(),
            "seconds": seconds, "threads": threads, "pair": int(pair.index), "asked": asked, "at_goal": at_goal,
            "true_offset_reach": ((session.written_position - kinematics.position(chain, session.q)) / network.space.radius).tolist()}


def census_markdown(document):
    lines = ["# Spike census of the pose inverse-kinematics network", "",
             "Written by `scripts/ik_spike_census.py`. %s with a %s goal, %s neurons (%s of them the inverse-kinematics layer), held at one pose for %.0f s of network time; "
             "\"fire\": at least one spike per second; a tick is 50 ms." % (document["chain"], document["task"], format(document["neurons"], ","),
                                                                      format(document["ik_neurons"], ","), document["seconds"]), ""]
    for key, title in (("asked", "While the hand is asked to move (a goal part of the way to another pose)"), ("at_goal", "At the goal (nothing is asked)")):
        part = document[key]
        lines += ["## %s" % title, "", "| Kind of population | How many | Neurons in each | Fire | Rate of those that fire | Spikes per tick, each | Gaps vary by |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in part["rows"]:
            lines.append("| %s | %d | %d | %s | %s | %.0f | %s |" % (
                row["kind"], row["populations"], row["neurons"], "–" if row["fire"] is None else "%.0f" % row["fire"],
                "–" if row["rate_hz"] is None else "%.0f Hz" % row["rate_hz"], row["spikes_per_tick"],
                "–" if row["gaps_vary"] is None else "%.0f%%" % (100 * row["gaps_vary"])))
        lines += ["", "All together %s spikes per tick, %s per second: %.1f per neuron and second. Shares (deg): %s. Lead of the goal rings (indices): %s. "
                  "Offset held (%% of the reach): %s." % (
                      format(int(round(part["spikes_per_tick"])), ","), format(int(round(part["spikes_per_second"])), ","), part["spikes_per_second"] / document["neurons"],
                      ", ".join("%+.1f" % np.degrees(v) for v in part["shares_rad"]), ", ".join("%+.1f" % v for v in part["lead_indices"]),
                      ", ".join("%+.1f" % (100 * v) for v in part["offset_reach"])), ""]
    return "\n".join(lines)


def weights():
    """How large one spike is at the new synapses (pA), worked out from the tunings and the stored readouts (no NEST)."""

    from ring_so3.nest import ik_calibrate as so3_ik_calibrate
    from ring_se3.nest import ik_calibrate
    from ring_se3.nest.sheet import ANCHOR_TAU_MS, LevelDesign, level_tuning
    from ring_se3.sheet import SheetDesign, SheetMap

    design, mapping, level = Design(), SheetMap(SheetDesign()), LevelDesign()
    ramp = ik_calibrate.load(design)
    tuning = ramp.tuning(design)
    into_ramp = tuning.input_weights(np.eye(2)) @ ik_calibrate.ramp_inputs(mapping, 0, ramp.total_rate_hz) / tuning.model.charge
    cells = level_tuning(design, level.tau_ms)
    into_level = cells.input_weights(np.ones((1, 1)))[:, 0] / (level.set_rate_hz * level.span) / cells.model.charge
    bounded = so3_ik_calibrate.load(design).decoders["bounded"][:, 0]
    onto_sheet = -level.gain_pa * bounded / float(np.e * ANCHOR_TAU_MS * 1e-3)

    def sizes(values):
        values = np.abs(np.asarray(values, dtype=float)).ravel()
        return {"median_pa": float(np.median(values)), "largest_pa": float(values.max())}

    return {"sheet_to_ramp": sizes(into_ramp), "sheet_to_level": sizes(into_level), "level_to_sheet": sizes(onto_sheet),
            "ramp_total_rate_hz": ramp.total_rate_hz, "ramp_pulse_ms": float(np.e * tuning.model.tau_syn),
            "ramp_pulses_at_once": float(ramp.total_rate_hz * np.e * tuning.model.tau_syn * 1e-3),
            "level_pulse_ms": float(np.e * level.tau_ms), "level_constant_pa": sizes(cells.input_weights(np.array([[1.0 / level.span]]))[:, 0])}


def sheets_figure(nest, path, threads, place=(0.35, -0.25, 0.4), step_pa=-30.0):
    """Three sheets held at one place; from 1 s to 3 s every sheet neuron gets ``step_pa`` (the anchor is that much too low)."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ring_so3.nest import calibrate as so3_calibrate
    from ring_so3.nest import ik_calibrate as so3_ik_calibrate
    from ring_so3.nest.populations import connect_weights, create_population, decoded_weights, set_value
    from ring_so3.nest.ring import RING_NEURON
    from ring_se3.nest import calibrate, ik_calibrate
    from ring_se3.nest.sheet import LevelDesign, build_sheet, connect_anchor, connect_anchor_bias, connect_level
    from ring_se3.sheet import PLANES, SheetDesign, SheetMap

    design, mapping = Design(), SheetMap(SheetDesign())
    base, pose = so3_calibrate.load(design), calibrate.load(design)
    bounded, ramp = so3_ik_calibrate.load(design).decoders["bounded"], ik_calibrate.load(design)
    place = np.asarray(place, dtype=float)
    ticks, on, off = 100, 20, 60
    results = {}
    for name, level in (("without", None), ("with", LevelDesign())):
        calibrate._kernel(nest, threads, 1)
        sources = [create_population(nest, design.entry, "source%d" % i, seed=50 + i) for i in range(3)]
        gates = [create_population(nest, design.entry, "gate%d" % i, seed=60 + i) for i in range(3)]
        for source, gate, value in zip(sources, gates, place):
            connect_weights(nest, source.neurons, gate.neurons, decoded_weights(base.decoders["source"], np.ones((1, 1)), gate))
            set_value(nest, source, [float(value)])
        bias_cells = calibrate.create_bias_cells(nest, design)
        sheets, cells = [], []
        for index, (plane, first, second) in enumerate(PLANES):
            sheet = build_sheet(nest, plane, mapping)
            connect_anchor(nest, sheet, [gates[first].neurons, gates[second].neurons], pose.decoders["gate"])
            connect_anchor_bias(nest, sheet, bias_cells, float(pose.fit["bias_rate_hz"]))
            if level is not None:
                cells.append(connect_level(nest, sheet, design, bounded, level, seed=90 + index, record=True))
            sheets.append(sheet)
        readouts = [ik_calibrate.create_ramp_readout(nest, ramp.tuning(design), sheets, i, ramp.total_rate_hz, "ramp%d" % i, seed=70 + i, record=True) for i in range(3)]
        for _ in range(30):
            nest.Simulate(50.0)
        for population in readouts + cells:
            population.counts(nest)
        for sheet in sheets:
            sheet.read(nest)
        rate, read, centre, held = [], [], [], []
        for tick in range(ticks):
            if tick in (on, off):
                for sheet in sheets:
                    nest.SetStatus(sheet.neurons, {"I_e": float(RING_NEURON["I_e"]) + (step_pa if tick == on else 0.0)})
            nest.Simulate(50.0)
            counts = np.array([sheet.read(nest) for sheet in sheets])
            rate.append(counts.sum(axis=1) / TICK_S)
            centre.append([ik_calibrate.pooled_centre(mapping, counts, i) for i in range(3)])
            read.append([float(readout.counts(nest) / TICK_S @ ramp.decoders["ramp"][:, 0]) for readout in readouts])
            held.append([float(cell.counts(nest) / TICK_S @ bounded[:, 0]) for cell in cells] if cells else [np.nan] * 3)
        results[name] = {"rate": np.array(rate), "read": np.array(read), "centre": np.array(centre), "held": np.array(held)}
        if level is not None:                       # then the place travels as a goal does: how far behind are packets and readouts?
            direction = np.array(FOLLOW_DIRECTION) / np.linalg.norm(FOLLOW_DIRECTION)
            at, written, packets, seen = place.copy(), [], [], []
            for tick in range(2 * FOLLOW_TICKS):
                if tick < FOLLOW_TICKS:
                    at = at + direction * FOLLOW_SPEED * TICK_S
                    for source, value in zip(sources, at):
                        set_value(nest, source, [float(value)])
                nest.Simulate(50.0)
                counts = np.array([sheet.read(nest) for sheet in sheets])
                written.append(at.copy())
                packets.append([ik_calibrate.pooled_centre(mapping, counts, i) for i in range(3)])
                seen.append([float(readout.counts(nest) / TICK_S @ ramp.decoders["ramp"][:, 0]) for readout in readouts])
            moving = slice(FOLLOW_TICKS // 3, FOLLOW_TICKS)

            def behind(rows):
                """Median, over the ticks of steady travel, of how far along the path a reading is behind what is written (s)."""

                return float(np.median((np.array(written)[moving] - np.array(rows)[moving]) @ direction) / FOLLOW_SPEED)

            follow = {"speed_units_per_s": FOLLOW_SPEED, "travelled_units": FOLLOW_SPEED * FOLLOW_TICKS * TICK_S,
                      "packets_behind_s": behind(packets), "readouts_behind_s": behind(seen),
                      "readouts_off_at_rest": float(np.linalg.norm(np.mean(np.array(seen)[-10:], axis=0) - at)),
                      "readout_sd_from_tick_to_tick": float(np.median(np.std(np.array(seen)[-20:], axis=0)))}

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                         "xtick.color": GREY, "ytick.color": GREY, "text.color": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                         "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
    t = (np.arange(ticks) + 1) * TICK_S
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.0), sharex=True)
    series = (("without", "without level cells", GREY), ("with", "with level cells", BLUE))
    def off_place(read):
        """Distance of what the three readouts give from the anchored place, each smoothed over 0.25 s (reaches)."""

        kernel = np.ones(SMOOTH_TICKS) / SMOOTH_TICKS
        smooth = np.stack([np.convolve(read[:, i], kernel, mode="same") for i in range(3)], axis=1)
        return np.linalg.norm(smooth - place[None, :], axis=1)

    for name, label, color in series:
        axes[0].plot(t, results[name]["rate"].mean(axis=1), color=color, lw=2, label=label, solid_capstyle="round")
        axes[2].plot(t[2:-2], 100.0 * off_place(results[name]["read"])[2:-2], color=color, lw=2, label=label, solid_capstyle="round")
    axes[1].plot(t, -LevelDesign().gain_pa * results["with"]["held"].mean(axis=1), color=BLUE, lw=2, solid_capstyle="round")
    for ax in axes:
        ax.axvspan(on * TICK_S, off * TICK_S, color=LIGHT, alpha=0.45, lw=0)
    axes[0].set_ylabel("spikes per second\nof one sheet's packet")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("The anchor of three sheets is %.0f pA too low from 1 s to 3 s (shaded)" % abs(step_pa), loc="left")
    axes[0].legend(loc="lower right", ncol=2)
    axes[1].set_ylabel("current the level cells\nadd to every neuron (pA)")
    axes[2].set_ylabel("ramp readouts: distance from\nthe anchored place (% of reach)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_xlabel("time (s)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    during, before = slice(on + 20, off), slice(0, on)
    out = {"place": place.tolist(), "step_pa": step_pa}
    for name in ("without", "with"):
        r = results[name]
        away = off_place(r["read"])
        out[name] = {"rate_before_hz": float(r["rate"][before].mean()), "rate_during_hz": float(r["rate"][during].mean()),
                     "read_error_before": float(np.median(away[2:on])), "read_error_during": float(np.median(away[during])),
                     "read_error_largest": float(np.max(away[2:-2]))}
    out["with"]["added_before_pa"] = float(-LevelDesign().gain_pa * results["with"]["held"][before].mean())
    out["with"]["added_during_pa"] = float(-LevelDesign().gain_pa * results["with"]["held"][during].mean())
    recovered = np.flatnonzero(results["with"]["rate"][on:off].mean(axis=1) >= 0.9 * out["with"]["rate_before_hz"])
    out["with"]["back_at_nine_tenths_after_s"] = float((recovered[0] + 1) * TICK_S) if recovered.size else None
    out["follow"] = follow
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago4")
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--only", choices=("census", "figure"), default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=os.path.join(ROOT, "docs", "ik", "spikes"))
    args = parser.parse_args(argv)
    import nest

    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    if args.only != "figure":
        document = census(nest, args.chain, args.task, args.threads)
        with open(os.path.join(args.out, "census.json"), "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=1)
        with open(os.path.join(args.out, "census.md"), "w", encoding="utf-8") as stream:
            stream.write(census_markdown(document))
        print("census of %s: %d neurons, %.0f spikes per tick while a move is asked (%.0f s)" % (
            document["chain"], document["neurons"], document["asked"]["spikes_per_tick"], time.time() - started), flush=True)
    if args.only != "census":
        figure = os.path.join(ROOT, "docs", "ik", "figures", "spikes_sheets.png") if args.out == os.path.join(ROOT, "docs", "ik", "spikes") else os.path.join(args.out, "spikes_sheets.png")
        numbers = sheets_figure(nest, figure, args.threads)
        numbers["weights"] = weights()
        with open(os.path.join(args.out, "figures.json"), "w", encoding="utf-8") as stream:
            json.dump(numbers, stream, indent=1)
        print("figure: %s (%.0f s)" % (os.path.relpath(figure), time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
