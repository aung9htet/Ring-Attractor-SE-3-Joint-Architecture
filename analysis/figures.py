"""Every data figure of the paper (main Figs 1-5 and the supplementary figures drawn from runs), from ``results/``.

Never simulates neurons or robots: it reads the run files the models' benchmarks wrote (``results/<model>/<benchmark>/``)
and the shared data that ``python3 -m analysis derive`` and ``python3 -m analysis activity`` put in
``analysis/source_data/``. Figures go to ``analysis/figures/`` (PDF, SVG, PNG) and the numbers behind every panel to
``analysis/source_data/plotexports/``.

Units of analysis are kept explicit: target pairs within a seed, seeds as separate realizations, and ticks within a
trajectory as repeated measurements. All closed-loop trials are scored at 5% of reach and 5 degrees.
"""
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from reaching_benchmark.geometry.chains import get_chain, workspace
from reaching_benchmark.geometry.kinematics import position
from reaching_benchmark.geometry.rotations import from_lift_pitch_yaw, lift_pitch_yaw, rotation_angle, wrap
from reaching_benchmark.limits import travel, travel_position
from reaching_benchmark.statistics import mcnemar, paired, censored, success_summary
from ring_attractor_kinematics.neurons.ring_layout import RingMap

from .common import DATA, EXPORT, FIGURES as FIG, HASHES, RAK, RESULTS, ensure_dirs, read, relative

ensure_dirs()
CHAIN = get_chain("tiago7")
RADIUS_MM = 805.74
#: a target is reached when, at the last sample, theta <= 5 deg and ||dp|| <= 5% of reach (40.3 mm on the TIAGo 7-DoF arm)
TOL_FRAC = .05
TOL_DEG, TOL_MM = 5.0, TOL_FRAC * RADIUS_MM
LIMIT_S = 25.0
ONSET_S = 2.0
# One visual key throughout (guide): shape as well as colour.
C = {"matrix": "#3A3F44", "lead": "#2166AC", "dls": "#7A7A7A", "observer": "#E08214", "goal": "#B8860B",
     "intact": "#3A3F44", "lock": "#8C510A", "motor": "#C51B7D", "allocation": "#1B7837", "diffuse": "#4D4D4D"}
MARK = {"matrix": "o", "lead": "^", "dls": "D", "observer": "s",
        "intact": "o", "lock": "s", "motor": "v", "allocation": "D", "diffuse": "X"}
#: Fig. 3's controllers, all on the same 30 Gazebo targets
CONTROLLERS = [("rak", "RAK (TIAGo 7-DoF)"), ("rak_pe", "RAK, predictive encoding (TIAGo 7-DoF)"), ("dls_ik", "DLS IK (TIAGo 7-DoF)"), ("ccd_ik", "CCD IK (TIAGo 7-DoF)"),
               ("mpc_nn_dynamics", "NN dynamics + MPC (TIAGo 7-DoF)"), ("pets", "PETS (TIAGo 7-DoF)")]
RAK_KEYS = ("rak", "rak_pe")
SHORT_CTRL = {"rak": "RAK", "rak_pe": "RAK, predictive enc.", "dls_ik": "DLS IK", "ccd_ik": "CCD IK", "mpc_nn_dynamics": "NN dynamics + MPC", "pets": "PETS"}
CTRL_COLOUR = {"rak": "#3A3F44", "rak_pe": C["lead"], "dls_ik": "#7A7A7A", "ccd_ik": "#A6761D", "mpc_nn_dynamics": "#1B9E77", "pets": "#7570B3"}
CTRL_MARK = {"rak": "o", "rak_pe": MARK["lead"], "dls_ik": "D", "ccd_ik": "s", "mpc_nn_dynamics": "p", "pets": "h"}
LABEL = {"matrix": "RAK model", "lead": "Predictive encoding", "dls": "DLS IK",
         "intact": "Intact", "lock": "Actuator lock", "motor": "Motor-population silencing",
         "allocation": "Allocation-population silencing"}
SHORT = {"intact": "Intact", "lock": "Lock", "motor": "Motor", "allocation": "Allocation"}
CONDITIONS = ("lock", "motor", "allocation")
TEAL = LinearSegmentedColormap.from_list("state", ["#FFFFFF", "#A4D9D8", "#087F8C"])
WIDTH = 178 / 25.4
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7.5, "legend.fontsize": 6.1, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "lines.linewidth": 1.2,
    "legend.handlelength": 1.8, "legend.columnspacing": 1.1,
})


def baseline_rows(model):
    """Every trial of one baseline on the 30 Gazebo targets (shards put back together), ."""

    out = []
    for path in sorted((RESULTS / model / "reach_gazebo").glob("*.json")):
        if path.name != "summary.json":
            out += read(path)["rows"]
    out.sort(key=lambda r: r["index"])
    assert [r["index"] for r in out] == list(range(30)), model
    return out


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        # no creation date, so redrawing an unchanged figure gives an identical file
        metadata = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}.get(ext, {})
        fig.savefig(FIG / (name + "." + ext), dpi=300, facecolor="white", metadata=metadata)
    plt.close(fig)


def panel(ax, letter, title, x=-.15):
    text = getattr(ax, "text2D", ax.text)
    text(x, 1.11, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
    ax.set_title(title, loc="left", pad=9, fontweight="normal")


def csvout(name, header, rows):
    with (EXPORT / name).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def nan(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


#: every RAK closed-loop run, scored at 5% of reach (with
#: exact-loop twins and reachability recomputed); short names as in the earlier evidence
RAK_RESULTS = RAK
BENCH = {"reach": "reach_kinematic", "gazebo": "reach_gazebo", "robustness": "reach_disturbance", "gain": "reach_gain"}
RAW = {}
for bench, folder in BENCH.items():
    if True:
        for path in sorted((RAK_RESULTS / folder).glob("*.json")):
            if path.name == "summary.json":
                continue
            doc = read(path)
            assert abs(doc["design"]["reach_reach"] - TOL_FRAC) < 1e-12, path
            run = doc["run"]
            seed = int(run["seed"])
            method = run.get("network_method", "matrix")
            if run.get("method", "ring") != "ring":
                method = run["method"]
            key = (bench, run["arm"], method, run.get("condition", "nominal"), float(run["kappa"]), seed)
            if key in RAW:
                # The update repeats the seed-1 file byte for byte; anything else is a real conflict.
                assert RAW[key]["rows"] == doc["rows"], key
                continue
            RAW[key] = doc


def raw(bench, arm="tiago7", method="matrix", condition="nominal", gain=1., seed=1):
    return RAW[(bench, arm, method, condition, gain, seed)]


def rows(*args, **kwargs):
    return raw(*args, **kwargs)["rows"]


def matched(bench, gain=1., seed=1):
    a = {int(r["index"]): r for r in rows(bench, gain=gain, seed=seed)}
    b = {int(r["index"]): r for r in rows(bench, method="lead", gain=gain, seed=seed)}
    assert sorted(a) == sorted(b) == list(range(30))
    # Starts and goals are paired by recorded values as well as seed/index.
    for i in a:
        for field in ("start", "goal_lpy", "goal_position"):
            assert np.allclose(a[i][field], b[i][field], atol=1e-5), (bench, i, field)
    return [a[i] for i in sorted(a)], [b[i] for i in sorted(b)]


JL = {}
JL_ASSIGN = None
for path in sorted((RAK_RESULTS / "joint_loss").glob("*.json")):
    if path.name == "summary.json":
        continue
    doc = read(path)
    assert abs(doc["design"]["reach_reach"] - TOL_FRAC) < 1e-12, path
    run = doc["run"]
    JL[(run["condition"], int(run["seed"]))] = doc
    if JL_ASSIGN is None:
        JL_ASSIGN = list(doc["disabled_joints"])
    assert list(doc["disabled_joints"]) == JL_ASSIGN


def intact_rows(seed):
    """Intact reference: the dedicated seed-1 run, else nominal core reaching at that seed (same harness, targets)."""
    if ("intact", seed) in JL:
        return JL[("intact", seed)]["rows"]
    return rows("reach", seed=seed)


def tolerance_scaled(deg, reach):
    """Pose error in units of the success tolerance: <= 1 means both requested components are within it."""
    return max(deg / TOL_DEG, reach * RADIUS_MM / TOL_MM)


def pose_error(q, goal_lpy, goal_position):
    deg = float(np.degrees(rotation_angle(CHAIN.rotation(q), from_lift_pitch_yaw(*goal_lpy))))
    reach = float(np.linalg.norm(np.asarray(goal_position) - position(CHAIN, q)) / (RADIUS_MM / 1000))
    return deg, reach


def goal_arrival_s(start_q, goal_lpy, goal_position, dt=.05):
    """When the encoded goal reaches the final goal: the session's own travel rules, from the hand at the start."""
    written_lpy = np.array(lift_pitch_yaw(CHAIN.rotation(start_q)))
    written_p = position(CHAIN, start_q)
    goal_lpy, goal_position = np.asarray(goal_lpy, float), np.asarray(goal_position, float)
    for step in range(2000):
        written_lpy = travel(written_lpy, goal_lpy, 1.0 * dt)
        written_p = travel_position(written_p, goal_position, .3 * (RADIUS_MM / 1000) * dt)
        if np.all(np.abs(wrap(written_lpy - goal_lpy)) < 1e-6) and np.linalg.norm(written_p - goal_position) < 1e-9:
            return (step + 1) * dt
    return float("nan")


def bootstrap_paired(b, a):
    s = paired(np.asarray(b, float), np.asarray(a, float))
    return s["median_difference"], s["ci"][0], s["ci"][1], s["p"]


def box(ax, xy, w, h, text, fc="#F4F6F7", ec="#9AA5AB", color="#1F2A30", size=6.2, weight="normal"):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=.7,
                           transform=ax.transAxes, clip_on=False)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            transform=ax.transAxes, weight=weight, linespacing=1.15)


def arrow(ax, start, end, color="#55626A", lw=.8, style="-|>", rad=0.):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle=style, mutation_scale=7,
                                 color=color, lw=lw, connectionstyle="arc3,rad=%g" % rad, clip_on=False))


# ----------------------------------------------------------------------------------------------------
# Figure 1 - what counts as a coordinated reach?
# ----------------------------------------------------------------------------------------------------
def make_reach_events():
    """Supplementary: what happens in one reach (the RAK model, Gazebo target pair 0), error against the goal over time."""
    r = next(r for r in rows("gazebo") if r["index"] == 0)
    arrival = goal_arrival_s(np.asarray(r["start"], float), r["goal_lpy"], r["goal_position"])
    entry = r["time_to_reach_s"]
    end = r["settled_s"] if r["settled_s"] is not None else r["duration_s"]
    fig = plt.figure(figsize=(WIDTH, 62 / 25.4))
    lower = fig.add_gridspec(1, 2, left=.075, right=.985, bottom=.17, top=.74, wspace=.24)
    errors = [("error_reach", RADIUS_MM, TOL_MM, "Position error (mm)", "a", "Hand position against the goal"),
              ("error_deg", 1., TOL_DEG, "Orientation error (°)", "b", "Hand orientation against the goal")]
    trace_rows = []
    for j, (field, scale, tol, ylabel, letter, title) in enumerate(errors):
        ax = fig.add_subplot(lower[0, j])
        tt = np.asarray(r["t"], float)
        yy = np.asarray(r[field], float) * scale
        ax.plot(tt, yy, color=C["matrix"], lw=1.1)
        ax.axhline(tol, color="#8A9296", lw=.8, ls=":")
        ax.text(.99, tol, " %s" % ("%.1f mm" % TOL_MM if j == 0 else "5°"), transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=5.8, color="#5B666C")
        ymax = max(yy.max(), tol) * 1.1
        ax.axvline(arrival, color=C["goal"], lw=.8, ls="--")
        ax.axvline(entry, color="#2E7D32", lw=.8, ls="-.")
        ax.axvspan(end - 2.0, end, color="#D9DEE1", alpha=.6, lw=0, zorder=0)
        ax.scatter([tt[-1]], [yy[-1]], s=16, color=C["matrix"], zorder=4)
        for xx, lab, col in ((arrival, "encoded goal\narrives", C["goal"]), (entry, "first sustained\nentry", "#2E7D32"),
                             (end, "2 s quiet window\n→ trial ends", "#5B666C")):
            ax.text(xx, 1.01, lab, fontsize=5.2, color=col, ha="right" if xx == end else "center", va="bottom", transform=ax.get_xaxis_transform())
        ax.annotate("final sample:\nsuccess" if r["reached"] else "final sample:\nfailure", xy=(tt[-1], yy[-1]),
                    xytext=(tt[-1] - 3.2, tol * 2.2), fontsize=5.2, color=C["matrix"], ha="center",
                    arrowprops=dict(arrowstyle="-", lw=.5, color="#8A9296"))
        ax.set(xlabel="Time (s)", ylabel=ylabel, xlim=(0, end + .3), ylim=(0, ymax))
        ax.text(-.12, 1.25, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.text(0, 1.25, title, transform=ax.transAxes, fontsize=7.5)
        trace_rows += [[field, a, b] for a, b in zip(tt, yy)]
    save(fig, "supp_reach_events")
    csvout("supp_reach_events_errors.csv", ["quantity", "time_s", "value"], trace_rows)
    csvout("supp_reach_events_times.csv", ["event", "time_s", "definition"],
           [["encoded_goal_arrival", arrival, "recomputed from prescribed start and goal with the session travel rules (1 rad/s per Euler component, 0.3 reach/s)"],
            ["first_sustained_entry", entry, "first logged sample after which both errors stay within tolerance to the end (time_to_reach_s)"],
            ["termination", end, "2 s quiet window after goal arrival (settled_s) or 25 s limit"],
            ["final_sample_success", int(r["reached"]), "success assessed at the final sample"]])


# ----------------------------------------------------------------------------------------------------
# Figure 2 - what does the population represent, and where does it lag?
# ----------------------------------------------------------------------------------------------------
def make_representation():
    path = DATA / "illustrative_activity.npz"
    HASHES[relative(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    act = np.load(path)
    t = act["t"]
    counts = act["ring_counts"][:, 3, :]
    # The replay follows the seed-1 kinematic matrix reach to target pair 0 (illustrative_activity_metadata.json).
    goal_row = [r for r in rows("reach", seed=1) if r["index"] == 0][0]
    goal_p = np.asarray(goal_row["goal_position"], float)
    # Panel b shows the xz sheet (PLANES[1]): this reach moves mainly in x and z.
    space = workspace(CHAIN)
    goal_xz = space.scaled(goal_p)[[0, 2]]
    path_xz = np.array([space.scaled(p)[[0, 2]] for p in act["true_position"]])
    span = np.vstack([path_xz, goal_xz])
    fig = plt.figure(figsize=(WIDTH, 128 / 25.4))
    gs = fig.add_gridspec(2, 3, left=.085, right=.985, bottom=.10, top=.92, hspace=.78, wspace=.56,
                          width_ratios=[1.28, 1.05, .9])
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(counts.T, aspect="auto", origin="lower", extent=[t[0], t[-1], 0, 200],
                   cmap=TEAL, vmin=0, vmax=5, interpolation="nearest", rasterized=True)
    ax.plot(t, act["ring_target"][:, 3], color="#E66101", lw=1, label="Supplied joint angle")
    mapping = RingMap.for_joint(CHAIN.joints[3], 200)
    centroid = np.array([mapping.position(a) if np.isfinite(a) else np.nan for a in act["held"][:, 3]])
    ax.plot(t, centroid, color="#253B43", lw=.8, ls="--", label="Decoded bump centre")
    goal_index = mapping.position(goal_row["final_q"][3])
    ax.axhline(goal_index, color=C["goal"], lw=1.1, ls=(0, (4, 1.5)), label="Joint 4 at end of reach")
    ax.plot(8, goal_index, marker="*", color=C["goal"], markersize=6, mec="white", mew=.3, clip_on=False, zorder=6)
    for edge in mapping.arc:
        ax.axhline(edge, color="#999999", lw=.55, ls=":")
    ax.set(xlabel="Replay time (s)", ylabel="Joint 4 ring neuron", xlim=(0, 8), ylim=(0, 200))
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=.9, fontsize=5.8)
    panel(ax, "a", "Joint-state spikes (forward replay)")
    cb = fig.colorbar(im, ax=ax, pad=.03, fraction=.04, ticks=[0, 5])
    cb.set_label("spikes per neuron\nper 50 ms", labelpad=2, fontsize=5.6)
    cb.ax.tick_params(labelsize=5.6, length=2)

    # Panel b: every spike of the xz sheet over the replay. Left, the spike trail on the sheet coloured by time;
    # right, the same spikes as spike trains along x and along z (cf. the ring raster in panel a).
    places = act["sheet_places"]                                   # (2304, 2): sheet coordinates (x, z) of each neuron
    tick, neuron = np.nonzero(act["sheet_counts"][:, 1] > 0)       # at most one spike per neuron per 50 ms bin
    spike_t, spike_xz = t[tick], places[neuron]
    jitter = np.random.default_rng(0).uniform(-.025, .025, spike_xz.shape)   # spread over the neuron's 0.05-wide cell, for visibility
    TIME = plt.get_cmap("viridis")
    edges = np.arange(0, 8.001, .5)
    trail = np.array([spike_xz[(spike_t >= a) & (spike_t < b)].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    trail_t = .5 * (edges[:-1] + edges[1:])
    # One space-time view: time along the long axis, the sheet's x and z across it. The goal is a line parallel to
    # the time axis (perpendicular to the x-z plane) at the goal's sheet coordinates.
    ax = fig.add_subplot(gs[0, 1:], projection="3d", computed_zorder=False)
    sx, sz = (spike_xz + jitter).T
    ax.scatter(spike_t, sx, sz, c=spike_t, cmap=TIME, vmin=0, vmax=8, s=.8, lw=0, alpha=.3, depthshade=False)
    ax.plot(trail_t, trail[:, 0], trail[:, 1], color="#526169", lw=.8, zorder=6)
    ax.scatter(trail_t, trail[:, 0], trail[:, 1], c=trail_t, cmap=TIME, vmin=0, vmax=8, s=14, edgecolors="white", lw=.5,
               depthshade=False, zorder=7)
    ax.plot([0, 8], [goal_xz[0]] * 2, [goal_xz[1]] * 2, color=C["goal"], lw=1.6, zorder=8)
    ax.scatter([8], [goal_xz[0]], [goal_xz[1]], marker="*", s=90, color=C["goal"], edgecolors="white", lw=.4,
               depthshade=False, zorder=9)
    ax.scatter([0], [path_xz[0, 0]], [path_xz[0, 1]], marker="o", s=16, facecolors="white", edgecolors="#222222", lw=.8,
               depthshade=False, zorder=9)
    lo = np.percentile(spike_xz, .5, axis=0) - .03
    hi = np.percentile(spike_xz, 99.5, axis=0) + .03
    ax.set(xlim=(0, 8), ylim=(lo[0], hi[0]), zlim=(lo[1], hi[1]))
    ax.set_box_aspect((2.4, 1.15, 1.05), zoom=1.38)
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-58)
    box = gs[0, 1:].get_position(fig)
    ax.set_position([box.x0 - .03, box.y0 - .06, box.width + .05, box.height + .10])
    ax.set_xlabel("Replay time (s)", labelpad=-6, fontsize=6.5)
    ax.set_ylabel("x / reach", labelpad=-7, fontsize=6.5)
    ax.set_zlabel("z / reach", labelpad=-7, fontsize=6.5)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.locator_params(axis="y", nbins=3)
    ax.locator_params(axis="z", nbins=3)
    ax.tick_params(labelsize=5.5, pad=-3)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#F6F8F9")
        axis.pane.set_edgecolor("#D4DBDF")
        axis._axinfo["grid"].update(color="#E3E8EB", linewidth=.4)
    ax.text(8, goal_xz[0], goal_xz[1] + .035, "goal", fontsize=6, color="#6B4E0D", ha="center", zorder=10)
    ax.text(0, path_xz[0, 0], path_xz[0, 1] + .05, "start", fontsize=5.6, color="#222222", ha="center", zorder=10)
    fig.text(box.x0 - .045, box.y1 + .035, "b", weight="bold", fontsize=9)
    fig.text(box.x0 - .005, box.y1 + .035, "xz position-sheet spikes in space and time", fontsize=7.5)

    ax = fig.add_subplot(gs[1, 0])
    true_dist = 1000 * np.linalg.norm(act["true_position"] - goal_p, axis=1)
    dec_dist = 1000 * np.linalg.norm(act["decoded_position"] - goal_p, axis=1)
    ax.plot(t, true_dist, color="#444444", lw=1.05, label="From supplied joints")
    ax.plot(t, dec_dist, color="#087F8C", lw=.9, label="Decoded from sheets")
    ax.axhline(TOL_MM, color=C["goal"], lw=.8, ls="--")
    ax.text(.3, TOL_MM, "goal tolerance %.1f mm" % TOL_MM, fontsize=5.4, color="#6B4E0D", va="bottom")
    ax.set(xlabel="Replay time (s)", ylabel="Hand distance to goal (mm)", xlim=(0, 8), ylim=(0, None))
    ax.legend(frameon=False, loc="upper right", fontsize=5.8)
    panel(ax, "c", "Hand approaching the goal, decoded")

    fwd = read(RAK / "held_pose" / "summary.json")["arms"]
    sub = gs[1, 1].subgridspec(2, 1, hspace=.25)
    staticrows = []
    for j, (field, scale, ylab) in enumerate((("orientation_error_deg", 1, "Orientation (°)"),
                                              ("position_error_reach", 100, "Position (% reach)"))):
        ax = fig.add_subplot(sub[j, 0])
        for fam, marker, col in (("generic", "o", "#7E858A"), ("tiago", "s", "#087F8C")):
            vals = np.array([fwd[fam + str(k)][field] for k in range(2, 8)]) * scale
            quant = np.percentile(vals, [25, 50, 75], axis=1)
            ax.fill_between(range(2, 8), quant[0], quant[2], color=col, alpha=.13, lw=0)
            ax.plot(range(2, 8), quant[1], marker=marker, color=col, markersize=2.8, lw=.9,
                    label="Synthetic arm" if fam == "generic" else "TIAGo arm")
        ax.set(ylabel=ylab, xticks=range(2, 8), ylim=(0, 1.95), xlim=(1.7, 7.3), yticks=[0, .5, 1, 1.5])
        ax.yaxis.label.set_size(6)
        if j == 0:
            ax.tick_params(labelbottom=False)
            ax.text(-.42, 1.30, "d", transform=ax.transAxes, fontweight="bold", fontsize=9)
            ax.text(-.04, 1.30, "Held-pose error, 12 arms", transform=ax.transAxes, fontsize=7.5)
            ax.legend(frameon=False, loc="upper left", fontsize=5.6, handlelength=1.2, ncol=2, columnspacing=.8)
        else:
            ax.set_xlabel("DoF used (first k joints of the arm)")
    for fam in ("generic", "tiago"):
        for k in range(2, 8):
            a = fam + str(k)
            for i, (o, p) in enumerate(zip(fwd[a]["orientation_error_deg"], fwd[a]["position_error_reach"])):
                staticrows.append([a, i, o, p * 100, p * fwd[a]["radius_m"] * 1000])

    ax = fig.add_subplot(gs[1, 2])
    tr = np.load(DATA / "common_path_traces.npz")
    HASHES[relative(DATA / "common_path_traces.npz")] = hashlib.sha256((DATA / "common_path_traces.npz").read_bytes()).hexdigest()
    lagrows = []
    meds, p90s = [], []
    for prefix, speed in (("common025", .25), ("common05", .5), ("common1", 1.)):
        moving = tr[prefix + "_moving"].astype(bool)
        v = tr[prefix + "_plain_orientation_deg"][moving]
        v = v[np.isfinite(v)]
        meds.append(np.median(v))
        p90s.append(np.percentile(v, 90))
        lagrows.append([prefix, speed, len(v), meds[-1], p90s[-1]])
    xs = np.arange(3)
    ax.plot(xs, meds, marker=MARK["matrix"], color=C["matrix"], markersize=3.6, lw=1)
    for x, m, p in zip(xs, meds, p90s):
        ax.plot([x, x], [m, p], color=C["matrix"], lw=.8, alpha=.55)
        ax.plot([x - .07, x + .07], [p, p], color=C["matrix"], lw=.8)
        ax.text(x + .1, m, "%.2f°" % m, fontsize=5.6, va="center")
    ax.axhline(1.243, color="#087F8C", ls=":", lw=.8)
    ax.text(2.25, 1.243, "held pose", fontsize=5.4, color="#087F8C", ha="right", va="bottom")
    ax.set(xticks=xs, xticklabels=["0.25", "0.5", "1"], xlabel="Peak joint speed (rad s$^{-1}$)",
           ylabel="Moving orientation error (°)", ylim=(0, None), xlim=(-.35, 2.4))
    ax.text(.03, .96, "same path at\nthree speeds;\nlag ≈ 0.20 s", transform=ax.transAxes, va="top", fontsize=5.6, color="#5B666C")
    panel(ax, "e", "Error during movement", x=-.33)
    save(fig, "fig2_representation")
    csvout("fig2_held_pose_trials.csv", ["arm", "pose_index", "orientation_deg", "position_percent_reach", "position_mm"], staticrows)
    csvout("fig2_replay_readouts.csv", ["time_s", "supplied_joint4_index", "joint4_centroid_index", "joint4_end_of_reach_index",
                                        "true_x_mm", "decoded_x_mm", "true_distance_to_goal_mm", "decoded_distance_to_goal_mm"],
           zip(t, act["ring_target"][:, 3], centroid, np.full(len(t), goal_index), 1000 * act["true_position"][:, 0],
               1000 * act["decoded_position"][:, 0], true_dist, dec_dist))
    csvout("fig2_sheet_trail_xz.csv", ["window_mid_s", "mean_x_scaled", "mean_z_scaled"], zip(trail_t, trail[:, 0], trail[:, 1]))
    csvout("fig2_sheet_spikes_xz.csv", ["time_s", "neuron", "x_scaled", "z_scaled"],
           zip(spike_t, neuron, spike_xz[:, 0], spike_xz[:, 1]))
    csvout("fig2_goal.csv", ["source", "pair_index", "goal_x_mm", "goal_y_mm", "goal_z_mm", "goal_x_scaled", "goal_z_scaled",
                             "joint4_end_of_reach_rad", "joint4_end_of_reach_index"],
           [["results/raw/reach/tiago7_pose_k1_seed1_0of1.json", 0, *(1000 * goal_p), *goal_xz, goal_row["final_q"][3], goal_index]])
    csvout("fig2_moving_error_core.csv", ["path", "peak_speed_rad_s", "moving_samples", "median_deg", "p90_deg"], lagrows)


# ----------------------------------------------------------------------------------------------------
# Figure 3 - does the controller reach the requested pose?
# ----------------------------------------------------------------------------------------------------
def make_reaching():
    EXTRA = 0
    HMM = 147

    def fy(mm):
        return mm / HMM

    fig = plt.figure(figsize=(WIDTH, HMM / 25.4))
    MISS = "#B2182B"
    TOL = {arm: TOL_FRAC * workspace(get_chain(arm)).radius * 1000 for arm in ("tiago7", "tiago4", "generic7")}
    REACH = {arm: workspace(get_chain(arm)).radius * 1000 for arm in TOL}

    # a - what "target reached" means: the two errors, the acceptance box, and when a reach is judged
    top = fig.add_gridspec(1, 3, left=.035, right=.985, bottom=fy(113.8 + EXTRA), top=fy(140 + EXTRA), wspace=.42, width_ratios=[1.2, .95, 1.2])
    ax = fig.add_subplot(top[0, 0])
    ax.axis("off")
    ax.text(-.04, 1.13, "a", transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
    ax.text(.07, 1.13, "What counts as a reached target?", transform=ax.transAxes, fontsize=7.5, va="bottom")
    # Illustration only: names and symbols; the definitions are in the caption and main text.
    from matplotlib.patches import Arc
    dr = ax.inset_axes([-.02, -.20, 1.04, 1.28])
    dr.axis("off")
    dr.set_xlim(0, 1.55)
    dr.set_ylim(-.02, .82)
    dr.set_aspect("equal")
    dr.set_anchor("W")
    P = np.array([[1, .5, 0], [0, .32, 1]])        # oblique view of x, y, z
    L = .36
    H, G = np.array([.38, .17]), np.array([1.02, .38])
    hand_R = from_lift_pitch_yaw(*np.radians([25, 5, -25]))

    def frame(o, R, colour):
        for k in range(3):
            tip = o + L * (P @ R[:, k])
            dr.add_patch(FancyArrowPatch(o, tip, arrowstyle="-|>", mutation_scale=7.5, color=colour, lw=1.3))
    frame(H, hand_R, C["matrix"])
    frame(G, np.eye(3), C["goal"])
    dr.scatter(*H, s=12, color=C["matrix"], zorder=4)
    dr.scatter(*G, marker="*", s=40, color=C["goal"], zorder=4)
    dr.text(H[0] - .02, H[1] - .09, "hand", ha="center", va="top", fontsize=6.4, color=C["matrix"])
    dr.text(G[0] - .06, G[1] + .06, "goal", ha="right", va="bottom", fontsize=6.4, color="#6B4E0D")
    dr.add_patch(FancyArrowPatch(H, G, arrowstyle="<|-|>", mutation_scale=5, color="#55626A", lw=.7, ls=(0, (3, 2)),
                                 shrinkA=2, shrinkB=4))
    mid = (H + G) / 2
    dr.text(mid[0] + .04, mid[1] - .035, r"$\Delta p$", ha="center", va="top", fontsize=7.5, color="#33393D")
    # rotations about the goal's three axes
    for k, (name, w, h) in enumerate(((r"lift $\alpha$", .05, .14), (r"pitch $\beta$", .10, .085), (r"yaw $\gamma$", .16, .05))):
        c = G + .86 * L * P[:, k]
        dr.add_patch(Arc(c, w, h, theta1=-60, theta2=240, color="#6B4E0D", lw=.7))
        dr.text(c[0] + w / 2 + .015, c[1], name, fontsize=6, color="#6B4E0D", ha="left", va="center")
    # theta: the single turn, about one axis n, that brings the hand's orientation onto the goal's (goal = identity here)
    rel = hand_R.T
    n3 = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    n3 = n3 / np.linalg.norm(n3)
    if (P @ n3)[1] < 0:
        n3 = -n3
    n2 = P @ n3
    dr.plot([H[0] - .55 * L * n2[0], H[0] + 1.25 * L * n2[0]], [H[1] - .55 * L * n2[1], H[1] + 1.25 * L * n2[1]],
            color=MISS, lw=.8, ls=(0, (3, 1.5)))
    dr.text(H[0] + 1.32 * L * n2[0], H[1] + 1.32 * L * n2[1], r"$n$", ha="center", va="bottom", fontsize=7.5, color=MISS)
    u = np.cross(n3, [0, 0, 1.]) if abs(n3[2]) < .9 else np.cross(n3, [1., 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(n3, u)
    centre = H + .95 * L * n2
    ring = np.array([centre + .065 * (P @ (np.cos(t) * u + np.sin(t) * v)) for t in np.linspace(.3, 2 * np.pi - .5, 60)])
    dr.plot(ring[:, 0], ring[:, 1], color=MISS, lw=1.1)
    dr.add_patch(FancyArrowPatch(ring[-3], ring[-1], arrowstyle="-|>", mutation_scale=6, color=MISS, lw=1.1))
    dr.text(centre[0] - .09, centre[1], r"$\theta$", ha="right", va="center", fontsize=8, color=MISS)

    ax = fig.add_subplot(top[0, 1])
    ax.add_patch(Rectangle((0, 0), TOL["tiago7"], TOL_DEG, fc="#E3F1E5", ec="none", zorder=0))
    ax.axvline(TOL["tiago7"], color="#2E7D32", lw=.8, ls="--")
    ax.axhline(TOL_DEG, color="#2E7D32", lw=.8, ls="--")
    ax.text(TOL["tiago7"] / 2, TOL_DEG / 2, "reached:\nboth within\ntolerance", ha="center", va="center", fontsize=5.7, color="#1B5E20")
    ax.text(TOL["tiago7"] / 2, TOL_DEG * 1.55, "missed\n(orientation)", ha="center", va="center", fontsize=5.4, color=MISS)
    ax.text(TOL["tiago7"] * 1.52, TOL_DEG / 2, "missed\n(position)", ha="center", va="center", fontsize=5.4, color=MISS)
    ax.text(TOL["tiago7"] * 1.52, TOL_DEG * 1.55, "missed\n(both)", ha="center", va="center", fontsize=5.4, color=MISS)
    ax.set(xlim=(0, TOL["tiago7"] * 2.05), ylim=(0, TOL_DEG * 2.1), xticks=[0, TOL["tiago7"]], yticks=[0, TOL_DEG],
           xlabel=r"Final position error $\Vert\Delta p\Vert$ (mm)", ylabel="Final orientation\n" r"error $\theta$ (°)")
    ax.set_xticklabels(["0", "%.1f\n%d%% of reach" % (TOL["tiago7"], round(100 * TOL_FRAC))])
    ax.set_yticklabels(["0", "5"])
    ax.text(-.05, 1.13, "Tolerances (TIAGo arm, 7 DoF)", transform=ax.transAxes, fontsize=6.4, va="bottom", color="#33393D")

    r = next(r for r in rows("gazebo") if r["index"] == 0)
    tt = np.asarray(r["t"], float)
    pos = np.asarray(r["error_reach"], float) * RADIUS_MM / TOL["tiago7"]
    ori = np.asarray(r["error_deg"], float) / TOL_DEG
    entry, end = r["time_to_reach_s"], (r["settled_s"] if r["settled_s"] is not None else r["duration_s"])
    ax = fig.add_subplot(top[0, 2])
    ax.axhspan(0, 1, color="#E3F1E5", lw=0, zorder=0)
    ax.axhline(1, color="#2E7D32", lw=.8, ls="--")
    ax.axvspan(end - 2.0, end, color="#D9DEE1", alpha=.7, lw=0, zorder=0)
    ax.plot(tt, pos, color=C["matrix"], lw=1.0, label=r"$\Vert\Delta p\Vert$ ÷ %.1f mm" % TOL["tiago7"])
    ax.plot(tt, ori, color=C["matrix"], lw=1.0, ls=(0, (2, 1.2)), label=r"$\theta$ ÷ 5°")
    ax.axvline(entry, color="#2E7D32", lw=.8, ls="-.")
    ax.scatter([tt[-1]] * 2, [pos[-1], ori[-1]], s=10, color=C["matrix"], zorder=4)
    ax.set_yscale("log")
    ax.set(xlim=(0, end + .4), ylim=(.025, 14), xlabel="Time from reach onset (s)", ylabel="Error ÷ tolerance")
    ax.set_yticks([.1, 1, 10])
    ax.set_yticklabels(["0.1", "1", "10"])
    ax.text(entry - .15, 9, "enters and\nstays (c)", ha="right", va="top", fontsize=5.2, color="#2E7D32")
    ax.text(end - 2.2, 9, "hand still for 2 s\n→ trial ends", ha="right", va="top", fontsize=5.2, color="#5B666C")
    ax.annotate("judged at the last sample (b, d)", xy=(tt[-1], max(pos[-1], ori[-1])), xytext=(end - 6.5, .045), fontsize=5.2,
                ha="center", va="center", color=C["matrix"], arrowprops=dict(arrowstyle="-", lw=.5, color="#8A9296"))
    ax.legend(frameon=False, loc="upper center", fontsize=5.0, handlelength=1.6, borderaxespad=.1, bbox_to_anchor=(.62, .80))
    ax.text(-.05, 1.13, "When a reach is judged (Gazebo target 0)", transform=ax.transAxes, fontsize=6.4, va="bottom", color="#33393D")

    # b, c, d - every controller on the same 30 Gazebo targets: the RAK model and the four baselines
    ctrls = [(key, label, CTRL_COLOUR[key], CTRL_MARK[key], rows("gazebo", method="matrix") if key == "rak" else rows("gazebo", method="lead") if key == "rak_pe" else baseline_rows(key))
             for key, label in CONTROLLERS]

    # b - success
    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=.255, right=.555, bottom=fy(60.7), top=fy(91.7))[0, 0])
    successrows = []
    for i, (key, label, colour, marker, rr) in enumerate(ctrls):
        sm = success_summary([r["reached"] for r in rr])
        rate, (lo, hi) = sm["rate"] * 100, np.array(sm["ci"]) * 100
        ax.errorbar(rate, i, xerr=[[rate - lo], [hi - rate]], fmt=marker, color=colour, capsize=2, markersize=3.6,
                    mfc=colour if key in RAK_KEYS else "white", lw=.9)
        ax.text(101.5, i, "%d/30" % sm["reached"], ha="left", va="center", color=colour, fontsize=5.6, clip_on=False)
        successrows.append([key, label, sm["n"], sm["reached"], sm["rate"], *sm["ci"]])
    ax.set(xlabel="Targets reached (%; Wilson 95% interval)", xlim=(70, 108), xticks=[70, 80, 90, 100],
           ylim=(len(ctrls) - .4, -.6))
    ax.set_yticks(range(len(ctrls)))
    ax.set_yticklabels([label for _, label, *_ in ctrls], fontsize=5.8)
    for t, (key, _, colour, *_) in zip(ax.get_yticklabels(), ctrls):
        t.set_color("#1F2A30" if key == "rak" else colour)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(70, 100)
    ax.xaxis.label.set_x(15 / 38)
    fig.text(.035, fy(95.6), "b", fontweight="bold", fontsize=9)
    fig.text(.06, fy(95.6), "How often is the requested pose reached? (Gazebo, same 30 targets)", fontsize=7.5)

    # c - when each reach enters tolerance for good (beside b)
    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=.705, right=.985, bottom=fy(60.7), top=fy(91.7))[0, 0])
    grid = np.arange(0, LIMIT_S + .001, .05)
    cumrows = []
    styles = {"rak": "-", "rak_pe": (0, (3, 1)), "dls_ik": (0, (5, 1.5)), "ccd_ik": (0, (1.2, 1.2)), "mpc_nn_dynamics": (0, (5, 1.5, 1, 1.5)), "pets": "-"}
    for key, label, colour, marker, rr in ctrls:
        times = np.array([r["time_to_reach_s"] if r["reached"] and r["time_to_reach_s"] is not None else np.inf for r in rr])
        frac = np.array([(times <= g).sum() / len(rr) for g in grid])
        ax.step(grid, frac, where="post", color=colour, ls=styles[key], lw=1.4 if key in RAK_KEYS else 1.0, label=SHORT_CTRL[key])
        cumrows += [[key, g, fr] for g, fr in zip(grid[::10], frac[::10])]
    ax.set(xlabel="Time from reach onset (s)", ylabel="Fraction of the 30 targets\nin tolerance from then on",
           xlim=(0, LIMIT_S), ylim=(0, 1.02), yticks=[0, .5, 1])
    ax.legend(frameon=False, loc="lower right", fontsize=5.0, ncol=1, handlelength=2.2)
    fig.text(.605, fy(95.6), "c", fontweight="bold", fontsize=9)
    fig.text(.63, fy(95.6), "When does each reach enter tolerance for good?", fontsize=7.5)

    # d - every endpoint, one facet per controller, common axes
    sub = fig.add_gridspec(1, len(ctrls), left=.075, right=.985, bottom=fy(9.7), top=fy(40.5), wspace=.12)
    every = [r for *_, rr in ctrls for r in rr]
    xmax = max(r["final_error_reach"] for r in every) * RADIUS_MM * 1.08
    ymax = max(r["final_error_deg"] for r in every) * 1.45
    endpointrows = []
    for j, (key, label, colour, marker, rr) in enumerate(ctrls):
        ax = fig.add_subplot(sub[0, j])
        ax.add_patch(Rectangle((0, 0), TOL_MM, TOL_DEG, fc="#E3F1E5", ec="none", zorder=0))
        ax.plot([TOL_MM, TOL_MM, 0], [0, TOL_DEG, TOL_DEG], color="#2E7D32", lw=.7, ls="--", zorder=1)
        xx = np.array([r["final_error_reach"] for r in rr]) * RADIUS_MM
        yy = np.array([r["final_error_deg"] for r in rr])
        ok = np.array([r["reached"] for r in rr], bool)
        ax.scatter(xx[ok], yy[ok], s=9, marker=marker, facecolors="none", edgecolors=colour, linewidths=.7)
        ax.scatter(xx[~ok], yy[~ok], s=14, marker="x", color=MISS, linewidths=.9)
        ax.text(.05, .97, "%d/30 reached\nmedian %.1f mm, %.2f°" % (ok.sum(), np.median(xx), np.median(yy)), transform=ax.transAxes,
                ha="left", va="top", fontsize=4.8, color=colour, bbox=dict(fc="white", ec="none", alpha=.85, pad=.4))
        ax.set_title(label.replace(" (TIAGo 7-DoF)", "").replace(", predictive", ",\npredictive"), fontsize=6.2, color="#1F2A30" if key == "rak" else colour, pad=3)
        ax.set(xlim=(0, xmax), ylim=(0, ymax), xticks=[0, TOL_MM], yticks=[0, TOL_DEG])
        ax.set_xticklabels(["0", "%.1f" % TOL_MM])
        if j == 0:
            ax.set_ylabel(r"Final orientation error $\theta$ (°)")
            ax.set_yticklabels(["0", "5"])
        else:
            ax.tick_params(labelleft=False)
        if j == len(ctrls) // 2:
            ax.set_xlabel(r"Final position error $\Vert\Delta p\Vert$ (mm)")
        for r, x, y, o in zip(rr, xx, yy, ok):
            hand = np.array(lift_pitch_yaw(CHAIN.rotation(np.asarray(r["final_q"], float))))
            d = np.degrees(np.abs([wrap(v) for v in hand - np.asarray(r["goal_lpy"], float)]))
            endpointrows.append([key, r["index"], x, y, *d, int(o)])
    fig.text(.035, fy(45.9), "d", fontweight="bold", fontsize=9)
    fig.text(.06, fy(45.9), "Final errors of every Gazebo reach", fontsize=7.5)
    save(fig, "fig3_reaching")
    csvout("fig3_reached_definition.csv", ["arm", "reach_mm", "position_tolerance_mm", "orientation_tolerance_deg", "requested"],
           [[a, REACH[a], TOL[a], "" if a == "tiago4" else TOL_DEG, "position" if a == "tiago4" else "position and orientation"]
            for a in ("tiago7", "tiago4", "generic7")])
    csvout("fig3_judging_trace_gazebo_target0.csv", ["time_s", "position_over_tolerance", "orientation_over_tolerance"],
           [[a, b, c] for a, b, c in zip(tt, pos, ori)])
    csvout("fig3_gazebo_success.csv", ["controller", "label", "n", "reached", "rate", "wilson_low", "wilson_high"], successrows)
    csvout("fig3_gazebo_endpoints.csv", ["controller", "target_index", "position_mm", "orientation_deg",
                                         "abs_lift_diff_deg", "abs_pitch_diff_deg", "abs_yaw_diff_deg", "reached"], endpointrows)
    csvout("fig3_sustained_entry.csv", ["controller", "time_s", "fraction_of_30"], cumrows)
    kinrows = []
    for arm, task, seed in (("tiago7", "position + orientation", 1), ("tiago7", "position + orientation", 2),
                            ("tiago7", "position + orientation", 3), ("generic7", "position + orientation", 1),
                            ("tiago4", "position only", 1)):
        rr = rows("reach", arm, "matrix", seed=seed)
        sm = success_summary([r["reached"] for r in rr])
        entry = [r["time_to_reach_s"] if r["reached"] and r["time_to_reach_s"] is not None else LIMIT_S for r in rr]
        reach_mm = workspace(get_chain(arm)).radius * 1000
        kinrows.append([arm, task, seed, sm["n"], sm["reached"], *sm["ci"], TOL[arm],
                        float(np.median([r["final_error_reach"] for r in rr])) * reach_mm,
                        float(np.median([r["final_error_deg"] for r in rr])) if arm != "tiago4" else "",
                        float(np.median(entry)), " ".join(str(r["index"]) for r in rr if not r["reached"])])
    csvout("supp_rak_kinematic_plant.csv", ["arm", "task", "kernel_seed", "n", "reached", "wilson_low", "wilson_high",
                                            "position_tolerance_mm", "median_final_position_mm", "median_final_orientation_deg",
                                            "median_entry_s_failures_25", "missed_target_indices"], kinrows)


def make_prediction():
    NO, YES = "Without predictive encoding", "With predictive encoding"
    fig = plt.figure(figsize=(WIDTH, 150 / 25.4))
    top = fig.add_gridspec(1, 2, left=.085, right=.985, bottom=.585, top=.94, wspace=.38, width_ratios=[1.55, .85])

    # a - estimation error over time, one shared path at 1 rad/s
    tr = np.load(DATA / "common_path_traces.npz")
    sub = top[0, 0].subgridspec(2, 1, hspace=.12, height_ratios=[.32, 1])
    axs, ax = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[1, 0])
    tt = tr["common1_t_s"]
    axs.fill_between(tt, 0, tr["common1_speed_rad_s"], color="#C9D1D5", lw=0)
    axs.set(xlim=(tt[0], tt[-1]), ylim=(0, 1.15), yticks=[0, 1], ylabel="joint speed\n(rad s$^{-1}$)")
    axs.yaxis.label.set_size(5.4)
    axs.tick_params(labelbottom=False, labelsize=5.4)
    ax.plot(tt, tr["common1_plain_orientation_deg"], color=C["matrix"], lw=.75, label=NO)
    ax.plot(tt, tr["common1_lead_orientation_deg"], color=C["lead"], lw=.75, label=YES)
    ax.set(xlim=(tt[0], tt[-1]), ylim=(0, 29), yticks=[0, 5, 10, 15, 20, 25], xlabel="Time on the replayed path (s)",
           ylabel="Hand-orientation\nestimate error (°)")
    ax.legend(frameon=False, fontsize=5.3, loc="upper left", handlelength=1.4, borderaxespad=.1)
    axs.text(-.11, 1.30, "a", transform=axs.transAxes, fontweight="bold", fontsize=9)
    axs.text(-.04, 1.30, "Forward-kinematics error in time", transform=axs.transAxes, fontsize=7.5)
    tracerows = list(zip(tt, tr["common1_speed_rad_s"], tr["common1_plain_orientation_deg"], tr["common1_lead_orientation_deg"]))

    # b - the path of a replayed at three peak speeds: median error while moving, without and with (paired bars)
    from matplotlib.patches import Patch
    ax = fig.add_subplot(top[0, 1])
    est = read(RAK / "moving_estimate" / "summary.json")["paths"]
    # every recorded path is exported; the panel shows the path of b, where only the speed changes
    estrows = [[name, est[name]["network_lag_s"], est[name]["sources"]["M0"]["moving_median_deg"],
                est[name]["sources"]["M1"]["moving_median_deg"]]
               for name in ("speed0.25", "speed0.5", "speed1", "pole", "common0.25", "common0.5", "common1")]
    width = .36
    for i, speed in enumerate((.25, .5, 1.)):
        name = "common%g" % speed
        for value, dx, key in ((est[name]["sources"]["M0"]["moving_median_deg"], -width / 2, "matrix"),
                               (est[name]["sources"]["M1"]["moving_median_deg"], width / 2, "lead")):
            ax.bar(i + dx, value, width, color=C[key], linewidth=0, zorder=2)
            ax.text(i + dx, value + .15, "%.2f" % value, ha="center", va="bottom", fontsize=4.9, color=C[key])
    ax.set(xticks=range(3), xlim=(-.6, 2.6), ylim=(0, 10), ylabel="Median error\nwhile moving (°)")
    ax.set_xticklabels(["0.25", "0.5", "1"], fontsize=5.8)
    ax.set_xlabel("Peak joint speed (rad s$^{-1}$)", fontsize=6)
    ax.legend(handles=[Patch(color=C["matrix"], label="Without"), Patch(color=C["lead"], label="With")],
              frameon=False, loc="upper left", fontsize=5.2, handlelength=1.0, borderaxespad=0)
    ax.text(-.62, 1.07, "b", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax.text(-.44, 1.07, "Forward-kinematics error by speed", transform=ax.transAxes, fontsize=7.5)

    # d, e - closed-loop reaching in Gazebo: every controller of Fig. 3 on the same 30 targets, paired against the RAK model
    MISS = "#B2182B"
    ctrls = [(key, rows("gazebo", method="matrix") if key == "rak" else rows("gazebo", method="lead") if key == "rak_pe"
              else baseline_rows(key)) for key, _ in CONTROLLERS]
    ref = ctrls[0][1]
    for key, rs in ctrls:
        assert [int(r["index"]) for r in rs] == list(range(30)), key
        for r, r0 in zip(rs, ref):   # the same targets: paired by recorded start and goal as well as index
            for name in ("start", "goal_lpy", "goal_position"):
                assert np.allclose(r[name], r0[name], atol=1e-4), (key, r["index"], name)
    bottom = fig.add_gridspec(1, 2, left=.13, right=.985, bottom=.115, top=.455, wspace=.06)
    effectrows, summaryrows, success_note = [], [], []
    for j, (field, xlabel, letter, title) in enumerate((
            ("path_ratio", "Hand path ratio (distance travelled / initial distance; 1 = straight)", "c",
             "How far does the hand travel?"),
            ("time_to_reach_s", "Time until the hand stays within tolerance (s; missed targets at 25 s)", "d",
             "When does the hand settle?"))):
        ax = fig.add_subplot(bottom[0, j])

        def values(rs):
            if field == "time_to_reach_s":
                return np.asarray(censored([r[field] for r in rs], LIMIT_S), float)
            return np.array([r[field] for r in rs], float)

        av = values(ref)
        for i, (key, rs) in enumerate(ctrls):
            v = values(rs)
            reached = np.array([bool(r["reached"]) for r in rs])
            colour, filled = CTRL_COLOUR[key], key in RAK_KEYS
            y = i + np.random.default_rng([7, i, j]).uniform(-.17, .17, len(v))
            ax.scatter(v[reached], y[reached], s=7, marker=CTRL_MARK[key], facecolors=colour if filled else "none",
                       edgecolors=colour, linewidths=0 if filled else .5, alpha=.5 if filled else .7, zorder=2)
            ax.scatter(v[~reached], y[~reached], s=9, marker="x", color=MISS, linewidths=.6, zorder=3)
            med = float(np.nanmedian(v))
            ax.plot([med] * 2, [i - .27, i + .27], color=colour, lw=1.6, zorder=4, solid_capstyle="butt")
            unit = "" if field == "path_ratio" else " s"
            if key == "rak":
                txt = ("median %.2f" + unit) % med
            else:
                d, lo, hi, pval = bootstrap_paired(v, av)
                txt = ("median %.2f" + unit + "; vs RAK %+.2f [%+.2f, %+.2f]") % (med, d, lo, hi)
                summaryrows.append(["gazebo", key, field, len(v), med, d, lo, hi, pval])
            ax.text(.995, i - .38, txt, transform=ax.get_yaxis_transform(), fontsize=4.8, ha="right", va="center",
                    color="#1F2A30" if key == "rak" else colour)
            for r, va, vb in zip(ref, av, v):
                effectrows.append(["gazebo", key, r["index"], field, va, vb, vb - va])
            if j == 0:
                success_note.append("%s: %d/30" % (SHORT_CTRL[key], int(reached.sum())))
        ax.set(yticks=range(len(ctrls)), ylim=(len(ctrls) - .5, -.66), xlabel=xlabel)
        ax.xaxis.label.set_size(6.2)
        ax.set_yticklabels([SHORT_CTRL[k] for k, _ in ctrls] if j == 0 else [], fontsize=5.8)
        if j == 0:
            for t, (key, _) in zip(ax.get_yticklabels(), ctrls):
                t.set_color("#1F2A30" if key == "rak" else CTRL_COLOUR[key])
        ax.tick_params(axis="y", length=0)
        if field == "path_ratio":
            top_x = max(float(np.nanmax(values(rs))) for _, rs in ctrls)
            ax.set_xscale("log")
            ax.set_xlim(.95, max(6, 1.15 * top_x))
            ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 50) if t <= max(6, 1.15 * top_x)]
            ax.set_xticks(ticks)
            ax.set_xticklabels(["%g" % t for t in ticks])
            ax.minorticks_off()
        else:
            ax.set_xlim(0, 26.5)
        panel(ax, letter, title, x=-.22 if j == 0 else -.04)
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", color=C["matrix"], markersize=3.5, label="RAK model (filled)"),
                        Line2D([], [], marker="D", ls="none", markerfacecolor="none", color="#7A7A7A", markersize=3.5,
                               label="baselines (hollow)"),
                        Line2D([], [], marker="x", ls="none", color=MISS, markersize=3.5, label="missed target"),
                        Line2D([], [], color="#59646A", lw=1.6, label="median of 30 targets")],
               loc="lower center", bbox_to_anchor=(.56, 0.), ncol=4, frameon=False, fontsize=5.8, handletextpad=.3,
               columnspacing=1.6)
    # Supplementary Table S4: with minus without predictive encoding, per kernel seed, gain and plant (the RAK model only)
    seedrows = []
    for bench, gain, seed in (("reach", 1., 1), ("reach", 1., 2), ("reach", 1., 3), ("gain", 1.5, 1), ("gain", 2., 1), ("gazebo", 1., 1)):
        a, b = matched(bench, gain, seed)
        line = [bench, gain, seed, sum(r["reached"] for r in a), sum(r["reached"] for r in b)]
        for field in ("path_ratio", "time_to_reach_s"):
            av = censored([r[field] for r in a], LIMIT_S) if field == "time_to_reach_s" else [r[field] for r in a]
            bv = censored([r[field] for r in b], LIMIT_S) if field == "time_to_reach_s" else [r[field] for r in b]
            line += list(bootstrap_paired(bv, av)) + [float(np.median(av)), float(np.median(bv))]
        seedrows.append(line)
    csvout("tableS4_predictive_encoding_paired.csv", ["plant", "gain_s_inverse", "kernel_seed", "reached_without", "reached_with",
           "path_ratio_change", "ci95_low", "ci95_high", "wilcoxon_p", "path_ratio_median_without", "path_ratio_median_with",
           "time_change_s", "time_ci95_low", "time_ci95_high", "time_wilcoxon_p", "time_median_without_s", "time_median_with_s"], seedrows)
    csvout("fig4_shared_path_trace_1rad_s.csv", ["time_s", "joint_speed_rad_s", "without_orientation_error_deg",
                                                 "with_orientation_error_deg"], tracerows)
    save(fig, "fig4_prediction")
    csvout("fig4_moving_estimation_by_path.csv", ["path", "network_lag_s", "without_moving_median_deg", "with_moving_median_deg"], estrows)
    csvout("fig4_paired_effects.csv", ["plant", "controller", "target_index", "quantity", "rak", "controller_value",
                                       "controller_minus_rak"], effectrows)
    csvout("fig4_paired_summaries.csv", ["plant", "controller", "quantity", "n_target_pairs", "median", "paired_median_vs_rak",
                                         "ci95_low", "ci95_high", "wilcoxon_p"], summaryrows)
    csvout("fig4_success_counts.csv", ["note"], [[n] for n in success_note])
    return summaryrows


# ----------------------------------------------------------------------------------------------------
# Figure 5 - what do the completed TIAGo ablations establish?
# ----------------------------------------------------------------------------------------------------
def onset_series(row, joint, key, rel):
    t = np.asarray(row["t"], float) - ONSET_S
    v = np.array([np.nan if x is None or x[joint] is None else x[joint] for x in row[key]], dtype=float)
    return np.interp(rel, t, v, left=np.nan, right=np.nan) if key == "shares" else \
        np.array([v[np.argmin(np.abs(t - g))] if (t.min() - .1 <= g <= t.max() + .1) else np.nan for g in rel])


def make_ablation():
    seeds = (1, 2, 3)
    data = {(c, s): JL[(c, s)]["rows"] for c in CONDITIONS for s in seeds}
    data.update({("intact", s): intact_rows(s) for s in seeds})
    for (c, s), rr in data.items():
        assert sorted(int(r["index"]) for r in rr) == list(range(30)), (c, s)
    # the interventions themselves are drawn on the circuit in Fig. 1 (badges and dashed outlines)
    fig = plt.figure(figsize=(WIDTH, 112 / 25.4))
    gs = fig.add_gridspec(2, 4, left=.09, right=.975, bottom=.09, top=.9, hspace=.62, wspace=.62, width_ratios=[1, 1, 1, 1.35])

    # a - intervention check in the circuit
    rel = np.arange(-1.0, 6.01, .2)
    ax1 = fig.add_subplot(gs[0, 0])
    checkrows = []
    for c in ("intact",) + CONDITIONS:
        shares, written = [], []
        for s in seeds:
            for r in data[(c, s)]:
                k = JL_ASSIGN[int(r["index"])]
                shares.append(np.abs(onset_series(r, k, "shares", rel)))
                lead = onset_series(r, k, "lead", rel)
                written.append(np.where(np.isnan(lead), 0., 1.))
        shares, written = np.array(shares), np.array(written)
        med = np.nanmedian(shares, axis=0)
        frac = np.mean(written, axis=0)
        # the cascade's requested turn for the stopped joint (its share s_k, rad per 50 ms tick), in degrees,
        # and how often a joint target is still written for it, in % of the 90 reaches
        ax1.plot(rel, np.degrees(med), color=C[c], lw=.9, marker=MARK[c], markersize=2, markevery=5)
        checkrows += [[c, g, m, f] for g, m, f in zip(rel, med, frac)]
    ax1.axvline(0, color="#59646A", lw=.7, ls="--")
    ax1.tick_params(labelsize=5.4)
    ax1.set(ylabel="Turn the circuit requests\n(° per 50 ms tick)", ylim=(0, None), xlim=(-1, 6),
            xlabel="Time after the joint stops (s)")
    ax1.yaxis.label.set_size(5.8)
    ax1.yaxis.labelpad = 1
    panel(ax1, "a", "Is the stopped joint\nstill asked to turn?", x=-.42)

    # b - hand error over time
    grid = np.arange(0, LIMIT_S + .001, .1)
    courserows = []
    for j, (field, scale, tol, ylabel, letter, title) in enumerate((
            ("error_reach", RADIUS_MM, TOL_MM, "Hand position error (mm)", "b", "Does the hand still reach the goal?"),
            ("error_deg", 1., TOL_DEG, "Hand orientation error (°)", "", ""))):
        ax = fig.add_subplot(gs[0, 1 + j])
        for c in ("intact",) + CONDITIONS:
            vals = np.array([np.interp(grid, np.asarray(r["t"], float), np.asarray(r[field], float) * scale)
                             for s in seeds for r in data[(c, s)]])
            q25, q50, q75 = np.percentile(vals, [25, 50, 75], axis=0)
            if c in ("intact", "lock"):
                ax.fill_between(grid, q25, q75, color=C[c], alpha=.12, lw=0)
            ax.plot(grid, q50, color=C[c], lw=1.0, label=SHORT[c])
            courserows += [[c, field, g, a, b, d] for g, a, b, d in zip(grid, q25, q50, q75)]
        ax.axvline(ONSET_S, color="#59646A", lw=.7, ls="--")
        ax.axhline(tol, color="#8A9296", lw=.8, ls=":")
        ax.set(xlabel="Time (s)", ylabel=ylabel, xlim=(0, 15), ylim=(0, None))
        ax.set_yscale("log")
        ax.set_ylim(.5 if field == "error_deg" else 5, None)
        if j == 0:
            panel(ax, letter, title, x=-.19)
            ax.legend(frameon=False, fontsize=5.3, loc="upper right", handlelength=1.2, borderaxespad=.1)
        ax.text(ONSET_S + .15, .03, "joint disabled", transform=ax.get_xaxis_transform(), fontsize=5.3, color="#59646A")

    # c - success per seed and condition, with the reachable ceiling
    ax = fig.add_subplot(gs[0, 3])
    successrows = []
    for i, c in enumerate(("intact",) + CONDITIONS):
        for j, s in enumerate(seeds):
            rr = data[(c, s)]
            sm = success_summary([r["reached"] for r in rr])
            x = i + (j - 1) * .22
            lo, hi = np.array(sm["ci"]) * 30
            ax.errorbar(x, sm["reached"], yerr=[[sm["reached"] - lo], [hi - sm["reached"]]], fmt=MARK[c], color=C[c], markersize=3.4,
                        capsize=1.5, lw=.8, mfc=C[c])
            reachable = sum(bool((r.get("reachability") or {}).get("reachable")) for r in rr) if c != "intact" else None
            if reachable is not None:
                ax.plot([x - .09, x + .09], [reachable] * 2, color=C[c], lw=1.1)
            successrows.append([c, s, sm["n"], sm["reached"], *sm["ci"], reachable])
        ax.text(i, 31.2, SHORT[c], ha="center", fontsize=5.4, color=C[c])
    ax.plot([], [], color="#59646A", lw=1.1, label="still reachable with the joint\nfrozen (DLS IK check)")
    ax.set(xticks=range(4), xticklabels=["s1 s2 s3"] * 4, ylabel="Targets reached (of 30)", ylim=(0, 33), xlim=(-.5, 3.5))
    ax.tick_params(axis="x", labelsize=4.8)
    ax.tick_params(axis="x", labelsize=5.0, length=0)
    ax.legend(frameon=False, fontsize=5.2, loc="lower center")
    panel(ax, "c", "Reaches that succeed", x=-.24)

    # d - absorption: counterfactual (other joints as intact) against actual
    ax = fig.add_subplot(gs[1, :3])
    absrows = []
    counts = {}
    for c in CONDITIONS:
        xs, ys = [], []
        for s in seeds:
            base = {int(r["index"]): r for r in data[("intact", s)]}
            for r in data[(c, s)]:
                a = base[int(r["index"])]
                k = int(r["disabled_joint"])
                qcf = np.asarray(a["final_q"], float).copy()
                qcf[k] = float(r["q_onset"][k])
                cf = tolerance_scaled(*pose_error(qcf, a["goal_lpy"], a["goal_position"]))
                act = tolerance_scaled(r["final_error_deg"], r["final_error_reach"])
                xs.append(cf)
                ys.append(act)
                absrows.append([c, s, r["index"], k + 1, cf, act, int(r["reached"]),
                                int(bool((r.get("reachability") or {}).get("reachable")))])
        xs, ys = np.array(xs), np.array(ys)
        miss = xs > 1
        counts[c] = (int(miss.sum()), int((miss & (ys <= 1)).sum()), int(((~miss) & (ys > 1)).sum()))
        ax.scatter(xs, ys, s=11, marker=MARK[c], facecolors="none" if c != "motor" else C[c], edgecolors=C[c], color=C[c],
                   linewidths=.7, alpha=.75, label="%s: %d of %d would-miss targets reached" % (SHORT[c], counts[c][1], counts[c][0]))
    ax.axhline(1, color="#59646A", lw=.8, ls=":")
    ax.axvline(1, color="#59646A", lw=.8, ls=":")
    ax.set(xscale="log", yscale="log", xlim=(.2, 20), ylim=(.2, 20),
           xlabel="Error if the other joints had moved as in the intact reach (× tolerance)",
           ylabel="Actual final error\n(× tolerance)")
    ax.plot([.2, 20], [.2, 20], color="#C9D0D4", lw=.6, zorder=0)
    ax.text(11, .23, "absorbed: would have\nmissed, was reached", fontsize=5.6, color="#1B5E20", va="bottom", ha="right")
    ax.text(11, 17, "not absorbed", fontsize=5.6, color="#7F1D1D", va="top", ha="right")
    ax.text(.22, .23, "freezing alone\nwould not miss", fontsize=5.6, color="#55626A", va="bottom")
    ax.legend(frameon=False, fontsize=5.4, loc="upper left")
    panel(ax, "d", "Do the remaining joints absorb the lost motion?", x=-.09)

    # e - diffuse silencing
    sub = gs[1, 3].subgridspec(2, 1, hspace=.65)
    audit = read(DATA / "sensitivity_metrics.json")
    base = read(RAK / "held_pose" / "summary.json")["arms"]["tiago7"]
    ax = fig.add_subplot(sub[0, 0])
    classes = [("all", "all"), ("rings", "rings"), ("value", "value"), ("sheets", "sheets")]
    diffrows = []
    for i, (name, label) in enumerate(classes):
        vals = []
        for row in audit["runs"]:
            lesion = row["lesion"]
            pop = "all" if len(lesion["classes"]) == 3 else lesion["classes"][0]
            if pop == name and np.isclose(float(lesion["fraction"]), .1):
                vals.append(float(row["orientation_median_deg"]))
                diffrows.append(["held_pose_10pct", name, int(lesion["lesion_seed"]), row["orientation_median_deg"], row["position_median_mm"]])
        ax.scatter([i] * len(vals), vals, s=9, color=C["diffuse"], marker="X", linewidths=0)
    ax.axhline(base["orientation_median_deg"], color="#087F8C", lw=.8, ls=":")
    ax.set(xticks=range(4), xticklabels=[c[1] for c in classes], ylabel="Held-pose\norientation (°)", yscale="log", ylim=(.8, 40))
    ax.tick_params(labelsize=5.4)
    ax.yaxis.label.set_size(5.8)
    ax.text(-.55, 1.30, "e", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax.text(-.08, 1.30, "10% diffuse silencing", transform=ax.transAxes, fontsize=7.5)
    ax = fig.add_subplot(sub[1, 0])
    loss = JL[("loss0.1_all_l1", 1)]["rows"]
    for i, (label, rr, colour) in enumerate((("intact", data[("intact", 1)], C["intact"]), ("10% silenced", loss, C["diffuse"]))):
        sm = success_summary([r["reached"] for r in rr])
        lo, hi = np.array(sm["ci"]) * 30
        ax.errorbar(sm["reached"], i, xerr=[[sm["reached"] - lo], [hi - sm["reached"]]], fmt="o" if i == 0 else "X", color=colour,
                    markersize=3.6, capsize=1.5, lw=.8)
        ax.text(31.5, i, "%d/30" % sm["reached"], fontsize=5.6, va="center", color=colour)
        diffrows.append(["closed_loop", label, 1, sm["reached"], sm["n"]])
    ax.set(yticks=[0, 1], xlim=(-1, 35), ylim=(1.6, -.6), xlabel="Targets reached (of 30)")
    ax.set_yticklabels(["intact", "10% silenced"], fontsize=5.4)
    ax.tick_params(labelsize=5.4)
    save(fig, "fig5_ablation")
    csvout("fig5_intervention_check.csv", ["condition", "time_from_onset_s", "median_abs_requested_turn_rad_per_tick", "fraction_of_reaches_with_written_target"], checkrows)
    csvout("fig5_error_courses.csv", ["condition", "quantity", "time_s", "q25", "median", "q75"], courserows)
    csvout("fig5_success.csv", ["condition", "kernel_seed", "n", "reached", "wilson_low", "wilson_high", "reachable_witnesses"], successrows)
    csvout("fig5_absorption.csv", ["condition", "kernel_seed", "target_index", "disabled_joint_one_based", "counterfactual_x_tolerance",
                                   "actual_x_tolerance", "reached", "reachable_witness"], absrows)
    csvout("fig5_diffuse.csv", ["measure", "group", "seed", "value_a", "value_b"], diffrows)
    return counts, successrows


# ----------------------------------------------------------------------------------------------------
# Supplementary companions
# ----------------------------------------------------------------------------------------------------
def hand_paths(fig, cell, rows_=6, cols=1):
    """One Gazebo reach (target pair 0) by every controller of Fig. 3, as hand paths on common axes (Fig. 1a)."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    ctrls = [(key, rows("gazebo", method="matrix") if key == "rak" else rows("gazebo", method="lead") if key == "rak_pe"
              else baseline_rows(key)) for key, _ in CONTROLLERS]
    paths, export = {}, []
    for key, rs in ctrls:
        r = next(r for r in rs if r["index"] == 0)
        # from the first logged sample: in Gazebo the arm is placed only to within a tolerance of the prescribed start
        pp = np.array([position(CHAIN, np.array(q)) for q in r["q"]]) * 1000
        paths[key] = (r, pp)
        export += [[key, t, *p] for t, p in zip(r["t"], pp)]
    goal = 1000 * np.array(paths["rak"][0]["goal_position"])
    for key, (r, _) in paths.items():           # the same target for every controller
        assert np.allclose(1000 * np.array(r["goal_position"]), goal, atol=.01), key
    allp = np.vstack([pp for _, pp in paths.values()] + [goal[None]])
    lo, hi = allp.min(0), allp.max(0)
    pad = .06 * (hi - lo)
    lims = list(zip(lo - pad, hi + pad))
    grid = GridSpecFromSubplotSpec(rows_, cols, subplot_spec=cell, wspace=.0, hspace=.38)
    last = len(CONTROLLERS) - 1
    for i, (key, label) in enumerate(CONTROLLERS):
        r, pp = paths[key]
        ax = fig.add_subplot(grid[i // cols, i % cols], projection="3d")
        colour = CTRL_COLOUR[key]
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):   # solid white background, dark frame
            axis.set_pane_color((1, 1, 1, 1))
            axis.pane.set_edgecolor("#8A969C")
            axis._axinfo["grid"].update(color="#D5DBDE", linewidth=.4)
            axis.line.set_color("#4A555B")
            axis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        artists = [ax.plot(*pp.T, color=colour, lw=1.4, solid_capstyle="round")[0],
                   ax.scatter(*pp[0], s=14, facecolors="white", edgecolors="#2F3A40", linewidths=.8, depthshade=False),
                   ax.scatter(*pp[-1], s=12, color=colour, edgecolors="#1F2A30", linewidths=.4, depthshade=False),
                   ax.scatter(*goal, marker="*", s=48, c=C["goal"], edgecolors="#5A3E06", linewidths=.4, depthshade=False)]
        for artist in artists:
            artist.set_clip_on(False)
        ax.set(xlim=lims[0], ylim=lims[1], zlim=lims[2])
        ax.view_init(elev=23, azim=-55)
        ax.set_box_aspect((1.45, 1.2, 1.05), zoom=1.3)   # fill the slot each controller gets
        ax.tick_params(pad=-3, labelsize=4.6, colors="#4A555B")
        # axis values once (every panel shares the same axes), on the last panel
        if i == last:
            ax.set_xlabel("x (mm)", fontsize=4.6, labelpad=-9)
            ax.set_ylabel("y (mm)", fontsize=4.6, labelpad=-9)
            ax.set_zlabel("z (mm)", fontsize=4.6, labelpad=-10)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        above = matplotlib.transforms.offset_copy(ax.transAxes, fig=fig, y=7.0, units="points")
        ax.text2D(.5, 1.0, SHORT_CTRL[key], transform=above, ha="center", va="bottom", fontsize=6.4,
                  fontweight="bold", color="#1F2A30" if key == "rak" else colour)
        ax.text2D(.5, 1.0, "path %.2f× straight line" % r["path_ratio"], transform=ax.transAxes, ha="center",
                  va="bottom", fontsize=5.4, color="#3A454B")
    csvout("fig1_hand_paths_target0.csv", ["controller", "time_s", "x_mm", "y_mm", "z_mm"], export)


def make_fig1_hand_paths():
    """Fig. 1a, drawn on its own: the architecture step places it beside the circuit (python -m analysis architecture)."""
    fig = plt.figure(figsize=(52.9 / 25.4, 153.1 / 25.4))   # the slot down the left of Fig. 1 (HAND_SLOT_MM)
    gs = fig.add_gridspec(1, 1, left=.03, right=.97, bottom=.085, top=.96)
    hand_paths(fig, gs[0, 0])
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor="#2F3A40", markersize=3, label="start"),
                        Line2D([], [], marker="*", ls="none", color=C["goal"], markersize=5, label="goal"),
                        Line2D([], [], marker="o", ls="none", color="#59646A", markersize=2.8, label="end")],
               loc="lower center", bbox_to_anchor=(.5, -.012), ncol=3, frameon=False, fontsize=5.2, handletextpad=.2,
               columnspacing=.8)
    save(fig, "fig1a_hand_paths")


def make_supp_joint_loss():
    """Per-joint outcomes, intervention drift and the exact-loop joint map."""
    seeds = (1, 2, 3)
    fig = plt.figure(figsize=(WIDTH, 112 / 25.4))
    gs = fig.add_gridspec(2, 2, left=.09, right=.985, bottom=.10, top=.92, hspace=.72, wspace=.32)
    ax = fig.add_subplot(gs[0, 0])
    perjoint = []
    for i, c in enumerate(("intact",) + CONDITIONS):
        for k in range(7):
            rr = [r for s in seeds for r in (JL[(c, s)]["rows"] if c != "intact" else intact_rows(s))
                  if JL_ASSIGN[int(r["index"])] == k]
            reached = sum(bool(r["reached"]) for r in rr)
            ax.scatter(k + (i - 1.5) * .17, reached / len(rr), marker=MARK[c], color=C[c], s=12,
                       label=SHORT[c] if k == 0 else None)
            perjoint.append([c, k + 1, len(rr), reached])
    ax.set(xticks=range(7), xticklabels=[str(k + 1) for k in range(7)], xlabel="Disabled joint (base = 1)",
           ylabel="Fraction reached\n(seeds 1-3 pooled)", ylim=(-.03, 1.05))
    ax.legend(frameon=False, ncol=4, fontsize=5.6, loc="lower left")
    panel(ax, "a", "Outcome by disabled joint (12-15 trials per point)")
    ax = fig.add_subplot(gs[0, 1])
    driftrows = []
    for i, c in enumerate(CONDITIONS):
        d = np.array([r["disabled_drift_rad"] for s in seeds for r in JL[(c, s)]["rows"]])
        jitter = np.random.default_rng(i).uniform(-.18, .18, len(d))
        ax.scatter(i + jitter, np.degrees(d), s=6, color=C[c], alpha=.5, linewidths=0)
        ax.plot([i - .25, i + .25], [np.degrees(np.median(d))] * 2, color=C[c], lw=1.8)
        driftrows += [[c, float(x)] for x in d]
    ax.set(xticks=range(3), xticklabels=[SHORT[c] for c in CONDITIONS], ylabel="Disabled-joint movement\nafter onset (°)")
    panel(ax, "b", "How still does the disabled joint stay?")
    jm = read(RAK_RESULTS / "joint_map" / "all.json")
    maprows = []
    for j, task in enumerate(("pose", "position")):
        ax = fig.add_subplot(gs[0, 1 + j])
        rr = [r for r in jm["rows"] if r["task"] == task and r["onset_s"] == 2.0]
        for i, (rule, colour, marker, label) in enumerate((("reachable", "#9AA5AB", "_", "still reachable"),
                                                           ("cascade_lock", C["lock"], "s", "cascade, uninformed"),
                                                           ("cascade_informed", C["allocation"], "D", "cascade, informed"),
                                                           ("dls_informed", C["dls"], "o", "DLS IK, informed"))):
            ys = []
            for k in range(7):
                mine = [r for r in rr if r["joint"] == k]
                v = sum(bool(r["reachability"]["reachable"]) if rule == "reachable" else bool(r["rules"][rule]["reached"]) for r in mine)
                ys.append(v)
                maprows.append([task, k + 1, rule, len(mine), v])
            ax.plot(np.arange(7) + (i - 1.5) * .12, ys, marker=marker, ls="none", color=colour, markersize=3.5 if marker != "_" else 7,
                    mew=1.2 if marker == "_" else .5, label=label)
        ax.set(xticks=range(7), xticklabels=[str(k + 1) for k in range(7)], xlabel="Disabled joint", ylim=(0, 31),
               ylabel="Targets of 30" if j == 0 else "")
        if j == 0:
            ax.legend(frameon=False, ncol=2, fontsize=5.4, loc="lower left")
        panel(ax, "cd"[j], "Exact loop, %s task, joint jammed at 2 s" % task)
    save(fig, "supp_joint_loss")
    csvout("supp_joint_loss_by_joint.csv", ["condition", "disabled_joint_one_based", "trials", "reached"], perjoint)
    csvout("supp_joint_loss_drift.csv", ["condition", "disabled_drift_rad"], driftrows)
    csvout("supp_joint_map.csv", ["task", "disabled_joint_one_based", "rule", "targets", "count"], maprows)


def make_supporting():
    fig = plt.figure(figsize=(WIDTH, 122 / 25.4))
    gs = fig.add_gridspec(2, 2, left=.10, right=.97, bottom=.11, top=.92, hspace=.75, wspace=.45)
    ax = fig.add_subplot(gs[0, 0])
    for i, cond in enumerate(("push", "noise")):
        rr = rows("robustness", condition=cond)
        s = success_summary([r["reached"] for r in rr])
        y = 100 * s["rate"]
        lo, hi = np.asarray(s["ci"]) * 100
        ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], fmt="o", color=C["matrix"], capsize=3)
        ax.text(i, y + 4, "%d/%d" % (s["reached"], s["n"]), ha="center", fontsize=7)
    ax.set(xticks=[0, 1], xticklabels=["0.3 rad push", "100 pA noise"], ylabel="Targets reached (%)", ylim=(45, 105), xlim=(-.5, 1.5))
    panel(ax, "a", "Perturbed reaching")
    ax = fig.add_subplot(gs[0, 1])
    grid = np.arange(0, 12.001, .2)
    e = np.array([np.interp(grid, r["t"], r["error_deg"]) for r in rows("robustness", condition="push")])
    q = np.percentile(e, [25, 50, 75], axis=0)
    ax.fill_between(grid, q[0], q[2], color=C["matrix"], alpha=.15, lw=0)
    ax.plot(grid, q[1], color=C["matrix"])
    ax.axvline(2, ls="--", color="#777777", lw=.8)
    ax.axhline(5, ls=":", color="#AAAAAA", lw=.8)
    ax.set(xlabel="Time (s)", ylabel="Orientation error (°)", xlim=(0, 12))
    panel(ax, "b", "Push recovery: median and IQR")
    ax = fig.add_subplot(gs[1, 0])
    gainrows = []
    for method, off in (("matrix", -.025), ("lead", .025)):
        for gain, bench in ((1., "reach"), (1.5, "gain"), (2., "gain")):
            rr = rows(bench, method=method, gain=gain)
            s = success_summary([r["reached"] for r in rr])
            y = s["rate"] * 100
            lo, hi = np.array(s["ci"]) * 100
            ax.errorbar(gain + off, y, yerr=[[y - lo], [hi - y]], fmt="o", color=C[method], capsize=2, markersize=3,
                        label=LABEL[method] if gain == 1 else None)
            gainrows.append([gain, method, s["n"], s["reached"], s["rate"], *s["ci"]])
    ax.set(xlabel="Control gain (s⁻¹)", ylabel="Targets reached (%)", xticks=[1, 1.5, 2], ylim=(50, 104), xlim=(.8, 2.2))
    ax.legend(frameon=False, loc="lower left")
    panel(ax, "c", "TIAGo 7 gain sweep")
    ax = fig.add_subplot(gs[1, 1])
    deltarows = []
    for gain, bench in ((1., "reach"), (1.5, "gain"), (2., "gain")):
        a, b = matched(bench, gain)
        s = paired([r["path_ratio"] for r in b], [r["path_ratio"] for r in a])
        med, (lo, hi) = s["median_difference"], s["ci"]
        ax.errorbar(gain, med, yerr=[[med - lo], [hi - med]], fmt="o", color=C["lead"], capsize=3, markersize=4)
        deltarows.append([gain, s["n"], med, lo, hi])
    ax.axhline(0, color="#999999", ls=":", lw=.8)
    ax.set(xlabel="Control gain (s⁻¹)", ylabel="With − without predictive\nencoding, path ratio", xticks=[1, 1.5, 2], xlim=(.8, 2.2))
    panel(ax, "d", "Paired path-ratio effect, 95% CI")
    save(fig, "supp_supporting")
    csvout("supp_gain_success.csv", ["gain_s_inverse", "method", "n", "reached", "rate", "ci_low", "ci_high"], gainrows)
    csvout("supp_gain_path_effect.csv", ["gain_s_inverse", "n_pairs", "median_with_minus_without_predictive", "ci_low", "ci_high"], deltarows)
    csvout("supp_push_recovery.csv", ["time_s", "q25_deg", "median_deg", "q75_deg"], zip(grid, *q))


def make_neuron_loss():
    """Read audited per-mask-seed descriptive statistics without pooling 90 poses."""
    audit = read(DATA / "sensitivity_metrics.json")
    # Each audited run reports its median over the same 30 poses; never pool masks.
    records = []
    for row in audit["runs"]:
        lesion = row["lesion"]
        records.append({
            "population": "all" if len(lesion["classes"]) == 3 else lesion["classes"][0],
            "fraction": float(lesion["fraction"]),
            "seed": int(lesion["lesion_seed"]),
            "orientation": float(row["orientation_median_deg"]),
            "position": float(row["position_median_mm"]),
        })
    baseline = read(RAK / "held_pose" / "summary.json")["arms"]["tiago7"]
    bases = {"orientation": baseline["orientation_median_deg"], "position": baseline["position_median_reach"] * .80574 * 1000}
    fig = plt.figure(figsize=(WIDTH, 117 / 25.4))
    gs = fig.add_gridspec(2, 2, left=.11, right=.97, bottom=.11, top=.92, hspace=.74, wspace=.47)
    for j, (quantity, ylabel) in enumerate((("orientation", "Orientation error (°)"), ("position", "Position error (mm)"))):
        ax = fig.add_subplot(gs[0, j])
        ax.axhline(bases[quantity], ls=":", color=C["matrix"], lw=1, label="Intact")
        for seed, marker in ((1, "o"), (2, "s"), (3, "^")):
            rr = sorted([r for r in records if r["population"] == "all" and r["seed"] == seed], key=lambda r: r["fraction"])
            xx = [0] + [r["fraction"] * 100 for r in rr]
            yy = [bases[quantity]] + [r[quantity] for r in rr]
            ax.plot(xx, yy, marker=marker, markersize=3, lw=.85, color=C["lead"], alpha=.55, label="Mask %d" % seed)
        ax.set(xlabel="Neurons silenced (%)", ylabel=ylabel, xticks=[0, 10, 20, 30], ylim=(0, None))
        if j == 0:
            ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1, columnspacing=.8)
        panel(ax, "ab"[j], "Diffuse silencing")
        ax = fig.add_subplot(gs[0, 1 + j])
        ax.axhline(bases[quantity], ls=":", color=C["matrix"], lw=1)
        classes = [("rings", "Joint rings"), ("value", "Value populations"), ("sheets", "Sheet family")]
        for i, (name, label) in enumerate(classes):
            for seed, off, marker in ((1, -.075, "o"), (2, 0, "s"), (3, .075, "^")):
                rr = [r for r in records if r["population"] == name and r["seed"] == seed and np.isclose(r["fraction"], .1)]
                assert len(rr) == 1, (name, seed, rr)
                ax.plot(i + off, rr[0][quantity], marker=marker, color=C["lead"], markersize=4, alpha=.7)
        ax.set(xticks=range(3), xticklabels=[x[1] for x in classes], ylabel=ylabel, xlim=(-.4, 2.4), ylim=(0, None))
        panel(ax, "cd"[j], "Ten per cent within one population family")
    save(fig, "supp_neuron_loss")


def make_closed_loop_loss():
    """One silencing mask and one kernel seed; not a recovery experiment."""
    doc = read(DATA / "closed_loop_loss.json")
    assert doc["n_targets"] == 30 and doc["kernel_seed"] == 1 and doc["lesion_seed"] == 1
    series = [("nominal", "Nominal\ns1", C["matrix"]),
              ("lesion", "10% silent\ns1 / m1", C["lead"])]
    records = {name: sorted(doc[name]["per_target"], key=lambda r: r["target_index"])
               for name, _, _ in series}
    assert [r["target_index"] for r in records["nominal"]] == [r["target_index"] for r in records["lesion"]] == list(range(30))
    fig = plt.figure(figsize=(WIDTH, 64 / 25.4))
    gs = fig.add_gridspec(1, 3, left=.075, right=.985, bottom=.23, top=.78, wspace=.53)
    ax = fig.add_subplot(gs[0, 0])
    for i, (name, label, col) in enumerate(series):
        s = success_summary([r["reached"] for r in records[name]])
        rate = s["rate"] * 100
        lo, hi = np.array(s["ci"]) * 100
        ax.errorbar(i, rate, yerr=[[rate-lo], [hi-rate]], fmt="o", color=col, capsize=3, markersize=4)
        ax.text(i, 109, "%d/30" % s["reached"], color=col, ha="center", fontsize=6)
    ax.set(xticks=[0, 1], xticklabels=[s[1] for s in series], ylim=(-3, 115),
           yticks=[0, 25, 50, 75, 100], xlim=(-.45, 1.45), ylabel="Targets reached (%)")
    panel(ax, "a", "Closed-loop success")

    for j, (field, ylabel, title) in enumerate((("orientation_deg", "Final orientation error (°)", "Final orientation"),
                                              ("position_mm", "Final position error (mm)", "Final position"))):
        ax = fig.add_subplot(gs[0, j+1])
        jitter = np.random.default_rng(32).uniform(-.075, .075, 30)
        core = np.array([r[field] for r in records["nominal"]])
        loss = np.array([r[field] for r in records["lesion"]])
        for off, a, b in zip(jitter, core, loss):
            ax.plot([off, 1+off], [a, b], lw=.45, color="#BCC8CF", alpha=.5, zorder=1)
        for i, (name, label, col) in enumerate(series):
            values = np.array([r[field] for r in records[name]])
            ax.scatter(i+jitter, values, s=9, color=col, alpha=.65, linewidths=0, zorder=2)
            ax.plot([i-.18, i+.18], [np.median(values)]*2, color=col, lw=2.0, zorder=3)
        ax.set(xticks=[0, 1], xticklabels=[s[1] for s in series], ylim=(0, None),
               xlim=(-.45, 1.45), ylabel=ylabel)
        panel(ax, "bc"[j], title)
    save(fig, "supp_closed_loop_loss")
    export = []
    for name, _, _ in series:
        for row in records[name]:
            export.append([name, 1, 0 if name == "nominal" else 1, row["target_index"],
                           int(row["reached"]), row["orientation_deg"], row["position_mm"],
                           row["path_ratio"], row["time_s_censored"]])
    csvout("supp_closed_loop_loss.csv", ["condition", "kernel_seed", "mask_seed", "target_index", "reached",
                                         "orientation_deg", "position_mm", "path_ratio", "time_s_censored"], export)


def make_ablation_all():
    absorption, _ = make_ablation()
    write_json(EXPORT / "ablation_counts.json", {
        "absorption_pooled_seeds": {c: {"would_miss": v[0], "absorbed": v[1], "harmed_although_counterfactual_within": v[2]}
                                    for c, v in absorption.items()},
        "definition": "Counterfactual: the intact reach's final joint angles at the same seed and target, with the disabled joint put back "
                      "at the angle where this trial froze it. Tolerance-scaled error = max(orientation/5 deg, position/%.2f mm)." % TOL_MM})


#: every figure this module draws: name -> (function, output file, what it needs besides results/)
FIGURE_MAKERS = {
    "fig1": (make_fig1_hand_paths, "fig1a_hand_paths", "-; then architecture composes Fig. 1"),
    "supp_reach_events": (make_reach_events, "supp_reach_events", "-"),
    "fig2": (make_representation, "fig2_representation", "derive, activity"),
    "fig3": (make_reaching, "fig3_reaching", "-"),
    "fig4": (make_prediction, "fig4_prediction", "derive"),
    "fig5": (make_ablation_all, "fig5_ablation", "derive"),
    "supp_supporting": (make_supporting, "supp_supporting", "-"),
    "supp_joint_loss": (make_supp_joint_loss, "supp_joint_loss", "-"),
    "supp_neuron_loss": (make_neuron_loss, "supp_neuron_loss", "derive"),
    "supp_closed_loop_loss": (make_closed_loop_loss, "supp_closed_loop_loss", "derive"),
}


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2) + "\n")


def finish():
    path = EXPORT / "inputs_sha256.json"           # merged, so drawing one figure keeps the others' inputs
    known = json.loads(path.read_text()) if path.exists() else {}
    known.update(HASHES)
    write_json(path, dict(sorted(known.items())))
    write_json(EXPORT / "selection_rules.json", {
        "figure1": "Gazebo RAK-model target pair 0, the lowest index, declared before inspection (as in the earlier Fig. 3c). Errors are the stored 200 ms traces; "
                   "goal arrival is recomputed from the prescribed start and goal; first sustained entry and termination are the logged event times.",
        "figure2": "Recorded illustrative forward replay, seed 11, joint 4, xy sheet at 2 and 6 s; no level-cell recording and no orientation trajectory "
                   "recording. Held pose: all 30 poses per chain. Moving error: RAK-model estimate without predictive encoding only, on the shared common path at three speeds.",
        "figure3": "All controllers on the same 30 TIAGo7 pose targets in Gazebo: the RAK model (kernel seed 1) and "
                   "DLS IK, CCD IK, NN dynamics + MPC and PETS (the "
                   "learned models' test reaches split over three Gazebos and put back together). Panel a example: Gazebo RAK-model "
                   "target 0, the lowest index (same trial as Figure 1). Kinematic-plant results in Supplementary Table S3. "
                   "All targets retained; failures never create a sustained-entry event.",
        "figure4": "Panels b-c: forward kinematics on the matched path (goals of speed0.5) at 0.25, 0.5 and 1 rad/s; medians over moving ticks (every recorded path exported). Panels d-e: every controller of Figure 3 on "
                   "the same 30 TIAGo7 Gazebo targets; paired effects are controller minus RAK model (without predictive encoding), median and percentile "
                   "95% interval from 4,000 target-pair resamples; missed targets retain 25 s.",
        "figure5": "Joint unavailability at 2 s, kinematic plant, seeds 1-3, 30 targets each, the same disabled joint per target in every condition and "
                   "seed. Intact reference: dedicated seed-1 run; nominal RAK-model reaching at seeds 2-3. Error courses pool the 90 trials per condition "
                   "(30 targets x 3 seeds) descriptively; success is shown per seed. Diffuse silencing: one mask, kernel seed 1.",
        "neuron_loss": "Every point is a median across the same 30 held poses for one lesion-mask seed. Three masks are not three neural simulation seeds or 90 independent targets.",
        "closed_loop_loss": "One fixed 10% silencing mask applied from network construction, kernel seed 1, same 30 targets as the nominal RAK model."
    })
