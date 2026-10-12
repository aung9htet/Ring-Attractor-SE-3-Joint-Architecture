"""The pose IK simulator's page: one HTML document, no external files (the SO-3 IK page with the position added)."""

IK_PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ring pose IK simulator</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1c2330; --muted: #5d6878; --line: #d9dee6; --grid: #e8ebf0;
  --decoded: #2a78d6; --error: #eb6834; --goal: #8a56c2; --x: #d9443a; --y: #2f9e5b; --z: #2a78d6; --cell: 28,35,48; --goalcell: 138,86,194;
  --m-ring: #2a78d6; --m-cascade: #1c2330; --m-dls: #eb6834; --m-transpose: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #12151b; --panel: #1a1f28; --ink: #e8ecf2; --muted: #9aa5b5; --line: #2f3744; --grid: #252c37;
          --decoded: #5aa2f0; --error: #f08a5d; --goal: #b58ae6; --x: #f06a60; --y: #53c183; --z: #5aa2f0; --cell: 232,236,242; --goalcell: 181,138,230;
          --m-ring: #5aa2f0; --m-cascade: #e8ecf2; --m-dls: #f08a5d; --m-transpose: #53c183; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: baseline; padding: 14px 16px 6px; }
h1 { font-size: 17px; margin: 0; font-weight: 650; }
header .status { color: var(--muted); font-variant-numeric: tabular-nums; }
main { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 12px; padding: 8px 16px 20px; }
@media (max-width: 900px) { main { grid-template-columns: minmax(0, 1fr); } }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; min-width: 0; }
h2 { font-size: 13px; margin: 0 0 8px; font-weight: 650; color: var(--muted); }
canvas { display: block; width: 100%; }
#view { height: 420px; cursor: grab; touch-action: none; }
.big { display: flex; flex-wrap: wrap; gap: 6px 24px; align-items: baseline; margin-top: 8px; }
.big b { font-size: 24px; font-variant-numeric: tabular-nums; }
.big span { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--muted); font-size: 12px; margin-top: 6px; }
.swatch { display: inline-block; width: 22px; height: 0; border-top: 2px solid var(--ink); vertical-align: middle; margin-right: 5px; }
.swatch.dash { border-top-style: dashed; } .swatch.dot { border-top-style: dotted; border-top-width: 3px; }
.row { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 10px; }
button, select { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 5px 11px; cursor: pointer; }
button:disabled, select:disabled { opacity: 0.5; cursor: default; }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--decoded); outline-offset: 2px; }
.joint { display: grid; grid-template-columns: 96px minmax(0, 1fr) 118px; gap: 8px; align-items: center; margin: 3px 0; }
.joint label { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.joint output { font-variant-numeric: tabular-nums; text-align: right; color: var(--muted); }
input[type=range] { width: 100%; accent-color: var(--decoded); }
.chart { height: 130px; margin-bottom: 6px; }
.chart-title { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.note { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
.share { display: grid; grid-template-columns: 96px minmax(0, 1fr) 170px; gap: 8px; align-items: center; margin: 3px 0; }
.share label { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.share output { font-variant-numeric: tabular-nums; text-align: right; color: var(--muted); }
.bar { position: relative; height: 12px; background: var(--grid); border-radius: 3px; }
.bar i { position: absolute; top: 0; bottom: 0; background: var(--goal); border-radius: 3px; }
.bar u { position: absolute; top: -2px; bottom: -2px; left: 50%; width: 1px; background: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; }
td .key { display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 6px; border-radius: 2px; }
.sheets { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.sheets canvas { aspect-ratio: 1 / 1; border: 1px solid var(--line); }
.sheets div { color: var(--muted); font-size: 12px; text-align: center; }
#gzbox { position: relative; background: var(--grid); border-radius: 6px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
#gzbox.live { min-height: 240px; }
#gz { display: block; width: 100%; }
#gznote { color: var(--muted); padding: 12px 16px; text-align: center; font-size: 13px; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<header>
  <h1>Pose inverse kinematics on ring attractors and attractor sheets</h1>
  <label>Chain <select id="chain"></select></label>
  <span class="status" id="status">connecting…</span>
</header>
<main>
  <section>
    <h2 id="viewtitle">The arm and the goal</h2>
    <canvas id="view"></canvas>
    <div class="legend">
      <span><i class="swatch"></i>true frame of the hand</span><span><i class="swatch dash"></i>read from the network</span>
      <span style="color:var(--goal)">● goal, ○ the goal as it travels</span>
      <span style="color:var(--x)">x</span><span style="color:var(--y)">y</span><span style="color:var(--z)">z</span>
      <span>drag elsewhere to turn the view</span>
    </div>
    <div class="big">
      <div><b id="gmm">–</b> <span id="gmmnote">from the goal</span></div>
      <div><b id="gdeg">–</b> <span>turned away from the goal</span></div>
    </div>
    <div class="big">
      <div><b id="left">–</b> <span>that no joint takes</span></div>
      <div><b id="err">–</b> <span>error of the network's readout</span></div>
    </div>
  </section>
  <section>
    <h2>Task, method, robot and goal</h2>
    <div class="row">
      <label>Task <select id="task"></select></label>
      <label>Position compared on <select id="pgoal"></select></label>
      <label>Method <select id="method"></select></label>
      <label>Robot <select id="robot"></select></label>
    </div>
    <div class="row">
      <button id="rgoal">Random goal nearby</button><button id="rfar">Random goal anywhere</button>
      <button id="replay">Same start and goal, next method</button><button id="stop">Stop here</button>
    </div>
    <div id="pos-sliders">
      <div class="joint"><label for="p0">goal x</label><input id="p0" type="range" step="0.002" value="0"><output id="po0"></output></div>
      <div class="joint"><label for="p1">goal y</label><input id="p1" type="range" step="0.002" value="0"><output id="po1"></output></div>
      <div class="joint"><label for="p2">goal z</label><input id="p2" type="range" step="0.002" value="0"><output id="po2"></output></div>
    </div>
    <div id="lpy-sliders">
      <div class="joint"><label for="g0">goal lift</label><input id="g0" type="range" min="-3.1416" max="3.1416" step="0.01" value="0"><output id="go0"></output></div>
      <div class="joint"><label for="g1">goal pitch</label><input id="g1" type="range" min="-1.5708" max="1.5708" step="0.01" value="0"><output id="go1"></output></div>
      <div class="joint"><label for="g2">goal yaw</label><input id="g2" type="range" min="-3.1416" max="3.1416" step="0.01" value="0"><output id="go2"></output></div>
    </div>
    <label id="anchorrow"><input id="ganchor" type="checkbox"> Goal anchor: the goal's three numbers hold the goal packets (off: the goal sheets remember the goal by themselves)</label>
    <p class="note" id="methodnote"></p>
    <h2 style="margin-top:12px">Joint angles (moving a slider takes the arm over by hand; the goal then stays on the hand)</h2>
    <div class="row"><label for="speed">Speed by hand</label>
      <input id="speed" type="range" min="0.1" max="3" step="0.05" value="1" style="width:160px"><output id="speedv"></output></div>
    <div id="joints"></div>
  </section>
  <section>
    <h2>Gazebo</h2>
    <div id="gzbox"><img id="gz" alt="Camera picture of the TIAGo in Gazebo" hidden><div id="gznote"></div></div>
    <p class="note" id="gzline"></p>
  </section>
  <section id="sheetsec">
    <h2>The sheets: the goal's packets and the hand's packets, spikes of the last 50 ms</h2>
    <div class="sheets">
      <div><canvas id="s0"></canvas>x across, y up</div><div><canvas id="s1"></canvas>x across, z up</div><div><canvas id="s2"></canvas>y across, z up</div>
    </div>
    <div class="legend">
      <span>■ hand: forward sheets</span><span style="color:var(--goal)">■ goal: goal sheets</span>
      <span style="color:var(--error)">+ true position of the hand</span><span style="color:var(--goal)">○ goal as written, × final goal</span>
    </div>
    <p class="note" id="offsets"></p>
    <p class="note">One cell per neuron of the 48 x 48 sheets of each plane; goal sheet and forward sheet are drawn on top of each other. The dashed circle is the edge of the workspace. The comparison reads both packets through their ramp readouts and holds goal minus hand.</p>
  </section>
  <section>
    <h2>Distance from the goal, one line per run (the last six)</h2>
    <div class="chart-title"><span>position (mm)</span><span id="rv0"></span></div><canvas class="chart" id="runs-mm"></canvas>
    <div class="chart-title"><span>orientation (deg)</span><span id="rv1"></span></div><canvas class="chart" id="runs-deg"></canvas>
    <table id="runtable"><thead><tr><th>Run</th><th>Robot</th><th>From</th><th>Within the marks after</th><th>Now / final</th><th>Path / straight</th></tr></thead><tbody></tbody></table>
    <p class="note" id="marks"></p>
  </section>
  <section>
    <h2>Each joint's share (bar: share, up to the largest it takes; number: lead of its goal ring)</h2>
    <div id="shares"></div>
    <p class="note" id="sharenote"></p>
    <div class="chart-title" style="margin-top:10px"><span>position of the hand (dark) and of the travelling goal (purple): x solid, y dashed, z dotted</span><span id="v1"></span></div>
    <canvas class="chart" id="xyz"></canvas>
    <div class="chart-title"><span>lift, pitch, yaw: true (dark) and goal on the goal rings (purple)</span><span id="v0"></span></div>
    <canvas class="chart" id="lpy"></canvas>
  </section>
  <section>
    <h2>Spikes of the last 50 ms: each joint's state ring and goal ring, the output rings, the three goal rings</h2>
    <canvas id="rings"></canvas>
    <p class="note">One row per ring, one cell per neuron; darker is more spikes. The tick under a state ring is where the joint angle was written; under an output ring the true angle; under a goal ring the goal that is written on it.</p>
  </section>
</main>
<script>
"use strict";
const $ = id => document.getElementById(id);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const deg = r => r * 180 / Math.PI;
let cfg = null, snap = null, hist = [], cam = {az: -0.9, el: 0.35}, dirty = true, sliders = [], lastSent = 0;
let goalSliders = [], posSliders = [], shareRows = [], goalHeld = 0, goalSent = 0, dragGoal = null, labels = {}, gzFrames = -1, gzLoading = false;

function post(path, body) {
  return fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
}
function fit(canvas) {
  const r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * d) || canvas.height !== Math.round(r.height * d)) {
    canvas.width = Math.round(r.width * d); canvas.height = Math.round(r.height * d);
  }
  const g = canvas.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, r.width, r.height);
  return [g, r.width, r.height];
}
function options(sel, items, value) {
  sel.innerHTML = "";
  for (const [v, text] of items) { const o = document.createElement("option"); o.value = v; o.textContent = text; sel.appendChild(o); }
  sel.value = value;
}
const TASK_LABELS = {pose: "Pose: orientation and position", position: "Position only", orientation: "Orientation only"};
const PGOAL_LABELS = {sheets: "the sheets (goal packets against hand packets)", cells: "numbers (goal against the position cells)"};

function buildControls() {
  options($("chain"), cfg.chains.map(n => [n, n]), cfg.chain);
  options($("task"), cfg.tasks.map(t => [t, TASK_LABELS[t]]), cfg.task);
  options($("pgoal"), cfg.position_goals.map(t => [t, PGOAL_LABELS[t]]), cfg.position_goal);
  labels = {}; for (const m of cfg.methods) labels[m.name] = m.label;
  options($("method"), cfg.methods.map(m => [m.name, m.label]), "ring");
  options($("robot"), cfg.robots.map(r => [r.name, r.label]), cfg.robot);
  for (const o of $("robot").options) o.disabled = o.value === "gazebo" && (!cfg.robots_there.includes("gazebo") || !cfg.gazebo_chains.includes(cfg.chain));
  const box = $("joints"); box.innerHTML = ""; sliders = [];
  cfg.joints.forEach((j, k) => {
    const row = document.createElement("div"); row.className = "joint";
    const label = document.createElement("label"); label.textContent = j.name + " (" + j.axis + ")"; label.htmlFor = "q" + k;
    const input = document.createElement("input"); input.type = "range"; input.id = "q" + k;
    input.min = j.lower; input.max = j.upper; input.step = 0.005; input.value = 0;
    const out = document.createElement("output");
    input.addEventListener("input", sendTargets);
    row.append(label, input, out); box.appendChild(row); sliders.push({input, out, held: 0});
  });
  const sh = $("shares"); sh.innerHTML = ""; shareRows = [];
  cfg.joints.forEach(j => {
    const row = document.createElement("div"); row.className = "share";
    const label = document.createElement("label"); label.textContent = j.name;
    const bar = document.createElement("div"); bar.className = "bar"; const fill = document.createElement("i"); bar.append(fill, document.createElement("u"));
    const out = document.createElement("output");
    row.append(label, bar, out); sh.appendChild(row); shareRows.push({fill, out});
  });
  goalSliders = [0, 1, 2].map(a => $("g" + a));
  posSliders = [0, 1, 2].map(a => { const s = $("p" + a); s.min = cfg.centre[a] - cfg.radius; s.max = cfg.centre[a] + cfg.radius; return s; });
  const hasPos = cfg.task !== "orientation", hasRot = cfg.task !== "position";
  $("pos-sliders").hidden = !hasPos; $("lpy-sliders").hidden = !hasRot;
  $("sheetsec").hidden = !hasPos; $("pgoal").disabled = !hasPos; $("anchorrow").hidden = !cfg.on_sheets;
  $("viewtitle").textContent = "The arm and the goal. " + (hasPos ? "Drag the goal's dot to move it, an arrow tip to move it along one axis" : "") +
    (hasPos && hasRot ? "; " : "") + (hasRot ? (hasPos ? "drag a ring to turn it" : "Drag a ring of the goal to turn it about that axis") : "");
  $("marks").textContent = "Marks: within " + (hasRot ? cfg.reach_deg + "°" : "") + (hasRot && hasPos ? " and " : "") +
    (hasPos ? (cfg.reach_reach * 100).toFixed(0) + "% of the reach (" + (1000 * cfg.reach_reach * cfg.radius).toFixed(0) + " mm)" : "") +
    ", and staying there. A run starts with a random goal, with a drag that is let go, or with \"same start and goal, next method\", which first puts the arm back where the last run started.";
  hist = []; snap = null; centre = null; span = null;
}
function sendTargets() {
  const now = performance.now(); sliders.forEach(s => s.held = now);
  if (now - lastSent < 40) return; lastSent = now;
  post("/api/target", {q: sliders.map(s => parseFloat(s.input.value))});
}
function sendGoalSliders(commit) {
  const now = performance.now(); goalHeld = now;
  if (!commit && now - goalSent < 80) return; goalSent = now;
  post("/api/goal", {lpy: goalSliders.map(g => parseFloat(g.value)), position: posSliders.map(g => parseFloat(g.value)), commit: !!commit});
}
[0, 1, 2].forEach(a => { for (const id of ["g" + a, "p" + a]) { $(id).addEventListener("input", () => sendGoalSliders(false)); $(id).addEventListener("change", () => sendGoalSliders(true)); } });
$("rgoal").onclick = () => post("/api/random_goal", {near: true});
$("rfar").onclick = () => post("/api/random_goal", {near: false});
$("replay").onclick = () => post("/api/replay");
$("stop").onclick = () => post("/api/stop");
$("speed").oninput = e => post("/api/speed", {speed: parseFloat(e.target.value)});
$("method").onchange = e => post("/api/method", {method: e.target.value});
$("robot").onchange = e => post("/api/robot", {robot: e.target.value});
$("task").onchange = e => { $("status").textContent = "rebuilding the network for this task…"; post("/api/task", {task: e.target.value}); };
$("pgoal").onchange = e => { $("status").textContent = "rebuilding the network…"; post("/api/position_goal", {on: e.target.value}); };
$("ganchor").onchange = e => post("/api/goal_anchor", {on: e.target.checked});
$("chain").onchange = e => { $("status").textContent = "building " + e.target.value + "…"; post("/api/chain", {chain: e.target.value}); };

function onTick(s) {
  if (hist.length && s.t <= hist[hist.length - 1].t) hist = hist.filter(p => p.t < s.t);
  snap = s; hist.push({t: s.t, tr: s.lpy_true, goal: s.ik.written, pt: s.pos_true, gpos: s.ik.written_position});
  while (hist.length && s.t - hist[0].t > 30) hist.shift();
  dirty = true;
}
async function loadHistory() {      // a page that is opened or reloaded fills its 30 s charts from what the server kept
  try {
    const h = await (await fetch("/history")).json();
    if (Array.isArray(h) && h.length) { hist = h.filter(p => p.t !== null && p.pt).map(p => ({t: p.t, tr: p.tr, goal: p.goal, pt: p.pt, gpos: p.gpos})); dirty = true; }
  } catch (e) {}
}
function connect() {
  const es = new EventSource("/events");
  es.addEventListener("config", e => { const first = cfg === null; cfg = JSON.parse(e.data); buildControls(); if (first) loadHistory(); });
  es.addEventListener("tick", e => onTick(JSON.parse(e.data)));
  es.onerror = () => { $("status").textContent = "connection lost; retrying…"; };
}
// "?poll" asks for the state ten times a second instead of keeping a stream open; "?poll&stop=N" stops
// after N answers, so that a headless browser can take a picture of a page that has come to rest.
async function poll(stop) {
  cfg = await (await fetch("/config")).json(); buildControls(); await loadHistory();
  let left = stop || Infinity, last = -1;
  const timer = setInterval(async () => {
    try {
      const s = await (await fetch("/state")).json();
      if (s.chain && (s.chain !== cfg.chain || s.ik.task !== cfg.task || s.robot !== cfg.robot)) { cfg = await (await fetch("/config")).json(); buildControls(); }
      if (s.t !== undefined && s.t !== last) { last = s.t; onTick(s); }
    } catch (e) { $("status").textContent = "connection lost; retrying…"; }
    if (--left <= 0) clearInterval(timer);
  }, 100);
}

// ---- small vector helpers (matrices are 9 numbers, row by row) ----
const col = (R, a) => [R[a], R[3 + a], R[6 + a]];
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
const add = (u, v, s) => [u[0] + s * v[0], u[1] + s * v[1], u[2] + s * v[2]];
const UNIT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
function rotAbout(axis, angle) {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1, x = axis[0] / n, y = axis[1] / n, z = axis[2] / n, c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}
function matmul(A, B) {
  const C = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return C;
}

// ---- 3D view ----
let centre = null, span = null, viewScale = 1, goalLen = 0.1, handles = {rings: [], tips: [], centre: null};
function camAxes() {      // screen right, screen up, and the direction toward the viewer, in world coordinates
  const ca = Math.cos(cam.az), sa = Math.sin(cam.az), ce = Math.cos(cam.el), se = Math.sin(cam.el);
  const right = [ca, -sa, 0], up = [se * sa, se * ca, ce];
  return [right, up, cross(right, up)];
}
function project(p, w, h, scale) {
  const [right, up] = camAxes(), d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
  return [w / 2 + scale * dot(d, right), h / 2 - scale * dot(d, up)];
}
function line(g, a, b, color, width, dash) {
  g.beginPath(); g.setLineDash(dash || []); g.strokeStyle = color; g.lineWidth = width; g.lineCap = "round";
  g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); g.setLineDash([]);
}
const goalMatrix = () => dragGoal ? dragGoal.matrix : snap.ik.goal_matrix;
const goalPosition = () => dragGoal ? dragGoal.position : snap.ik.goal_position;
function drawView() {
  const [g, w, h] = fit($("view"));
  if (!snap || !cfg) return;
  const pts = snap.points, n = cfg.joints.length, hasPos = cfg.task !== "orientation", hasRot = cfg.task !== "position";
  // the picture holds the arm and the goal: its middle and its size follow them slowly, so that it does not jump
  const shown = hasPos ? pts.concat([goalPosition(), snap.ik.written_position]) : pts;
  const mid = [0, 1, 2].map(a => (Math.min(...shown.map(p => p[a])) + Math.max(...shown.map(p => p[a]))) / 2);
  centre = centre ? centre.map((c, a) => c + 0.05 * (mid[a] - c)) : mid;
  const reach = Math.max(cfg.reach, 0.2);
  const far = Math.max(0.3 * reach, ...shown.map(p => Math.hypot(p[0] - mid[0], p[1] - mid[1], p[2] - mid[2]))) + 0.22 * reach;
  span = span ? span + 0.05 * (far - span) : far;
  const scale = Math.min(w, h) * 0.5 / span, P = p => project(p, w, h, scale);
  viewScale = scale;
  const ink = css("--ink"), muted = css("--muted"), axes = [css("--x"), css("--y"), css("--z")], goal = css("--goal");
  const f = reach * 0.35; line(g, P([-f, 0, 0]), P([f, 0, 0]), css("--line"), 1); line(g, P([0, -f, 0]), P([0, f, 0]), css("--line"), 1);
  for (let k = 0; k + 1 < pts.length; k++) line(g, P(pts[k]), P(pts[k + 1]), muted, 5);
  for (let k = 0; k < pts.length; k++) { const q = P(pts[k]); g.beginPath(); g.fillStyle = ink; g.arc(q[0], q[1], k === pts.length - 1 ? 4.5 : 3.5, 0, 7); g.fill(); }
  const len = reach * 0.11;
  for (let k = 0; k <= n; k++) {
    const o = pts[k + 1], big = k === n, L = big ? len * 1.5 : len * 0.7;
    for (const [R, dash, width] of [[snap.true[k], null, big ? 3 : 1.5], [snap.decoded[k], [5, 4], big ? 2 : 1]]) {
      for (let a = 0; a < 3; a++) {
        const c = col(R, a), s = L * (dash ? 1.15 : 1);
        line(g, P(o), P(add(o, c, s)), axes[a], width, dash);
      }
    }
  }
  if (snap.pos && snap.pos.every(v => v !== null)) {      // where the forward sheets hold the hand
    const d = P(snap.pos); g.beginPath(); g.strokeStyle = css("--decoded"); g.lineWidth = 2; g.arc(d[0], d[1], 7, 0, 7); g.stroke();
  }
  // the goal: at its own place when the task has a position, else at the hand
  const hand = pts[n + 1], o = hasPos ? goalPosition() : hand, G = goalMatrix(), L = len * 2.0; goalLen = L;
  handles = {rings: [], tips: [], centre: null};
  if (hasPos) {
    line(g, P(hand), P(o), goal, 1.2, [3, 4]);
    const wr = P(snap.ik.written_position); g.beginPath(); g.strokeStyle = goal; g.lineWidth = 2; g.arc(wr[0], wr[1], 6, 0, 7); g.stroke();
    for (let a = 0; a < 3; a++) {       // arrows along the base axes: drag a tip to move the goal along that axis
      const tip = P(add(o, UNIT[a], L * 1.45));
      line(g, P(o), tip, axes[a], dragGoal && dragGoal.kind === "axis" && dragGoal.axis === a ? 3 : 1.5);
      g.beginPath(); g.fillStyle = axes[a]; g.arc(tip[0], tip[1], 5, 0, 7); g.fill();
      handles.tips.push(tip);
    }
    const c = P(o); g.beginPath(); g.fillStyle = goal; g.arc(c[0], c[1], 7, 0, 7); g.fill(); handles.centre = c;
  }
  if (hasRot) {
    for (let a = 0; a < 3; a++) {
      const c = col(G, a), u = col(G, (a + 1) % 3), v = col(G, (a + 2) % 3);
      line(g, P(o), P(add(o, c, L)), axes[a], 3, [1, 6]);
      const path = [];
      for (let i = 0; i <= 48; i++) {
        const t = i / 48 * 2 * Math.PI;
        path.push(P(add(add(o, u, Math.cos(t) * L), v, Math.sin(t) * L)));
      }
      handles.rings.push(path);
      const hot = dragGoal && dragGoal.kind === "ring" && dragGoal.axis === a;
      g.beginPath(); g.strokeStyle = axes[a]; g.globalAlpha = hot ? 1 : 0.55; g.lineWidth = hot ? 3 : 1.6;
      path.forEach((q, i) => i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])); g.stroke(); g.globalAlpha = 1;
    }
  }
  handles.origin = P(o);
}
function hit(x, y) {      // what of the goal is under the pointer: {kind, axis} or null
  if (handles.centre && Math.hypot(handles.centre[0] - x, handles.centre[1] - y) < 10) return {kind: "plane", axis: -1};
  for (let a = 0; a < handles.tips.length; a++) if (Math.hypot(handles.tips[a][0] - x, handles.tips[a][1] - y) < 10) return {kind: "axis", axis: a};
  let best = null, bestD = 9;
  handles.rings.forEach((path, a) => { for (const q of path) { const d = Math.hypot(q[0] - x, q[1] - y); if (d < bestD) { bestD = d; best = a; } } });
  if (best !== null) return {kind: "ring", axis: best};
  if (handles.rings.length && handles.origin && Math.hypot(handles.origin[0] - x, handles.origin[1] - y) < goalLen * viewScale) return {kind: "free", axis: -1};
  return null;
}
(function () {
  const v = $("view"); let drag = null;
  const at = e => { const r = v.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const send = commit => post("/api/goal", {matrix: dragGoal.matrix, position: dragGoal.position, commit});
  v.addEventListener("pointerdown", e => {
    const [x, y] = at(e), what = snap ? hit(x, y) : null;
    v.setPointerCapture(e.pointerId);
    if (what === null) { drag = [e.clientX, e.clientY, cam.az, cam.el]; v.style.cursor = "grabbing"; return; }
    dragGoal = {kind: what.kind, axis: what.axis, matrix: snap.ik.goal_matrix.slice(), position: snap.ik.goal_position.slice(), last: [x, y], sent: 0};
    goalHeld = performance.now(); dirty = true;
  });
  v.addEventListener("pointermove", e => {
    if (dragGoal) {
      const [x, y] = at(e), c = handles.origin, [right, up, toward] = camAxes(), dx = x - dragGoal.last[0], dy = y - dragGoal.last[1];
      if (dragGoal.kind === "ring") {            // about one axis of the goal: by the angle the pointer went round the centre
        const a0 = Math.atan2(-(dragGoal.last[1] - c[1]), dragGoal.last[0] - c[0]), a1 = Math.atan2(-(y - c[1]), x - c[0]);
        let d = a1 - a0; if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI;
        const axis = col(dragGoal.matrix, dragGoal.axis);
        dragGoal.matrix = matmul(rotAbout(axis, d * (dot(axis, toward) >= 0 ? 1 : -1)), dragGoal.matrix);
      } else if (dragGoal.kind === "free") {     // freely: sideways about the screen's up, up and down about its right
        dragGoal.matrix = matmul(matmul(rotAbout(up, dx * 0.012), rotAbout(right, dy * 0.012)), dragGoal.matrix);
      } else if (dragGoal.kind === "plane") {    // in the plane of the screen
        dragGoal.position = add(add(dragGoal.position, right, dx / viewScale), up, -dy / viewScale);
      } else {                                   // along one base axis: as far as the pointer went along that axis on the screen
        const axis = UNIT[dragGoal.axis], sx = dot(axis, right), sy = -dot(axis, up), size = sx * sx + sy * sy;
        if (size > 1e-4) dragGoal.position = add(dragGoal.position, axis, (dx * sx + dy * sy) / (viewScale * size));
      }
      dragGoal.last = [x, y]; goalHeld = performance.now(); dirty = true;
      const now = performance.now();
      if (now - dragGoal.sent > 80) { dragGoal.sent = now; send(false); }
      return;
    }
    if (!drag) { const [x, y] = at(e), what = snap ? hit(x, y) : null; v.style.cursor = what === null ? "grab" : what.kind === "ring" ? "crosshair" : "move"; return; }
    cam.az = drag[2] + (e.clientX - drag[0]) * 0.01; cam.el = Math.max(-1.5, Math.min(1.5, drag[3] + (e.clientY - drag[1]) * 0.01)); dirty = true;
  });
  const end = () => {
    if (dragGoal) { send(true); const done = dragGoal; dragGoal = null; if (snap) { snap.ik.goal_matrix = done.matrix; snap.ik.goal_position = done.position; } setTimeout(() => { dirty = true; }, 300); }
    drag = null; v.style.cursor = "grab"; dirty = true;
  };
  v.addEventListener("pointerup", end); v.addEventListener("pointercancel", end);
})();

// ---- charts ----
function axesFrame(g, w, h, lo, hi, ticks, xmax) {
  const x0 = 38, x1 = w - 6, y0 = 8, y1 = h - 20;
  const X = t => x0 + t / xmax * (x1 - x0), Y = v => y1 - (v - lo) / (hi - lo) * (y1 - y0);
  g.font = "11px system-ui, sans-serif"; g.fillStyle = css("--muted"); g.textBaseline = "middle"; g.textAlign = "right";
  for (const v of ticks) { g.strokeStyle = css("--grid"); g.lineWidth = 1; g.beginPath(); g.moveTo(x0, Y(v)); g.lineTo(x1, Y(v)); g.stroke(); g.fillText(String(v), x0 - 5, Y(v)); }
  g.textAlign = "center"; g.textBaseline = "top";
  for (let t = 0; t <= xmax + 1e-9; t += xmax / 4) g.fillText(t.toFixed(0) + (t === 0 ? " s" : ""), X(t), y1 + 4);
  return [X, Y];
}
function drawRunChart(id, key, mark, step) {
  const [g, w, h] = fit($(id));
  if (!snap) return;
  const runs = snap.runs || [];
  let xmax = 10, top = step;
  for (const r of runs) { if (r.t.length) xmax = Math.max(xmax, r.t[r.t.length - 1]); top = Math.max(top, ...r[key]); }
  xmax = Math.ceil(xmax / 4) * 4; top = Math.ceil(top / step) * step;
  const [X, Y] = axesFrame(g, w, h, 0, top, [0, top / 2, top], xmax);
  if (mark !== null) { g.strokeStyle = css("--muted"); g.setLineDash([4, 4]); g.lineWidth = 1; g.beginPath(); g.moveTo(X(0), Y(mark)); g.lineTo(X(xmax), Y(mark)); g.stroke(); g.setLineDash([]); }
  for (const r of runs) {
    g.strokeStyle = css("--m-" + r.method); g.lineWidth = r.live ? 2.6 : 1.8; g.globalAlpha = r.live ? 1 : 0.8; g.lineJoin = "round";
    g.setLineDash(r.robot === "gazebo" ? [6, 3] : []); g.beginPath();
    r.t.forEach((t, i) => i ? g.lineTo(X(t), Y(r[key][i])) : g.moveTo(X(t), Y(r[key][i]))); g.stroke(); g.globalAlpha = 1; g.setLineDash([]);
  }
}
function drawRuns() {
  if (!snap || !cfg) return;
  const hasPos = cfg.task !== "orientation", hasRot = cfg.task !== "position";
  drawRunChart("runs-mm", "mm", hasPos ? 1000 * cfg.reach_reach * cfg.radius : null, 100);
  drawRunChart("runs-deg", "deg", hasRot ? cfg.reach_deg : null, 30);
  const body = $("runtable").tBodies[0]; body.innerHTML = "";
  for (const r of (snap.runs || []).slice().reverse()) {
    const tr = document.createElement("tr");
    const cells = ["", r.robot === "gazebo" ? "Gazebo" : "light", r.start_mm.toFixed(0) + " mm, " + r.start_deg.toFixed(0) + "°", r.reached_s === null ? "not yet" : r.reached_s.toFixed(1) + " s",
                   r.final_mm === null ? "–" : r.final_mm.toFixed(0) + " mm, " + r.final_deg.toFixed(1) + "°", r.path_ratio === null ? "–" : r.path_ratio.toFixed(2)];
    cells.forEach(text => { const td = document.createElement("td"); td.textContent = text; tr.appendChild(td); });
    const key = document.createElement("span"); key.className = "key"; key.style.background = css("--m-" + r.method);
    tr.firstChild.append(key, document.createTextNode("#" + r.id + " " + (labels[r.method] || r.method) + (r.task !== cfg.task ? " [" + r.task + "]" : "") + (r.live ? " (running)" : "")));
    body.appendChild(tr);
  }
}
function drawSeries(id, lo, hi, ticks, rows) {
  const [g, w, h] = fit($(id));
  if (!hist.length) return;
  const x0 = 38, x1 = w - 4, y0 = 8, y1 = h - 8, now = hist[hist.length - 1].t;
  const X = t => x1 - (now - t) / 30 * (x1 - x0), Y = v => y1 - (v - lo) / (hi - lo) * (y1 - y0);
  g.font = "11px system-ui, sans-serif"; g.fillStyle = css("--muted"); g.textBaseline = "middle"; g.textAlign = "right";
  for (const v of ticks) { g.strokeStyle = css("--grid"); g.lineWidth = 1; g.beginPath(); g.moveTo(x0, Y(v)); g.lineTo(x1, Y(v)); g.stroke(); g.fillText(String(v), x0 - 5, Y(v)); }
  for (const row of rows) {
    g.strokeStyle = row.color; g.lineWidth = row.width; g.setLineDash(row.dash); g.beginPath(); let prev = null;
    for (const p of hist) { const v = row.value(p); if (v === null || v === undefined) { prev = null; continue; }
      if (prev === null || Math.abs(v - prev) > (hi - lo) * 0.5) g.moveTo(X(p.t), Y(v)); else g.lineTo(X(p.t), Y(v)); prev = v; }
    g.stroke();
  }
  g.setLineDash([]);
}
function drawTraces() {
  if (!snap || !cfg) return;
  const dashes = [[], [6, 3], [2, 3]], goal = css("--goal"), ink = css("--ink"), rows = [], lrows = [];
  const span = Math.round(100 * (cfg.radius + Math.max(...cfg.centre.map(Math.abs)))) / 100;
  for (let a = 0; a < 3; a++) {
    rows.push({color: goal, width: 1.5, dash: dashes[a], value: p => p.gpos ? p.gpos[a] : null});
    rows.push({color: ink, width: 1.8, dash: dashes[a], value: p => p.pt ? p.pt[a] : null});
    lrows.push({color: goal, width: 1.5, dash: dashes[a], value: p => p.goal && p.goal[a] !== null ? deg(p.goal[a]) : null});
    lrows.push({color: ink, width: 1.8, dash: dashes[a], value: p => p.tr ? deg(p.tr[a]) : null});
  }
  drawSeries("xyz", -span, span, [-span, 0, span], rows);
  drawSeries("lpy", -180, 180, [-180, 0, 180], lrows);
  const k = snap.ik;
  $("v1").textContent = ["x", "y", "z"].map((nm, a) => nm + " " + snap.pos_true[a].toFixed(3) + " → " + k.goal_position[a].toFixed(3)).join("   ") + " m";
  $("v0").textContent = ["lift", "pitch", "yaw"].map((nm, a) => nm + " " + deg(snap.lpy_true[a]).toFixed(0) + "° → " + (k.goal[a] === null ? "–" : deg(k.goal[a]).toFixed(0) + "°")).join("   ");
}

// ---- sheets ----
function drawSheets() {
  if (!snap || !cfg || cfg.task === "orientation") return;
  const n = cfg.sheet.size, rgb = css("--cell"), grgb = css("--goalcell");
  for (let k = 0; k < 3; k++) {
    const [g, w, h] = fit($("s" + k)), cw = w / n, ch = h / n;
    const X = i => (i + 0.5) * cw, Y = j => h - (j + 0.5) * ch;       // first coordinate across, second up
    g.strokeStyle = css("--line"); g.setLineDash([4, 4]); g.beginPath(); g.arc(w / 2, h / 2, w / 2 / cfg.sheet.extent, 0, 7); g.stroke(); g.setLineDash([]);
    for (const [cells, color] of [[snap.goal_sheets[k], grgb], [snap.sheets[k], rgb]]) {
      const top = Math.max(2, ...cells.map(c => c[1]));
      for (const [index, count] of cells) {
        const i = Math.floor(index / n), j = index % n;
        g.fillStyle = "rgba(" + color + "," + (0.25 + 0.6 * count / top) + ")"; g.fillRect(i * cw, h - (j + 1) * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    const t = snap.sheet_true[k]; g.strokeStyle = css("--error"); g.lineWidth = 2; g.beginPath();
    g.moveTo(X(t[0]) - 7, Y(t[1])); g.lineTo(X(t[0]) + 7, Y(t[1])); g.moveTo(X(t[0]), Y(t[1]) - 7); g.lineTo(X(t[0]), Y(t[1]) + 7); g.stroke();
    const wr = snap.sheet_goal[k], fin = snap.sheet_final[k]; g.strokeStyle = css("--goal");
    g.beginPath(); g.arc(X(wr[0]), Y(wr[1]), 8, 0, 7); g.stroke();
    g.beginPath(); g.moveTo(X(fin[0]) - 5, Y(fin[1]) - 5); g.lineTo(X(fin[0]) + 5, Y(fin[1]) + 5); g.moveTo(X(fin[0]) - 5, Y(fin[1]) + 5); g.lineTo(X(fin[0]) + 5, Y(fin[1]) - 5); g.stroke();
  }
  const k = snap.ik, mm = v => v === null ? "–" : (v >= 0 ? "+" : "") + v.toFixed(0);
  $("offsets").textContent = "Goal minus hand as the network holds it: x " + mm(k.offset_mm[0]) + ", y " + mm(k.offset_mm[1]) + ", z " + mm(k.offset_mm[2]) +
    " mm (exactly: " + k.true_offset_mm.map(mm).join(", ") + " mm)." + (snap.pos_error_mm === null ? "" : " The forward sheets hold the hand " + snap.pos_error_mm.toFixed(0) + " mm from where it is.");
}

// ---- rings ----
function drawRings() {
  if (!snap || !cfg) return;
  const canvas = $("rings"), rowH = 16, gap = 8, labelW = 78, ik = snap.ik;
  const short = name => name.replace("_joint", "");
  const rows = [];
  snap.rings.forEach((counts, k) => {
    rows.push({label: short(cfg.joints[k].name), counts, mark: snap.written[k], color: css("--ink"), tight: true});
    rows.push({label: "  its goal", counts: ik.joint_goal_rings[k], mark: null, color: css("--goal"), goal: true});
  });
  ["lift", "pitch", "yaw"].forEach((name, a) => {
    const n = snap.out_rings[a].length;
    rows.push({label: name, counts: snap.out_rings[a], mark: ((snap.lpy_true[a] / (2 * Math.PI)) % 1 + 1) % 1 * n, color: css("--decoded"), space: a === 0});
  });
  if (cfg.task !== "position") ["lift", "pitch", "yaw"].forEach((name, a) => {
    const n = ik.goal_rings[a].length, w = ik.written[a];
    rows.push({label: "goal " + name, counts: ik.goal_rings[a], mark: w === null ? null : ((w / (2 * Math.PI)) % 1 + 1) % 1 * n, color: css("--goal"), goal: true, space: a === 0});
  });
  let total = 16; rows.forEach(r => { total += rowH + (r.tight ? 2 : gap) + (r.space ? 10 : 0); });
  canvas.style.height = total + "px";
  const [g, w] = fit(canvas);
  const rgb = css("--cell"), muted = css("--muted");
  g.font = "11px system-ui, sans-serif"; g.textBaseline = "middle";
  let y = 4;
  rows.forEach(r => {
    if (r.space) y += 10;
    const n = r.counts.length, cw = (w - labelW) / n, top = Math.max(3, ...r.counts);
    g.fillStyle = r.goal ? css("--goal") : muted; g.textAlign = "left"; g.fillText(r.label, 0, y + rowH / 2);
    g.strokeStyle = css("--line"); g.lineWidth = 1; g.strokeRect(labelW - 0.5, y - 0.5, w - labelW, rowH + 1);
    for (let i = 0; i < n; i++) if (r.counts[i] > 0) { g.fillStyle = "rgba(" + rgb + "," + (0.18 + 0.82 * r.counts[i] / top) + ")"; g.fillRect(labelW + i * cw, y, Math.max(cw, 1), rowH); }
    if (r.mark !== null && r.mark !== undefined) { g.fillStyle = r.color; const x = labelW + (r.mark + 0.5) * cw; g.fillRect(x - 1, y + rowH + 1, 2, r.tight ? 2 : 5); }
    y += rowH + (r.tight ? 2 : gap);
  });
}

// ---- Gazebo panel ----
function drawGazebo() {
  if (!snap || !cfg) return;
  const img = $("gz"), note = $("gznote"), box = $("gzbox"), gzb = snap.gazebo;
  const say = text => { img.hidden = true; note.hidden = false; note.textContent = text; };
  if (!cfg.robots_there.includes("gazebo")) {
    box.className = ""; $("gzline").textContent = "";
    say("Gazebo is not running beside this simulator. Start it with docker/run.bash gazebo up (see docs/ik/README.md), then start the simulator inside it; the light robot works without.");
    return;
  }
  if (!gzb) {
    box.className = ""; $("gzline").textContent = "";
    say(cfg.gazebo_chains.includes(cfg.chain) ? "Gazebo is running. Choose \"TIAGo in Gazebo\" as the robot to move its arm." : "Gazebo has the TIAGo arm only: choose one of the tiago chains.");
    return;
  }
  box.className = "live";
  if (gzb.frames > 0 && gzb.frames !== gzFrames && !gzLoading) {
    gzLoading = true; gzFrames = gzb.frames;
    const next = new Image();
    next.onload = () => { img.src = next.src; img.hidden = false; note.hidden = true; gzLoading = false; };
    next.onerror = () => { gzLoading = false; };
    next.src = "/gazebo.jpg?f=" + gzb.frames;
  }
  if (gzb.frames === 0) say("Gazebo moves the arm, but its camera gives no picture (it needs the container started with --gpu). The arm on the left is drawn from Gazebo's joint angles.");
  const lag = Math.max(...gzb.lag.map(v => Math.abs(deg(v))));
  $("gzline").textContent = "Gazebo time " + gzb.sim_time.toFixed(2) + " s, stepped 50 ms per tick of the network · largest distance of a joint from its target " + lag.toFixed(1) + "°";
}

const METHOD_NOTES = {
  ring: "Ring IK: the spiking network compares goal and decoded pose, hands each joint its share down the arm, and each joint follows its goal ring.",
  cascade: "The same cascade computed exactly from the true joint angles, no neurons. The network still runs and shows its rings and sheets.",
  dls: "Damped least squares, the usual method: the smallest joint motion that makes turn and move. Exact joint angles, no neurons.",
  transpose: "Jacobian transpose: every joint moves by its part of the whole error. Exact joint angles, no neurons."
};
function drawText() {
  if (!snap || !cfg) return;
  const k = snap.ik, now = performance.now(), hasPos = cfg.task !== "orientation", hasRot = cfg.task !== "position";
  $("gmm").textContent = k.goal_error_mm.toFixed(0) + " mm";
  $("gmmnote").textContent = "from the goal (" + (k.goal_error_mm / 10 / cfg.radius).toFixed(1) + "% of the reach)" + (hasPos ? "" : ", not asked for");
  $("gdeg").textContent = k.goal_error.toFixed(1) + "°";
  $("left").textContent = (hasPos && k.left_mm !== null ? k.left_mm.toFixed(0) + " mm" : "") + (hasPos && hasRot ? ", " : "") + (hasRot && k.left_deg !== null ? k.left_deg.toFixed(1) + "°" : "");
  $("err").textContent = (snap.pos_error_mm === null ? "–" : snap.pos_error_mm.toFixed(0) + " mm") + ", " + snap.error.toFixed(1) + "°";
  const modes = {ik: "going to the goal", hold: "at rest", manual: "moved by hand", placing: "going back to the start pose"};
  $("status").textContent = cfg.chain + (cfg.stub ? " · stand-in network (no spikes)" : " · " + cfg.neurons.toLocaleString() + " neurons") +
    (snap.robot === "gazebo" ? " · TIAGo in Gazebo" : "") +
    " · " + (modes[snap.mode] || snap.mode) + " · " + snap.rate.toFixed(2) + "× real time · t = " + snap.t.toFixed(1) + " s";
  if (document.activeElement !== $("method")) $("method").value = snap.method;
  if (document.activeElement !== $("robot")) $("robot").value = snap.robot;
  $("methodnote").textContent = METHOD_NOTES[snap.method] || "";
  $("sharenote").textContent = snap.method === "ring" ? "Shares as the network holds them." : "Shares as the chosen method computes them; the lead is still that of the network's goal rings.";
  $("speedv").textContent = snap.speed.toFixed(2) + " rad/s"; if (document.activeElement !== $("speed")) $("speed").value = snap.speed;
  $("replay").disabled = !(snap.runs && snap.runs.length);
  $("ganchor").checked = !!k.goal_anchor;
  if (now - goalHeld > 800) {
    goalSliders.forEach((g, a) => { if (k.goal[a] !== null) g.value = k.goal[a]; });
    posSliders.forEach((g, a) => { g.value = k.goal_position[a]; });
  }
  goalSliders.forEach((g, a) => { $("go" + a).textContent = k.goal[a] === null ? "" : deg(parseFloat(g.value)).toFixed(0) + "°"; });
  posSliders.forEach((g, a) => { $("po" + a).textContent = parseFloat(g.value).toFixed(3) + " m"; });
  shareRows.forEach((r, j) => {
    const share = k.shares[j] || 0, part = Math.max(-1, Math.min(1, share / cfg.caps[j])) * 50;
    r.fill.style.left = (part >= 0 ? 50 : 50 + part) + "%"; r.fill.style.width = Math.abs(part) + "%";
    r.out.textContent = deg(share).toFixed(1) + "°, lead " + (k.lead[j] === null ? "–" : k.lead[j].toFixed(1)) + " idx";
  });
  sliders.forEach((s, j) => {
    if (now - s.held > 600) s.input.value = snap.mode === "manual" ? snap.target[j] : snap.q[j];
    s.out.textContent = deg(snap.q[j]).toFixed(1) + "°";
  });
}
function frame() {
  try {
    if (dirty) { dirty = false; drawView(); drawRuns(); drawGazebo(); drawSheets(); drawTraces(); drawRings(); drawText(); }
  } finally { requestAnimationFrame(frame); }
}
window.addEventListener("resize", () => { dirty = true; });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { dirty = true; });
const params = new URLSearchParams(location.search);
if (params.has("poll")) poll(parseInt(params.get("stop") || "0", 10)); else connect();
frame();
</script>
</body>
</html>
"""

__all__ = ["IK_PAGE_HTML"]
