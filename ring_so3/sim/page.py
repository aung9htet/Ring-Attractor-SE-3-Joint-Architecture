"""The simulator's page: one HTML document, no external files."""

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ring FK simulator</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1c2330; --muted: #5d6878; --line: #d9dee6; --grid: #e8ebf0;
  --decoded: #2a78d6; --error: #eb6834; --x: #d9443a; --y: #2f9e5b; --z: #2a78d6; --cell: 28,35,48;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #12151b; --panel: #1a1f28; --ink: #e8ecf2; --muted: #9aa5b5; --line: #2f3744; --grid: #252c37;
          --decoded: #5aa2f0; --error: #f08a5d; --x: #f06a60; --y: #53c183; --z: #5aa2f0; --cell: 232,236,242; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: baseline; padding: 14px 16px 6px; }
h1 { font-size: 17px; margin: 0; font-weight: 650; }
header .status { color: var(--muted); font-variant-numeric: tabular-nums; }
main { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 12px; padding: 8px 16px 20px; }
@media (max-width: 900px) { main { grid-template-columns: minmax(0, 1fr); } }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; min-width: 0; }
h2 { font-size: 13px; margin: 0 0 8px; font-weight: 650; color: var(--muted); text-transform: none; }
canvas { display: block; width: 100%; }
#view { height: 380px; cursor: grab; touch-action: none; }
.big { display: flex; flex-wrap: wrap; gap: 6px 24px; align-items: baseline; margin-top: 8px; }
.big b { font-size: 26px; font-variant-numeric: tabular-nums; }
.big span { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--muted); font-size: 12px; margin-top: 6px; }
.swatch { display: inline-block; width: 22px; height: 0; border-top: 2px solid var(--ink); vertical-align: middle; margin-right: 5px; }
.swatch.dash { border-top-style: dashed; }
.row { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 10px; }
button, select { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 5px 11px; cursor: pointer; }
button.on { background: var(--decoded); border-color: var(--decoded); color: #fff; }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--decoded); outline-offset: 2px; }
.joint { display: grid; grid-template-columns: 96px minmax(0, 1fr) 118px; gap: 8px; align-items: center; margin: 3px 0; }
.joint label { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.joint output { font-variant-numeric: tabular-nums; text-align: right; color: var(--muted); }
input[type=range] { width: 100%; accent-color: var(--decoded); }
.chart { height: 100px; margin-bottom: 6px; }
.chart-title { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
#links { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; margin-top: 6px; }
.note { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
</style>
</head>
<body>
<header>
  <h1>Orientation from ring attractors</h1>
  <label>Chain <select id="chain"></select></label>
  <span class="status" id="status">connecting…</span>
</header>
<main>
  <section>
    <h2>The arm, with the true frame and the frame read from the network at every link</h2>
    <canvas id="view"></canvas>
    <div class="legend">
      <span><i class="swatch"></i>true frame</span><span><i class="swatch dash"></i>read from the network</span>
      <span style="color:var(--x)">x</span><span style="color:var(--y)">y</span><span style="color:var(--z)">z</span>
      <span>drag to turn the view</span>
    </div>
    <div class="big">
      <div><b id="err">–</b> <span>orientation error of the end effector now</span></div>
      <div><b id="errmed">–</b> <span>median of the last 10 s</span></div>
    </div>
    <div id="links"></div>
  </section>
  <section>
    <h2>Joint angles</h2>
    <div class="row">
      <button id="manual">Sliders</button><button id="goals">Keep moving to random goals</button><button id="random">One random pose</button>
    </div>
    <div class="row"><label for="speed">Joint speed limit</label>
      <input id="speed" type="range" min="0.1" max="1.5" step="0.05" value="0.5" style="width:180px"><output id="speedv"></output></div>
    <div class="row"><label><input id="lead" type="checkbox"> Write ahead: each ring gets the angle its joint will have about 0.15 s later, which is how late the readout is</label></div>
    <div id="joints"></div>
    <p class="note">A slider sets where a joint should go; the number is where it is now. The arm moves at the speed limit, and every 50 ms its angles are written on the joint rings.</p>
  </section>
  <section>
    <h2>Lift, pitch and yaw of the end effector: true and read from the output rings</h2>
    <div class="legend" style="margin:0 0 6px"><span><i class="swatch"></i>true</span><span><i class="swatch" style="border-color:var(--decoded)"></i>output ring</span></div>
    <div class="chart-title"><span>lift</span><span id="v0"></span></div><canvas class="chart" id="c0"></canvas>
    <div class="chart-title"><span>pitch</span><span id="v1"></span></div><canvas class="chart" id="c1"></canvas>
    <div class="chart-title"><span>yaw</span><span id="v2"></span></div><canvas class="chart" id="c2"></canvas>
    <div class="chart-title"><span>orientation error (deg)</span><span id="v3"></span></div><canvas class="chart" id="c3"></canvas>
    <p class="note">Last 30 s. Lift and yaw have no value when pitch is near ±90°: their rings fall silent there and the line stops.</p>
  </section>
  <section>
    <h2>Spikes of the last 50 ms: joint rings (in) and output rings (out)</h2>
    <canvas id="rings"></canvas>
    <p class="note">One row per ring, one cell per neuron; darker is more spikes. The tick under a joint ring is where the angle was written; under an output ring, the true angle.</p>
  </section>
</main>
<script>
"use strict";
const $ = id => document.getElementById(id);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const deg = r => r * 180 / Math.PI;
let cfg = null, snap = null, hist = [], cam = {az: -0.9, el: 0.35}, dirty = true, sliders = [], lastSent = 0;

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

function buildControls() {
  const sel = $("chain"); sel.innerHTML = "";
  for (const name of cfg.chains) { const o = document.createElement("option"); o.value = o.textContent = name; sel.appendChild(o); }
  sel.value = cfg.chain;
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
  hist = []; snap = null; centre = null;
}
function sendTargets() {
  const now = performance.now(); sliders.forEach(s => s.held = now);
  if (now - lastSent < 40) return; lastSent = now;
  post("/api/target", {q: sliders.map(s => parseFloat(s.input.value))});
}
$("manual").onclick = () => post("/api/mode", {mode: "manual"});
$("goals").onclick = () => post("/api/mode", {mode: "goals"});
$("random").onclick = () => post("/api/random");
$("speed").oninput = e => post("/api/speed", {speed: parseFloat(e.target.value)});
$("lead").onchange = e => post("/api/lead", {on: e.target.checked});
$("chain").onchange = e => { $("status").textContent = "building " + e.target.value + "…"; post("/api/chain", {chain: e.target.value}); };

function onTick(s) {
  if (hist.length && s.t <= hist[hist.length - 1].t) hist = hist.filter(p => p.t < s.t);
  snap = s; hist.push({t: s.t, error: s.error, tr: s.lpy_true, ring: s.lpy_ring});
  while (hist.length && s.t - hist[0].t > 30) hist.shift();
  dirty = true;
}
async function loadHistory() {
  try { const h = await (await fetch("/history")).json(); if (Array.isArray(h) && h.length) { hist = h.filter(p => p.t !== null); dirty = true; } } catch (e) {}
}
function connect() {
  const es = new EventSource("/events");
  es.addEventListener("config", e => { const first = cfg === null; cfg = JSON.parse(e.data); buildControls(); if (first) loadHistory(); });
  es.addEventListener("tick", e => onTick(JSON.parse(e.data)));
  es.onerror = () => { $("status").textContent = "connection lost; retrying…"; };
}
// "?poll" asks for the state ten times a second instead of keeping a stream open (for browsers or proxies
// that do not pass event streams); "?poll&stop=N" stops after N answers, so that a headless browser can
// take a picture of a page that has come to rest.
async function poll(stop) {
  cfg = await (await fetch("/config")).json(); buildControls(); await loadHistory();
  let left = stop || Infinity, last = -1;
  const timer = setInterval(async () => {
    try {
      const s = await (await fetch("/state")).json();
      if (s.chain && s.chain !== cfg.chain) { cfg = await (await fetch("/config")).json(); buildControls(); }
      if (s.t !== undefined && s.t !== last) { last = s.t; onTick(s); }
    } catch (e) { $("status").textContent = "connection lost; retrying…"; }
    if (--left <= 0) clearInterval(timer);
  }, 100);
}

// ---- 3D view ----
let centre = null;                       // middle of the arm, followed slowly so the picture does not jump
function project(p, w, h, scale) {
  const ca = Math.cos(cam.az), sa = Math.sin(cam.az), ce = Math.cos(cam.el), se = Math.sin(cam.el);
  const px = p[0] - centre[0], py = p[1] - centre[1], pz = p[2] - centre[2];
  const x = ca * px - sa * py, y = sa * px + ca * py;
  return [w / 2 + scale * x, h / 2 - scale * (ce * pz + se * y), -se * pz + ce * y];
}
function line(g, a, b, color, width, dash) {
  g.beginPath(); g.setLineDash(dash || []); g.strokeStyle = color; g.lineWidth = width; g.lineCap = "round";
  g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); g.setLineDash([]);
}
function drawView() {
  const [g, w, h] = fit($("view"));
  if (!snap || !cfg) return;
  const pts = snap.points, n = cfg.joints.length;
  const mid = [0, 1, 2].map(a => (Math.min(...pts.map(p => p[a])) + Math.max(...pts.map(p => p[a]))) / 2);
  centre = centre ? centre.map((c, a) => c + 0.05 * (mid[a] - c)) : mid;
  const reach = Math.max(cfg.reach, 0.2), scale = Math.min(w, h) * 0.78 / reach, P = p => project(p, w, h, scale);
  const ink = css("--ink"), muted = css("--muted"), axes = [css("--x"), css("--y"), css("--z")];
  // floor cross and base
  const f = reach * 0.35; line(g, P([-f, 0, 0]), P([f, 0, 0]), css("--line"), 1); line(g, P([0, -f, 0]), P([0, f, 0]), css("--line"), 1);
  for (let k = 0; k + 1 < pts.length; k++) line(g, P(pts[k]), P(pts[k + 1]), muted, 5);
  for (let k = 0; k < pts.length; k++) { const q = P(pts[k]); g.beginPath(); g.fillStyle = ink; g.arc(q[0], q[1], k === pts.length - 1 ? 4.5 : 3.5, 0, 7); g.fill(); }
  const len = reach * 0.13;
  for (let k = 0; k <= n; k++) {
    const o = pts[k + 1], big = k === n, L = big ? len * 1.5 : len;
    for (const [R, dash, width] of [[snap.true[k], null, big ? 3 : 2], [snap.decoded[k], [5, 4], big ? 2.2 : 1.5]]) {
      for (let a = 0; a < 3; a++) {
        const tip = [o[0] + L * (dash ? 1.18 : 1) * R[a], o[1] + L * (dash ? 1.18 : 1) * R[3 + a], o[2] + L * (dash ? 1.18 : 1) * R[6 + a]];
        line(g, P(o), P(tip), axes[a], width, dash);
      }
    }
  }
}
(function () {
  const v = $("view"); let drag = null;
  v.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, cam.az, cam.el]; v.setPointerCapture(e.pointerId); v.style.cursor = "grabbing"; });
  v.addEventListener("pointermove", e => { if (!drag) return; cam.az = drag[2] + (e.clientX - drag[0]) * 0.01; cam.el = Math.max(-1.5, Math.min(1.5, drag[3] + (e.clientY - drag[1]) * 0.01)); dirty = true; });
  v.addEventListener("pointerup", () => { drag = null; v.style.cursor = "grab"; });
})();

// ---- charts ----
function chart(id, lo, hi, series, ticks) {
  const [g, w, h] = fit($(id));
  const x0 = 34, x1 = w - 4, y0 = 9, y1 = h - 9, now = hist.length ? hist[hist.length - 1].t : 0;
  const X = t => x1 - (now - t) / 30 * (x1 - x0), Y = v => y1 - (v - lo) / (hi - lo) * (y1 - y0);
  g.font = "11px system-ui, sans-serif"; g.fillStyle = css("--muted"); g.textBaseline = "middle"; g.textAlign = "right";
  for (const v of ticks) { g.strokeStyle = css("--grid"); g.lineWidth = 1; g.beginPath(); g.moveTo(x0, Y(v)); g.lineTo(x1, Y(v)); g.stroke(); g.fillText(String(v), x0 - 5, Y(v)); }
  for (const s of series) {
    g.strokeStyle = s.color; g.lineWidth = s.width; g.lineJoin = "round"; g.beginPath(); let prev = null;
    for (const p of hist) {
      const v = s.value(p);
      if (v === null || v === undefined || Number.isNaN(v)) { prev = null; continue; }
      if (prev === null || Math.abs(v - prev) > (hi - lo) * 0.5) g.moveTo(X(p.t), Y(v)); else g.lineTo(X(p.t), Y(v));
      prev = v;
    }
    g.stroke();
  }
}
function drawCharts() {
  if (!hist.length) return;
  const ink = css("--ink"), dec = css("--decoded");
  for (let a = 0; a < 3; a++) {
    const half = a === 1 ? 90 : 180;
    chart("c" + a, -half, half, [
      {color: ink, width: 2, value: p => deg(p.tr[a])},
      {color: dec, width: 2, value: p => p.ring[a] === null ? null : deg(p.ring[a])}], [-half, 0, half]);
    if (snap) {
      const r = snap.lpy_ring[a];
      $("v" + a).textContent = "true " + deg(snap.lpy_true[a]).toFixed(1) + "°   ring " + (r === null ? "silent" : deg(r).toFixed(1) + "°");
    }
  }
  const top = Math.max(5, Math.ceil(Math.max(...hist.map(p => p.error)) / 5) * 5);
  chart("c3", 0, top, [{color: css("--error"), width: 2, value: p => p.error}], [0, top / 2, top]);
  if (snap) $("v3").textContent = snap.error.toFixed(2) + "°";
}

// ---- rings ----
function drawRings() {
  if (!snap || !cfg) return;
  const canvas = $("rings"), rows = snap.rings.length + snap.out_rings.length, rowH = 20, gap = 9, labelW = 62;
  canvas.style.height = (rows * (rowH + gap) + 16) + "px";
  const [g, w] = fit(canvas);
  const rgb = css("--cell"), ink = css("--ink"), muted = css("--muted");
  g.font = "11px system-ui, sans-serif"; g.textBaseline = "middle";
  const names = cfg.joints.map(j => j.name.replace("_joint", "")).concat(["lift", "pitch", "yaw"]);
  const all = snap.rings.concat(snap.out_rings);
  all.forEach((counts, r) => {
    const y = 4 + r * (rowH + gap) + (r >= snap.rings.length ? 10 : 0), n = counts.length, cw = (w - labelW) / n;
    const top = Math.max(3, ...counts);
    g.fillStyle = muted; g.textAlign = "left"; g.fillText(names[r], 0, y + rowH / 2);
    g.strokeStyle = css("--line"); g.lineWidth = 1; g.strokeRect(labelW - 0.5, y - 0.5, w - labelW, rowH + 1);
    for (let i = 0; i < n; i++) if (counts[i] > 0) { g.fillStyle = "rgba(" + rgb + "," + (0.18 + 0.82 * counts[i] / top) + ")"; g.fillRect(labelW + i * cw, y, Math.max(cw, 1), rowH); }
    let mark = null;
    if (r < snap.rings.length) mark = snap.written[r];
    else { const a = snap.lpy_true[r - snap.rings.length]; mark = ((a / (2 * Math.PI)) % 1 + 1) % 1 * n; }
    if (mark !== null) { g.fillStyle = r < snap.rings.length ? ink : css("--decoded"); const x = labelW + (mark + 0.5) * cw; g.fillRect(x - 1, y + rowH + 1, 2, 5); }
  });
}

function drawText() {
  if (!snap || !cfg) return;
  $("err").textContent = snap.error.toFixed(2) + "°";
  const recent = hist.filter(p => snap.t - p.t <= 10).map(p => p.error).sort((a, b) => a - b);
  $("errmed").textContent = recent.length ? recent[Math.floor(recent.length / 2)].toFixed(2) + "°" : "–";
  $("links").textContent = "error of each link's frame: " + snap.link_error.map((e, k) => (k + 1) + ": " + e.toFixed(1) + "°").join("   ");
  $("status").textContent = cfg.chain + (cfg.stub ? " · stand-in network (no spikes)" : " · " + cfg.neurons.toLocaleString() + " neurons") +
    " · running at " + snap.rate.toFixed(2) + "× real time · t = " + snap.t.toFixed(1) + " s";
  $("manual").classList.toggle("on", snap.mode === "manual"); $("goals").classList.toggle("on", snap.mode === "goals");
  $("speedv").textContent = snap.speed.toFixed(2) + " rad/s"; if (document.activeElement !== $("speed")) $("speed").value = snap.speed;
  $("lead").checked = !!snap.lead;
  const now = performance.now();
  sliders.forEach((s, k) => {
    if (now - s.held > 600) s.input.value = snap.target[k];
    s.out.textContent = deg(snap.q[k]).toFixed(1) + "°";
  });
}
function frame() {
  try {
    if (dirty) { dirty = false; drawView(); drawCharts(); drawRings(); drawText(); }
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

__all__ = ["PAGE_HTML"]
