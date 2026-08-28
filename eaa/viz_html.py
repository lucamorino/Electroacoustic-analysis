"""A self-contained interactive HTML view of a descriptor table.

Writes one file with the data inlined -- no CDN, no server, no build step, so
it keeps working on a machine with no network and can be mailed to a
collaborator.  What it adds over the static plots is the thing that matters
when a run has four hundred columns: you can show and hide descriptors as you
read, and a shared crosshair reports every visible descriptor at one instant.

Same layout logic as the matplotlib side: small multiples by default, because
a spectral centroid in Hz and a roughness in asper do not belong on one axis.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .table import segment_times

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --series: #2a78d6;
  --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --series: #3987e5;
    --border: rgba(255,255,255,0.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --page: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --series: #3987e5;
  --border: rgba(255,255,255,0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--text-primary);
  font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1400px; margin: 0 auto; padding: 20px 20px 60px; }
header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 4px; }
h1 { font-size: 16px; font-weight: 600; margin: 0; }
.meta { color: var(--text-secondary); font-size: 12px; }
.spacer { flex: 1; }
button {
  font: inherit; font-size: 12px; color: var(--text-secondary);
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 10px; cursor: pointer;
}
button:hover { color: var(--text-primary); }
button[aria-pressed="true"] { color: var(--text-primary); border-color: var(--axis); }

/* One filter row above everything it scopes. */
.controls {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; margin: 14px 0 18px;
}
.controls-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
input[type="search"] {
  font: inherit; font-size: 12px; padding: 5px 9px; min-width: 220px;
  color: var(--text-primary); background: var(--page);
  border: 1px solid var(--border); border-radius: 6px;
}
.count { color: var(--muted); font-size: 12px; }
.groups { display: flex; flex-direction: column; gap: 2px; max-height: 320px; overflow-y: auto; }
details { border-top: 1px solid var(--grid); }
details:first-child { border-top: none; }
summary {
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); padding: 6px 2px; cursor: pointer; list-style-position: inside;
}
summary:hover { color: var(--text-primary); }
summary b { color: var(--series); font-weight: 600; }
.boxes { display: flex; flex-wrap: wrap; gap: 2px 14px; }
label.box {
  display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
  font-size: 12px; color: var(--text-secondary);
  padding: 3px 4px; border-radius: 4px; min-height: 24px;
}
label.box:hover { background: var(--page); color: var(--text-primary); }
label.box input { accent-color: var(--series); margin: 0; }

.panels { position: relative; }
.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; margin-bottom: 8px; padding: 8px 10px 2px;
}
.panel-title { font-size: 12px; color: var(--text-primary); margin: 0 0 2px 4px; }
.panel-title span { color: var(--muted); font-weight: 400; }
svg { display: block; width: 100%; }
.axis-band {
  padding: 2px 10px 6px; position: sticky; bottom: 0;
  background: var(--page); border-top: 1px solid var(--border);
}
.empty { color: var(--muted); padding: 40px 0; text-align: center; }

#tip {
  position: fixed; pointer-events: none; z-index: 10; opacity: 0;
  transition: opacity .08s; background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px;
  padding: 8px 10px; font-size: 12px; min-width: 170px;
  box-shadow: 0 4px 14px rgba(0,0,0,.14);
}
#tip .t-time { color: var(--text-primary); font-weight: 600; margin-bottom: 5px; }
#tip .t-row { display: flex; justify-content: space-between; gap: 14px; color: var(--text-secondary); }
#tip .t-row b { color: var(--text-primary); font-weight: 500; font-variant-numeric: tabular-nums; }

table { border-collapse: collapse; font-size: 12px; width: 100%; }
th, td {
  text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums; white-space: nowrap;
}
th { color: var(--muted); font-weight: 500; text-align: right; position: sticky; top: 0; background: var(--surface); }
th:first-child, td:first-child { text-align: left; }
.table-wrap { overflow: auto; max-height: 70vh; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; padding: 0 10px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>__TITLE__</h1>
    <div class="meta">__META__</div>
    <div class="spacer"></div>
    <button id="view-toggle" aria-pressed="false">Table view</button>
    <button id="theme-toggle">Theme</button>
  </header>

  <div class="controls">
    <div class="controls-row">
      <input type="search" id="filter" placeholder="Filter descriptors…" aria-label="Filter descriptors">
      <button id="all">Show all</button>
      <button id="none">Hide all</button>
      <button id="reset">Reset</button>
      <span class="count" id="count"></span>
    </div>
    <div class="groups" id="groups"></div>
  </div>

  <div id="chart-view">
    <div class="panels" id="panels"></div>
    <div class="axis-band"><svg id="axis" height="24" role="img" aria-label="time axis"></svg></div>
  </div>
  <div id="table-view" hidden><div class="table-wrap"><table id="table"></table></div></div>
</div>
<div id="tip" role="status" aria-live="polite"></div>

<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const M = { left: 66, right: 14, top: 6, bottom: 6 };
const PANEL_H = 88;

const state = {
  visible: new Set(DATA.initial),
  // Panels appear in the order they were asked for, not in CSV column order.
  order: DATA.initial.slice(),
  filter: "",
  table: false,
};
function addVisible(name) {
  state.visible.add(name);
  if (!state.order.includes(name)) state.order.push(name);
}

const el = (id) => document.getElementById(id);
const svgEl = (name, attrs) => {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
};

function fmtTime(s) {
  s = Math.max(0, s);
  const m = Math.floor(s / 60), r = s - m * 60;
  return m + ":" + (r < 10 ? "0" : "") + r.toFixed(s < 60 ? 1 : 0).replace(/\.0$/, "");
}
function fmtValue(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  return v.toPrecision(3);
}

/* ---- controls -------------------------------------------------------- */
function buildControls() {
  const host = el("groups");
  host.innerHTML = "";
  for (const [group, names] of Object.entries(DATA.groups)) {
    const shown = names.filter((n) => n.toLowerCase().includes(state.filter));
    if (!shown.length) continue;
    const active = shown.filter((n) => state.visible.has(n)).length;
    // Collapsed by default: a run can carry 200 bark-band columns, and an
    // open list of them buries the groups you actually came for.
    const wrap = document.createElement("details");
    wrap.open = active > 0 || !!state.filter;
    const head = document.createElement("summary");
    head.innerHTML = group + " (" + shown.length + ")" +
      (active ? " · <b>" + active + " shown</b>" : "");
    wrap.appendChild(head);
    const boxes = document.createElement("div");
    boxes.className = "boxes";
    for (const name of shown) {
      const label = document.createElement("label");
      label.className = "box";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = state.visible.has(name);
      input.addEventListener("change", () => {
        input.checked ? addVisible(name) : state.visible.delete(name);
        render();
      });
      label.appendChild(input);
      label.appendChild(document.createTextNode(DATA.short[name]));
      label.title = name;
      boxes.appendChild(label);
    }
    wrap.appendChild(boxes);
    host.appendChild(wrap);
  }
  el("count").textContent =
    state.visible.size + " of " + DATA.names.length + " descriptors shown";
}

/* ---- chart ----------------------------------------------------------- */
function visibleNames() {
  return state.order.filter((n) => state.visible.has(n));
}

function drawPanels() {
  const host = el("panels");
  host.innerHTML = "";
  const names = visibleNames();
  if (!names.length) {
    host.innerHTML = '<div class="empty">Nothing selected — tick a descriptor above.</div>';
    return;
  }
  const width = host.clientWidth || 900;
  for (const name of names) {
    const panel = document.createElement("div");
    panel.className = "panel";
    const title = document.createElement("p");
    title.className = "panel-title";
    const [lo, hi] = DATA.range[name];
    title.innerHTML = name + " <span>· " + fmtValue(lo) + " to " + fmtValue(hi) + "</span>";
    panel.appendChild(title);

    const svg = svgEl("svg", { height: PANEL_H, viewBox: `0 0 ${width} ${PANEL_H}` });
    svg.dataset.name = name;
    const plotW = width - M.left - M.right;
    const plotH = PANEL_H - M.top - M.bottom;
    const x = (t) => M.left + ((t - DATA.t0) / (DATA.t1 - DATA.t0)) * plotW;
    const y = (v) => M.top + plotH - ((v - lo) / (hi - lo || 1)) * plotH;

    // Recessive hairline grid: three y rules, solid.
    for (const frac of [0, 0.5, 1]) {
      const value = lo + frac * (hi - lo);
      svg.appendChild(svgEl("line", {
        x1: M.left, x2: width - M.right, y1: y(value), y2: y(value),
        stroke: "var(--grid)", "stroke-width": 1,
      }));
      const label = svgEl("text", {
        x: M.left - 8, y: y(value) + 3.5, "text-anchor": "end",
        fill: "var(--muted)", "font-size": 10,
      });
      label.textContent = fmtValue(value);
      svg.appendChild(label);
    }

    // Time gridlines, so a moment can be located without scrolling to the axis.
    for (const t of timeTicks()) {
      svg.appendChild(svgEl("line", {
        x1: x(t), x2: x(t), y1: M.top, y2: M.top + plotH,
        stroke: "var(--grid)", "stroke-width": 1,
      }));
    }

    // Steps: a segment's value describes an extent, not an instant.
    let d = "";
    const values = DATA.values[name];
    for (let i = 0; i < values.length; i++) {
      d += (i ? "L" : "M") + x(DATA.starts[i]) + " " + y(values[i]) +
           "L" + x(DATA.ends[i]) + " " + y(values[i]);
    }
    svg.appendChild(svgEl("path", {
      d, fill: "none", stroke: "var(--series)", "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
    svg.appendChild(svgEl("line", {
      class: "cross", x1: 0, x2: 0, y1: M.top, y2: M.top + plotH,
      stroke: "var(--axis)", "stroke-width": 1, opacity: 0,
    }));
    panel.appendChild(svg);
    host.appendChild(panel);
  }
  drawAxis(width);
}

function timeTicks() {
  const span = DATA.t1 - DATA.t0;
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];
  const step = steps.find((s) => span / s <= 10) || 3600;
  const out = [];
  for (let t = Math.ceil(DATA.t0 / step) * step; t <= DATA.t1; t += step) out.push(t);
  return out;
}

function drawAxis(width) {
  const svg = el("axis");
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${width} 24`);
  const plotW = width - M.left - M.right;
  const span = DATA.t1 - DATA.t0;
  svg.appendChild(svgEl("line", {
    x1: M.left, x2: width - M.right, y1: 1, y2: 1,
    stroke: "var(--axis)", "stroke-width": 1,
  }));
  for (const t of timeTicks()) {
    const px = M.left + ((t - DATA.t0) / span) * plotW;
    const text = svgEl("text", {
      x: px, y: 15, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10,
    });
    text.textContent = fmtTime(t);
    svg.appendChild(text);
  }
}

/* ---- crosshair + tooltip --------------------------------------------- */
function segmentAt(t) {
  let lo = 0, hi = DATA.starts.length - 1;
  if (t <= DATA.starts[0]) return 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (DATA.starts[mid] <= t) lo = mid + 1; else hi = mid - 1;
  }
  return Math.max(0, Math.min(hi, DATA.starts.length - 1));
}

function onMove(event) {
  const panels = el("panels");
  const first = panels.querySelector("svg");
  if (!first) return hideTip();
  const box = first.getBoundingClientRect();
  const scale = box.width / first.viewBox.baseVal.width;
  const px = event.clientX - box.left;
  const plotL = M.left * scale, plotR = box.width - M.right * scale;
  if (px < plotL || px > plotR) return hideTip();

  const t = DATA.t0 + ((px - plotL) / (plotR - plotL)) * (DATA.t1 - DATA.t0);
  const i = segmentAt(t);
  for (const svg of panels.querySelectorAll("svg")) {
    const line = svg.querySelector(".cross");
    if (!line) continue;
    const vx = px / scale;
    line.setAttribute("x1", vx);
    line.setAttribute("x2", vx);
    line.setAttribute("opacity", 1);
  }

  const tip = el("tip");
  let html = '<div class="t-time">' + fmtTime(DATA.starts[i]) + " – " +
             fmtTime(DATA.ends[i]) + " <span style=\"color:var(--muted)\">#" + i + "</span></div>";
  for (const name of visibleNames()) {
    html += '<div class="t-row"><span>' + DATA.short[name] + "</span><b>" +
            fmtValue(DATA.values[name][i]) + "</b></div>";
  }
  tip.innerHTML = html;
  tip.style.opacity = 1;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let left = event.clientX + 16, top = event.clientY + 16;
  if (left + w > window.innerWidth - 8) left = event.clientX - w - 16;
  if (top + h > window.innerHeight - 8) top = Math.max(8, window.innerHeight - h - 8);
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}

function hideTip() {
  el("tip").style.opacity = 0;
  for (const line of document.querySelectorAll(".cross")) line.setAttribute("opacity", 0);
}

/* ---- table view (every value reachable without hovering) -------------- */
function drawTable() {
  const names = visibleNames();
  const rows = [];
  rows.push("<thead><tr><th>#</th><th>start</th><th>end</th>" +
    names.map((n) => "<th>" + DATA.short[n] + "</th>").join("") + "</tr></thead>");
  const body = [];
  for (let i = 0; i < DATA.starts.length; i++) {
    body.push("<tr><td>" + i + "</td><td>" + fmtTime(DATA.starts[i]) + "</td><td>" +
      fmtTime(DATA.ends[i]) + "</td>" +
      names.map((n) => "<td>" + fmtValue(DATA.values[n][i]) + "</td>").join("") + "</tr>");
  }
  rows.push("<tbody>" + body.join("") + "</tbody>");
  el("table").innerHTML = rows.join("");
}

function render() {
  buildControls();
  if (state.table) drawTable(); else drawPanels();
}

/* ---- wiring ---------------------------------------------------------- */
el("filter").addEventListener("input", (e) => {
  state.filter = e.target.value.toLowerCase();
  buildControls();
});
el("all").addEventListener("click", () => {
  DATA.names.forEach(addVisible);
  render();
});
el("none").addEventListener("click", () => { state.visible.clear(); render(); });
el("reset").addEventListener("click", () => {
  state.visible = new Set(DATA.initial);
  state.order = DATA.initial.slice();
  el("filter").value = ""; state.filter = "";
  render();
});
el("view-toggle").addEventListener("click", (e) => {
  state.table = !state.table;
  e.target.setAttribute("aria-pressed", String(state.table));
  e.target.textContent = state.table ? "Chart view" : "Table view";
  el("chart-view").hidden = state.table;
  el("table-view").hidden = !state.table;
  render();
});
el("theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
});
el("panels").addEventListener("mousemove", onMove);
el("panels").addEventListener("mouseleave", hideTip);
window.addEventListener("resize", () => { if (!state.table) drawPanels(); });

render();
</script>
</body>
</html>
"""


def _short(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[1:]) if len(parts) > 2 else name


def write_html(
    df: pd.DataFrame,
    columns: Sequence[str],
    path: str,
    title: str = "",
    initial: Sequence[str] = (),
    smooth: int = 0,
) -> str:
    """Write the interactive page.  ``columns`` are offered, ``initial`` shown."""
    starts, ends = segment_times(df)
    columns = [c for c in columns if c in df.columns]
    if not columns:
        raise ValueError("no descriptor columns to plot")

    values: Dict[str, List[float]] = {}
    ranges: Dict[str, List[float]] = {}
    groups: Dict[str, List[str]] = {}
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        if smooth and smooth > 1:
            series = series.rolling(smooth, center=True, min_periods=1).median()
        array = series.to_numpy(dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.06 * (hi - lo)
        values[column] = [None if not np.isfinite(v) else round(float(v), 6) for v in array]
        ranges[column] = [round(lo - pad, 6), round(hi + pad, 6)]
        groups.setdefault(column.split(".")[0], []).append(column)

    # Put the small, interpretable groups first; the wide band/cepstral ones
    # last, where their hundreds of columns do not bury everything else.
    order = ["dynamics", "psycho", "spectral", "noisiness", "harmonic", "tonal"]
    groups = dict(
        sorted(groups.items(), key=lambda kv: (order.index(kv[0])
               if kv[0] in order else len(order), kv[0]))
    )

    names = list(values)
    shown = [c for c in (initial or names[:6]) if c in values] or names[:6]
    payload = {
        "names": names,
        "groups": groups,
        "short": {c: _short(c) for c in names},
        "initial": shown,
        "values": values,
        "range": ranges,
        "starts": [round(float(v), 4) for v in starts],
        "ends": [round(float(v), 4) for v in ends],
        "t0": round(float(starts[0]), 4),
        "t1": round(float(ends[-1]), 4),
    }
    meta = (
        f"{len(df)} segments · {float(ends[-1]) - float(starts[0]):.1f} s · "
        f"{len(names)} descriptors · time in m:ss"
    )
    html = (
        _TEMPLATE.replace("__DATA__", json.dumps(payload, allow_nan=False))
        .replace("__TITLE__", title or os.path.basename(path))
        .replace("__META__", meta)
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
