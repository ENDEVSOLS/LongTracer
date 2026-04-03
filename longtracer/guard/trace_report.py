"""
Trace Report — Rich console visualization, JSON export, and HTML export.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime as _dt


# ── Console report ──────────────────────────────────────────────────

def print_trace_report(
    tracer,
    result: Optional[Dict] = None,
    verbose: bool = True,
) -> None:
    """Print a rich trace report to the console."""
    if not tracer or not tracer.root_run:
        print("  ⚠️  No trace data available.")
        return

    root = tracer.root_run
    trace_id = root.get("trace_id", "N/A")
    duration = root.get("duration_ms")
    dur_str = f"{duration:.0f}ms" if duration else "in-progress"

    print()
    print("══════════════════════════════════════════════════════════")
    print("📊 TRACE REPORT")
    print("══════════════════════════════════════════════════════════")
    print(f"  Trace ID: {trace_id}")
    print(f"  Duration: {dur_str}")

    runs_by_name = {}
    trace_id = root.get("trace_id")

    if trace_id:
        try:
            all_runs = tracer.backend.get_runs_by_trace(trace_id)
            for run in all_runs:
                if run["run_id"] != trace_id:
                    runs_by_name[run["name"]] = run
        except Exception:
            pass

    if not runs_by_name and hasattr(tracer, "_run_stack"):
        for run in tracer._run_stack:
            if run["run_id"] != trace_id:
                runs_by_name[run["name"]] = run

    for name, run in runs_by_name.items():
        out = run.get("outputs", {})
        dur = run.get("duration_ms")
        dur_s = f"({dur:.0f}ms)" if dur else ""
        err = run.get("error")
        status = "ERR" if err else "OK"
        print(f"\n  [{status}] {name} {dur_s}")
        if err:
            print(f"     ERROR: {err}")
        for k, v in out.items():
            if k in ("duration_ms", "tags"):
                continue
            print(f"     {k}: {str(v)[:80]}")

    print("══════════════════════════════════════════════════════════")
    project = root.get("project_name", "longtracer")
    print(f"  📁 Project: {project}")
    print()


# ── JSON export ─────────────────────────────────────────────────────

def export_trace_json(tracer, filepath: Optional[str] = None) -> dict:
    """Export the full trace as a JSON-serializable dictionary."""
    if not tracer or not tracer.root_run:
        return {}

    root = tracer.root_run
    trace_id = root.get("trace_id")

    child_runs = []
    if trace_id:
        try:
            all_runs = tracer.backend.get_runs_by_trace(trace_id)
        except Exception:
            all_runs = []

        for run in all_runs:
            if run["run_id"] == trace_id:
                continue
            clean = {}
            for k, v in run.items():
                if k == "_id":
                    continue
                if hasattr(v, "isoformat"):
                    clean[k] = v.isoformat()
                else:
                    clean[k] = v
            child_runs.append(clean)

    clean_root = {}
    for k, v in root.items():
        if k == "_id":
            continue
        if hasattr(v, "isoformat"):
            clean_root[k] = v.isoformat()
        else:
            clean_root[k] = v

    trace_data = {
        "root": clean_root,
        "claim_evidence_map": tracer.claim_evidence_map,
        "runs": child_runs,
    }

    if filepath:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)
        print(f"  📄 Trace exported: {filepath}")

    return trace_data


# ── HTML export ─────────────────────────────────────────────────────

def _html_escape(s: str) -> str:
    """Minimal HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _serialize(obj):
    """JSON serializer for datetime and other non-standard types."""
    if isinstance(obj, _dt):
        return obj.isoformat()
    return str(obj)


def export_trace_html(
    tracer,
    filepath: Optional[str] = None,
) -> str:
    """
    Generate a LangSmith-inspired self-contained HTML trace report.
    """
    if not tracer or not tracer.root_run:
        return "<html><body><p>No trace data available.</p></body></html>"

    root = tracer.root_run
    trace_id = root.get("trace_id", "N/A")

    # Collect all child runs with full data
    all_child_runs = []
    try:
        runs = tracer.backend.get_runs_by_trace(trace_id)
        for run in runs:
            if run["run_id"] != trace_id:
                all_child_runs.append({
                    "run_id": run.get("run_id", ""),
                    "name": run.get("name", "unknown"),
                    "run_type": run.get("run_type", "chain"),
                    "duration_ms": run.get("duration_ms", 0) or 0,
                    "start_time": run.get("start_time", 0) or 0,
                    "end_time": run.get("end_time", 0) or 0,
                    "parent_id": run.get("parent_id"),
                    "inputs": run.get("inputs", {}),
                    "outputs": run.get("outputs", {}),
                    "error": run.get("error"),
                })
    except Exception:
        pass

    # Build data payload
    trace_data = {
        "trace_id": trace_id,
        "project_name": root.get("project_name", "longtracer"),
        "run_name": root.get("run_name", ""),
        "duration_ms": root.get("duration_ms") or 0,
        "created_at": str(root.get("created_at", "")),
        "inputs": root.get("inputs", {}),
        "outputs": root.get("outputs", {}),
    }

    trace_json = json.dumps(trace_data, default=_serialize)
    runs_json = json.dumps(all_child_runs, default=_serialize)

    html = _LANGSMITH_HTML_TEMPLATE.replace("__TRACE_DATA__", trace_json).replace("__RUNS_DATA__", runs_json)

    if filepath:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(html)
        print(f"  HTML report exported: {filepath}")

    return html


_LANGSMITH_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LongTracer — Trace Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0f1117; --surface: #1a1d27; --surface-2: #242838; --border: #2d3248;
  --text: #e4e6ef; --text-muted: #8b8fa3; --accent: #6366f1;
  --green: #22c55e; --green-bg: rgba(34,197,94,0.12);
  --red: #ef4444; --red-bg: rgba(239,68,68,0.12);
  --blue: #3b82f6; --blue-bg: rgba(59,130,246,0.15);
  --purple: #a855f7; --purple-bg: rgba(168,85,247,0.15);
  --orange: #f59e0b; --orange-bg: rgba(245,158,11,0.15);
  --yellow: #eab308;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }

/* Header */
.header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 2rem; }
.header-row { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
.header h1 { font-size: 1.2rem; font-weight: 700; }
.header .logo { color: var(--accent); }
.meta-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }
.pill { background: var(--surface-2); border: 1px solid var(--border); padding: 0.2rem 0.6rem;
        border-radius: 4px; font-size: 0.75rem; color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace; }
.pill b { color: var(--text); font-weight: 500; }

/* Badges */
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
         font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
.badge-pass { background: var(--green-bg); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
.badge-fail { background: var(--red-bg); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }
.badge-na   { background: var(--surface-2); color: var(--text-muted); border: 1px solid var(--border); }
.type-badge { font-size: 0.6rem; padding: 0.1rem 0.35rem; border-radius: 3px; font-weight: 600; }
.type-retriever { background: var(--blue-bg); color: var(--blue); }
.type-llm       { background: var(--purple-bg); color: var(--purple); }
.type-chain     { background: var(--green-bg); color: var(--green); }
.type-tool      { background: var(--orange-bg); color: var(--orange); }

/* Layout */
.main { max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem; }

/* Summary cards */
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 0.75rem; margin-bottom: 1.25rem; }
.s-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
.s-card .label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
.s-card .val { font-size: 1.6rem; font-weight: 700; }
.s-card .sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.15rem; }

/* Section card */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 1rem; overflow: hidden; }
.card-title { padding: 0.65rem 1rem; border-bottom: 1px solid var(--border); font-weight: 600;
              font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.card-body { padding: 1rem; }

/* Waterfall */
.wf-row { display: grid; grid-template-columns: 130px 70px 1fr 70px; align-items: center;
          padding: 0.45rem 0; border-bottom: 1px solid var(--border); font-size: 0.8rem; }
.wf-row:last-child { border-bottom: none; }
.wf-name { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wf-bar-wrap { height: 18px; position: relative; }
.wf-bar { position: absolute; height: 100%; border-radius: 3px; min-width: 4px;
          transition: opacity 0.15s; opacity: 0.8; }
.wf-bar:hover { opacity: 1; }
.wf-bar.retriever { background: var(--blue); } .wf-bar.llm { background: var(--purple); }
.wf-bar.chain { background: var(--green); } .wf-bar.tool { background: var(--orange); }
.wf-dur { text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); }

/* Span cards */
.span-card { border: 1px solid var(--border); border-radius: 6px; margin-bottom: 0.5rem; overflow: hidden; }
.span-hdr { display: flex; align-items: center; justify-content: space-between;
            padding: 0.55rem 1rem; cursor: pointer; user-select: none;
            background: var(--surface-2); transition: background 0.15s; }
.span-hdr:hover { background: #2a2e42; }
.span-hdr-left { display: flex; align-items: center; gap: 0.65rem; }
.span-hdr .arrow { transition: transform 0.2s; color: var(--text-muted); font-size: 0.65rem; }
.span-hdr.open .arrow { transform: rotate(90deg); }
.span-name { font-weight: 600; font-size: 0.82rem; }
.span-dur  { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); }
.span-ok   { color: var(--green); font-size: 0.72rem; }
.span-err  { color: var(--red);   font-size: 0.72rem; }
.span-body { display: none; border-top: 1px solid var(--border); background: var(--bg); }
.span-body.open { display: block; }
.span-section { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); }
.span-section:last-child { border-bottom: none; }
.span-section h4 { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;
                   letter-spacing: 0.05em; margin-bottom: 0.4rem; }
pre.json { background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
           padding: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
           overflow-x: auto; white-space: pre-wrap; word-break: break-word; color: var(--text); max-height: 300px; overflow-y: auto; }

/* Claims table */
table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
th, td { text-align: left; padding: 0.5rem 0.65rem; border-bottom: 1px solid var(--border); }
th { background: var(--surface-2); color: var(--text-muted); font-weight: 600; font-size: 0.72rem;
     text-transform: uppercase; letter-spacing: 0.04em; }
.status-supported   { color: var(--green); font-weight: 600; }
.status-unsupported { color: var(--red); font-weight: 600; }
.status-hallucination { color: var(--red); font-weight: 700; }

/* I/O section */
.io-section { margin-bottom: 1rem; }
.io-section h4 { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 0.4rem; }

.footer { text-align: center; color: var(--text-muted); font-size: 0.7rem; margin-top: 2rem; padding: 1rem; opacity: 0.5; }
</style>
</head>
<body>

<div class="header">
  <div class="header-row">
    <h1><span class="logo">◆</span> LongTracer Trace Report</h1>
  </div>
  <div class="meta-pills" id="metaPills"></div>
</div>

<div class="main">
  <div class="summary" id="summary"></div>

  <!-- Trace I/O -->
  <div class="card" id="traceIOCard" style="display:none">
    <div class="card-title">Trace Input / Output</div>
    <div class="card-body" id="traceIO"></div>
  </div>

  <!-- Waterfall -->
  <div class="card">
    <div class="card-title">Pipeline Timeline</div>
    <div class="card-body" id="waterfall"></div>
  </div>

  <!-- Spans -->
  <div class="card">
    <div class="card-title">Span Details</div>
    <div class="card-body" id="spans"></div>
  </div>

  <!-- Claims -->
  <div class="card" id="claimsCard" style="display:none">
    <div class="card-title">Claims Verification</div>
    <div class="card-body" id="claims"></div>
  </div>
</div>

<div class="footer">Generated by LongTracer &middot; <span id="genTime"></span></div>

<script>
const T = __TRACE_DATA__;
const R = __RUNS_DATA__;

// Helpers
function fmtDur(ms) {
  if (!ms && ms !== 0) return 'N/A';
  return ms < 1000 ? ms.toFixed(0) + 'ms' : (ms / 1000).toFixed(2) + 's';
}
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function prettyJSON(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}
function typeBadge(t) {
  const cls = 'type-' + (t || 'chain');
  return '<span class="type-badge ' + cls + '">' + esc(t || 'chain') + '</span>';
}

// Header pills
(function() {
  const p = document.getElementById('metaPills');
  const pills = [
    ['Trace', T.trace_id],
    ['Project', T.project_name],
    ['Duration', fmtDur(T.duration_ms)],
    ['Created', (T.created_at || '').substring(0, 19)],
  ];
  p.innerHTML = pills.map(([l,v]) => '<span class="pill"><b>' + esc(l) + ':</b> ' + esc(String(v)) + '</span>').join('');
})();

// Summary cards
(function() {
  const el = document.getElementById('summary');
  // Find grounding span
  const grounding = R.find(r => r.name === 'grounding');
  const evalClaims = R.find(r => r.name === 'eval_claims');
  const gOut = grounding ? (grounding.outputs || {}) : {};
  const claims = evalClaims ? (evalClaims.outputs || {}).claims || [] : [];
  const trustScore = gOut.grounding_score || 0;
  const verdict = gOut.verdict || 'N/A';
  const hCount = gOut.hallucination_count || 0;
  const supported = claims.filter(c => c.status === 'supported').length;
  const verdictClass = verdict === 'PASS' ? 'badge-pass' : verdict === 'FAIL' ? 'badge-fail' : 'badge-na';

  el.innerHTML = [
    '<div class="s-card"><div class="label">Verdict</div><div class="val"><span class="badge ' + verdictClass + '">' + esc(verdict) + '</span></div></div>',
    '<div class="s-card"><div class="label">Trust Score</div><div class="val">' + (trustScore * 100).toFixed(0) + '%</div></div>',
    '<div class="s-card"><div class="label">Total Duration</div><div class="val">' + fmtDur(T.duration_ms) + '</div></div>',
    '<div class="s-card"><div class="label">Claims</div><div class="val">' + claims.length + '</div><div class="sub">' + supported + ' supported, ' + hCount + ' hallucinations</div></div>',
    '<div class="s-card"><div class="label">Spans</div><div class="val">' + R.length + '</div></div>',
  ].join('');
})();

// Trace I/O
(function() {
  const el = document.getElementById('traceIO');
  const card = document.getElementById('traceIOCard');
  const hasInputs = T.inputs && Object.keys(T.inputs).length > 0;
  const hasOutputs = T.outputs && Object.keys(T.outputs).length > 0;
  if (!hasInputs && !hasOutputs) return;
  card.style.display = '';
  let html = '';
  if (hasInputs) {
    html += '<div class="io-section"><h4>Input</h4><pre class="json">' + esc(prettyJSON(T.inputs)) + '</pre></div>';
  }
  if (hasOutputs) {
    html += '<div class="io-section"><h4>Output</h4><pre class="json">' + esc(prettyJSON(T.outputs)) + '</pre></div>';
  }
  el.innerHTML = html;
})();

// Waterfall timeline
(function() {
  const el = document.getElementById('waterfall');
  if (!R.length) { el.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;">No spans recorded</div>'; return; }

  const totalDur = T.duration_ms || 1;
  // Compute offsets: use start_time relative to first span
  const traceStart = R.reduce((min, r) => Math.min(min, r.start_time || Infinity), Infinity);

  let html = '';
  R.forEach(r => {
    const offset = ((r.start_time || traceStart) - traceStart) * 1000;
    const dur = r.duration_ms || 0;
    const leftPct = Math.min((offset / totalDur) * 100, 95);
    const widthPct = Math.max(Math.min((dur / totalDur) * 100, 100 - leftPct), 0.5);
    const barClass = r.run_type || 'chain';
    html += '<div class="wf-row">' +
      '<div class="wf-name">' + esc(r.name) + '</div>' +
      '<div>' + typeBadge(r.run_type) + '</div>' +
      '<div class="wf-bar-wrap"><div class="wf-bar ' + barClass + '" style="left:' + leftPct.toFixed(1) + '%;width:' + widthPct.toFixed(1) + '%" title="' + esc(r.name) + ': ' + fmtDur(dur) + '"></div></div>' +
      '<div class="wf-dur">' + fmtDur(dur) + '</div>' +
      '</div>';
  });
  el.innerHTML = html;
})();

// Span detail cards
(function() {
  const el = document.getElementById('spans');
  if (!R.length) { el.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;">No spans</div>'; return; }

  let html = '';
  R.forEach((r, i) => {
    const err = r.error;
    const statusHtml = err
      ? '<span class="span-err">✕ ERROR</span>'
      : '<span class="span-ok">✓ OK</span>';
    const outputs = r.outputs || {};
    // Remove internal keys
    const cleanOutputs = {};
    Object.keys(outputs).forEach(k => { if (k !== 'duration_ms' && k !== 'tags') cleanOutputs[k] = outputs[k]; });
    const inputs = r.inputs || {};

    html += '<div class="span-card">' +
      '<div class="span-hdr" onclick="toggleSpan(this)">' +
        '<div class="span-hdr-left">' +
          '<span class="arrow">▶</span>' +
          '<span class="span-name">' + esc(r.name) + '</span>' +
          typeBadge(r.run_type) +
        '</div>' +
        '<div style="display:flex;align-items:center;gap:0.75rem;">' +
          '<span class="span-dur">' + fmtDur(r.duration_ms) + '</span>' +
          statusHtml +
        '</div>' +
      '</div>' +
      '<div class="span-body" id="spanBody' + i + '">';

    if (err) {
      html += '<div class="span-section"><h4>Error</h4><pre class="json" style="color:var(--red)">' + esc(String(err)) + '</pre></div>';
    }
    if (Object.keys(inputs).length) {
      html += '<div class="span-section"><h4>Input</h4><pre class="json">' + esc(prettyJSON(inputs)) + '</pre></div>';
    }
    if (Object.keys(cleanOutputs).length) {
      html += '<div class="span-section"><h4>Output</h4><pre class="json">' + esc(prettyJSON(cleanOutputs)) + '</pre></div>';
    }

    html += '</div></div>';
  });
  el.innerHTML = html;
})();

// Claims table
(function() {
  const evalClaims = R.find(r => r.name === 'eval_claims');
  if (!evalClaims) return;
  const claims = (evalClaims.outputs || {}).claims || [];
  if (!claims.length) return;

  const card = document.getElementById('claimsCard');
  const el = document.getElementById('claims');
  card.style.display = '';

  let html = '<table><thead><tr><th>#</th><th>Claim</th><th>Status</th><th>Score</th></tr></thead><tbody>';
  claims.forEach((c, i) => {
    const status = c.is_hallucination ? 'HALLUCINATION' : (c.status || 'unknown');
    const cls = c.is_hallucination ? 'status-hallucination' : (status === 'supported' ? 'status-supported' : 'status-unsupported');
    const icon = status === 'supported' ? '✓' : '✕';
    html += '<tr><td>' + (i+1) + '</td><td>' + esc(c.text || '') + '</td>' +
            '<td class="' + cls + '">' + icon + ' ' + esc(status) + '</td>' +
            '<td>' + (c.score != null ? Number(c.score).toFixed(2) : 'N/A') + '</td></tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
})();

// Toggle
function toggleSpan(hdr) {
  hdr.classList.toggle('open');
  const body = hdr.nextElementSibling;
  body.classList.toggle('open');
}

// Gen time
document.getElementById('genTime').textContent = new Date().toISOString().replace('T',' ').substring(0,19) + ' UTC';
</script>
</body>
</html>"""
