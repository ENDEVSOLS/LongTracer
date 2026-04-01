"""
Trace Report — Rich console visualization and JSON export for trace data.
"""

import json
import os
from typing import Dict, List, Optional


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

    _print_retrieval(runs_by_name.get("retrieval"), verbose)
    _print_prompt_build(runs_by_name.get("prompt_build"), verbose)
    _print_llm_call(runs_by_name.get("llm_call"), verbose)
    _print_eval_relevance(runs_by_name.get("eval_relevance"), verbose)
    _print_eval_claims(runs_by_name.get("eval_claims"), verbose)
    _print_grounding(runs_by_name.get("grounding"), verbose)

    print("══════════════════════════════════════════════════════════")

    project = root.get("project_name", "longtracer")
    mongo_db = os.environ.get("MONGODB_DATABASE", "longtracer")
    print(f"  📁 Project: {project}")
    mongo_uri = os.environ.get("MONGODB_URI")
    if mongo_uri:
        print(f"  🔗 MongoDB: db.runs.find({{trace_id: \"{trace_id}\"}})")

    print()


def _span_header(name: str, run: Optional[Dict]) -> str:
    if run and run.get("duration_ms"):
        return f"─── {name} ({run['duration_ms']:.0f}ms) {'─' * max(1, 40 - len(name))}"
    return f"─── {name} {'─' * max(1, 48 - len(name))}"


def _print_retrieval(run: Optional[Dict], verbose: bool) -> None:
    if not run:
        return
    out = run.get("outputs", {})
    print(_span_header("retrieval", run))
    count = out.get("count", 0)
    ms = out.get("retrieval_ms", 0)
    print(f"  Chunks: {count} retrieved ({ms:.0f}ms)")
    chunks = out.get("chunks", [])
    for c in chunks[:5]:
        src = c.get("source", "?")
        page = c.get("page", "?")
        preview = c.get("text", "")[:60].replace("\n", " ")
        print(f"  [{c.get('chunk_index', '?')}] p{page} / {src}")
        if verbose and preview:
            print(f"      \"{preview}...\"")
    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more chunks")
    print()


def _print_prompt_build(run: Optional[Dict], verbose: bool) -> None:
    if not run:
        return
    out = run.get("outputs", {})
    print(_span_header("prompt_build", run))
    ctx_len = out.get("context_length_chars", 0)
    print(f"  Context: {ctx_len:,} chars → system prompt")
    if verbose:
        prompt = out.get("system_prompt", "")
        if prompt:
            lines = prompt.split("\n")
            preview_lines = lines[:3]
            print(f"  Preview: {preview_lines[0][:80]}...")
    print()


def _print_llm_call(run: Optional[Dict], verbose: bool) -> None:
    if not run:
        return
    out = run.get("outputs", {})
    print(_span_header("llm_call", run))
    model = out.get("model", "unknown")
    ms = out.get("llm_ms", 0)
    print(f"  Model: {model} ({ms:.0f}ms)")
    answer = out.get("answer", "")
    if answer:
        preview = answer[:120].replace("\n", " ")
        print(f"  Answer: \"{preview}{'...' if len(answer) > 120 else ''}\"")
    print()


def _print_eval_relevance(run: Optional[Dict], verbose: bool) -> None:
    if not run:
        return
    out = run.get("outputs", {})
    print(_span_header("eval_relevance", run))
    avg = out.get("average_relevance", 0)
    top = out.get("top_relevance", 0)
    passed = out.get("threshold_pass", False)
    icon = "✅" if passed else "❌"
    print(f"  Average: {avg:.2f} | Top: {top:.2f} | Pass: {icon}")
    if verbose:
        rankings = out.get("chunk_rankings", [])
        for r in rankings[:3]:
            cid = r.get("chunk_id", "?")
            score = r.get("score", 0)
            print(f"    [{cid}] → {score:.3f}")
    print()


def _print_eval_claims(run: Optional[Dict], verbose: bool) -> None:
    if not run:
        return
    out = run.get("outputs", {})
    print(_span_header("eval_claims", run))
    claims = out.get("claims", [])
    ms = out.get("verify_ms", 0)
    print(f"  Total: {len(claims)} claims ({ms:.0f}ms)")
    for c in claims:
        icon = "✅" if c.get("status") == "supported" else "❌"
        conf = c.get("confidence", 0)
        text = c.get("text", "")[:60]
        cid = c.get("claim_id", "?")
        print(f"  {icon} {cid}: \"{text}\" ({conf:.2f})")
        if c.get("is_hallucination"):
            print("      ⚠️  HALLUCINATION detected")
        if c.get("is_meta_statement"):
            print("      ℹ️  Meta-statement (honest uncertainty)")
    print()


def _print_grounding(run: Optional[Dict], verbose: bool) -> None:
    if not run:
        return
    out = run.get("outputs", {})
    print(_span_header("grounding", run))
    score = out.get("grounding_score", 0)
    h_count = out.get("hallucination_count", 0)
    flags = out.get("flags_triggered", [])
    verdict = out.get("verdict", "?")
    verdict_icon = "✅" if verdict == "PASS" else "⚠️"

    print(f"  Score: {score:.2f} | Hallucinations: {h_count}")
    if flags:
        print(f"  Flags: {flags}")
    print(f"  Verdict: {verdict_icon} {verdict}")
    h_ids = out.get("hallucinated_claim_ids", [])
    if h_ids:
        print(f"  Hallucinated: {h_ids}")
    print()


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

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LongTracer Report — {trace_id}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f8f9fa; color: #212529; padding: 2rem; line-height: 1.6; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
  .meta {{ color: #6c757d; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 4px;
            font-weight: 600; font-size: 0.9rem; }}
  .badge-pass {{ background: #d4edda; color: #155724; }}
  .badge-fail {{ background: #f8d7da; color: #721c24; }}
  .card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 8px;
           padding: 1.25rem; margin-bottom: 1rem; }}
  .card h2 {{ font-size: 1.1rem; margin-bottom: 0.75rem; border-bottom: 1px solid #eee;
              padding-bottom: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee; }}
  th {{ background: #f1f3f5; font-weight: 600; }}
  .supported {{ color: #155724; }}
  .unsupported {{ color: #721c24; }}
  .hallucination {{ color: #721c24; font-weight: 600; }}
  .timing {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .timing-item {{ background: #e9ecef; padding: 0.5rem 1rem; border-radius: 4px; font-size: 0.85rem; }}
  .score-bar {{ display: inline-block; height: 10px; border-radius: 2px; }}
  .score-fill {{ background: #28a745; height: 100%; border-radius: 2px; display: block; }}
  .footer {{ text-align: center; color: #adb5bd; font-size: 0.8rem; margin-top: 2rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>LongTracer Verification Report</h1>
  <div class="meta">
    Trace ID: <code>{trace_id}</code> &middot;
    Project: {project_name} &middot;
    Duration: {duration} &middot;
    Generated: {generated_at}
  </div>

  <div class="card">
    <h2>Verdict</h2>
    <p>Trust Score: <strong>{trust_score_pct}</strong>
       &nbsp; <span class="badge {verdict_class}">{verdict}</span></p>
    <p>Claims: {total_claims} total, {supported_count} supported, {hallucination_count} hallucinations</p>
  </div>

  <div class="card">
    <h2>Claims</h2>
    <table>
      <thead><tr><th>#</th><th>Claim</th><th>Status</th><th>Score</th><th>Source</th></tr></thead>
      <tbody>{claims_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Pipeline Timing</h2>
    <div class="timing">{timing_items}</div>
  </div>

  <div class="card">
    <h2>Spans</h2>
    <table>
      <thead><tr><th>Span</th><th>Type</th><th>Duration</th><th>Status</th></tr></thead>
      <tbody>{spans_rows}</tbody>
    </table>
  </div>

  <div class="footer">Generated by LongTracer v0.1.0</div>
</div>
</body>
</html>"""


def export_trace_html(
    tracer,
    filepath: Optional[str] = None,
) -> str:
    """
    Generate a self-contained HTML trace report.

    Args:
        tracer: Tracer instance with completed root run.
        filepath: Optional path to write the HTML file.

    Returns:
        The HTML string.
    """
    if not tracer or not tracer.root_run:
        return "<html><body><p>No trace data available.</p></body></html>"

    root = tracer.root_run
    trace_id = root.get("trace_id", "N/A")
    project_name = root.get("project_name", "longtracer")
    duration_ms = root.get("duration_ms")
    duration = f"{duration_ms:.0f}ms" if duration_ms else "N/A"

    # Collect child runs
    runs_by_name: Dict = {}
    try:
        all_runs = tracer.backend.get_runs_by_trace(trace_id)
        for run in all_runs:
            if run["run_id"] != trace_id:
                runs_by_name[run["name"]] = run
    except Exception:
        pass

    # Extract claims from eval_claims span
    claims_data = []
    eval_claims = runs_by_name.get("eval_claims", {})
    if eval_claims:
        claims_data = eval_claims.get("outputs", {}).get("claims", [])

    # Extract grounding
    grounding = runs_by_name.get("grounding", {})
    grounding_out = grounding.get("outputs", {}) if grounding else {}
    trust_score = grounding_out.get("grounding_score", 0)
    verdict = grounding_out.get("verdict", "N/A")
    hallucination_count = grounding_out.get("hallucination_count", 0)

    supported_count = sum(1 for c in claims_data if c.get("status") == "supported")

    # Build claims rows
    claims_rows = ""
    for i, c in enumerate(claims_data, 1):
        status = c.get("status", "unknown")
        css = "supported" if status == "supported" else "unsupported"
        if c.get("is_hallucination"):
            css = "hallucination"
            status = "HALLUCINATION"
        text = _html_escape(c.get("text", "")[:200])
        score = c.get("confidence", c.get("score", 0))
        source = _html_escape(str(c.get("supporting_chunks", ""))[:80])
        claims_rows += f'<tr><td>{i}</td><td>{text}</td><td class="{css}">{status}</td><td>{score:.2f}</td><td>{source}</td></tr>\n'

    if not claims_rows:
        claims_rows = '<tr><td colspan="5">No claims data available</td></tr>'

    # Build timing items
    timing_items = ""
    for span_name in ["retrieval", "llm_call", "eval_relevance", "eval_claims", "grounding"]:
        span = runs_by_name.get(span_name)
        if span:
            ms = span.get("duration_ms", 0)
            timing_items += f'<div class="timing-item">{span_name}: {ms:.0f}ms</div>\n'

    # Build spans rows
    spans_rows = ""
    for run in runs_by_name.values():
        name = _html_escape(run.get("name", "?"))
        rtype = run.get("run_type", "chain")
        dur = f"{run.get('duration_ms', 0):.0f}ms" if run.get("duration_ms") else "N/A"
        err = run.get("error")
        status = "OK" if not err else f"ERROR: {_html_escape(str(err)[:60])}"
        spans_rows += f'<tr><td>{name}</td><td>{rtype}</td><td>{dur}</td><td>{status}</td></tr>\n'

    from datetime import datetime as _dt
    html = _HTML_TEMPLATE.format(
        trace_id=trace_id,
        project_name=_html_escape(project_name),
        duration=duration,
        generated_at=_dt.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        trust_score_pct=f"{trust_score:.0%}",
        verdict=verdict,
        verdict_class="badge-pass" if verdict == "PASS" else "badge-fail",
        total_claims=len(claims_data),
        supported_count=supported_count,
        hallucination_count=hallucination_count,
        claims_rows=claims_rows,
        timing_items=timing_items or '<div class="timing-item">No timing data</div>',
        spans_rows=spans_rows or '<tr><td colspan="4">No spans</td></tr>',
    )

    if filepath:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(html)
        print(f"  HTML report exported: {filepath}")

    return html


def _html_escape(s: str) -> str:
    """Minimal HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
