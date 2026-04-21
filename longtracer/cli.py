"""
LongTracer CLI — View and inspect verification traces.

Installed as ``longtracer`` console command via pyproject.toml entry point.

Usage:
    longtracer view                     # list recent traces
    longtracer view --id <trace_id>     # view specific trace
    longtracer view --last              # view most recent trace
    longtracer view --export <trace_id> # export trace to JSON
    longtracer view --html <trace_id>   # export trace to HTML
    longtracer view --project <name>    # filter by project
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def _load_dotenv():
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if not os.environ.get(key.strip()):
                        os.environ[key.strip()] = value.strip()


def _get_tracer():
    from longtracer.guard.tracer import Tracer
    return Tracer(run_name="longtracer_cli")


def _fmt_dt(dt) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(dt, str):
        return dt[:19]
    return str(dt) if dt else "N/A"


def _fmt_dur(ms) -> str:
    if ms is None:
        return "N/A"
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.2f}s"


def cmd_list(args):
    tracer = _get_tracer()
    if not tracer.is_connected():
        print("Backend not connected. Check your configuration.")
        return
    traces = tracer.list_recent_traces(limit=args.limit, project_name=args.project)
    if not traces:
        print("No traces found.")
        return
    print()
    print("=" * 85)
    print("  RECENT TRACES")
    print("=" * 85)
    print(f"{'#':<4} {'Trace ID':<38} {'Duration':<10} {'Created':<20} {'Project':<15} {'Query'}")
    print("-" * 85)
    for i, t in enumerate(traces, 1):
        tid = t.get("trace_id", "N/A")
        dur = _fmt_dur(t.get("duration_ms"))
        cre = _fmt_dt(t.get("created_at"))
        proj = t.get("project_name", "-")[:14]
        q = t.get("inputs", {}).get("query", "N/A")
        if len(q) > 30:
            q = q[:27] + "..."
        print(f"{i:<4} {tid:<38} {dur:<10} {cre:<20} {proj:<15} {q}")
    print("-" * 85)
    print(f"Total: {len(traces)} trace(s)")
    print()


def cmd_view(args):
    tracer = _get_tracer()
    trace = tracer.get_trace(args.id)
    if not trace:
        print(f"Trace not found: {args.id}")
        return
    print()
    print("=" * 80)
    print("  TRACE DETAILS")
    print("=" * 80)
    print(f"  Trace ID:  {trace.get('trace_id', 'N/A')}")
    print(f"  Project:   {trace.get('project_name', 'N/A')}")
    print(f"  Run Name:  {trace.get('run_name', 'N/A')}")
    print(f"  Created:   {_fmt_dt(trace.get('created_at'))}")
    print(f"  Duration:  {_fmt_dur(trace.get('duration_ms'))}")
    inputs = trace.get("inputs", {})
    if inputs:
        print("\n--- INPUTS " + "-" * 68)
        for k, v in inputs.items():
            print(f"  {k}: {str(v)[:100]}")
    outputs = trace.get("outputs", {})
    if outputs:
        print("\n--- OUTPUTS " + "-" * 67)
        for k, v in outputs.items():
            if k == "claim_evidence_map":
                print(f"  {k}: ({len(v)} claims)")
                continue
            print(f"  {k}: {str(v)[:100]}")
    evidence_map = trace.get("claim_evidence_map", {})
    if evidence_map:
        print("\n--- CLAIM -> EVIDENCE MAP " + "-" * 54)
        for claim_id, evidences in evidence_map.items():
            print(f'\n  "{claim_id[:60]}"')
            if isinstance(evidences, dict):
                for src, score in evidences.items():
                    bar = "#" * int(float(score) * 10) + "." * (10 - int(float(score) * 10))
                    print(f'     [{bar}] {float(score):.2f} <- "{src[:50]}"')
    runs = tracer.get_runs_by_trace(args.id)
    child_runs = [r for r in runs if r.get("run_id") != args.id]
    if child_runs:
        print("\n--- PIPELINE SPANS " + "-" * 60)
        for run in child_runs:
            name = run.get("name", "?")
            dur = _fmt_dur(run.get("duration_ms"))
            err = run.get("error")
            status = "OK" if not err else "ERR"
            print(f"\n  [{status}] {name} ({dur})")
            if err:
                print(f"     ERROR: {err}")
            for k, v in run.get("outputs", {}).items():
                if k in ("duration_ms", "tags"):
                    continue
                print(f"     {k}: {str(v)[:80]}")
    print()
    print("=" * 80)


def cmd_last(args):
    tracer = _get_tracer()
    traces = tracer.list_recent_traces(limit=1, project_name=args.project)
    if not traces:
        print("No traces found.")
        return
    args.id = traces[0].get("trace_id")
    if args.id:
        cmd_view(args)


def cmd_export_json(args):
    from longtracer.guard.trace_report import export_trace_json
    tracer = _get_tracer()
    trace = tracer.get_trace(args.export)
    if not trace:
        print(f"Trace not found: {args.export}")
        return
    tracer.root_run = trace
    out = args.output or f"trace_{args.export[:8]}.json"
    export_trace_json(tracer, filepath=out)
    print(f"Exported to: {out}")


def cmd_export_html(args):
    from longtracer.guard.trace_report import export_trace_html
    tracer = _get_tracer()
    trace = tracer.get_trace(args.html)
    if not trace:
        print(f"Trace not found: {args.html}")
        return
    tracer.root_run = trace
    out = args.output or f"trace_{args.html[:8]}.html"
    export_trace_html(tracer, filepath=out)
    print(f"HTML report exported to: {out}")


def cmd_check(args):
    """Run a one-shot hallucination check from the CLI."""
    import json as _json
    import sys as _sys
    from longtracer.guard.verifier import CitationVerifier

    if args.json_output:
        # Suppress model-loading progress bars that pollute JSON stdout
        _orig_stdout = _sys.stdout
        _sys.stdout = open(os.devnull, "w")
        try:
            verifier = CitationVerifier(threshold=args.threshold)
            result = verifier.verify_parallel(args.response, args.sources)
        finally:
            _sys.stdout.close()
            _sys.stdout = _orig_stdout
    else:
        verifier = CitationVerifier(threshold=args.threshold)
        result = verifier.verify_parallel(args.response, args.sources)

    if args.json_output:
        out = {
            "verdict": result.verdict,
            "trust_score": round(result.trust_score, 4),
            "summary": result.summary,
            "hallucination_count": result.hallucination_count,
            "claims": [
                {
                    "claim": c.get("claim", ""),
                    "supported": c.get("supported", False),
                    "score": round(c.get("score", 0), 4),
                    "is_hallucination": c.get("is_hallucination", False),
                }
                for c in result.claims
            ],
        }
        print(_json.dumps(out, indent=2))
        return

    icon = "✓" if result.verdict == "PASS" else "✗"
    print(
        f"\n{icon} {result.verdict}  "
        f"trust={result.trust_score:.2f}  "
        f"hallucinations={result.hallucination_count}"
    )
    print(f"  {result.summary}\n")
    for c in result.claims:
        status = "✓" if c.get("supported") else "✗"
        hall = " [HALLUCINATION]" if c.get("is_hallucination") else ""
        print(f"  {status} {c.get('claim', '')[:100]}{hall}")
        if c.get("best_source"):
            print(f"    ↳ source: {c['best_source'][:80]}")
    print()


def cmd_serve(args):
    """Start the REST API server."""
    try:
        from longtracer.server import run_server
    except ImportError:
        print(
            "Server dependencies not installed.\n"
            "Install with: pip install 'longtracer[server]'"
        )
        return
    print(f"Starting LongTracer API server on {args.host}:{args.port}")
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


def main():
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="longtracer",
        description="LongTracer — View and inspect verification traces",
    )
    sub = parser.add_subparsers(dest="command")
    vp = sub.add_parser("view", help="View traces")
    vp.add_argument("--id", help="View a specific trace by ID")
    vp.add_argument("--last", action="store_true", help="View most recent trace")
    vp.add_argument("--export", metavar="TRACE_ID", help="Export trace to JSON")
    vp.add_argument("--html", metavar="TRACE_ID", help="Export trace to HTML report")
    vp.add_argument("--output", "-o", help="Output file path")
    vp.add_argument("--project", "-p", help="Filter by project name")
    vp.add_argument("--limit", type=int, default=10, help="Max traces to list")

    # check subcommand
    cp = sub.add_parser("check", help="Verify a response against sources")
    cp.add_argument("response", help="LLM response text to verify")
    cp.add_argument("sources", nargs="+", help="Source text(s) to verify against")
    cp.add_argument("--json", dest="json_output", action="store_true",
                     help="Output results as JSON")
    cp.add_argument("--threshold", type=float, default=0.5,
                     help="Verification threshold (default: 0.5)")

    # serve subcommand
    sp = sub.add_parser("serve", help="Start the REST API server")
    sp.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    sp.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    sp.add_argument("--workers", type=int, default=1, help="Worker processes (default: 1)")
    sp.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")

    args = parser.parse_args()
    if args.command is None:
        args.command = "view"
        args.id = args.last = args.export = args.html = args.output = args.project = None
        args.limit = 10
        args.last = False

    if args.command == "check":
        cmd_check(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "view":
        if args.id:
            cmd_view(args)
        elif args.last:
            cmd_last(args)
        elif args.export:
            cmd_export_json(args)
        elif args.html:
            cmd_export_html(args)
        else:
            cmd_list(args)


if __name__ == "__main__":
    main()
