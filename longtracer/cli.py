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
    longtracer check <response> <src>   # one-shot hallucination check
    longtracer serve                    # start REST API + dashboard server
    longtracer doctor                   # inspect installation health
    longtracer models prepare           # pre-download model weights
"""

import argparse
import os
import sys
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
    from longtracer.guard.verifier import CitationVerifier

    if args.json_output:
        # Suppress model-loading progress bars that pollute JSON stdout
        _orig_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            verifier = CitationVerifier(threshold=args.threshold)
            result = verifier.verify_parallel(args.response, args.sources)
        finally:
            sys.stdout.close()
            sys.stdout = _orig_stdout
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

    icon = "\u2713" if result.verdict == "PASS" else "\u2717"
    print(
        f"\n{icon} {result.verdict}  "
        f"trust={result.trust_score:.2f}  "
        f"hallucinations={result.hallucination_count}"
    )
    print(f"  {result.summary}\n")
    for c in result.claims:
        status = "\u2713" if c.get("supported") else "\u2717"
        hall = " [HALLUCINATION]" if c.get("is_hallucination") else ""
        print(f"  {status} {c.get('claim', '')[:100]}{hall}")
        if c.get("best_source"):
            print(f"    \u21b3 source: {c['best_source'][:80]}")
    print()


def cmd_serve(args):
    """Start the REST API and dashboard server."""
    try:
        from longtracer.server import run_server
    except ImportError:
        print(
            "Server dependencies not installed.\n"
            "Install with: pip install 'longtracer[server]'"
        )
        return
    print(f"Starting LongTracer server on http://{args.host}:{args.port}")
    print(f"Dashboard: http://localhost:{args.port}/dashboard")
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


# ---------------------------------------------------------------------------
# doctor  (Roadmap Queue 6 — longtracer doctor)
# ---------------------------------------------------------------------------

def _print_check(label: str, ok: bool, detail: str = "", warn: bool = False) -> bool:
    """Print a single health-check result line. Returns True if ok."""
    if ok and not warn:
        icon = "  \u2713"       # ✓
    elif warn:
        icon = "  \u26a0"       # ⚠
    else:
        icon = "  \u2717"       # ✗
    suffix = f"  \u2014 {detail}" if detail else ""  # em-dash
    print(f"{icon}  {label}{suffix}")
    return ok and not warn


def cmd_doctor(args):  # noqa: C901
    """
    Inspect the LongTracer installation and report health.

    All checks are read-only — this command never mutates user data or traces.

    Checks performed:
      - Python version (>=3.10 required)
      - longtracer package installed and version
      - Core dependencies: pydantic, sentence_transformers, transformers, numpy
      - STS model cached  (sentence-transformers/all-MiniLM-L6-v2)
      - NLI model cached  (cross-encoder/nli-deberta-v3-xsmall)
      - Optional extras:  otel, server, mongo, postgres, redis, slm, langchain,
                          langgraph, llamaindex, haystack
      - ~/.longtracer directory writable (SQLite default storage)
      - pyproject.toml [tool.longtracer] config present
      - LONGTRACER_* environment variables set
      - Default backend connectivity
    """
    import importlib
    import importlib.metadata
    import tempfile

    errors = 0
    warnings = 0

    print()
    print("=" * 62)
    print("  longtracer doctor")
    print("=" * 62)

    # ── Runtime ──────────────────────────────────────────────────
    print("\n[ Runtime ]")
    py_ver = sys.version_info
    py_ok = (py_ver.major, py_ver.minor) >= (3, 10)
    ver_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if not _print_check("Python version", py_ok, ver_str):
        if py_ok:
            warnings += 1
        else:
            errors += 1

    try:
        pkg_ver = importlib.metadata.version("longtracer")
        _print_check("longtracer", True, f"v{pkg_ver}")
    except importlib.metadata.PackageNotFoundError:
        _print_check("longtracer", False, "not found in environment")
        errors += 1

    # ── Core dependencies ────────────────────────────────────────
    print("\n[ Core dependencies ]")
    for dep in ("pydantic", "sentence_transformers", "transformers", "numpy"):
        try:
            m = importlib.import_module(dep)
            dep_ver = getattr(m, "__version__", "?")
            _print_check(dep, True, f"v{dep_ver}")
        except ImportError:
            _print_check(dep, False, "not installed")
            errors += 1

    # ── Model cache ──────────────────────────────────────────────
    print("\n[ Model cache ]")

    def _is_model_cached(model_id: str) -> bool:
        """Check if a HuggingFace model is present in the local cache."""
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        hub = Path(hf_home) / "hub"
        safe_name = "models--" + model_id.replace("/", "--")
        model_dir = hub / safe_name
        return model_dir.exists() and any(
            f.suffix in (".bin", ".safetensors", ".pt")
            for f in model_dir.rglob("*")
        )

    sts_id = "sentence-transformers/all-MiniLM-L6-v2"
    sts_ok = _is_model_cached(sts_id)
    if not _print_check(
        "STS bi-encoder  (all-MiniLM-L6-v2)",
        sts_ok,
        "cached" if sts_ok else "not cached \u2014 run: longtracer models prepare",
        warn=not sts_ok,
    ):
        warnings += 1

    nli_id = "cross-encoder/nli-deberta-v3-xsmall"
    nli_ok = _is_model_cached(nli_id)
    if not _print_check(
        "NLI cross-encoder  (nli-deberta-v3-xsmall)",
        nli_ok,
        "cached" if nli_ok else "not cached \u2014 run: longtracer models prepare",
        warn=not nli_ok,
    ):
        warnings += 1

    # ── Optional extras ──────────────────────────────────────────
    print("\n[ Optional extras ]")
    _EXTRAS = [
        ("otel",                "opentelemetry.sdk"),
        ("server (FastAPI)",    "fastapi"),
        ("mongo",               "pymongo"),
        ("postgres",            "psycopg2"),
        ("redis",               "redis"),
        ("slm (llama-cpp)",     "llama_cpp"),
        ("langchain",           "langchain"),
        ("langgraph",           "langgraph"),
        ("llamaindex",          "llama_index"),
        ("haystack",            "haystack"),
    ]
    for label, mod in _EXTRAS:
        try:
            importlib.import_module(mod)
            _print_check(label, True, "installed")
        except ImportError:
            _print_check(label, True, "not installed (optional)", warn=True)
            # Warnings only — extras are optional by design

    # ── Storage writability ──────────────────────────────────────
    print("\n[ Storage ]")
    default_dir = Path.home() / ".longtracer"
    try:
        default_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=default_dir, delete=True):
            pass
        _print_check("~/.longtracer writable", True, str(default_dir))
    except OSError as exc:
        _print_check("~/.longtracer writable", False, str(exc))
        errors += 1

    # ── Configuration ────────────────────────────────────────────
    print("\n[ Configuration ]")
    from longtracer.config import load_config, _find_pyproject
    pyproject_path = _find_pyproject()
    if pyproject_path:
        cfg = load_config()
        cfg_keys = ", ".join(cfg.keys()) if cfg else "(defaults only)"
        _print_check("pyproject.toml", True, str(pyproject_path))
        _print_check("[tool.longtracer] keys", True, cfg_keys)
    else:
        _print_check("pyproject.toml", True, "not found \u2014 using built-in defaults", warn=True)
        warnings += 1

    lt_env = sorted(k for k in os.environ if k.startswith("LONGTRACER_"))
    if lt_env:
        _print_check("LONGTRACER_* env vars", True, ", ".join(lt_env))
    else:
        _print_check("LONGTRACER_* env vars", True, "none set \u2014 using defaults", warn=True)

    # ── Backend connectivity ─────────────────────────────────────
    print("\n[ Backend ]")
    try:
        from longtracer.guard.cache import get_default_backend
        backend = get_default_backend()
        if backend is not None:
            _print_check("Default backend", True, type(backend).__name__)
        else:
            _print_check("Default backend", False, "returned None")
            errors += 1
    except Exception as exc:
        _print_check("Default backend", False, str(exc)[:80])
        errors += 1

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 62)
    if errors == 0 and warnings == 0:
        print("  \u2713  All checks passed. LongTracer is healthy.")
    elif errors == 0:
        print(f"  \u26a0  {warnings} warning(s) \u2014 installation is functional.")
        if not sts_ok or not nli_ok:
            print("     Tip: run `longtracer models prepare` to cache model weights")
            print("     and eliminate the cold-start download on first verification.")
    else:
        print(f"  \u2717  {errors} error(s), {warnings} warning(s).")
        print("     Fix errors above before running verification.")
    print("=" * 62)
    print()

    if errors > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# models prepare  (Roadmap Queue 6 — longtracer models prepare)
# ---------------------------------------------------------------------------

def cmd_models_prepare(args):
    """
    Pre-download STS and NLI model weights to the local HuggingFace cache.

    Safe to re-run — already-cached weights are loaded from disk instantly.
    Eliminates the cold-start model download that occurs on the first
    ``longtracer check`` or ``CitationVerifier()`` call.

    Disk usage:
      STS (all-MiniLM-L6-v2)        ~90 MB
      NLI (nli-deberta-v3-xsmall)   ~90 MB
    """
    import time

    print()
    print("=" * 62)
    print("  longtracer models prepare")
    print("=" * 62)
    print()
    print("Downloading model weights (first run may take a few minutes).")
    print("Weights are cached in ~/.cache/huggingface/hub and")
    print("reused automatically on every subsequent call.")
    print()

    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
    except ImportError:
        print("  \u2717  sentence-transformers is not installed.")
        print("     Install with: pip install longtracer")
        sys.exit(1)

    errors = []

    # 1. STS bi-encoder
    sts_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[ 1/2 ]  Loading STS bi-encoder")
    print(f"         {sts_name}")
    t0 = time.time()
    try:
        SentenceTransformer(sts_name)
        elapsed = (time.time() - t0) * 1000
        print(f"         \u2713  Ready  ({elapsed:.0f} ms)")
    except Exception as exc:
        print(f"         \u2717  FAILED: {exc}")
        errors.append(("STS bi-encoder", str(exc)))
    print()

    # 2. NLI cross-encoder
    nli_name = "cross-encoder/nli-deberta-v3-xsmall"
    print(f"[ 2/2 ]  Loading NLI cross-encoder")
    print(f"         {nli_name}")
    t0 = time.time()
    try:
        CrossEncoder(nli_name)
        elapsed = (time.time() - t0) * 1000
        print(f"         \u2713  Ready  ({elapsed:.0f} ms)")
    except Exception as exc:
        print(f"         \u2717  FAILED: {exc}")
        errors.append(("NLI cross-encoder", str(exc)))
    print()

    print("=" * 62)
    if not errors:
        print("  \u2713  Both models ready.")
        print("     The cold-start penalty on your first verification is now")
        print("     eliminated. Run `longtracer doctor` to verify full health.")
    else:
        print(f"  \u2717  {len(errors)} model(s) failed:")
        for name, msg in errors:
            print(f"     \u2022 {name}: {msg}")
        print()
        print("  Possible causes:")
        print("    - No internet connection")
        print("    - Insufficient disk space (~180 MB needed for both models)")
        print("    - HuggingFace Hub outage  https://status.huggingface.co")
    print("=" * 62)
    print()

    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def main():
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="longtracer",
        description="LongTracer \u2014 RAG grounding verification and trace management",
    )
    sub = parser.add_subparsers(dest="command")

    # ── view ──────────────────────────────────────────────────────
    vp = sub.add_parser("view", help="View traces")
    vp.add_argument("--id", help="View a specific trace by ID")
    vp.add_argument("--last", action="store_true", help="View most recent trace")
    vp.add_argument("--export", metavar="TRACE_ID", help="Export trace to JSON")
    vp.add_argument("--html", metavar="TRACE_ID", help="Export trace to HTML report")
    vp.add_argument("--output", "-o", help="Output file path")
    vp.add_argument("--project", "-p", help="Filter by project name")
    vp.add_argument("--limit", type=int, default=10, help="Max traces to list (default: 10)")

    # ── check ─────────────────────────────────────────────────────
    cp = sub.add_parser("check", help="Verify a response against sources")
    cp.add_argument("response", help="LLM response text to verify")
    cp.add_argument("sources", nargs="+", help="Source text(s) to verify against")
    cp.add_argument("--json", dest="json_output", action="store_true",
                    help="Output results as JSON")
    cp.add_argument("--threshold", type=float, default=0.5,
                    help="Verification threshold (default: 0.5)")

    # ── serve ─────────────────────────────────────────────────────
    sp = sub.add_parser("serve", help="Start the REST API and dashboard server")
    sp.add_argument("--host", default="0.0.0.0",
                    help="Bind address (default: 0.0.0.0)")
    sp.add_argument("--port", type=int, default=8000,
                    help="Port (default: 8000 — dashboard at http://localhost:8000/dashboard)")
    sp.add_argument("--workers", type=int, default=1,
                    help="Worker processes (default: 1)")
    sp.add_argument("--reload", action="store_true",
                    help="Enable auto-reload (dev mode only)")

    # ── doctor ────────────────────────────────────────────────────
    sub.add_parser(
        "doctor",
        help=(
            "Inspect installation health: Python version, package version, "
            "model cache, optional extras, storage writability, and configuration. "
            "Read-only — never mutates traces or data."
        ),
    )

    # ── models ────────────────────────────────────────────────────
    mp = sub.add_parser("models", help="Manage model weights")
    msub = mp.add_subparsers(dest="models_command")
    msub.add_parser(
        "prepare",
        help=(
            "Pre-download STS and NLI model weights (~180 MB total) so the "
            "first `longtracer check` starts instantly without a cold-start download."
        ),
    )

    args = parser.parse_args()

    # Default to `view` when called with no subcommand
    if args.command is None:
        args.command = "view"
        args.id = None
        args.last = False
        args.export = None
        args.html = None
        args.output = None
        args.project = None
        args.limit = 10

    if args.command == "check":
        cmd_check(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "models":
        if getattr(args, "models_command", None) == "prepare":
            cmd_models_prepare(args)
        else:
            mp.print_help()
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
