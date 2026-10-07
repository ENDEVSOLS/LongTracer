"""
Evaluator benchmark harness for LongTracer v0.3.0 (handover Workstream D.2).

Runs ``CitationVerifier.verify_case`` with REAL model weights over a split of
``benchmarks/evaluator/dataset.json`` and writes a Markdown + JSON report with
per-label precision/recall, confusion counts, false-alert rate, evaluator error
count, sample counts, and Wilson 95% intervals. Every rate is shown with its
numerator and denominator.

Opt-in only: refuses to run under CI (``CI=true``) unless ``--allow-ci`` is given,
and refuses mocked models.

Usage:
    make benchmark
    HF_HUB_OFFLINE=1 python benchmarks/evaluator/run.py --split heldout --out benchmarks/evaluator/reports/
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

from longtracer.contracts.result import ExecutionStatus
from longtracer.guard.verifier import CitationVerifier

DATASET_DEFAULT = Path(__file__).resolve().parent / "dataset.json"
REPORTS_DEFAULT = Path(__file__).resolve().parent / "reports"
LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_SOURCES"]
MODELS = {
    "sts": "sentence-transformers/all-MiniLM-L6-v2",
    "nli": "cross-encoder/nli-deberta-v3-xsmall",
}
CI_MIN_N = 10  # below this, intervals are not reported


def wilson_score_interval(positives: int, total: int, z: float = 1.95996) -> Optional[Tuple[float, float]]:
    """Wilson score 95% interval for a binomial proportion (None when total == 0)."""
    if total <= 0:
        return None
    p = positives / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(positives: int, total: int) -> str:
    """'x% (num/den) [95% CI]' — never a bare rate."""
    if total == 0:
        return f"n/a (0/0)"
    rate = positives / total * 100.0
    if total < CI_MIN_N:
        return f"{rate:.1f}% ({positives}/{total}) [n<{CI_MIN_N}, no CI]"
    lo, hi = wilson_score_interval(positives, total)  # type: ignore[misc]
    return f"{rate:.1f}% ({positives}/{total}) [95% CI {lo * 100:.1f}–{hi * 100:.1f}%]"


def md_escape(text: str, limit: int = 140) -> str:
    """Escape case text for Markdown tables and HTML renderers (handover §9)."""
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return html.escape(text).replace("|", "&#124;")


def _model_revision(model_id: str) -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        ref = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
        return ref.read_text().strip()
    except Exception:
        return "unknown"


def get_environment_info() -> Dict[str, Any]:
    import sentence_transformers
    import torch
    import transformers

    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "longtracer"], capture_output=True, text=True
        ).stdout.strip()
    )
    return {
        "commit": commit + ("-dirty" if dirty else ""),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "models": {k: {"id": v, "revision": _model_revision(v)} for k, v in MODELS.items()},
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
    }


def _align_claims(expected: List[Dict[str, Any]], predicted: List[Any]) -> Tuple[List[Tuple[str, str]], int]:
    """Pair expected and predicted claims by text, then by position.

    Returns (pairs of (expected_label, predicted_label), alignment_failures). An
    expected claim with no predicted counterpart is an alignment failure — it is
    reported, never silently scored as a label.
    """
    remaining = list(predicted)
    pairs: List[Tuple[str, str]] = []
    failures = 0
    for i, exp in enumerate(expected):
        match = next((p for p in remaining if p.claim_text.strip() == exp["claim_text"].strip()), None)
        if match is None and i < len(predicted) and predicted[i] in remaining:
            match = predicted[i]
        if match is None:
            failures += 1
            continue
        remaining.remove(match)
        pairs.append((exp["assessment"], match.assessment.value))
    return pairs, failures


def evaluate_split(
    verifier: CitationVerifier,
    cases: List[Dict[str, Any]],
    split: str,
    limit: Optional[int] = None,
    detect_conflicts: bool = True,
) -> Dict[str, Any]:
    selected = [c for c in cases if split == "all" or c.get("split") == split]
    if limit:
        selected = selected[:limit]

    confusion = {e: {p: 0 for p in LABELS} for e in LABELS}
    evaluator_errors: List[Dict[str, str]] = []
    alignment_failures = 0
    gate_agree = 0
    expected_pass = false_alerts = expected_pass_indeterminate = 0
    by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "gate_agree": 0})
    disagreements: List[Dict[str, Any]] = []
    durations: List[float] = []

    print(f"Running {len(selected)} cases (split={split}, detect_conflicts={detect_conflicts}) ...")
    for i, case in enumerate(selected, 1):
        t0 = time.perf_counter()
        res = verifier.verify_case(
            case["response"],
            [s["text"] for s in case["sources"]],
            source_metadata=[{"source_id": s["source_id"], **s.get("metadata", {})} for s in case["sources"]],
            case_id=case["case_id"],
            detect_conflicts=detect_conflicts,
        )
        durations.append((time.perf_counter() - t0) * 1000.0)
        exp = case["expected"]
        cat = case["category"]
        by_category[cat]["total"] += 1

        if res.execution != ExecutionStatus.SUCCESS:
            evaluator_errors.append(
                {
                    "case_id": case["case_id"],
                    "execution": res.execution.value,
                    "reason": res.reason.value if res.reason else "",
                    "error": res.error_message or "",
                }
            )
            continue  # nothing was assessed: excluded from claim metrics, counted as an error

        gate_ok = res.quality_gate.value == exp["quality_gate"]
        gate_agree += gate_ok
        by_category[cat]["gate_agree"] += gate_ok
        if exp["quality_gate"] == "PASS":
            expected_pass += 1
            false_alerts += res.quality_gate.value == "FAIL"
            expected_pass_indeterminate += res.quality_gate.value == "INDETERMINATE"

        pairs, fails = _align_claims(exp.get("claims", []), res.claims)
        alignment_failures += fails
        for e_lbl, p_lbl in pairs:
            confusion[e_lbl][p_lbl] += 1
        if not gate_ok or any(e != p for e, p in pairs) or fails:
            disagreements.append(
                {
                    "case_id": case["case_id"],
                    "category": cat,
                    "response": case["response"],
                    "expected_gate": exp["quality_gate"],
                    "predicted_gate": res.quality_gate.value,
                    "expected_claims": [c["assessment"] for c in exp.get("claims", [])],
                    "predicted_claims": [c.assessment.value for c in res.claims],
                }
            )
        if i % 20 == 0 or i == len(selected):
            print(f"  {i}/{len(selected)}")

    per_label = {}
    for lbl in LABELS:
        tp = confusion[lbl][lbl]
        support = sum(confusion[lbl].values())
        predicted = sum(confusion[e][lbl] for e in LABELS)
        per_label[lbl] = {"tp": tp, "predicted": predicted, "support": support}

    contra = per_label["CONTRADICTED"]
    s = sorted(durations)
    return {
        "split": split,
        "detect_conflicts": detect_conflicts,
        "total_cases": len(selected),
        "assessed_cases": len(selected) - len(evaluator_errors),
        "evaluator_errors": evaluator_errors,
        "alignment_failures": alignment_failures,
        "gate_agreement": {"positives": gate_agree, "total": len(selected) - len(evaluator_errors)},
        "false_alert_rate": {
            "positives": false_alerts,
            "total": expected_pass,
            "indeterminate_on_expected_pass": expected_pass_indeterminate,
        },
        "contradiction_alert_precision": {"positives": contra["tp"], "total": contra["predicted"]},
        "confusion_matrix": confusion,
        "per_label": per_label,
        "category_results": dict(by_category),
        "disagreements": disagreements,
        "latency_ms": {
            "n": len(s),
            "p50": s[len(s) // 2] if s else 0.0,
            "p95": s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)] if s else 0.0,
        },
    }


def write_markdown_report(results: Dict[str, Any], env: Dict[str, Any], meta: Dict[str, Any], out_path: Path) -> None:
    fa, cp = results["false_alert_rate"], results["contradiction_alert_precision"]
    reviewed = meta.get("human_reviewed_cases", 0)
    md: List[str] = [
        "# LongTracer v0.3.0 Evaluator Benchmark Report",
        "",
        "> Internal engineering measurement. Not a public accuracy claim. English, synthetic",
        "> general-knowledge cases only; no domain or multilingual conclusions are supported.",
        "",
    ]
    if reviewed < results["total_cases"]:
        md += [
            f"> **Provisional:** {reviewed} of {meta.get('total_cases', '?')} dataset cases are human-reviewed. "
            "Labels were drafted and have not yet been signed off by two human reviewers, so these numbers "
            "measure agreement with draft labels.",
            "",
        ]
    md += [
        "| Item | Value |",
        "|---|---|",
        f"| Split | `{results['split']}` ({results['total_cases']} cases) |",
        f"| Dataset version | `{meta.get('dataset_version', '?')}` |",
        f"| Conflict detection | `{results['detect_conflicts']}` |",
        f"| Commit | `{env['commit']}` |",
        f"| Hardware | {env['cpu']} ({env['logical_cpus']} threads), device `{env['device']}`, {env['os']} |",
        f"| Python / torch / sentence-transformers / transformers | {env['python']} / {env['torch']} / "
        f"{env['sentence_transformers']} / {env['transformers']} |",
        f"| STS model | `{env['models']['sts']['id']}` @ `{env['models']['sts']['revision']}` |",
        f"| NLI model | `{env['models']['nli']['id']}` @ `{env['models']['nli']['revision']}` |",
        f"| Thresholds | support 0.40, NLI gate 0.25, contradiction/entailment 0.5 (unchanged) |",
        f"| Latency per case | p50 {results['latency_ms']['p50']:.0f} ms, p95 {results['latency_ms']['p95']:.0f} ms "
        f"(n={results['latency_ms']['n']}) |",
        f"| Generated | {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} |",
        "",
        "## 1. Release-gate hypotheses vs. measured",
        "",
        "Targets are hypotheses to measure, not achievements to assert. Misses are reported as-is.",
        "",
        "| Metric | Target | Measured | Meets target |",
        "|---|---|---|---|",
    ]
    cp_rate = cp["positives"] / cp["total"] if cp["total"] else 0.0
    fa_rate = fa["positives"] / fa["total"] if fa["total"] else 0.0
    md.append(
        f"| Contradiction-alert precision | ≥ 95% | {fmt_rate(cp['positives'], cp['total'])} | "
        f"{'yes' if cp['total'] and cp_rate >= 0.95 else '**no**'} |"
    )
    md.append(
        f"| False alerts on expected-PASS cases (gate FAIL) | ≤ 5% | {fmt_rate(fa['positives'], fa['total'])} | "
        f"{'yes' if fa['total'] and fa_rate <= 0.05 else '**no**'} |"
    )
    n_err = len(results["evaluator_errors"])
    md += [
        f"| Evaluator errors (ERROR / TIMEOUT / CANCELLED) | report | {n_err}/{results['total_cases']} cases | — |",
        f"| Expected-PASS cases gated INDETERMINATE | report | {fa['indeterminate_on_expected_pass']}/{fa['total']} | — |",
        f"| Claim alignment failures (expected claim not produced) | report | {results['alignment_failures']} | — |",
        f"| Case-level quality-gate agreement | report | "
        f"{fmt_rate(results['gate_agreement']['positives'], results['gate_agreement']['total'])} | — |",
        "",
        "## 2. Per-label precision and recall (claim level)",
        "",
        "| Label | Precision | Recall | Expected (support) | Predicted |",
        "|---|---|---|---|---|",
    ]
    for lbl, m in results["per_label"].items():
        md.append(
            f"| `{lbl}` | {fmt_rate(m['tp'], m['predicted'])} | {fmt_rate(m['tp'], m['support'])} | "
            f"{m['support']} | {m['predicted']} |"
        )
    md += [
        "",
        "## 3. Confusion counts (rows = expected, columns = predicted)",
        "",
        "| Expected \\ Predicted | " + " | ".join(f"`{x}`" for x in LABELS) + " | Total |",
        "|---|" + "---|" * (len(LABELS) + 1),
    ]
    for e in LABELS:
        row = results["confusion_matrix"][e]
        md.append(f"| `{e}` | " + " | ".join(str(row[p]) for p in LABELS) + f" | {sum(row.values())} |")
    md += ["", "## 4. Case-level gate agreement by category", "", "| Category | Agreement |", "|---|---|"]
    for cat, c in sorted(results["category_results"].items()):
        md.append(f"| `{cat}` | {fmt_rate(c['gate_agree'], c['total'])} |")
    if results["evaluator_errors"]:
        md += ["", "## 5. Evaluator errors", "", "| Case | Execution | Reason | Error |", "|---|---|---|---|"]
        for e in results["evaluator_errors"]:
            md.append(f"| `{e['case_id']}` | {e['execution']} | {e['reason']} | {md_escape(e['error'])} |")
    md += [
        "",
        f"## {6 if results['evaluator_errors'] else 5}. Disagreements ({len(results['disagreements'])})",
        "",
        "| Case | Category | Response | Gate exp → pred | Claims exp → pred |",
        "|---|---|---|---|---|",
    ]
    for d in results["disagreements"]:
        md.append(
            f"| `{d['case_id']}` | {d['category']} | {md_escape(d['response'])} | "
            f"{d['expected_gate']} → {d['predicted_gate']} | "
            f"{', '.join(d['expected_claims']) or '—'} → {', '.join(d['predicted_claims']) or '—'} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["heldout", "calibration", "all"], default="heldout")
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--out", type=Path, default=REPORTS_DEFAULT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-conflicts", action="store_true", help="Run with detect_conflicts=False")
    parser.add_argument("--allow-ci", action="store_true", help="Allow running when CI=true")
    args = parser.parse_args(argv)

    if os.environ.get("CI", "").lower() in ("1", "true", "yes") and not args.allow_ci:
        print(
            "Refusing to run in CI: the benchmark needs real weights and real time (use --allow-ci).", file=sys.stderr
        )
        return 2
    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}. Run `python benchmarks/evaluator/dataset.py`.", file=sys.stderr)
        return 1

    verifier = CitationVerifier()
    if isinstance(verifier.model, Mock) or type(verifier.model).__name__ != "HybridVerificationModel":
        print("Refusing to run with a mocked model: benchmarks need real weights.", file=sys.stderr)
        return 1

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    meta = {**data.get("metadata", {}), "dataset_version": data.get("dataset_version")}
    env = get_environment_info()
    results = evaluate_split(verifier, data["cases"], args.split, args.limit, detect_conflicts=not args.no_conflicts)
    results["environment"] = env
    results["dataset"] = meta

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark_report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(results, env, meta, args.out / "benchmark_report.md")
    cp = results["contradiction_alert_precision"]
    fa = results["false_alert_rate"]
    print(f"\nReport: {args.out / 'benchmark_report.md'}")
    print(f"Contradiction-alert precision: {fmt_rate(cp['positives'], cp['total'])}")
    print(f"False-alert rate:              {fmt_rate(fa['positives'], fa['total'])}")
    print(f"Evaluator errors:              {len(results['evaluator_errors'])}/{results['total_cases']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
