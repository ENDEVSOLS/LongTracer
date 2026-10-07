"""
Performance baseline — cold model load, warm verification latency, and peak memory.

Internal engineering measurement for docs/development/current-baseline.md.
Not a public performance guarantee. Requires real model weights
(run ``longtracer models prepare`` first).

Usage:
    python benchmarks/perf_baseline.py                      # legacy verify_parallel path
    python benchmarks/perf_baseline.py --mode case          # verify_case (v0.3.0+)
    python benchmarks/perf_baseline.py --mode case-conflicts
    python benchmarks/perf_baseline.py --cold-runs 3 --warm-runs 50 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

# ── Reference workload (fixed, deterministic) ───────────────────────────────
# 5 sources x 5 sentences; response with 8 claims (6 grounded, 1 contradicted, 1 unrelated).
SOURCES = [
    (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was designed by the engineering company of Gustave Eiffel. "
        "Construction of the tower finished in 1889 for the World's Fair. "
        "The tower is 330 metres tall including its antennas. "
        "It is the most visited paid monument in the world."
    ),
    (
        "Water boils at 100 degrees Celsius at standard sea-level pressure. "
        "At higher altitudes the boiling point of water is lower. "
        "Pure water freezes at 0 degrees Celsius under standard conditions. "
        "Water has its maximum density at about 4 degrees Celsius. "
        "The chemical formula of water is H2O."
    ),
    (
        "The Python programming language was created by Guido van Rossum. "
        "Its first public release, version 0.9.0, appeared in 1991. "
        "Python 3.0 was released in December 2008. "
        "Python emphasises code readability with significant indentation. "
        "The language is maintained by the Python Software Foundation."
    ),
    (
        "Mount Everest is Earth's highest mountain above sea level. "
        "Its summit lies on the border between Nepal and China. "
        "The officially recognised height is 8,848.86 metres. "
        "The first confirmed ascent was made in 1953 by Tenzing Norgay and Edmund Hillary. "
        "Climbers typically use supplemental oxygen above 8,000 metres."
    ),
    (
        "The Amazon rainforest covers much of the Amazon basin in South America. "
        "The majority of the forest is contained within Brazil. "
        "It is home to an estimated 390 billion individual trees. "
        "The rainforest plays a major role in regulating the global climate. "
        "Deforestation is a significant threat to the region."
    ),
]

RESPONSE = (
    "The Eiffel Tower was designed by the company of Gustave Eiffel. "
    "Its construction was completed in 1889 for the World's Fair. "
    "Water boils at 100 degrees Celsius at sea-level pressure. "
    "Python was created by Guido van Rossum and first released in 1991. "
    "Mount Everest's summit lies on the border between Nepal and China. "
    "Most of the Amazon rainforest is located within Brazil. "
    "The first confirmed ascent of Everest happened in 1975. "
    "Bananas are an excellent source of dietary potassium."
)


def _pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation) — simple and reproducible."""
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[k]


def _peak_rss_mb() -> float:
    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cold_load_once() -> dict:
    """Run in a child process: time import + shared model load."""
    t0 = time.perf_counter()
    from longtracer.guard.nli_model import get_shared_model

    t_import = time.perf_counter()
    get_shared_model(verbose=False)
    t_load = time.perf_counter()
    return {
        "import_s": t_import - t0,
        "model_load_s": t_load - t_import,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _environment() -> dict:
    import torch
    import sentence_transformers
    import transformers

    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "commit": commit,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["legacy", "case", "case-conflicts"], default="legacy")
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--warm-runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", dest="json_path", default=None, help="Write results to this JSON file")
    parser.add_argument("--_cold-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._cold_child:
        print(json.dumps(_cold_load_once()))
        return 0

    # Cold load: fresh interpreter per run so nothing is cached in-process.
    cold = []
    for _ in range(args.cold_runs):
        out = (
            subprocess.run([sys.executable, __file__, "--_cold-child"], capture_output=True, text=True, check=True)
            .stdout.strip()
            .splitlines()[-1]
        )
        cold.append(json.loads(out))

    # Warm: one process, models loaded once, repeated verification.
    from longtracer.guard.verifier import CitationVerifier

    verifier = CitationVerifier()

    def run_once() -> None:
        if args.mode == "legacy":
            verifier.verify_parallel(RESPONSE, SOURCES)
        else:
            verifier.verify_case(RESPONSE, SOURCES, detect_conflicts=(args.mode == "case-conflicts"))

    for _ in range(args.warmup):
        run_once()
    timings_ms = []
    for _ in range(args.warm_runs):
        t0 = time.perf_counter()
        run_once()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    from longtracer.guard.claim_splitter import split_into_claims

    result = {
        "mode": args.mode,
        "environment": _environment(),
        "models": {
            "sts": "sentence-transformers/all-MiniLM-L6-v2",
            "nli": "cross-encoder/nli-deberta-v3-xsmall",
        },
        "workload": {
            "sources": len(SOURCES),
            "source_chars": sum(len(s) for s in SOURCES),
            "response_chars": len(RESPONSE),
            "claims": len(split_into_claims(RESPONSE)),
            "batch_size": 1,
            "cache": False,
        },
        "cold": {
            "runs": len(cold),
            "model_load_s_median": statistics.median(c["model_load_s"] for c in cold),
            "model_load_s_max": max(c["model_load_s"] for c in cold),
            "import_s_median": statistics.median(c["import_s"] for c in cold),
            "peak_rss_mb_after_load_max": max(c["peak_rss_mb"] for c in cold),
        },
        "warm": {
            "runs": len(timings_ms),
            "warmup_runs": args.warmup,
            "p50_ms": _pct(timings_ms, 50),
            "p95_ms": _pct(timings_ms, 95),
            "min_ms": min(timings_ms),
            "max_ms": max(timings_ms),
            "peak_rss_mb": _peak_rss_mb(),
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_path:
        with open(args.json_path, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
