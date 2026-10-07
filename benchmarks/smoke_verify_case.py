"""
Real-model smoke test for the v0.3.0 typed result path (handover C.5).

Runs ``verify_case`` / ``check_case`` with real STS + NLI weights on every C.3
edge case and checks the expected states. Mocks prove control flow; this proves
evaluator behaviour on this machine. Not part of the default CI job.

Usage:
    HF_HUB_OFFLINE=1 python benchmarks/smoke_verify_case.py
"""

from __future__ import annotations

import sys

from longtracer import check_case
from longtracer.contracts.result import AssessmentAvailability as A
from longtracer.contracts.result import ClaimAssessment as C
from longtracer.contracts.result import ExecutionStatus as E
from longtracer.contracts.result import QualityGate as G
from longtracer.contracts.result import ReasonCode as R
from longtracer.guard.verifier import CitationVerifier

WATER = ["Water boils at 100 degrees Celsius at standard sea-level pressure."]
TOWER = ["The Eiffel Tower was completed in 1889.", "The Eiffel Tower was completed in 1925, not 1889."]


def main() -> int:
    v = CitationVerifier()
    checks = [
        # name, result, {field: expected}
        (
            "empty",
            v.verify_case("", WATER),
            {"availability": A.NO_ASSESSABLE_CLAIMS, "reason": R.EMPTY_RESPONSE, "quality_gate": G.FAIL},
        ),
        ("whitespace", v.verify_case("  \n ", WATER), {"availability": A.NO_ASSESSABLE_CLAIMS, "quality_gate": G.FAIL}),
        (
            "too short",
            v.verify_case("Yes, it does.", WATER),
            {"reason": R.NO_EXTRACTABLE_CLAIMS, "quality_gate": G.FAIL},
        ),
        (
            "no sources",
            v.verify_case("Water boils at 100 degrees Celsius at sea level.", []),
            {"reason": R.NO_SOURCES_SUPPLIED, "claim0": C.INSUFFICIENT_EVIDENCE, "quality_gate": G.FAIL},
        ),
        (
            "empty source text",
            v.verify_case("Water boils at 100 degrees Celsius at sea level.", [""]),
            {"claim0": C.INSUFFICIENT_EVIDENCE, "claim0_reason": R.NO_SOURCE_TEXT, "quality_gate": G.FAIL},
        ),
        (
            "supported",
            v.verify_case("Water boils at 100 degrees Celsius at standard pressure.", WATER),
            {"claim0": C.SUPPORTED, "quality_gate": G.PASS},
        ),
        (
            "contradicted",
            v.verify_case("Water boils at 50 degrees Celsius at standard sea-level pressure.", WATER),
            {"claim0": C.CONTRADICTED, "quality_gate": G.FAIL},
        ),
        (
            "refusal",
            v.verify_case("The provided documents do not contain information about the population of Paris.", WATER),
            {"reason": R.HONEST_UNCERTAINTY_ONLY, "quality_gate": G.PASS},
        ),
        (
            "conflict (default off)",
            v.verify_case("The Eiffel Tower was completed in 1889.", TOWER),
            {"claim0": C.SUPPORTED, "quality_gate": G.PASS},
        ),
        (
            "conflict (opt-in)",
            v.verify_case("The Eiffel Tower was completed in 1889.", TOWER, detect_conflicts=True),
            {"claim0": C.CONFLICTING_SOURCES, "quality_gate": G.FAIL},
        ),
        (
            "timeout",
            v.verify_case("Water boils at 100 degrees Celsius at standard pressure.", WATER, timeout=0.001),
            {"execution": E.TIMEOUT, "quality_gate": G.INDETERMINATE},
        ),
        (
            "check_case",
            check_case("Water boils at 100 degrees Celsius at standard pressure.", WATER),
            {"execution": E.SUCCESS, "claim0": C.SUPPORTED},
        ),
    ]

    failures = 0
    for name, res, expected in checks:
        actual = {
            "execution": res.execution,
            "availability": res.availability,
            "reason": res.reason,
            "quality_gate": res.quality_gate,
            "claim0": res.claims[0].assessment if res.claims else None,
            "claim0_reason": res.claims[0].reason if res.claims else None,
        }
        bad = {k: (actual[k], v_) for k, v_ in expected.items() if actual[k] != v_}
        status = "ok  " if not bad else "FAIL"
        failures += bool(bad)
        claim = f" claim0={actual['claim0'].value}" if actual["claim0"] else ""
        print(
            f"{status} {name:24} exec={res.execution.value:8} avail={res.availability.value:21} "
            f"gate={res.quality_gate.value:13} reason={(res.reason.value if res.reason else '-'):24}{claim}"
        )
        for k, (got, want) in bad.items():
            print(f"       expected {k}={want.value}, got {getattr(got, 'value', got)}")
        assert res.schema_version == "1"
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
