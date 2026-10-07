"""
Harness self-tests (no model weights). Run explicitly:

    pytest benchmarks/evaluator/test_harness.py

Not collected by the default suite (testpaths = ["tests"]).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as ds  # noqa: E402
import run  # noqa: E402

DATA = json.loads((Path(__file__).resolve().parent / "dataset.json").read_text())


def test_wilson_interval_known_values():
    lo, hi = run.wilson_score_interval(33, 40)
    assert round(lo, 3) == 0.681 and round(hi, 3) == 0.913
    assert run.wilson_score_interval(0, 0) is None


def test_rates_always_show_denominator():
    assert "(3/4)" in run.fmt_rate(3, 4) and "no CI" in run.fmt_rate(3, 4)
    assert "(33/40)" in run.fmt_rate(33, 40) and "95% CI" in run.fmt_rate(33, 40)
    assert run.fmt_rate(0, 0) == "n/a (0/0)"


def test_report_escapes_html_and_pipes():
    out = run.md_escape("<script>alert(1)</script> a | b")
    assert "<script>" not in out and "&lt;script&gt;" in out and "|" not in out


def test_alignment_never_scores_missing_claims():
    pred = [SimpleNamespace(claim_text="A claim.", assessment=SimpleNamespace(value="SUPPORTED"))]
    pairs, fails = run._align_claims(
        [{"claim_text": "A claim.", "assessment": "SUPPORTED"}, {"claim_text": "B.", "assessment": "CONTRADICTED"}],
        pred,
    )
    assert pairs == [("SUPPORTED", "SUPPORTED")] and fails == 1


def test_dataset_validates_against_contracts():
    assert ds.validate_cases(DATA["cases"]) == []


def test_all_14_categories_present():
    cats = {c["category"] for c in DATA["cases"]}
    assert len(cats) == 14


def test_both_splits_present_and_stratified():
    for cat in {c["category"] for c in DATA["cases"]}:
        splits = {c["split"] for c in DATA["cases"] if c["category"] == cat}
        assert splits == {"calibration", "heldout"}, cat


def test_no_leakage_between_splits():
    assert ds.check_dataset_leakage(DATA["cases"]) == []


def test_leakage_checker_detects_duplicates():
    a = {"case_id": "a", "split": "calibration", "response": "The sky is blue today.", "sources": [{"text": "x"}]}
    b = {"case_id": "b", "split": "heldout", "response": "the sky  is blue today.", "sources": [{"text": "y"}]}
    assert ds.check_dataset_leakage([a, b])


def test_review_status_is_honest():
    """No case may claim 'reviewed' without two named reviewers."""
    for c in DATA["cases"]:
        if c["review"]["status"] == "reviewed":
            assert ds.is_reviewed(c), c["case_id"]
    assert DATA["metadata"]["human_reviewed_cases"] == sum(ds.is_reviewed(c) for c in DATA["cases"])


def test_refuses_to_run_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert run.main(["--limit", "1"]) == 2
