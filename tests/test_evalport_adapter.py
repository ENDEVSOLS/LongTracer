"""Tests for the EvalPort adapter (longtracer.adapters.evalport / longtracer.to_openeval).

`VerificationResult` and its claim dicts are built directly against the real
`longtracer.guard.verifier.VerificationResult` dataclass and the claim dict
shape produced by `longtracer.guard.nli_model.HybridVerificationModel` --
unlike a standalone adapter package with no dependency on the real
`longtracer`, `sentence-transformers` is already a core dependency of this
package (see pyproject.toml), so there's no reason to fake the dataclass
here. This follows the same hand-built-claim-dict convention already used by
tests/test_verifier_edge_cases.py, and avoids ever loading the real NLI
models in CI.

The design constraints this adapter satisfies (per the maintainer's review on
ENDEVSOLS/LongTracer#15) are exercised directly:
  * `to_openeval()` only, no `from_openeval()` -- LongTracer produces
    verification results, it doesn't consume EvalPort suites.
  * Response-level fields (trust_score, verdict, summary, latency_stats) and
    claim-level evidence (supported, score, best_source, is_hallucination)
    are exact passthroughs, never recalculated.
  * Unsupported claims stay distinct from confirmed hallucinations.
  * Both a single `VerificationResult` and a batch are supported.

`evalport-sdk` (the `openeval.*` package) is an optional extra
(`pip install "longtracer[evalport]"`), and this repo's CI does a bare
`pip install -e "."` with no extras -- so every real-schema-validation
assertion below goes through `pytest.importorskip("openeval.validate", ...)`
immediately before use, exactly mirroring the graceful-degradation pattern
`scripts/export_openeval.py` uses in TIGER-AI-Lab/ClawBench. When
evalport-sdk *is* installed (e.g. `pip install -e ".[evalport]"` or
`".[all]"` locally, or via CI's `evalport` matrix job if one is added later),
these assertions run for real and confirm the adapter's output is valid
against the actual EvalPort schema, not just internally self-consistent.
"""

from __future__ import annotations

import sys
from collections import deque
from importlib import metadata as importlib_metadata

import pytest

from longtracer.adapters.evalport import to_openeval
from longtracer.guard.verifier import VerificationResult


def _skip_reason() -> str:
    return "evalport-sdk not installed; skipping real-schema validation"


def _claim(
    claim="Paris is the capital of France.",
    supported=True,
    score=0.87,
    best_source="Paris is the capital and largest city of France.",
    is_hallucination=False,
    **overrides,
):
    """Build a claim dict matching HybridVerificationModel.verify_claim()'s output shape."""
    base = {
        "claim": claim,
        "supported": supported,
        "score": score,
        "best_score": score,
        "sentence_results": [
            {"claim_sentence": claim, "score": score, "matched_source": best_source[:100]}
        ],
        "contradiction_score": 0.02,
        "entailment_score": 0.9 if supported else 0.1,
        "nli_ran": True,
        "best_source": best_source,
        "best_source_index": 0,
        "best_source_metadata": {"doc_id": "doc_1"},
        "is_hallucination": is_hallucination,
        "is_meta_statement": False,
        "has_hallucination_pattern": is_hallucination,
    }
    base.update(overrides)
    return base


def _all_supported_result() -> VerificationResult:
    claims = [
        _claim(),
        _claim(claim="Water boils at 100C.", best_source="Water boils at 100 degrees Celsius."),
    ]
    return VerificationResult(
        trust_score=0.9,
        claims=claims,
        flagged_claims=[],
        hallucinations=[],
        all_supported=True,
        hallucination_count=0,
        latency_stats={
            "sts_calls": 2, "sts_avg_ms": 12.0,
            "nli_calls": 2, "nli_avg_ms": 45.0,
            "nli_skipped": 0, "total_ms": 114.0,
        },
    )


def _mixed_result() -> VerificationResult:
    supported_claim = _claim()
    unsupported_claim = _claim(
        claim="The Eiffel Tower is in Berlin.",
        supported=False,
        score=0.15,
        best_source="",
        is_hallucination=False,
    )
    hallucinated_claim = _claim(
        claim="Napoleon was born in 1600.",
        supported=False,
        score=0.05,
        contradiction_score=0.91,
        entailment_score=0.02,
        best_source="Napoleon was born in 1769.",
        is_hallucination=True,
    )
    claims = [supported_claim, unsupported_claim, hallucinated_claim]
    return VerificationResult(
        trust_score=0.35,
        claims=claims,
        flagged_claims=[unsupported_claim, hallucinated_claim],
        hallucinations=[hallucinated_claim],
        all_supported=False,
        hallucination_count=1,
        latency_stats={
            "sts_calls": 3, "sts_avg_ms": 10.0,
            "nli_calls": 3, "nli_avg_ms": 40.0,
            "nli_skipped": 0, "total_ms": 150.0,
        },
    )


# --- shape / schema validity -------------------------------------------------


def test_single_result_shape_and_validity():
    vr = _all_supported_result()
    rs = to_openeval(vr, run_id="run_1", started_at="2026-08-31T00:00:00Z")

    assert rs["run_id"] == "run_1"
    assert rs["suite_id"] == "longtracer_citation_verification"
    assert len(rs["results"]) == 1

    result = rs["results"][0]
    assert result["test_case_id"] == "claim_verification_0"
    assert result["passed"] is True
    assert result["duration_ms"] == 114
    assert result["metadata"]["trust_score"] == 0.9
    assert result["metadata"]["verdict"] == "PASS"
    assert len(result["grader_results"]) == 2

    openeval_validate = pytest.importorskip("openeval.validate", reason=_skip_reason())
    validation = openeval_validate.validate_result_set(rs)
    assert validation.valid, validation.errors


def test_response_level_fields_preserved_exactly():
    """trust_score, verdict, summary, latency_stats must be exact passthroughs."""
    vr = _mixed_result()
    rs = to_openeval(vr)
    result = rs["results"][0]

    assert result["metadata"]["trust_score"] == vr.trust_score
    assert result["metadata"]["verdict"] == vr.verdict == "FAIL"
    assert result["metadata"]["summary"] == vr.summary
    assert result["metadata"]["latency_stats"] == vr.latency_stats
    assert result["metadata"]["hallucination_count"] == 1
    assert result["metadata"]["flagged_claim_count"] == 2
    # Result.passed is a direct rename of verdict == "PASS", not a fresh AND
    # over grader_results computed independently.
    assert result["passed"] == (vr.verdict == "PASS") == False  # noqa: E712


def test_claim_level_evidence_preserved():
    vr = _mixed_result()
    rs = to_openeval(vr)
    graders = rs["results"][0]["grader_results"]

    supported_gr, unsupported_gr, hallucination_gr = graders

    assert supported_gr["passed"] is True
    assert supported_gr["score"] == 0.87
    assert supported_gr["metadata"]["best_source"] == "Paris is the capital and largest city of France."
    assert supported_gr["metadata"]["is_hallucination"] is False

    assert unsupported_gr["passed"] is False
    assert unsupported_gr["metadata"]["is_hallucination"] is False

    assert hallucination_gr["passed"] is False
    assert hallucination_gr["metadata"]["is_hallucination"] is True


def test_unsupported_distinct_from_hallucination():
    """The core requirement from the maintainer's review (ENDEVSOLS/LongTracer#15):
    an unsupported claim that is NOT flagged as a hallucination must be
    distinguishable from one that IS -- both have passed=False, but only the
    latter has is_hallucination=True / claim_status="hallucination"."""
    vr = _mixed_result()
    rs = to_openeval(vr)
    _, unsupported_gr, hallucination_gr = rs["results"][0]["grader_results"]

    assert unsupported_gr["passed"] is False
    assert unsupported_gr["metadata"]["is_hallucination"] is False
    assert unsupported_gr["metadata"]["openeval"]["claim_status"] == "unsupported"

    assert hallucination_gr["passed"] is False
    assert hallucination_gr["metadata"]["is_hallucination"] is True
    assert hallucination_gr["metadata"]["openeval"]["claim_status"] == "hallucination"

    # And a genuinely supported claim is a third, distinct bucket.
    supported_gr = rs["results"][0]["grader_results"][0]
    assert supported_gr["metadata"]["openeval"]["claim_status"] == "supported"


def test_score_is_never_recalculated_only_clamped():
    """A claim score outside [0, 1] (cosine similarity is unbounded to
    [-1.0, 1.0]) is clamped for the schema-required `score` field, but the
    exact original LongTracer value is preserved in metadata.openeval.raw_score."""
    claim = _claim(score=-0.3)
    vr = VerificationResult(
        trust_score=0.5, claims=[claim], flagged_claims=[claim],
        hallucinations=[], all_supported=False, hallucination_count=0,
    )
    rs = to_openeval(vr)
    gr = rs["results"][0]["grader_results"][0]

    assert gr["score"] == 0.0  # clamped
    assert gr["metadata"]["openeval"]["raw_score"] == -0.3  # untouched original

    openeval_validate = pytest.importorskip("openeval.validate", reason=_skip_reason())
    validation = openeval_validate.validate_result_set(rs)
    assert validation.valid, validation.errors


# --- batches ------------------------------------------------------------------


def test_batch_of_results():
    vr1 = _all_supported_result()
    vr2 = _mixed_result()
    rs = to_openeval([vr1, vr2], run_id="batch_run")

    assert rs["run_id"] == "batch_run"
    assert len(rs["results"]) == 2
    assert rs["results"][0]["test_case_id"] == "claim_verification_0"
    assert rs["results"][1]["test_case_id"] == "claim_verification_1"
    assert rs["summary"]["total"] == 2
    assert rs["summary"]["passed"] == 1
    assert rs["summary"]["failed"] == 1
    assert rs["summary"]["avg_score"] == (vr1.trust_score + vr2.trust_score) / 2

    openeval_validate = pytest.importorskip("openeval.validate", reason=_skip_reason())
    validation = openeval_validate.validate_result_set(rs)
    assert validation.valid, validation.errors


def test_batch_detection_works_for_non_list_tuple_sequences():
    """`to_openeval()` is annotated to accept `Sequence[Any]`, not just
    `list`/`tuple` -- a `deque` (or any other real Sequence) must also be
    treated as a batch, not misread as a single VerificationResult."""
    vr1 = _all_supported_result()
    vr2 = _mixed_result()

    rs_deque = to_openeval(deque([vr1, vr2]), run_id="deque_run")
    assert len(rs_deque["results"]) == 2
    assert rs_deque["results"][0]["test_case_id"] == "claim_verification_0"
    assert rs_deque["results"][1]["test_case_id"] == "claim_verification_1"
    assert rs_deque["summary"]["total"] == 2

    openeval_validate = pytest.importorskip("openeval.validate", reason=_skip_reason())
    validation = openeval_validate.validate_result_set(rs_deque)
    assert validation.valid, validation.errors

    rs_tuple = to_openeval((vr1, vr2), run_id="tuple_run")
    assert len(rs_tuple["results"]) == 2


def test_batch_with_explicit_test_case_ids_and_response_texts():
    vr1 = _all_supported_result()
    vr2 = _mixed_result()
    rs = to_openeval(
        [vr1, vr2],
        test_case_ids=["resp_a", "resp_b"],
        response_texts=["Paris is the capital of France. Water boils at 100C.", None],
    )

    assert rs["results"][0]["test_case_id"] == "resp_a"
    assert rs["results"][1]["test_case_id"] == "resp_b"
    assert rs["results"][0]["actual_output"] == "Paris is the capital of France. Water boils at 100C."
    assert "actual_output" not in rs["results"][1]


def test_mismatched_test_case_ids_length_raises():
    vr1 = _all_supported_result()
    with pytest.raises(ValueError):
        to_openeval([vr1], test_case_ids=["a", "b"])


def test_mismatched_response_texts_length_raises():
    vr1 = _all_supported_result()
    with pytest.raises(ValueError):
        to_openeval([vr1], response_texts=["a", "b"])


# --- edge cases -----------------------------------------------------------


def test_empty_claims_result_still_valid():
    """A VerificationResult with no claims at all -- a vacuous PASS."""
    vr = VerificationResult(
        trust_score=1.0, claims=[], flagged_claims=[], hallucinations=[],
        all_supported=True, hallucination_count=0,
        latency_stats={
            "sts_calls": 0, "sts_avg_ms": 0,
            "nli_calls": 0, "nli_avg_ms": 0,
            "nli_skipped": 0, "total_ms": 0.0,
        },
    )
    rs = to_openeval(vr)
    result = rs["results"][0]

    assert result["passed"] is True
    assert result["grader_results"] == []

    openeval_validate = pytest.importorskip("openeval.validate", reason=_skip_reason())
    validation = openeval_validate.validate_result_set(rs)
    assert validation.valid, validation.errors


def test_clamp01_handles_none_nan_and_unparseable_values():
    """`_clamp01()` (used for every claim's `score`) must degrade to `None`
    for every value that can't become a valid, finite float -- `None` itself,
    a NaN float, and a value that can't be coerced to float at all -- rather
    than raising, since a malformed/missing per-claim score should surface
    as EvalPort's own "unscored" (`null`) representation, not crash the
    whole export."""
    from longtracer.adapters.evalport import _clamp01

    assert _clamp01(None) is None
    assert _clamp01(float("nan")) is None
    assert _clamp01("not-a-number") is None
    assert _clamp01([1, 2, 3]) is None  # unhashable/uncoercible type
    # And the values it *does* handle are still clamped/passed through correctly.
    assert _clamp01(0.5) == 0.5
    assert _clamp01(-5) == 0.0
    assert _clamp01(5) == 1.0
    assert _clamp01("0.42") == 0.42  # numeric strings coerce via float()


def test_is_batch_excludes_str_bytes_bytearray():
    """`_is_batch()` must never treat a `str`/`bytes`/`bytearray` as a batch
    of results, even though all three are technically `Sequence`s -- a
    `VerificationResult` is never actually any of these, but the check
    exists so a caller's mistake (e.g. accidentally passing a `str`) fails
    with a clear downstream `AttributeError`/`TypeError` from treating it as
    a single malformed result, not a silently-wrong per-character batch."""
    from longtracer.adapters.evalport import _is_batch

    assert _is_batch("not a result") is False
    assert _is_batch(b"not a result") is False
    assert _is_batch(bytearray(b"not a result")) is False
    assert _is_batch([_all_supported_result()]) is True
    assert _is_batch((_all_supported_result(),)) is True
    assert _is_batch(deque([_all_supported_result()])) is True
    assert _is_batch(_all_supported_result()) is False


def test_dict_input_also_supported():
    """to_openeval() duck-types via attribute-or-key access, so a plain dict
    (e.g. a VerificationResult round-tripped through JSON) works too."""
    vr_dict = {
        "trust_score": 0.5,
        "verdict": "FAIL",
        "summary": "1/2 claims supported.",
        "claims": [_claim(supported=False, score=0.1, is_hallucination=False)],
        "flagged_claims": [_claim(supported=False, score=0.1, is_hallucination=False)],
        "hallucinations": [],
        "all_supported": False,
        "hallucination_count": 0,
        "latency_stats": None,
    }
    rs = to_openeval(vr_dict)
    assert rs["results"][0]["passed"] is False
    assert rs["results"][0]["metadata"]["trust_score"] == 0.5

    openeval_validate = pytest.importorskip("openeval.validate", reason=_skip_reason())
    validation = openeval_validate.validate_result_set(rs)
    assert validation.valid, validation.errors


# --- runner metadata --------------------------------------------------------


def test_runner_name_is_longtracer():
    vr = _all_supported_result()
    rs = to_openeval(vr)
    assert rs["runner"]["name"] == "longtracer"


def test_runner_version_reads_installed_distribution_metadata():
    """`_longtracer_version()` resolves the installed `longtracer`
    distribution's version via `importlib.metadata.version()`. This package
    is installed (editable or otherwise) in every environment this test
    suite runs in, so the adapter's reported version must match exactly
    what `importlib.metadata.version("longtracer")` itself reports."""
    vr = _all_supported_result()
    rs = to_openeval(vr)

    try:
        expected = importlib_metadata.version("longtracer")
    except importlib_metadata.PackageNotFoundError:
        expected = None

    assert rs["runner"]["version"] == expected


def test_runner_version_uses_importlib_metadata_not_a_manual_lookup(monkeypatch):
    """When `importlib.metadata.version()` reports a given version for the
    `longtracer` distribution, `to_openeval()` reports that exact value --
    confirming `_longtracer_version()` goes through `importlib.metadata`
    rather than any other source (e.g. a hardcoded `__version__`
    attribute)."""
    import longtracer.adapters.evalport as evalport_module

    monkeypatch.setattr(
        evalport_module._importlib_metadata,
        "version",
        lambda name: "9.9.9" if name == "longtracer" else None,
    )
    vr = _all_supported_result()
    rs = to_openeval(vr)
    assert rs["runner"]["version"] == "9.9.9"


def test_runner_version_is_none_when_distribution_not_found(monkeypatch):
    """If the `longtracer` distribution metadata can't be found at all (e.g.
    a vendored copy running outside of pip), `_longtracer_version()` returns
    None rather than raising."""
    import longtracer.adapters.evalport as evalport_module

    def _raise(name):
        raise importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(evalport_module._importlib_metadata, "version", _raise)
    vr = _all_supported_result()
    rs = to_openeval(vr)
    assert rs["runner"]["version"] is None


# --- export surface ---------------------------------------------------------


def test_lazy_export_from_adapters_package():
    """`to_openeval` must also be reachable via the lazy `longtracer.adapters`
    re-export, matching every other adapter in this package (see
    longtracer/adapters/__init__.py)."""
    from longtracer.adapters import to_openeval as lazy_to_openeval

    assert lazy_to_openeval is to_openeval


def test_lazy_export_from_top_level_package():
    """`to_openeval` is also reachable as `from longtracer import
    to_openeval`, matching the module's own documented usage example."""
    import longtracer

    vr = _all_supported_result()
    rs = longtracer.to_openeval(vr, run_id="top_level_run")
    assert rs["run_id"] == "top_level_run"


def test_openeval_import_failure_falls_back_to_default_version():
    """When `openeval` (evalport-sdk) genuinely isn't importable,
    `longtracer.adapters.evalport` still imports cleanly and its
    `OPENEVAL_VERSION` constant falls back to a fixed, valid semver string
    -- this is the exact contract documented in the module's own docstring
    ("no hard dependency on EvalPort").

    Run in a subprocess with `openeval` blocked via a `sys.meta_path`
    finder, so this can't disturb -- or be masked by -- this test session's
    own already-imported `openeval` package (installed here whenever the
    `evalport` extra is present, e.g. local dev with `pip install -e
    ".[all]"`)."""
    import subprocess

    code = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'openeval' or name.startswith('openeval.') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError(f'blocked for test: {name}')\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "from longtracer.adapters.evalport import OPENEVAL_VERSION\n"
        "assert OPENEVAL_VERSION == '1.0.0', OPENEVAL_VERSION\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
