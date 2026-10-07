"""
Tests: Honest result semantics and CaseResult contracts for v0.3.0.

Guards the core v0.3.0 promise:
1. Empty and too-short answers cannot produce a passing QualityGate.
2. Clear reason codes distinguish empty answers, missing sources, and refusals.
3. Evaluator failures (model load error, timeout, crash) surface as INDETERMINATE quality gates.
4. Legacy fields and public APIs remain backward-compatible.
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch

from longtracer import check_case
from longtracer.errors import (
    EvaluatorError,
    ModelUnavailableError,
    EvaluationFailedError,
)
from longtracer.guard.verifier import CitationVerifier, VerificationResult
from longtracer.contracts.result import (
    CaseResult,
    ClaimResult,
    ExecutionStatus,
    AssessmentAvailability,
    ClaimAssessment,
    QualityGate,
    ReasonCode,
    LegacyVerificationAdapter,
    compute_quality_gate,
)


def _make_mock_model():
    model = MagicMock()
    model.get_latency_stats.return_value = {
        "sts_calls": 1,
        "sts_avg_ms": 10.0,
        "nli_calls": 1,
        "nli_avg_ms": 20.0,
        "nli_skipped": 0,
        "total_ms": 30.0,
    }
    model.reset_latency_log.return_value = None
    return model


def _make_claim_dict(
    text: str,
    supported: bool = True,
    contradiction_score: float = 0.0,
    entailment_score: float = 0.8,
    is_meta_statement: bool = False,
    is_hallucination: bool = False,
    nli_ran: bool = True,
    best_source: str = "source text here",
) -> dict:
    return {
        "claim": text,
        "supported": supported,
        "score": 0.85 if supported else 0.2,
        "best_score": 0.90 if supported else 0.25,
        "sentence_results": [],
        "contradiction_score": contradiction_score,
        "entailment_score": entailment_score,
        "nli_ran": nli_ran,
        "best_source": best_source,
        "best_source_index": 0,
        "best_source_metadata": {},
        "is_hallucination": is_hallucination,
        "is_meta_statement": is_meta_statement,
        "has_hallucination_pattern": False,
    }


@pytest.fixture
def mock_verifier():
    mock_model = _make_mock_model()
    with patch("longtracer.guard.verifier.get_shared_model", return_value=mock_model):
        v = CitationVerifier()
        v.model = mock_model
        yield v, mock_model


class TestEmptyAndShortResponses:
    """Headline v0.3.0 release gate: empty and too-short answers cannot pass."""

    def test_empty_string_cannot_pass_gate(self, mock_verifier):
        v, _ = mock_verifier
        case_res = v.verify_case("", sources=["valid source document"])
        assert case_res.quality_gate != QualityGate.PASS
        assert case_res.quality_gate == QualityGate.FAIL
        assert case_res.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS
        assert case_res.reason == ReasonCode.EMPTY_RESPONSE
        assert case_res.trust_score == 1.0  # Legacy field preserved
        assert case_res.schema_version == "1"

    def test_whitespace_string_cannot_pass_gate(self, mock_verifier):
        v, _ = mock_verifier
        case_res = v.verify_case("   \n\t  ", sources=["valid source document"])
        assert case_res.quality_gate == QualityGate.FAIL
        assert case_res.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS
        assert case_res.reason == ReasonCode.EMPTY_RESPONSE
        assert case_res.trust_score == 1.0

    def test_too_short_string_cannot_pass_gate(self, mock_verifier):
        v, _ = mock_verifier
        # Shorter than claim_splitter threshold (>15 chars per claim)
        case_res = v.verify_case("Short.", sources=["valid source document"])
        assert case_res.quality_gate == QualityGate.FAIL
        assert case_res.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS
        assert case_res.reason == ReasonCode.NO_EXTRACTABLE_CLAIMS
        assert case_res.trust_score == 1.0

    def test_distinct_reasons_empty_vs_short(self, mock_verifier):
        v, _ = mock_verifier
        empty_res = v.verify_case("", sources=["source"])
        short_res = v.verify_case("No.", sources=["source"])
        assert empty_res.reason == ReasonCode.EMPTY_RESPONSE
        assert short_res.reason == ReasonCode.NO_EXTRACTABLE_CLAIMS
        assert empty_res.reason != short_res.reason


class TestNoSourcesSupplied:
    """When no sources are provided, claims are INSUFFICIENT_EVIDENCE, never CONTRADICTED."""

    def test_no_sources_produces_insufficient_evidence(self, mock_verifier):
        v, _ = mock_verifier
        resp = "The Eiffel Tower is located in Paris and was completed in 1889."
        case_res = v.verify_case(resp, sources=[])

        assert case_res.quality_gate == QualityGate.FAIL
        assert case_res.reason == ReasonCode.NO_SOURCES_SUPPLIED
        assert len(case_res.claims) > 0
        for claim in case_res.claims:
            assert claim.assessment == ClaimAssessment.INSUFFICIENT_EVIDENCE
            assert claim.assessment != ClaimAssessment.CONTRADICTED
            assert claim.reason == ReasonCode.NO_SOURCES_SUPPLIED


class TestClaimAssessmentsAndReasons:
    """Assessments map to explicit typed states with explainable reason codes."""

    def test_supported_claim(self, mock_verifier):
        v, mock_model = mock_verifier
        resp = "The Eiffel Tower is located in Paris and was completed in 1889."
        mock_model.verify_claims_batch.return_value = [
            _make_claim_dict(
                "The Eiffel Tower is located in Paris and was completed in 1889.",
                supported=True,
                best_source="The Eiffel Tower is in Paris, built in 1889.",
            )
        ]

        case_res = v.verify_case(resp, sources=["Some history source"])
        assert case_res.quality_gate == QualityGate.PASS
        assert len(case_res.claims) == 1
        claim = case_res.claims[0]
        assert claim.assessment == ClaimAssessment.SUPPORTED
        assert claim.reason == ReasonCode.SUPPORTED_BY_EVIDENCE
        assert claim.supporting_sources == ["source_0"]  # source IDs, not source text
        assert claim.confidence == pytest.approx(0.85)
        assert claim.details and "sts_similarity" in claim.details

    def test_contradicted_claim(self, mock_verifier):
        v, mock_model = mock_verifier
        resp = "The Eiffel Tower is located in Berlin and was completed in 1999."
        mock_model.verify_claims_batch.return_value = [
            _make_claim_dict(
                "The Eiffel Tower is located in Berlin and was completed in 1999.",
                supported=False,
                contradiction_score=0.85,
                is_hallucination=True,
                best_source="The Eiffel Tower is in Paris, not Berlin.",
            )
        ]

        case_res = v.verify_case(resp, sources=["Some history source"])
        assert case_res.quality_gate == QualityGate.FAIL
        assert len(case_res.claims) == 1
        claim = case_res.claims[0]
        assert claim.assessment == ClaimAssessment.CONTRADICTED
        assert claim.reason == ReasonCode.CONTRADICTED_BY_EVIDENCE
        assert claim.contradicting_sources == ["source_0"]
        assert claim.confidence == pytest.approx(0.85)  # contradiction probability

    def test_refusal_honest_uncertainty_passes(self, mock_verifier):
        v, mock_model = mock_verifier
        resp = "The provided documents do not contain information regarding this topic."
        mock_model.verify_claims_batch.return_value = [
            _make_claim_dict(
                resp,
                supported=False,
                is_meta_statement=True,
                is_hallucination=False,
            )
        ]

        case_res = v.verify_case(resp, sources=["Some document"])
        assert case_res.quality_gate == QualityGate.PASS
        assert len(case_res.claims) == 1
        claim = case_res.claims[0]
        assert claim.availability == AssessmentAvailability.NOT_EVALUATED
        assert claim.reason == ReasonCode.HONEST_UNCERTAINTY
        assert case_res.reason == ReasonCode.HONEST_UNCERTAINTY_ONLY
        assert case_res.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS


class TestErrorAndTimeoutHandling:
    """Evaluator failures become ERROR/TIMEOUT + INDETERMINATE, never FAIL."""

    def test_model_unavailable_error(self, mock_verifier):
        v, mock_model = mock_verifier
        resp = "The Eiffel Tower is located in Paris and was completed in 1889."
        mock_model.verify_claims_batch.side_effect = ModelUnavailableError("Weights missing")

        case_res = v.verify_case(resp, sources=["source"])
        assert case_res.execution == ExecutionStatus.ERROR
        assert case_res.quality_gate == QualityGate.INDETERMINATE
        assert case_res.reason == ReasonCode.MODEL_UNAVAILABLE

    def test_evaluation_failed_error(self, mock_verifier):
        v, mock_model = mock_verifier
        resp = "The Eiffel Tower is located in Paris and was completed in 1889."
        mock_model.verify_claims_batch.side_effect = EvaluationFailedError("Out of memory")

        case_res = v.verify_case(resp, sources=["source"])
        assert case_res.execution == ExecutionStatus.ERROR
        assert case_res.quality_gate == QualityGate.INDETERMINATE
        assert case_res.reason == ReasonCode.EVALUATION_FAILED

    def test_check_case_constructor_failure(self):
        with patch("longtracer.guard.verifier.get_shared_model", side_effect=ModelUnavailableError("Not downloaded")):
            res = check_case(
                "The Eiffel Tower is located in Paris and was completed in 1889.",
                sources=["source"],
            )
            assert res.execution == ExecutionStatus.ERROR
            assert res.quality_gate == QualityGate.INDETERMINATE
            assert res.reason == ReasonCode.MODEL_UNAVAILABLE

    def test_verify_case_timeout(self, mock_verifier):
        v, mock_model = mock_verifier
        resp = "The Eiffel Tower is located in Paris and was completed in 1889."

        def slow_verify(*args, **kwargs):
            time.sleep(0.15)
            return []

        mock_model.verify_claims_batch.side_effect = slow_verify

        case_res = v.verify_case(resp, sources=["source"], timeout=0.03)
        assert case_res.execution == ExecutionStatus.TIMEOUT
        assert case_res.quality_gate == QualityGate.INDETERMINATE
        assert case_res.reason == ReasonCode.EVALUATION_TIMEOUT

    def test_invalid_input_type(self, mock_verifier):
        v, _ = mock_verifier
        # In verify_case: returns ERROR + INVALID_INPUT + INDETERMINATE
        case_res = v.verify_case(12345, sources=["source"])  # type: ignore[arg-type]
        assert case_res.execution == ExecutionStatus.ERROR
        assert case_res.quality_gate == QualityGate.INDETERMINATE
        assert case_res.reason == ReasonCode.INVALID_INPUT

        # In verify_parallel: still raises TypeError (legacy compatibility preserved)
        with pytest.raises(TypeError):
            v.verify_parallel(12345, sources=["source"])  # type: ignore[arg-type]


class TestComputeQualityGatePureFunction:
    """Quality gate is a decoupled pure policy function."""

    def test_success_all_supported_passes(self):
        case = CaseResult(
            execution=ExecutionStatus.SUCCESS,
            availability=AssessmentAvailability.ASSESSED,
            claims=[
                ClaimResult(
                    claim_id="c1",
                    claim_text="Valid claim",
                    assessment=ClaimAssessment.SUPPORTED,
                    availability=AssessmentAvailability.ASSESSED,
                )
            ],
        )
        assert compute_quality_gate(case) == QualityGate.PASS

    def test_success_with_contradiction_fails(self):
        case = CaseResult(
            execution=ExecutionStatus.SUCCESS,
            availability=AssessmentAvailability.ASSESSED,
            claims=[
                ClaimResult(
                    claim_id="c1",
                    claim_text="Contradicted claim",
                    assessment=ClaimAssessment.CONTRADICTED,
                    availability=AssessmentAvailability.ASSESSED,
                )
            ],
        )
        assert compute_quality_gate(case) == QualityGate.FAIL

    def test_success_with_insufficient_evidence_fails(self):
        case = CaseResult(
            execution=ExecutionStatus.SUCCESS,
            availability=AssessmentAvailability.ASSESSED,
            claims=[
                ClaimResult(
                    claim_id="c1",
                    claim_text="Ungrounded claim",
                    assessment=ClaimAssessment.INSUFFICIENT_EVIDENCE,
                    availability=AssessmentAvailability.ASSESSED,
                )
            ],
        )
        assert compute_quality_gate(case) == QualityGate.FAIL

    def test_empty_response_fails(self):
        case = CaseResult(
            execution=ExecutionStatus.SUCCESS,
            availability=AssessmentAvailability.NO_ASSESSABLE_CLAIMS,
            reason=ReasonCode.EMPTY_RESPONSE,
        )
        assert compute_quality_gate(case) == QualityGate.FAIL

    def test_refusal_only_passes(self):
        case = CaseResult(
            execution=ExecutionStatus.SUCCESS,
            availability=AssessmentAvailability.NO_ASSESSABLE_CLAIMS,
            reason=ReasonCode.HONEST_UNCERTAINTY_ONLY,
        )
        assert compute_quality_gate(case) == QualityGate.PASS

    def test_error_and_timeout_indeterminate(self):
        err_case = CaseResult(execution=ExecutionStatus.ERROR)
        timeout_case = CaseResult(execution=ExecutionStatus.TIMEOUT)
        assert compute_quality_gate(err_case) == QualityGate.INDETERMINATE
        assert compute_quality_gate(timeout_case) == QualityGate.INDETERMINATE


class TestRoundtripAndLegacyCompatibility:
    """Legacy results convert losslessly to CaseResult and back."""

    def test_roundtrip_preserves_verdict_and_trust_score(self):
        legacy = VerificationResult(
            trust_score=0.92,
            claims=[{"claim": "Paris is capital of France."}],
            flagged_claims=[],
            hallucinations=[],
            all_supported=True,
            hallucination_count=0,
            summary="All 1 claim(s) supported.",
        )
        case_res = LegacyVerificationAdapter.from_legacy(legacy)
        assert case_res.trust_score == 0.92
        assert case_res.quality_gate == QualityGate.PASS

        back_to_legacy = LegacyVerificationAdapter.to_legacy(case_res)
        assert back_to_legacy.trust_score == 0.92
        assert back_to_legacy.verdict == "PASS"
        assert back_to_legacy.all_supported is True


# ─────────────────────────────────────────────────────────────────────
# Additional coverage from the v0.3.0 requirements traceability audit
# ─────────────────────────────────────────────────────────────────────

import ast
import asyncio
import dataclasses
import inspect
from concurrent.futures import CancelledError as FuturesCancelledError
from pathlib import Path

import numpy as np
import torch

from longtracer.contracts.result import summarize_case, unassessed_case
from longtracer.errors import EvaluationTimeoutError, InvalidInputError

FACT = "The Eiffel Tower is located in Paris and was completed in 1889."


def _no_text_claim(text: str) -> dict:
    """Claim dict as produced by nli_model._empty_result (sources had no usable sentences)."""
    d = _make_claim_dict(text, supported=False, nli_ran=False, entailment_score=0.0, best_source="")
    d.update({"score": 0.0, "best_score": 0.0, "best_source_index": -1, "best_source_metadata": None})
    return d


class TestReleaseRegressionGuard:
    """The headline release gate, on every public entry point of the new layer."""

    @pytest.mark.parametrize("response", ["", "   \n\t  ", "Yes, it does."])
    def test_verify_case(self, mock_verifier, response):
        v, _ = mock_verifier
        res = v.verify_case(response, sources=["some source"])
        assert res.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS
        assert res.quality_gate != QualityGate.PASS
        assert res.reason in (ReasonCode.EMPTY_RESPONSE, ReasonCode.NO_EXTRACTABLE_CLAIMS)
        # Legacy layer keeps its documented meaning at the same time
        assert v.verify_parallel(response, sources=["some source"]).trust_score == 1.0

    @pytest.mark.parametrize("response", ["", "   ", "Yes, it does."])
    def test_verify_case_async(self, mock_verifier, response):
        v, _ = mock_verifier
        res = asyncio.run(v.verify_case_async(response, sources=["some source"]))
        assert res.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS
        assert res.quality_gate != QualityGate.PASS

    @pytest.mark.parametrize("response", ["", "  ", "Yes, it does."])
    def test_check_case(self, response):
        with patch("longtracer.guard.verifier.get_shared_model", return_value=_make_mock_model()):
            res = check_case(response, ["some source"])
        assert res.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS
        assert res.quality_gate != QualityGate.PASS

    def test_empty_and_short_summaries_differ(self, mock_verifier):
        v, _ = mock_verifier
        empty = v.verify_case("", sources=["s"])
        short = v.verify_case("Yes, it does.", sources=["s"])
        assert empty.availability == short.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS
        assert empty.summary != short.summary
        assert "empty" in empty.summary.lower() and "too short" in short.summary.lower()


class TestSilentSuccessFailureModes:
    """Handover §2.5: every listed failure mode becomes an explicit typed state."""

    def test_model_download_failure_via_check_case(self):
        """A network/hub failure while loading weights → ModelUnavailableError → ERROR."""
        from longtracer.guard import nli_model

        with (
            patch.object(nli_model, "_shared_model", None),
            patch(
                "longtracer.guard.nli_model.SentenceTransformer",
                side_effect=OSError("We couldn't connect to 'https://huggingface.co' to load this model"),
            ),
        ):
            res = check_case(FACT, ["source"])
        assert res.execution == ExecutionStatus.ERROR
        assert res.quality_gate == QualityGate.INDETERMINATE
        assert res.reason == ReasonCode.MODEL_UNAVAILABLE
        assert "models prepare" in (res.error_message or "")

    def test_timeout_returns_early(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.side_effect = lambda *a, **k: time.sleep(1.0) or []
        t0 = time.perf_counter()
        res = v.verify_case(FACT, sources=["source"], timeout=0.05)
        elapsed = time.perf_counter() - t0
        assert res.execution == ExecutionStatus.TIMEOUT
        assert res.quality_gate == QualityGate.INDETERMINATE
        assert elapsed < 0.6, f"verify_case waited {elapsed:.2f}s; the timeout did not return early"

    def test_timeout_async(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.side_effect = lambda *a, **k: time.sleep(1.0) or []
        res = asyncio.run(v.verify_case_async(FACT, sources=["source"], timeout=0.05))
        assert res.execution == ExecutionStatus.TIMEOUT
        assert res.reason == ReasonCode.EVALUATION_TIMEOUT

    def test_engine_raises_typed_timeout_error(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.side_effect = lambda *a, **k: time.sleep(1.0) or []
        with pytest.raises(EvaluationTimeoutError) as exc_info:
            v._run_legacy_engine(FACT, ["source"], None, 0.05)
        assert exc_info.value.timeout_s == 0.05
        assert isinstance(exc_info.value, TimeoutError)

    @pytest.mark.parametrize("bad", [0, -1, "5", True])
    def test_invalid_timeout_raises_invalid_input_error(self, mock_verifier, bad):
        v, _ = mock_verifier
        with pytest.raises(InvalidInputError):
            v.verify_case(FACT, sources=["source"], timeout=bad)

    def test_cancelled_run(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.side_effect = FuturesCancelledError()
        res = v.verify_case(FACT, sources=["source"])
        assert res.execution == ExecutionStatus.CANCELLED
        assert res.quality_gate == QualityGate.INDETERMINATE
        assert res.reason == ReasonCode.EXECUTION_CANCELLED

    def test_unknown_engine_exception_is_error_not_success(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.side_effect = ValueError("tokenizer blew up")
        res = v.verify_case(FACT, sources=["source"])
        assert res.execution == ExecutionStatus.ERROR
        assert res.quality_gate == QualityGate.INDETERMINATE
        assert "ValueError" in (res.error_message or "")

    @pytest.mark.parametrize(
        "side_effect",
        [ModelUnavailableError("x"), EvaluationFailedError("x"), FuturesCancelledError(), RuntimeError("x")],
    )
    def test_incomplete_evaluation_reports_no_partial_claims(self, mock_verifier, side_effect):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.side_effect = side_effect
        res = v.verify_case(FACT, sources=["source"])
        assert res.availability == AssessmentAvailability.NOT_EVALUATED
        assert res.claims == []
        assert res.quality_gate == QualityGate.INDETERMINATE

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"response": FACT, "sources": "not a list"},
            {"response": FACT, "sources": [123]},
            {"response": FACT, "sources": ["ok"], "source_metadata": "nope"},
            {"response": None, "sources": ["ok"]},
        ],
    )
    def test_malformed_or_unsupported_input(self, mock_verifier, kwargs):
        v, _ = mock_verifier
        res = v.verify_case(**kwargs)
        assert res.execution == ExecutionStatus.ERROR
        assert res.reason == ReasonCode.INVALID_INPUT
        assert res.quality_gate == QualityGate.INDETERMINATE
        with pytest.raises(TypeError):  # legacy public behaviour is unchanged
            v.verify_parallel(**kwargs)

    def test_no_sources_never_contradicted(self, mock_verifier):
        v, _ = mock_verifier
        res = v.verify_case(FACT, sources=[])
        assert res.reason == ReasonCode.NO_SOURCES_SUPPLIED
        assert res.claims and all(c.assessment == ClaimAssessment.INSUFFICIENT_EVIDENCE for c in res.claims)
        assert all(c.reason == ReasonCode.NO_SOURCES_SUPPLIED for c in res.claims)
        assert res.quality_gate == QualityGate.FAIL
        assert "No sources were supplied" in res.summary

    def test_sources_without_usable_text(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.return_value = [_no_text_claim(FACT)]
        res = v.verify_case(FACT, sources=[""])
        assert res.claims[0].assessment == ClaimAssessment.INSUFFICIENT_EVIDENCE
        assert res.claims[0].reason == ReasonCode.NO_SOURCE_TEXT
        assert res.quality_gate == QualityGate.FAIL


class TestClaimMappingDetails:
    def test_pattern_hallucination_is_insufficient_not_contradicted(self, mock_verifier):
        """'Based on my knowledge…' with low similarity: nothing contradicted it."""
        v, mock_model = mock_verifier
        claim = _make_claim_dict(FACT, supported=False, nli_ran=False, is_hallucination=True, entailment_score=0.0)
        claim["has_hallucination_pattern"] = True
        mock_model.verify_claims_batch.return_value = [claim]
        res = v.verify_case(FACT, sources=["source"])
        assert res.claims[0].assessment == ClaimAssessment.INSUFFICIENT_EVIDENCE
        assert res.claims[0].reason == ReasonCode.LOW_EVIDENCE_SIMILARITY
        assert res.quality_gate == QualityGate.FAIL
        assert res.metadata["legacy_verdict"] == "FAIL"

    def test_not_entailed_when_nli_ran(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.return_value = [
            _make_claim_dict(FACT, supported=False, nli_ran=True, contradiction_score=0.2, entailment_score=0.1)
        ]
        res = v.verify_case(FACT, sources=["source"])
        assert res.claims[0].reason == ReasonCode.NOT_ENTAILED

    def test_mixed_refusal_and_supported_claim(self, mock_verifier):
        v, mock_model = mock_verifier
        refusal = "The provided documents do not contain information about the opening hours."
        resp = f"{FACT} {refusal}"
        mock_model.verify_claims_batch.return_value = [
            _make_claim_dict(FACT, supported=True),
            _make_claim_dict(refusal, supported=False, is_meta_statement=True),
        ]
        res = v.verify_case(resp, sources=["source"])
        assert res.availability == AssessmentAvailability.ASSESSED
        assert res.claims[1].availability == AssessmentAvailability.NOT_EVALUATED
        assert res.quality_gate == QualityGate.PASS
        assert "1 not evaluated" in res.summary

    def test_source_ids_from_metadata(self, mock_verifier):
        v, mock_model = mock_verifier
        claim = _make_claim_dict(FACT, supported=True)
        claim["best_source_metadata"] = {"source_id": "doc-42"}
        mock_model.verify_claims_batch.return_value = [claim]
        res = v.verify_case(FACT, sources=["source"], source_metadata=[{"source_id": "doc-42"}])
        assert res.claims[0].supporting_sources == ["doc-42"]

    def test_char_offsets_attached(self, mock_verifier):
        v, mock_model = mock_verifier
        resp = f"Intro sentence that is long enough. {FACT}"
        mock_model.verify_claims_batch.return_value = [
            _make_claim_dict("Intro sentence that is long enough."),
            _make_claim_dict(FACT),
        ]
        res = v.verify_case(resp, sources=["source"])
        c = res.claims[1]
        assert resp[c.char_start : c.char_end] == FACT

    def test_metadata_records_legacy_verdict_and_conflict_status(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.return_value = [_make_claim_dict(FACT)]
        res = v.verify_case(FACT, sources=["source"])
        assert res.metadata == {"legacy_verdict": "PASS", "conflict_detection": "disabled"}


class TestSchemaVersionOnEveryRecord:
    def test_all_emitted_paths(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.return_value = [_make_claim_dict(FACT)]
        results = [
            v.verify_case(FACT, sources=["s"]),
            v.verify_case("", sources=["s"]),
            v.verify_case("Yes, it does.", sources=["s"]),
            v.verify_case(FACT, sources=[]),
            v.verify_case(123, sources=["s"]),  # type: ignore[arg-type]
        ]
        mock_model.verify_claims_batch.side_effect = EvaluationFailedError("x")
        results.append(v.verify_case(FACT, sources=["s"]))
        results.append(unassessed_case(ExecutionStatus.TIMEOUT, ReasonCode.EVALUATION_TIMEOUT))
        for r in results:
            assert r.schema_version == "1"
            assert r.model_dump(mode="json")["schema_version"] == "1"


class TestGatePolicyEdges:
    def test_assessed_with_zero_assessed_claims_fails(self):
        case = CaseResult(availability=AssessmentAvailability.ASSESSED, claims=[])
        assert compute_quality_gate(case) == QualityGate.FAIL

    def test_no_assessable_claims_without_reason_fails(self):
        case = CaseResult(availability=AssessmentAvailability.NO_ASSESSABLE_CLAIMS, reason=None)
        assert compute_quality_gate(case) == QualityGate.FAIL

    @pytest.mark.parametrize("status", [ExecutionStatus.CANCELLED, ExecutionStatus.SKIPPED, ExecutionStatus.ERROR])
    def test_non_success_is_indeterminate_even_with_supported_claims(self, status):
        case = CaseResult(
            execution=status,
            claims=[ClaimResult(claim_id="c", claim_text="t", assessment=ClaimAssessment.SUPPORTED)],
        )
        assert compute_quality_gate(case) == QualityGate.INDETERMINATE

    def test_conflicting_sources_fails_gate(self):
        case = CaseResult(
            claims=[ClaimResult(claim_id="c", claim_text="t", assessment=ClaimAssessment.CONFLICTING_SOURCES)]
        )
        assert compute_quality_gate(case) == QualityGate.FAIL

    def test_gate_does_not_mutate_claims(self):
        case = CaseResult(claims=[ClaimResult(claim_id="c", claim_text="t", assessment=ClaimAssessment.CONTRADICTED)])
        before = case.model_dump()
        compute_quality_gate(case)
        assert case.model_dump() == before


class TestRoundTripScenarios:
    """legacy → CaseResult → legacy preserves verdict and trust_score."""

    @pytest.mark.parametrize(
        "claims, flagged, hallucinations, trust",
        [
            ([_make_claim_dict(FACT)], [], [], 0.85),
            (
                [_make_claim_dict(FACT, supported=False, contradiction_score=0.9, is_hallucination=True)],
                "all",
                "all",
                0.93,
            ),
            (
                [_make_claim_dict(FACT), _make_claim_dict("Second claim that is not supported.", supported=False)],
                [1],
                [],
                0.52,
            ),
            ([], [], [], 1.0),
            ([_make_claim_dict(FACT, supported=False, is_meta_statement=True)], "all", [], 0.21),
            ([{"claim": "Plain synthetic dict claim."}], [], [], 0.7),
        ],
    )
    def test_roundtrip(self, claims, flagged, hallucinations, trust):
        def pick(sel):
            return list(claims) if sel == "all" else [claims[i] for i in sel]

        legacy = VerificationResult(
            trust_score=trust,
            claims=claims,
            flagged_claims=pick(flagged),
            hallucinations=pick(hallucinations),
            all_supported=not pick(flagged),
            hallucination_count=len(pick(hallucinations)),
        )
        back = LegacyVerificationAdapter.to_legacy(LegacyVerificationAdapter.from_legacy(legacy))
        assert back.verdict == legacy.verdict
        assert back.trust_score == legacy.trust_score


class TestConflictDetection:
    """CONFLICTING_SOURCES (opt-in) with a fake model; real-model behaviour is in the smoke test."""

    def _fake_model(self, logits, id2label=None):
        m = _make_mock_model()
        m.extract_source_sentences.side_effect = lambda s: [s] if s else []
        m.sts_model.encode.side_effect = lambda sents, **k: torch.ones((len(sents), 4))
        m.nli_model.predict.return_value = np.array(logits, dtype=float)
        m.nli_model.model.config.id2label = (
            id2label if id2label is not None else {0: "contradiction", 1: "entailment", 2: "neutral"}
        )
        m.verify_claims_batch.return_value = [_make_claim_dict(FACT, supported=True)]
        return m

    def _verifier(self, model):
        with patch("longtracer.guard.verifier.get_shared_model", return_value=model):
            v = CitationVerifier()
        v.model = model
        return v

    def test_disagreeing_sources_flagged(self):
        model = self._fake_model([[-5, 5, -5], [5, -5, -5]])  # src0 entails, src1 contradicts
        v = self._verifier(model)
        res = v.verify_case(
            FACT,
            ["Completed in 1889.", "Completed in 1925, not 1889."],
            source_metadata=[{"id": "A"}, {"id": "B"}],
            detect_conflicts=True,
        )
        claim = res.claims[0]
        assert claim.assessment == ClaimAssessment.CONFLICTING_SOURCES
        assert claim.reason == ReasonCode.CONFLICTING_EVIDENCE
        assert claim.supporting_sources == ["A"] and claim.contradicting_sources == ["B"]
        assert res.quality_gate == QualityGate.FAIL
        assert res.metadata["conflict_detection"] == "enabled: 1 conflicting claim(s)"

    def test_label_mapping_read_from_model_config(self):
        # Same logits, swapped label order: now src0 contradicts and src1 entails.
        model = self._fake_model(
            [[-5, 5, -5], [5, -5, -5]], id2label={0: "entailment", 1: "contradiction", 2: "neutral"}
        )
        res = self._verifier(model).verify_case(FACT, ["a source text", "b source text"], detect_conflicts=True)
        c = res.claims[0]
        assert c.supporting_sources == ["source_1"] and c.contradicting_sources == ["source_0"]

    def test_agreeing_sources_not_flagged(self):
        model = self._fake_model([[-5, 5, -5], [-5, 5, -5]])
        res = self._verifier(model).verify_case(FACT, ["a source text", "b source text"], detect_conflicts=True)
        assert res.claims[0].assessment == ClaimAssessment.SUPPORTED
        assert res.quality_gate == QualityGate.PASS

    def test_unknown_label_mapping_reported_not_guessed(self):
        model = self._fake_model([[-5, 5, -5], [5, -5, -5]], id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        res = self._verifier(model).verify_case(FACT, ["a source text", "b source text"], detect_conflicts=True)
        assert res.claims[0].assessment == ClaimAssessment.SUPPORTED
        assert res.metadata["conflict_detection"].startswith("unavailable")

    def test_conflict_detection_failure_is_error_not_silent(self):
        model = self._fake_model([[0, 0, 0]])
        model.nli_model.predict.side_effect = RuntimeError("oom")
        res = self._verifier(model).verify_case(FACT, ["a source text", "b source text"], detect_conflicts=True)
        assert res.execution == ExecutionStatus.ERROR
        assert res.quality_gate == QualityGate.INDETERMINATE

    def test_off_by_default(self):
        model = self._fake_model([[-5, 5, -5], [5, -5, -5]])
        res = self._verifier(model).verify_case(FACT, ["a source text", "b source text"])
        assert res.claims[0].assessment == ClaimAssessment.SUPPORTED
        model.nli_model.predict.assert_not_called()


class TestArchitectureAndLegacyIsolation:
    def test_contracts_never_import_guard_at_module_level(self):
        """Handover §2.4: longtracer/contracts must not import longtracer.guard at module level."""
        contracts_dir = Path(__file__).resolve().parent.parent / "longtracer" / "contracts"
        for py in sorted(contracts_dir.glob("*.py")):
            tree = ast.parse(py.read_text())
            for node in tree.body:  # module level only; lazy imports inside functions are allowed
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("longtracer.guard"):
                    pytest.fail(f"{py.name} imports {node.module} at module level")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("longtracer.guard"), f"{py.name} imports {alias.name}"

    def test_thresholds_unchanged(self):
        """C.2: thresholds are not tuned in v0.3.0."""
        from longtracer.guard import nli_model

        sig = inspect.signature(nli_model.HybridVerificationModel.__init__)
        assert sig.parameters["support_threshold"].default == 0.40
        src = inspect.getsource(nli_model)
        assert src.count("avg_score >= 0.25") == 2  # NLI gate in verify_claim + verify_claims_batch
        assert src.count("max_contradiction > 0.5") >= 4

    def test_verify_case_does_not_change_legacy_output(self, mock_verifier):
        v, mock_model = mock_verifier
        mock_model.verify_claims_batch.return_value = [_make_claim_dict(FACT)]
        before = dataclasses.asdict(v.verify_parallel(FACT, ["source"]))
        v.verify_case(FACT, ["source"], timeout=1.0)
        after = dataclasses.asdict(v.verify_parallel(FACT, ["source"]))
        assert before == after

    def test_summarize_case_reason_texts(self):
        assert "empty" in summarize_case([], ReasonCode.EMPTY_RESPONSE).lower()
        assert (
            "refusal"
            in summarize_case(
                [
                    ClaimResult(
                        claim_id="c",
                        claim_text="t",
                        assessment=ClaimAssessment.INSUFFICIENT_EVIDENCE,
                        availability=AssessmentAvailability.NOT_EVALUATED,
                    )
                ],
                ReasonCode.HONEST_UNCERTAINTY_ONLY,
            ).lower()
        )
