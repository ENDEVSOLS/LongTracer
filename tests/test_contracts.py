"""
Unit tests for LongTracer versioned contracts and result semantics.

Covers:
  - SourceEvidence contract, automatic text hashing, and URI metadata
  - Multi-layer result states: ExecutionStatus, AssessmentAvailability,
    ClaimAssessment, and QualityGate enums
  - ClaimResult and CaseResult models, serialization, and JSON schema export
  - ApplicationOutput and TestCase models
  - RunManifest, ReviewRecord, and BaselineRecord provenance models
  - LegacyVerificationAdapter bidirectional conversion and edge-case handling
"""

import json
import sys
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

# Offline mock fallback for heavy ML dependencies
for _mod in ("sentence_transformers", "transformers"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from longtracer.contracts import (
    ApplicationOutput,
    AssessmentAvailability,
    BaselineRecord,
    CaseResult,
    ClaimAssessment,
    ClaimResult,
    ExecutionStatus,
    LegacyVerificationAdapter,
    QualityGate,
    ReviewRecord,
    ReviewState,
    RunManifest,
    SourceEvidence,
    TestCase,
    compute_text_hash,
)
from longtracer.guard.verifier import VerificationResult


# ---------------------------------------------------------------------------
# SourceEvidence Tests
# ---------------------------------------------------------------------------

class TestSourceEvidence:
    """Tests for SourceEvidence and text hashing."""

    def test_auto_computes_sha256_text_hash(self):
        text = "Water freezes at 0 degrees Celsius."
        expected_hash = compute_text_hash(text)

        evidence = SourceEvidence(source_id="src_1", text=text)
        assert evidence.text_hash == expected_hash
        assert len(evidence.text_hash) == 64
        assert evidence.schema_version == "1"

    def test_preserves_explicit_text_hash(self):
        explicit = "a" * 64
        evidence = SourceEvidence(source_id="src_2", text="Some text", text_hash=explicit)
        assert evidence.text_hash == explicit

    def test_metadata_and_provenance_preservation(self):
        evidence = SourceEvidence(
            source_id="doc_chunk_42",
            text="Revenue increased by 14% in Q3.",
            document_id="annual_report_2025.pdf",
            page=14,
            section="Financial Highlights",
            uri="https://example.com/reports/2025.pdf",
            provenance={"retrieval_score": 0.89, "rank": 1},
            metadata={"department": "finance"},
        )
        data = evidence.model_dump()
        assert data["document_id"] == "annual_report_2025.pdf"
        assert data["page"] == 14
        assert data["uri"] == "https://example.com/reports/2025.pdf"
        assert data["provenance"]["rank"] == 1

        # JSON Roundtrip
        raw_json = evidence.model_dump_json()
        reconstructed = SourceEvidence.model_validate_json(raw_json)
        assert reconstructed == evidence


# ---------------------------------------------------------------------------
# Result Contracts & Enums Tests
# ---------------------------------------------------------------------------

class TestResultContracts:
    """Tests for ClaimResult, CaseResult, and state enums."""

    def test_enums_members(self):
        assert ExecutionStatus.SUCCESS.value == "SUCCESS"
        assert ExecutionStatus.ERROR.value == "ERROR"
        assert ExecutionStatus.TIMEOUT.value == "TIMEOUT"

        assert AssessmentAvailability.ASSESSED.value == "ASSESSED"
        assert AssessmentAvailability.NO_ASSESSABLE_CLAIMS.value == "NO_ASSESSABLE_CLAIMS"

        assert ClaimAssessment.SUPPORTED.value == "SUPPORTED"
        assert ClaimAssessment.CONTRADICTED.value == "CONTRADICTED"
        assert ClaimAssessment.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"
        assert ClaimAssessment.CONFLICTING_SOURCES.value == "CONFLICTING_SOURCES"

        assert QualityGate.PASS.value == "PASS"
        assert QualityGate.FAIL.value == "FAIL"
        assert QualityGate.INDETERMINATE.value == "INDETERMINATE"

    def test_case_result_counts(self):
        claims = [
            ClaimResult(
                claim_id="c1",
                claim_text="Claim one",
                assessment=ClaimAssessment.SUPPORTED,
                supporting_sources=["src_1"],
            ),
            ClaimResult(
                claim_id="c2",
                claim_text="Claim two",
                assessment=ClaimAssessment.CONTRADICTED,
                contradicting_sources=["src_2"],
            ),
            ClaimResult(
                claim_id="c3",
                claim_text="Claim three",
                assessment=ClaimAssessment.INSUFFICIENT_EVIDENCE,
            ),
        ]
        result = CaseResult(
            case_id="test_case_1",
            execution=ExecutionStatus.SUCCESS,
            availability=AssessmentAvailability.ASSESSED,
            quality_gate=QualityGate.FAIL,
            claims=claims,
            trust_score=0.33,
        )

        assert result.supported_claims_count == 1
        assert result.contradicted_claims_count == 1
        assert result.ungrounded_claims_count == 2
        assert result.schema_version == "1"

    def test_json_schema_exportable(self):
        schema = CaseResult.model_json_schema()
        assert "properties" in schema
        assert "schema_version" in schema["properties"]
        assert "execution" in schema["properties"]
        assert "quality_gate" in schema["properties"]


# ---------------------------------------------------------------------------
# TestCase & ApplicationOutput Tests
# ---------------------------------------------------------------------------

class TestCaseContracts:
    """Tests for TestCase and ApplicationOutput."""

    def test_application_output_serialization(self):
        ev = SourceEvidence(source_id="s1", text="Evidence text")
        out = ApplicationOutput(
            response="The policy covers dental care.",
            sources=[ev],
            latency_ms=124.5,
        )
        assert out.response_type == "text"
        assert len(out.sources) == 1

        json_str = out.model_dump_json()
        parsed = ApplicationOutput.model_validate_json(json_str)
        assert parsed.response == out.response
        assert parsed.sources[0].source_id == "s1"

    def test_test_case_saved_mode(self):
        ev = SourceEvidence(source_id="s1", text="Company founded in 2020.")
        tc = TestCase(
            case_id="founding-date-01",
            mode="saved",
            question="When was the company founded?",
            saved_answer="In 2020.",
            saved_evidence=[ev],
            assertions={"forbidden_claims": ["Founded in 2018"]},
            tags=["history", "baseline"],
        )
        assert tc.schema_version == "1"
        assert tc.case_revision == 1
        assert tc.mode == "saved"
        assert tc.saved_evidence[0].source_id == "s1"


# ---------------------------------------------------------------------------
# Provenance & Review Records Tests
# ---------------------------------------------------------------------------

class TestProvenanceAndReviewContracts:
    """Tests for RunManifest, ReviewRecord, and BaselineRecord."""

    def test_run_manifest_serialization(self):
        manifest = RunManifest(
            run_id="run_20260928_001",
            dataset_digest="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            evaluator_fingerprint="sts:all-MiniLM-L6-v2+nli:deberta-v3-xsmall",
            policy_fingerprint="strict_no_hallucination_v1",
            created_at="2026-09-28T10:00:00Z",
            status_counts={"PASS": 95, "FAIL": 5},
        )
        assert manifest.schema_version == "1"
        assert manifest.repetition_count == 1
        data = manifest.model_dump()
        assert data["run_id"] == "run_20260928_001"

    def test_review_and_baseline_records(self):
        rev = ReviewRecord(
            review_id="rev_1",
            trace_id="tr_99",
            case_id="case_12",
            state=ReviewState.CONFIRMED_BUG,
            reviewer="mudassir",
            notes="LLM generated hallucinated refund window.",
        )
        assert rev.state == ReviewState.CONFIRMED_BUG

        base = BaselineRecord(
            baseline_id="base_v1",
            run_id="run_20260928_001",
            dataset_digest="e3b0c442...",
            evaluator_fingerprint="fp1",
            policy_fingerprint="pol1",
            approver="mudassir",
            approved_at="2026-09-28T10:30:00Z",
            reason="Validated golden run for release 0.3.0.",
        )
        assert base.approver == "mudassir"
        assert base.schema_version == "1"


# ---------------------------------------------------------------------------
# LegacyVerificationAdapter Tests
# ---------------------------------------------------------------------------

class TestLegacyVerificationAdapter:
    """Tests for lossless bidirectional mapping between legacy and modern results."""

    def test_from_legacy_all_supported(self):
        legacy = VerificationResult(
            trust_score=1.0,
            claims=[
                {"claim": "The sky is blue.", "confidence": 0.95, "sources": ["Atmosphere source"]},
            ],
            flagged_claims=[],
            hallucinations=[],
            all_supported=True,
            hallucination_count=0,
            summary="All 1 claim(s) supported.",
            latency_stats={"total_ms": 45.2},
        )

        case_res = LegacyVerificationAdapter.from_legacy(legacy, case_id="c_sky")
        assert case_res.case_id == "c_sky"
        assert case_res.quality_gate == QualityGate.PASS
        assert case_res.availability == AssessmentAvailability.ASSESSED
        assert len(case_res.claims) == 1
        assert case_res.claims[0].assessment == ClaimAssessment.SUPPORTED
        assert case_res.claims[0].supporting_sources == ["Atmosphere source"]
        assert case_res.trust_score == 1.0
        assert case_res.latency_ms == 45.2

    def test_from_legacy_with_hallucination(self):
        legacy = VerificationResult(
            trust_score=0.0,
            claims=[
                {"claim": "Paris is in Germany.", "confidence": 0.1, "sources": ["Geography doc"]},
            ],
            flagged_claims=[{"claim": "Paris is in Germany."}],
            hallucinations=[{"claim": "Paris is in Germany."}],
            all_supported=False,
            hallucination_count=1,
            summary="0/1 claims supported, 1 hallucination(s) detected.",
        )

        case_res = LegacyVerificationAdapter.from_legacy(legacy)
        assert case_res.quality_gate == QualityGate.FAIL
        assert len(case_res.claims) == 1
        assert case_res.claims[0].assessment == ClaimAssessment.CONTRADICTED

    def test_from_legacy_empty_claims(self):
        legacy = VerificationResult(
            trust_score=1.0,
            claims=[],
            flagged_claims=[],
            hallucinations=[],
            all_supported=True,
            hallucination_count=0,
        )

        case_res = LegacyVerificationAdapter.from_legacy(legacy)
        assert case_res.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS
        assert len(case_res.claims) == 0

    def test_roundtrip_to_legacy(self):
        case_res = CaseResult(
            case_id="case_100",
            execution=ExecutionStatus.SUCCESS,
            availability=AssessmentAvailability.ASSESSED,
            quality_gate=QualityGate.FAIL,
            claims=[
                ClaimResult(
                    claim_id="c1",
                    claim_text="Earth is flat.",
                    assessment=ClaimAssessment.CONTRADICTED,
                    contradicting_sources=["NASA doc"],
                    confidence=0.05,
                )
            ],
            trust_score=0.0,
            summary="Contradiction detected.",
            latency_ms=62.0,
        )

        legacy = LegacyVerificationAdapter.to_legacy(case_res)
        assert isinstance(legacy, VerificationResult)
        assert legacy.trust_score == 0.0
        assert legacy.verdict == "FAIL"
        assert legacy.hallucination_count == 1
        assert len(legacy.hallucinations) == 1
        assert legacy.hallucinations[0]["claim"] == "Earth is flat."
        assert legacy.latency_stats == {"total_ms": 62.0}
