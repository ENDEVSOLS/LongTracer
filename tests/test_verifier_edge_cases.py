"""
Tests: CitationVerifier edge cases — empty inputs, no sources, result structure.

These tests mock HybridVerificationModel to avoid loading ML models in CI.
"""

import pytest
from unittest.mock import MagicMock, patch
from longtracer.guard.verifier import CitationVerifier, VerificationResult


def _make_mock_model():
    """Create a mock HybridVerificationModel that returns predictable results."""
    model = MagicMock()
    model.get_latency_stats.return_value = {
        "sts_calls": 1, "sts_avg_ms": 50.0,
        "nli_calls": 1, "nli_avg_ms": 100.0,
        "nli_skipped": 0, "total_ms": 150.0,
    }
    model.reset_latency_log.return_value = None
    return model


def _make_supported_claim(text: str) -> dict:
    return {
        "claim": text, "supported": True, "score": 0.75,
        "best_score": 0.80, "sentence_results": [],
        "contradiction_score": 0.05, "entailment_score": 0.85,
        "nli_ran": True, "best_source": "source text...",
        "best_source_index": 0, "best_source_metadata": {},
        "is_hallucination": False, "is_meta_statement": False,
        "has_hallucination_pattern": False,
    }


def _make_unsupported_claim(text: str) -> dict:
    return {
        "claim": text, "supported": False, "score": 0.10,
        "best_score": 0.15, "sentence_results": [],
        "contradiction_score": 0.70, "entailment_score": 0.05,
        "nli_ran": True, "best_source": "different source...",
        "best_source_index": 0, "best_source_metadata": {},
        "is_hallucination": True, "is_meta_statement": False,
        "has_hallucination_pattern": False,
    }


@pytest.fixture
def verifier():
    with patch("longtracer.guard.verifier.HybridVerificationModel") as MockModel:
        mock_instance = _make_mock_model()
        MockModel.return_value = mock_instance
        v = CitationVerifier()
        v.model = mock_instance
        yield v


class TestEmptyInputHandling:

    def test_empty_response_returns_vacuous_truth(self, verifier):
        result = verifier.verify_parallel("", sources=["some source"])
        assert result.trust_score == 1.0
        assert result.claims == []
        assert result.all_supported is True
        assert result.hallucination_count == 0

    def test_whitespace_only_response_returns_vacuous_truth(self, verifier):
        result = verifier.verify_parallel("   \n\t  ", sources=["some source"])
        assert result.trust_score == 1.0
        assert result.claims == []
        assert result.all_supported is True

    def test_empty_response_sequential_returns_vacuous_truth(self, verifier):
        result = verifier.verify("", sources=["some source"])
        assert result.trust_score == 1.0
        assert result.all_supported is True

    def test_empty_sources_marks_all_claims_unsupported(self, verifier):
        # Need a response long enough to generate claims (>500 chars)
        long_response = "The Eiffel Tower is in Paris. " * 20
        result = verifier.verify_parallel(long_response, sources=[])
        assert result.all_supported is False
        assert all(c["supported"] is False for c in result.claims)
        assert all(c["score"] == 0.0 for c in result.claims)

    def test_empty_sources_trust_score_is_zero(self, verifier):
        long_response = "The Eiffel Tower is in Paris. " * 20
        result = verifier.verify_parallel(long_response, sources=[])
        assert result.trust_score == 0.0

    def test_empty_sources_sequential_marks_unsupported(self, verifier):
        long_response = "The Eiffel Tower is in Paris. " * 20
        result = verifier.verify(long_response, sources=[])
        assert result.all_supported is False


class TestVerificationResultStructure:

    def test_result_is_dataclass(self, verifier):
        result = verifier.verify_parallel("", sources=[])
        assert isinstance(result, VerificationResult)

    def test_result_has_all_fields(self, verifier):
        result = verifier.verify_parallel("", sources=[])
        assert hasattr(result, "trust_score")
        assert hasattr(result, "claims")
        assert hasattr(result, "flagged_claims")
        assert hasattr(result, "hallucinations")
        assert hasattr(result, "all_supported")
        assert hasattr(result, "hallucination_count")
        assert hasattr(result, "latency_stats")

    def test_trust_score_range(self, verifier):
        verifier.model.verify_claims_batch.return_value = [
            _make_supported_claim("claim one"),
            _make_supported_claim("claim two"),
        ]
        long_response = "Claim one is true. " * 30
        result = verifier.verify_parallel(long_response, sources=["source"])
        assert 0.0 <= result.trust_score <= 1.0

    def test_all_supported_true_when_no_flagged(self, verifier):
        verifier.model.verify_claims_batch.return_value = [
            _make_supported_claim("claim one"),
        ]
        long_response = "Claim one is true. " * 30
        result = verifier.verify_parallel(long_response, sources=["source"])
        assert result.all_supported is True
        assert result.flagged_claims == []

    def test_all_supported_false_when_flagged(self, verifier):
        verifier.model.verify_claims_batch.return_value = [
            _make_unsupported_claim("fabricated claim"),
        ]
        long_response = "Fabricated claim is here. " * 30
        result = verifier.verify_parallel(long_response, sources=["source"])
        assert result.all_supported is False
        assert len(result.flagged_claims) == 1

    def test_hallucination_count_matches_hallucinations_list(self, verifier):
        verifier.model.verify_claims_batch.return_value = [
            _make_supported_claim("good claim"),
            _make_unsupported_claim("bad claim"),
        ]
        long_response = "Good claim is true. Bad claim is false. " * 15
        result = verifier.verify_parallel(long_response, sources=["source"])
        assert result.hallucination_count == len(result.hallucinations)


class TestVerifyWithRagResult:

    def test_verify_with_rag_result_returns_dict(self, verifier):
        rag_result = {
            "answer": "",
            "source_texts": [],
            "sources": [],
        }
        result = verifier.verify_with_rag_result(rag_result)
        assert isinstance(result, dict)
        assert "trust_score" in result
        assert "hallucination_count" in result
        assert "claims" in result

    def test_verify_with_rag_result_extracts_metadata(self, verifier):
        mock_source = MagicMock()
        mock_source.metadata = {"source": "doc.pdf", "page": 1}
        rag_result = {
            "answer": "",
            "source_texts": ["text"],
            "sources": [mock_source],
        }
        result = verifier.verify_with_rag_result(rag_result)
        assert isinstance(result, dict)

    def test_verify_with_rag_result_handles_missing_keys(self, verifier):
        result = verifier.verify_with_rag_result({})
        assert result["trust_score"] == 1.0  # empty answer → vacuous truth


class TestTracerIntegration:

    def test_tracer_receives_claim_evidence(self, verifier):
        from longtracer.guard.tracer import Tracer
        from longtracer.guard.cache.memory import MemoryBackend

        tracer = Tracer(project_name="test", backend=MemoryBackend())
        verifier.tracer = tracer

        verifier.model.verify_claims_batch.return_value = [
            _make_supported_claim("the claim"),
        ]
        long_response = "The claim is supported. " * 30
        verifier.verify_parallel(long_response, sources=["source text"])
        assert len(tracer.claim_evidence_map) > 0
