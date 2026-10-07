"""
Tests: typed evaluator errors (Workstream B).

These tests mock the underlying model classes so they run without network
access or real model weights, following the pattern in
tests/test_verifier_edge_cases.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from longtracer.errors import EvaluationFailedError, ModelUnavailableError


class TestModelLoadErrors:
    """Model-load failures raise a typed, actionable LongTracer error."""

    def test_sts_load_failure_raises_model_unavailable_error(self):
        with patch("longtracer.guard.nli_model.SentenceTransformer", side_effect=OSError("not found")):
            from longtracer.guard.nli_model import HybridVerificationModel

            with pytest.raises(ModelUnavailableError) as exc_info:
                HybridVerificationModel(verbose=False)

            assert "models prepare" in str(exc_info.value)
            assert exc_info.value.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert isinstance(exc_info.value.__cause__, OSError)

    def test_sts_load_failure_is_still_an_import_error(self):
        """Backward compatibility: existing `except ImportError` call sites keep working."""
        with patch("longtracer.guard.nli_model.SentenceTransformer", side_effect=OSError("not found")):
            from longtracer.guard.nli_model import HybridVerificationModel

            with pytest.raises(ImportError):
                HybridVerificationModel(verbose=False)

    def test_nli_load_failure_raises_model_unavailable_error(self):
        with (
            patch("longtracer.guard.nli_model.SentenceTransformer"),
            patch("longtracer.guard.nli_model.CrossEncoder", side_effect=OSError("not found")),
        ):
            from longtracer.guard.nli_model import HybridVerificationModel

            with pytest.raises(ModelUnavailableError) as exc_info:
                HybridVerificationModel(verbose=False)

            assert exc_info.value.model_name == "cross-encoder/nli-deberta-v3-xsmall"
            assert isinstance(exc_info.value.__cause__, OSError)

    def test_load_failure_preserves_original_exception_as_cause(self):
        original = RuntimeError("disk full")
        with patch("longtracer.guard.nli_model.SentenceTransformer", side_effect=original):
            from longtracer.guard.nli_model import HybridVerificationModel

            with pytest.raises(ModelUnavailableError) as exc_info:
                HybridVerificationModel(verbose=False)
            assert exc_info.value.__cause__ is original


class TestInferenceErrors:
    """Failures during scoring (not loading) raise EvaluationFailedError."""

    def _make_model_with_mocked_backends(self):
        with patch("longtracer.guard.nli_model.SentenceTransformer"), patch("longtracer.guard.nli_model.CrossEncoder"):
            from longtracer.guard.nli_model import HybridVerificationModel

            model = HybridVerificationModel(verbose=False, use_slm=False)
        return model

    def test_nli_predict_failure_raises_evaluation_failed_error(self):
        model = self._make_model_with_mocked_backends()
        model.nli_model.predict = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with pytest.raises(EvaluationFailedError) as exc_info:
            model.compute_nli_scores("some source sentence", "some claim")

        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_sts_encode_failure_raises_evaluation_failed_error(self):
        model = self._make_model_with_mocked_backends()
        model.sts_model.encode = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with pytest.raises(EvaluationFailedError):
            model.verify_claim("The sky is blue.", ["The sky appears blue during the day."])

    def test_evaluation_failed_error_does_not_escape_as_a_success(self):
        """A scoring failure must propagate as an error, never as a passing result."""
        model = self._make_model_with_mocked_backends()
        model.nli_model.predict = MagicMock(side_effect=RuntimeError("boom"))

        # The real STS model is loaded (SentenceTransformer/CrossEncoder classes are
        # mocked, but we still want a real embedding model to guarantee avg_score
        # clears the 0.25 NLI gate). Instead, force the gate deterministically by
        # stubbing sts_model.encode to return embeddings with cosine similarity 1.0.
        import torch

        def fake_encode(sentences, **kwargs):
            return torch.ones((len(sentences), 4))

        model.sts_model.encode = fake_encode

        with pytest.raises(EvaluationFailedError):
            model.verify_claim(
                "Water boils at 100 degrees Celsius.",
                ["Water boils at 100 degrees Celsius at sea level."],
            )
