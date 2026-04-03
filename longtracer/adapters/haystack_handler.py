"""
Haystack Integration for LongTracer.

Provides a Haystack v2 ``@component`` that verifies LLM responses
against retrieved documents. Drop it into any Haystack pipeline.

Usage:
    from longtracer import instrument_haystack

    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("llm", generator)
    instrument_haystack(pipeline)  # adds verifier component

    # Or use the component directly:
    from longtracer.adapters.haystack_handler import LongTracerVerifier
    pipeline.add_component("verifier", LongTracerVerifier())
"""

import logging
from typing import Dict, List, Optional

try:
    from haystack import component, Document
    _HAYSTACK_AVAILABLE = True
except ImportError:
    _HAYSTACK_AVAILABLE = False

from longtracer.core import LongTracer

logger = logging.getLogger("longtracer")


def _check_haystack():
    if not _HAYSTACK_AVAILABLE:
        raise ImportError(
            "Haystack is not installed. "
            "Install with: pip install 'longtracer[haystack]'"
        )


if _HAYSTACK_AVAILABLE:

    @component
    class LongTracerVerifier:
        """
        Haystack v2 component that verifies LLM responses against
        source documents using LongTracer's hybrid STS + NLI pipeline.

        Inputs:
            response: The LLM-generated text to verify.
            documents: Retrieved documents used as sources.

        Outputs:
            response: The original response (pass-through).
            trust_score: Float 0.0–1.0.
            verdict: "PASS" or "FAIL".
            summary: Human-readable summary string.
            claims: List of per-claim verification dicts.
            hallucination_count: Number of hallucinated claims.
        """

        def __init__(
            self,
            threshold: float = 0.5,
            verbose: bool = False,
        ):
            self.threshold = threshold
            self.verbose = verbose
            self._verifier = None

        def warm_up(self):
            """Load models. Called by Haystack before first run."""
            from longtracer.guard.verifier import CitationVerifier
            tracer = None
            if LongTracer.is_enabled():
                tracer = LongTracer.get_tracer()
            self._verifier = CitationVerifier(
                threshold=self.threshold, tracer=tracer
            )

        @component.output_types(
            response=str,
            trust_score=float,
            verdict=str,
            summary=str,
            claims=list,
            hallucination_count=int,
        )
        def run(
            self,
            response: str,
            documents: List[Document],
        ) -> Dict:
            if self._verifier is None:
                self.warm_up()

            source_texts = []
            source_metadata = []
            for doc in documents:
                source_texts.append(doc.content or "")
                source_metadata.append(doc.meta or {})

            try:
                result = self._verifier.verify_parallel(
                    response, source_texts, source_metadata
                )
            except Exception as exc:
                logger.warning(
                    "LongTracerVerifier: verification failed (%s)", exc
                )
                return {
                    "response": response,
                    "trust_score": -1.0,
                    "verdict": "ERROR",
                    "summary": f"Verification failed: {exc}",
                    "claims": [],
                    "hallucination_count": 0,
                }

            return {
                "response": response,
                "trust_score": result.trust_score,
                "verdict": result.verdict,
                "summary": result.summary,
                "claims": result.claims,
                "hallucination_count": result.hallucination_count,
            }

else:
    # Stub when Haystack is not installed
    class LongTracerVerifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _check_haystack()


def instrument_haystack(pipeline, verbose: Optional[bool] = None) -> None:
    """Add LongTracerVerifier to an existing Haystack pipeline.

    Connects the verifier after the generator component. Expects the
    pipeline to have a component whose output includes ``replies``
    (standard Haystack generator output) and a retriever that outputs
    ``documents``.

    Args:
        pipeline: A Haystack v2 Pipeline instance.
        verbose: Override verbose setting.
    """
    _check_haystack()

    verifier = LongTracerVerifier(
        verbose=verbose if verbose is not None else LongTracer.is_verbose()
    )
    pipeline.add_component("longtracer_verifier", verifier)
    logger.info("LongTracer: Haystack verifier component added to pipeline")
