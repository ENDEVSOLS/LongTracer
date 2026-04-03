"""
LongTracer SDK — One-liner RAG verification.

Usage:
    from longtracer import LongTracer, CitationVerifier
    LongTracer.init()

    # Quick check (no setup needed):
    from longtracer import check
    result = check("LLM said this", ["source text"])

    # Or with framework adapters:
    from longtracer import instrument_langchain, instrument_llamaindex
"""

from longtracer.core import LongTracer
from longtracer.guard.verifier import CitationVerifier, VerificationResult


def check(
    response: str,
    sources: list[str],
    source_metadata: list[dict] | None = None,
    threshold: float = 0.5,
) -> VerificationResult:
    """One-liner hallucination check — no class instantiation needed.

    Args:
        response: LLM-generated response to verify.
        sources: Source document chunks to verify against.
        source_metadata: Optional metadata for each source.
        threshold: Verification threshold (default 0.5).

    Returns:
        VerificationResult with trust_score, verdict, claims, etc.
    """
    verifier = CitationVerifier(threshold=threshold)
    return verifier.verify_parallel(response, sources, source_metadata)


def instrument_langchain(chain, verbose=None):
    """Lazy-loaded LangChain adapter."""
    from longtracer.adapters.langchain_handler import instrument_langchain as _impl
    return _impl(chain, verbose=verbose)


def instrument_llamaindex(query_engine, verbose=None):
    """Lazy-loaded LlamaIndex adapter."""
    from longtracer.adapters.llamaindex_handler import instrument_llamaindex as _impl
    return _impl(query_engine, verbose=verbose)


def instrument_haystack(pipeline, verbose=None):
    """Lazy-loaded Haystack adapter."""
    from longtracer.adapters.haystack_handler import instrument_haystack as _impl
    return _impl(pipeline, verbose=verbose)


# Backward compatibility
CitationGuard = LongTracer

__all__ = [
    "LongTracer",
    "CitationGuard",  # backward compat
    "CitationVerifier",
    "VerificationResult",
    "check",
    "instrument_langchain",
    "instrument_llamaindex",
    "instrument_haystack",
]
