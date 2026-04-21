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


def check_batch(
    items: list[dict],
    threshold: float = 0.5,
    max_workers: int = 4,
) -> list[VerificationResult]:
    """One-liner batch hallucination check — verify multiple responses at once.

    Args:
        items: List of dicts, each with "response" (str) and "sources" (list[str]).
        threshold: Verification threshold (default 0.5).
        max_workers: Max parallel workers (default 4).

    Returns:
        List of VerificationResult, one per item.

    Example::

        results = check_batch([
            {"response": "Paris is in France.", "sources": ["Paris is the capital of France."]},
            {"response": "Water boils at 50°C.", "sources": ["Water boils at 100°C."]},
        ])
    """
    verifier = CitationVerifier(threshold=threshold)
    return verifier.verify_batch(items, max_workers=max_workers)


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


def instrument_langgraph(graph, threshold=0.5, verbose=None):
    """Lazy-loaded LangGraph agent adapter."""
    from longtracer.adapters.langgraph_handler import instrument_langgraph as _impl
    return _impl(graph, threshold=threshold, verbose=verbose)


def instrument_langchain_agent(agent_executor, threshold=0.5, verbose=None):
    """Lazy-loaded LangChain AgentExecutor adapter."""
    from longtracer.adapters.langgraph_handler import instrument_langchain_agent as _impl
    return _impl(agent_executor, threshold=threshold, verbose=verbose)


def instrument_openai_assistant(client, threshold=0.5, verbose=None):
    """Lazy-loaded OpenAI Assistants API adapter."""
    from longtracer.adapters.openai_handler import instrument_openai_assistant as _impl
    return _impl(client, threshold=threshold, verbose=verbose)


def instrument_crewai(crew, threshold=0.5, verbose=None):
    """Lazy-loaded CrewAI adapter."""
    from longtracer.adapters.crewai_handler import instrument_crewai as _impl
    return _impl(crew, threshold=threshold, verbose=verbose)


def instrument_autogen(agent, threshold=0.5, verbose=None):
    """Lazy-loaded AutoGen adapter."""
    from longtracer.adapters.autogen_handler import instrument_autogen as _impl
    return _impl(agent, threshold=threshold, verbose=verbose)


# Backward compatibility
CitationGuard = LongTracer

__all__ = [
    "LongTracer",
    "CitationGuard",  # backward compat
    "CitationVerifier",
    "VerificationResult",
    "check",
    "check_batch",
    "instrument_langchain",
    "instrument_langchain_agent",
    "instrument_langgraph",
    "instrument_llamaindex",
    "instrument_haystack",
    "instrument_openai_assistant",
    "instrument_crewai",
    "instrument_autogen",
]

