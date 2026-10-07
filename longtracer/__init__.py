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

try:
    import importlib.metadata as _metadata

    __version__ = _metadata.version("longtracer")
except Exception:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

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


def check_case(
    response: str,
    sources: list[str],
    source_metadata: list[dict] | None = None,
    *,
    case_id: str | None = None,
    timeout: float | None = None,
    detect_conflicts: bool = False,
    **kw,
):
    """One-liner typed verification — returns a ``CaseResult``, never raises for evaluator failures.

    Unlike ``check()``, model loading happens inside this call, so a missing or
    undownloadable model (``ModelUnavailableError``) is reported as
    ``execution=ERROR`` / ``quality_gate=INDETERMINATE`` / ``reason=MODEL_UNAVAILABLE``.

    Args:
        response: LLM response text to verify.
        sources: Source texts to verify against.
        source_metadata: Optional metadata per source.
        case_id: Optional identifier copied onto the result.
        timeout: Optional wall-clock limit in seconds.
        detect_conflicts: Opt-in ``CONFLICTING_SOURCES`` detection.
        **kw: Passed to ``CitationVerifier(...)`` (e.g. ``cache=True``).

    Returns:
        CaseResult with execution, availability, claim assessments and quality gate.
    """
    from longtracer.contracts.result import ExecutionStatus, ReasonCode, unassessed_case
    from longtracer.errors import ModelUnavailableError

    try:
        verifier = CitationVerifier(**kw)
    except ModelUnavailableError as e:
        return unassessed_case(ExecutionStatus.ERROR, ReasonCode.MODEL_UNAVAILABLE, case_id=case_id, error_message=str(e))
    except Exception as e:  # any other construction failure is still never a success
        return unassessed_case(
            ExecutionStatus.ERROR, ReasonCode.EVALUATION_FAILED, case_id=case_id, error_message=f"{type(e).__name__}: {e}"
        )

    return verifier.verify_case(
        response,
        sources,
        source_metadata=source_metadata,
        case_id=case_id,
        timeout=timeout,
        detect_conflicts=detect_conflicts,
    )


# Typed evaluation contracts
from longtracer.contracts.result import (
    CaseResult,
    ClaimResult,
    ExecutionStatus,
    AssessmentAvailability,
    ClaimAssessment,
    QualityGate,
    ReasonCode,
)

# Backward compatibility
CitationGuard = LongTracer

__all__ = [
    "__version__",
    "LongTracer",
    "CitationGuard",  # backward compat
    "CitationVerifier",
    "VerificationResult",
    "check",
    "check_batch",
    "check_case",
    "CaseResult",
    "ClaimResult",
    "ExecutionStatus",
    "AssessmentAvailability",
    "ClaimAssessment",
    "QualityGate",
    "ReasonCode",
    "instrument_langchain",
    "instrument_langchain_agent",
    "instrument_langgraph",
    "instrument_llamaindex",
    "instrument_haystack",
    "instrument_openai_assistant",
    "instrument_crewai",
    "instrument_autogen",
]

