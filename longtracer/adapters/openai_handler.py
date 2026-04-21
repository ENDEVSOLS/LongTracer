"""
OpenAI Assistants API adapter for LongTracer.

Wraps an OpenAI client to automatically verify assistant responses
against retrieved context (file_search citations). Verification runs
after each completed assistant run.

Usage:
    from longtracer import instrument_openai_assistant

    from openai import OpenAI
    client = OpenAI()
    instrument_openai_assistant(client)

    # Use the client normally — verification happens automatically:
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    # Access verification result:
    result = run._longtracer_result  # VerificationResult or None
"""

import logging
import time
from typing import Any, Dict, List, Optional

try:
    import openai as _openai_mod
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from longtracer.core import LongTracer
from longtracer.logging_config import log_span, log_trace_id

logger = logging.getLogger("longtracer")


def _check_openai() -> None:
    """Raise ImportError with install instructions if openai is missing."""
    if not _OPENAI_AVAILABLE:
        raise ImportError(
            "The openai package is required for the OpenAI Assistants adapter. "
            "Install with: pip install 'longtracer[openai]'"
        )


def _extract_assistant_response(client: Any, thread_id: str) -> str:
    """Extract the latest assistant message text from a thread.

    Returns:
        The concatenated text of the last assistant message, or empty string.
    """
    try:
        messages = client.beta.threads.messages.list(
            thread_id=thread_id, order="desc", limit=10,
        )
        for msg in messages.data:
            if msg.role == "assistant" and msg.content:
                parts = []
                for block in msg.content:
                    if hasattr(block, "text") and block.text:
                        parts.append(block.text.value)
                if parts:
                    return "\n".join(parts)
    except Exception as exc:
        logger.warning("LongTracer: failed to extract assistant response: %s", exc)
    return ""


def _extract_citations(client: Any, thread_id: str, run_id: str) -> List[str]:
    """Extract source text from file_search citation annotations.

    Retrieves run steps to find file_search results and extracts
    the cited text content.

    Returns:
        List of source text strings from citations.
    """
    sources: List[str] = []
    try:
        # Get run steps to find retrieval results
        steps = client.beta.threads.runs.steps.list(
            thread_id=thread_id, run_id=run_id,
        )
        for step in steps.data:
            if step.type == "tool_calls" and step.step_details:
                for tool_call in step.step_details.tool_calls:
                    if getattr(tool_call, "type", None) == "file_search":
                        # Extract results from file_search
                        fs = getattr(tool_call, "file_search", None)
                        if fs and hasattr(fs, "results") and fs.results:
                            for result in fs.results:
                                content = getattr(result, "content", None)
                                if content:
                                    for c in content:
                                        text = getattr(c, "text", "")
                                        if text:
                                            sources.append(text[:500])
                                # Fallback: use the file name as context
                                elif hasattr(result, "file_name"):
                                    sources.append(f"[File: {result.file_name}]")

        # Also extract inline citation annotations from messages
        messages = client.beta.threads.messages.list(
            thread_id=thread_id, order="desc", limit=5,
        )
        for msg in messages.data:
            if msg.role == "assistant" and msg.content:
                for block in msg.content:
                    if hasattr(block, "text") and block.text:
                        annotations = getattr(block.text, "annotations", [])
                        for ann in annotations:
                            if hasattr(ann, "file_citation"):
                                quote = getattr(ann.file_citation, "quote", "")
                                if quote and quote not in sources:
                                    sources.append(quote[:500])

    except Exception as exc:
        logger.debug("LongTracer: citation extraction partial/failed: %s", exc)

    return sources


def _run_verification(
    response: str,
    sources: List[str],
    threshold: float = 0.5,
) -> Any:
    """Run CitationVerifier on the response against sources.

    Returns:
        VerificationResult or None on failure.
    """
    if not response or not sources:
        return None

    try:
        from longtracer.guard.verifier import CitationVerifier

        verify_start = time.time()
        verifier = CitationVerifier(threshold=threshold)
        result = verifier.verify_parallel(response, sources)
        verify_ms = (time.time() - verify_start) * 1000

        # Log to tracer if enabled
        tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
        if tracer:
            tracer.start_root(inputs={"adapter": "openai_assistants"})

            with tracer.span("retrieval", run_type="retriever") as span:
                span.set_output({
                    "count": len(sources),
                    "source_preview": [s[:100] for s in sources[:3]],
                })

            claims_data = []
            for i, claim in enumerate(result.claims):
                claims_data.append({
                    "claim_id": f"claim_{i}",
                    "text": claim["claim"][:200],
                    "status": "supported" if claim["supported"] else "unsupported",
                    "score": claim["score"],
                    "is_hallucination": claim.get("is_hallucination", False),
                })

            with tracer.span("eval_claims", run_type="chain") as span:
                span.set_output({
                    "claims": claims_data,
                    "total_claims": len(claims_data),
                    "verify_ms": round(verify_ms, 1),
                })

            with tracer.span("grounding", run_type="chain") as span:
                span.set_output({
                    "trust_score": result.trust_score,
                    "verdict": result.verdict,
                    "summary": result.summary,
                    "hallucination_count": result.hallucination_count,
                })

            tracer.end_root(outputs={
                "verdict": result.verdict,
                "trust_score": result.trust_score,
                "adapter": "openai_assistants",
            })

            if LongTracer.is_verbose():
                log_span(
                    "grounding",
                    score=f"{result.trust_score:.2f}",
                    verdict=result.verdict,
                )
                trace_id = tracer.root_run.get("trace_id", "") if tracer.root_run else ""
                if trace_id:
                    log_trace_id(trace_id)

        return result

    except Exception as exc:
        logger.error("LongTracer: OpenAI verification failed: %s", exc)
        return None


def instrument_openai_assistant(
    client: Any,
    threshold: float = 0.5,
    verbose: Optional[bool] = None,
) -> Any:
    """Instrument an OpenAI client to verify assistant responses.

    Monkey-patches ``client.beta.threads.runs.create_and_poll`` to
    automatically run hallucination verification after each completed run.

    Args:
        client: An ``openai.OpenAI`` or ``openai.AsyncOpenAI`` client instance.
        threshold: Verification threshold (default 0.5).
        verbose: Override verbose setting.

    Returns:
        The patched client (same instance, modified in-place).

    Usage::

        from openai import OpenAI
        from longtracer import instrument_openai_assistant

        client = OpenAI()
        instrument_openai_assistant(client)

        # Normal usage — verification happens automatically
        run = client.beta.threads.runs.create_and_poll(...)
        if hasattr(run, '_longtracer_result'):
            print(run._longtracer_result.verdict)
    """
    _check_openai()

    if not LongTracer.is_enabled():
        LongTracer.init(verbose=verbose)

    # Guard against double-patching
    if getattr(client, "_longtracer_patched", False):
        logger.debug("LongTracer: OpenAI client already instrumented, skipping")
        return client

    runs_resource = client.beta.threads.runs

    # Patch create_and_poll
    original_create_and_poll = runs_resource.create_and_poll

    def patched_create_and_poll(*args: Any, **kwargs: Any) -> Any:
        run = original_create_and_poll(*args, **kwargs)

        if getattr(run, "status", None) != "completed":
            return run

        try:
            thread_id = run.thread_id
            response_text = _extract_assistant_response(client, thread_id)
            sources = _extract_citations(client, thread_id, run.id)
            result = _run_verification(response_text, sources, threshold=threshold)

            # Attach result to the run object
            try:
                run._longtracer_result = result
            except (AttributeError, TypeError):
                # Frozen/immutable object — store in a side dict
                pass

        except Exception as exc:
            logger.warning("LongTracer: post-run verification failed: %s", exc)

        return run

    runs_resource.create_and_poll = patched_create_and_poll

    # Also patch create (non-polling) to add a verify helper
    original_create = runs_resource.create

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        run = original_create(*args, **kwargs)

        def verify_run(completed_run: Any = None) -> Any:
            """Manually verify this run after it completes.

            Args:
                completed_run: The completed Run object (if re-fetched).
                               If None, uses the original run's thread_id.

            Returns:
                VerificationResult or None.
            """
            target = completed_run or run
            tid = getattr(target, "thread_id", None)
            rid = getattr(target, "id", None)
            if not tid or not rid:
                return None
            response_text = _extract_assistant_response(client, tid)
            sources = _extract_citations(client, tid, rid)
            return _run_verification(response_text, sources, threshold=threshold)

        try:
            run._longtracer_verify = verify_run
        except (AttributeError, TypeError):
            pass

        return run

    runs_resource.create = patched_create

    client._longtracer_patched = True
    logger.info("LongTracer: OpenAI Assistants client instrumented")
    return client


def verify_assistant_run(
    client: Any,
    thread_id: str,
    run_id: str,
    threshold: float = 0.5,
) -> Any:
    """Manually verify a completed assistant run.

    Standalone function for verifying an assistant run without
    monkey-patching the client.

    Args:
        client: An ``openai.OpenAI`` client instance.
        thread_id: The thread ID.
        run_id: The run ID.
        threshold: Verification threshold.

    Returns:
        VerificationResult or None.

    Usage::

        result = verify_assistant_run(client, thread.id, run.id)
        print(result.verdict, result.trust_score)
    """
    _check_openai()

    response_text = _extract_assistant_response(client, thread_id)
    sources = _extract_citations(client, thread_id, run_id)
    return _run_verification(response_text, sources, threshold=threshold)


# Backward compatibility alias
CitationGuardOpenAIHandler = instrument_openai_assistant
