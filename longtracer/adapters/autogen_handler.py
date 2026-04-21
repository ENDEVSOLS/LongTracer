"""
AutoGen adapter for LongTracer.

Wraps AutoGen ≥0.4 (agentchat API) agents to verify their responses
against retrieved context or conversation history.

Supports:
- ``autogen_agentchat.AssistantAgent``
- Any agent with ``on_messages`` method

Usage:
    from longtracer import instrument_autogen

    from autogen_agentchat.agents import AssistantAgent
    agent = AssistantAgent(name="assistant", model_client=model)
    instrument_autogen(agent)

    # Or standalone:
    from longtracer.adapters.autogen_handler import verify_autogen_result
    result = verify_autogen_result(response_text, sources)
"""

import logging
import time
from typing import Any, Dict, List, Optional

try:
    from autogen_agentchat.agents import AssistantAgent as _AssistantAgent
    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False
    _AssistantAgent = None

from longtracer.core import LongTracer
from longtracer.logging_config import log_span, log_trace_id

logger = logging.getLogger("longtracer")


def _check_autogen() -> None:
    """Raise ImportError with install instructions if autogen is missing."""
    if not _AUTOGEN_AVAILABLE:
        raise ImportError(
            "The autogen-agentchat package (≥0.4) is required for the AutoGen adapter. "
            "Install with: pip install 'longtracer[autogen]'"
        )


def _extract_message_text(message: Any) -> str:
    """Extract text content from an AutoGen message.

    Handles various message types:
    - ``TextMessage`` — ``.content`` is a string
    - ``MultiModalMessage`` — ``.content`` is a list of items
    - ``ToolCallResultMessage`` — ``.content`` is a list of results
    - Plain strings
    - Dicts with ``content`` key

    Returns:
        Extracted text, or empty string.
    """
    if isinstance(message, str):
        return message

    # TextMessage, ChatMessage, etc.
    content = getattr(message, "content", None)
    if content is None:
        return str(message) if message else ""

    if isinstance(content, str):
        return content

    # MultiModal: list of content items
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)

    return str(content)


def _extract_sources_from_messages(messages: List[Any]) -> List[str]:
    """Extract potential source texts from a conversation history.

    Uses all non-assistant messages as potential sources — this is
    the best approximation since AutoGen agents receive context
    through their message history.

    Args:
        messages: List of AutoGen message objects.

    Returns:
        List of source text strings.
    """
    sources: List[str] = []
    for msg in messages:
        role = getattr(msg, "source", getattr(msg, "role", ""))

        # Skip the agent's own responses
        if role in ("assistant",):
            continue

        text = _extract_message_text(msg)
        if text and len(text) > 20:  # Skip very short messages
            sources.append(text[:1000])

    return sources


def _run_verification(
    response: str,
    sources: List[str],
    threshold: float,
    agent_name: str,
) -> Any:
    """Run CitationVerifier on the response against sources.

    Returns:
        VerificationResult or None.
    """
    if not response or not response.strip() or not sources:
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
            claims_data = []
            for i, claim in enumerate(result.claims):
                claims_data.append({
                    "claim_id": f"claim_{i}",
                    "text": claim["claim"][:200],
                    "status": "supported" if claim["supported"] else "unsupported",
                    "score": claim["score"],
                    "is_hallucination": claim.get("is_hallucination", False),
                })

            with tracer.span("autogen_eval", run_type="chain") as span:
                span.set_output({
                    "agent": agent_name,
                    "claims": claims_data,
                    "total_claims": len(claims_data),
                    "trust_score": result.trust_score,
                    "verdict": result.verdict,
                    "verify_ms": round(verify_ms, 1),
                })

        if LongTracer.is_verbose():
            log_span(
                "autogen_eval",
                agent=agent_name,
                verdict=result.verdict,
                score=f"{result.trust_score:.2f}",
            )

        return result

    except Exception as exc:
        logger.error("LongTracer: AutoGen verification failed: %s", exc)
        return None


def instrument_autogen(
    agent: Any,
    threshold: float = 0.5,
    verbose: Optional[bool] = None,
) -> Any:
    """Instrument an AutoGen agent to verify its responses.

    Wraps the agent's ``on_messages`` method to run hallucination
    verification on each response.

    Args:
        agent: An ``autogen_agentchat.AssistantAgent`` or compatible agent.
        threshold: Verification threshold (default 0.5).
        verbose: Override verbose setting.

    Returns:
        The agent (same instance, modified in-place).

    Usage::

        from autogen_agentchat.agents import AssistantAgent
        from longtracer import instrument_autogen

        agent = AssistantAgent(name="assistant", model_client=model)
        instrument_autogen(agent)

        # Normal usage — verification happens automatically
        result = await agent.on_messages(messages, ctx)
    """
    _check_autogen()

    if not LongTracer.is_enabled():
        LongTracer.init(verbose=verbose)

    # Guard against double-patching
    if getattr(agent, "_longtracer_patched", False):
        logger.debug("LongTracer: AutoGen agent already instrumented, skipping")
        return agent

    agent_name = getattr(agent, "name", "unknown_agent")

    # Wrap on_messages (the primary response method in autogen ≥0.4)
    if hasattr(agent, "on_messages"):
        original_on_messages = agent.on_messages

        async def patched_on_messages(messages: Any, cancellation_token: Any = None, **kwargs: Any) -> Any:
            """Wrapped on_messages that verifies the agent's response."""
            # Initialize tracer for this agent call
            tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
            if tracer:
                tracer.start_root(inputs={
                    "adapter": "autogen",
                    "agent": agent_name,
                })

            # Call the original method
            response = await original_on_messages(messages, cancellation_token, **kwargs)

            # Extract response text
            response_text = ""
            if hasattr(response, "chat_message"):
                response_text = _extract_message_text(response.chat_message)
            elif hasattr(response, "content"):
                response_text = _extract_message_text(response)

            # Extract sources from input messages
            msg_list = messages if isinstance(messages, list) else [messages]
            sources = _extract_sources_from_messages(msg_list)

            # Run verification
            if response_text and sources:
                verification = _run_verification(
                    response_text, sources, threshold, agent_name,
                )
                try:
                    response._longtracer_result = verification
                except (AttributeError, TypeError):
                    pass

            # Close tracer root
            if tracer and tracer.root_run:
                tracer.end_root(outputs={
                    "adapter": "autogen",
                    "agent": agent_name,
                    "response_len": len(response_text),
                })
                if LongTracer.is_verbose():
                    trace_id = tracer.root_run.get("trace_id", "") if tracer.root_run else ""
                    if trace_id:
                        log_trace_id(trace_id)

            return response

        agent.on_messages = patched_on_messages

    # Also wrap on_messages_stream if present
    if hasattr(agent, "on_messages_stream"):
        original_stream = agent.on_messages_stream

        async def patched_stream(messages: Any, cancellation_token: Any = None, **kwargs: Any):
            """Wrapped stream that verifies the final response."""
            final_response = None
            async for item in original_stream(messages, cancellation_token, **kwargs):
                final_response = item
                yield item

            # Verify the final response
            if final_response is not None:
                response_text = ""
                if hasattr(final_response, "chat_message"):
                    response_text = _extract_message_text(final_response.chat_message)
                elif hasattr(final_response, "content"):
                    response_text = _extract_message_text(final_response)

                msg_list = messages if isinstance(messages, list) else [messages]
                sources = _extract_sources_from_messages(msg_list)

                if response_text and sources:
                    verification = _run_verification(
                        response_text, sources, threshold, agent_name,
                    )
                    try:
                        final_response._longtracer_result = verification
                    except (AttributeError, TypeError):
                        pass

        agent.on_messages_stream = patched_stream

    agent._longtracer_patched = True
    logger.info("LongTracer: AutoGen agent '%s' instrumented", agent_name)
    return agent


def verify_autogen_result(
    response: Any,
    sources: List[str],
    threshold: float = 0.5,
) -> Any:
    """Manually verify an AutoGen agent response.

    Standalone function — no monkey-patching needed.

    Args:
        response: Agent response text (string) or message object.
        sources: Source texts to verify against.
        threshold: Verification threshold.

    Returns:
        VerificationResult or None.

    Usage::

        result = await agent.on_messages(messages, ctx)
        vr = verify_autogen_result(result.chat_message.content, sources)
        print(vr.verdict, vr.trust_score)
    """
    response_text = _extract_message_text(response)
    if not response_text or not sources:
        return None

    try:
        from longtracer.guard.verifier import CitationVerifier

        verifier = CitationVerifier(threshold=threshold)
        return verifier.verify_parallel(response_text, sources)
    except Exception as exc:
        logger.error("LongTracer: AutoGen result verification failed: %s", exc)
        return None
