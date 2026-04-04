"""
LangGraph & LangChain Agent Callback Handler for LongTracer.

A single handler that traces and verifies agent executions across:
- LangGraph StateGraph agents
- LangGraph ``create_react_agent`` (prebuilt)
- LangGraph Functional API (``@entrypoint``)
- LangChain ``AgentExecutor``
- LangChain ``create_react_agent`` / ``create_tool_calling_agent``

The handler accumulates retrieved documents across agent steps, captures
tool calls and LLM responses as spans, and runs hallucination verification
once at agent completion.

Usage:
    from longtracer import instrument_langgraph
    handler = instrument_langgraph(graph)

    # Or pass directly in config:
    from longtracer.adapters.langgraph_handler import LongTracerAgentHandler
    result = graph.invoke(input, config={"callbacks": [LongTracerAgentHandler()]})
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID
from contextvars import ContextVar

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from langchain_core.agents import AgentAction, AgentFinish

    _LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    _LANGCHAIN_CORE_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[misc,assignment]

from longtracer.core import LongTracer
from longtracer.logging_config import log_span, log_trace_id

logger = logging.getLogger("longtracer")


def _check_langchain_core() -> None:
    if not _LANGCHAIN_CORE_AVAILABLE:
        raise ImportError(
            "langchain-core is required for LangGraph/LangChain agent tracing. "
            "Install with: pip install 'longtracer[langgraph]'"
        )


# ── Thread-safe agent state ─────────────────────────────────────

_agent_state_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "lt_agent_state", default=None
)


def _get_agent_state() -> Dict[str, Any]:
    """Get or create thread-safe agent run state."""
    state = _agent_state_var.get()
    if state is None:
        state = {
            "sources": [],
            "source_metadata": [],
            "tool_calls": [],
            "llm_responses": [],
            "final_answer": None,
            "root_run_id": None,
            "chain_depth": 0,
            "retriever_start_time": None,
            "llm_start_time": None,
            "tool_start_time": None,
        }
        _agent_state_var.set(state)
    return state


def _reset_agent_state() -> None:
    _agent_state_var.set(None)


def _normalize_document(doc: Any) -> Dict[str, Any]:
    """Normalize a LangChain Document into a stable dict."""
    content = ""
    metadata: Dict[str, Any] = {}

    if hasattr(doc, "page_content"):
        content = doc.page_content or ""
        metadata = doc.metadata or {}
    elif isinstance(doc, dict):
        content = doc.get("page_content", doc.get("text", ""))
        metadata = doc.get("metadata", {})

    source = metadata.get("source", "unknown")
    page = metadata.get("page", 0)
    raw = f"{source}:{page}:{content[:100]}"
    chunk_id = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    return {
        "chunk_id": chunk_id,
        "text": content[:500],
        "source": source,
        "page": page,
        "metadata": metadata,
    }


def _extract_text_from_message(message: Any) -> str:
    """Extract text content from a LangChain message or string."""
    if isinstance(message, str):
        return message
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(parts)
    return str(message)


# ── Main handler ────────────────────────────────────────────────

if _LANGCHAIN_CORE_AVAILABLE:

    class LongTracerAgentHandler(BaseCallbackHandler):
        """Callback handler for LangGraph and LangChain agent tracing.

        Captures retriever outputs, tool calls, LLM responses, and agent
        steps. Runs hallucination verification once at agent completion.

        Works with:
        - LangGraph ``StateGraph`` agents
        - LangGraph ``create_react_agent``
        - LangChain ``AgentExecutor``
        - LangChain ``create_react_agent`` / ``create_tool_calling_agent``
        - Any ``Runnable`` that uses ``BaseCallbackHandler``
        """

        name = "LongTracerAgentHandler"

        def __init__(
            self,
            threshold: float = 0.5,
            verbose: Optional[bool] = None,
        ):
            self.threshold = threshold
            self._verbose = verbose

        @property
        def _is_verbose(self) -> bool:
            if self._verbose is not None:
                return self._verbose
            return LongTracer.is_verbose()

        # ── Chain / graph node events ───────────────────────────

        def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            state["chain_depth"] += 1
            if state["root_run_id"] is None:
                state["root_run_id"] = str(run_id)
                tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
                if tracer:
                    tracer.start_root(
                        run_name="agent_execution",
                        inputs={"run_id": str(run_id)},
                    )

        def on_chain_end(
            self,
            outputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            state["chain_depth"] -= 1

            if str(run_id) != state.get("root_run_id"):
                return

            # Root chain ended — run verification
            self._finalize(state, outputs)

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            state["chain_depth"] -= 1
            if str(run_id) == state.get("root_run_id"):
                tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
                if tracer and tracer.root_run:
                    tracer.end_root(outputs={"error": str(error)})
                _reset_agent_state()

        # ── Retriever events ────────────────────────────────────

        def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            state["retriever_start_time"] = time.time()

        def on_retriever_end(
            self,
            documents: List[Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            retrieval_ms = 0.0
            if state.get("retriever_start_time"):
                retrieval_ms = (time.time() - state["retriever_start_time"]) * 1000

            for doc in documents:
                normalized = _normalize_document(doc)
                state["sources"].append(normalized["text"])
                state["source_metadata"].append(normalized["metadata"])

            tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
            if tracer:
                with tracer.span("retrieval", run_type="retriever") as span:
                    span.set_output({
                        "count": len(documents),
                        "retrieval_ms": round(retrieval_ms, 1),
                    })

            if self._is_verbose:
                log_span("retrieval", chunks=len(documents))

        # ── Tool events ─────────────────────────────────────────

        def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            state["tool_start_time"] = time.time()
            tool_name = serialized.get("name", "unknown_tool")
            state["tool_calls"].append({
                "name": tool_name,
                "input": input_str[:500],
                "run_id": str(run_id),
            })

        def on_tool_end(
            self,
            output: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            tool_ms = 0.0
            if state.get("tool_start_time"):
                tool_ms = (time.time() - state["tool_start_time"]) * 1000

            # If tool output contains documents, accumulate as sources
            output_str = str(output) if output else ""
            if hasattr(output, "documents"):
                for doc in output.documents:
                    normalized = _normalize_document(doc)
                    state["sources"].append(normalized["text"])
                    state["source_metadata"].append(normalized["metadata"])

            tool_name = "unknown_tool"
            if state["tool_calls"]:
                for tc in reversed(state["tool_calls"]):
                    if tc["run_id"] == str(run_id):
                        tool_name = tc["name"]
                        break

            tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
            if tracer:
                with tracer.span("tool_call", run_type="tool") as span:
                    span.set_output({
                        "tool": tool_name,
                        "output_preview": output_str[:300],
                        "tool_ms": round(tool_ms, 1),
                    })

            if self._is_verbose:
                log_span("tool_call", tool=tool_name)

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            logger.warning("Tool error during agent execution: %s", error)

        # ── LLM / Chat model events ────────────────────────────

        def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[Any]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            state["llm_start_time"] = time.time()

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            state["llm_start_time"] = time.time()

        def on_chat_model_end(
            self,
            response: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            self._handle_llm_response(response)

        def on_llm_end(
            self,
            response: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            self._handle_llm_response(response)

        def _handle_llm_response(self, response: Any) -> None:
            """Extract text from LLM response and log span."""
            state = _get_agent_state()
            llm_ms = 0.0
            if state.get("llm_start_time"):
                llm_ms = (time.time() - state["llm_start_time"]) * 1000

            text = ""
            model = ""

            # ChatGeneration (LangGraph / chat models)
            if hasattr(response, "generations") and response.generations:
                gen = response.generations[0]
                if isinstance(gen, list) and gen:
                    gen = gen[0]
                if hasattr(gen, "message"):
                    text = _extract_text_from_message(gen.message)
                elif hasattr(gen, "text"):
                    text = gen.text or ""

            if hasattr(response, "llm_output") and response.llm_output:
                model = response.llm_output.get("model_name", "")

            if text:
                state["llm_responses"].append(text)
                state["final_answer"] = text

            tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
            if tracer:
                with tracer.span("llm_call", run_type="llm") as span:
                    span.set_output({
                        "answer_preview": text[:500],
                        "model": model,
                        "llm_ms": round(llm_ms, 1),
                    })

            if self._is_verbose:
                log_span("llm_call", model=model, answer_len=len(text))

        # ── Agent-specific events (LangChain AgentExecutor) ─────

        def on_agent_action(
            self,
            action: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            tool_name = getattr(action, "tool", "unknown")
            tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
            if tracer:
                with tracer.span("agent_action", run_type="chain") as span:
                    span.set_output({
                        "tool": tool_name,
                        "tool_input_preview": str(
                            getattr(action, "tool_input", "")
                        )[:300],
                    })

            if self._is_verbose:
                log_span("agent_action", tool=tool_name)

        def on_agent_finish(
            self,
            finish: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            state = _get_agent_state()
            output_text = getattr(finish, "return_values", {})
            if isinstance(output_text, dict):
                output_text = (
                    output_text.get("output")
                    or output_text.get("result")
                    or output_text.get("answer")
                    or str(output_text)
                )
            state["final_answer"] = str(output_text)

        # ── Verification ────────────────────────────────────────

        def _finalize(
            self,
            state: Dict[str, Any],
            outputs: Any,
        ) -> None:
            """Run verification and close the trace."""
            tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None

            answer = state.get("final_answer")
            if answer is None:
                # Try to extract from outputs
                if isinstance(outputs, dict):
                    # LangGraph: messages list
                    messages = outputs.get("messages", [])
                    if messages:
                        last = messages[-1] if isinstance(messages, list) else messages
                        answer = _extract_text_from_message(last)
                    if not answer:
                        answer = (
                            outputs.get("output")
                            or outputs.get("result")
                            or outputs.get("answer")
                            or ""
                        )
                elif isinstance(outputs, str):
                    answer = outputs

            sources = state.get("sources", [])
            source_metadata = state.get("source_metadata", [])

            if answer and sources:
                self._run_verification(tracer, answer, sources, source_metadata)
            elif answer and not sources:
                if self._is_verbose:
                    log_span("grounding", verdict="SKIP", reason="no_sources")

            if tracer and tracer.root_run:
                tracer.end_root(outputs={
                    "answer_preview": (answer or "")[:300],
                    "sources_count": len(sources),
                    "tool_calls_count": len(state.get("tool_calls", [])),
                    "llm_calls_count": len(state.get("llm_responses", [])),
                })
                if self._is_verbose:
                    trace_id = tracer.root_run.get("trace_id", "")
                    if trace_id:
                        log_trace_id(trace_id)

            _reset_agent_state()

        def _run_verification(
            self,
            tracer: Any,
            answer: str,
            sources: List[str],
            source_metadata: List[Dict],
        ) -> None:
            """Run STS + NLI verification on the final answer."""
            try:
                from longtracer.guard.verifier import CitationVerifier

                verify_start = time.time()
                verifier = CitationVerifier(threshold=self.threshold)
                result = verifier.verify_parallel(
                    answer, sources, source_metadata=source_metadata
                )
                verify_ms = (time.time() - verify_start) * 1000

                claims_data = []
                for i, claim in enumerate(result.claims):
                    claims_data.append({
                        "claim_id": f"claim_{i}",
                        "text": claim["claim"][:200],
                        "status": "supported" if claim["supported"] else "unsupported",
                        "score": claim["score"],
                        "is_hallucination": claim.get("is_hallucination", False),
                    })

                if tracer:
                    with tracer.span("eval_claims", run_type="chain") as span:
                        span.set_output({
                            "claims": claims_data,
                            "total_claims": len(claims_data),
                            "verify_ms": round(verify_ms, 1),
                        })

                if self._is_verbose:
                    supported = sum(1 for c in claims_data if c["status"] == "supported")
                    log_span("eval_claims", total=len(claims_data), supported=supported)

                hallucinated = [c["claim_id"] for c in claims_data if c["is_hallucination"]]
                flags: List[str] = []
                if hallucinated:
                    flags.append("HALLUCINATION")
                if result.trust_score < 0.5:
                    flags.append("LOW_TRUST")

                if tracer:
                    with tracer.span("grounding", run_type="chain") as span:
                        span.set_output({
                            "trust_score": result.trust_score,
                            "verdict": result.verdict,
                            "summary": result.summary,
                            "hallucination_count": len(hallucinated),
                            "flags": flags,
                        })

                if self._is_verbose:
                    log_span(
                        "grounding",
                        score=f"{result.trust_score:.2f}",
                        verdict=result.verdict,
                    )

            except Exception as exc:
                logger.error("Agent verification failed: %s", exc)

else:
    # Stub when langchain-core is not installed
    class LongTracerAgentHandler:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _check_langchain_core()


# ── Convenience functions ───────────────────────────────────────


def instrument_langgraph(
    graph: Any,
    threshold: float = 0.5,
    verbose: Optional[bool] = None,
) -> "LongTracerAgentHandler":
    """Attach LongTracer verification to a LangGraph agent.

    The handler is returned so it can also be passed via ``config``
    for streaming calls.

    Args:
        graph: A compiled LangGraph ``StateGraph`` or ``create_react_agent`` graph.
        threshold: Verification threshold (default 0.5).
        verbose: Override verbose setting.

    Returns:
        The ``LongTracerAgentHandler`` instance.

    Usage::

        from longtracer import instrument_langgraph

        handler = instrument_langgraph(graph)
        result = graph.invoke(input, config={"callbacks": [handler]})

        # Streaming:
        for chunk in graph.stream(input, config={"callbacks": [handler]}):
            print(chunk)
    """
    _check_langchain_core()

    if not LongTracer.is_enabled():
        LongTracer.init(verbose=verbose)

    handler = LongTracerAgentHandler(threshold=threshold, verbose=verbose)
    logger.info("LongTracer: LangGraph agent handler created")
    return handler


def instrument_langchain_agent(
    agent_executor: Any,
    threshold: float = 0.5,
    verbose: Optional[bool] = None,
) -> "LongTracerAgentHandler":
    """Attach LongTracer verification to a LangChain AgentExecutor.

    Args:
        agent_executor: A LangChain ``AgentExecutor`` instance.
        threshold: Verification threshold (default 0.5).
        verbose: Override verbose setting.

    Returns:
        The ``LongTracerAgentHandler`` instance.

    Usage::

        from longtracer import instrument_langchain_agent

        handler = instrument_langchain_agent(agent_executor)
        result = agent_executor.invoke({"input": "..."})
    """
    _check_langchain_core()

    if not LongTracer.is_enabled():
        LongTracer.init(verbose=verbose)

    handler = LongTracerAgentHandler(threshold=threshold, verbose=verbose)

    if hasattr(agent_executor, "callbacks"):
        if agent_executor.callbacks is None:
            agent_executor.callbacks = [handler]
        else:
            agent_executor.callbacks.append(handler)
    else:
        logger.warning(
            "Could not attach handler to AgentExecutor. "
            "Pass it manually: agent.invoke(input, config={'callbacks': [handler]})"
        )

    logger.info("LongTracer: LangChain agent handler attached")
    return handler
