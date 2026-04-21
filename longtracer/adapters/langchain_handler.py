"""
LangChain Callback Handler for LongTracer.

Hooks into LangChain's callback system to automatically capture
retrieval, prompt, LLM, and verification spans.

Usage:
    from longtracer import instrument_langchain
    chain = RetrievalQA.from_chain_type(...)
    instrument_langchain(chain)
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from contextvars import ContextVar

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.documents import Document
    from langchain_core.outputs import LLMResult
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[misc,assignment]

from longtracer.core import LongTracer
from longtracer.logging_config import log_span, log_trace_id

logger = logging.getLogger("longtracer")

_run_state_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "lt_run_state", default=None
)


def _get_state() -> Dict[str, Any]:
    """Get or create thread-local run state."""
    state = _run_state_var.get()
    if state is None:
        state = {
            "chunks": [],
            "prompts": [],
            "final_answer": None,
            "root_run_id": None,
            "retrieval_ms": 0,
            "llm_ms": 0,
            "retriever_start_time": None,
            "llm_start_time": None,
        }
        _run_state_var.set(state)
    return state


def _reset_state():
    """Reset thread-local run state."""
    _run_state_var.set(None)


def _check_langchain():
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this adapter. "
            "Install with: pip install 'longtracer[langchain]'"
        )


def normalize_doc(doc) -> Dict[str, Any]:
    """
    Normalize a LangChain Document into a stable dict with chunk_id.
    """
    content = doc.page_content or ""
    metadata = doc.metadata or {}
    source = metadata.get("source", "unknown")
    page = metadata.get("page", 0)
    section = metadata.get("section", "")

    raw = f"{source}:{page}:{content[:100]}"
    chunk_id = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    return {
        "chunk_id": chunk_id,
        "text": content[:500],
        "source": source,
        "page": page,
        "section": section,
        "metadata": metadata,
    }


class LongTracerCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for LongTracer tracing.

    Captures retrieval, prompt, LLM, and verification spans automatically.
    Verification is triggered ONLY at root chain end when chunks + answer exist.
    """

    name = "LongTracerCallbackHandler"

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        state = _get_state()
        if state["root_run_id"] is None:
            state["root_run_id"] = str(run_id)
            logger.debug(f"Root chain started: {run_id}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        state = _get_state()

        if str(run_id) != state.get("root_run_id"):
            return

        tracer = LongTracer.get_tracer()
        if not tracer or not LongTracer.is_enabled():
            _reset_state()
            return

        answer = state.get("final_answer")
        if answer is None:
            answer = (
                outputs.get("result")
                or outputs.get("output")
                or outputs.get("answer")
                or outputs.get("text")
                or str(outputs)
            )

        chunks = state.get("chunks", [])

        if chunks and answer:
            self._run_verification(tracer, state, answer, chunks)

        if tracer.root_run:
            trace_id = tracer.root_run.get("trace_id", "")
            if trace_id and LongTracer.is_verbose():
                log_trace_id(trace_id)

        _reset_state()

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        state = _get_state()
        state["retriever_start_time"] = time.time()

    def on_retriever_end(
        self,
        documents: List[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        state = _get_state()
        tracer = LongTracer.get_tracer()

        retriever_duration_ms = 0
        if state.get("retriever_start_time"):
            retriever_duration_ms = (time.time() - state["retriever_start_time"]) * 1000

        normalized = [normalize_doc(doc) for doc in documents]
        state["chunks"].extend(normalized)

        if tracer and LongTracer.is_enabled():
            with tracer.span("retrieval", run_type="retriever") as span:
                span.set_output({
                    "chunks": normalized,
                    "count": len(normalized),
                })
                span_run_id = span.run["run_id"]
            if retriever_duration_ms > 0:
                tracer._safe_update_run(span_run_id, {"duration_ms": retriever_duration_ms})

            if LongTracer.is_verbose():
                log_span("retrieval", chunks=len(normalized))

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        state = _get_state()
        state["prompts"].extend(prompts)
        state["llm_start_time"] = time.time()

        tracer = LongTracer.get_tracer()
        if tracer and LongTracer.is_enabled():
            combined = "\n---\n".join(prompts)
            with tracer.span("llm_prep", run_type="llm") as span:
                span.set_output({
                    "system_prompt": combined[:2000],
                    "context_length_chars": len(combined),
                })

            if LongTracer.is_verbose():
                log_span("llm_prep", chars=len(combined))

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        state = _get_state()
        tracer = LongTracer.get_tracer()

        llm_duration_ms = 0
        if state.get("llm_start_time"):
            llm_duration_ms = (time.time() - state["llm_start_time"]) * 1000

        text = ""
        if response.generations:
            text = response.generations[0][0].text if response.generations[0] else ""

        state["final_answer"] = text

        model = ""
        if response.llm_output:
            model = response.llm_output.get("model_name", "")

        if tracer and LongTracer.is_enabled():
            with tracer.span("llm_call", run_type="llm") as span:
                span.set_output({
                    "answer": text[:1000],
                    "model": model,
                })
                span_run_id = span.run["run_id"]
            if llm_duration_ms > 0:
                tracer._safe_update_run(span_run_id, {"duration_ms": llm_duration_ms})

            if LongTracer.is_verbose():
                log_span("llm_call", model=model, answer_len=len(text))

    def _run_verification(
        self,
        tracer,
        state: Dict[str, Any],
        answer: str,
        chunks: List[Dict],
    ):
        try:
            from longtracer.guard.verifier import CitationVerifier
            from longtracer.guard.context_relevance import ContextRelevanceScorer
        except ImportError:
            logger.warning("Could not import verifier/scorer — skipping verification")
            return

        try:
            source_texts = [c.get("text", "") for c in chunks]
            source_metadata = [c.get("metadata", {}) for c in chunks]

            verify_start = time.time()
            verifier = CitationVerifier()
            result = verifier.verify_parallel(answer, source_texts, source_metadata=source_metadata)
            verify_duration_ms = (time.time() - verify_start) * 1000

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
                })
                span_run_id = span.run["run_id"]
            tracer._safe_update_run(span_run_id, {"duration_ms": verify_duration_ms})

            if LongTracer.is_verbose():
                supported = sum(1 for c in claims_data if c["status"] == "supported")
                log_span("eval_claims", total=len(claims_data), supported=supported)

            grounding_start = time.time()
            hallucinated = [
                c["claim_id"] for c in claims_data if c["is_hallucination"]
            ]
            flags = []
            if hallucinated:
                flags.append("HALLUCINATION")
            if result.trust_score < 0.5:
                flags.append("LOW_TRUST")
            grounding_duration_ms = (time.time() - grounding_start) * 1000

            with tracer.span("grounding", run_type="chain") as span:
                span.set_output({
                    "grounding_score": result.trust_score,
                    "hallucinated_claim_ids": hallucinated,
                    "hallucination_count": len(hallucinated),
                    "flags_triggered": flags,
                    "verdict": "PASS" if not flags else "FAIL",
                })
                span_run_id = span.run["run_id"]
            tracer._safe_update_run(span_run_id, {"duration_ms": grounding_duration_ms})

            if LongTracer.is_verbose():
                log_span(
                    "grounding",
                    score=f"{result.trust_score:.2f}",
                    verdict="PASS" if not flags else "FAIL",
                    flags=flags,
                )

        except Exception as e:
            logger.error(f"Verification failed: {e}")


def instrument_langchain(chain, verbose: Optional[bool] = None):
    """
    Attach LongTracer to a LangChain chain.

    Usage:
        chain = RetrievalQA.from_chain_type(...)
        instrument_langchain(chain)
    """
    _check_langchain()

    if not LongTracer.is_enabled():
        LongTracer.init(verbose=verbose)

    handler = LongTracerCallbackHandler()

    if hasattr(chain, "callbacks"):
        if chain.callbacks is None:
            chain.callbacks = [handler]
        else:
            chain.callbacks.append(handler)
    elif hasattr(chain, "config"):
        config = chain.config if hasattr(chain, "config") else {}
        callbacks = config.get("callbacks", [])
        callbacks.append(handler)
        chain.config = {**config, "callbacks": callbacks}
    else:
        logger.warning(
            "Could not attach handler — pass handler manually to chain.invoke(config={'callbacks': [handler]})"
        )

    logger.info("LongTracer instrumented for LangChain")
    return handler
