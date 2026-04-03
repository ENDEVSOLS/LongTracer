"""
LlamaIndex Callback Handler for LongTracer.

Hooks into LlamaIndex's callback manager to automatically capture
retrieval, prompt, LLM, and verification spans.

Usage:
    from longtracer import instrument_llamaindex
    query_engine = index.as_query_engine()
    instrument_llamaindex(query_engine)
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional
from contextvars import ContextVar

from longtracer.core import LongTracer
from longtracer.logging_config import log_span, log_trace_id

logger = logging.getLogger("longtracer")

_run_state_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "lt_llama_run_state", default=None
)


def _get_state() -> Dict[str, Any]:
    state = _run_state_var.get()
    if state is None:
        state = {
            "chunks": [],
            "prompts": [],
            "final_answer": None,
        }
        _run_state_var.set(state)
    return state


def _reset_state():
    _run_state_var.set(None)


def normalize_node(node) -> Dict[str, Any]:
    """Normalize a LlamaIndex NodeWithScore or TextNode into a stable dict."""
    actual_node = getattr(node, "node", node)
    content = getattr(actual_node, "text", "") or getattr(actual_node, "get_content", lambda: "")()
    metadata = getattr(actual_node, "metadata", {}) or getattr(actual_node, "extra_info", {}) or {}
    score = getattr(node, "score", None)
    source = metadata.get("file_name", metadata.get("source", "unknown"))
    page = metadata.get("page_label", metadata.get("page", 0))

    raw = f"{source}:{page}:{content[:100]}"
    chunk_id = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    result = {
        "chunk_id": chunk_id,
        "text": content[:500],
        "source": source,
        "page": page,
        "metadata": metadata,
    }
    if score is not None:
        result["score"] = float(score)
    return result


from llama_index.core.callbacks.base_handler import BaseCallbackHandler

class CitationGuardLlamaIndexHandler(BaseCallbackHandler):
    """
    LlamaIndex callback handler for LongTracer tracing.

    Compatible with LlamaIndex's CallbackManager system.
    Verification triggers only at SYNTHESIZE/QUERY event end.
    """

    def __init__(self):
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self._event_starts = {}

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def on_event_start(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs,
    ) -> str:
        self._event_starts[event_id] = {
            "type": event_type,
            "parent_id": parent_id,
        }
        return event_id

    def on_event_end(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs,
    ) -> None:
        tracer = LongTracer.get_tracer()
        if not tracer or not LongTracer.is_enabled():
            return

        payload = payload or {}
        state = _get_state()

        try:
            from llama_index.core.callbacks import CBEventType
            retrieve_type = CBEventType.RETRIEVE
            llm_type = CBEventType.LLM
            synth_type = CBEventType.SYNTHESIZE
            query_type = CBEventType.QUERY
        except ImportError:
            retrieve_type = "retrieve"
            llm_type = "llm"
            synth_type = "synthesize"
            query_type = "query"

        if event_type == retrieve_type:
            self._handle_retrieve(tracer, state, payload)
        elif event_type == llm_type:
            self._handle_llm(tracer, state, payload)
        elif event_type in (synth_type, query_type):
            self._handle_synthesize(tracer, state, payload)

        self._event_starts.pop(event_id, None)

    def _handle_retrieve(self, tracer, state, payload):
        nodes = payload.get("nodes", [])
        normalized = [normalize_node(n) for n in nodes]
        state["chunks"].extend(normalized)

        with tracer.span("retrieval", run_type="retriever") as span:
            span.set_output({
                "chunks": normalized,
                "count": len(normalized),
            })

        if LongTracer.is_verbose():
            log_span("retrieval", chunks=len(normalized))

    def _handle_llm(self, tracer, state, payload):
        messages = payload.get("messages", [])
        prompt_text = "\n".join(str(m) for m in messages) if messages else ""

        if prompt_text:
            state["prompts"].append(prompt_text)
            with tracer.span("llm_prep", run_type="llm") as span:
                span.set_output({
                    "system_prompt": prompt_text[:2000],
                    "context_length_chars": len(prompt_text),
                })
            if LongTracer.is_verbose():
                log_span("llm_prep", chars=len(prompt_text))

        response = payload.get("response", "")
        if hasattr(response, "text"):
            response = response.text
        response = str(response) if response else ""

        if response:
            state["final_answer"] = response
            model = payload.get("serialized", {}).get("model", "")

            with tracer.span("llm_call", run_type="llm") as span:
                span.set_output({
                    "answer": response[:1000],
                    "model": model,
                })

            if LongTracer.is_verbose():
                log_span("llm_call", model=model, answer_len=len(response))

    def _handle_synthesize(self, tracer, state, payload):
        chunks = state.get("chunks", [])
        answer = state.get("final_answer")

        if not answer:
            resp = payload.get("response", "")
            if hasattr(resp, "response"):
                answer = resp.response
            elif hasattr(resp, "text"):
                answer = resp.text
            else:
                answer = str(resp) if resp else ""

        if chunks and answer:
            self._run_verification(tracer, answer, chunks)

        if tracer.root_run:
            trace_id = tracer.root_run.get("trace_id", "")
            if trace_id and LongTracer.is_verbose():
                log_trace_id(trace_id)

        _reset_state()

    def _run_verification(self, tracer, answer: str, chunks: List[Dict]):
        try:
            from longtracer.guard.verifier import CitationVerifier

            source_texts = [c.get("text", "") for c in chunks]
            source_metadata = [c.get("metadata", {}) for c in chunks]

            verifier = CitationVerifier()
            result = verifier.verify_parallel(answer, source_texts, source_metadata=source_metadata)

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

            if LongTracer.is_verbose():
                supported = sum(1 for c in claims_data if c["status"] == "supported")
                log_span("eval_claims", total=len(claims_data), supported=supported)

            hallucinated = [c["claim_id"] for c in claims_data if c["is_hallucination"]]
            flags = []
            if hallucinated:
                flags.append("HALLUCINATION")
            if result.trust_score < 0.5:
                flags.append("LOW_TRUST")

            with tracer.span("grounding", run_type="chain") as span:
                span.set_output({
                    "grounding_score": result.trust_score,
                    "hallucinated_claim_ids": hallucinated,
                    "hallucination_count": len(hallucinated),
                    "flags_triggered": flags,
                    "verdict": "PASS" if not flags else "FAIL",
                })

            if LongTracer.is_verbose():
                log_span(
                    "grounding",
                    score=f"{result.trust_score:.2f}",
                    verdict="PASS" if not flags else "FAIL",
                )

        except Exception as e:
            logger.error(f"Verification failed: {e}")


def instrument_llamaindex(query_engine, verbose: Optional[bool] = None):
    """
    Attach LongTracer to a LlamaIndex query engine.

    Usage:
        query_engine = index.as_query_engine()
        instrument_llamaindex(query_engine)
    """
    if not LongTracer.is_enabled():
        LongTracer.init(verbose=verbose)

    handler = CitationGuardLlamaIndexHandler()

    try:
        from llama_index.core.callbacks import CallbackManager

        if hasattr(query_engine, "callback_manager"):
            cb_manager = query_engine.callback_manager
            if cb_manager is None:
                cb_manager = CallbackManager([handler])
                query_engine.callback_manager = cb_manager
            else:
                cb_manager.add_handler(handler)
        elif hasattr(query_engine, "_callback_manager"):
            cb_manager = query_engine._callback_manager
            if cb_manager is None:
                cb_manager = CallbackManager([handler])
                query_engine._callback_manager = cb_manager
            else:
                cb_manager.add_handler(handler)
        else:
            logger.warning(
                "Could not find callback_manager on query engine. "
                "Pass handler manually via Settings.callback_manager."
            )
    except ImportError:
        logger.warning(
            "llama_index.core not found — install with: pip install llama-index-core"
        )

    logger.info("LongTracer instrumented for LlamaIndex")
    return handler
