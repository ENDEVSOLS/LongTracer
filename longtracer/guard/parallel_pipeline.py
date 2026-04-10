"""
Parallel Pipeline - ThreadPool-based orchestration for LongTracer.

Runs context relevance scoring in parallel with LLM generation,
then performs batch claim verification.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.retriever import RAGRetriever
    from longtracer.guard.verifier import CitationVerifier
    from longtracer.guard.context_relevance import ContextRelevanceScorer
    from longtracer.guard.tracer import Tracer


class ParallelPipeline:
    """
    Parallel pipeline for LongTracer.

    Optimizations:
    1. Context relevance runs IN PARALLEL with LLM generation
    2. Claim verification uses batch processing (all claims at once)
    """

    def __init__(
        self,
        max_workers: int = 4,
        tracer: Optional["Tracer"] = None,
    ):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tracer = tracer

    def run(
        self,
        query: str,
        retriever: "RAGRetriever",
        verifier: "CitationVerifier",
        relevance_scorer: "ContextRelevanceScorer",
        k: int = 10,
    ) -> Dict:
        """Run the parallel RAG + verification pipeline."""
        pipeline_start = time.time()
        t = self.tracer

        # Step 1: Retrieve documents
        retrieve_start = time.time()
        docs = retriever.retrieve(query, k=k)
        source_texts = [doc.page_content for doc in docs]
        source_metadata = [
            doc.metadata if hasattr(doc, "metadata") else {}
            for doc in docs
        ]
        retrieve_ms = (time.time() - retrieve_start) * 1000

        if t:
            chunks_data = []
            for i, (doc, meta) in enumerate(zip(docs, source_metadata)):
                chunks_data.append({
                    "chunk_index": i,
                    "text": doc.page_content[:500],
                    "text_length": len(doc.page_content),
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "N/A"),
                    "section": meta.get("section", ""),
                })
            with t.span("retrieval", run_type="retriever",
                        inputs={"query": query, "k": k}) as span:
                span.set_output({
                    "chunks": chunks_data,
                    "count": len(docs),
                    "retrieval_ms": round(retrieve_ms, 1),
                })

        if not docs:
            return {
                "answer": "No relevant documents found.",
                "sources": [], "source_texts": [],
                "relevance_result": {}, "verification_result": {},
                "timing": {"retrieve_ms": retrieve_ms, "total_ms": retrieve_ms},
            }

        # Step 2: Capture prompt text
        context_text = "\n\n".join([
            f"[Source: {m.get('source', 'Unknown')}]\n{t_}"
            for t_, m in zip(source_texts, source_metadata)
        ])
        try:
            full_prompt = retriever.prompt.format(
                context=context_text, question=query
            )
        except Exception:
            full_prompt = f"<context>\n{context_text}\n</context>\n\n{query}"

        if t:
            with t.span("prompt_build", run_type="chain",
                        inputs={"question": query, "chunk_count": len(docs)}) as span:
                span.set_output({
                    "system_prompt": full_prompt,
                    "context_length_chars": len(context_text),
                })

        # Step 3: PARALLEL - Context relevance + LLM
        parallel_start = time.time()

        relevance_future = self.executor.submit(
            relevance_scorer.score_with_metadata, query, source_texts, source_metadata,
        )
        llm_future = self.executor.submit(retriever.generate, query, docs)

        relevance_result = relevance_future.result()
        answer = llm_future.result()

        parallel_ms = (time.time() - parallel_start) * 1000

        if t:
            model_name = getattr(retriever.llm, "model", "unknown")
            with t.span("llm_call", run_type="llm",
                        inputs={"question": query,
                                "context_preview": context_text[:200]}) as span:
                span.set_output({
                    "answer": answer, "model": model_name,
                    "llm_ms": round(parallel_ms, 1),
                })

        if t:
            with t.span("eval_relevance", run_type="chain",
                        inputs={"query": query, "chunk_count": len(docs)}) as span:
                span.set_output({
                    "average_relevance": relevance_result.get("average_relevance", 0),
                    "top_relevance": relevance_result.get("top_relevance", 0),
                    "threshold_pass": relevance_result.get("threshold_pass", False),
                    "chunk_rankings": relevance_result.get("chunk_rankings", []),
                    "latency_ms": relevance_result.get("latency_ms", 0),
                })

        # Step 4: Batch verify claims
        verify_start = time.time()
        metadata_list = [
            doc.metadata if hasattr(doc, "metadata") else {}
            for doc in docs
        ]
        verification_result = verifier.verify_parallel(
            answer, source_texts, metadata_list
        )
        verify_ms = (time.time() - verify_start) * 1000

        if t:
            claims_data = []
            for i, claim in enumerate(verification_result.claims):
                supporting_chunks = []
                if claim.get("best_source_index") is not None:
                    idx = claim["best_source_index"]
                    if idx < len(source_metadata):
                        supporting_chunks.append({
                            "chunk_index": idx,
                            "source": source_metadata[idx].get("source", "unknown"),
                            "page": source_metadata[idx].get("page", "N/A"),
                        })
                claims_data.append({
                    "claim_id": f"claim_{i}",
                    "text": claim.get("claim", ""),
                    "status": "supported" if claim.get("supported") else "unsupported",
                    "confidence": claim.get("score", 0),
                    "best_source": claim.get("best_source", ""),
                    "contradiction_score": claim.get("contradiction_score", 0),
                    "entailment_score": claim.get("entailment_score", 0),
                    "supporting_chunks": supporting_chunks,
                    "is_hallucination": claim.get("is_hallucination", False),
                    "is_meta_statement": claim.get("is_meta_statement", False),
                    "nli_ran": claim.get("nli_ran", False),
                })

            with t.span("eval_claims", run_type="chain",
                        inputs={"answer_preview": answer[:200],
                                "source_count": len(source_texts)}) as span:
                span.set_output({
                    "claims": claims_data,
                    "total_claims": len(claims_data),
                    "verify_ms": round(verify_ms, 1),
                })

        # Step 5: Grounding / flags
        hallucinated_ids = [
            f"claim_{i}"
            for i, c in enumerate(verification_result.claims)
            if c.get("is_hallucination")
        ]

        flags_triggered = []
        if verification_result.hallucination_count > 0:
            flags_triggered.append("HALLUCINATION")
        if relevance_result.get("average_relevance", 1) < 0.5:
            flags_triggered.append("LOW_RELEVANCE")
        if verification_result.trust_score < 0.5:
            flags_triggered.append("LOW_TRUST")

        verdict = "PASS" if (
            verification_result.all_supported
            and verification_result.hallucination_count == 0
        ) else "FAIL"

        if t:
            with t.span("grounding", run_type="chain",
                        inputs={"claim_count": len(verification_result.claims)}) as span:
                span.set_output({
                    "grounding_score": verification_result.trust_score,
                    "hallucinated_claim_ids": hallucinated_ids,
                    "hallucination_count": verification_result.hallucination_count,
                    "flags_triggered": flags_triggered,
                    "verdict": verdict,
                })

        total_ms = (time.time() - pipeline_start) * 1000

        result = {
            "answer": answer,
            "sources": docs,
            "source_texts": source_texts,
            "relevance_result": relevance_result,
            "verification_result": {
                "trust_score": verification_result.trust_score,
                "all_supported": verification_result.all_supported,
                "claims": verification_result.claims,
                "flagged_claims": verification_result.flagged_claims,
                "hallucinations": verification_result.hallucinations,
                "hallucination_count": verification_result.hallucination_count,
                "latency_stats": verification_result.latency_stats,
            },
            "timing": {
                "retrieve_ms": retrieve_ms,
                "parallel_relevance_llm_ms": parallel_ms,
                "verify_ms": verify_ms,
                "total_ms": total_ms,
            },
            "verdict": verdict,
            "flags": flags_triggered,
        }

        if t and t.root_run:
            result["trace_id"] = t.root_run["trace_id"]

        return result

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


def create_parallel_pipeline(
    max_workers: int = 4,
    tracer: Optional["Tracer"] = None,
) -> ParallelPipeline:
    """Create a new parallel pipeline instance."""
    return ParallelPipeline(max_workers=max_workers, tracer=tracer)
