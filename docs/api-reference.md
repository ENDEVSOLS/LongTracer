# API Reference

## CitationVerifier

The main verification class.

```python
from longtracer import CitationVerifier

verifier = CitationVerifier(threshold=0.5, tracer=None)
```

### Methods

#### `verify_parallel(response, sources, source_metadata=None)`

Verify an LLM response against source documents using parallel batch processing.

```python
result = verifier.verify_parallel(
    response="The Eiffel Tower is in Berlin.",
    sources=["The Eiffel Tower is in Paris, France."],
    source_metadata=[{"source": "geo.pdf", "page": 1}]
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | `str` | LLM-generated response to verify |
| `sources` | `List[str]` | Source document chunks |
| `source_metadata` | `List[Dict]` | Optional metadata for each source |

**Returns:** `VerificationResult`

---

#### `verify(response, sources, source_metadata=None)`

Sequential (non-parallel) verification. Same signature as `verify_parallel`.

---

#### `verify_with_rag_result(rag_result)`

Convenience wrapper for RAG pipeline outputs.

```python
result = verifier.verify_with_rag_result({
    "answer": "...",
    "source_documents": [
        {"page_content": "...", "metadata": {"source": "doc.pdf"}}
    ]
})
```

---

## VerificationResult

```python
@dataclass
class VerificationResult:
    trust_score: float           # 0.0–1.0
    claims: List[Dict]           # all claims with verification details
    flagged_claims: List[Dict]   # claims that failed
    hallucinations: List[Dict]   # claims flagged as hallucinations
    all_supported: bool          # True if all claims pass
    hallucination_count: int     # number of hallucinated claims
    latency_stats: Optional[Dict]  # STS/NLI timing breakdown
```

---

## LongTracer

Singleton for global configuration and multi-project tracing.

```python
from longtracer import LongTracer
```

### `LongTracer.init()`

```python
LongTracer.init(
    project_name="my-project",   # default: "longtracer"
    backend="sqlite",            # sqlite | memory | mongo | postgres | redis
    verbose=False,               # print per-span summaries
    **backend_kwargs             # passed to backend constructor
)
```

### `LongTracer.get_tracer(project_name=None)`

Returns the `Tracer` instance for the given project (or the default project).

### `LongTracer.auto()`

Auto-enable from environment variables (`LONGTRACER_ENABLED=true`).

### `LongTracer.reset()`

Reset all state. Useful in tests.

---

## instrument_langchain

```python
from longtracer import instrument_langchain

instrument_langchain(chain, verbose=None)
```

Attaches `CitationGuardCallbackHandler` to a LangChain chain.

---

## instrument_llamaindex

```python
from longtracer import instrument_llamaindex

instrument_llamaindex(query_engine, verbose=None)
```

Attaches `CitationGuardLlamaIndexHandler` to a LlamaIndex query engine.

---

## ParallelPipeline

For advanced usage with custom RAG pipelines:

```python
from longtracer.guard.parallel_pipeline import ParallelPipeline

pipeline = ParallelPipeline(max_workers=4, tracer=tracer)
result = pipeline.run(
    query="...",
    retriever=retriever,
    verifier=verifier,
    relevance_scorer=scorer,
    k=10
)
```

---

## ContextRelevanceScorer

Score how relevant retrieved chunks are to the query:

```python
from longtracer.guard.context_relevance import ContextRelevanceScorer

scorer = ContextRelevanceScorer()
result = scorer.score(query="...", chunks=["chunk1", "chunk2"])

print(result["average_relevance"])  # 0.0–1.0
print(result["threshold_pass"])     # bool
print(result["chunk_rankings"])     # sorted by relevance
```
