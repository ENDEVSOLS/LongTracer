# API Reference

## `check()` and `check_batch()` — One-Liners

The fastest way to verify responses:

```python
from longtracer import check, check_batch

# Single response
result = check(
    "The Eiffel Tower is in Berlin.",
    ["The Eiffel Tower is in Paris, France."]
)
print(result.verdict)  # "FAIL"

# Batch of responses (parallel)
results = check_batch([
    {"response": "Paris is in France.", "sources": ["Paris is in France."]},
    {"response": "Water boils at 50C.", "sources": ["Water boils at 100C."]}
])
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | `str` | LLM-generated response to verify |
| `sources` | `list[str]` | Source document chunks |
| `source_metadata` | `list[dict] \| None` | Optional metadata for each source |
| `threshold` | `float` | Verification threshold (default 0.5) |

**Returns:** `VerificationResult`

---

## CitationVerifier

The main verification class with full control.

```python
from longtracer import CitationVerifier

verifier = CitationVerifier(
    threshold=0.5,
    tracer=None,
    cache=False,   # enable in-memory result caching
)
```

### `verify_parallel(response, sources, source_metadata=None)`

Verify using parallel batch processing (recommended).

```python
result = verifier.verify_parallel(
    response="The Eiffel Tower is in Berlin.",
    sources=["The Eiffel Tower is in Paris, France."],
    source_metadata=[{"source": "geo.pdf", "page": 1}]
)
```

**Raises:** `TypeError` if inputs have wrong types.

---

### `verify_batch(items)`

Verify a batch of responses in parallel (returns a list of `VerificationResult`).

```python
results = verifier.verify_batch([
    {"response": "Paris is in France.", "sources": ["Paris is in France."]},
    {"response": "Water boils at 50C.", "sources": ["Water boils at 100C."]}
])
```

---

### `verify(response, sources, source_metadata=None)`

Sequential (non-parallel) verification. Same signature.

---

### `verify_parallel_async(response, sources, source_metadata=None)`

Async wrapper — runs verification in a thread pool executor.

```python
result = await verifier.verify_parallel_async(response, sources)
```

Use this in async frameworks (FastAPI, LangChain async, etc.).

---

### `verify_batch_async(items)`

Async wrapper for batch verification.

```python
results = await verifier.verify_batch_async(items)
```

---

### `verify_with_rag_result(rag_result)`

Convenience wrapper for RAG pipeline outputs.

---

### `cache_stats()`

Returns cache statistics when `cache=True`:

```python
verifier = CitationVerifier(cache=True)
verifier.verify_parallel(...)
print(verifier.cache_stats())
# {"enabled": True, "entries": 3}
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
    verdict: str                 # "PASS" or "FAIL" (auto-computed)
    summary: str                 # human-readable summary (auto-computed)
    latency_stats: Optional[Dict]  # STS/NLI timing breakdown
```

### Jupyter Notebook Display

In Jupyter, `VerificationResult` renders as a color-coded HTML table:

- 🟢 Green = supported claim
- 🔴 Red = hallucination
- 🟡 Yellow = unsupported (not hallucination)

```python
result = verifier.verify_parallel(response, sources)
result  # displays rich HTML in Jupyter
```

---

## Input Validation

All public `verify*` methods validate inputs and raise `TypeError` with helpful messages:

```python
verifier.verify_parallel(response=123, sources="not a list")
# TypeError: `response` must be a string, got int

verifier.verify_parallel(response="ok", sources=[123])
# TypeError: `sources[0]` must be a string, got int
```

---

## Configuration

### `pyproject.toml` (Recommended)

LongTracer automatically loads defaults from your `pyproject.toml`:

```toml
[tool.longtracer]
project = "my-rag-app"
backend = "sqlite"
threshold = 0.5
verbose = true
log_level = "INFO"
```

### `LongTracer.init()`

Override configuration programmatically:

```python
LongTracer.init(
    project_name="my-project",
    backend="sqlite",
    verbose=False,
    **backend_kwargs
)
```

### `LongTracer.get_tracer(project_name=None)`

Returns the `Tracer` instance for the given project.

### `LongTracer.auto()`

Auto-enable from environment variables (`LONGTRACER_ENABLED=true`).

### `LongTracer.reset()`

Reset all state. Useful in tests.

---

## Framework Adapters

### `instrument_langchain(chain, verbose=None)`

Attaches verification to a LangChain chain.

### `instrument_langgraph(graph, threshold=0.5, verbose=None)`

Creates a `LongTracerAgentHandler` for LangGraph agents. Returns the handler
to pass via `config={"callbacks": [handler]}`.

### `instrument_langchain_agent(agent_executor, threshold=0.5, verbose=None)`

Attaches verification to a LangChain `AgentExecutor`.

### `instrument_llamaindex(query_engine, verbose=None)`

Attaches verification to a LlamaIndex query engine.

### `instrument_haystack(pipeline, verbose=None)`

Adds `LongTracerVerifier` component to a Haystack v2 pipeline.

---

## LongTracerAgentHandler

For direct usage with any LangChain/LangGraph Runnable:

```python
from longtracer.adapters.langgraph_handler import LongTracerAgentHandler

handler = LongTracerAgentHandler(threshold=0.5, verbose=True)
result = runnable.invoke(input, config={"callbacks": [handler]})
```

---

## CLI

### `longtracer check`

```bash
longtracer check "response text" "source 1" "source 2"
longtracer check "response" "source" --json
longtracer check "response" "source" --threshold 0.7
```

### `longtracer view`

```bash
longtracer view                        # list recent traces
longtracer view --last                 # most recent
longtracer view --id <trace_id>        # specific trace
longtracer view --html <trace_id>      # export HTML report
longtracer view --export <trace_id>    # export JSON
longtracer view --project <name>       # filter by project
```
