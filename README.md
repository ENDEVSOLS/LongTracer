<p align="center">
  <img src="https://raw.githubusercontent.com/ENDEVSOLS/LongTracer/main/assets/logo.png" alt="LongTracer Logo" width="320"/>
</p>


<p align="center"><strong>RAG hallucination detection, multi-project tracing, and pluggable backends — all batteries included.</strong></p>

<p align="center">
<a href="https://pypi.org/project/longtracer/"><img src="https://img.shields.io/pypi/v/longtracer" alt="PyPI Version"></a>
<a href="https://pepy.tech/project/longtracer"><img src="https://static.pepy.tech/badge/longtracer" alt="Total Downloads"></a>
<a href="https://pepy.tech/project/longtracer"><img src="https://static.pepy.tech/badge/longtracer/month" alt="Monthly Downloads"></a>
<a href="https://github.com/ENDEVSOLS/LongTracer/stargazers"><img src="https://img.shields.io/github/stars/ENDEVSOLS/LongTracer?style=flat" alt="GitHub Stars"></a>
<a href="https://github.com/ENDEVSOLS/LongTracer/actions/workflows/ci.yml"><img src="https://github.com/ENDEVSOLS/LongTracer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/pypi/pyversions/longtracer" alt="Python Versions">
<a href="https://github.com/ENDEVSOLS/LongTracer/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ENDEVSOLS/LongTracer" alt="License"></a>
</p>

<p align="center">
<a href="https://endevsols.github.io/LongTracer/"><strong>📖 Documentation</strong></a> &nbsp;·&nbsp;
<a href="https://endevsols.github.io/LongTracer/getting-started/quickstart/"><strong>Quick Start</strong></a> &nbsp;·&nbsp;
<a href="https://endevsols.github.io/LongTracer/api-reference/"><strong>API Reference</strong></a> &nbsp;·&nbsp;
<a href="https://github.com/ENDEVSOLS/LongTracer/blob/main/CHANGELOG.md"><strong>Changelog</strong></a>
</p>

Detect hallucinations in LLM-generated responses. LongTracer verifies every claim against source documents using hybrid STS + NLI, works with any RAG framework, and traces the full verification pipeline.

## Quick Start

```bash
pip install longtracer
```

### One-Liner & Batch API

```python
from longtracer import check, check_batch

# Verify a single response
result = check(
    "The Eiffel Tower is 330 meters tall and located in Berlin.",
    ["The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall."]
)

print(result.verdict)             # "FAIL"
print(result.trust_score)         # 0.0 - 1.0
print(result.hallucination_count) # 1 ("Berlin" contradicts "Paris")

# Verify in bulk
results = check_batch([
    {"response": "P is NP.", "sources": ["It is not known if P is NP."]},
    {"response": "Water boils at 100C.", "sources": ["Water boils at 100C."]}
])
```

### CLI (no Python needed)

```bash
longtracer check "The Eiffel Tower is in Berlin." "The Eiffel Tower is in Paris."
# ✗ FAIL  trust=0.50  hallucinations=1
```

### Full API

```python
from longtracer import CitationVerifier

verifier = CitationVerifier(cache=True)  # optional result caching
result = verifier.verify_parallel(
    response="The Eiffel Tower is 330 meters tall and located in Berlin.",
    sources=["The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall."]
)
```

No vector store dependency. No LLM dependency. Just strings in, verification out.

## How It Works

1. **Claim splitting** — LLM response is split into individual sentences/claims
2. **STS matching** — Fast bi-encoder (`all-MiniLM-L6-v2`) finds the best-matching source sentence for each claim
3. **NLI verification** — Cross-encoder (`nli-deberta-v3-xsmall`) classifies entailment/contradiction/neutral
4. **Verdict** — Trust score computed, hallucinations flagged

## Framework Adapters

### LangChain (3 lines)

```bash
pip install "longtracer[langchain]"
```

```python
from longtracer import LongTracer, instrument_langchain

LongTracer.init(verbose=True)
instrument_langchain(your_chain)
# Your chain.invoke() now auto-verifies every response
```

### LlamaIndex (3 lines)

```bash
pip install "longtracer[llamaindex]"
```

```python
from longtracer import LongTracer, instrument_llamaindex

LongTracer.init(verbose=True)
instrument_llamaindex(your_query_engine)
```

### Direct API (any framework)

```python
from longtracer.guard.verifier import CitationVerifier

verifier = CitationVerifier()
result = verifier.verify_parallel(
    response="LLM said this...",
    sources=["chunk 1 text", "chunk 2 text"],
    source_metadata=[{"source": "doc.pdf", "page": 1}, {"source": "doc.pdf", "page": 2}]
)
```

### Haystack v2

```bash
pip install "longtracer[haystack]"
```

```python
from longtracer.adapters.haystack_handler import LongTracerVerifier

pipeline.add_component("verifier", LongTracerVerifier())
pipeline.connect("generator.replies", "verifier.response")
pipeline.connect("retriever.documents", "verifier.documents")
```

### LangGraph Agents

```bash
pip install "longtracer[langgraph]"
```

```python
from longtracer import instrument_langgraph

handler = instrument_langgraph(graph)
result = agent.invoke(
    {"messages": [("user", "What is X?")]},
    config={"callbacks": [handler]}
)
```

### LangChain Agents

```python
from longtracer import instrument_langchain_agent

handler = instrument_langchain_agent(agent_executor)
result = agent_executor.invoke({"input": "What is X?"})
```

### Async Support

```python
result = await verifier.verify_parallel_async(response, sources)
```

Works with Haystack, custom pipelines, or any code that produces strings.

## Multi-Project Tracing

Track multiple RAG applications independently:

```python
from longtracer import LongTracer

LongTracer.init(project_name="chatbot-prod", backend="sqlite")

# Get project-specific tracers
chatbot = LongTracer.get_tracer("chatbot-prod")
search  = LongTracer.get_tracer("search-api")

# Each project's traces are tagged and filterable
chatbot.start_root(inputs={"query": "..."})
```

## Vector Store & LLM Agnostic

The SDK core takes plain `str` and `List[str]`. It does not depend on any vector store (Chroma, FAISS, Pinecone, Weaviate, Qdrant, pgvector) or any LLM provider (OpenAI, Anthropic, Ollama, Bedrock). Use whatever you want — LongTracer just verifies the output.

## Trace Storage Backends

```python
LongTracer.init(backend="sqlite")   # default — persists to ~/.longtracer/traces.db
LongTracer.init(backend="memory")   # in-memory, lost on restart
LongTracer.init(backend="mongo")    # production, distributed
```

| Backend | Install | Where traces live |
|---------|---------|-------------------|
| SQLite | built-in (default) | `~/.longtracer/traces.db` |
| Memory | built-in | RAM only, lost on restart |
| MongoDB | `pip install "longtracer[mongo]"` | MongoDB database |
| PostgreSQL | `pip install "longtracer[postgres]"` | PostgreSQL database |
| Redis | `pip install "longtracer[redis]"` | Redis key-value store |

## Viewing Traces

### CLI

```bash
longtracer view                        # list recent traces
longtracer view --last                 # view most recent
longtracer view --id <trace_id>        # view specific trace
longtracer view --project chatbot-prod # filter by project
longtracer view --export <trace_id>    # export to JSON
longtracer view --html <trace_id>      # export to HTML report
```

### Console (verbose mode)

```
[longtracer] span=retrieval    chunks=5
[longtracer] span=llm_call     answer_len=179
[longtracer] span=eval_claims  total=3 supported=2
[longtracer] span=grounding    score=0.67 verdict=FAIL
```

### HTML Report

```python
from longtracer.guard.trace_report import export_trace_html
export_trace_html(tracer, filepath="report.html")
```

Generates a standalone HTML file with trust scores, a summary stats bar, and clickable per-claim evidence diffs — viewable in any browser, zero external dependencies.

### JSON Export

```python
from longtracer.guard.trace_report import export_trace_json
export_trace_json(tracer, filepath="trace.json")
```

## Optional Dependencies

| Extra | Install | What it adds |
|-------|---------|-------------|
| `langchain` | `pip install "longtracer[langchain]"` | LangChain callback adapter |
| `llamaindex` | `pip install "longtracer[llamaindex]"` | LlamaIndex event adapter |
| `haystack` | `pip install "longtracer[haystack]"` | Haystack v2 component adapter |
| `langgraph` | `pip install "longtracer[langgraph]"` | LangGraph & LangChain agent tracing |
| `mongo` | `pip install "longtracer[mongo]"` | MongoDB trace backend |
| `postgres` | `pip install "longtracer[postgres]"` | PostgreSQL trace backend |
| `redis` | `pip install "longtracer[redis]"` | Redis trace backend |
| `chroma` | `pip install "longtracer[chroma]"` | ChromaDB + HuggingFace embeddings |
| `all` | `pip install "longtracer[all]"` | Everything |

## Configuration

Set project-level defaults effortlessly via `pyproject.toml` or environment variables (env vars override file).

### `pyproject.toml`
```toml
[tool.longtracer]
project = "my-rag-app"
backend = "sqlite"
threshold = 0.5
verbose = true
log_level = "INFO"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LONGTRACER_ENABLED` | `false` | Auto-enable with `LongTracer.auto()` |
| `LONGTRACER_VERBOSE` | `false` | Print per-span summaries |
| `LONGTRACER_LOG_LEVEL` | `INFO` | Python logging level |
| `LONGTRACER_PROJECT` | `longtracer`| Default project name |
| `TRACE_CACHE_BACKEND` | `sqlite` | Trace storage: sqlite, memory, mongo, postgres, redis |
| `MONGODB_URI` | — | MongoDB connection URI |
| `POSTGRES_HOST` | — | PostgreSQL host |
| `REDIS_HOST` | — | Redis host |

## Demo Application

The `examples/` directory contains a complete RAG demo using ChromaDB + Ollama. It is NOT part of the published PyPI package. See [examples/README.md](examples/README.md) for setup instructions.

## Documentation

Full documentation at **[endevsols.github.io/LongTracer](https://endevsols.github.io/LongTracer)**

- [Installation](https://endevsols.github.io/LongTracer/getting-started/installation/)
- [Quick Start](https://endevsols.github.io/LongTracer/getting-started/quickstart/)
- [How It Works](https://endevsols.github.io/LongTracer/how-it-works/)
- [LangChain Integration](https://endevsols.github.io/LongTracer/integrations/langchain/)
- [LangGraph & Agent Integration](https://endevsols.github.io/LongTracer/integrations/langgraph/)
- [LlamaIndex Integration](https://endevsols.github.io/LongTracer/integrations/llamaindex/)
- [Haystack Integration](https://endevsols.github.io/LongTracer/integrations/haystack/)
- [API Reference](https://endevsols.github.io/LongTracer/api-reference/)
- [CLI Reference](https://endevsols.github.io/LongTracer/cli/)

## License

MIT
