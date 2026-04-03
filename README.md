<p align="center">
  <img src="assets/logo.png" alt="LongTracer Logo" width="320"/>
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

Detect hallucinations in LLM-generated responses. LongTracer verifies every claim against source documents using hybrid STS + NLI, works with any RAG framework, and traces the full verification pipeline.

## Quick Start

```bash
pip install longtracer
```

```python
from longtracer import CitationVerifier

verifier = CitationVerifier()
result = verifier.verify_parallel(
    response="The Eiffel Tower is 330 meters tall and located in Berlin.",
    sources=["The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall."]
)

print(result.trust_score)         # 0.0 - 1.0
print(result.hallucination_count) # 1 ("Berlin" contradicts "Paris")
print(result.all_supported)       # False
```

That's it. No vector store dependency, no LLM dependency. Just strings in, verification out.

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

Generates a self-contained HTML file with trust score, per-claim results, timing breakdown — viewable in any browser, no external dependencies.

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
| `mongo` | `pip install "longtracer[mongo]"` | MongoDB trace backend |
| `postgres` | `pip install "longtracer[postgres]"` | PostgreSQL trace backend |
| `redis` | `pip install "longtracer[redis]"` | Redis trace backend |
| `chroma` | `pip install "longtracer[chroma]"` | ChromaDB + HuggingFace embeddings |
| `all` | `pip install "longtracer[all]"` | Everything |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LONGTRACER_ENABLED` | `false` | Auto-enable with `LongTracer.auto()` |
| `LONGTRACER_VERBOSE` | `false` | Print per-span summaries |
| `LONGTRACER_LOG_LEVEL` | `INFO` | Python logging level |
| `TRACE_CACHE_BACKEND` | `sqlite` | Trace storage: sqlite, memory, mongo, postgres, redis |
| `MONGODB_URI` | — | MongoDB connection URI |
| `POSTGRES_HOST` | — | PostgreSQL host |
| `REDIS_HOST` | — | Redis host |
| `TRACE_PROJECT` | `longtracer` | Default project name |

## Demo Application

The `examples/` directory contains a complete RAG demo using ChromaDB + Ollama. It is NOT part of the published PyPI package. See [examples/README.md](examples/README.md) for setup instructions.

## License

MIT
