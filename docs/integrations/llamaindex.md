# LlamaIndex Integration

LongTracer hooks into LlamaIndex's event system to automatically verify responses from any query engine.

## Install

```bash
pip install "longtracer[llamaindex]"
```

## Usage

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from longtracer import LongTracer, instrument_llamaindex

# 1. Init LongTracer
LongTracer.init(verbose=True)

# 2. Build your query engine as normal
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 3. Instrument — one line
instrument_llamaindex(query_engine)

# 4. Use your query engine as normal
response = query_engine.query("What is the capital of France?")
```

## What Gets Captured

| Span | What it records |
|------|----------------|
| `retrieval` | Retrieved nodes, scores, latency |
| `llm_call` | LLM response, latency |
| `eval_claims` | Per-claim verification results |
| `grounding` | Trust score, hallucination count, verdict |

## Viewing Results

```bash
longtracer view --last
longtracer view --html <trace_id>
```

## Notes

- Works with any LlamaIndex query engine that uses a retriever
- Handles both `node.text` and `node.get_content()` node formats
- Verification triggered at `SYNTHESIZE` or `QUERY` event end
- A failing verification never crashes your query engine
