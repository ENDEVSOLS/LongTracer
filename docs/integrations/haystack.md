# Haystack Integration

LongTracer provides a native Haystack v2 `@component` for verifying LLM responses in any Haystack pipeline.

## Install

```bash
pip install "longtracer[haystack]"
```

## Usage as a Component

```python
from haystack import Pipeline, Document
from longtracer.adapters.haystack_handler import LongTracerVerifier

# Create the verifier component
verifier = LongTracerVerifier(threshold=0.5)

# Add to your pipeline
pipeline = Pipeline()
pipeline.add_component("retriever", your_retriever)
pipeline.add_component("generator", your_generator)
pipeline.add_component("verifier", verifier)

# Connect: generator output → verifier
pipeline.connect("retriever.documents", "verifier.documents")
pipeline.connect("generator.replies", "verifier.response")
```

## Quick Instrument

```python
from longtracer import instrument_haystack

pipeline = Pipeline()
# ... add your components ...
instrument_haystack(pipeline)
```

This adds a `longtracer_verifier` component to the pipeline. You still need to connect it to your generator and retriever outputs.

## Component Inputs / Outputs

| Input | Type | Description |
|-------|------|-------------|
| `response` | `str` | LLM-generated text to verify |
| `documents` | `List[Document]` | Retrieved Haystack Documents |

| Output | Type | Description |
|--------|------|-------------|
| `response` | `str` | Pass-through of original response |
| `trust_score` | `float` | 0.0–1.0 |
| `verdict` | `str` | "PASS" or "FAIL" |
| `summary` | `str` | Human-readable summary |
| `claims` | `list` | Per-claim verification results |
| `hallucination_count` | `int` | Number of hallucinated claims |

## With Tracing

```python
from longtracer import LongTracer

LongTracer.init(verbose=True, backend="sqlite")

# The verifier component automatically picks up the active tracer
verifier = LongTracerVerifier()
```

## Notes

- Models are loaded on first `run()` call (or explicitly via `warm_up()`)
- A failing verification never crashes your pipeline — returns `verdict="ERROR"` with the error message
- Works with any Haystack v2 pipeline that produces documents and text responses
