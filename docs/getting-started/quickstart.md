# Quick Start

## 0. One-Liner Check (Fastest)

No setup, no class instantiation:

```python
from longtracer import check, check_batch

# Single check
result = check(
    "The Eiffel Tower is 330 meters tall and located in Berlin.",
    ["The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall."]
)

print(result.verdict)   # "FAIL"
print(result.summary)   # "1/1 claims supported, 1 hallucination(s) detected."

# Batch check
results = check_batch([
    {"response": "Water boils at 100C.", "sources": ["At 1 atm, water boiling point is 100C."]},
    {"response": "Paris is in Germany.", "sources": ["Paris is in France."]}
])
```

Or from the command line:

```bash
longtracer check "The Eiffel Tower is in Berlin." "The Eiffel Tower is in Paris."
# ✗ FAIL  trust=0.50  hallucinations=1
```

## 1. Direct Verification

The simplest usage — no setup required:

```python
from longtracer import CitationVerifier

verifier = CitationVerifier()

result = verifier.verify_parallel(
    response="The Eiffel Tower is 330 meters tall and located in Berlin.",
    sources=[
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "It stands 330 metres tall on the Champ de Mars."
    ]
)

print(f"Trust score:   {result.trust_score:.2f}")
print(f"Hallucinations: {result.hallucination_count}")
print(f"All supported:  {result.all_supported}")

for claim in result.claims:
    status = "✓" if claim["supported"] else "✗"
    print(f"  {status} {claim['claim'][:80]}")
```

---

## 2. With Source Metadata

Pass metadata to trace which document each claim came from:

```python
result = verifier.verify_parallel(
    response="Paris is the capital of France and has a population of 2 million.",
    sources=["Paris is the capital and most populous city of France."],
    source_metadata=[{"source": "geography.pdf", "page": 12}]
)

for claim in result.claims:
    print(claim["claim"])
    print(f"  Best source: {claim['best_source_metadata']}")
    print(f"  Score: {claim['score']:.3f}")
```

---

## 3. With Tracing Enabled

Enable tracing to persist verification results across runs:

```python
from longtracer import LongTracer, CitationVerifier

LongTracer.init(backend="sqlite", verbose=True)
tracer = LongTracer.get_tracer()

verifier = CitationVerifier(tracer=tracer)

with tracer.span("my_pipeline"):
    result = verifier.verify_parallel(
        response="...",
        sources=["..."]
    )

# View traces later
# longtracer view --last
```

---

## 4. Multi-Project Tracing

Track multiple RAG applications independently:

```python
from longtracer import LongTracer

LongTracer.init(project_name="chatbot-prod", backend="sqlite")
LongTracer.init(project_name="search-api", backend="sqlite")

chatbot = LongTracer.get_tracer("chatbot-prod")
search  = LongTracer.get_tracer("search-api")

# Each project's traces are stored and filterable separately
```

---

## 5. View Traces via CLI

```bash
longtracer view                        # list recent traces
longtracer view --last                 # view most recent
longtracer view --project chatbot-prod # filter by project
longtracer view --html <trace_id>      # export HTML report
```

---

## Next Steps

- [LangChain integration](../integrations/langchain.md)
- [LlamaIndex integration](../integrations/llamaindex.md)
- [Haystack integration](../integrations/haystack.md)
- [Trace storage backends](../backends.md)
- [CLI reference](../cli.md)
