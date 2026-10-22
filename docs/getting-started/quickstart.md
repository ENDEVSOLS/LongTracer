# Quick Start

## 0. Start with Typed Evaluation Results

Requires Python 3.11 or newer. Install LongTracer, prepare the model cache, and inspect your installation:

```bash
pip install longtracer
longtracer models prepare
longtracer doctor
```

The first model preparation requires network access. Cached models are reused on subsequent runs, although loading them still takes time.

Use `check_case()` for new evaluation and gating logic:

```python
from longtracer import check_case

case = check_case(
    "The Eiffel Tower is located in Berlin, Germany.",
    ["The Eiffel Tower is located in Paris, France."],
    timeout=5.0,
)

print(case.execution.value)
print(case.availability.value)
for claim in case.claims:
    print(claim.assessment.value, claim.reason.value)

if case.quality_gate.value == "PASS":
    print("The response passes the evaluation policy.")
elif case.quality_gate.value == "FAIL":
    print("Review the claim assessments and supplied evidence.")
else:
    print("Evaluation did not complete; inspect the reason and retry or alert.")
```

`CaseResult` separates evaluator execution, assessment availability, individual claim assessments, and the final quality gate. Model loading or evaluation failures produce an `INDETERMINATE` gate, with a reason describing the failure.

### Empty Responses

With models available, an empty answer explicitly fails the quality gate:

```python
case = check_case("", ["The Eiffel Tower is located in Paris, France."])
print(case.availability.value)  # NO_ASSESSABLE_CLAIMS
print(case.reason.value)        # EMPTY_RESPONSE
print(case.quality_gate.value)  # FAIL
```

A recognized refusal-only response may pass the refusal policy while remaining unassessed for factual grounding. Inspect `availability` as well as `quality_gate` when your application requires assessed factual claims.

LongTracer checks agreement with supplied sources, not whether those sources are true. See the [Result States guide](../result-states.md) for policy details, reason codes, and limitations. Timeout values must be finite and positive; expiry returns a result without terminating background evaluation.

### Legacy One-Liner and Batch APIs

Existing `check()` and `check_batch()` APIs retain their previous behavior and return `VerificationResult`. Their legacy `trust_score` measures similarity, and `check()` can still return `PASS` for an empty answer. Use `check_case()` for the typed result semantics above.

```python
from longtracer import check, check_batch

result = check(
    "The Eiffel Tower is located in Berlin, Germany.",
    ["The Eiffel Tower is located in Paris, France."],
)
print(result.verdict)
print(result.summary)

results = check_batch([
    {"response": "Water boils at 100C.", "sources": ["At 1 atm, water boiling point is 100C."]},
    {"response": "Paris is in Germany.", "sources": ["Paris is in France."]},
])
```

The command-line check also uses legacy result semantics:

```bash
longtracer check "The Eiffel Tower is in Berlin." "The Eiffel Tower is in Paris."
```

## 1. Direct Verification

Use `CitationVerifier` directly when you need to reuse an evaluator instance. These methods return the legacy result type:

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
