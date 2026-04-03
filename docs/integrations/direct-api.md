# Direct API

Use LongTracer with any framework — Haystack, custom pipelines, or any code that produces strings.

## Basic Usage

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

print(result.trust_score)          # 0.5
print(result.hallucination_count)  # 1
print(result.all_supported)        # False
```

## With Source Metadata

```python
result = verifier.verify_parallel(
    response="...",
    sources=["chunk 1", "chunk 2"],
    source_metadata=[
        {"source": "doc.pdf", "page": 1},
        {"source": "doc.pdf", "page": 2}
    ]
)

for claim in result.claims:
    print(f"Claim: {claim['claim']}")
    print(f"  Supported: {claim['supported']}")
    print(f"  Score: {claim['score']:.3f}")
    print(f"  Source: {claim['best_source_metadata']}")
    print(f"  Hallucination: {claim['is_hallucination']}")
```

## VerificationResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `trust_score` | `float` | 0.0–1.0, fraction of supported claims |
| `all_supported` | `bool` | True if all claims are supported |
| `hallucination_count` | `int` | Number of hallucinated claims |
| `claims` | `List[Dict]` | Per-claim results (see below) |
| `flagged_claims` | `List[Dict]` | Claims that failed verification |
| `hallucinations` | `List[Dict]` | Claims flagged as hallucinations |
| `latency_stats` | `Dict` | STS/NLI call counts and timing |

## Per-Claim Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | The claim text |
| `supported` | `bool` | Whether the claim is supported |
| `score` | `float` | Final verification score |
| `best_score` | `float` | Best STS similarity score |
| `entailment_score` | `float` | NLI entailment probability |
| `contradiction_score` | `float` | NLI contradiction probability |
| `nli_ran` | `bool` | Whether NLI was executed (gated by STS) |
| `best_source` | `str` | Best matching source sentence |
| `best_source_index` | `int` | Index into sources list |
| `best_source_metadata` | `Dict` | Metadata of best source |
| `is_hallucination` | `bool` | Whether flagged as hallucination |
| `is_meta_statement` | `bool` | Whether it's an honest uncertainty statement |

## With RAG Result Dict

```python
rag_result = {
    "answer": "...",
    "source_documents": [
        {"page_content": "...", "metadata": {"source": "doc.pdf"}}
    ]
}

result = verifier.verify_with_rag_result(rag_result)
```

## Threshold Configuration

```python
# Default threshold is 0.5 — adjust for stricter/looser verification
verifier = CitationVerifier(threshold=0.7)
```
