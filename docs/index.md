# LongTracer

<p align="center">
  <img src="https://raw.githubusercontent.com/ENDEVSOLS/LongTracer/main/assets/logo.png" alt="LongTracer" width="280"/>
</p>

**RAG hallucination detection, multi-project tracing, and pluggable backends — all batteries included.**

[![PyPI](https://img.shields.io/pypi/v/longtracer)](https://pypi.org/project/longtracer/)
[![CI](https://github.com/ENDEVSOLS/LongTracer/actions/workflows/ci.yml/badge.svg)](https://github.com/ENDEVSOLS/LongTracer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/longtracer)](https://pypi.org/project/longtracer/)
[![License](https://img.shields.io/github/license/ENDEVSOLS/LongTracer)](https://github.com/ENDEVSOLS/LongTracer/blob/main/LICENSE)

---

## What is LongTracer?

LongTracer is an open-source Python SDK that detects hallucinations in LLM-generated responses. It verifies every claim in an LLM output against your source documents using a two-stage hybrid pipeline:

1. **STS (Semantic Textual Similarity)** — fast bi-encoder finds the best-matching source sentence for each claim
2. **NLI (Natural Language Inference)** — cross-encoder classifies entailment / contradiction / neutral

The result is a `trust_score` (0.0–1.0), a list of flagged claims, and a full trace of the verification pipeline.

---

## Why LongTracer?

| Problem | LongTracer's answer |
|---------|---------------------|
| LLMs hallucinate facts not in your documents | Detects contradictions at the claim level |
| Hard to debug which claim failed | Full trace with per-claim evidence mapping |
| Tied to a specific vector store or LLM | Works with any RAG framework — just strings in |
| Verification adds too much latency | Parallel pipeline: relevance scoring runs alongside LLM generation |
| Need to track verification across projects | Multi-project tracing with pluggable storage backends |

---

## Install

```bash
pip install longtracer
```

## 30-Second Example

```python
from longtracer import CitationVerifier

verifier = CitationVerifier()
result = verifier.verify_parallel(
    response="The Eiffel Tower is 330 meters tall and located in Berlin.",
    sources=["The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall."]
)

print(result.trust_score)         # 0.5
print(result.hallucination_count) # 1  ("Berlin" contradicts "Paris")
print(result.all_supported)       # False
```

No vector store dependency. No LLM dependency. Just strings in, verification out.

---

## Next Steps

- [Installation guide](getting-started/installation.md) — all install options including extras
- [Quick Start](getting-started/quickstart.md) — working examples in 5 minutes
- [How It Works](how-it-works.md) — deep dive into the STS + NLI pipeline
- [Integrations](integrations/langchain.md) — LangChain, LlamaIndex, direct API
