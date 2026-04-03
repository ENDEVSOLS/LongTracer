# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip

## Core Install

```bash
pip install longtracer
```

This installs the core SDK with:

- `CitationVerifier` — claim-level hallucination detection
- `LongTracer` — singleton for multi-project tracing
- SQLite and in-memory trace backends (no extra dependencies)
- `longtracer` CLI command

Models are downloaded automatically on first use (~100MB total):

- `all-MiniLM-L6-v2` — fast bi-encoder for STS
- `nli-deberta-v3-xsmall` — cross-encoder for NLI

---

## Optional Extras

Install only what you need:

```bash
# LangChain integration
pip install "longtracer[langchain]"

# LlamaIndex integration
pip install "longtracer[llamaindex]"

# MongoDB trace backend
pip install "longtracer[mongo]"

# PostgreSQL trace backend
pip install "longtracer[postgres]"

# Redis trace backend
pip install "longtracer[redis]"

# ChromaDB + HuggingFace embeddings (for the RAG demo)
pip install "longtracer[chroma]"

# Everything
pip install "longtracer[all]"
```

---

## Verify Installation

```bash
python -c "from longtracer import CitationVerifier; print('OK')"
longtracer --help
```

---

## Upgrading

```bash
pip install --upgrade longtracer
```
