# LangChain Integration

LongTracer hooks into LangChain's callback system to automatically capture retrieval, LLM, and verification spans — no changes to your chain required.

## Install

```bash
pip install "longtracer[langchain]"
```

## Usage

```python
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from longtracer import LongTracer, instrument_langchain

# 1. Init LongTracer
LongTracer.init(verbose=True)

# 2. Build your chain as normal
chain = RetrievalQA.from_chain_type(
    llm=your_llm,
    retriever=your_vectorstore.as_retriever()
)

# 3. Instrument — one line
instrument_langchain(chain)

# 4. Use your chain as normal — verification happens automatically
result = chain.invoke({"query": "What is the capital of France?"})
```

## What Gets Captured

| Span | What it records |
|------|----------------|
| `retrieval` | Retrieved chunks, count, latency |
| `llm_prep` | Prompt text, context length |
| `llm_call` | LLM answer, model name, latency |
| `eval_claims` | Per-claim verification results |
| `grounding` | Trust score, hallucination count, verdict |

## Viewing Results

```bash
longtracer view --last
longtracer view --html <trace_id>
```

## Notes

- Works with any LangChain chain that uses a retriever (`RetrievalQA`, `ConversationalRetrievalChain`, custom chains)
- Verification is triggered at the end of the root chain, after the LLM has responded
- If no chunks are retrieved, verification is skipped gracefully
- A failing verification never crashes your chain — all errors are logged as warnings
