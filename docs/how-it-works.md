# How It Works

LongTracer uses a two-stage hybrid pipeline to verify each claim in an LLM response against source documents.

---

## Pipeline Overview

```
LLM Response
     │
     ▼
┌─────────────────┐
│  Claim Splitter │  Split response into individual sentences/claims
└────────┬────────┘
         │  ["claim 1", "claim 2", ...]
         ▼
┌─────────────────────────────────────────────────────┐
│  For each claim:                                    │
│                                                     │
│  Step A: STS (Bi-Encoder)                           │
│  ┌──────────────────────────────────────────────┐   │
│  │ all-MiniLM-L6-v2                             │   │
│  │ Encode claim + all source sentences          │   │
│  │ Cosine similarity → best matching source     │   │
│  │ O(N+M) — fast, ~10ms                         │   │
│  └──────────────────┬───────────────────────────┘   │
│                     │ STS score ≥ 0.25?              │
│                     ▼                               │
│  Step B: NLI (Cross-Encoder)  [gated]               │
│  ┌──────────────────────────────────────────────┐   │
│  │ nli-deberta-v3-xsmall                        │   │
│  │ (claim, best_source) → [contra, neutral, ent]│   │
│  │ O(1) per claim — accurate, ~150ms            │   │
│  └──────────────────┬───────────────────────────┘   │
│                     │                               │
│                     ▼                               │
│  Verdict: supported / hallucination / neutral       │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Trust Score    │  supported_claims / total_claims
└─────────────────┘
```

---

## Stage 1: Claim Splitting

The LLM response is split into individual sentences using a regex-based splitter that:

- Protects decimal numbers (`98.5` is not split at the `.`)
- Protects abbreviations (`Dr.`, `Inc.`, `e.g.`) from triggering splits
- Filters out very short fragments (< 15 chars)
- Detects **meta-statements** — honest uncertainty phrases like "the documents do not contain..." (never flagged as hallucinations)
- Detects **hallucination patterns** — outside-knowledge signals like "based on my knowledge..." (flagged regardless of NLI)

---

## Stage 2A: STS Evidence Selection

For each claim, the bi-encoder (`all-MiniLM-L6-v2`) computes cosine similarity between the claim embedding and every source sentence embedding.

- Complexity: **O(N + M)** where N = claim tokens, M = total source tokens
- Typical latency: **< 10ms per claim**
- Output: best-matching source sentence + similarity score

**Gating:** If the best STS score is below `0.25`, NLI is skipped entirely. This avoids wasting compute on claims that have no plausible source match.

---

## Stage 2B: NLI Verification

The cross-encoder (`nli-deberta-v3-xsmall`) takes the `(claim, best_source_sentence)` pair and outputs three scores:

| Label | Meaning |
|-------|---------|
| `entailment` | Source supports the claim |
| `neutral` | Source neither supports nor contradicts |
| `contradiction` | Source contradicts the claim → hallucination |

- Complexity: **O(1)** per claim (single pair)
- Typical latency: **~150ms per claim**

---

## Hallucination Detection Logic

A claim is flagged as a hallucination if **any** of these conditions are true:

1. `contradiction_score > 0.5` (NLI says it's contradicted)
2. Low STS score + hallucination pattern detected + NLI didn't rescue it
3. Claim contains explicit outside-knowledge signals (`"based on my knowledge..."`)

Meta-statements are **never** flagged as hallucinations regardless of scores.

---

## Trust Score

```
trust_score = supported_claims / total_claims
```

Where `supported_claims` = claims with `entailment_score > threshold` and no contradiction.

- `1.0` = all claims supported
- `0.0` = no claims supported (or no sources provided)

---

## Parallel Pipeline

When using `ParallelPipeline`, context relevance scoring runs **in parallel** with LLM generation:

```
Retrieve docs
     │
     ├──────────────────────────────┐
     │                              │
     ▼                              ▼
Context Relevance Scoring      LLM Generation
(bi-encoder cosine sim)        (your LLM call)
     │                              │
     └──────────────┬───────────────┘
                    │
                    ▼
           Batch Claim Verification
                    │
                    ▼
              Verdict + Flags
```

This means relevance scoring adds **zero latency** to the pipeline — it runs while the LLM is thinking.
