# ADR 0001: Result-State Semantics & Versioned Contracts

- **Status:** Accepted
- **Date:** 2026-09-28
- **Context Milestone:** Phase 1 — v0.3.0 (Queue 3)
- **Deciders:** LongTracer Core Team

---

## Context

In LongTracer `0.2.0`, verification results were represented primarily via a single float score (`trust_score: 0.0 - 1.0`) and a binary verdict (`verdict: "PASS" | "FAIL"`).

This design created several critical correctness and developer-experience issues:

1. **Silent False Confidence on Empty Outputs:** If an application generated an empty answer or produced zero assessable claims, `all_supported` evaluated to `True` and `trust_score` was reported as `1.0` ("PASS").
2. **Conflation of System Errors with Quality Failures:** If an evaluator timed out, ran out of memory, or lacked model weights, it was impossible to distinguish an infrastructure failure from an unsupported factual claim.
3. **Ambiguity Between Contradiction and Missing Evidence:** A claim that contradicts evidence is materially different from a claim for which no retrieved evidence was supplied. Conflating both as generic "hallucinations" led to misleading reports.
4. **Silent Resolution of Conflicting Sources:** When retrieved documents disagreed, previous models were forced to either pick one source as truth or arbitrarily penalize the claim.
5. **No Versioning on Exported Artifacts:** Exported JSON/HTML traces had no explicit schema version, making regression suites fragile across releases.

---

## Decision

We introduce an explicit, multi-layered result state model and versioned contract layer under `longtracer.contracts`.

### 1. Multi-Layer Result States

Instead of collapsing all outcomes into a single float, verification evaluates four distinct, orthogonal layers:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Execution Status                                         │
│    SUCCESS | ERROR | TIMEOUT | SKIPPED | CANCELLED          │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Assessment Availability                                  │
│    ASSESSED | NO_ASSESSABLE_CLAIMS | NOT_EVALUATED           │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Claim Assessment (per claim)                             │
│    SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE |       │
│    CONFLICTING_SOURCES                                      │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Quality Gate                                             │
│    PASS | FAIL | INDETERMINATE                              │
└─────────────────────────────────────────────────────────────┘
```

#### Layer Rules:
- **ExecutionStatus.ERROR / TIMEOUT:** Results in `QualityGate.INDETERMINATE`. Infrastructure errors must never masquerade as a successful pass or a legitimate evaluation failure.
- **AssessmentAvailability.NO_ASSESSABLE_CLAIMS:** If an application was expected to produce an answer but gave none, the policy gate evaluates to `QualityGate.FAIL`, not `PASS`.
- **ClaimAssessment.INSUFFICIENT_EVIDENCE:** The claim is unsupported by provided evidence, but is not labeled objectively false in the world.
- **ClaimAssessment.CONFLICTING_SOURCES:** Assigned when multiple provided evidence passages directly contradict each other regarding the claim.

### 2. Versioned Contracts Layer (`longtracer.contracts`)

All structured data models inherit from Pydantic `BaseModel` and carry an explicit `schema_version = "1"`.

The contracts package provides:
- **`evidence.py` (`SourceEvidence`):** Normalized evidence passage with deterministic SHA-256 `text_hash` and source identity.
- **`result.py` (`ClaimResult`, `CaseResult`):** Explicit typed verification results.
- **`case.py` (`TestCase`, `ApplicationOutput`):** Portable representation for regression test suites and app runner outputs.
- **`run.py` (`RunManifest`):** Complete provenance metadata (dataset digest, evaluator fingerprint, timestamps).
- **`review.py` (`ReviewRecord`, `BaselineRecord`):** Human review states (`UNREVIEWED`, `CONFIRMED_BUG`, `EXPECTED_BEHAVIOR`, `ARCHIVED`) and approved baselines.

### 3. Backward Compatibility

To preserve 100% backward compatibility for existing code, pipelines, and framework adapters (LangChain, LlamaIndex, Haystack, EvalPort):
- The legacy `VerificationResult` dataclass remains fully supported in `longtracer.guard.verifier`.
- A bidirectional adapter (`LegacyVerificationAdapter`) allows lossless translation between `VerificationResult` and the new `CaseResult`.
- `CaseResult.trust_score` is retained as a calculated legacy compatibility field with documented semantics.

---

## Consequences

### Positive
- Gating in CI is deterministic: zero risk of empty responses passing CI.
- Infrastructure crashes yield `INDETERMINATE` exit codes (exit 2) instead of quality failure (exit 1) or false success (exit 0).
- Granular claim classifications (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_SOURCES`) enable precise root-cause analysis.
- Portable JSON Schema export enables tooling across languages and external review dashboards.

### Negative / Trade-offs
- More types and fields to learn; mitigated by clear documentation and convenient helper methods.
- Slight serialization overhead compared to raw dicts; mitigated by Pydantic v2 Rust core performance.
