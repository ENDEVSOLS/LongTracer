# LongTracer Product Roadmap

> **Mission:** Fast, truthful, and local RAG verification guardrails. Detect hallucinations and groundedness contradictions in LLM responses using hybrid STS + NLI without paid LLM-as-judge dependencies.

---

## Release Train

| Version | Status | Focus | Delivered / Planned Scope |
|---|---|---|---|
| **v0.1.x** | Shipped | SDK & Core Verification | Fast hybrid STS + NLI verification engine, multi-project tracing, memory/SQLite/Mongo backends. |
| **v0.1.5** | Shipped | Developer Experience | Async verifiers, rich notebook displays, input validation, CLI commands. |
| **v0.1.6** | Shipped | Framework Integrations | LangChain, LlamaIndex, Haystack, LangGraph, CrewAI, AutoGen adapters. |
| **v0.2.0** | Shipped | Observability & Analytics | Built-in analytics dashboard, OpenTelemetry exporter, Slack/webhook alerting. |
| **v0.3.0** | **Release candidate** | **Trustworthy Evaluation** | Typed `CaseResult` alongside the legacy result (`verify_case()`, `check_case()`): execution / availability / claim assessment / quality gate with `ReasonCode`s; empty answers can no longer pass the new gate; typed `EvaluatorError` hierarchy; published JSON Schema; opt-in experimental conflicting-sources detection; benchmark dataset (drafted, pending human review) and harness. |
| **v0.4.0** | Planned | Regression Workflow | JSONL regression datasets, `longtracer compare`, deterministic digests, pytest runner integration. |
| **v0.5.0** | Planned | Reviewed Feedback Loop | Human-in-the-loop trace review, `longtracer case from-trace`, golden fixture promotion, PII redaction. |
| **v0.6.0** | Planned | Evidence Hardening | Citation span attachment, OpenTelemetry GenAI semantic conventions, resource limits. |
| **v1.0.0** | Planned | Production Stability | Long-term API freeze, migration guide, production pilot hardening. |

---

## Detailed Version Objectives

### v0.3.0 — Trustworthy Evaluation (Release candidate)
- **Typed results:** `CaseResult` reports execution, availability, per-claim assessment
  (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_SOURCES`) with reason codes,
  and a quality gate computed separately. It is returned alongside the unchanged legacy `VerificationResult`.
- **Headline fix:** empty, whitespace and too-short answers report `NO_ASSESSABLE_CLAIMS` and cannot
  pass the new gate. The legacy `trust_score` keeps its documented meaning.
- **No silent success:** model unavailable, scoring failure, timeout, cancellation and invalid input
  become typed `ERROR` / `TIMEOUT` / `CANCELLED` states with an `INDETERMINATE` gate.
- **Typed errors:** `EvaluatorError` hierarchy, compatible with existing `ImportError` handlers.
- **Published schema:** `docs/schema/case_result.v1.json` with a compatibility test.
- **Benchmark:** 140-case, 14-category dataset and an opt-in harness reporting denominators and Wilson
  intervals. Labels are drafted and awaiting human review; results are provisional.
- **Not in this release:** threshold tuning, storage of `CaseResult`, and a fix for the legacy NLI
  label-order issue (documented as a known issue).

---

### v0.4.0 — Regression Workflow (Planned)

- JSONL dataset loader for offline evaluation workflows.
- `TestCase` and `ApplicationOutput` contract runner implementation.
- `longtracer compare` CLI affordance for comparing evaluation runs.
- `RunManifest` population capturing environment fingerprints, model digests, and policy states.
- Native `pytest` test-runner plugin for CI assertion against golden baselines.

---

### v0.5.0 — Reviewed Feedback Loop (Planned)

- Promotion of production failure traces to regression fixtures via `longtracer case from-trace`.
- `ReviewRecord` and `BaselineRecord` state lifecycle management.
- Automatic sanitization and PII redaction for production-to-test promotion.
- Near-duplicate detection to prevent redundant regression test cases.

---

### v0.6.0 — Evidence Hardening (Planned)

- Citation span attachment and citation coverage metrics.
- OpenTelemetry GenAI semantic-convention full alignment.
- Runtime hardening: process sandboxing, hard CPU time budgets, and memory bounds.

---

### v1.0.0 — Production Stability (Planned)

- Long-term public API stability guarantee.
- Migration guides from legacy v0.1/v0.2 verification paths to contracts.
- Pilot-driven enterprise hardening.

---

## Architectural Principles

1. **Local & Private:** Zero hidden cloud telemetry; zero paid LLM-as-judge network dependencies.
2. **Honesty Over Optimism:** Evaluator failures, timeouts, and unassessed responses never produce false successes.
3. **Lossless Compatibility:** Modern layers wrap and adapt existing legacy surfaces without breaking changes.
