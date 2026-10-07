# Result States

LongTracer v0.3.0 adds a typed result, `CaseResult`, next to the existing `VerificationResult`. It reports what happened in four separate layers and never reports success when nothing was actually assessed.

!!! important "Evidence grounding, not objective truth"
    LongTracer checks whether a response is **consistent with the sources you supply**. It does not decide whether those sources are true. If a source says *"The moon is made of cheese"* and the answer repeats it, the claim is `SUPPORTED`.

## Getting a `CaseResult`

The new result comes from **new methods**. Existing methods are unchanged and still return `VerificationResult`.

```python
from longtracer import check_case, CitationVerifier

# One-liner. Model loading happens inside, so a missing model is reported, not raised.
case = check_case("Water boils at 100 °C at sea level.", ["Water boils at 100 °C at sea-level pressure."])

# Or on a verifier you already have
verifier = CitationVerifier()
case = verifier.verify_case(response, sources, source_metadata=None,
                            case_id=None, timeout=None, detect_conflicts=False)
case = await verifier.verify_case_async(response, sources)

print(case.execution, case.availability, case.quality_gate, case.reason)
for claim in case.claims:
    print(claim.assessment, claim.reason, claim.supporting_sources, claim.details)
```

`verify_case` runs the same engine as `verify_parallel`, with the same models and thresholds. It adds the reasons the legacy result cannot carry, catches evaluator failures, and computes the quality gate separately. A `VerificationResult` you already have can be converted with `LegacyVerificationAdapter.from_legacy(result)`.

## The four layers

| Layer | Field | Values | Question it answers |
|---|---|---|---|
| 1. Execution | `execution` | `SUCCESS`, `ERROR`, `TIMEOUT`, `CANCELLED`, `SKIPPED` | Did the evaluator run to completion? |
| 2. Availability | `availability` | `ASSESSED`, `NO_ASSESSABLE_CLAIMS`, `NOT_EVALUATED` | Was there anything to assess? |
| 3. Claim assessment | `claims[].assessment` | `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_SOURCES` | How does each claim relate to the supplied evidence? |
| 4. Quality gate | `quality_gate` | `PASS`, `FAIL`, `INDETERMINATE` | Does the case pass policy? |

Layers 1–3 are measurement. Layer 4 is policy, computed by a separate pure function, `longtracer.contracts.compute_quality_gate(case)`. It never changes the claims.

### Availability

| Value | When |
|---|---|
| `ASSESSED` | At least one claim was assessed against evidence |
| `NO_ASSESSABLE_CLAIMS` | Empty response, response too short to contain claims, or a response that is only an honest refusal |
| `NOT_EVALUATED` | Nothing was assessed because evaluation did not complete (error, timeout, cancellation, invalid input). `claims` is always empty; partial results are never reported. |

### Claim assessment

| Assessment | Signal it comes from (thresholds unchanged) | Reason codes |
|---|---|---|
| `SUPPORTED` | Engine marked the claim `supported` | `SUPPORTED_BY_EVIDENCE` |
| `CONTRADICTED` | NLI ran and contradiction probability > 0.5 | `CONTRADICTED_BY_EVIDENCE` |
| `INSUFFICIENT_EVIDENCE` | Not supported, not contradicted | `NOT_ENTAILED` (NLI ran), `LOW_EVIDENCE_SIMILARITY` (similarity below the 0.25 NLI gate), `NO_SOURCES_SUPPLIED` (`sources=[]`), `NO_SOURCE_TEXT` (sources had no usable sentences), `HONEST_UNCERTAINTY` (refusal; claim availability `NOT_EVALUATED`) |
| `CONFLICTING_SOURCES` | One source entails the claim and a different source contradicts it (opt-in, see below) | `CONFLICTING_EVIDENCE` |

Each claim also carries:

- `supporting_sources` / `contradicting_sources`: source IDs (`source_id` or `id` from `source_metadata`, otherwise `source_<index>`)
- `confidence`: the similarity score for supported or insufficient claims, and the contradiction probability for contradicted claims
- `details`: the raw scores
- `char_start` / `char_end`: the claim's position, when the claim text appears verbatim in the response

### Quality gate policy

| Situation | Gate |
|---|---|
| `execution` is not `SUCCESS` | `INDETERMINATE` (nothing was assessed, so it is neither a pass nor a fail) |
| `availability = NOT_EVALUATED` | `INDETERMINATE` |
| `NO_ASSESSABLE_CLAIMS` with reason `HONEST_UNCERTAINTY_ONLY` | `PASS` (a correct refusal may pass policy) |
| `NO_ASSESSABLE_CLAIMS` for any other reason | `FAIL` |
| Assessed, every assessed claim `SUPPORTED` | `PASS` |
| Assessed, any claim not `SUPPORTED` | `FAIL` |

## Edge cases

| Input | `execution` | `availability` | `reason` | Claims | Gate |
|---|---|---|---|---|---|
| `""` or whitespace | `SUCCESS` | `NO_ASSESSABLE_CLAIMS` | `EMPTY_RESPONSE` | none | `FAIL` |
| Too short to contain a claim | `SUCCESS` | `NO_ASSESSABLE_CLAIMS` | `NO_EXTRACTABLE_CLAIMS` | none | `FAIL` |
| Claims, `sources=[]` | `SUCCESS` | `ASSESSED` | `NO_SOURCES_SUPPLIED` | all `INSUFFICIENT_EVIDENCE`, never `CONTRADICTED` | `FAIL` |
| Sources with no usable text (e.g. `[""]`) | `SUCCESS` | `ASSESSED` | — | `INSUFFICIENT_EVIDENCE` / `NO_SOURCE_TEXT` | `FAIL` |
| Only an honest refusal ("The provided documents do not contain…") | `SUCCESS` | `NO_ASSESSABLE_CLAIMS` | `HONEST_UNCERTAINTY_ONLY` | `NOT_EVALUATED` / `HONEST_UNCERTAINTY` | `PASS` |
| Refusal plus factual claims | `SUCCESS` | `ASSESSED` | — | refusal claim `NOT_EVALUATED`; others assessed normally | from the assessed claims |
| Model missing or failed to download | `ERROR` | `NOT_EVALUATED` | `MODEL_UNAVAILABLE` | none | `INDETERMINATE` |
| Model failed while scoring (e.g. out of memory) | `ERROR` | `NOT_EVALUATED` | `EVALUATION_FAILED` | none | `INDETERMINATE` |
| `timeout` exceeded | `TIMEOUT` | `NOT_EVALUATED` | `EVALUATION_TIMEOUT` | none | `INDETERMINATE` |
| Evaluation future cancelled | `CANCELLED` | `NOT_EVALUATED` | `EXECUTION_CANCELLED` | none | `INDETERMINATE` |
| Wrong argument types | `ERROR` | `NOT_EVALUATED` | `INVALID_INPUT` | none | `INDETERMINATE` |

`summary` states the reason in words. For example, empty and too-short responses have different summaries even though both are `NO_ASSESSABLE_CLAIMS`.

Notes on failures:

- **Model availability.** `check_case` loads the model, so `MODEL_UNAVAILABLE` is reported. `CitationVerifier()` itself still raises `ModelUnavailableError` (an `ImportError` subclass) as before. Run `longtracer models prepare` to cache the weights.
- **Timeout.** `verify_case(timeout=s)` returns as soon as the limit passes. The background thread cannot be killed; it finishes and its result is discarded. A non-positive `timeout` raises `InvalidInputError`.
- **Async cancellation.** If the task awaiting `verify_case_async` is cancelled, `asyncio.CancelledError` propagates as asyncio requires. `CANCELLED` is reported when the evaluation future itself is cancelled.
- **Legacy methods** (`verify`, `verify_parallel`, `check`, …) still raise `TypeError` for wrong argument types and still raise evaluator errors directly.

## `CONFLICTING_SOURCES` (opt-in, experimental)

The default engine runs NLI only against the single best-matching source sentence, so it cannot see that sources disagree. With `detect_conflicts=True`, `verify_case` does the following:

1. For each assessed claim, it takes the best-matching sentence from each distinct source (similarity ≥ 0.25), for up to 3 sources.
2. It scores those pairs with the NLI model, reading the label order from the model's own config.
3. It marks the claim `CONFLICTING_SOURCES` when one source entails it (> 0.5) and another contradicts it (> 0.5).

The outcome is recorded in `case.metadata["conflict_detection"]`. Failures become `ERROR`, never a silent skip.

**Cost.** The setting is off by default, and with it off, latency is unchanged. Enabling it adds roughly **55% to warm p95 latency**.

Measured on the internal reference workload (8 claims, 5 sources, 50 warm runs, Ryzen 5 PRO 5650U, CPU):

| Path | Warm p95 | Peak memory |
|---|---|---|
| Pre-v0.3.0 baseline (`verify_parallel`) | 482 ms | 1,050 MB |
| `verify_case`, default (conflicts off) | 454 ms | 1,009 MB |
| `verify_case(detect_conflicts=True)` | 751 ms | 1,027 MB |

The opt-in path is about +55% p95 against the pre-v0.3.0 baseline, which exceeds the internal 20% budget. That is why the feature is opt-in and marked experimental. Memory overhead is small (about +2%). These are internal engineering numbers, not a performance guarantee.

## Legacy fields: what they actually mean

`VerificationResult` is unchanged in v0.3.0. Its fields keep their existing meaning, and that meaning is stated plainly here.

| Field | Actual meaning |
|---|---|
| `trust_score` | Mean STS (cosine) similarity between each claim and its best-matching source sentence. It is **not** the share of supported claims. A contradicted claim usually has high similarity, so `trust_score` can be 0.95 with `verdict="FAIL"`. It is `1.0` when there are no claims (empty or very short answer) and `0.0` when no sources are given. |
| `verdict` | `"PASS"` if no claim is flagged and none is a hallucination, otherwise `"FAIL"`. An empty answer is `"PASS"`. |
| `all_supported` | `True` when no claim is flagged, including when there are no claims at all |
| `hallucinations` / `hallucination_count` | Claims contradicted by NLI, or with very low similarity plus phrases like "based on my knowledge". Honest refusals are never hallucinations. |
| `claims[].entailment_score` | Known issue: holds the NLI model's *neutral* probability, not entailment. The label order in the legacy engine does not match the model config. Unchanged in v0.3.0 so that legacy verdicts stay stable. `CaseResult` does not rely on this field. |
| `CitationVerifier(threshold=...)` | Accepted and stored, but not used by the verification logic |

**Why both results can be true at once.** For an empty answer, `verify_parallel` returns `trust_score=1.0` and `verdict="PASS"` because existing dashboards, alerts, webhooks and stored traces depend on that meaning. `verify_case` returns `NO_ASSESSABLE_CLAIMS` and `quality_gate=FAIL`. Use `CaseResult` for any new gating logic.

Other differences between the two layers:

- A pattern-based legacy "hallucination" with low similarity becomes `INSUFFICIENT_EVIDENCE`, because nothing contradicted it.
- An honest refusal is `FAIL` in the legacy verdict but can `PASS` the new gate.
- `CaseResult.trust_score` mirrors the legacy value for compatibility. Do not gate on it.
- `case.metadata["legacy_verdict"]` records the legacy verdict for comparison.

## Schema

`CaseResult` is a versioned Pydantic model. Every record carries `schema_version` (currently `"1"`).

- Committed JSON Schema: [`schema/case_result.v1.json`](schema/case_result.v1.json)
- Regenerate: `python -m longtracer.contracts.schema > docs/schema/case_result.v1.json`
- Compatibility rule (enforced by `tests/test_schema_compat.py`):
    - Adding optional fields or enum members is allowed.
    - Removing or re-typing a field, removing an enum member, or making a field required needs a `schema_version` bump.

`CaseResult` is returned to callers only. It is not written to trace storage in v0.3.0, so no storage migration is needed.
