# EvalPort Integration

LongTracer can export any verification result to [EvalPort](https://github.com/adhabnr-ux/evalport), a small open standard for portable LLM evaluation datasets and results. Use `to_openeval()` to turn a `VerificationResult` (or a batch of them) into a plain-dict `ResultSet` you can validate against the EvalPort schema, save as JSON, or feed into any EvalPort-compatible tool.

This is a one-way, `to_openeval()`-only integration: LongTracer *produces* verification results, it does not consume EvalPort suites, so there is no `from_openeval()`.

## Install

`to_openeval()` itself has **no hard dependency** on EvalPort — it works out of the box. Installing the `evalport` extra only sharpens the `version` field of the exported `ResultSet` to match your installed EvalPort spec revision; without it, a fixed fallback version string is used and everything else is identical.

```bash
pip install "longtracer[evalport]"
```

## Usage

```python
from longtracer import to_openeval
from longtracer.guard.verifier import CitationVerifier

verifier = CitationVerifier()
result = verifier.verify_parallel(response, sources)

result_set = to_openeval(result, run_id="my_run")
# result_set is a plain dict conforming to the EvalPort ResultSet schema

import json
with open("resultset.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

Batches (e.g. from `CitationVerifier.verify_batch()`) work the same way — pass a list, tuple, or any other `Sequence` of `VerificationResult`s and each one becomes a separate `Result` entry:

```python
results = verifier.verify_batch(items)
result_set = to_openeval(
    results,
    run_id="nightly_eval",
    test_case_ids=["case_1", "case_2", "case_3"],
    response_texts=[r["response"] for r in items],
)
```

`to_openeval` is also reachable as `longtracer.adapters.to_openeval` and `longtracer.adapters.evalport.to_openeval`.

## Validating the output

If `evalport-sdk` is installed, you can confirm the exported `ResultSet` is schema-valid:

```python
from openeval.validate import validate_result_set

validation = validate_result_set(result_set)
assert validation.valid, validation.errors
```

## Field mapping

Every field below is a **direct passthrough** of what LongTracer already computed — `to_openeval()` never recalculates a score or a verdict. The only transformation applied is clamping each claim's `score` into EvalPort's required `[0.0, 1.0]` range (LongTracer's cosine-similarity score is mathematically unbounded to `[-1.0, 1.0]`); the untouched original is always preserved in `metadata.openeval.raw_score`.

### Response level (`VerificationResult` → EvalPort `Result`)

| LongTracer field | EvalPort field | Notes |
|---|---|---|
| `verdict == "PASS"` | `passed` | Direct rename, not a recomputation |
| `trust_score` | `metadata.trust_score` | |
| `verdict` | `metadata.verdict` | `"PASS"` / `"FAIL"` |
| `summary` | `metadata.summary` | |
| `all_supported` | `metadata.all_supported` | |
| `hallucination_count` | `metadata.hallucination_count` | |
| `len(flagged_claims)` | `metadata.flagged_claim_count` | |
| `latency_stats` | `metadata.latency_stats` | Also mirrored to `duration_ms` when `latency_stats.total_ms` is present |
| `claims` | `grader_results` | One `GraderResult` per claim, see below |
| *(not tracked by LongTracer)* | `actual_output` | Only set when you pass `response_texts=[...]` yourself |

### Per-claim (claim dict → EvalPort `GraderResult`)

Each claim becomes one `GraderResult` of `type: "custom"`, with `grader_id` set to `lt_claim_<index>`:

| LongTracer field | EvalPort field | Notes |
|---|---|---|
| `supported` | `passed` | |
| `score` (clamped) | `score` | Original preserved at `metadata.openeval.raw_score` |
| `is_hallucination` | `metadata.is_hallucination` | |
| `best_source`, `best_source_index`, `best_source_metadata` | `metadata.best_source*` | |
| `best_score`, `contradiction_score`, `entailment_score`, `nli_ran` | `metadata.*` | |
| `is_meta_statement`, `has_hallucination_pattern` | `metadata.*` | |
| `sentence_results` | `metadata.sentence_results` | |

`metadata.openeval.claim_status` additionally categorizes each claim as one of `"supported"` / `"unsupported"` / `"hallucination"` — this is what keeps an unsupported-but-not-flagged claim distinct from a confirmed hallucination, both of which have `passed=False`.

## Notes

- `run_id` / `suite_id` / `started_at` populate `ResultSet` fields LongTracer has no equivalent for; sane defaults are used when omitted (`started_at` defaults to the current UTC time at conversion).
- `test_case_ids` defaults to `"claim_verification_0"`, `"claim_verification_1"`, ... when not supplied.
- `runner.version` reports the installed `longtracer` distribution's version via `importlib.metadata` (not by importing `longtracer` itself), and is `None` if the package isn't installed as a distribution.
- A `VerificationResult` with no claims converts to a vacuous `passed=True` result with an empty `grader_results` list, matching LongTracer's own empty-input handling.
