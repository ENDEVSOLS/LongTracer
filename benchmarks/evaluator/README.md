# Evaluator benchmark

Internal engineering benchmark for the v0.3.0 typed result path (`CitationVerifier.verify_case`). It is not shipped in the wheel or sdist, and it is not run in CI.

## Status: labels are drafted, not yet human-reviewed

All 140 cases have `review.status = "unreviewed"`. The handover requires human-reviewed labels, and **that requirement is not met yet.** Until two named people have reviewed a case, any benchmark number measures agreement with *draft* labels and must be reported as provisional. The harness prints this warning in every report.

**To review a case:**

1. Check the response, the sources, and every expected claim label against the category definition below.
2. Fix the label (not the case) if it is wrong. Never drop a case because the evaluator gets it wrong.
3. Set `review.reviewers` to two names and `review.status` to `"reviewed"`.
4. Re-run `python benchmarks/evaluator/dataset.py`.

A case counts as reviewed only with two reviewers (`dataset.is_reviewed`).

## Contents

| Item | Value |
|---|---|
| Dataset version | `0.3.0` (`DATASET_VERSION` in `dataset.py`) |
| Cases | 140: 14 categories × 10 |
| Splits | calibration 42 / held-out 98, every category in both |
| Language | English only |
| Content | Synthetic general-knowledge sentences. No customer data, no secrets. |
| Format | `sources[]` validate as `longtracer.contracts.SourceEvidence`; expected labels use `ExecutionStatus`, `AssessmentAvailability`, `QualityGate` and `ClaimAssessment` values (`dataset.validate_cases`) |
| Models | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243…`, `cross-encoder/nli-deberta-v3-xsmall` @ `a1508764…` |
| Thresholds | support 0.40, NLI gate 0.25, contradiction/entailment 0.5 (unchanged in v0.3.0) |
| Preprocessing | `split_into_claims` (responses ≤ 10 chars and sentences ≤ 15 chars dropped); sources split by `HybridVerificationModel.split_into_sentences` |

**Categories:** support, contradiction, insufficient evidence, conflicting sources, numeric change, unit change, date change, negation, mixed supported/unsupported, multi-passage support, table-derived text, empty answers, refusals, citation mistakes.

## Leakage check

`check_dataset_leakage()` compares every calibration × held-out pair. A pair leaks if any of these holds:

- the normalized responses (lower-cased, whitespace collapsed) are identical
- the token Jaccard similarity of the responses is ≥ 0.8
- the token Jaccard similarity of the concatenated source texts is ≥ 0.8

Empty responses are compared on sources only. Result for version 0.3.0: **0 pairs flagged**, so nothing was removed. The checker is tested against a known duplicate in `test_harness.py`.

## Running

```bash
python benchmarks/evaluator/dataset.py                   # regenerate + validate + leakage check
HF_HUB_OFFLINE=1 make benchmark                          # held-out split, real weights
python benchmarks/evaluator/run.py --split calibration --out /tmp/calib
python benchmarks/evaluator/run.py --no-conflicts        # default verify_case path
pytest benchmarks/evaluator/test_harness.py              # harness self-tests, no weights
```

The harness refuses to run with a mocked model, or under `CI=true` unless `--allow-ci` is passed.

## Reporting rules

- Every rate is printed as `x% (num/den)`, with a Wilson 95% interval when n ≥ 10.
- Cases that end in `ERROR` / `TIMEOUT` / `CANCELLED` are counted as evaluator errors and excluded from claim metrics. They are never scored as a label.
- An expected claim the evaluator did not produce is counted as an alignment failure. It is never scored as a label.
- Release-gate targets (contradiction-alert precision ≥ 95%, false alerts ≤ 5%) are hypotheses. Misses are reported as measured, and thresholds and cases are not tuned to pass.
