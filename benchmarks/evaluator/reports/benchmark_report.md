# LongTracer v0.3.0 Evaluator Benchmark Report

> Internal engineering measurement. Not a public accuracy claim. English, synthetic
> general-knowledge cases only; no domain or multilingual conclusions are supported.

> **Provisional:** 0 of 140 dataset cases are human-reviewed. Labels were drafted and have not yet been signed off by two human reviewers, so these numbers measure agreement with draft labels.

| Item | Value |
|---|---|
| Split | `heldout` (98 cases) |
| Dataset version | `0.3.0` |
| Conflict detection | `True` |
| Commit | `ca4e5f0` |
| Hardware | AMD Ryzen 5 PRO 5650U with Radeon Graphics (12 threads), device `cpu`, Linux 7.0.0-34-generic |
| Python / torch / sentence-transformers / transformers | 3.12.14 / 2.14.1+cpu / 6.1.0 / 5.18.0 |
| STS model | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| NLI model | `cross-encoder/nli-deberta-v3-xsmall` @ `a150876415327c80daeff35ca6f68f5ed8cf5c24` |
| Thresholds | support 0.40, NLI gate 0.25, contradiction/entailment 0.5 (unchanged) |
| Latency per case | p50 63 ms, p95 170 ms (n=98) |
| Generated | 2026-10-03 03:58:42 UTC |

## 1. Release-gate hypotheses vs. measured

Targets are hypotheses to measure, not achievements to assert. Misses are reported as-is.

| Metric | Target | Measured | Meets target |
|---|---|---|---|
| Contradiction-alert precision | ≥ 95% | 82.5% (33/40) [95% CI 68.1–91.3%] | **no** |
| False alerts on expected-PASS cases (gate FAIL) | ≤ 5% | 3.6% (1/28) [95% CI 0.6–17.7%] | yes |
| Evaluator errors (ERROR / TIMEOUT / CANCELLED) | report | 0/98 cases | — |
| Expected-PASS cases gated INDETERMINATE | report | 0/28 | — |
| Claim alignment failures (expected claim not produced) | report | 0 | — |
| Case-level quality-gate agreement | report | 81.6% (80/98) [95% CI 72.8–88.1%] | — |

## 2. Per-label precision and recall (claim level)

| Label | Precision | Recall | Expected (support) | Predicted |
|---|---|---|---|---|
| `SUPPORTED` | 62.2% (28/45) [95% CI 47.6–74.9%] | 100.0% (28/28) [95% CI 87.9–100.0%] | 28 | 45 |
| `CONTRADICTED` | 82.5% (33/40) [95% CI 68.1–91.3%] | 80.5% (33/41) [95% CI 66.0–89.8%] | 41 | 40 |
| `INSUFFICIENT_EVIDENCE` | 100.0% (11/11) [95% CI 74.1–100.0%] | 50.0% (11/22) [95% CI 30.7–69.3%] | 22 | 11 |
| `CONFLICTING_SOURCES` | 100.0% (2/2) [n<10, no CI] | 28.6% (2/7) [n<10, no CI] | 7 | 2 |

## 3. Confusion counts (rows = expected, columns = predicted)

| Expected \ Predicted | `SUPPORTED` | `CONTRADICTED` | `INSUFFICIENT_EVIDENCE` | `CONFLICTING_SOURCES` | Total |
|---|---|---|---|---|---|
| `SUPPORTED` | 28 | 0 | 0 | 0 | 28 |
| `CONTRADICTED` | 8 | 33 | 0 | 0 | 41 |
| `INSUFFICIENT_EVIDENCE` | 4 | 7 | 11 | 0 | 22 |
| `CONFLICTING_SOURCES` | 5 | 0 | 0 | 2 | 7 |

## 4. Case-level gate agreement by category

| Category | Agreement |
|---|---|
| `citation_mistakes` | 85.7% (6/7) [n<10, no CI] |
| `conflicting_sources` | 28.6% (2/7) [n<10, no CI] |
| `contradiction` | 100.0% (7/7) [n<10, no CI] |
| `date_change` | 100.0% (7/7) [n<10, no CI] |
| `empty_answers` | 100.0% (7/7) [n<10, no CI] |
| `insufficient_evidence` | 71.4% (5/7) [n<10, no CI] |
| `mixed_supported_unsupported` | 71.4% (5/7) [n<10, no CI] |
| `multi_passage_support` | 100.0% (7/7) [n<10, no CI] |
| `negation` | 71.4% (5/7) [n<10, no CI] |
| `numeric_change` | 100.0% (7/7) [n<10, no CI] |
| `refusals` | 85.7% (6/7) [n<10, no CI] |
| `support` | 100.0% (7/7) [n<10, no CI] |
| `table_derived_text` | 100.0% (7/7) [n<10, no CI] |
| `unit_change` | 28.6% (2/7) [n<10, no CI] |

## 5. Disagreements (25)

| Case | Category | Response | Gate exp → pred | Claims exp → pred |
|---|---|---|---|---|
| `case_insufficient_evidence_004` | insufficient_evidence | The ancient kingdom possessed advanced quantum computing devices in 500 BC. | FAIL → FAIL | INSUFFICIENT_EVIDENCE → CONTRADICTED |
| `case_insufficient_evidence_005` | insufficient_evidence | Certain species of deep sea cephalopods communicate exclusively through psychic telepathy. | FAIL → PASS | INSUFFICIENT_EVIDENCE → SUPPORTED |
| `case_insufficient_evidence_008` | insufficient_evidence | Domestic feline sleep cycles are directly influenced by Jupiter&#x27;s gravitational field. | FAIL → FAIL | INSUFFICIENT_EVIDENCE → CONTRADICTED |
| `case_insufficient_evidence_009` | insufficient_evidence | The original blueprint was drafted using green ink imported from Belgium. | FAIL → PASS | INSUFFICIENT_EVIDENCE → SUPPORTED |
| `case_conflicting_sources_002` | conflicting_sources | The company reported ten million dollars in annual operating profit. | FAIL → PASS | CONFLICTING_SOURCES → SUPPORTED |
| `case_conflicting_sources_004` | conflicting_sources | The newly opened transit bridge was designed by Chief Engineer Martinez. | FAIL → PASS | CONFLICTING_SOURCES → SUPPORTED |
| `case_conflicting_sources_005` | conflicting_sources | Clinical trial results showed the drug reduced symptom duration by four days. | FAIL → PASS | CONFLICTING_SOURCES → SUPPORTED |
| `case_conflicting_sources_006` | conflicting_sources | The ancient manuscript contains exactly forty-two illustrated botanical plates. | FAIL → PASS | CONFLICTING_SOURCES → SUPPORTED |
| `case_conflicting_sources_010` | conflicting_sources | The solar array provides eighty percent of the facility&#x27;s electric power. | FAIL → PASS | CONFLICTING_SOURCES → SUPPORTED |
| `case_unit_change_002` | unit_change | The dry cargo container weighs approximately two thousand pounds when empty. | FAIL → PASS | CONTRADICTED → SUPPORTED |
| `case_unit_change_004` | unit_change | The chemical storage tank holds twenty thousand gallons of industrial solvent. | FAIL → PASS | CONTRADICTED → SUPPORTED |
| `case_unit_change_005` | unit_change | The microchip fabrication gate length measures five micrometers in thickness. | FAIL → PASS | CONTRADICTED → SUPPORTED |
| `case_unit_change_008` | unit_change | The fiber optic connection transmits data at twenty gigabits per day. | FAIL → PASS | CONTRADICTED → SUPPORTED |
| `case_unit_change_010` | unit_change | The subterranean pressure valve operates at forty Pascals of hydraulic pressure. | FAIL → PASS | CONTRADICTED → SUPPORTED |
| `case_negation_009` | negation | The water filtration system does not eliminate heavy metal contaminants. | FAIL → PASS | CONTRADICTED → SUPPORTED |
| `case_negation_010` | negation | The mobile device battery is removable by end users. | FAIL → PASS | CONTRADICTED → SUPPORTED |
| `case_mixed_supported_unsupported_004` | mixed_supported_unsupported | Mount Everest is Earth&#x27;s highest mountain. Space aliens built a radio telescope on its summit. | FAIL → FAIL | SUPPORTED, INSUFFICIENT_EVIDENCE → SUPPORTED, CONTRADICTED |
| `case_mixed_supported_unsupported_005` | mixed_supported_unsupported | Mercury is the planet closest to the Sun. Mercury is completely covered in liquid chocolate. | FAIL → FAIL | SUPPORTED, INSUFFICIENT_EVIDENCE → SUPPORTED, CONTRADICTED |
| `case_mixed_supported_unsupported_006` | mixed_supported_unsupported | The Pacific Ocean is the largest ocean. Deep sea mermaids manage fish trade routes there. | FAIL → PASS | SUPPORTED, INSUFFICIENT_EVIDENCE → SUPPORTED, SUPPORTED |
| `case_mixed_supported_unsupported_008` | mixed_supported_unsupported | DNA carries genetic instructions. DNA was first synthesized by medieval alchemists in Venice. | FAIL → PASS | SUPPORTED, INSUFFICIENT_EVIDENCE → SUPPORTED, SUPPORTED |
| `case_mixed_supported_unsupported_010` | mixed_supported_unsupported | Photosynthesis produces glucose and oxygen. Plants communicate by sending text messages. | FAIL → FAIL | SUPPORTED, INSUFFICIENT_EVIDENCE → SUPPORTED, CONTRADICTED |
| `case_refusals_009` | refusals | The provided context does not provide any answers for the query submitted. | PASS → FAIL | INSUFFICIENT_EVIDENCE → INSUFFICIENT_EVIDENCE |
| `case_citation_mistakes_005` | citation_mistakes | According to Table Three, the product warranty period is five full years. | FAIL → FAIL | INSUFFICIENT_EVIDENCE → CONTRADICTED |
| `case_citation_mistakes_009` | citation_mistakes | According to the weather bulletin, snowfall will commence at midnight. | FAIL → PASS | CONTRADICTED → SUPPORTED |
| `case_citation_mistakes_010` | citation_mistakes | Source Document Gamma proves that the team completed twenty sprints in total. | FAIL → FAIL | INSUFFICIENT_EVIDENCE → CONTRADICTED |
