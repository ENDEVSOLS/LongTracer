# Engineering Baseline (pre-v0.3.0)

> **Internal engineering artifact.** These numbers describe one machine on one day.
> They are not performance guarantees and must not be quoted as marketing claims.

This is the "before" snapshot for v0.3.0. Every later change to evaluator semantics or
performance is compared against it. **Nothing was fixed while producing this report.**

## 1. What was measured

| Item | Value |
|---|---|
| Commit (`main`) | `cb3407a` |
| Commit with PR #20 applied | `90c1377` (adds `longtracer/contracts/` only; `longtracer/guard/` is byte-identical to `cb3407a`) |
| Package version | `0.2.0` (`pyproject.toml`) |
| Python measured | Tests: 3.10.21, 3.11.16, 3.12.14 (CPython, `uv venv --python 3.1x`). Timings: 3.12.14 only. |
| Install | `pip install -e .` (bare, no extras, same as CI) + dev tools |
| torch | `2.14.1+cpu` (CPU wheel from `https://download.pytorch.org/whl/cpu`) |
| sentence-transformers / transformers / pydantic | `6.1.0` / `5.18.0` / `2.13.5` |
| Hardware | AMD Ryzen 5 PRO 5650U (6 cores / 12 threads), 26 GiB RAM, no GPU |
| OS | Ubuntu 26.04.1 LTS, Linux 7.0.0-34-generic, x86_64 |
| Device | CPU; `torch.get_num_threads() == 6` |

The test suite was run on all three CI versions (3.10 / 3.11 / 3.12). At `cb3407a` it gives 158 passed on each, with a bare `pip install -e .` (no extras) and `HF_HUB_OFFLINE=1`. Performance was measured on 3.12 only.

## 2. Quality gates

| Gate | Command | `cb3407a` | `90c1377` (PR #20) |
|---|---|---|---|
| Tests collected | `pytest --collect-only -q` | 158 | 172 |
| Tests result | `pytest -q` | **158 passed**, 0 failed, 0 skipped | **172 passed**, 0 failed, 0 skipped |
| Lint | `ruff check .` | `All checks passed!` | `All checks passed!` |
| Format | `ruff format --check .` | **fails**: 65 files would be reformatted | **fails**: 68 files would be reformatted |
| Types | `mypy longtracer/ --ignore-missing-imports` | **114 errors** in 18 files | **117 errors** in 19 files (+3 in `contracts/result.py`) |
| Build | `python -m build && twine check dist/*` | wheel + sdist built, both `PASSED` | same |
| Docs | `mkdocs build --strict` | passes (INFO: `observability-analytics/blueprint.md` not in nav) | passes |
| Health | `longtracer doctor` | exit 0, "All checks passed" | same |

Notes:

- `make lint` runs `ruff format --check .` and therefore **fails at baseline**. The pre-existing
  formatting backlog is not fixed in v0.3.0 (no opportunistic refactoring). New files are formatted;
  existing files are left as they are.
- The mypy backlog is the comparison point for v0.3.0: **no new errors relative to 117**.
  Per-file counts: `tracer.py` 19, `cache/sqlite.py` 18, `cache/mongo.py` 14, `cache/postgres.py` 13,
  `nli_model.py` 11, `cache/kv_mongo.py` 11, `otel.py` 7, `cache/redis_backend.py` 6, `webhooks.py` 4,
  `contracts/result.py` 3, others ≤ 2.
- `pytest` took 105.6 s on the first run (it downloaded the NLI model) and 27.6 s once models were
  cached. See §5.6: the test suite loads real model weights.

## 3. Models

| Role | Identifier (as loaded by code) | Revision observed in local HF cache |
|---|---|---|
| STS (bi-encoder) | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| NLI (cross-encoder) | `cross-encoder/nli-deberta-v3-xsmall` | `a150876415327c80daeff35ca6f68f5ed8cf5c24` |

**The code does not pin a revision.** `HybridVerificationModel` loads by name, so a fresh install
gets whatever the Hub's `main` points to on that day. The revisions above are what was measured,
read from `~/.cache/huggingface/hub/models--<org>--<name>/refs/main`.

Thresholds in effect (hard-coded in `longtracer/guard/nli_model.py`): STS support `0.40`,
NLI gate `avg_score >= 0.25`, contradiction / entailment `> 0.5`, hallucination-pattern
similarity `< 0.20`.

## 4. Performance

Command (models already cached; offline to prove no network use):

```bash
HF_HUB_OFFLINE=1 python benchmarks/perf_baseline.py --cold-runs 3 --warm-runs 50 --json baseline.json
```

**Reference workload:** 5 sources (1,490 chars, 25 sentences), one response of 482 chars that
splits into 8 claims (6 grounded, 1 contradicted, 1 unrelated). `verify_parallel`, cache off,
batch size 1, 5 warm-up runs discarded.

| Metric | Value | State |
|---|---|---|
| Cold `import longtracer.guard.nli_model` | 6.42 s (median of 3) | fresh interpreter |
| Cold model load (`get_shared_model()`) | 1.10 s median, 1.13 s max (3 runs) | fresh interpreter, weights on local disk |
| Peak RSS after cold load | 674 MB | |
| Warm `verify_parallel` p50 | **433 ms** | 50 runs |
| Warm `verify_parallel` p95 | **482 ms** | 50 runs |
| Warm min / max | 426 ms / 584 ms | |
| Peak RSS, warm process | **1,050 MB** | |

The v0.3.0 performance budget (≤ 20% degradation) applies to warm p95 (482 ms) and warm
peak RSS (1,050 MB) on this workload and machine.

## 5. Behaviour observed

### 5.1 Offline behaviour

After `longtracer models prepare`, the following all succeed with `HF_HUB_OFFLINE=1`:
`longtracer models prepare` (reports both models ready, ~1 s), `longtracer doctor` (exit 0),
and the full performance run above. No network access is needed once weights are cached.

### 5.2 Edge cases (real models, `verify_parallel`)

Reproduce with the snippet in §6.

| Input | `trust_score` | `verdict` | Claims | Observation |
|---|---|---|---|---|
| `""` | 1.00 | PASS | 0 | **False success** (handover §1.4) |
| `"   \n "` | 1.00 | PASS | 0 | **False success** |
| `"Yes, it does."` (≤ 15 chars) | 1.00 | PASS | 0 | **False success.** Same result object as empty; reason cannot be told apart |
| Claim, `sources=[]` | 0.00 | FAIL | 1 | Not a hallucination (correct), but **no reason code**; looks like any low score |
| Honest refusal ("The provided documents do not contain…") | 0.21 | FAIL | 1 | `is_meta_statement=True`, not a hallucination, but still flagged, so verdict FAIL |
| Contradiction (50 °C vs 100 °C) | **0.95** | FAIL | 1 | High `trust_score` despite contradiction: score is mean STS similarity, not support rate |
| Date change (1899 vs 1889) | 0.93 | FAIL | 1 | Detected by NLI contradiction |
| Negation | 0.89 | FAIL | 1 | Detected by NLI contradiction |
| Supported claim | 0.95 | PASS | 1 | |
| Sources disagree (1889 vs "1925, not 1889") | 1.00 | PASS | 1 | **Conflict invisible**: NLI only checks the single best-matching sentence |

### 5.3 Defect: NLI label order is swapped for entailment/neutral

`HybridVerificationModel.compute_nli_scores` reads the softmax as
`[contradiction, neutral, entailment]`. The model's own config says otherwise:

```json
"id2label": {"0": "contradiction", "1": "entailment", "2": "neutral"}
```

So the value stored as `entailment_score` is actually the **neutral** probability, and vice versa.
Measured directly with `CrossEncoder.predict`:

| Premise / hypothesis | p[0] | p[1] | p[2] |
|---|---|---|---|
| identical sentences | 0.001 | **0.995** | 0.004 |
| "boils at 100 C" / "boils at 50 C" | **0.998** | 0.000 | 0.002 |

Consequences: the "NLI entailment rescue" (`entailment_score > 0.5`) fires on *neutral* pairs, and
real entailment never rescues a low-similarity claim. Contradiction (index 0) is read correctly, so
contradiction detection is unaffected. **Not fixed here.** Fixing it changes legacy `supported`
and `verdict` values, so it needs a maintainer decision.

### 5.4 Observation: `longtracer/__init__.py` eagerly imports model libraries

`from longtracer import ...` (even `import longtracer.errors`, since Python always runs the
parent package's `__init__.py` first) triggers `longtracer/__init__.py` → `CitationVerifier` →
`longtracer.guard.nli_model` → `sentence_transformers`/`torch` at **module import time**, not
first use. Measured with `python -X importtime -c "import longtracer.errors"`: `sentence_transformers`
alone accounts for ~4.3 s of the ~4.4 s total. This is pre-existing (confirmed present at `cb3407a`
too) and is a `longtracer/__init__.py` design choice, not specific to any new module. `tests/test_imports.py`
guards individual new modules (e.g. `longtracer/errors.py` has no heavy imports in its own source)
but cannot guard against this at the package level without changing `longtracer/__init__.py`,
which is out of scope for v0.3.0 (no opportunistic refactoring).

### 5.5 Defect: `threshold` argument has no effect

`CitationVerifier(threshold=...)` and `check(..., threshold=...)` store the value, but no
verification code reads `self.threshold`. Verified: `threshold=0.0` and `threshold=0.99` give the
same verdict. Not fixed (threshold work is out of scope for v0.3.0).

### 5.6 Semantically ambiguous

- `trust_score` is the mean STS similarity of the claims. A contradicted claim usually has *high*
  similarity, so `trust_score` can be 0.95 with `verdict=FAIL`.
- Claim splitting drops any sentence of 15 characters or fewer and any whole response of 10
  characters or fewer, silently.
- Optional SLM fallback (when `llama-cpp-python` is installed) can override NLI. It was not installed
  for this baseline, so its behaviour was not measured.

### 5.7 Test-suite observation

`tests/test_verifier_edge_cases.py` patches `longtracer.guard.verifier.HybridVerificationModel`, but
`CitationVerifier.__init__` calls `get_shared_model()`, which builds the model from
`longtracer.guard.nli_model`. The patch therefore does not prevent real weights from loading. The
mock is assigned afterwards (`v.model = mock`), so assertions still use the mock, but the suite needs
model weights (network or cache) to run.

### 5.8 Security observation: `serve` binds to all interfaces by default

The handover's privacy defaults (§9) require "loopback dashboard binding". The code does otherwise:

- `longtracer serve` defaults to `--host 0.0.0.0` (`longtracer/cli.py`)
- `run_server(host="0.0.0.0")` (`longtracer/server.py`)

So the dashboard and REST API are reachable from the network unless `--host 127.0.0.1` is passed. **Not changed:** `server.py` is out of scope for v0.3.0, and changing a CLI default alters CLI behaviour. Raised for maintainer decision (handover §8: security issue → stop and ask).

## 6. Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e . ruff mypy pytest pytest-cov hypothesis build twine mkdocs-material "mkdocstrings[python]"
longtracer models prepare

pytest -q
ruff check . ; ruff format --check .
mypy longtracer/ --ignore-missing-imports
python -m build && twine check dist/*
mkdocs build --strict
HF_HUB_OFFLINE=1 longtracer doctor
HF_HUB_OFFLINE=1 python benchmarks/perf_baseline.py --json baseline.json
```

Edge-case probe used for §5.2:

```python
from longtracer.guard.verifier import CitationVerifier

v = CitationVerifier()
src = ["Water boils at 100 degrees Celsius at standard sea-level pressure.",
       "The Eiffel Tower was completed in 1889 in Paris."]
for response, sources in [
    ("", src), ("   \n ", src), ("Yes, it does.", src),
    ("Water boils at 100 degrees Celsius at sea level.", []),
    ("The provided documents do not contain information about the population of Paris.", src),
    ("Water boils at 50 degrees Celsius at standard sea-level pressure.", src),
]:
    r = v.verify_parallel(response, sources)
    print(repr(response[:40]), r.trust_score, r.verdict, len(r.claims))
```

NLI label check used for §5.3:

```python
from sentence_transformers import CrossEncoder
import numpy as np

m = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")
print(m.model.config.id2label)
s = m.predict([("The Eiffel Tower was completed in 1889.", "The Eiffel Tower was completed in 1889.")])[0]
print(np.exp(s) / np.exp(s).sum())
```
