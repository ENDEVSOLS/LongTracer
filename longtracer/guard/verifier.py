"""
Citation Verifier - Main verification class with latency tracking.
"""

import asyncio
import hashlib
import json
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from longtracer.guard.claim_splitter import split_into_claims
from longtracer.guard.nli_model import HybridVerificationModel, get_shared_model

if TYPE_CHECKING:
    from longtracer.guard.tracer import Tracer
    from longtracer.contracts.result import CaseResult


@dataclass
class VerificationResult:
    """Result of verifying an LLM response."""
    trust_score: float
    claims: List[Dict]
    flagged_claims: List[Dict]
    hallucinations: List[Dict]
    all_supported: bool
    hallucination_count: int
    verdict: str = "PASS"
    summary: str = ""
    latency_stats: Optional[Dict] = None

    def __post_init__(self):
        self.verdict = "PASS" if (
            self.all_supported and self.hallucination_count == 0
        ) else "FAIL"
        total = len(self.claims)
        supported = total - len(self.flagged_claims)
        if total == 0:
            self.summary = "No claims to verify."
        elif self.all_supported:
            self.summary = f"All {total} claim(s) supported."
        else:
            parts = [f"{supported}/{total} claims supported"]
            if self.hallucination_count > 0:
                parts.append(
                    f"{self.hallucination_count} hallucination(s) detected"
                )
            self.summary = ", ".join(parts) + "."

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        score_pct = int(self.trust_score * 100)
        bar_color = (
            "#22c55e" if score_pct >= 80
            else "#eab308" if score_pct >= 50
            else "#ef4444"
        )
        verdict_color = "#22c55e" if self.verdict == "PASS" else "#ef4444"

        rows = ""
        for c in self.claims:
            if c.get("is_hallucination"):
                bg, icon = "#fef2f2", "🔴"
            elif c.get("supported"):
                bg, icon = "#f0fdf4", "🟢"
            else:
                bg, icon = "#fefce8", "🟡"
            claim_text = c.get("claim", "")[:120]
            score = c.get("score", 0)
            source = c.get("best_source", "")[:80]
            rows += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px">{icon}</td>'
                f'<td style="padding:6px">{claim_text}</td>'
                f'<td style="padding:6px;text-align:center">{score:.2f}</td>'
                f'<td style="padding:6px;font-size:0.85em;color:#666">'
                f'{source}</td></tr>'
            )

        return (
            f'<div style="font-family:system-ui;max-width:800px">'
            f'<div style="display:flex;gap:16px;margin-bottom:12px">'
            f'<div style="padding:12px 20px;border-radius:8px;'
            f'background:{verdict_color};color:white;font-weight:bold;'
            f'font-size:1.2em">{self.verdict}</div>'
            f'<div style="flex:1;padding:12px">'
            f'<div style="font-size:0.85em;color:#666">Trust Score</div>'
            f'<div style="background:#e5e7eb;border-radius:4px;height:20px;'
            f'margin-top:4px">'
            f'<div style="background:{bar_color};height:100%;'
            f'border-radius:4px;width:{score_pct}%;min-width:2px"></div>'
            f'</div>'
            f'<div style="font-size:0.85em;margin-top:2px">'
            f'{self.trust_score:.2f} &mdash; {self.summary}</div>'
            f'</div></div>'
            f'<table style="width:100%;border-collapse:collapse;'
            f'font-size:0.9em">'
            f'<tr style="background:#f3f4f6;font-weight:600">'
            f'<th style="padding:6px;width:30px"></th>'
            f'<th style="padding:6px;text-align:left">Claim</th>'
            f'<th style="padding:6px">Score</th>'
            f'<th style="padding:6px;text-align:left">Best Source</th></tr>'
            f'{rows}</table></div>'
        )


class CitationVerifier:
    """
    LongTracer - Verify LLM responses against source documents.
    Uses hybrid STS + NLI with gating and latency tracking.

    Models are loaded once and shared across instances for performance.
    """

    _SENTINEL = object()  # distinguish "not passed" from explicit 0.5

    def __init__(
        self,
        threshold: float = _SENTINEL,  # type: ignore[assignment]
        tracer: Optional["Tracer"] = None,
        cache: bool = False,
    ):
        # Priority: code arg > pyproject.toml > default (0.5)
        if threshold is self._SENTINEL:
            from longtracer.config import load_config
            cfg = load_config()
            threshold = cfg.get("threshold", 0.5)

        self.model = get_shared_model()
        self.threshold = threshold
        self.tracer = tracer
        self._cache: Dict[str, Dict] = {} if cache else None

    @staticmethod
    def _validate_inputs(
        response: object,
        sources: object,
        source_metadata: object = None,
    ) -> None:
        """Validate types for public verify methods."""
        if not isinstance(response, str):
            raise TypeError(
                f"`response` must be a string, got {type(response).__name__}"
            )
        if not isinstance(sources, list):
            raise TypeError(
                f"`sources` must be a list of strings, "
                f"got {type(sources).__name__}"
            )
        for i, s in enumerate(sources):
            if not isinstance(s, str):
                raise TypeError(
                    f"`sources[{i}]` must be a string, "
                    f"got {type(s).__name__}"
                )
        if source_metadata is not None and not isinstance(source_metadata, list):
            raise TypeError(
                f"`source_metadata` must be a list or None, "
                f"got {type(source_metadata).__name__}"
            )

    def _cache_key(self, claim: str, sources: List[str]) -> str:
        """Compute a deterministic cache key for a claim + sources pair."""
        raw = json.dumps({"c": claim, "s": sorted(sources)}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _empty_result(self) -> VerificationResult:
        """Return a vacuous-truth result for empty/no-claim inputs."""
        return VerificationResult(
            trust_score=1.0, claims=[], flagged_claims=[],
            hallucinations=[], all_supported=True,
            hallucination_count=0, latency_stats=self.model.get_latency_stats()
        )

    def _unsupported_claims_result(
        self, claims_text: List[str]
    ) -> VerificationResult:
        """Return a result where all claims are unsupported."""
        unsupported = []
        for claim in claims_text:
            unsupported.append({
                "claim": claim, "supported": False, "score": 0.0,
                "best_score": 0.0, "sentence_results": [],
                "contradiction_score": 0.0, "entailment_score": 0.0,
                "nli_ran": False, "best_source": "",
                "best_source_index": -1,
                "best_source_metadata": None,
                "is_hallucination": False,
                "is_meta_statement": False,
                "has_hallucination_pattern": False,
            })
        return VerificationResult(
            trust_score=0.0, claims=unsupported,
            flagged_claims=unsupported.copy(),
            hallucinations=[], all_supported=False,
            hallucination_count=0,
            latency_stats=self.model.get_latency_stats()
        )

    def _build_result(
        self, verified_claims: List[Dict]
    ) -> VerificationResult:
        """Build VerificationResult from a list of verified claim dicts."""
        flagged = [c for c in verified_claims if not c["supported"]]
        hallucinations = [c for c in verified_claims if c["is_hallucination"]]

        if verified_claims:
            trust_score = (
                sum(c["score"] for c in verified_claims) / len(verified_claims)
            )
        else:
            trust_score = 1.0

        return VerificationResult(
            trust_score=trust_score,
            claims=verified_claims,
            flagged_claims=flagged,
            hallucinations=hallucinations,
            all_supported=len(flagged) == 0,
            hallucination_count=len(hallucinations),
            latency_stats=self.model.get_latency_stats(),
        )

    def _log_claims_to_tracer(self, verified_claims: List[Dict]) -> None:
        """Log claim-evidence pairs to the tracer if attached."""
        if not self.tracer:
            return
        for result in verified_claims:
            claim_id = result.get("claim", "")[:50]
            best_source = result.get("best_source", "")[:50]
            sts_score = result.get("score", 0.0)
            ent_score = result.get("entailment_score", 0.0)
            log_score = (
                max(sts_score, ent_score)
                if result.get("nli_ran") else sts_score
            )
            self.tracer.log_claim_evidence(claim_id, best_source, log_score)

    def verify(
        self,
        response: str,
        sources: List[str],
        source_metadata: Optional[List[dict]] = None,
    ) -> VerificationResult:
        """Verify an LLM response against source documents (sequential)."""
        self._validate_inputs(response, sources, source_metadata)
        self.model.reset_latency_log()

        if not response or not response.strip():
            return self._empty_result()

        claims_text = split_into_claims(response)
        if not claims_text:
            return self._empty_result()

        if not sources:
            return self._unsupported_claims_result(claims_text)

        verified_claims = []
        for claim in claims_text:
            result = self.model.verify_claim(claim, sources, source_metadata)
            verified_claims.append(result)

        self._log_claims_to_tracer(verified_claims)
        return self._build_result(verified_claims)

    def verify_parallel(
        self,
        response: str,
        sources: List[str],
        source_metadata: Optional[List[dict]] = None,
    ) -> VerificationResult:
        """Verify an LLM response using PARALLEL batch processing."""
        self._validate_inputs(response, sources, source_metadata)
        self.model.reset_latency_log()

        if not response or not response.strip():
            return self._empty_result()

        claims_text = split_into_claims(response)
        if not claims_text:
            return self._empty_result()

        if not sources:
            return self._unsupported_claims_result(claims_text)

        # Check cache for all claims
        if self._cache is not None:
            cached_results = []
            uncached_claims = []
            uncached_indices = []
            for i, claim in enumerate(claims_text):
                key = self._cache_key(claim, sources)
                if key in self._cache:
                    cached_results.append((i, self._cache[key]))
                else:
                    uncached_claims.append(claim)
                    uncached_indices.append(i)

            if uncached_claims:
                fresh = self.model.verify_claims_batch(
                    uncached_claims, sources, source_metadata
                )
                for idx, result in zip(uncached_indices, fresh):
                    key = self._cache_key(claims_text[idx], sources)
                    self._cache[key] = result
                    cached_results.append((idx, result))
            else:
                fresh = []

            cached_results.sort(key=lambda x: x[0])
            verified_claims = [r for _, r in cached_results]
        else:
            verified_claims = self.model.verify_claims_batch(
                claims_text, sources, source_metadata
            )

        self._log_claims_to_tracer(verified_claims)
        return self._build_result(verified_claims)

    async def verify_parallel_async(
        self,
        response: str,
        sources: List[str],
        source_metadata: Optional[List[dict]] = None,
    ) -> VerificationResult:
        """Async wrapper for verify_parallel.

        Runs the CPU-bound verification in a thread pool executor
        so it doesn't block the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.verify_parallel, response, sources, source_metadata
        )

    def cache_stats(self) -> Dict[str, int]:
        """Return cache hit statistics."""
        if self._cache is None:
            return {"enabled": False, "entries": 0}
        return {"enabled": True, "entries": len(self._cache)}

    def verify_with_rag_result(self, rag_result: dict) -> dict:
        """Verify a RAG result (convenience method)."""
        answer = rag_result.get("answer", "")
        source_texts = rag_result.get("source_texts", [])

        sources = rag_result.get("sources", [])
        source_metadata = []
        for src in sources:
            if hasattr(src, "metadata"):
                source_metadata.append(src.metadata)
            else:
                source_metadata.append({})

        result = self.verify_parallel(answer, source_texts, source_metadata)

        return {
            "answer": answer,
            "trust_score": result.trust_score,
            "verdict": result.verdict,
            "summary": result.summary,
            "all_supported": result.all_supported,
            "claims": result.claims,
            "flagged_claims": result.flagged_claims,
            "hallucinations": result.hallucinations,
            "hallucination_count": result.hallucination_count,
            "latency_stats": result.latency_stats,
        }

    def verify_batch(
        self,
        items: List[Dict],
        max_workers: int = 4,
    ) -> List[VerificationResult]:
        """Verify multiple responses in one call.

        Each item must be a dict with:
            - "response" (str): The LLM response to verify.
            - "sources" (list[str]): Source texts to verify against.
            - "source_metadata" (list[dict], optional): Metadata per source.

        Args:
            items: List of dicts, each with "response" and "sources".
            max_workers: Max parallel workers (default 4).

        Returns:
            List of VerificationResult, one per item (same order).

        Example::

            results = verifier.verify_batch([
                {"response": "Paris is in France.", "sources": ["Paris is the capital of France."]},
                {"response": "Water boils at 50°C.", "sources": ["Water boils at 100°C."]},
            ])
        """
        if not isinstance(items, list):
            raise TypeError(
                f"`items` must be a list of dicts, got {type(items).__name__}"
            )

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise TypeError(
                    f"`items[{i}]` must be a dict with 'response' and 'sources', "
                    f"got {type(item).__name__}"
                )
            if "response" not in item:
                raise TypeError(
                    f"`items[{i}]` missing required key 'response'"
                )
            if "sources" not in item:
                raise TypeError(
                    f"`items[{i}]` missing required key 'sources'"
                )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _verify_one(idx_item):
            idx, item = idx_item
            return idx, self.verify_parallel(
                item["response"],
                item["sources"],
                item.get("source_metadata"),
            )

        if len(items) == 1:
            # Skip ThreadPool overhead for single item
            _, result = _verify_one((0, items[0]))
            return [result]

        results: List[Optional[VerificationResult]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_verify_one, (i, item))
                for i, item in enumerate(items)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results  # type: ignore[return-value]

    async def verify_batch_async(
        self,
        items: List[Dict],
        max_workers: int = 4,
    ) -> List[VerificationResult]:
        """Async wrapper for verify_batch.

        Runs the CPU-bound batch verification in a thread pool executor
        so it doesn't block the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.verify_batch, items, max_workers
        )

    # ── Typed result path (v0.3.0) ──────────────────────────────────
    # Additive only: nothing below changes the legacy methods above.

    @staticmethod
    def _source_ids(sources: List[str], source_metadata: Optional[List[dict]]) -> List[str]:
        """Stable per-source identifiers (metadata ``source_id``/``id`` if given)."""
        ids = []
        for i in range(len(sources)):
            meta = source_metadata[i] if source_metadata and i < len(source_metadata) else None
            sid = None
            if isinstance(meta, dict):
                sid = meta.get("source_id", meta.get("id"))
            ids.append(str(sid) if sid is not None else f"source_{i}")
        return ids

    def _nli_label_indices(self) -> Optional[Dict[str, int]]:
        """Read entailment/contradiction indices from the NLI model's own config.

        Returns None if the label mapping cannot be determined, so the caller
        reports detection as unavailable instead of guessing.
        """
        cfg = getattr(getattr(self.model.nli_model, "model", None), "config", None)
        id2label = getattr(cfg, "id2label", None)
        if not isinstance(id2label, dict):
            return None
        by_name = {str(v).lower(): int(k) for k, v in id2label.items()}
        if "entailment" not in by_name or "contradiction" not in by_name:
            return None
        return {"entailment": by_name["entailment"], "contradiction": by_name["contradiction"]}

    def _detect_and_apply_conflicts(
        self,
        case_res: "CaseResult",
        sources: List[str],
        source_metadata: Optional[List[dict]] = None,
        max_sources: int = 3,
    ) -> str:
        """Opt-in: mark claims where one source entails and another contradicts.

        For each assessed claim, the best-matching sentence from each distinct
        source (STS similarity >= 0.25, the existing NLI gate) is scored with
        NLI, for up to ``max_sources`` sources. Uses the model's own label
        mapping. Returns a status string recorded in ``case_res.metadata``.
        Errors propagate as EvaluationFailedError (never silently skipped).
        """
        import numpy as np
        from sentence_transformers import util

        from longtracer.contracts.result import AssessmentAvailability, ClaimAssessment, ReasonCode
        from longtracer.errors import EvaluationFailedError

        labels = self._nli_label_indices()
        if labels is None:
            return "unavailable: NLI label mapping not found"

        ids = self._source_ids(sources, source_metadata)
        per_source = [(sid, self.model.extract_source_sentences(src)) for sid, src in zip(ids, sources)]
        per_source = [(sid, sents) for sid, sents in per_source if sents]
        if len(per_source) < 2:
            return "skipped: fewer than two sources with usable text"

        targets = [c for c in case_res.claims if c.availability == AssessmentAvailability.ASSESSED]
        if not targets:
            return "skipped: no assessed claims"

        try:
            claim_embs = self.model.sts_model.encode(
                [c.claim_text for c in targets], convert_to_tensor=True, show_progress_bar=False
            )
            src_embs = [
                self.model.sts_model.encode(sents, convert_to_tensor=True, show_progress_bar=False)
                for _, sents in per_source
            ]
            pairs: List[tuple] = []  # (claim_idx, source_id, premise, claim_text)
            for ci, claim in enumerate(targets):
                cands = []
                for (sid, sents), embs in zip(per_source, src_embs):
                    sims = util.cos_sim(claim_embs[ci : ci + 1], embs)[0]
                    j = int(sims.argmax())
                    if float(sims[j]) >= 0.25:
                        cands.append((float(sims[j]), sid, sents[j]))
                cands.sort(key=lambda x: x[0], reverse=True)
                if len(cands) >= 2:
                    pairs.extend((ci, sid, sent, claim.claim_text) for _, sid, sent in cands[:max_sources])
            if not pairs:
                return "enabled: no claim matched two or more sources"
            logits = np.asarray(self.model.nli_model.predict([(p[2], p[3]) for p in pairs]))
        except Exception as e:
            raise EvaluationFailedError(f"Conflict detection failed. (Original error: {e})") from e

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

        verdicts: Dict[int, Dict[str, List[str]]] = {}
        for (ci, sid, _, _), p in zip(pairs, probs):
            v = verdicts.setdefault(ci, {"entail": [], "contra": []})
            if p[labels["entailment"]] > 0.5:
                v["entail"].append(sid)
            elif p[labels["contradiction"]] > 0.5:
                v["contra"].append(sid)

        flagged = 0
        for ci, v in verdicts.items():
            if v["entail"] and v["contra"]:
                claim = targets[ci]
                claim.assessment = ClaimAssessment.CONFLICTING_SOURCES
                claim.reason = ReasonCode.CONFLICTING_EVIDENCE
                claim.supporting_sources = v["entail"]
                claim.contradicting_sources = v["contra"]
                flagged += 1
        return f"enabled: {flagged} conflicting claim(s)"

    def _run_legacy_engine(
        self,
        response: str,
        sources: List[str],
        source_metadata: Optional[List[dict]],
        timeout: Optional[float],
    ) -> VerificationResult:
        """Run verify_parallel, optionally bounded by a timeout.

        On timeout raises EvaluationTimeoutError and returns immediately. The
        worker thread cannot be killed; it finishes in the background and its
        result is discarded.
        """
        if timeout is None:
            return self.verify_parallel(response, sources, source_metadata)

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        from longtracer.errors import EvaluationTimeoutError

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="longtracer-verify-case")
        try:
            future = executor.submit(self.verify_parallel, response, sources, source_metadata)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as e:
                future.cancel()
                raise EvaluationTimeoutError(
                    f"Evaluation did not finish within {timeout} s.", timeout_s=timeout
                ) from e
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def _attach_offsets(case_res: "CaseResult", response: str) -> None:
        """Fill char_start/char_end where the claim text appears verbatim in the response."""
        cursor = 0
        for claim in case_res.claims:
            pos = response.find(claim.claim_text, cursor)
            if pos >= 0:
                claim.char_start = pos
                claim.char_end = pos + len(claim.claim_text)
                cursor = claim.char_end

    def verify_case(
        self,
        response: str,
        sources: List[str],
        source_metadata: Optional[List[dict]] = None,
        *,
        case_id: Optional[str] = None,
        timeout: Optional[float] = None,
        detect_conflicts: bool = False,
    ) -> "CaseResult":
        """Verify a response and return an honest, typed ``CaseResult``.

        Additive companion to ``verify_parallel``: it runs the same engine, so
        legacy behaviour is unchanged, and reports the outcome in four layers
        (execution, availability, per-claim assessment with reason codes, and a
        separately computed quality gate). Failures become typed states and are
        never reported as a pass.

        Args:
            response: LLM response text to verify.
            sources: Source texts to verify against.
            source_metadata: Optional metadata per source (``source_id``/``id``
                keys are used as source identifiers).
            case_id: Optional identifier copied onto the result.
            timeout: Optional wall-clock limit in seconds (> 0). On expiry the
                result is ``TIMEOUT``/``INDETERMINATE``; the background work is
                not killed.
            detect_conflicts: Opt-in multi-source NLI check that can emit
                ``CONFLICTING_SOURCES``. Off by default; experimental. Adds
                roughly 55% to warm p95 latency when enabled.

        Returns:
            A ``CaseResult`` with ``schema_version`` set.

        Raises:
            InvalidInputError: If ``timeout`` is not a positive number.
        """
        import time
        from concurrent.futures import CancelledError as FuturesCancelledError

        from longtracer.contracts.result import (
            ExecutionStatus,
            LegacyVerificationAdapter,
            ReasonCode,
            compute_quality_gate,
            unassessed_case,
        )
        from longtracer.errors import (
            EvaluationTimeoutError,
            EvaluatorError,
            InvalidInputError,
            ModelUnavailableError,
        )

        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise InvalidInputError(f"`timeout` must be a positive number of seconds, got {timeout!r}")

        start = time.perf_counter()

        def _elapsed() -> float:
            return (time.perf_counter() - start) * 1000.0

        def _unassessed(
            execution: ExecutionStatus, reason: ReasonCode, message: Optional[str] = None
        ) -> "CaseResult":
            return unassessed_case(
                execution, reason, case_id=case_id, error_message=message, latency_ms=_elapsed()
            )

        # ① Malformed / unsupported input → typed ERROR (legacy methods still raise TypeError)
        try:
            self._validate_inputs(response, sources, source_metadata)
        except TypeError as e:
            return _unassessed(ExecutionStatus.ERROR, ReasonCode.INVALID_INPUT, str(e))

        # ②③④ Reasons the legacy result cannot carry
        if not response.strip():
            case_reason = ReasonCode.EMPTY_RESPONSE
        elif not split_into_claims(response):
            case_reason = ReasonCode.NO_EXTRACTABLE_CLAIMS
        elif not sources:
            case_reason = ReasonCode.NO_SOURCES_SUPPLIED
        else:
            case_reason = None

        # ⑤ Run the unchanged legacy engine
        try:
            legacy_res = self._run_legacy_engine(response, sources, source_metadata, timeout)
        except EvaluationTimeoutError as e:
            return _unassessed(ExecutionStatus.TIMEOUT, ReasonCode.EVALUATION_TIMEOUT, str(e))
        except FuturesCancelledError:
            return _unassessed(ExecutionStatus.CANCELLED, ReasonCode.EXECUTION_CANCELLED, "Evaluation cancelled.")
        except ModelUnavailableError as e:
            return _unassessed(ExecutionStatus.ERROR, ReasonCode.MODEL_UNAVAILABLE, str(e))
        except EvaluatorError as e:
            return _unassessed(ExecutionStatus.ERROR, ReasonCode.EVALUATION_FAILED, str(e))
        except Exception as e:  # unknown engine failure: still never a success
            return _unassessed(ExecutionStatus.ERROR, ReasonCode.EVALUATION_FAILED, f"{type(e).__name__}: {e}")

        # ⑥ Measurement layer: map claims from raw signals (adapter, thresholds unchanged)
        case_res = LegacyVerificationAdapter.from_legacy(legacy_res, case_id=case_id, case_reason=case_reason)
        self._attach_offsets(case_res, response)
        case_res.metadata["legacy_verdict"] = legacy_res.verdict

        # ⑦ Optional conflict detection
        if detect_conflicts:
            try:
                case_res.metadata["conflict_detection"] = self._detect_and_apply_conflicts(
                    case_res, sources, source_metadata
                )
            except EvaluatorError as e:
                return _unassessed(ExecutionStatus.ERROR, ReasonCode.EVALUATION_FAILED, str(e))
        else:
            case_res.metadata["conflict_detection"] = "disabled"

        # ⑧ Policy layer, computed separately from measurement
        from longtracer.contracts.result import summarize_case

        case_res.summary = summarize_case(case_res.claims, case_res.reason)
        case_res.quality_gate = compute_quality_gate(case_res)
        case_res.latency_ms = _elapsed()
        return case_res

    async def verify_case_async(
        self,
        response: str,
        sources: List[str],
        source_metadata: Optional[List[dict]] = None,
        *,
        case_id: Optional[str] = None,
        timeout: Optional[float] = None,
        detect_conflicts: bool = False,
    ) -> "CaseResult":
        """Async wrapper for ``verify_case``.

        Runs in a thread pool executor so the event loop is not blocked. The
        timeout is enforced inside ``verify_case``. If the awaiting task itself
        is cancelled, ``asyncio.CancelledError`` propagates (asyncio contract);
        there is no caller left to receive a ``CANCELLED`` result.
        """
        loop = asyncio.get_running_loop()

        def _run() -> "CaseResult":
            return self.verify_case(
                response,
                sources,
                source_metadata,
                case_id=case_id,
                timeout=timeout,
                detect_conflicts=detect_conflicts,
            )

        return await loop.run_in_executor(None, _run)
