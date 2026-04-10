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

