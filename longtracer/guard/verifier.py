"""
Citation Verifier - Main verification class with latency tracking.
"""

from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass

from longtracer.guard.claim_splitter import split_into_claims
from longtracer.guard.nli_model import HybridVerificationModel

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
    latency_stats: Optional[Dict] = None


class CitationVerifier:
    """
    LongTracer - Verify LLM responses against source documents.
    Uses hybrid STS + NLI with gating and latency tracking.
    """

    def __init__(self, threshold: float = 0.5, tracer: Optional["Tracer"] = None):
        self.model = HybridVerificationModel()
        self.threshold = threshold
        self.tracer = tracer

    def _empty_result(self) -> VerificationResult:
        """Return a vacuous-truth result for empty/no-claim inputs."""
        return VerificationResult(
            trust_score=1.0, claims=[], flagged_claims=[],
            hallucinations=[], all_supported=True,
            hallucination_count=0, latency_stats=self.model.get_latency_stats()
        )

    def _unsupported_claims_result(self, claims_text: List[str]) -> VerificationResult:
        """Return a result where all claims are unsupported (no sources provided)."""
        unsupported = []
        for claim in claims_text:
            unsupported.append({
                "claim": claim, "supported": False, "score": 0.0,
                "best_score": 0.0, "sentence_results": [],
                "contradiction_score": 0.0, "entailment_score": 0.0,
                "nli_ran": False, "best_source": "", "best_source_index": -1,
                "best_source_metadata": None, "is_hallucination": False,
                "is_meta_statement": False, "has_hallucination_pattern": False,
            })
        return VerificationResult(
            trust_score=0.0, claims=unsupported,
            flagged_claims=unsupported.copy(),
            hallucinations=[], all_supported=False,
            hallucination_count=0, latency_stats=self.model.get_latency_stats()
        )

    def verify(
        self,
        response: str,
        sources: List[str],
        source_metadata: Optional[List[dict]] = None
    ) -> VerificationResult:
        """Verify an LLM response against source documents (sequential)."""
        self.model.reset_latency_log()

        # Empty/whitespace response → vacuous truth
        if not response or not response.strip():
            return self._empty_result()

        claims_text = split_into_claims(response)
        if not claims_text:
            return self._empty_result()

        # No sources → all claims unsupported
        if not sources:
            return self._unsupported_claims_result(claims_text)

        verified_claims = []
        flagged = []
        hallucinations = []

        for claim in claims_text:
            result = self.model.verify_claim(claim, sources, source_metadata)
            verified_claims.append(result)

            if self.tracer:
                claim_id = claim[:50]
                best_source = result.get("best_source", "")[:50]
                sts_score = result.get("score", 0.0)
                ent_score = result.get("entailment_score", 0.0)
                log_score = max(sts_score, ent_score) if result.get("nli_ran") else sts_score
                self.tracer.log_claim_evidence(claim_id, best_source, log_score)

            if not result["supported"]:
                flagged.append(result)
            if result["is_hallucination"]:
                hallucinations.append(result)

        if verified_claims:
            trust_score = sum(c["score"] for c in verified_claims) / len(verified_claims)
        else:
            trust_score = 1.0

        return VerificationResult(
            trust_score=trust_score, claims=verified_claims,
            flagged_claims=flagged, hallucinations=hallucinations,
            all_supported=len(flagged) == 0,
            hallucination_count=len(hallucinations),
            latency_stats=self.model.get_latency_stats()
        )

    def verify_parallel(
        self,
        response: str,
        sources: List[str],
        source_metadata: Optional[List[dict]] = None
    ) -> VerificationResult:
        """Verify an LLM response using PARALLEL batch processing."""
        self.model.reset_latency_log()

        # Empty/whitespace response → vacuous truth
        if not response or not response.strip():
            return self._empty_result()

        claims_text = split_into_claims(response)
        if not claims_text:
            return self._empty_result()

        # No sources → all claims unsupported
        if not sources:
            return self._unsupported_claims_result(claims_text)

        verified_claims = self.model.verify_claims_batch(
            claims_text, sources, source_metadata
        )

        flagged = []
        hallucinations = []

        for result in verified_claims:
            if self.tracer:
                claim_id = result.get("claim", "")[:50]
                best_source = result.get("best_source", "")[:50]
                sts_score = result.get("score", 0.0)
                ent_score = result.get("entailment_score", 0.0)
                log_score = max(sts_score, ent_score) if result.get("nli_ran") else sts_score
                self.tracer.log_claim_evidence(claim_id, best_source, log_score)

            if not result["supported"]:
                flagged.append(result)
            if result["is_hallucination"]:
                hallucinations.append(result)

        if verified_claims:
            trust_score = sum(c["score"] for c in verified_claims) / len(verified_claims)
        else:
            trust_score = 1.0

        return VerificationResult(
            trust_score=trust_score, claims=verified_claims,
            flagged_claims=flagged, hallucinations=hallucinations,
            all_supported=len(flagged) == 0,
            hallucination_count=len(hallucinations),
            latency_stats=self.model.get_latency_stats()
        )

    def verify_with_rag_result(self, rag_result: dict) -> dict:
        """Verify a RAG result (convenience method)."""
        answer = rag_result.get("answer", "")
        source_texts = rag_result.get("source_texts", [])

        sources = rag_result.get("sources", [])
        source_metadata = []
        for src in sources:
            if hasattr(src, 'metadata'):
                source_metadata.append(src.metadata)
            else:
                source_metadata.append({})

        result = self.verify_parallel(answer, source_texts, source_metadata)

        return {
            "answer": answer,
            "trust_score": result.trust_score,
            "all_supported": result.all_supported,
            "claims": result.claims,
            "flagged_claims": result.flagged_claims,
            "hallucinations": result.hallucinations,
            "hallucination_count": result.hallucination_count,
            "latency_stats": result.latency_stats
        }
