"""
Hybrid Verification Model - Optimized Bi-Encoder STS + NLI + SLM Fallback.

Pipeline:
- Step A: Fast Bi-Encoder (all-MiniLM-L6-v2) for Evidence Selection (O(N+M))
- Step B: Cross-Encoder (DeBERTa) for Verification (O(1))
- Step C: SLM Fallback (Qwen2.5-1.5B GGUF) for uncertain/numeric claims (~5-10%)
"""

import time
import re
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from longtracer.guard.claim_splitter import analyze_claim

logger = logging.getLogger("longtracer")


class HybridVerificationModel:
    """
    Optimized hybrid verification.
    STS: Bi-Encoder (Fast, <100ms)
    NLI: Cross-Encoder (Accurate, ~150ms)
    SLM: Generative fallback for uncertain/numeric claims (~200-400ms, ~5-10% of claims)
    """

    def __init__(
        self,
        sts_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        support_threshold: float = 0.40,
        verbose: bool = True,
        use_slm: Optional[bool] = None,
    ):
        self.verbose = verbose
        self.support_threshold = support_threshold

        self.latency_log = {
            "sts_calls": 0,
            "nli_calls": 0,
            "sts_total_ms": 0.0,
            "nli_total_ms": 0.0,
            "nli_skipped": 0
        }

        if verbose:
            print("  ⏳ Loading Fast STS (all-MiniLM-L6-v2)...")
        start = time.time()
        try:
            self.sts_model = SentenceTransformer(sts_model_name)
        except Exception as e:
            raise ImportError(
                f"Failed to load STS model '{sts_model_name}'. "
                f"Install with: pip install sentence-transformers>=5.0 "
                f"(Original error: {e})"
            ) from e
        if verbose:
            print(f"     ✓ Fast STS loaded in {(time.time()-start)*1000:.0f}ms")

        if verbose:
            print("  ⏳ Loading NLI model (deberta-v3-xsmall)...")
        start = time.time()
        try:
            self.nli_model = CrossEncoder(nli_model_name)
        except Exception as e:
            raise ImportError(
                f"Failed to load NLI model '{nli_model_name}'. "
                f"Install with: pip install sentence-transformers>=5.0 "
                f"(Original error: {e})"
            ) from e
        if verbose:
            print(f"     ✓ NLI loaded in {(time.time()-start)*1000:.0f}ms")

        # Step C: SLM fallback (auto-detect or explicit)
        self.slm_verifier = None
        _slm_enabled = use_slm if use_slm is not None else None  # None = auto
        if _slm_enabled is not False:
            try:
                from longtracer.guard.slm_verifier import is_slm_available, SLMVerifier
                if _slm_enabled or is_slm_available():
                    self.slm_verifier = SLMVerifier(verbose=verbose)
                    if verbose:
                        print("     ✓ SLM fallback enabled (auto-detected llama-cpp-python)")
                    logger.info("SLM fallback enabled for uncertain/numeric claims")
            except Exception:
                pass  # SLM not available, continue without it

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        text = re.sub(r'\s+', ' ', text.strip())
        protected = text
        protected = re.sub(r'(\d+)\.(\d+)', r'\1<DEC>\2', protected)
        protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr|i\.e|e\.g)\.\s', r'\1<ABR> ', protected)
        sentences = re.split(r'(?<=[.!?])\s+', protected)

        result = []
        for s in sentences:
            s = s.replace('<DEC>', '.').replace('<ABR>', '.').strip()
            if len(s) > 10:
                result.append(s)
        return result

    def extract_source_sentences(self, source: str) -> List[str]:
        return self.split_into_sentences(source)

    def compute_nli_scores(self, source: str, claim: str) -> Dict[str, float]:
        """Compute NLI probabilities."""
        start = time.time()
        scores = self.nli_model.predict([(source, claim)])
        latency_ms = (time.time() - start) * 1000

        self.latency_log["nli_calls"] += 1
        self.latency_log["nli_total_ms"] += latency_ms

        if len(scores.shape) > 1:
            scores = scores[0]

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2])
        }

    def verify_claim(
        self,
        claim: str,
        contexts: List[str],
        source_metadata: Optional[List[dict]] = None
    ) -> Dict:
        """Verify claim using Fast Batch Encoding + NLI."""
        claim_analysis = analyze_claim(claim)

        claim_sentences = self.split_into_sentences(claim)
        if not claim_sentences:
            claim_sentences = [claim]

        all_source_sentences = []
        source_to_metadata = {}

        for idx, ctx in enumerate(contexts):
            sentences = self.extract_source_sentences(ctx)
            for sent in sentences:
                all_source_sentences.append(sent)
                source_to_metadata[sent] = {
                    "source_idx": idx,
                    "metadata": source_metadata[idx] if source_metadata and idx < len(source_metadata) else {}
                }

        if not all_source_sentences:
            return self._empty_result(claim, claim_analysis)

        sts_start = time.time()
        claim_embs = self.sts_model.encode(claim_sentences, convert_to_tensor=True, show_progress_bar=False)
        source_embs = self.sts_model.encode(all_source_sentences, convert_to_tensor=True, show_progress_bar=False)
        sim_matrix = util.cos_sim(claim_embs, source_embs)
        sts_latency = (time.time() - sts_start) * 1000
        self.latency_log["sts_calls"] += 1
        self.latency_log["sts_total_ms"] += sts_latency

        sentence_results = []
        best_overall_score = 0.0
        best_matching_source = ""
        best_source_index = -1
        best_source_metadata = None

        for i, claim_sent in enumerate(claim_sentences):
            scores = sim_matrix[i]
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            best_match_text = all_source_sentences[best_idx]

            sentence_results.append({
                "claim_sentence": claim_sent,
                "score": best_score,
                "matched_source": best_match_text[:100] + "..."
            })

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_matching_source = best_match_text
                if best_match_text in source_to_metadata:
                    info = source_to_metadata[best_match_text]
                    best_source_index = info["source_idx"]
                    best_source_metadata = info["metadata"]

        avg_score = sum(r["score"] for r in sentence_results) / len(sentence_results) if sentence_results else 0.0

        max_contradiction = 0.0
        entailment_score = 0.0
        nli_ran = False

        if best_matching_source and avg_score >= 0.25:
            nli_result = self.compute_nli_scores(best_matching_source, claim)
            max_contradiction = nli_result["contradiction"]
            entailment_score = nli_result["entailment"]
            nli_ran = True
        else:
            self.latency_log["nli_skipped"] += 1

        is_supported = (
            (avg_score >= self.support_threshold) or (nli_ran and entailment_score > 0.5)
        ) and not (nli_ran and max_contradiction > 0.5)

        match_score_is_low_and_no_nli_rescue = not nli_ran or entailment_score < 0.3
        is_meta_statement = claim_analysis["is_meta_statement"]
        has_hallucination_pattern = claim_analysis["has_hallucination_pattern"]

        is_hallucination = (
            (nli_ran and max_contradiction > 0.5) or
            (avg_score < 0.20 and has_hallucination_pattern and not is_meta_statement and match_score_is_low_and_no_nli_rescue)
        )

        if is_meta_statement:
            is_hallucination = False

        # SLM fallback for uncertain or numeric claims
        if self.slm_verifier and best_matching_source and nli_ran:
            if self._slm_should_verify(claim, entailment_score, max_contradiction):
                slm_result = self.slm_verifier.verify(claim, best_matching_source)
                if "support" in slm_result.get("raw_output", ""):
                    is_supported = True
                    is_hallucination = False
                else:
                    is_supported = False
                    is_hallucination = True

        return {
            "claim": claim,
            "supported": is_supported,
            "score": avg_score,
            "best_score": best_overall_score,
            "sentence_results": sentence_results,
            "contradiction_score": max_contradiction,
            "entailment_score": entailment_score,
            "nli_ran": nli_ran,
            "best_source": best_matching_source[:300] + "...",
            "best_source_index": best_source_index,
            "best_source_metadata": best_source_metadata,
            "is_hallucination": is_hallucination,
            "is_meta_statement": is_meta_statement,
            "has_hallucination_pattern": has_hallucination_pattern
        }

    def verify_claims_batch(
        self,
        claims: List[str],
        contexts: List[str],
        source_metadata: Optional[List[dict]] = None,
        max_workers: int = 8
    ) -> List[Dict]:
        """Verify multiple claims in PARALLEL - optimized batch processing."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not claims:
            return []

        claim_analyses = [analyze_claim(c) for c in claims]

        all_source_sentences = []
        source_to_metadata = {}

        for idx, ctx in enumerate(contexts):
            sentences = self.extract_source_sentences(ctx)
            for sent in sentences:
                all_source_sentences.append(sent)
                source_to_metadata[sent] = {
                    "source_idx": idx,
                    "metadata": source_metadata[idx] if source_metadata and idx < len(source_metadata) else {}
                }

        if not all_source_sentences:
            return [self._empty_result(c, a) for c, a in zip(claims, claim_analyses)]

        all_claim_sentences = []
        claim_sentence_map = []

        for claim_idx, claim in enumerate(claims):
            sentences = self.split_into_sentences(claim)
            if not sentences:
                sentences = [claim]
            for sent_idx, sent in enumerate(sentences):
                all_claim_sentences.append(sent)
                claim_sentence_map.append((claim_idx, sent_idx, sent))

        sts_start = time.time()
        claim_embs = self.sts_model.encode(
            all_claim_sentences, convert_to_tensor=True,
            show_progress_bar=False, batch_size=64
        )
        source_embs = self.sts_model.encode(
            all_source_sentences, convert_to_tensor=True,
            show_progress_bar=False, batch_size=64
        )
        full_sim_matrix = util.cos_sim(claim_embs, source_embs)
        sts_latency = (time.time() - sts_start) * 1000
        self.latency_log["sts_calls"] += 1
        self.latency_log["sts_total_ms"] += sts_latency

        claim_results = {}

        for i, (claim_idx, sent_idx, claim_sent) in enumerate(claim_sentence_map):
            scores = full_sim_matrix[i]
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            best_match_text = all_source_sentences[best_idx]

            if claim_idx not in claim_results:
                claim_results[claim_idx] = {
                    "sentence_results": [],
                    "best_overall_score": 0.0,
                    "best_matching_source": "",
                    "best_source_index": -1,
                    "best_source_metadata": None
                }

            claim_results[claim_idx]["sentence_results"].append({
                "claim_sentence": claim_sent,
                "score": best_score,
                "matched_source": best_match_text[:100] + "..."
            })

            if best_score > claim_results[claim_idx]["best_overall_score"]:
                claim_results[claim_idx]["best_overall_score"] = best_score
                claim_results[claim_idx]["best_matching_source"] = best_match_text
                if best_match_text in source_to_metadata:
                    info = source_to_metadata[best_match_text]
                    claim_results[claim_idx]["best_source_index"] = info["source_idx"]
                    claim_results[claim_idx]["best_source_metadata"] = info["metadata"]

        nli_inputs = []
        for claim_idx, claim in enumerate(claims):
            if claim_idx not in claim_results:
                continue
            cr = claim_results[claim_idx]
            sr = cr["sentence_results"]
            avg_score = sum(r["score"] for r in sr) / len(sr) if sr else 0.0

            if cr["best_matching_source"] and avg_score >= 0.25:
                nli_inputs.append((claim_idx, claim, cr["best_matching_source"]))

        nli_results = {}

        def run_nli(args):
            claim_idx, claim, source = args
            return claim_idx, self.compute_nli_scores(source, claim)

        if nli_inputs:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_nli, inp) for inp in nli_inputs]
                for future in as_completed(futures):
                    claim_idx, scores = future.result()
                    nli_results[claim_idx] = scores

        self.latency_log["nli_skipped"] += len(claims) - len(nli_inputs)

        results = []
        for claim_idx, claim in enumerate(claims):
            analysis = claim_analyses[claim_idx]

            if claim_idx not in claim_results:
                results.append(self._empty_result(claim, analysis))
                continue

            cr = claim_results[claim_idx]
            sr = cr["sentence_results"]
            avg_score = sum(r["score"] for r in sr) / len(sr) if sr else 0.0

            nli_ran = claim_idx in nli_results
            if nli_ran:
                nli_res = nli_results[claim_idx]
                max_contradiction = nli_res["contradiction"]
                entailment_score = nli_res["entailment"]
            else:
                max_contradiction = 0.0
                entailment_score = 0.0

            is_supported = (
                (avg_score >= self.support_threshold) or (nli_ran and entailment_score > 0.5)
            ) and not (nli_ran and max_contradiction > 0.5)

            match_score_is_low_and_no_nli_rescue = not nli_ran or entailment_score < 0.3
            is_meta_statement = analysis["is_meta_statement"]
            has_hallucination_pattern = analysis["has_hallucination_pattern"]

            is_hallucination = (
                (nli_ran and max_contradiction > 0.5) or
                (avg_score < 0.20 and has_hallucination_pattern and not is_meta_statement and match_score_is_low_and_no_nli_rescue)
            )

            if is_meta_statement:
                is_hallucination = False

            # SLM fallback for uncertain or numeric claims
            best_src = cr["best_matching_source"]
            if self.slm_verifier and best_src and nli_ran:
                if self._slm_should_verify(claim, entailment_score, max_contradiction):
                    slm_result = self.slm_verifier.verify(claim, best_src)
                    if "support" in slm_result.get("raw_output", ""):
                        is_supported = True
                        is_hallucination = False
                    else:
                        is_supported = False
                        is_hallucination = True

            results.append({
                "claim": claim,
                "supported": is_supported,
                "score": avg_score,
                "best_score": cr["best_overall_score"],
                "sentence_results": sr,
                "contradiction_score": max_contradiction,
                "entailment_score": entailment_score,
                "nli_ran": nli_ran,
                "best_source": cr["best_matching_source"][:300] + "..." if cr["best_matching_source"] else "",
                "best_source_index": cr["best_source_index"],
                "best_source_metadata": cr["best_source_metadata"],
                "is_hallucination": is_hallucination,
                "is_meta_statement": is_meta_statement,
                "has_hallucination_pattern": has_hallucination_pattern
            })

        return results

    @staticmethod
    def _slm_should_verify(claim: str, entailment: float, contradiction: float) -> bool:
        """
        Decide if a claim should be sent to the SLM fallback.

        Only triggers for claims containing numbers/dates, where the NLI
        cross-encoder is known to be unreliable:
        1. Claim has numbers AND NLI says entailed (might be wrong year/amount)
        2. Claim has numbers AND NLI says contradicted (might be valid approximation)
        3. Claim has numbers AND NLI is uncertain (can't decide)
        """
        has_numbers = bool(re.search(r'\d{2,}', claim))
        if not has_numbers:
            return False

        # NLI is uncertain on a numeric claim
        is_uncertain = entailment < 0.5 and contradiction < 0.5

        # NLI says entailed but might be wrong (e.g., wrong year)
        suspicious_entailment = entailment > 0.5 and contradiction < 0.3

        # NLI says contradicted but might be valid approximation
        suspicious_contradiction = contradiction > 0.5

        return is_uncertain or suspicious_entailment or suspicious_contradiction

    def _empty_result(self, claim, analysis):
        return {
            "claim": claim,
            "supported": False,
            "score": 0.0,
            "best_score": 0.0,
            "sentence_results": [],
            "contradiction_score": 0.0,
            "entailment_score": 0.0,
            "nli_ran": False,
            "best_source": "",
            "best_source_index": -1,
            "best_source_metadata": None,
            "is_hallucination": False,
            "is_meta_statement": analysis["is_meta_statement"],
            "has_hallucination_pattern": analysis["has_hallucination_pattern"]
        }

    def get_latency_stats(self) -> Dict:
        sts_avg = (self.latency_log["sts_total_ms"] / self.latency_log["sts_calls"]
                   if self.latency_log["sts_calls"] > 0 else 0)
        nli_avg = (self.latency_log["nli_total_ms"] / self.latency_log["nli_calls"]
                   if self.latency_log["nli_calls"] > 0 else 0)

        return {
            "sts_calls": self.latency_log["sts_calls"],
            "sts_avg_ms": sts_avg,
            "nli_calls": self.latency_log["nli_calls"],
            "nli_avg_ms": nli_avg,
            "nli_skipped": self.latency_log["nli_skipped"],
            "total_ms": self.latency_log["sts_total_ms"] + self.latency_log["nli_total_ms"]
        }

    def reset_latency_log(self):
        self.latency_log = {
            "sts_calls": 0, "nli_calls": 0, "sts_total_ms": 0.0, "nli_total_ms": 0.0, "nli_skipped": 0
        }

    def get_slm_stats(self) -> Optional[Dict]:
        """Return SLM fallback statistics, or None if SLM is not enabled."""
        if self.slm_verifier:
            return self.slm_verifier.get_stats()
        return None


# Backward compatibility
NLIModel = HybridVerificationModel

# ── Shared model singleton ──────────────────────────────────────────
# Avoids reloading STS + NLI + SLM on every CitationVerifier() call.
# First access loads models (~10s), subsequent access returns instantly.

_shared_model: Optional[HybridVerificationModel] = None


def get_shared_model(**kwargs) -> HybridVerificationModel:
    """Get or create the shared HybridVerificationModel singleton.

    Models are loaded once and reused across all CitationVerifier instances.
    Pass any kwargs to override model defaults on first creation.

    Returns:
        The shared HybridVerificationModel instance.
    """
    global _shared_model
    if _shared_model is None:
        _shared_model = HybridVerificationModel(**kwargs)
    return _shared_model


def reset_shared_model() -> None:
    """Clear the shared model (useful for testing)."""
    global _shared_model
    _shared_model = None

