"""
Result Contracts — Typed multi-layer result states and verification outcomes.

Replaces the single opaque float trust score with explicit execution,
availability, claim assessment, and quality gate layers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ExecutionStatus(str, Enum):
    """Execution status of the verification pipeline."""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class AssessmentAvailability(str, Enum):
    """Availability of assessable factual claims in the application output."""
    ASSESSED = "ASSESSED"
    NO_ASSESSABLE_CLAIMS = "NO_ASSESSABLE_CLAIMS"
    NOT_EVALUATED = "NOT_EVALUATED"


class ClaimAssessment(str, Enum):
    """Grounding assessment for an individual factual claim against evidence."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_SOURCES = "CONFLICTING_SOURCES"


class QualityGate(str, Enum):
    """Final decision of the verification quality gate for CI or runtime policy."""
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


class ReasonCode(str, Enum):
    """Reason code providing explainability for claim- and case-level outcomes."""
    # Case execution and availability reasons
    INVALID_INPUT = "INVALID_INPUT"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    NO_EXTRACTABLE_CLAIMS = "NO_EXTRACTABLE_CLAIMS"
    NO_SOURCES_SUPPLIED = "NO_SOURCES_SUPPLIED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    EVALUATION_TIMEOUT = "EVALUATION_TIMEOUT"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    HONEST_UNCERTAINTY_ONLY = "HONEST_UNCERTAINTY_ONLY"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    EXECUTION_SKIPPED = "EXECUTION_SKIPPED"

    # Claim assessment reasons
    SUPPORTED_BY_EVIDENCE = "SUPPORTED_BY_EVIDENCE"
    CONTRADICTED_BY_EVIDENCE = "CONTRADICTED_BY_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    LOW_EVIDENCE_SIMILARITY = "LOW_EVIDENCE_SIMILARITY"
    NOT_ENTAILED = "NOT_ENTAILED"
    HONEST_UNCERTAINTY = "HONEST_UNCERTAINTY"
    NO_SOURCE_TEXT = "NO_SOURCE_TEXT"


class ClaimResult(BaseModel):
    """
    Verification outcome for a single decomposed claim.

    Attributes:
        claim_id: Unique identifier for the claim within the case.
        claim_text: Text of the factual claim that was verified.
        assessment: Evaluator assessment against supplied evidence.
        availability: Whether this claim was assessable.
        reason: Structured reason code explaining the assessment.
        supporting_sources: List of source_ids that support this claim.
        contradicting_sources: List of source_ids that contradict this claim.
        confidence: Evaluator model confidence score (0.0 to 1.0).
        details: Explanation or rationale from the verifier.
        char_start: Character offset start in the parent response text (optional).
        char_end: Character offset end in the parent response text (optional).
    """

    model_config = ConfigDict(extra="ignore")

    claim_id: str
    claim_text: str
    assessment: ClaimAssessment
    availability: AssessmentAvailability = AssessmentAvailability.ASSESSED
    reason: Optional[ReasonCode] = None
    supporting_sources: List[str] = Field(default_factory=list)
    contradicting_sources: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    details: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class CaseResult(BaseModel):
    """
    Full verification result for an application response across all claims.

    Attributes:
        schema_version: Version identifier for contract schema compatibility.
        case_id: Identifier of the evaluated test case or request (optional).
        execution: Pipeline execution status (SUCCESS, ERROR, TIMEOUT, etc.).
        availability: Claim availability status (ASSESSED, NO_ASSESSABLE_CLAIMS).
        quality_gate: Policy gate outcome (PASS, FAIL, INDETERMINATE).
        reason: Structured reason code explaining the case-level outcome.
        claims: Detailed assessment for each evaluated claim.
        trust_score: Backward-compatibility trust score (0.0 to 1.0).
        summary: Human-readable evaluation summary.
        error_message: Error description if execution failed.
        latency_ms: Total verification latency in milliseconds.
        metadata: Arbitrary user-defined key-value attributes.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1"
    case_id: Optional[str] = None
    execution: ExecutionStatus = ExecutionStatus.SUCCESS
    availability: AssessmentAvailability = AssessmentAvailability.ASSESSED
    quality_gate: QualityGate = QualityGate.PASS
    reason: Optional[ReasonCode] = None
    claims: List[ClaimResult] = Field(default_factory=list)
    trust_score: float = 0.0
    summary: str = ""
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def supported_claims_count(self) -> int:
        """Count of claims with assessment == SUPPORTED."""
        return sum(1 for c in self.claims if c.assessment == ClaimAssessment.SUPPORTED)

    @property
    def contradicted_claims_count(self) -> int:
        """Count of claims with assessment == CONTRADICTED."""
        return sum(1 for c in self.claims if c.assessment == ClaimAssessment.CONTRADICTED)

    @property
    def ungrounded_claims_count(self) -> int:
        """Count of claims not fully supported."""
        return sum(1 for c in self.claims if c.assessment != ClaimAssessment.SUPPORTED)


def compute_quality_gate(case: CaseResult) -> QualityGate:
    """Compute the quality gate outcome from case execution, availability, and claims.

    Policy layer: completely separated from claim-level measurement.
    """
    if case.execution != ExecutionStatus.SUCCESS:
        return QualityGate.INDETERMINATE
    if case.availability == AssessmentAvailability.NOT_EVALUATED:
        return QualityGate.INDETERMINATE
    if case.availability == AssessmentAvailability.NO_ASSESSABLE_CLAIMS:
        # Policy choice (documented): an honest refusal may pass; anything else with
        # no assessable claims (empty, too short, unknown) never passes.
        return QualityGate.PASS if case.reason == ReasonCode.HONEST_UNCERTAINTY_ONLY else QualityGate.FAIL
    assessed = [c for c in case.claims if c.availability == AssessmentAvailability.ASSESSED]
    if not assessed:
        # ASSESSED availability with nothing actually assessed is not a pass.
        return QualityGate.FAIL
    if all(c.assessment == ClaimAssessment.SUPPORTED for c in assessed):
        return QualityGate.PASS
    return QualityGate.FAIL


_CASE_REASON_SUMMARIES: Dict[ReasonCode, str] = {
    ReasonCode.EMPTY_RESPONSE: "Response is empty or whitespace-only; nothing could be assessed.",
    ReasonCode.NO_EXTRACTABLE_CLAIMS: "Response contains no extractable claims (too short to assess).",
    ReasonCode.HONEST_UNCERTAINTY_ONLY: "Response is an honest refusal / statement of uncertainty; grounding not applicable.",
    ReasonCode.INVALID_INPUT: "Input could not be evaluated (invalid or unsupported input).",
    ReasonCode.MODEL_UNAVAILABLE: "Evaluator model unavailable; nothing was assessed.",
    ReasonCode.EVALUATION_FAILED: "Evaluator failed while scoring; nothing was assessed.",
    ReasonCode.EVALUATION_TIMEOUT: "Evaluation timed out; nothing was assessed.",
    ReasonCode.EXECUTION_CANCELLED: "Evaluation was cancelled; nothing was assessed.",
}


def summarize_case(claims: List[ClaimResult], reason: Optional[ReasonCode] = None) -> str:
    """Build a human-readable summary that states the reason when nothing was assessed."""
    prefix = ""
    if reason in _CASE_REASON_SUMMARIES and (not claims or reason == ReasonCode.HONEST_UNCERTAINTY_ONLY):
        return _CASE_REASON_SUMMARIES[reason]  # type: ignore[index]
    if reason == ReasonCode.NO_SOURCES_SUPPLIED:
        prefix = "No sources were supplied. "
    assessed = [c for c in claims if c.availability == AssessmentAvailability.ASSESSED]
    counts = {a: sum(1 for c in assessed if c.assessment == a) for a in ClaimAssessment}
    parts = [f"{counts[ClaimAssessment.SUPPORTED]}/{len(assessed)} assessed claim(s) supported"]
    for label, name in (
        (ClaimAssessment.CONTRADICTED, "contradicted"),
        (ClaimAssessment.INSUFFICIENT_EVIDENCE, "insufficient evidence"),
        (ClaimAssessment.CONFLICTING_SOURCES, "conflicting sources"),
    ):
        if counts[label]:
            parts.append(f"{counts[label]} {name}")
    not_evaluated = len(claims) - len(assessed)
    if not_evaluated:
        parts.append(f"{not_evaluated} not evaluated (honest uncertainty)")
    return prefix + ", ".join(parts) + "."


def unassessed_case(
    execution: ExecutionStatus,
    reason: ReasonCode,
    *,
    case_id: Optional[str] = None,
    error_message: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> CaseResult:
    """Build a CaseResult for a run where nothing was assessed (error, timeout, cancel, bad input).

    Availability is always ``NOT_EVALUATED`` and no partial claims are reported.
    """
    case = CaseResult(
        case_id=case_id,
        execution=execution,
        availability=AssessmentAvailability.NOT_EVALUATED,
        quality_gate=QualityGate.INDETERMINATE,
        reason=reason,
        claims=[],
        trust_score=0.0,
        summary=_CASE_REASON_SUMMARIES.get(reason, "Nothing was assessed."),
        error_message=error_message,
        latency_ms=latency_ms,
    )
    case.quality_gate = compute_quality_gate(case)
    return case


class LegacyVerificationAdapter:
    """
    Bidirectional adapter between legacy VerificationResult and modern CaseResult.

    Ensures zero disruption to existing applications, downstream integrations,
    and storage engines.
    """

    @staticmethod
    def _source_id(index: Any, metadata: Any) -> Optional[str]:
        """Return a stable source identifier for a legacy best-source index."""
        if isinstance(metadata, dict):
            for key in ("source_id", "id"):
                if metadata.get(key) is not None:
                    return str(metadata[key])
        if isinstance(index, int) and index >= 0:
            return f"source_{index}"
        return None

    @staticmethod
    def _claim_from_signals(cid: str, item: Dict[str, Any]) -> ClaimResult:
        """Map one raw verifier claim dict onto a ClaimResult (thresholds unchanged)."""
        ctext = str(item.get("claim", item.get("text", "")))
        is_meta = bool(item.get("is_meta_statement", False))
        is_supp = bool(item.get("supported", False))
        nli_ran = bool(item.get("nli_ran", False))
        score = float(item.get("score", 0.0) or 0.0)
        contra_score = float(item.get("contradiction_score", 0.0) or 0.0)
        best_index = item.get("best_source_index", -1)
        src_id = LegacyVerificationAdapter._source_id(best_index, item.get("best_source_metadata"))
        no_source_text = (not item.get("best_source")) and best_index == -1 and not nli_ran

        availability = AssessmentAvailability.ASSESSED
        supp: List[str] = []
        contra: List[str] = []
        if is_meta and not is_supp:
            assessment = ClaimAssessment.INSUFFICIENT_EVIDENCE
            availability = AssessmentAvailability.NOT_EVALUATED
            reason = ReasonCode.HONEST_UNCERTAINTY
            confidence = 1.0
        elif is_supp:
            assessment = ClaimAssessment.SUPPORTED
            reason = ReasonCode.SUPPORTED_BY_EVIDENCE
            supp = [src_id] if src_id else []
            confidence = score
        elif nli_ran and contra_score > 0.5:
            assessment = ClaimAssessment.CONTRADICTED
            reason = ReasonCode.CONTRADICTED_BY_EVIDENCE
            contra = [src_id] if src_id else []
            confidence = contra_score
        elif no_source_text:
            assessment = ClaimAssessment.INSUFFICIENT_EVIDENCE
            reason = ReasonCode.NO_SOURCE_TEXT
            confidence = 0.0
        else:
            assessment = ClaimAssessment.INSUFFICIENT_EVIDENCE
            reason = ReasonCode.NOT_ENTAILED if nli_ran else ReasonCode.LOW_EVIDENCE_SIMILARITY
            confidence = score

        details = (
            f"sts_similarity={score:.3f}; nli_ran={nli_ran}; "
            f"contradiction={contra_score:.3f}; "
            f"legacy_entailment_field={float(item.get('entailment_score', 0.0) or 0.0):.3f}"
        )
        return ClaimResult(
            claim_id=cid,
            claim_text=ctext,
            assessment=assessment,
            availability=availability,
            reason=reason,
            supporting_sources=supp,
            contradicting_sources=contra,
            confidence=min(1.0, max(0.0, confidence)),
            details=details,
        )

    @staticmethod
    def from_legacy(
        legacy: Any,
        case_id: Optional[str] = None,
        *,
        case_reason: Optional[ReasonCode] = None,
    ) -> CaseResult:
        """
        Convert a legacy VerificationResult dataclass into a modern CaseResult.

        Args:
            legacy: A ``VerificationResult`` (or any object with the same attributes).
            case_id: Optional identifier for the case.
            case_reason: Optional case-level reason known to the caller but not
                recoverable from the legacy object (e.g. ``EMPTY_RESPONSE`` vs
                ``NO_EXTRACTABLE_CLAIMS``, or ``NO_SOURCES_SUPPLIED``).

        Mapping:
          - No claims: ``NO_ASSESSABLE_CLAIMS``; the gate is ``FAIL`` (never ``PASS``).
          - Real verifier claim dicts (with ``supported`` / ``nli_ran`` keys) are
            mapped from raw signals without changing any threshold.
          - Other claim dicts and strings use the PR #20 text-based fallback.
          - Refusal-only responses: ``NO_ASSESSABLE_CLAIMS`` + ``HONEST_UNCERTAINTY_ONLY``.
          - The gate is always computed by ``compute_quality_gate()``.
        """
        claims_list: List[ClaimResult] = []
        raw_claims = getattr(legacy, "claims", []) or []
        flagged_raw = getattr(legacy, "flagged_claims", []) or []
        hallucinations_raw = getattr(legacy, "hallucinations", []) or []

        contradicted_texts = {h.get("claim", "") if isinstance(h, dict) else str(h) for h in hallucinations_raw}
        flagged_texts = {f.get("claim", "") if isinstance(f, dict) else str(f) for f in flagged_raw}

        for idx, item in enumerate(raw_claims):
            cid = f"claim_{idx + 1}"
            if isinstance(item, dict) and ("supported" in item or "nli_ran" in item):
                claims_list.append(LegacyVerificationAdapter._claim_from_signals(cid, item))
                continue

            # Text-based fallback (PR #20 behaviour for synthetic / non-verifier dicts)
            if isinstance(item, dict):
                ctext = str(item.get("claim", item.get("text", "")))
                sources_val = item.get("sources", [])
                sources_list: List[str] = [str(s) for s in sources_val] if isinstance(sources_val, list) else []
                raw_conf = item.get("confidence", item.get("score", 1.0))
                try:
                    conf = float(raw_conf) if raw_conf is not None else 1.0
                except (ValueError, TypeError):
                    conf = 1.0
            else:
                ctext = str(item)
                sources_list = []
                conf = 1.0

            supp: List[str] = []
            contra: List[str] = []
            if ctext in contradicted_texts:
                assessment = ClaimAssessment.CONTRADICTED
                reason = ReasonCode.CONTRADICTED_BY_EVIDENCE
                contra = sources_list
            elif ctext in flagged_texts:
                assessment = ClaimAssessment.INSUFFICIENT_EVIDENCE
                reason = ReasonCode.LOW_EVIDENCE_SIMILARITY
            else:
                assessment = ClaimAssessment.SUPPORTED
                reason = ReasonCode.SUPPORTED_BY_EVIDENCE
                supp = sources_list
            claims_list.append(
                ClaimResult(
                    claim_id=cid,
                    claim_text=ctext,
                    assessment=assessment,
                    reason=reason,
                    supporting_sources=supp,
                    contradicting_sources=contra,
                    confidence=min(1.0, max(0.0, conf)),
                )
            )

        if not claims_list:
            availability = AssessmentAvailability.NO_ASSESSABLE_CLAIMS
            reason_out: Optional[ReasonCode] = case_reason or ReasonCode.NO_EXTRACTABLE_CLAIMS
        elif all(c.availability == AssessmentAvailability.NOT_EVALUATED for c in claims_list):
            availability = AssessmentAvailability.NO_ASSESSABLE_CLAIMS
            reason_out = case_reason or ReasonCode.HONEST_UNCERTAINTY_ONLY
        else:
            availability = AssessmentAvailability.ASSESSED
            reason_out = case_reason

        if reason_out == ReasonCode.NO_SOURCES_SUPPLIED:
            for c in claims_list:
                if c.reason in (ReasonCode.NO_SOURCE_TEXT, ReasonCode.LOW_EVIDENCE_SIMILARITY):
                    c.reason = ReasonCode.NO_SOURCES_SUPPLIED

        latency_stats = getattr(legacy, "latency_stats", None)
        latency_ms = latency_stats.get("total_ms") if isinstance(latency_stats, dict) else None

        trust_score_val = getattr(legacy, "trust_score", 0.0)
        try:
            trust_score = float(trust_score_val) if trust_score_val is not None else 0.0
        except (ValueError, TypeError):
            trust_score = 0.0

        case_res = CaseResult(
            case_id=case_id,
            execution=ExecutionStatus.SUCCESS,
            availability=availability,
            quality_gate=QualityGate.INDETERMINATE,
            reason=reason_out,
            claims=claims_list,
            trust_score=trust_score,
            summary=summarize_case(claims_list, reason_out),
            latency_ms=latency_ms,
        )
        case_res.quality_gate = compute_quality_gate(case_res)
        return case_res

    @staticmethod
    def to_legacy(case_result: CaseResult) -> Any:
        """
        Convert a modern CaseResult back into a legacy VerificationResult dataclass.
        """
        # Lazy import to avoid circular dependency
        from longtracer.guard.verifier import VerificationResult

        claims_dicts = []
        flagged_dicts = []
        hallucinations_dicts = []

        for c in case_result.claims:
            cdict = {
                "claim": c.claim_text,
                "assessment": c.assessment.value,
                "confidence": c.confidence,
                "sources": c.supporting_sources,
            }
            claims_dicts.append(cdict)
            if c.assessment in (ClaimAssessment.INSUFFICIENT_EVIDENCE, ClaimAssessment.CONFLICTING_SOURCES):
                flagged_dicts.append(cdict)
            elif c.assessment == ClaimAssessment.CONTRADICTED:
                flagged_dicts.append(cdict)
                hallucinations_dicts.append(cdict)

        all_supported = (len(flagged_dicts) == 0)
        verdict = "PASS" if case_result.quality_gate == QualityGate.PASS else "FAIL"

        return VerificationResult(
            trust_score=case_result.trust_score,
            claims=claims_dicts,
            flagged_claims=flagged_dicts,
            hallucinations=hallucinations_dicts,
            all_supported=all_supported,
            hallucination_count=len(hallucinations_dicts),
            verdict=verdict,
            summary=case_result.summary,
            latency_stats={"total_ms": case_result.latency_ms} if case_result.latency_ms is not None else None,
        )
