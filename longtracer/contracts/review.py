"""
Review & Baseline Contracts — Explicit human review and baseline promotion records.

Implements the core principle that production traces and benchmark baselines
require explicit human authorization before promotion into permanent regression suites.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ReviewState(str, Enum):
    """Workflow state for a trace or candidate evaluation case undergoing human review."""
    UNREVIEWED = "UNREVIEWED"
    CONFIRMED_BUG = "CONFIRMED_BUG"
    EXPECTED_BEHAVIOR = "EXPECTED_BEHAVIOR"
    ARCHIVED = "ARCHIVED"


class ReviewRecord(BaseModel):
    """
    Audit record documenting human review of a trace or candidate test case.

    Attributes:
        schema_version: Version identifier for contract schema compatibility.
        review_id: Unique identifier for this review action.
        trace_id: Associated production/staging trace ID (if promoted from live trace).
        case_id: Identifier of the test case created or inspected.
        state: Assigned human review state.
        reviewer: Name, email, or GitHub handle of the human reviewer.
        reviewed_at: ISO-8601 UTC timestamp of review decision.
        notes: Contextual notes, bug description, or rationale.
        tags: Categorization tags.
        metadata: Arbitrary user-defined key-value attributes.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1"
    review_id: str
    trace_id: Optional[str] = None
    case_id: Optional[str] = None
    state: ReviewState = ReviewState.UNREVIEWED
    reviewer: Optional[str] = None
    reviewed_at: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaselineRecord(BaseModel):
    """
    Explicit authorization record establishing a specific evaluation run
    as the golden regression baseline.

    Attributes:
        schema_version: Version identifier for contract schema compatibility.
        baseline_id: Unique identifier for this approved baseline.
        run_id: Reference to the source evaluation run manifest.
        dataset_digest: Cryptographic digest of the dataset evaluated.
        evaluator_fingerprint: Fingerprint of the verifier models & thresholds.
        policy_fingerprint: Fingerprint of the pass/fail policy rules.
        approver: Identity of the person approving this baseline.
        approved_at: ISO-8601 UTC timestamp of approval.
        reason: Justification or release context for establishing this baseline.
        superseded_baseline_id: Identifier of the previous baseline this replaces.
        metadata: Arbitrary user-defined key-value attributes.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1"
    baseline_id: str
    run_id: str
    dataset_digest: str
    evaluator_fingerprint: str
    policy_fingerprint: str
    approver: str
    approved_at: str
    reason: str
    superseded_baseline_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
