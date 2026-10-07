"""
LongTracer Contracts — Portable, versioned schemas and evaluation data models.

Exports:
  - Evidence: SourceEvidence, compute_text_hash
  - Results: ExecutionStatus, AssessmentAvailability, ClaimAssessment, QualityGate,
             ClaimResult, CaseResult, LegacyVerificationAdapter
  - Case: ApplicationOutput, TestCase
  - Run: RunManifest
  - Review: ReviewState, ReviewRecord, BaselineRecord
"""

from longtracer.contracts.case import ApplicationOutput, TestCase
from longtracer.contracts.evidence import SourceEvidence, compute_text_hash
from longtracer.contracts.result import (
    AssessmentAvailability,
    CaseResult,
    ClaimAssessment,
    ClaimResult,
    ExecutionStatus,
    LegacyVerificationAdapter,
    QualityGate,
    ReasonCode,
    compute_quality_gate,
    summarize_case,
    unassessed_case,
)
from longtracer.contracts.review import BaselineRecord, ReviewRecord, ReviewState
from longtracer.contracts.run import RunManifest

__all__ = [
    # Evidence
    "SourceEvidence",
    "compute_text_hash",
    # Results & Layers
    "ExecutionStatus",
    "AssessmentAvailability",
    "ClaimAssessment",
    "QualityGate",
    "ReasonCode",
    "compute_quality_gate",
    "summarize_case",
    "unassessed_case",
    "ClaimResult",
    "CaseResult",
    "LegacyVerificationAdapter",
    # Cases & Runners
    "ApplicationOutput",
    "TestCase",
    # Run Provenance
    "RunManifest",
    # Review & Baselines
    "ReviewState",
    "ReviewRecord",
    "BaselineRecord",
]
