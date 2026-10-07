"""
Run Manifest Contract — Provenance and environment metadata for evaluation runs.

Captures sufficient context to explain comparisons, guarantee auditability,
and ensure reproducibility across different machines and CI environments.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class RunManifest(BaseModel):
    """
    Provenance manifest recording the parameters, environment, and outcome
    of an evaluation execution.

    Attributes:
        schema_version: Version identifier for contract schema compatibility.
        run_id: Unique identifier for this evaluation run.
        dataset_digest: Cryptographic SHA-256 digest of the evaluated dataset.
        runner_identity: Name or import path of the runner callback (if app mode).
        evaluator_fingerprint: Hash or identifier of the evaluator model & config.
        policy_fingerprint: Hash or identifier of the applied quality gate rules.
        application_commit: Git commit SHA of the evaluated application.
        application_config_fingerprint: Hash of application settings or prompt templates.
        platform: System information (OS, Python version, hardware).
        device: Execution device (e.g. 'cpu', 'cuda:0').
        created_at: ISO-8601 UTC timestamp of run initiation.
        completed_at: ISO-8601 UTC timestamp of run completion.
        mode: Evaluation mode ('saved' or 'app').
        seed: Execution seed used for local reproducibility (if applicable).
        repetition_count: Number of evaluation repetitions per case.
        status_counts: Summary mapping of outcome counts (e.g. {'PASS': 42, 'FAIL': 3}).
        metadata: Arbitrary user-defined key-value attributes.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1"
    run_id: str
    dataset_digest: str
    evaluator_fingerprint: str
    runner_identity: Optional[str] = None
    policy_fingerprint: Optional[str] = None
    application_commit: Optional[str] = None
    application_config_fingerprint: Optional[str] = None
    platform: Dict[str, Any] = Field(default_factory=dict)
    device: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    mode: str = "saved"
    seed: Optional[int] = None
    repetition_count: int = 1
    status_counts: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
