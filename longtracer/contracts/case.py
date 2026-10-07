"""
Case & Application Output Contracts.

Defines the structure for regression test cases and outputs produced by
tested applications or static snapshots.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from longtracer.contracts.evidence import SourceEvidence


class ApplicationOutput(BaseModel):
    """
    Standard output produced by an application callback or live model query.

    Attributes:
        schema_version: Version identifier for contract schema compatibility.
        response: Generated text response from the model/application.
        response_type: Format or nature of the output (e.g. 'text', 'json', 'refusal').
        sources: Evidence passages supplied or retrieved for this generation.
        citations: Citation markers or spans identified in the response.
        metadata: Application-specific context (model name, prompt tokens, etc.).
        latency_ms: Execution duration in milliseconds.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1"
    response: str
    response_type: Optional[str] = "text"
    sources: List[SourceEvidence] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[float] = None


class TestCase(BaseModel):
    """
    A single evaluation test case within a LongTracer regression dataset.

    Supports two primary evaluation modes:
      - `saved`: Evaluates an existing saved answer + evidence snapshot.
      - `app`: Invokes a trusted application runner callback to generate a fresh answer.

    Attributes:
        schema_version: Version identifier for contract schema compatibility.
        case_id: Unique and stable identifier for this test case.
        case_revision: Monotonically increasing revision number for case edits.
        mode: Evaluation mode ('saved' or 'app').
        question: User query or input prompt.
        conversation: Optional preceding multi-turn dialogue history.
        tags: Categorization tags (e.g. ['finance', 'refusal', 'edge-case']).
        expectations: High-level expected outcomes (e.g. expect_refusal, max_hallucinations).
        assertions: Explicit validation assertions (e.g. forbidden_claims, required_sources).
        provenance: Origin tracking metadata (e.g. source trace_id, author, dataset name).
        saved_answer: Static answer snapshot (used in 'saved' mode).
        saved_evidence: Static evidence snapshot (used in 'saved' mode).
    """

    model_config = ConfigDict(extra="ignore")
    __test__ = False

    schema_version: str = "1"
    case_id: str
    case_revision: int = 1
    mode: Literal["saved", "app"] = "saved"
    question: str
    conversation: Optional[List[Dict[str, str]]] = None
    tags: List[str] = Field(default_factory=list)
    expectations: Dict[str, Any] = Field(default_factory=dict)
    assertions: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[Dict[str, Any]] = None
    saved_answer: Optional[str] = None
    saved_evidence: List[SourceEvidence] = Field(default_factory=list)
