"""
Source Evidence Contract — Normalized representation of grounding evidence.

Every piece of evidence supplied to LongTracer for claim verification is
modeled as a SourceEvidence instance with deterministic hashing and source
identity metadata.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


def compute_text_hash(text: str) -> str:
    """Compute a deterministic SHA-256 hex digest for evidence text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SourceEvidence(BaseModel):
    """
    Normalized representation of a single evidence document or passage chunk.

    Attributes:
        schema_version: Version identifier for contract schema compatibility.
        source_id: Unique identifier for this evidence item within a verification case.
        text: Raw textual content of the evidence.
        text_hash: SHA-256 digest of the text content. Automatically populated if omitted.
        document_id: Identifier of the parent document (optional).
        document_version: Version string or commit hash of the document (optional).
        chunk_id: Identifier of the specific chunk or segment (optional).
        page: Page number if source is paginated (optional).
        section: Section name or header path (optional).
        uri: Location metadata (e.g. s3://, file://, https://). Treated strictly
             as metadata — LongTracer never automatically fetches arbitrary URLs.
        provenance: Optional metadata describing retrieval score, rank, or retrieval pipeline.
        metadata: Arbitrary user-defined key-value attributes.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_version: str = "1"
    source_id: str
    text: str
    text_hash: str = ""
    document_id: Optional[str] = None
    document_version: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    uri: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_text_hash(self) -> SourceEvidence:
        """Automatically populate text_hash from text if not explicitly provided."""
        if not self.text_hash:
            self.text_hash = compute_text_hash(self.text)
        return self
