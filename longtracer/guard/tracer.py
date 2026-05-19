"""
Unified Tracer - Uses pluggable cache backend for trace storage.

Supports multiple projects — each Tracer instance is scoped to a project_name.
All backend operations are wrapped in try/except so the tracer never crashes
the caller's pipeline.

Usage:
    from longtracer.guard.tracer import Tracer

    tracer = Tracer(project_name="my-chatbot")
    tracer = Tracer(backend_type="sqlite", path="./traces.db")
    tracer = Tracer(backend_type="mongo", uri="mongodb://localhost:27017")
    tracer = Tracer(backend_type="memory")
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from datetime import datetime
from uuid import uuid4

from .cache import TraceCacheBackend, create_backend, get_default_backend

logger = logging.getLogger("longtracer")


class Tracer:
    """
    Unified tracer for RAG guardrail debugging with pluggable storage.

    Each Tracer instance is scoped to a ``project_name``. Multiple Tracer
    instances can coexist in the same process, each tracking a different
    project against the same or different backends.

    All backend I/O is wrapped in try/except — a failing backend will never
    crash the caller's pipeline.
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        run_name: str = "longtracer_pipeline",
        backend: Optional[TraceCacheBackend] = None,
        backend_type: Optional[str] = None,
        **backend_kwargs,
    ):
        self.project_name = project_name or os.environ.get(
            "TRACE_PROJECT", "longtracer"
        )
        self.run_name = run_name
        self.root_run: Optional[Dict[str, Any]] = None
        self._run_stack: List[Dict[str, Any]] = []
        self.claim_evidence_map: Dict[str, Dict[str, float]] = {}

        if backend is not None:
            self.backend = backend
        elif backend_type is not None:
            self.backend = create_backend(backend_type, **backend_kwargs)
        else:
            self.backend = get_default_backend()

    # ── safe backend helpers ────────────────────────────────────

    def _safe_save_run(self, run: Dict[str, Any]) -> None:
        try:
            self.backend.save_run(run)
        except Exception as exc:
            logger.warning("Tracer: save_run failed (%s), continuing", exc)

    def _safe_update_run(self, run_id: str, updates: Dict[str, Any]) -> None:
        try:
            self.backend.update_run(run_id, updates)
        except Exception as exc:
            logger.warning("Tracer: update_run failed (%s), continuing", exc)

    def _safe_save_trace(self, trace: Dict[str, Any]) -> None:
        try:
            self.backend.save_trace(trace)
        except Exception as exc:
            logger.warning("Tracer: save_trace failed (%s), continuing", exc)

    # ── root lifecycle ──────────────────────────────────────────

    def start_root(self, inputs: Optional[Dict[str, Any]] = None):
        """Start the root trace run."""
        trace_id = str(uuid4())
        self.root_run = {
            "trace_id": trace_id,
            "run_id": trace_id,
            "name": self.run_name,
            "run_type": "chain",
            "project_name": self.project_name,
            "inputs": inputs or {},
            "outputs": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "start_time": time.time(),
            "end_time": None,
            "duration_ms": None,
            "parent_id": None,
            "children": [],
            "error": None,
        }
        self._run_stack.append(self.root_run)
        self._safe_save_run(self.root_run.copy())

    def end_root(
        self,
        outputs: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """End the root trace run and save to backend.

        Args:
            outputs: Optional output data for the trace.
            metrics: Optional verification metrics dict with keys:
                trust_score (float), hallucination_count (int),
                claim_count (int). When provided, these are stored as
                top-level fields in the trace document for metrics
                aggregation.
        """
        if not self.root_run:
            return

        end_time = time.time()
        duration_ms = (end_time - self.root_run["start_time"]) * 1000

        self.root_run["outputs"] = outputs or {
            "claim_evidence_map": self.claim_evidence_map,
        }
        self.root_run["end_time"] = end_time
        self.root_run["duration_ms"] = duration_ms
        self.root_run["updated_at"] = datetime.utcnow()

        self._safe_update_run(
            self.root_run["run_id"],
            {
                "outputs": self.root_run["outputs"],
                "end_time": self.root_run["end_time"],
                "duration_ms": self.root_run["duration_ms"],
                "updated_at": self.root_run["updated_at"],
            },
        )

        trace_doc = {
            "trace_id": self.root_run["trace_id"],
            "project_name": self.project_name,
            "run_name": self.run_name,
            "inputs": self.root_run["inputs"],
            "outputs": self.root_run["outputs"],
            "claim_evidence_map": self.claim_evidence_map,
            "created_at": self.root_run["created_at"],
            "duration_ms": duration_ms,
            "run_count": len(self._run_stack),
        }

        # Lift verification metrics to top-level trace fields
        if metrics:
            trace_doc["trust_score"] = metrics.get("trust_score")
            trace_doc["hallucination_count"] = metrics.get("hallucination_count")
            trace_doc["claim_count"] = metrics.get("claim_count")

        self._safe_save_trace(trace_doc)

        if self._run_stack:
            self._run_stack.pop()

    # ── span context manager ────────────────────────────────────

    @contextmanager
    def span(
        self,
        name: str,
        run_type: str = "chain",
        inputs: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for creating a traced span."""
        run_id = str(uuid4())
        parent = self._run_stack[-1] if self._run_stack else None

        run = {
            "run_id": run_id,
            "trace_id": self.root_run["trace_id"] if self.root_run else run_id,
            "name": name,
            "run_type": run_type,
            "project_name": self.project_name,
            "inputs": inputs or {},
            "outputs": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "start_time": time.time(),
            "end_time": None,
            "duration_ms": None,
            "parent_id": parent["run_id"] if parent else None,
            "children": [],
            "error": None,
        }

        self._safe_save_run(run.copy())

        if parent:
            parent["children"].append(run_id)

        self._run_stack.append(run)
        context = SpanContext(run)
        start_time = time.time()

        try:
            yield context
        except Exception as e:
            run["error"] = str(e)
            run["end_time"] = time.time()
            run["duration_ms"] = (run["end_time"] - run["start_time"]) * 1000
            run["updated_at"] = datetime.utcnow()
            self._safe_update_run(run_id, {
                "error": run["error"],
                "end_time": run["end_time"],
                "duration_ms": run["duration_ms"],
                "updated_at": run["updated_at"],
            })
            self._run_stack.pop()
            raise
        else:
            duration_ms = (time.time() - start_time) * 1000
            context._outputs["duration_ms"] = duration_ms
            run["outputs"] = context._outputs
            run["end_time"] = time.time()
            run["duration_ms"] = duration_ms
            run["updated_at"] = datetime.utcnow()
            self._safe_update_run(run_id, {
                "outputs": run["outputs"],
                "end_time": run["end_time"],
                "duration_ms": run["duration_ms"],
                "updated_at": run["updated_at"],
            })
            self._run_stack.pop()

    # ── claim evidence ──────────────────────────────────────────

    def log_claim_evidence(self, claim_id: str, chunk_id: str, score: float):
        """Log claim-to-evidence mapping for backtracking."""
        if claim_id not in self.claim_evidence_map:
            self.claim_evidence_map[claim_id] = {}
        self.claim_evidence_map[claim_id][chunk_id] = score

    # ── query methods (multi-project aware) ─────────────────────

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a trace by ID."""
        try:
            return self.backend.get_trace(trace_id)
        except Exception as exc:
            logger.warning("Tracer: get_trace failed (%s)", exc)
            return None

    def list_recent_traces(
        self,
        limit: int = 10,
        project_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List recent traces, optionally filtered by project.

        Args:
            limit: Max traces to return.
            project_name: Filter to this project only. If None, returns
                          traces from all projects.
        """
        try:
            traces = self.backend.list_traces(limit=limit * 3 if project_name else limit)
        except Exception as exc:
            logger.warning("Tracer: list_traces failed (%s)", exc)
            return []

        if project_name:
            traces = [
                t for t in traces if t.get("project_name") == project_name
            ]
        return traces[:limit]

    def get_runs_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all runs for a specific trace."""
        try:
            return self.backend.get_runs_by_trace(trace_id)
        except Exception as exc:
            logger.warning("Tracer: get_runs_by_trace failed (%s)", exc)
            return []

    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self.backend.is_connected()


class SpanContext:
    """Context for managing outputs of a span."""

    def __init__(self, run: Dict[str, Any]):
        self.run = run
        self._outputs: Dict[str, Any] = {}

    def set_output(self, outputs: Dict[str, Any]):
        self._outputs.update(outputs)

    def add_tag(self, tag: str):
        if "tags" not in self._outputs:
            self._outputs["tags"] = []
        self._outputs["tags"].append(tag)


# Backward compatibility
MongoTracer = Tracer
