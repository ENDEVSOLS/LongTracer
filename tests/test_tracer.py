"""
Tests: Tracer — span lifecycle, multi-project, backend resilience.

All tests use MemoryBackend to avoid I/O.
"""

import pytest
from unittest.mock import MagicMock, patch
from longtracer.guard.tracer import Tracer, SpanContext
from longtracer.guard.cache.memory import MemoryBackend


@pytest.fixture
def backend():
    return MemoryBackend()


@pytest.fixture
def tracer(backend):
    return Tracer(project_name="test-project", backend=backend)


class TestTracerInit:

    def test_default_project_name_from_env(self, monkeypatch):
        monkeypatch.setenv("TRACE_PROJECT", "env-project")
        t = Tracer(backend=MemoryBackend())
        assert t.project_name == "env-project"

    def test_explicit_project_name(self, backend):
        t = Tracer(project_name="my-project", backend=backend)
        assert t.project_name == "my-project"

    def test_backend_type_creates_backend(self):
        t = Tracer(project_name="p", backend_type="memory")
        assert t.backend is not None
        assert t.is_connected()

    def test_is_connected_delegates_to_backend(self, tracer):
        assert tracer.is_connected() is True


class TestRootLifecycle:

    def test_start_root_creates_root_run(self, tracer):
        tracer.start_root(inputs={"query": "test"})
        assert tracer.root_run is not None
        assert tracer.root_run["project_name"] == "test-project"
        assert tracer.root_run["inputs"]["query"] == "test"

    def test_start_root_generates_uuid_trace_id(self, tracer):
        tracer.start_root()
        tid = tracer.root_run["trace_id"]
        assert len(tid) == 36  # UUID format
        assert tid.count("-") == 4

    def test_end_root_sets_duration(self, tracer):
        tracer.start_root()
        tracer.end_root(outputs={"trust_score": 0.9})
        assert tracer.root_run["duration_ms"] is not None
        assert tracer.root_run["duration_ms"] > 0

    def test_end_root_saves_trace_to_backend(self, tracer, backend):
        tracer.start_root(inputs={"query": "hello"})
        trace_id = tracer.root_run["trace_id"]
        tracer.end_root(outputs={"verdict": "PASS"})
        saved = backend.get_trace(trace_id)
        assert saved is not None
        assert saved["outputs"]["verdict"] == "PASS"

    def test_end_root_without_start_does_not_crash(self, tracer):
        tracer.end_root()  # should not raise

    def test_root_run_stored_in_backend(self, tracer, backend):
        tracer.start_root()
        trace_id = tracer.root_run["trace_id"]
        runs = backend.get_runs_by_trace(trace_id)
        assert any(r["run_id"] == trace_id for r in runs)


class TestSpanContextManager:

    def test_span_creates_child_run(self, tracer, backend):
        tracer.start_root()
        trace_id = tracer.root_run["trace_id"]
        with tracer.span("retrieval", run_type="retriever") as span:
            span.set_output({"count": 5})
        runs = backend.get_runs_by_trace(trace_id)
        span_runs = [r for r in runs if r["name"] == "retrieval"]
        assert len(span_runs) == 1
        assert span_runs[0]["outputs"]["count"] == 5

    def test_span_records_duration(self, tracer, backend):
        tracer.start_root()
        trace_id = tracer.root_run["trace_id"]
        with tracer.span("llm_call") as span:
            span.set_output({"answer": "hello"})
        runs = backend.get_runs_by_trace(trace_id)
        span_run = next(r for r in runs if r["name"] == "llm_call")
        assert span_run["duration_ms"] > 0

    def test_span_records_error_on_exception(self, tracer, backend):
        tracer.start_root()
        trace_id = tracer.root_run["trace_id"]
        with pytest.raises(ValueError):
            with tracer.span("failing_span"):
                raise ValueError("test error")
        runs = backend.get_runs_by_trace(trace_id)
        span_run = next(r for r in runs if r["name"] == "failing_span")
        assert span_run["error"] == "test error"

    def test_span_exception_propagates_to_caller(self, tracer):
        tracer.start_root()
        with pytest.raises(RuntimeError, match="propagated"):
            with tracer.span("test"):
                raise RuntimeError("propagated")

    def test_nested_spans_have_parent_id(self, tracer, backend):
        tracer.start_root()
        trace_id = tracer.root_run["trace_id"]
        with tracer.span("outer") as outer_span:
            outer_run_id = outer_span.run["run_id"]
            with tracer.span("inner"):
                pass
        runs = backend.get_runs_by_trace(trace_id)
        inner = next(r for r in runs if r["name"] == "inner")
        assert inner["parent_id"] == outer_run_id

    def test_span_without_root_still_works(self, tracer):
        """Span can be used without start_root (no crash)."""
        with tracer.span("standalone") as span:
            span.set_output({"ok": True})


class TestBackendResilience:
    """Tracer never crashes the caller when backend fails."""

    def test_failing_save_run_does_not_crash(self):
        bad_backend = MagicMock()
        bad_backend.save_run.side_effect = Exception("DB down")
        bad_backend.update_run.side_effect = Exception("DB down")
        bad_backend.save_trace.side_effect = Exception("DB down")
        bad_backend.is_connected.return_value = False

        t = Tracer(project_name="test", backend=bad_backend)
        t.start_root(inputs={"query": "test"})  # must not raise
        t.end_root(outputs={"result": "ok"})    # must not raise

    def test_failing_backend_span_does_not_crash(self):
        bad_backend = MagicMock()
        bad_backend.save_run.side_effect = Exception("DB down")
        bad_backend.update_run.side_effect = Exception("DB down")
        bad_backend.is_connected.return_value = False

        t = Tracer(project_name="test", backend=bad_backend)
        t.start_root()
        with t.span("test_span") as span:  # must not raise
            span.set_output({"data": 1})

    def test_failing_get_trace_returns_none(self):
        bad_backend = MagicMock()
        bad_backend.get_trace.side_effect = Exception("DB down")
        t = Tracer(project_name="test", backend=bad_backend)
        result = t.get_trace("some-id")
        assert result is None

    def test_failing_list_traces_returns_empty(self):
        bad_backend = MagicMock()
        bad_backend.list_traces.side_effect = Exception("DB down")
        t = Tracer(project_name="test", backend=bad_backend)
        result = t.list_recent_traces()
        assert result == []


class TestClaimEvidence:

    def test_log_claim_evidence(self, tracer):
        tracer.log_claim_evidence("claim text", "source text", 0.85)
        assert "claim text" in tracer.claim_evidence_map
        assert tracer.claim_evidence_map["claim text"]["source text"] == 0.85

    def test_multiple_sources_for_same_claim(self, tracer):
        tracer.log_claim_evidence("claim", "source-1", 0.7)
        tracer.log_claim_evidence("claim", "source-2", 0.9)
        assert len(tracer.claim_evidence_map["claim"]) == 2

    def test_claim_evidence_included_in_trace(self, tracer, backend):
        tracer.start_root()
        trace_id = tracer.root_run["trace_id"]
        tracer.log_claim_evidence("the claim", "the source", 0.75)
        tracer.end_root()
        saved = backend.get_trace(trace_id)
        assert "the claim" in saved["claim_evidence_map"]


class TestMultiProjectTracing:

    def test_different_project_names(self):
        backend = MemoryBackend()
        t1 = Tracer(project_name="project-a", backend=backend)
        t2 = Tracer(project_name="project-b", backend=backend)
        assert t1.project_name == "project-a"
        assert t2.project_name == "project-b"

    def test_traces_tagged_with_project_name(self):
        backend = MemoryBackend()
        t1 = Tracer(project_name="proj-a", backend=backend)
        t2 = Tracer(project_name="proj-b", backend=backend)

        t1.start_root()
        t1.end_root()
        t2.start_root()
        t2.end_root()

        all_traces = backend.list_traces(limit=10)
        proj_a = [t for t in all_traces if t["project_name"] == "proj-a"]
        proj_b = [t for t in all_traces if t["project_name"] == "proj-b"]
        assert len(proj_a) == 1
        assert len(proj_b) == 1

    def test_list_recent_traces_filters_by_project(self):
        backend = MemoryBackend()
        t1 = Tracer(project_name="proj-a", backend=backend)
        t2 = Tracer(project_name="proj-b", backend=backend)

        t1.start_root()
        t1.end_root()
        t2.start_root()
        t2.end_root()

        only_a = t1.list_recent_traces(project_name="proj-a")
        assert all(t["project_name"] == "proj-a" for t in only_a)
        assert len(only_a) == 1

    def test_list_recent_traces_no_filter_returns_all(self):
        backend = MemoryBackend()
        for i in range(3):
            t = Tracer(project_name=f"proj-{i}", backend=backend)
            t.start_root()
            t.end_root()
        t_any = Tracer(project_name="any", backend=backend)
        all_traces = t_any.list_recent_traces()
        assert len(all_traces) == 3


class TestSpanContext:

    def test_set_output_updates_outputs(self):
        run = {"run_id": "r1"}
        ctx = SpanContext(run)
        ctx.set_output({"key": "value"})
        assert ctx._outputs["key"] == "value"

    def test_set_output_merges(self):
        run = {"run_id": "r1"}
        ctx = SpanContext(run)
        ctx.set_output({"a": 1})
        ctx.set_output({"b": 2})
        assert ctx._outputs["a"] == 1
        assert ctx._outputs["b"] == 2

    def test_add_tag(self):
        run = {"run_id": "r1"}
        ctx = SpanContext(run)
        ctx.add_tag("hallucination")
        ctx.add_tag("low_trust")
        assert "hallucination" in ctx._outputs["tags"]
        assert "low_trust" in ctx._outputs["tags"]
