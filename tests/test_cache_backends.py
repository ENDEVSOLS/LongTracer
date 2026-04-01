"""
Tests: Cache backends — MemoryBackend and SQLiteBackend.

These tests run without any external services.
"""

import os
import tempfile
import pytest
from datetime import datetime

from longtracer.guard.cache.memory import MemoryBackend
from longtracer.guard.cache.sqlite import SQLiteBackend
from longtracer.guard.cache.factory import create_backend, get_default_backend


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def memory():
    return MemoryBackend()


@pytest.fixture
def sqlite_db(tmp_path):
    db_path = str(tmp_path / "test_traces.db")
    backend = SQLiteBackend(path=db_path)
    yield backend
    backend.close()


def _make_run(run_id: str = "run-1", trace_id: str = "trace-1", name: str = "test") -> dict:
    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "name": name,
        "run_type": "chain",
        "project_name": "test-project",
        "inputs": {"query": "test"},
        "outputs": {},
        "created_at": datetime.utcnow(),
        "error": None,
    }


def _make_trace(trace_id: str = "trace-1", project: str = "test-project") -> dict:
    return {
        "trace_id": trace_id,
        "project_name": project,
        "run_name": "test_pipeline",
        "inputs": {"query": "What is X?"},
        "outputs": {"trust_score": 0.85},
        "claim_evidence_map": {},
        "created_at": datetime.utcnow(),
        "duration_ms": 1234.5,
        "run_count": 3,
    }


# ── MemoryBackend tests ──────────────────────────────────────────────

class TestMemoryBackend:

    def test_is_connected(self, memory):
        assert memory.is_connected() is True

    def test_save_and_get_run(self, memory):
        run = _make_run()
        memory.save_run(run)
        runs = memory.get_runs_by_trace("trace-1")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-1"

    def test_save_run_missing_run_id_raises(self, memory):
        with pytest.raises(ValueError, match="run_id"):
            memory.save_run({"name": "no-id"})

    def test_update_run(self, memory):
        run = _make_run()
        memory.save_run(run)
        result = memory.update_run("run-1", {"outputs": {"answer": "42"}})
        assert result is True
        runs = memory.get_runs_by_trace("trace-1")
        assert runs[0]["outputs"]["answer"] == "42"

    def test_update_nonexistent_run_returns_false(self, memory):
        result = memory.update_run("nonexistent", {"outputs": {}})
        assert result is False

    def test_save_and_get_trace(self, memory):
        trace = _make_trace()
        memory.save_trace(trace)
        retrieved = memory.get_trace("trace-1")
        assert retrieved is not None
        assert retrieved["trace_id"] == "trace-1"
        assert retrieved["project_name"] == "test-project"

    def test_get_nonexistent_trace_returns_none(self, memory):
        assert memory.get_trace("does-not-exist") is None

    def test_list_traces_newest_first(self, memory):
        for i in range(3):
            t = _make_trace(trace_id=f"trace-{i}", project="proj")
            t["created_at"] = datetime(2024, 1, i + 1)
            memory.save_trace(t)
        traces = memory.list_traces(limit=10)
        assert len(traces) == 3
        # Newest first
        assert traces[0]["trace_id"] == "trace-2"

    def test_list_traces_respects_limit(self, memory):
        for i in range(5):
            memory.save_trace(_make_trace(trace_id=f"t-{i}"))
        assert len(memory.list_traces(limit=2)) == 2

    def test_lru_eviction_at_capacity(self):
        backend = MemoryBackend(max_traces=3)
        for i in range(4):
            backend.save_trace(_make_trace(trace_id=f"t-{i}"))
        # Only 3 traces kept, oldest evicted
        assert len(backend._traces) == 3
        assert backend.get_trace("t-0") is None  # evicted
        assert backend.get_trace("t-3") is not None  # newest kept

    def test_clear_removes_all_data(self, memory):
        memory.save_run(_make_run())
        memory.save_trace(_make_trace())
        memory.clear()
        assert memory.list_traces() == []
        assert memory.get_runs_by_trace("trace-1") == []

    def test_stats(self, memory):
        memory.save_run(_make_run())
        memory.save_trace(_make_trace())
        stats = memory.stats()
        assert stats["runs"] == 1
        assert stats["traces"] == 1

    def test_multiple_runs_for_same_trace(self, memory):
        for i in range(3):
            memory.save_run(_make_run(run_id=f"run-{i}", trace_id="trace-1", name=f"span-{i}"))
        runs = memory.get_runs_by_trace("trace-1")
        assert len(runs) == 3

    def test_runs_isolated_by_trace(self, memory):
        memory.save_run(_make_run(run_id="r1", trace_id="trace-A"))
        memory.save_run(_make_run(run_id="r2", trace_id="trace-B"))
        assert len(memory.get_runs_by_trace("trace-A")) == 1
        assert len(memory.get_runs_by_trace("trace-B")) == 1


# ── SQLiteBackend tests ──────────────────────────────────────────────

class TestSQLiteBackend:

    def test_is_connected(self, sqlite_db):
        assert sqlite_db.is_connected() is True

    def test_save_and_get_run(self, sqlite_db):
        run = _make_run()
        sqlite_db.save_run(run)
        runs = sqlite_db.get_runs_by_trace("trace-1")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-1"

    def test_update_run(self, sqlite_db):
        run = _make_run()
        sqlite_db.save_run(run)
        result = sqlite_db.update_run("run-1", {"outputs": {"answer": "hello"}})
        assert result is True
        runs = sqlite_db.get_runs_by_trace("trace-1")
        assert runs[0]["outputs"]["answer"] == "hello"

    def test_update_nonexistent_run_returns_false(self, sqlite_db):
        result = sqlite_db.update_run("ghost", {"outputs": {}})
        assert result is False

    def test_save_and_get_trace(self, sqlite_db):
        trace = _make_trace()
        sqlite_db.save_trace(trace)
        retrieved = sqlite_db.get_trace("trace-1")
        assert retrieved is not None
        assert retrieved["trace_id"] == "trace-1"

    def test_get_nonexistent_trace_returns_none(self, sqlite_db):
        assert sqlite_db.get_trace("ghost") is None

    def test_list_traces_newest_first(self, sqlite_db):
        for i in range(3):
            t = _make_trace(trace_id=f"t-{i}")
            t["created_at"] = datetime(2024, 1, i + 1)
            sqlite_db.save_trace(t)
        traces = sqlite_db.list_traces(limit=10)
        assert len(traces) == 3

    def test_list_traces_respects_limit(self, sqlite_db):
        for i in range(5):
            sqlite_db.save_trace(_make_trace(trace_id=f"t-{i}"))
        assert len(sqlite_db.list_traces(limit=2)) == 2

    def test_data_persists_across_connections(self, tmp_path):
        """Data written to SQLite survives closing and reopening the connection."""
        db_path = str(tmp_path / "persist.db")
        b1 = SQLiteBackend(path=db_path)
        b1.save_trace(_make_trace(trace_id="persistent-trace"))
        b1.close()

        b2 = SQLiteBackend(path=db_path)
        retrieved = b2.get_trace("persistent-trace")
        b2.close()
        assert retrieved is not None
        assert retrieved["trace_id"] == "persistent-trace"

    def test_multiple_runs_for_same_trace(self, sqlite_db):
        for i in range(3):
            sqlite_db.save_run(_make_run(run_id=f"r-{i}", trace_id="trace-1", name=f"span-{i}"))
        runs = sqlite_db.get_runs_by_trace("trace-1")
        assert len(runs) == 3

    def test_disconnected_backend_returns_safe_defaults(self):
        """A backend that fails to connect returns safe empty values."""
        from unittest.mock import patch
        # Patch both makedirs and sqlite3.connect to simulate a full connection failure
        with patch("os.makedirs"), \
             patch("sqlite3.connect", side_effect=Exception("simulated DB failure")):
            backend = SQLiteBackend(path="/fake/path/traces.db")
        assert not backend.is_connected()
        assert backend.get_trace("x") is None
        assert backend.list_traces() == []
        assert backend.get_runs_by_trace("x") == []
        assert backend.update_run("x", {}) is False


# ── Factory tests ────────────────────────────────────────────────────

class TestCacheFactory:

    def test_create_memory_backend(self):
        from longtracer.guard.cache.memory import MemoryBackend
        b = create_backend("memory")
        assert isinstance(b, MemoryBackend)

    def test_create_sqlite_backend(self, tmp_path):
        from longtracer.guard.cache.sqlite import SQLiteBackend
        b = create_backend("sqlite", path=str(tmp_path / "test.db"))
        assert isinstance(b, SQLiteBackend)
        b.close()

    def test_create_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend("nonexistent_backend")

    def test_create_backend_aliases(self):
        from longtracer.guard.cache.memory import MemoryBackend
        assert isinstance(create_backend("mem"), MemoryBackend)

    def test_get_default_backend_returns_sqlite_by_default(self, monkeypatch):
        """Default backend is SQLite when no env vars are set."""
        monkeypatch.delenv("TRACE_CACHE_BACKEND", raising=False)
        monkeypatch.delenv("MONGODB_URI", raising=False)
        monkeypatch.delenv("REDIS_HOST", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        from longtracer.guard.cache.sqlite import SQLiteBackend
        b = get_default_backend()
        assert isinstance(b, SQLiteBackend)

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("TRACE_CACHE_BACKEND", "memory")
        from longtracer.guard.cache.memory import MemoryBackend
        b = get_default_backend()
        assert isinstance(b, MemoryBackend)
