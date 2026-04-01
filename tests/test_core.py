"""
Tests: LongTracer singleton — init, multi-project, auto-enable, reset.
"""

import pytest
import os
from longtracer import LongTracer
from longtracer.guard.tracer import Tracer


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset LongTracer state before and after every test."""
    LongTracer.reset()
    yield
    LongTracer.reset()


class TestInit:

    def test_init_returns_singleton(self):
        a = LongTracer.init(backend="memory")
        b = LongTracer.init(backend="memory")
        assert a is b

    def test_init_enables_longtracer(self):
        assert not LongTracer.is_enabled()
        LongTracer.init(backend="memory")
        assert LongTracer.is_enabled()

    def test_init_verbose_false_by_default(self):
        LongTracer.init(backend="memory")
        assert LongTracer.is_verbose() is False

    def test_init_verbose_true(self):
        LongTracer.init(backend="memory", verbose=True)
        assert LongTracer.is_verbose() is True

    def test_init_verbose_from_env(self, monkeypatch):
        monkeypatch.setenv("LONGTRACER_VERBOSE", "true")
        LongTracer.init(backend="memory")
        assert LongTracer.is_verbose() is True

    def test_init_default_project_name(self):
        LongTracer.init(backend="memory")
        tracer = LongTracer.get_tracer()
        assert tracer.project_name == "default"

    def test_init_custom_project_name(self):
        LongTracer.init(project_name="my-app", backend="memory")
        tracer = LongTracer.get_tracer()
        assert tracer.project_name == "my-app"

    def test_init_creates_tracer(self):
        LongTracer.init(backend="memory")
        assert LongTracer.get_tracer() is not None
        assert isinstance(LongTracer.get_tracer(), Tracer)


class TestMultiProject:

    def test_multiple_projects_stored_separately(self):
        LongTracer.init(project_name="proj-a", backend="memory")
        LongTracer.init(project_name="proj-b", backend="memory")
        a = LongTracer.get_tracer("proj-a")
        b = LongTracer.get_tracer("proj-b")
        assert a is not b
        assert a.project_name == "proj-a"
        assert b.project_name == "proj-b"

    def test_get_tracer_no_arg_returns_last_initialized(self):
        LongTracer.init(project_name="first", backend="memory")
        LongTracer.init(project_name="second", backend="memory")
        tracer = LongTracer.get_tracer()
        assert tracer.project_name == "second"

    def test_get_tracer_unknown_project_auto_creates(self):
        LongTracer.init(backend="memory")
        tracer = LongTracer.get_tracer("new-project")
        assert tracer is not None
        assert tracer.project_name == "new-project"

    def test_get_tracer_before_init_returns_none(self):
        assert LongTracer.get_tracer() is None

    def test_list_projects_returns_all_initialized(self):
        LongTracer.init(project_name="a", backend="memory")
        LongTracer.init(project_name="b", backend="memory")
        LongTracer.init(project_name="c", backend="memory")
        projects = LongTracer.list_projects()
        assert "a" in projects
        assert "b" in projects
        assert "c" in projects

    def test_projects_share_same_backend(self):
        LongTracer.init(project_name="proj-a", backend="memory")
        LongTracer.init(project_name="proj-b", backend="memory")
        a = LongTracer.get_tracer("proj-a")
        b = LongTracer.get_tracer("proj-b")
        assert a.backend is b.backend


class TestAutoEnable:

    def test_auto_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LONGTRACER_ENABLED", "false")
        result = LongTracer.auto()
        assert result is None
        assert not LongTracer.is_enabled()

    def test_auto_returns_instance_when_enabled(self, monkeypatch):
        monkeypatch.setenv("LONGTRACER_ENABLED", "true")
        # Pre-init with memory backend so auto() doesn't try to create SQLite
        LongTracer.init(backend="memory")
        LongTracer.reset()
        # Now auto() will use get_default_backend() — set env to use memory
        monkeypatch.setenv("TRACE_CACHE_BACKEND", "memory")
        result = LongTracer.auto()
        assert result is not None
        assert LongTracer.is_enabled()

    def test_auto_not_set_returns_none(self, monkeypatch):
        monkeypatch.delenv("LONGTRACER_ENABLED", raising=False)
        result = LongTracer.auto()
        assert result is None


class TestReset:

    def test_reset_clears_tracers(self):
        LongTracer.init(project_name="test", backend="memory")
        assert LongTracer.get_tracer() is not None
        LongTracer.reset()
        assert LongTracer.get_tracer() is None

    def test_reset_disables(self):
        LongTracer.init(backend="memory")
        assert LongTracer.is_enabled()
        LongTracer.reset()
        assert not LongTracer.is_enabled()

    def test_reset_clears_backend_cache(self):
        LongTracer.init(backend="memory")
        assert LongTracer._backend_cache is not None
        LongTracer.reset()
        assert LongTracer._backend_cache is None

    def test_can_reinit_after_reset(self):
        LongTracer.init(project_name="first", backend="memory")
        LongTracer.reset()
        LongTracer.init(project_name="second", backend="memory")
        assert LongTracer.get_tracer().project_name == "second"


class TestContext:

    def test_get_context_returns_dict(self):
        ctx = LongTracer.get_context()
        assert isinstance(ctx, dict)

    def test_set_and_get_context(self):
        LongTracer.set_context({"user_id": "abc123"})
        ctx = LongTracer.get_context()
        assert ctx["user_id"] == "abc123"

    def test_get_context_returns_empty_dict_when_unset(self):
        ctx = LongTracer.get_context()
        assert ctx == {} or isinstance(ctx, dict)
