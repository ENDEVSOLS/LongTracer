"""
Tests: Package imports and public API surface.

Verifies the package can be imported cleanly without optional dependencies
and that all public symbols are accessible.
"""

import pytest


class TestPublicAPI:
    """The public API surface is importable and complete."""

    def test_top_level_imports(self):
        """Core symbols are importable from the top-level package."""
        from longtracer import LongTracer, CitationVerifier, VerificationResult
        assert LongTracer is not None
        assert CitationVerifier is not None
        assert VerificationResult is not None

    def test_backward_compat_alias(self):
        """CitationGuard is a backward-compatible alias for LongTracer."""
        from longtracer import LongTracer, CitationGuard
        assert CitationGuard is LongTracer

    def test_instrument_functions_importable(self):
        """instrument_langchain and instrument_llamaindex are importable."""
        from longtracer import instrument_langchain, instrument_llamaindex
        assert callable(instrument_langchain)
        assert callable(instrument_llamaindex)

    def test_adapters_module_importable_without_frameworks(self):
        """longtracer.adapters imports without LangChain/LlamaIndex installed."""
        import longtracer.adapters  # must not raise
        assert longtracer.adapters is not None

    def test_guard_module_importable(self):
        """longtracer.guard imports cleanly."""
        from longtracer.guard import CitationVerifier, Tracer, ContextRelevanceScorer
        assert CitationVerifier is not None
        assert Tracer is not None
        assert ContextRelevanceScorer is not None

    def test_cache_module_importable(self):
        """longtracer.guard.cache imports cleanly."""
        from longtracer.guard.cache import (
            TraceCacheBackend, create_backend, get_default_backend,
            CacheBackend, CacheStats, cache_key, get_cache,
        )
        assert TraceCacheBackend is not None
        assert callable(create_backend)
        assert callable(get_default_backend)

    def test_py_typed_marker_exists(self):
        """py.typed marker file exists for PEP 561 support."""
        import importlib.resources
        import longtracer
        import os
        pkg_dir = os.path.dirname(longtracer.__file__)
        assert os.path.exists(os.path.join(pkg_dir, "py.typed"))

    def test_all_exports_defined(self):
        """All symbols in __all__ are actually importable."""
        import longtracer
        for name in longtracer.__all__:
            assert hasattr(longtracer, name), f"__all__ lists '{name}' but it's not importable"
