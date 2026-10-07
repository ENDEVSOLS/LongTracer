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
            TraceCacheBackend,
            create_backend,
            get_default_backend,
            CacheBackend,
            CacheStats,
            cache_key,
            get_cache,
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


class TestErrorsModuleIsImportLight:
    """longtracer.errors must be importable without pulling in model libraries."""

    def test_errors_module_importable(self):
        """longtracer.errors imports cleanly on its own."""
        import longtracer.errors as errors_mod

        assert errors_mod is not None

    def test_errors_module_has_no_heavy_top_level_imports(self):
        """longtracer/errors.py itself must not import model libraries at module level.

        Note: importing `longtracer.errors` still triggers `longtracer/__init__.py`,
        which (pre-existing, since cb3407a) eagerly imports CitationVerifier and
        therefore sentence_transformers. That is a package-level eagerness issue,
        not something longtracer/errors.py introduces. This test checks the
        module's own source so it stays meaningful regardless of that.
        """
        import ast
        import inspect

        import longtracer.errors as errors_mod

        source = inspect.getsource(errors_mod)
        tree = ast.parse(source)
        heavy_prefixes = ("torch", "transformers", "sentence_transformers")
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

        heavy = imported_roots & set(heavy_prefixes)
        assert not heavy, f"longtracer/errors.py imports heavy module(s) at module level: {heavy}"

    def test_error_hierarchy_importable(self):
        """All typed evaluator errors are importable from longtracer.errors."""
        from longtracer.errors import (
            EvaluatorError,
            ModelUnavailableError,
            EvaluationTimeoutError,
            InvalidInputError,
            EvaluationFailedError,
        )

        assert issubclass(ModelUnavailableError, EvaluatorError)
        assert issubclass(EvaluationTimeoutError, EvaluatorError)
        assert issubclass(InvalidInputError, EvaluatorError)
        assert issubclass(EvaluationFailedError, EvaluatorError)

    def test_model_unavailable_is_also_import_error(self):
        """ModelUnavailableError subclasses ImportError for backward compatibility."""
        from longtracer.errors import ModelUnavailableError

        assert issubclass(ModelUnavailableError, ImportError)

    def test_timeout_is_also_timeout_error(self):
        """EvaluationTimeoutError subclasses the built-in TimeoutError."""
        from longtracer.errors import EvaluationTimeoutError

        assert issubclass(EvaluationTimeoutError, TimeoutError)

    def test_invalid_input_is_also_value_error(self):
        """InvalidInputError subclasses the built-in ValueError."""
        from longtracer.errors import InvalidInputError

        assert issubclass(InvalidInputError, ValueError)
