"""
Core LongTracer SDK.

Provides the singleton `LongTracer` instance for global configuration
and context management. Supports multiple named projects via
``get_tracer(project_name)``.

Usage:
    from longtracer import LongTracer

    # Single project (default)
    LongTracer.init(verbose=True)

    # Multiple projects
    LongTracer.init(project_name="chatbot-prod", backend="sqlite")
    chatbot_tracer = LongTracer.get_tracer("chatbot-prod")
    search_tracer  = LongTracer.get_tracer("search-api")
"""

import os
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from contextvars import ContextVar

from .logging_config import configure_logging

if TYPE_CHECKING:
    from longtracer.guard.tracer import Tracer

# Thread-safe context storage
_longtracer_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "longtracer_context", default=None
)

logger = logging.getLogger("longtracer")


class LongTracer:
    """
    Singleton for configuring and accessing LongTracer.

    Supports multiple projects: each ``init()`` call with a different
    ``project_name`` creates a separate Tracer instance stored in
    ``_tracers``. ``get_tracer()`` without arguments returns the most
    recently initialized tracer (backward compatible).
    """

    _instance: Optional["LongTracer"] = None
    _tracers: Dict[str, "Tracer"] = {}
    _default_project: Optional[str] = None
    _enabled: bool = False
    _verbose: bool = False
    _backend_cache: Optional[Any] = None  # shared backend across projects

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LongTracer, cls).__new__(cls)
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: Optional[str] = None,
        backend: str = "auto",
        verbose: Optional[bool] = None,
        log_level: Optional[str] = None,
    ) -> "LongTracer":
        """
        Initialize LongTracer for a project.

        Can be called multiple times with different ``project_name`` values
        to set up multi-project tracing. All projects share the same backend.

        Config priority (highest to lowest):
            Code args > Environment variables > pyproject.toml > Built-in defaults

        Args:
            project_name: Name of the project (default: "default").
            backend: "auto", "mongo", "sqlite", "memory".
            verbose: Print per-span summaries to console.
            log_level: Python logging level.

        Returns:
            The singleton LongTracer instance.
        """
        from longtracer.config import load_config
        cfg = load_config()

        # Priority: code args > env vars > pyproject.toml > defaults
        if project_name is None:
            project_name = os.environ.get("LONGTRACER_PROJECT") or cfg.get("project")

        if backend == "auto":
            backend = os.environ.get("LONGTRACER_BACKEND") or cfg.get("backend", "auto")

        if verbose is None:
            env_v = os.environ.get("LONGTRACER_VERBOSE")
            if env_v is not None:
                verbose = env_v.lower() == "true"
            else:
                verbose = cfg.get("verbose", False)

        if log_level is None:
            log_level = os.environ.get("LONGTRACER_LOG_LEVEL") or cfg.get("log_level", "INFO")

        cls._verbose = verbose
        cls._enabled = True
        configure_logging(level=log_level, verbose=verbose)

        from longtracer.guard.tracer import Tracer
        from longtracer.guard.cache import create_backend, get_default_backend

        # Create or reuse the backend (shared across projects)
        if cls._backend_cache is None:
            if backend == "memory":
                cls._backend_cache = create_backend("memory")
            elif backend == "sqlite":
                cls._backend_cache = create_backend("sqlite")
            elif backend == "mongo":
                cls._backend_cache = create_backend("mongo")
            else:
                cls._backend_cache = get_default_backend()

        resolved_name = project_name or "default"

        cls._tracers[resolved_name] = Tracer(
            project_name=resolved_name,
            backend=cls._backend_cache,
        )
        cls._default_project = resolved_name

        # Ensure singleton instance exists
        if cls._instance is None:
            cls._instance = cls.__new__(cls)

        logger.info(
            "LongTracer initialized (project=%s, backend=%s, verbose=%s)",
            resolved_name, backend, verbose,
        )
        return cls._instance

    @classmethod
    def auto(cls) -> Optional["LongTracer"]:
        """
        Auto-enable if LONGTRACER_ENABLED=true.
        Returns the instance if enabled, None otherwise.
        """
        if os.environ.get("LONGTRACER_ENABLED", "").lower() == "true":
            return cls.init()
        return None

    @classmethod
    def get_tracer(cls, project_name: Optional[str] = None) -> Optional["Tracer"]:
        """
        Get a tracer by project name.

        Args:
            project_name: Project to retrieve. If None, returns the most
                          recently initialized tracer (backward compatible).

        Returns:
            Tracer instance, or None if not initialized.
        """
        if project_name:
            tracer = cls._tracers.get(project_name)
            if tracer is None:
                # Auto-create a tracer for this project using the shared backend
                if cls._backend_cache is not None:
                    from longtracer.guard.tracer import Tracer
                    tracer = Tracer(
                        project_name=project_name,
                        backend=cls._backend_cache,
                    )
                    cls._tracers[project_name] = tracer
            return tracer
        # Default: return the most recently initialized tracer
        if cls._default_project:
            return cls._tracers.get(cls._default_project)
        return None

    @classmethod
    def list_projects(cls) -> list:
        """Return names of all initialized projects."""
        return list(cls._tracers.keys())

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if LongTracer is enabled."""
        return cls._enabled

    @classmethod
    def is_verbose(cls) -> bool:
        """Check if verbose logging is enabled."""
        return cls._verbose

    @staticmethod
    def get_context() -> Dict[str, Any]:
        """Get the current thread-local context."""
        ctx = _longtracer_context.get()
        if ctx is None:
            ctx = {}
            _longtracer_context.set(ctx)
        return ctx

    @staticmethod
    def set_context(ctx: Dict[str, Any]):
        """Set the current thread-local context."""
        _longtracer_context.set(ctx)

    @classmethod
    def reset(cls) -> None:
        """Reset all state (useful for testing)."""
        cls._tracers.clear()
        cls._default_project = None
        cls._enabled = False
        cls._verbose = False
        cls._backend_cache = None


# Backward compatibility
CitationGuard = LongTracer
