"""
Evaluator errors — typed LongTracer exceptions for verification failures.

Import-light by design: this module uses only the standard library, so it can be
imported (and caught) without loading model libraries or weights.

Each subclass also inherits from the built-in exception that callers already
catch today, so existing ``except ImportError`` / ``except TimeoutError`` /
``except ValueError`` handlers keep working.
"""

from typing import Optional

PREPARE_HINT = "Run `longtracer models prepare` to download and cache the models, then retry."


class EvaluatorError(Exception):
    """Base class for all LongTracer evaluator errors."""


class ModelUnavailableError(EvaluatorError, ImportError):
    """A verification model could not be loaded (missing, not downloaded, or corrupt).

    Subclasses ``ImportError`` because model-load failures were raised as
    ``ImportError`` before v0.3.0.

    Attributes:
        model_name: Identifier of the model that failed to load, if known.
    """

    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(message)
        self.model_name = model_name


class EvaluationTimeoutError(EvaluatorError, TimeoutError):
    """Verification did not finish within the allowed time.

    Attributes:
        timeout_s: The timeout that was exceeded, in seconds.
    """

    def __init__(self, message: str, timeout_s: Optional[float] = None):
        super().__init__(message)
        self.timeout_s = timeout_s


class InvalidInputError(EvaluatorError, ValueError):
    """Input that the evaluator cannot assess (malformed or unsupported).

    Note: ``CitationVerifier`` public methods still raise ``TypeError`` for
    wrong argument types, as documented. This error is used by the typed
    result path (``verify_case``) and for inputs the models cannot process.
    """


class EvaluationFailedError(EvaluatorError, RuntimeError):
    """A model raised an unexpected error while scoring (e.g. out of memory)."""


__all__ = [
    "EvaluatorError",
    "ModelUnavailableError",
    "EvaluationTimeoutError",
    "InvalidInputError",
    "EvaluationFailedError",
    "PREPARE_HINT",
]
