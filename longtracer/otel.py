"""
OpenTelemetry integration for LongTracer.

Provides optional OTel span emission for verify() calls, exportable
via OTLP to Jaeger, Grafana Tempo, Datadog, or any OTLP-compatible backend.

All OTel functionality is optional. When opentelemetry-* packages are
not installed, all functions become no-ops with zero overhead.

Configuration:
    Env vars:
        LONGTRACER_OTEL_ENABLED       — "true" to enable (default: false)
        OTEL_EXPORTER_OTLP_ENDPOINT   — e.g. "http://localhost:4318"
        OTEL_SERVICE_NAME             — service name (default: "longtracer")
    pyproject.toml:
        [tool.longtracer]
        otel_enabled = true

Usage (auto-instrumentation):
    from longtracer.otel import setup_otel
    setup_otel(endpoint="http://localhost:4318")
    # Now all verify() calls create OTel spans automatically

Usage (manual span):
    from longtracer.otel import verify_span
    with verify_span("response text", ["source1"], project="my-chatbot"):
        result = verifier.verify_parallel(response, sources)
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("longtracer")

# ── Lazy imports with graceful fallback ──────────────────────────

_OTEL_AVAILABLE = False
_TracerProvider = None
_BatchSpanProcessor = None
_OTLPSpanExporter = None
_trace_api = None

try:
    from opentelemetry import trace as _trace_api_mod
    from opentelemetry.sdk.trace import TracerProvider as _TP
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BSP
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as _OTLP,
    )
    _OTEL_AVAILABLE = True
    _TracerProvider = _TP
    _BatchSpanProcessor = _BSP
    _OTLPSpanExporter = _OTLP
    _trace_api = _trace_api_mod
except ImportError:
    pass

# ── Module state ─────────────────────────────────────────────────

_tracer = None  # OTel tracer instance, set by setup_otel()
_setup_done = False


# ── Public API ───────────────────────────────────────────────────


def setup_otel(
    endpoint: Optional[str] = None,
    service_name: str = "longtracer",
) -> None:
    """Initialize OpenTelemetry TracerProvider with OTLP exporter.

    When opentelemetry-* packages are not installed, logs a warning
    and returns silently (all span creation becomes a no-op).

    Args:
        endpoint: OTLP HTTP endpoint (e.g. "http://localhost:4318").
                  Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var.
        service_name: Service name for spans.
                      Falls back to OTEL_SERVICE_NAME env var.
    """
    global _tracer, _setup_done

    if not _OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install with: pip install 'longtracer[otel]'"
        )
        return

    if _setup_done:
        logger.debug("OTel already set up, skipping")
        return

    # Resolve config
    resolved_endpoint = endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", ""
    )
    resolved_service = os.environ.get(
        "OTEL_SERVICE_NAME", service_name
    )

    if not resolved_endpoint:
        logger.warning(
            "OTel enabled but no endpoint configured. "
            "Set OTEL_EXPORTER_OTLP_ENDPOINT or pass endpoint= to setup_otel(). "
            "Falling back to console exporter."
        )
        exporter = None
    else:
        exporter = _OTLPSpanExporter(endpoint=resolved_endpoint)

    # Create TracerProvider
    provider = _TracerProvider()

    if exporter is not None:
        processor = _BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        logger.info(
            "OTel initialized: service=%s, endpoint=%s",
            resolved_service, resolved_endpoint,
        )
    else:
        # Fallback: console exporter so spans are at least logged
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            processor = _BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            logger.info(
                "OTel initialized with console exporter: service=%s",
                resolved_service,
            )
        except Exception:
            logger.warning("OTel console exporter fallback failed")

    _trace_api.set_tracer_provider(provider)
    _tracer = _trace_api.get_tracer(
        "longtracer.otel",
        "0.4.0",
    )
    _setup_done = True


def is_otel_enabled() -> bool:
    """Check if OTel is configured and available.

    Returns True if:
      - opentelemetry-* packages are installed
      - setup_otel() has been called OR LONGTRACER_OTEL_ENABLED=true
      - A tracer is available
    """
    if not _OTEL_AVAILABLE:
        return False

    # If already set up, it's enabled
    if _tracer is not None:
        return True

    # Check config for auto-enable
    from longtracer.config import load_config
    cfg = load_config()
    env_val = os.environ.get("LONGTRACER_OTEL_ENABLED", "").lower()
    cfg_val = cfg.get("otel_enabled", False)

    if env_val == "true" or cfg_val is True:
        # Auto-setup
        setup_otel()
        return _tracer is not None

    return False


@contextmanager
def verify_span(
    response: str,
    sources: list,
    project: str = "default",
):
    """Context manager that wraps verify() in an OTel span.

    Creates a span named ``longtracer.verify`` with attributes set
    from the verification result after the block completes.

    When OTel is not available or not set up, this is a no-op.

    Usage::

        from longtracer.otel import verify_span, setup_otel

        setup_otel(endpoint="http://localhost:4318")

        with verify_span(response, sources, project="my-chatbot") as span_ctx:
            result = verifier.verify_parallel(response, sources)

        # span_ctx["result"] contains the VerificationResult
    """
    span_ctx = {"result": None}

    if not _OTEL_AVAILABLE or _tracer is None:
        yield span_ctx
        return

    start = time.time()
    with _tracer.start_as_current_span("longtracer.verify") as span:
        # Set input attributes
        span.set_attribute("longtracer.project_id", project)
        span.set_attribute("longtracer.source_count", len(sources))
        span.set_attribute(
            "longtracer.response_length", len(response)
        )

        try:
            yield span_ctx
            result = span_ctx.get("result")

            if result is not None:
                # Set result attributes
                span.set_attribute(
                    "longtracer.trust_score",
                    getattr(result, "trust_score", 0.0),
                )
                span.set_attribute(
                    "longtracer.claim_count",
                    len(getattr(result, "claims", [])),
                )
                span.set_attribute(
                    "longtracer.hallucination_count",
                    getattr(result, "hallucination_count", 0),
                )
                span.set_attribute(
                    "longtracer.verdict",
                    getattr(result, "verdict", "UNKNOWN"),
                )
                span.set_attribute(
                    "longtracer.threshold",
                    getattr(result, "threshold", 0.5)
                    if hasattr(result, "threshold")
                    else 0.5,
                )
                span.set_attribute(
                    "longtracer.duration_ms",
                    round((time.time() - start) * 1000, 2),
                )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(
                _trace_api.StatusCode.ERROR,
                str(exc),
            )
            raise


def get_tracer():
    """Return the OTel tracer instance (or None if not set up)."""
    return _tracer
