"""
LongTracer REST API Server.

Exposes LongTracer verification as HTTP endpoints with security:
    - API key authentication (x-api-key header + Bearer fallback)
    - CORS with configurable origins
    - Rate limiting (token bucket per IP)
    - Input validation (Pydantic models with size limits)
    - Timing-safe key comparison

Usage:
    longtracer serve                  # start on 0.0.0.0:8100
    longtracer serve --port 9000      # custom port
    longtracer serve --reload         # dev mode with auto-reload

    # Set API key (required):
    export LONGTRACER_API_KEY="your-secret-key"

Endpoints:
    GET  /api/v1/health             — Health check (no auth)
    POST /api/v1/verify             — Verify a single response
    POST /api/v1/verify/batch       — Verify multiple responses
    GET  /api/v1/traces             — List recent traces
    GET  /api/v1/traces/{trace_id}  — Get a specific trace
"""

import logging
import os
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("longtracer")

# ── Constants ───────────────────────────────────────────────────

MAX_RESPONSE_LENGTH = 50_000  # 50K chars
MAX_SOURCE_LENGTH = 10_000  # 10K chars per source
MAX_SOURCES_COUNT = 100  # max sources per request
MAX_BATCH_SIZE = 20  # max items in batch
DEFAULT_RATE_LIMIT = 60  # requests per minute per IP
API_VERSION = "v1"


# ── Pydantic Models ─────────────────────────────────────────────

class VerifyRequest(BaseModel):
    """Request body for single verification."""

    response: str = Field(
        ...,
        min_length=1,
        max_length=MAX_RESPONSE_LENGTH,
        description="LLM-generated response text to verify.",
    )
    sources: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_SOURCES_COUNT,
        description="Source document chunks to verify against.",
    )
    source_metadata: Optional[List[dict]] = Field(
        default=None,
        description="Optional metadata for each source.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Verification threshold (0.0–1.0).",
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: List[str]) -> List[str]:
        """Validate and truncate individual source strings."""
        validated = []
        for i, src in enumerate(v):
            if not isinstance(src, str):
                raise ValueError(f"sources[{i}] must be a string")
            if len(src) > MAX_SOURCE_LENGTH:
                validated.append(src[:MAX_SOURCE_LENGTH])
            else:
                validated.append(src)
        return validated


class VerifyBatchRequest(BaseModel):
    """Request body for batch verification."""

    items: List[VerifyRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of verification requests.",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Max parallel workers.",
    )


class ClaimResponse(BaseModel):
    """Individual claim in a verification response."""

    claim: str
    supported: bool
    score: float
    is_hallucination: bool


class VerifyResponse(BaseModel):
    """Response body for verification."""

    verdict: str
    trust_score: float
    summary: str
    hallucination_count: int
    claims: List[ClaimResponse]
    all_supported: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = ""
    uptime_seconds: float = 0.0


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str


# ── Rate Limiter ────────────────────────────────────────────────

class TokenBucketRateLimiter:
    """Thread-safe in-memory token bucket rate limiter.

    Each IP address gets a separate bucket with a configurable
    rate limit (requests per minute).
    """

    def __init__(self, rate_per_minute: int = DEFAULT_RATE_LIMIT):
        self.rate = rate_per_minute
        self.interval = 60.0 / rate_per_minute  # seconds per token
        self._buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": float(rate_per_minute), "last_refill": time.time()}
        )
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Check if a request is allowed for the given key.

        Returns True if allowed, False if rate limited.
        """
        with self._lock:
            now = time.time()
            bucket = self._buckets[key]

            # Refill tokens
            elapsed = now - bucket["last_refill"]
            refill = elapsed / self.interval
            bucket["tokens"] = min(float(self.rate), bucket["tokens"] + refill)
            bucket["last_refill"] = now

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            return False


# ── Time parsing helper ───────────────────────────────────────

def _parse_metrics_time(
    value: Optional[str],
    default_offset_days: Optional[int] = None,
) -> Optional[datetime]:
    """Parse an ISO 8601 datetime string or return a default.

    Args:
        value: ISO 8601 datetime string, or None.
        default_offset_days: If value is None, return now minus this many days.

    Returns:
        A datetime object, or None.
    """
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    if default_offset_days is not None:
        return datetime.now(timezone.utc) - timedelta(days=default_offset_days)
    return None


# ── App Factory ────────────────────────────────────────────────

def create_app() -> Any:
    """Create and configure the FastAPI application.

    Returns:
        A configured FastAPI app instance.

    Raises:
        ImportError: If FastAPI or uvicorn is not installed.
    """
    try:
        from fastapi import FastAPI, Header, HTTPException, Request, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
    except ImportError:
        raise ImportError(
            "FastAPI and uvicorn are required for the REST API server. "
            "Install with: pip install 'longtracer[server]'"
        )

    try:
        from fastapi.templating import Jinja2Templates
        from fastapi.staticfiles import StaticFiles
        import pathlib
        _DASHBOARD_DIR = pathlib.Path(__file__).parent / "dashboard"
        templates = Jinja2Templates(directory=str(_DASHBOARD_DIR / "templates"))
        _TEMPLATES_AVAILABLE = True
    except Exception:
        _TEMPLATES_AVAILABLE = False

    # ── Configuration ───────────────────────────────────────
    api_key = os.environ.get("LONGTRACER_API_KEY", "")
    cors_origins_str = os.environ.get("LONGTRACER_CORS_ORIGINS", "")
    cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()] if cors_origins_str else []
    rate_limit = int(os.environ.get("LONGTRACER_RATE_LIMIT", str(DEFAULT_RATE_LIMIT)))

    # ── State ───────────────────────────────────────────────
    start_time = time.time()
    rate_limiter = TokenBucketRateLimiter(rate_per_minute=rate_limit)

    # ── App ─────────────────────────────────────────────────
    app = FastAPI(
        title="LongTracer API",
        description="RAG verification guardrails — detect hallucinations in LLM responses.",
        version="0.1.6",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["x-api-key", "authorization", "content-type"],
        )

    # ── Auth dependency ─────────────────────────────────────

    async def verify_api_key(
        request: Request,
        x_api_key: Optional[str] = Header(None, alias="x-api-key"),
        authorization: Optional[str] = Header(None),
    ) -> None:
        """Validate API key from x-api-key header or Bearer token.

        Uses timing-safe comparison to prevent timing attacks.
        """
        if not api_key:
            # No API key configured — allow all (dev mode)
            return

        provided_key = ""

        # Priority 1: x-api-key header (LangSmith standard)
        if x_api_key:
            provided_key = x_api_key
        # Priority 2: Authorization: Bearer <key>
        elif authorization and authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()

        if not provided_key:
            raise HTTPException(
                status_code=401,
                detail="API key required. Provide via x-api-key header.",
            )

        if not secrets.compare_digest(provided_key, api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
            )

    # ── Rate limit dependency ───────────────────────────────

    async def check_rate_limit(request: Request) -> None:
        """Check rate limit for the requesting IP."""
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
            )

    # ── Endpoints ───────────────────────────────────────────

    @app.get(
        f"/api/{API_VERSION}/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
    )
    async def health():
        """Health check — no authentication required."""
        return HealthResponse(
            status="ok",
            version="0.1.6",
            uptime_seconds=round(time.time() - start_time, 1),
        )

    @app.post(
        f"/api/{API_VERSION}/verify",
        response_model=VerifyResponse,
        tags=["Verification"],
        summary="Verify a single response",
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def verify(req: VerifyRequest):
        """Verify an LLM response against source documents.

        Returns claim-level verification with trust score and verdict.
        """
        try:
            from longtracer.guard.verifier import CitationVerifier

            verifier = CitationVerifier(threshold=req.threshold)
            result = verifier.verify_parallel(
                req.response, req.sources, req.source_metadata,
            )

            # Dispatch webhook if configured
            try:
                from longtracer.webhooks import dispatch_verification_result
                dispatch_verification_result(result)
            except Exception:
                pass  # Webhook failure should never fail the API

            # Save verification as a trace (for metrics / observability)
            try:
                _save_verification_trace(req.response, req.sources, result)
            except Exception:
                pass  # Trace save failure should never fail the API

            # Dispatch alert if trust_score < threshold
            try:
                from longtracer.alerts import dispatch_alert
                dispatch_alert(
                    result,
                    project="longtracer_api",
                    trace_id=getattr(result, '_trace_id', None),
                )
            except Exception:
                pass  # Alert failure should never fail the API

            return VerifyResponse(
                verdict=result.verdict,
                trust_score=round(result.trust_score, 4),
                summary=result.summary,
                hallucination_count=result.hallucination_count,
                all_supported=result.all_supported,
                claims=[
                    ClaimResponse(
                        claim=c.get("claim", "")[:500],
                        supported=c.get("supported", False),
                        score=round(c.get("score", 0), 4),
                        is_hallucination=c.get("is_hallucination", False),
                    )
                    for c in result.claims
                ],
            )

        except Exception as exc:
            logger.error("Verification error: %s", exc)
            raise HTTPException(status_code=500, detail="Verification failed.")

    @app.post(
        f"/api/{API_VERSION}/verify/batch",
        response_model=List[VerifyResponse],
        tags=["Verification"],
        summary="Verify multiple responses in batch",
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def verify_batch(req: VerifyBatchRequest):
        """Verify multiple LLM responses in one call."""
        try:
            from longtracer.guard.verifier import CitationVerifier

            verifier = CitationVerifier()
            items = [
                {
                    "response": item.response,
                    "sources": item.sources,
                    "source_metadata": item.source_metadata,
                }
                for item in req.items
            ]
            results = verifier.verify_batch(items, max_workers=req.max_workers)

            responses = []
            for result in results:
                responses.append(VerifyResponse(
                    verdict=result.verdict,
                    trust_score=round(result.trust_score, 4),
                    summary=result.summary,
                    hallucination_count=result.hallucination_count,
                    all_supported=result.all_supported,
                    claims=[
                        ClaimResponse(
                            claim=c.get("claim", "")[:500],
                            supported=c.get("supported", False),
                            score=round(c.get("score", 0), 4),
                            is_hallucination=c.get("is_hallucination", False),
                        )
                        for c in result.claims
                    ],
                ))

            return responses

        except Exception as exc:
            logger.error("Batch verification error: %s", exc)
            raise HTTPException(status_code=500, detail="Batch verification failed.")

    @app.get(
        f"/api/{API_VERSION}/traces",
        tags=["Traces"],
        summary="List recent traces",
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def list_traces(
        limit: int = 10,
        project: Optional[str] = None,
    ):
        """List recent verification traces."""
        try:
            from longtracer.guard.tracer import Tracer
            tracer = Tracer(run_name="longtracer_api")
            traces = tracer.list_recent_traces(limit=limit, project_name=project)

            # Sanitize trace output — remove internal fields
            sanitized = []
            for t in traces:
                sanitized.append({
                    "trace_id": t.get("trace_id"),
                    "project_name": t.get("project_name"),
                    "run_name": t.get("run_name"),
                    "created_at": str(t.get("created_at", "")),
                    "duration_ms": t.get("duration_ms"),
                })
            return sanitized
        except Exception as exc:
            logger.error("List traces error: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to list traces.")

    @app.get(
        f"/api/{API_VERSION}/traces/{{trace_id}}",
        tags=["Traces"],
        summary="Get a specific trace",
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def get_trace(trace_id: str):
        """Get details of a specific trace by ID."""
        try:
            from longtracer.guard.tracer import Tracer
            tracer = Tracer(run_name="longtracer_api")
            trace = tracer.get_trace(trace_id)
            if not trace:
                raise HTTPException(status_code=404, detail="Trace not found.")
            return trace
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Get trace error: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to get trace.")

    # ── Helper: save verification trace ────────────────────────

    def _save_verification_trace(
        response_text: str,
        sources: List[str],
        result: Any,
    ) -> None:
        """Save a verification result as a trace with metrics."""
        try:
            from longtracer.guard.tracer import Tracer
            from longtracer.guard.cache import get_default_backend

            tracer = Tracer(
                project_name="longtracer_api",
                run_name="verify_endpoint",
                backend=get_default_backend(),
            )
            tracer.start_root(inputs={
                "response": response_text[:200],
                "source_count": len(sources),
            })
            tracer.end_root(
                outputs={
                    "verdict": result.verdict,
                    "trust_score": result.trust_score,
                    "summary": result.summary,
                },
                metrics={
                    "trust_score": result.trust_score,
                    "hallucination_count": result.hallucination_count,
                    "claim_count": len(result.claims),
                },
            )
        except Exception as exc:
            logger.debug("Trace save failed (non-critical): %s", exc)

    # ── Metrics endpoints ────────────────────────────────────────

    @app.get(
        f"/api/{API_VERSION}/metrics/summary",
        tags=["Metrics"],
        summary="Get aggregated metrics summary",
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def metrics_summary(
        project: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """Get aggregated verification metrics.

        Query params:
            project: Filter by project name.
            start: ISO 8601 datetime (default: 7 days ago).
            end: ISO 8601 datetime (default: now).
        """
        try:
            from longtracer.guard.cache import get_default_backend

            start_dt = _parse_metrics_time(start, default_offset_days=7)
            end_dt = _parse_metrics_time(end) or datetime.now(timezone.utc)

            backend = get_default_backend()
            return backend.get_metrics_summary(
                project=project,
                start_time=start_dt,
                end_time=end_dt,
            )
        except Exception as exc:
            logger.error("Metrics summary error: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to get metrics.")

    @app.get(
        f"/api/{API_VERSION}/metrics/timeseries",
        tags=["Metrics"],
        summary="Get time-bucketed metrics",
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def metrics_timeseries(
        project: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ):
        """Get time-bucketed verification metrics for charting.

        Query params:
            project: Filter by project name.
            start: ISO 8601 datetime (default: 7 days ago).
            end: ISO 8601 datetime (default: now).
            interval: Bucket size — "1h", "6h", "1d", "1w" (default: "1d").
        """
        try:
            from longtracer.guard.cache import get_default_backend

            if interval not in ("1h", "6h", "1d", "1w"):
                interval = "1d"

            start_dt = _parse_metrics_time(start, default_offset_days=7)
            end_dt = _parse_metrics_time(end) or datetime.now(timezone.utc)

            backend = get_default_backend()
            return backend.get_metrics_timeseries(
                project=project,
                start_time=start_dt,
                end_time=end_dt,
                interval=interval,
            )
        except Exception as exc:
            logger.error("Metrics timeseries error: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to get metrics.")

    # ── Dashboard routes ─────────────────────────────────────────

    # Cookie-based auth helper for dashboard
    def _check_dashboard_auth(request: Request) -> bool:
        """Check if the dashboard request has a valid API key cookie.
        Returns True if authenticated (or if no API key is configured)."""
        if not api_key:
            return True  # Dev mode — no auth required
        cookie_key = request.cookies.get("longtracer_api_key", "")
        return bool(cookie_key) and secrets.compare_digest(cookie_key, api_key)

    def _get_tracer_for_dashboard():
        """Get a tracer instance for dashboard data fetching."""
        try:
            from longtracer.guard.tracer import Tracer
            from longtracer.guard.cache import get_default_backend
            return Tracer(
                run_name="longtracer_dashboard",
                backend=get_default_backend(),
            )
        except Exception:
            return None

    def _fmt_dur(ms) -> str:
        if ms is None:
            return "N/A"
        return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.2f}s"

    def _fmt_dt(dt) -> str:
        if dt is None:
            return "N/A"
        if isinstance(dt, datetime):
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return str(dt)[:19] if dt else "N/A"

    # Mount static files (must be before routes)
    if _TEMPLATES_AVAILABLE:
        try:
            app.mount(
                "/static",
                StaticFiles(directory=str(_DASHBOARD_DIR / "static")),
                name="dashboard-static",
            )
        except Exception as exc:
            logger.warning("Failed to mount dashboard static files: %s", exc)

    @app.get("/dashboard/login", include_in_schema=False)
    async def dashboard_login_page(request: Request):
        """Render the login page."""
        if not _TEMPLATES_AVAILABLE:
            return HTMLResponse("<h1>Dashboard unavailable (Jinja2 not installed)</h1>")
        # If already authenticated, redirect to dashboard
        if _check_dashboard_auth(request):
            return RedirectResponse(url="/dashboard", status_code=302)
        return templates.TemplateResponse(request, "login.html", {
            "error": None,
            "api_key_configured": bool(api_key),
        })

    @app.post("/dashboard/login", include_in_schema=False)
    async def dashboard_login_submit(request: Request):
        """Process login form submission."""
        if not _TEMPLATES_AVAILABLE:
            return HTMLResponse("<h1>Dashboard unavailable</h1>")
        form = await request.form()
        provided_key = form.get("api_key", "")

        # If no API key configured on server, allow any input
        if not api_key or secrets.compare_digest(provided_key, api_key):
            response = RedirectResponse(url="/dashboard", status_code=302)
            response.set_cookie(
                key="longtracer_api_key",
                value=provided_key or "",
                httponly=True,
                samesite="strict",
                max_age=86400 * 7,  # 7 days
            )
            return response

        return templates.TemplateResponse(request, "login.html", {
            "error": "Invalid API key. Please try again.",
            "api_key_configured": bool(api_key),
        })

    @app.post("/dashboard/logout", include_in_schema=False)
    async def dashboard_logout():
        """Clear the auth cookie and redirect to login."""
        response = RedirectResponse(url="/dashboard/login", status_code=302)
        response.delete_cookie("longtracer_api_key")
        return response

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_home(request: Request):
        """Dashboard overview page (metrics placeholder)."""
        if not _TEMPLATES_AVAILABLE:
            return HTMLResponse("<h1>Dashboard unavailable (Jinja2 not installed)</h1>")
        if not _check_dashboard_auth(request):
            return RedirectResponse(url="/dashboard/login", status_code=302)

        tracer = _get_tracer_for_dashboard()
        projects = []
        if tracer:
            try:
                traces = tracer.list_recent_traces(limit=100)
                projects = sorted(set(
                    t.get("project_name", "") for t in traces if t.get("project_name")
                ))
            except Exception:
                pass

        return templates.TemplateResponse(request, "metrics.html", {
            "page": "metrics",
            "projects": projects,
            "selected_project": None,
            "api_key_configured": bool(api_key),
            "error_message": None,
        })

    @app.get("/dashboard/traces", include_in_schema=False)
    async def dashboard_traces(
        request: Request,
        page: int = 1,
        project: Optional[str] = None,
    ):
        """Trace list page with pagination and project filter."""
        if not _TEMPLATES_AVAILABLE:
            return HTMLResponse("<h1>Dashboard unavailable</h1>")
        if not _check_dashboard_auth(request):
            return RedirectResponse(url="/dashboard/login", status_code=302)

        per_page = 20
        tracer = _get_tracer_for_dashboard()
        traces_raw = []
        projects = []

        if tracer:
            try:
                all_traces = tracer.list_recent_traces(limit=500, project_name=project or None)
                projects = sorted(set(
                    t.get("project_name", "") for t in all_traces if t.get("project_name")
                ))
                traces_raw = all_traces
            except Exception:
                pass

        # Format traces for display
        traces = []
        for t in traces_raw:
            traces.append({
                "trace_id": t.get("trace_id", ""),
                "project_name": t.get("project_name", ""),
                "trust_score": t.get("trust_score"),
                "hallucination_count": t.get("hallucination_count"),
                "claim_count": t.get("claim_count"),
                "duration_ms": t.get("duration_ms"),
                "duration_fmt": _fmt_dur(t.get("duration_ms")),
                "created_at": t.get("created_at"),
                "created_fmt": _fmt_dt(t.get("created_at")),
            })

        # Pagination
        total = len(traces)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * per_page
        traces_page = traces[start_idx:start_idx + per_page]

        return templates.TemplateResponse(request, "traces.html", {
            "page": "traces",
            "traces": traces_page,
            "projects": projects,
            "selected_project": project,
            "page_num": page,
            "total_pages": total_pages,
            "api_key_configured": bool(api_key),
            "error_message": None,
        })

    @app.get("/dashboard/traces/{trace_id}", include_in_schema=False)
    async def dashboard_trace_detail(request: Request, trace_id: str):
        """Single trace detail page."""
        if not _TEMPLATES_AVAILABLE:
            return HTMLResponse("<h1>Dashboard unavailable</h1>")
        if not _check_dashboard_auth(request):
            return RedirectResponse(url="/dashboard/login", status_code=302)

        tracer = _get_tracer_for_dashboard()
        if not tracer:
            return HTMLResponse("<h1>No data source configured</h1>", status_code=503)

        trace = tracer.get_trace(trace_id)
        if not trace:
            return HTMLResponse(
                '<h1>Trace Not Found</h1><p>The requested trace does not exist.</p>'
                '<a href="/dashboard/traces">← Back to Traces</a>',
                status_code=404,
            )

        # Get child runs
        runs_raw = tracer.get_runs_by_trace(trace_id)
        child_runs = [r for r in runs_raw if r.get("run_id") != trace_id]

        # Format runs for display
        runs = []
        for r in child_runs:
            runs.append({
                "name": r.get("name", "unknown"),
                "run_type": r.get("run_type", "chain"),
                "duration_ms": r.get("duration_ms"),
                "duration_fmt": _fmt_dur(r.get("duration_ms")),
                "error": r.get("error"),
                "outputs": r.get("outputs", {}),
            })

        # Format trace for display
        trace_display = {
            "trace_id": trace.get("trace_id", ""),
            "project_name": trace.get("project_name", ""),
            "run_name": trace.get("run_name", ""),
            "trust_score": trace.get("trust_score"),
            "hallucination_count": trace.get("hallucination_count"),
            "claim_count": trace.get("claim_count"),
            "duration_ms": trace.get("duration_ms"),
            "duration_fmt": _fmt_dur(trace.get("duration_ms")),
            "created_at": trace.get("created_at"),
            "created_fmt": _fmt_dt(trace.get("created_at")),
            "inputs": trace.get("inputs", {}),
            "outputs": trace.get("outputs", {}),
        }

        claim_evidence_map = trace.get("claim_evidence_map", {})

        return templates.TemplateResponse(request, "trace_detail.html", {
            "page": "traces",
            "trace": trace_display,
            "runs": runs,
            "claim_evidence_map": claim_evidence_map,
            "api_key_configured": bool(api_key),
            "error_message": None,
        })

    # ── Global error handler ────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Catch-all handler — never expose internal details."""
        logger.error("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8100,
    workers: int = 1,
    reload: bool = False,
) -> None:
    """Start the LongTracer REST API server.

    Args:
        host: Bind address (default 0.0.0.0).
        port: Port number (default 8100).
        workers: Number of worker processes.
        reload: Enable auto-reload for development.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install 'longtracer[server]'"
        )

    uvicorn.run(
        "longtracer.server:create_app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        factory=True,
        log_level="info",
    )
