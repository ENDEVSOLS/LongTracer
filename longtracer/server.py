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
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError(
            "FastAPI and uvicorn are required for the REST API server. "
            "Install with: pip install 'longtracer[server]'"
        )

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
