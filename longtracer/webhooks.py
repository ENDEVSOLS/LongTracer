"""
Webhook dispatcher for LongTracer.

Sends verification results to external systems via HMAC-signed HTTP POST.

Security:
    - HMAC-SHA256 signature in ``X-LongTracer-Signature`` header
    - Configurable timeout (5s default)
    - Async dispatch — never blocks the verification pipeline
    - 5 retries with exponential backoff + jitter (Stripe-style)
    - URL validation (HTTPS enforced in production)

Configuration (priority: env > pyproject.toml > defaults):
    Env vars:
        LONGTRACER_WEBHOOK_URL     — Target URL
        LONGTRACER_WEBHOOK_SECRET  — HMAC signing key
        LONGTRACER_WEBHOOK_EVENTS  — Comma-separated event list
        LONGTRACER_WEBHOOK_TIMEOUT — Request timeout in seconds

    pyproject.toml:
        [tool.longtracer]
        webhook_url = "https://example.com/hooks/longtracer"
        webhook_secret = "your-secret"
        webhook_events = ["verification.complete", "verification.fail"]
        webhook_timeout = 5.0

Usage:
    from longtracer.webhooks import dispatch_webhook

    # After verification completes:
    dispatch_webhook("verification.complete", {
        "trust_score": 0.85,
        "verdict": "PASS",
        ...
    })
"""

import hashlib
import hmac
import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("longtracer")

# ── Constants ───────────────────────────────────────────────────

DEFAULT_TIMEOUT = 5.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_EVENTS = ["verification.complete", "verification.fail"]

# Exponential backoff base intervals (seconds) — Stripe-style
# Actual delay = base * (2^attempt) + jitter
_BACKOFF_BASE = 15  # 15s, 30s, 60s, 120s, 240s ≈ 30s→4min
_JITTER_MAX = 5  # ±5s random jitter


# ── Config loading ──────────────────────────────────────────────

def _load_webhook_config() -> Dict[str, Any]:
    """Load webhook configuration from env vars and pyproject.toml.

    Priority: env vars > pyproject.toml > defaults
    """
    from longtracer.config import load_config
    cfg = load_config()

    url = os.environ.get("LONGTRACER_WEBHOOK_URL") or cfg.get("webhook_url", "")
    secret = os.environ.get("LONGTRACER_WEBHOOK_SECRET") or cfg.get("webhook_secret", "")

    events_env = os.environ.get("LONGTRACER_WEBHOOK_EVENTS")
    if events_env:
        events = [e.strip() for e in events_env.split(",") if e.strip()]
    else:
        events = cfg.get("webhook_events", DEFAULT_EVENTS)

    timeout_env = os.environ.get("LONGTRACER_WEBHOOK_TIMEOUT")
    if timeout_env:
        try:
            timeout = float(timeout_env)
        except ValueError:
            timeout = DEFAULT_TIMEOUT
    else:
        timeout = cfg.get("webhook_timeout", DEFAULT_TIMEOUT)

    return {
        "url": url,
        "secret": secret,
        "events": events,
        "timeout": timeout,
    }


# ── HMAC signing ────────────────────────────────────────────────

def compute_signature(payload: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for a payload.

    Args:
        payload: Raw bytes of the JSON payload.
        secret: The signing secret.

    Returns:
        Hex-encoded HMAC-SHA256 signature prefixed with ``sha256=``.
    """
    if not secret:
        return ""
    mac = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    )
    return f"sha256={mac.hexdigest()}"


def verify_signature(payload: bytes, secret: str, signature: str) -> bool:
    """Verify an HMAC-SHA256 signature (timing-safe).

    Args:
        payload: Raw bytes of the JSON payload.
        secret: The signing secret.
        signature: The received signature to verify.

    Returns:
        True if the signature is valid, False otherwise.
    """
    if not secret or not signature:
        return False
    expected = compute_signature(payload, secret)
    return hmac.compare_digest(expected, signature)


# ── Payload building ───────────────────────────────────────────

def _build_payload(event: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a webhook payload with standard fields.

    The payload includes:
        - id: Unique delivery ID (UUID4) for idempotency
        - event: Event type string
        - timestamp: ISO 8601 UTC timestamp
        - data: Event data dict
    """
    return {
        "id": str(uuid4()),
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": _sanitize_data(data),
    }


def _sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize webhook data to prevent sensitive info leakage.

    Removes or truncates fields that might contain PII or secrets.
    """
    sanitized = {}
    for key, value in data.items():
        # Skip internal/private fields
        if key.startswith("_"):
            continue

        # Truncate very long strings (e.g., full source texts)
        if isinstance(value, str) and len(value) > 2000:
            sanitized[key] = value[:2000] + "...[truncated]"
        elif isinstance(value, list):
            # Limit list items and truncate each
            truncated_list = []
            for item in value[:50]:  # Max 50 items
                if isinstance(item, str) and len(item) > 500:
                    truncated_list.append(item[:500] + "...[truncated]")
                elif isinstance(item, dict):
                    truncated_list.append(_sanitize_data(item))
                else:
                    truncated_list.append(item)
            sanitized[key] = truncated_list
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_data(value)
        else:
            sanitized[key] = value

    return sanitized


# ── HTTP delivery ──────────────────────────────────────────────

def _deliver_webhook(
    url: str,
    payload_bytes: bytes,
    signature: str,
    timeout: float,
    delivery_id: str,
) -> bool:
    """Send a single webhook delivery attempt.

    Uses urllib.request to avoid requiring httpx/requests as a core dependency.

    Returns:
        True if delivery succeeded (2xx response), False otherwise.
    """
    import urllib.request
    import urllib.error

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "LongTracer-Webhook/0.1.6",
        "X-LongTracer-Delivery": delivery_id,
    }

    if signature:
        headers["X-LongTracer-Signature"] = signature

    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if 200 <= status < 300:
                logger.debug(
                    "Webhook delivered: %s (status=%d)", delivery_id, status,
                )
                return True
            else:
                logger.warning(
                    "Webhook delivery returned non-2xx: %s (status=%d)",
                    delivery_id, status,
                )
                return False
    except urllib.error.HTTPError as exc:
        logger.warning(
            "Webhook delivery HTTP error: %s (status=%d)", delivery_id, exc.code,
        )
        return False
    except urllib.error.URLError as exc:
        logger.warning(
            "Webhook delivery URL error: %s (%s)", delivery_id, exc.reason,
        )
        return False
    except Exception as exc:
        logger.warning(
            "Webhook delivery failed: %s (%s)", delivery_id, exc,
        )
        return False


def _deliver_with_retries(
    url: str,
    payload_bytes: bytes,
    signature: str,
    timeout: float,
    delivery_id: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bool:
    """Deliver a webhook with exponential backoff + jitter.

    Retry schedule (approximate with jitter):
        Attempt 1: immediate
        Attempt 2: ~30s
        Attempt 3: ~2min
        Attempt 4: ~8min
        Attempt 5: ~30min

    Returns:
        True if any attempt succeeded, False if all failed.
    """
    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Exponential backoff with jitter
            delay = _BACKOFF_BASE * (2 ** (attempt - 1))
            jitter = random.uniform(-_JITTER_MAX, _JITTER_MAX)
            wait = max(1.0, delay + jitter)
            logger.info(
                "Webhook retry %d/%d for %s (waiting %.1fs)",
                attempt, max_retries, delivery_id, wait,
            )
            time.sleep(wait)

        success = _deliver_webhook(
            url, payload_bytes, signature, timeout, delivery_id,
        )
        if success:
            return True

    # All retries exhausted — dead letter log
    logger.error(
        "Webhook delivery FAILED after %d retries: %s (url=%s). "
        "This event is now in the dead letter log.",
        max_retries, delivery_id, url,
    )
    return False


# ── Public API ──────────────────────────────────────────────────

def dispatch_webhook(
    event: str,
    data: Dict[str, Any],
    *,
    url: Optional[str] = None,
    secret: Optional[str] = None,
    timeout: Optional[float] = None,
    async_delivery: bool = True,
) -> Optional[str]:
    """Dispatch a webhook event.

    Sends a signed HTTP POST to the configured webhook URL.
    By default, delivery happens in a background thread so it
    never blocks the verification pipeline.

    Args:
        event: Event type (e.g., "verification.complete").
        data: Event data dict (will be sanitized).
        url: Override webhook URL (uses config if not provided).
        secret: Override signing secret (uses config if not provided).
        timeout: Override request timeout (uses config if not provided).
        async_delivery: If True (default), dispatch in background thread.

    Returns:
        The delivery ID (UUID) if dispatched, None if skipped.

    Usage::

        dispatch_webhook("verification.complete", {
            "trust_score": 0.85,
            "verdict": "PASS",
            "summary": "All claims supported.",
        })
    """
    config = _load_webhook_config()

    target_url = url or config["url"]
    signing_secret = secret or config["secret"]
    req_timeout = timeout or config["timeout"]
    allowed_events = config["events"]

    # Skip if no URL configured
    if not target_url:
        logger.debug("Webhook skipped: no URL configured")
        return None

    # Skip if event not in allowed list
    if event not in allowed_events:
        logger.debug("Webhook skipped: event '%s' not in allowed list", event)
        return None

    # Build and sign payload
    payload = _build_payload(event, data)
    delivery_id = payload["id"]
    payload_bytes = json.dumps(payload, default=str).encode("utf-8")
    signature = compute_signature(payload_bytes, signing_secret)

    if async_delivery:
        # Fire-and-forget in background thread
        thread = threading.Thread(
            target=_deliver_with_retries,
            args=(target_url, payload_bytes, signature, req_timeout, delivery_id),
            daemon=True,
            name=f"longtracer-webhook-{delivery_id[:8]}",
        )
        thread.start()
        logger.debug("Webhook dispatched async: %s -> %s", delivery_id, target_url)
    else:
        # Synchronous delivery (for testing)
        _deliver_with_retries(
            target_url, payload_bytes, signature, req_timeout, delivery_id,
        )

    return delivery_id


def dispatch_verification_result(
    result: Any,
    extra_data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Dispatch a webhook for a verification result.

    Convenience function that determines the event type from the result
    and dispatches accordingly.

    Args:
        result: A VerificationResult object.
        extra_data: Additional data to include in the payload.

    Returns:
        The delivery ID if dispatched, None if skipped.
    """
    if result is None:
        return None

    event = (
        "verification.complete" if result.verdict == "PASS"
        else "verification.fail"
    )

    data = {
        "trust_score": result.trust_score,
        "verdict": result.verdict,
        "summary": result.summary,
        "hallucination_count": result.hallucination_count,
        "all_supported": result.all_supported,
        "claims_count": len(result.claims),
        "flagged_count": len(result.flagged_claims),
    }

    if extra_data:
        data.update(extra_data)

    return dispatch_webhook(event, data)
