"""
Alert dispatcher for LongTracer.

Sends alerts when trust scores drop below a configurable threshold.
Supports four channels: webhook, Slack, Discord, and email (SMTP).

All channel dispatches run in background daemon threads — they never
block the verification pipeline or API responses.

Configuration (priority: env vars > pyproject.toml > defaults):

    Env vars:
        LONGTRACER_ALERT_THRESHOLD   — float, fire alert if trust_score < this
        LONGTRACER_ALERT_CHANNELS    — comma-separated: webhook,slack,discord,email
        LONGTRACER_SLACK_WEBHOOK_URL — Slack Incoming Webhook URL
        LONGTRACER_DISCORD_WEBHOOK_URL — Discord Webhook URL
        LONGTRACER_ALERT_EMAIL_FROM  — Sender email address
        LONGTRACER_ALERT_EMAIL_TO    — Comma-separated recipient emails
        LONGTRACER_SMTP_HOST         — SMTP server hostname
        LONGTRACER_SMTP_PORT         — SMTP port (default: 587)
        LONGTRACER_SMTP_USER         — SMTP username
        LONGTRACER_SMTP_PASSWORD     — SMTP password or app password

    pyproject.toml:
        [tool.longtracer]
        alert_threshold = 0.6
        alert_channels = ["webhook", "slack"]
        slack_webhook_url = "https://hooks.slack.com/services/..."
        discord_webhook_url = "https://discord.com/api/webhooks/..."
        alert_email_from = "longtracer@example.com"
        alert_email_to = ["oncall@example.com"]
        smtp_host = "smtp.gmail.com"
        smtp_port = "587"
        smtp_user = "longtracer@example.com"
        smtp_password = "app-password"

Usage (SDK users):
    from longtracer.alerts import dispatch_alert

    result = check("response", ["source"])
    dispatch_alert(result, project="my-chatbot")
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger("longtracer")

# ── Config loading ──────────────────────────────────────────────


def _load_alert_config() -> Dict[str, Any]:
    """Load alert configuration from env vars and pyproject.toml."""
    from longtracer.config import load_config
    cfg = load_config()

    def _get(env_key: str, cfg_key: str, default: Any = None) -> Any:
        val = os.environ.get(env_key)
        if val is not None:
            return val
        return cfg.get(cfg_key, default)

    threshold_str = _get("LONGTRACER_ALERT_THRESHOLD", "alert_threshold")
    threshold = None
    if threshold_str is not None:
        try:
            threshold = float(threshold_str)
        except (ValueError, TypeError):
            pass

    channels_str = _get("LONGTRACER_ALERT_CHANNELS", "alert_channels")
    channels = []
    if isinstance(channels_str, str):
        channels = [c.strip() for c in channels_str.split(",") if c.strip()]
    elif isinstance(channels_str, list):
        channels = channels_str

    email_to_str = _get("LONGTRACER_ALERT_EMAIL_TO", "alert_email_to")
    email_to = []
    if isinstance(email_to_str, str):
        email_to = [e.strip() for e in email_to_str.split(",") if e.strip()]
    elif isinstance(email_to_str, list):
        email_to = email_to_str

    smtp_port_str = _get("LONGTRACER_SMTP_PORT", "smtp_port", "587")
    try:
        smtp_port = int(smtp_port_str)
    except (ValueError, TypeError):
        smtp_port = 587

    return {
        "threshold": threshold,
        "channels": channels,
        "slack_webhook_url": _get("LONGTRACER_SLACK_WEBHOOK_URL", "slack_webhook_url", ""),
        "discord_webhook_url": _get("LONGTRACER_DISCORD_WEBHOOK_URL", "discord_webhook_url", ""),
        "email_from": _get("LONGTRACER_ALERT_EMAIL_FROM", "alert_email_from", ""),
        "email_to": email_to,
        "smtp_host": _get("LONGTRACER_SMTP_HOST", "smtp_host", ""),
        "smtp_port": smtp_port,
        "smtp_user": _get("LONGTRACER_SMTP_USER", "smtp_user", ""),
        "smtp_password": _get("LONGTRACER_SMTP_PASSWORD", "smtp_password", ""),
    }


# ── Public API ──────────────────────────────────────────────────


def check_alert_threshold(trust_score: float, threshold: float) -> bool:
    """Check if trust_score is below the alert threshold.

    Args:
        trust_score: The verification trust score (0.0–1.0).
        threshold: The alert threshold (0.0–1.0).

    Returns:
        True if trust_score < threshold (alert should fire).
    """
    return trust_score < threshold


def dispatch_alert(
    result: Any,
    project: str = "default",
    trace_id: Optional[str] = None,
    dashboard_url: Optional[str] = None,
) -> Optional[str]:
    """Dispatch alerts to all configured channels if trust_score < threshold.

    For SDK users to call manually after verify(). Checks the configured
    alert_threshold and fires alerts to each channel listed in alert_channels.
    All dispatches run in background threads.

    Args:
        result: A VerificationResult object.
        project: Project name for the alert payload.
        trace_id: Optional trace ID to include in the alert.
        dashboard_url: Optional link to the dashboard trace view.

    Returns:
        A dispatch ID string if alerts were dispatched, None if skipped.
    """
    config = _load_alert_config()

    threshold = config.get("threshold")
    if threshold is None:
        logger.debug("Alerting disabled: no alert_threshold configured")
        return None

    trust_score = getattr(result, "trust_score", None)
    if trust_score is None:
        return None

    if not check_alert_threshold(trust_score, threshold):
        logger.debug(
            "Alert skipped: trust_score %.2f >= threshold %.2f",
            trust_score, threshold,
        )
        return None

    channels = config.get("channels", [])
    if not channels:
        logger.debug("Alert skipped: no alert_channels configured")
        return None

    # Build alert payload
    payload = _build_alert_payload(
        result=result,
        threshold=threshold,
        project=project,
        trace_id=trace_id,
        dashboard_url=dashboard_url,
    )

    dispatch_id = f"alert-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    # Dispatch each channel in a background thread
    for channel in channels:
        thread = threading.Thread(
            target=_dispatch_channel,
            args=(channel, payload, config),
            daemon=True,
            name=f"longtracer-alert-{channel}",
        )
        thread.start()

    logger.info(
        "Alert dispatched: %s (trust_score=%.2f, threshold=%.2f, channels=%s)",
        dispatch_id, trust_score, threshold, channels,
    )
    return dispatch_id


# ── Alert payload ───────────────────────────────────────────────


def _build_alert_payload(
    result: Any,
    threshold: float,
    project: str,
    trace_id: Optional[str],
    dashboard_url: Optional[str],
) -> Dict[str, Any]:
    """Build the standard alert payload."""
    trust_score = getattr(result, "trust_score", 0.0)
    verdict = getattr(result, "verdict", "FAIL")
    hallucination_count = getattr(result, "hallucination_count", 0)
    claims = getattr(result, "claims", [])
    claim_count = len(claims) if claims else 0

    return {
        "alert_type": "low_trust_score",
        "threshold": threshold,
        "trust_score": trust_score,
        "verdict": verdict,
        "hallucination_count": hallucination_count,
        "claim_count": claim_count,
        "project": project,
        "trace_id": trace_id or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dashboard_url": dashboard_url or "",
    }


# ── Channel dispatcher ──────────────────────────────────────────


def _dispatch_channel(
    channel: str,
    payload: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Dispatch an alert to a single channel (runs in a thread)."""
    try:
        if channel == "webhook":
            _send_webhook_alert(payload, config)
        elif channel == "slack":
            _send_slack(payload, config)
        elif channel == "discord":
            _send_discord(payload, config)
        elif channel == "email":
            _send_email(payload, config)
        else:
            logger.warning("Unknown alert channel: %s", channel)
    except Exception as exc:
        logger.warning("Alert dispatch failed for %s: %s", channel, exc)


# ── Channel implementations ─────────────────────────────────────


def _send_slack(payload: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Send alert to Slack via Incoming Webhook."""
    url = config.get("slack_webhook_url", "")
    if not url:
        logger.warning("Slack alert skipped: slack_webhook_url not configured")
        return

    score_pct = f"{payload['trust_score'] * 100:.0f}%"
    slack_payload = {
        "text": f"🚨 LongTracer Alert: Low Trust Score ({score_pct})",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Trust Score Alert",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Score:* {payload['trust_score']:.2f} / 1.0"},
                    {"type": "mrkdwn", "text": f"*Threshold:* {payload['threshold']}"},
                    {"type": "mrkdwn", "text": f"*Hallucinations:* {payload['hallucination_count']}"},
                    {"type": "mrkdwn", "text": f"*Project:* {payload['project']}"},
                    {"type": "mrkdwn", "text": f"*Verdict:* {payload['verdict']}"},
                    {"type": "mrkdwn", "text": f"*Claims:* {payload['claim_count']}"},
                ],
            },
        ],
    }

    if payload.get("dashboard_url"):
        slack_payload["blocks"].append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View in Dashboard"},
                    "url": payload["dashboard_url"],
                },
            ],
        })

    _http_post_json(url, slack_payload, "Slack")


def _send_discord(payload: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Send alert to Discord via Webhook."""
    url = config.get("discord_webhook_url", "")
    if not url:
        logger.warning("Discord alert skipped: discord_webhook_url not configured")
        return

    content = (
        f"🚨 **Trust Score Alert** — "
        f"Score: {payload['trust_score']:.2f} (threshold: {payload['threshold']}) | "
        f"Hallucinations: {payload['hallucination_count']} | "
        f"Project: {payload['project']} | "
        f"Verdict: {payload['verdict']}"
    )
    if payload.get("dashboard_url"):
        content += f" | [View Dashboard]({payload['dashboard_url']})"

    _http_post_json(url, {"content": content}, "Discord")


def _send_webhook_alert(payload: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Send alert via existing webhook infrastructure."""
    from longtracer.webhooks import dispatch_webhook

    dispatch_webhook(
        event="alert.low_trust_score",
        data=payload,
        async_delivery=False,  # Already in a background thread
    )


def _send_email(payload: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Send alert via email (SMTP with TLS)."""
    email_to = config.get("email_to", [])
    smtp_host = config.get("smtp_host", "")
    if not email_to or not smtp_host:
        logger.warning(
            "Email alert skipped: missing email_to or smtp_host config"
        )
        return

    smtp_port = config.get("smtp_port", 587)
    smtp_user = config.get("smtp_user", "")
    smtp_password = config.get("smtp_password", "")
    email_from = config.get("email_from", smtp_user or "longtracer@localhost")

    score_pct = f"{payload['trust_score'] * 100:.0f}%"
    subject = f"🚨 LongTracer Alert: Low Trust Score ({score_pct})"

    body_lines = [
        f"LongTracer Trust Score Alert",
        f"",
        f"  Trust Score:    {payload['trust_score']:.2f} / 1.0",
        f"  Threshold:      {payload['threshold']}",
        f"  Verdict:        {payload['verdict']}",
        f"  Hallucinations: {payload['hallucination_count']}",
        f"  Claims:         {payload['claim_count']}",
        f"  Project:        {payload['project']}",
        f"  Trace ID:       {payload.get('trace_id', 'N/A')}",
        f"  Timestamp:      {payload['timestamp']}",
    ]
    if payload.get("dashboard_url"):
        body_lines.append(f"  Dashboard:      {payload['dashboard_url']}")

    body = "\n".join(body_lines)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = ", ".join(email_to)

    import smtplib

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.debug("Email alert sent to %s", email_to)
    except smtplib.SMTPAuthenticationError as exc:
        logger.warning(
            "Email alert failed: SMTP auth error. "
            "Check smtp_user/smtp_password (may need an app password). "
            "Details: %s",
            exc,
        )
    except Exception as exc:
        logger.warning("Email alert failed: %s", exc)


# ── HTTP helper ─────────────────────────────────────────────────


def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    label: str,
) -> bool:
    """Send a JSON POST request using urllib (zero extra deps)."""
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "LongTracer-Alert/0.4.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if 200 <= resp.status < 300:
                logger.debug("%s alert delivered (status=%d)", label, resp.status)
                return True
            else:
                logger.warning(
                    "%s alert returned non-2xx: status=%d", label, resp.status
                )
                return False
    except urllib.error.HTTPError as exc:
        logger.warning("%s alert HTTP error: status=%d", label, exc.code)
        return False
    except urllib.error.URLError as exc:
        logger.warning("%s alert URL error: %s", label, exc.reason)
        return False
    except Exception as exc:
        logger.warning("%s alert failed: %s", label, exc)
        return False
