# Alerting System

LongTracer features an active, asynchronous alerting system designed to notify you the moment an LLM response falls below your acceptable trust threshold.

## How It Works

Alerts are dispatched in **background daemon threads**. This means that network delays (like an SMTP timeout or a slow Slack API) will **never** block your main verification pipeline or add latency to your RAG generation loop.

## Configuration

You can configure alerting via environment variables or inside your `pyproject.toml` file under the `[tool.longtracer]` table.

### 1. Webhook

Sends an HMAC-signed POST request with the verification payload.

**Environment Variables:**
- `LONGTRACER_WEBHOOK_URL`
- `LONGTRACER_WEBHOOK_SECRET`

### 2. Slack

Sends a formatted notification to a Slack channel via Incoming Webhook.

**Environment Variables:**
- `LONGTRACER_SLACK_WEBHOOK`

### 3. Discord

Sends an embed notification to a Discord channel.

**Environment Variables:**
- `LONGTRACER_DISCORD_WEBHOOK`

### 4. Email (SMTP)

Sends an email report.

**Environment Variables:**
- `LONGTRACER_SMTP_HOST`
- `LONGTRACER_SMTP_PORT`
- `LONGTRACER_SMTP_USER`
- `LONGTRACER_SMTP_PASS`
- `LONGTRACER_EMAIL_FROM`
- `LONGTRACER_EMAIL_TO`

## Threshold

By default, an alert is triggered if the trust score falls below `0.5`. You can change this threshold:

```bash
export LONGTRACER_ALERT_THRESHOLD="0.8"
```
