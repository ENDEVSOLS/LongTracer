# Blueprint: Observability & Analytics (v0.4.0)

## Overview

Transform LongTracer from a verification-only SDK into a full observability platform for RAG pipelines. This adds five integrated features: (1) a built-in web dashboard for browsing traces across projects, (2) aggregated metrics APIs deriving trust-score trends and hallucination rates from existing trace data, (3) multi-channel alerting (webhook, Slack, Discord, email) when trust scores drop below a configurable threshold, (4) OpenTelemetry span export for `verify()` calls via OTLP, and (5) a pre-built Grafana dashboard JSON template. All features are served from the existing `longtracer serve` FastAPI server with zero Node.js dependencies.

## Context

- **Current state**: LongTracer v0.1.6 has core STS+NLI verification, REST API server (`server.py`), webhook dispatch (`webhooks.py`), pluggable trace backends (SQLite/Memory/Mongo/Postgres/Redis), and a CLI viewer.
- **This touches**: `server.py` (new endpoints + dashboard routes), `guard/tracer.py` (metrics in trace docs), `guard/cache/backend.py` (metrics query interface), all backend implementations, `config.py` (new config keys), `webhooks.py` (extend for alerts), plus new modules: `alerts.py`, `otel.py`, `dashboard/` package.
- **Patterns to follow**: Config priority chain (code args > env vars > pyproject.toml > defaults), safe backend I/O wrapped in try/except, background thread dispatch for network calls, Pydantic models for API validation, optional dependency groups via `[project.optional-dependencies]`.

## Requirements

1. **Dashboard UI**: A web interface at `/dashboard/*` served by `longtracer serve` using Jinja2 templates + HTMX + Chart.js. Shows trace list, trace detail, and metrics overview. Protected by the existing API key auth (stored in browser localStorage).
2. **Metrics Data Layer**: Every trace saved by the server's verify endpoint includes `trust_score`, `hallucination_count`, and `claim_count` as top-level fields in the trace JSON document. No new database tables or migrations.
3. **Metrics API**: `GET /api/v1/metrics/summary` returns aggregated stats (avg trust score, pass rate, total traces, total hallucinations) filtered by project and date range. `GET /api/v1/metrics/timeseries` returns time-bucketed data for charting.
4. **Alerting — Automatic (Server)**: When the server's verify endpoint returns a result with `trust_score < alert_threshold`, alerts fire automatically to all configured channels (webhook, Slack, Discord, email) in background threads.
5. **Alerting — Manual (SDK)**: A public `dispatch_alert(result)` function in `longtracer.alerts` that SDK users can call explicitly after `verify()`.
6. **Alert Channels**: Slack (Incoming Webhook HTTP POST), Discord (Webhook HTTP POST), Email (stdlib `smtplib`), and existing webhook infrastructure. Zero new external dependencies.
7. **OpenTelemetry Export**: Optional `opentelemetry-api` + `opentelemetry-sdk` + `opentelemetry-exporter-otlp` dependency group. When enabled, each `verify()` call is wrapped in an OTel span with attributes: `trust_score`, `claim_count`, `hallucination_count`, `verdict`, `project_id`. Export via OTLP protocol.
8. **Grafana Template**: A `grafana/longtracer.json` file in the repo with pre-built panels: trust score over time, hallucination rate, claims per project, verification latency. Users import into Grafana and point at their OTel data source.
9. **Backward Compatibility**: All new features are opt-in. Existing code works identically without any configuration changes. No breaking changes to public APIs.

## Out of Scope

- React-based UI or any Node.js build tooling
- New database tables, collections, or schema migrations
- FastAPI auto-instrumentation via `opentelemetry-instrumentation-fastapi`
- Role-based access control or audit logging (v0.5.0)
- PII redaction (v0.5.0)
- Cloud-hosted trace storage (v0.5.0)
- Native Slack Bot API or Discord Bot API (only webhook-based notifications)
- Dashboard real-time WebSocket updates (HTMX polling is sufficient)
- Multi-tenant isolation or user management

## Technical Design

### Architecture

```
longtracer/
  server.py            # MODIFIED — add dashboard routes + metrics API + alert dispatch
  alerts.py            # NEW — multi-channel alert dispatcher
  otel.py              # NEW — OpenTelemetry integration (optional)
  config.py            # MODIFIED — add alert + otel config keys
  webhooks.py          # UNCHANGED — existing webhook infra reused by alerts
  cli.py               # MODIFIED — minor: dashboard link in serve output
  dashboard/           # NEW — Jinja2 templates + static assets
    __init__.py
    templates/
      base.html        # Layout: sidebar, nav, API key auth
      login.html       # API key input form
      traces.html      # Trace list with HTMX pagination
      trace_detail.html # Single trace with spans + claims
      metrics.html     # Charts overview
    static/
      css/
        dashboard.css
      js/
        dashboard.js   # HTMX + Chart.js logic
  guard/
    tracer.py          # MODIFIED — end_root() accepts metrics dict
    cache/
      backend.py       # MODIFIED — add get_metrics_summary() + get_metrics_timeseries()
      sqlite.py        # MODIFIED — optimized metrics via json_extract
      memory.py        # MODIFIED — optimized metrics via iteration
      mongo.py         # UNCHANGED — falls back to base class default
      postgres.py      # UNCHANGED — falls back to base class default
      redis_backend.py # UNCHANGED — falls back to base class default
grafana/
  longtracer.json      # NEW — Grafana dashboard template
```

### Data Models / Schemas

#### Trace Document (Extended)

The existing trace JSON document gains three optional top-level fields:

```python
# Current shape (in end_root → save_trace):
{
    "trace_id": "uuid",
    "project_name": "string",
    "run_name": "string",
    "inputs": {...},
    "outputs": {...},
    "claim_evidence_map": {...},
    "created_at": "datetime",
    "duration_ms": 1234.5,
    "run_count": 3
}

# New shape (with metrics):
{
    "trace_id": "uuid",
    "project_name": "string",
    "run_name": "string",
    "inputs": {...},
    "outputs": {...},
    "claim_evidence_map": {...},
    "created_at": "datetime",
    "duration_ms": 1234.5,
    "run_count": 3,
    "trust_score": 0.85,            # NEW — nullable float
    "hallucination_count": 1,        # NEW — nullable int
    "claim_count": 5                 # NEW — nullable int
}
```

These fields are optional and nullable — existing traces without them simply won't contribute to metrics aggregation.

#### Metrics Summary Response

```python
{
    "project": "my-chatbot",         # null if unfiltered
    "start_time": "2026-05-07T00:00:00Z",
    "end_time": "2026-05-14T23:59:59Z",
    "total_traces": 150,
    "avg_trust_score": 0.82,
    "min_trust_score": 0.45,
    "max_trust_score": 1.0,
    "pass_rate": 0.87,               # verdict==PASS / total
    "total_hallucinations": 23,
    "total_claims": 750,
    "hallucination_rate": 0.031      # total_hallucinations / total_claims
}
```

#### Metrics Timeseries Response

```python
{
    "project": "my-chatbot",
    "interval": "1d",
    "data_points": [
        {
            "bucket": "2026-05-07",
            "trace_count": 18,
            "avg_trust_score": 0.88,
            "hallucination_count": 2,
            "claim_count": 90,
            "pass_rate": 0.94
        },
        ...
    ]
}
```

#### Alert Config

```toml
# pyproject.toml [tool.longtracer]
alert_threshold = 0.6                # float — fire alert if trust_score < this
alert_channels = ["webhook", "slack"] # list — which channels to use
slack_webhook_url = "https://hooks.slack.com/services/T.../B.../xxx"
discord_webhook_url = "https://discord.com/api/webhooks/.../..."
alert_email_from = "longtracer@example.com"
alert_email_to = ["oncall@example.com"]
smtp_host = "smtp.gmail.com"
smtp_port = "587"                    # string in config, parsed at usage
smtp_user = "longtracer@example.com"
smtp_password = "app-password"
```

Environment variable equivalents (higher priority):
`LONGTRACER_ALERT_THRESHOLD`, `LONGTRACER_ALERT_CHANNELS`, `LONGTRACER_SLACK_WEBHOOK_URL`, `LONGTRACER_DISCORD_WEBHOOK_URL`, `LONGTRACER_ALERT_EMAIL_FROM`, `LONGTRACER_ALERT_EMAIL_TO`, `LONGTRACER_SMTP_HOST`, `LONGTRACER_SMTP_PORT`, `LONGTRACER_SMTP_USER`, `LONGTRACER_SMTP_PASSWORD`.

#### Alert Payload

```python
{
    "alert_type": "low_trust_score",
    "threshold": 0.6,
    "trust_score": 0.42,
    "verdict": "FAIL",
    "hallucination_count": 3,
    "claim_count": 8,
    "project": "my-chatbot",
    "trace_id": "uuid",
    "timestamp": "2026-05-14T10:30:00Z",
    "dashboard_url": "http://localhost:8100/dashboard/traces/uuid"  # if server running
}
```

#### OTel Span Attributes

```python
{
    "longtracer.trust_score": 0.85,
    "longtracer.claim_count": 5,
    "longtracer.hallucination_count": 1,
    "longtracer.verdict": "PASS",
    "longtracer.project_id": "my-chatbot",
    "longtracer.threshold": 0.5,
    "longtracer.duration_ms": 234.5
}
```

### API / Interface Changes

#### New API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/dashboard` | API key (cookie) | Metrics overview with charts |
| `GET` | `/dashboard/traces` | API key (cookie) | Trace list with filters |
| `GET` | `/dashboard/traces/{id}` | API key (cookie) | Trace detail view |
| `GET` | `/dashboard/login` | None | API key input form |
| `POST` | `/dashboard/login` | None | Validate API key, set cookie |
| `GET` | `/api/v1/metrics/summary` | API key (header) | Aggregated metrics stats |
| `GET` | `/api/v1/metrics/timeseries` | API key (header) | Time-bucketed metrics |

Query params for metrics endpoints:
- `project` (optional): filter by project name
- `start` (optional): ISO 8601 datetime, default 7 days ago
- `end` (optional): ISO 8601 datetime, default now
- `interval` (timeseries only): `1h`, `6h`, `1d`, `1w` — default `1d`

#### Modified: `TraceCacheBackend` (abstract interface)

```python
# OLD: no metrics methods
class TraceCacheBackend(ABC):
    def save_run(self, run): ...
    def update_run(self, run_id, updates): ...
    def save_trace(self, trace): ...
    def get_trace(self, trace_id): ...
    def list_traces(self, limit=10): ...
    def get_runs_by_trace(self, trace_id): ...
    def is_connected(self): ...

# NEW: adds two methods with default implementations
class TraceCacheBackend(ABC):
    # ... existing methods unchanged ...

    def get_metrics_summary(
        self,
        project: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Default: list_traces() → filter + aggregate in Python.
        Subclasses override with optimized queries."""
        ...

    def get_metrics_timeseries(
        self,
        project: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "1d",
    ) -> List[Dict[str, Any]]:
        """Default: list_traces() → filter + bucket + aggregate in Python.
        Subclasses override with optimized queries."""
        ...
```

Both methods have **non-abstract default implementations** in the base class. This means all existing backends (Mongo, Postgres, Redis) work immediately via the default. Only SQLite and Memory get optimized overrides.

#### Modified: `Tracer.end_root()`

```python
# OLD:
def end_root(self, outputs=None): ...

# NEW:
def end_root(self, outputs=None, metrics=None): ...
```

The `metrics` parameter is an optional dict:
```python
metrics = {
    "trust_score": 0.85,
    "hallucination_count": 1,
    "claim_count": 5,
}
```

When provided, these fields are added as top-level keys in the trace document. When `None` (default), behavior is identical to before — **fully backward compatible**.

#### Modified: `config.py` `_VALID_KEYS`

```python
# NEW keys added:
_VALID_KEYS = {
    # ... existing keys ...
    "alert_threshold": float,
    "alert_channels": list,
    "slack_webhook_url": str,
    "discord_webhook_url": str,
    "alert_email_from": str,
    "alert_email_to": list,
    "smtp_host": str,
    "smtp_port": str,
    "smtp_user": str,
    "smtp_password": str,
    "otel_enabled": bool,
}
```

#### New Public API: `longtracer.alerts`

```python
from longtracer.alerts import dispatch_alert, check_alert_threshold

def dispatch_alert(
    result: VerificationResult,
    project: str = "default",
    trace_id: Optional[str] = None,
) -> Optional[str]:
    """Dispatch alerts to all configured channels if trust_score < threshold.
    For SDK users to call manually after verify()."""

def check_alert_threshold(trust_score: float, threshold: float) -> bool:
    """Check if trust_score is below threshold."""
```

#### New Public API: `longtracer.otel`

```python
from longtracer.otel import setup_otel, verify_span

def setup_otel(
    endpoint: Optional[str] = None,
    service_name: str = "longtracer",
) -> None:
    """Initialize OTel TracerProvider with OTLP exporter.
    Reads OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_SERVICE_NAME env vars."""

@contextmanager
def verify_span(
    response: str,
    sources: list[str],
    project: str = "default",
):
    """Context manager that wraps verify() in an OTel span."""
```

#### New pyproject.toml Optional Dependency Group

```toml
[project.optional-dependencies]
# ... existing groups ...
otel = [
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "opentelemetry-exporter-otlp-proto-http>=1.20",
]
```

### Key Implementation Details

#### Dashboard Auth Flow

1. User visits `/dashboard/*` → server checks for `longtracer_api_key` cookie.
2. No cookie → redirect to `/dashboard/login`.
3. User submits API key → server validates via `secrets.compare_digest`.
4. Valid → set `longtracer_api_key` cookie (HttpOnly, SameSite=Strict) → redirect to `/dashboard`.
5. All dashboard endpoints check the cookie before rendering.
6. If no API key is configured on the server (dev mode), dashboard is accessible without login.

**Why cookie instead of localStorage**: HttpOnly cookies can't be read by JavaScript (XSS protection). The login form POSTs the key, server validates, sets cookie. HTMX requests automatically include cookies — no manual header management.

Actually, we need both approaches. HTMX sends cookies automatically, but Chart.js `fetch()` calls to `/api/v1/metrics/*` need the API key in the `x-api-key` header. Solution: the login page stores the key in localStorage AND the server sets a cookie. HTMX uses the cookie; JavaScript fetch uses localStorage.

#### Metrics Derivation Strategy

For the default base class implementation:
1. Call `self.list_traces(limit=10000)` — fetch up to 10K recent traces.
2. Filter in Python: drop traces without `trust_score`, filter by `project_name` and `created_at`.
3. Aggregate: compute avg/min/max trust_score, sum hallucination_count, count total.
4. For timeseries: bucket by interval (truncate `created_at` to hour/day/week), aggregate per bucket.

For SQLite optimized override:
```sql
SELECT
    COUNT(*) as total_traces,
    AVG(CAST(json_extract(data, '$.trust_score') AS REAL)) as avg_trust_score,
    MIN(CAST(json_extract(data, '$.trust_score') AS REAL)) as min_trust_score,
    MAX(CAST(json_extract(data, '$.trust_score') AS REAL)) as max_trust_score,
    SUM(CAST(json_extract(data, '$.hallucination_count') AS INTEGER)) as total_hallucinations,
    SUM(CAST(json_extract(data, '$.claim_count') AS INTEGER)) as total_claims,
    SUM(CASE WHEN json_extract(data, '$.trust_score') >= ? THEN 1 ELSE 0 END) as pass_count
FROM traces
WHERE json_extract(data, '$.trust_score') IS NOT NULL
  AND (? IS NULL OR project_name = ?)
  AND created_at >= ? AND created_at <= ?;
```

**Why this works**: SQLite 3.38+ (shipped with Python 3.10+) has built-in `json_extract()`. The `data` column already stores the full trace JSON.

#### Alert Dispatch Flow

```
verify() → VerificationResult
    ↓ (server only)
trust_score < alert_threshold?
    ↓ yes
alerts.dispatch_alert(result)
    ↓
for channel in alert_channels:
    ↓ background thread per channel
    ├─→ webhook: reuse webhooks.dispatch_webhook()
    ├─→ slack:   urllib.request.urlopen(slack_url, json_payload)
    ├─→ discord: urllib.request.urlopen(discord_url, json_payload)
    └─→ email:   smtplib.SMTP(smtp_host, smtp_port) → send_message()
```

**Error handling**: Each channel dispatch is wrapped in try/except. A failing channel logs a warning but never blocks other channels or the API response. All dispatch happens in daemon threads (fire-and-forget).

**Slack payload format**:
```json
{
    "text": "🚨 LongTracer Alert: Low Trust Score",
    "blocks": [
        {"type": "header", "text": {"type": "plain_text", "text": "🚨 Trust Score Alert"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": "*Score:* 0.42 / 1.0"},
            {"type": "mrkdwn", "text": "*Threshold:* 0.6"},
            {"type": "mrkdwn", "text": "*Hallucinations:* 3"},
            {"type": "mrkdwn", "text": "*Project:* my-chatbot"}
        ]}
    ]
}
```

**Discord payload format**:
```json
{
    "content": "🚨 **Trust Score Alert** — Score: 0.42 (threshold: 0.6) | Hallucinations: 3 | Project: my-chatbot"
}
```

**Email format**: Plain text email with subject `🚨 LongTracer Alert: Low Trust Score (0.42)`.

#### OpenTelemetry Integration

The OTel module is entirely optional. When `opentelemetry-api` is not installed, all functions become no-ops.

```python
# otel.py — lazy imports, graceful fallback
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
```

Configuration:
- `LONGTRACER_OTEL_ENABLED=true` or `[tool.longtracer] otel_enabled = true`
- `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` (standard OTel env var)
- `OTEL_SERVICE_NAME=longtracer` (standard OTel env var)

If `otel_enabled` is true but OTel packages are not installed → log a warning on startup, skip silently at runtime.

The `verify_span()` context manager creates a span named `longtracer.verify` and sets attributes from the VerificationResult after the block completes.

#### Dashboard Static Assets — CDN Loading

HTMX and Chart.js are loaded from CDN in the `base.html` template:
```html
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7"></script>
```

For air-gapped/offline environments, users can download these files and place them in the static directory. A `LONGTRACER_OFFLINE_MODE=true` env var switches to local asset paths.

### Dependencies

| Library | Why | Install |
|---------|-----|---------|
| `jinja2` | Template rendering for dashboard | Already a FastAPI dependency; add to `[server]` extras |
| `htmx.org` | Frontend interactivity (loaded from CDN) | No install needed |
| `chart.js` | Metrics charts (loaded from CDN) | No install needed |
| `opentelemetry-api` | OTel tracing API | `pip install 'longtracer[otel]'` |
| `opentelemetry-sdk` | OTel SDK for span processing | `pip install 'longtracer[otel]'` |
| `opentelemetry-exporter-otlp-proto-http` | OTLP HTTP exporter | `pip install 'longtracer[otel]'` |

**Zero new mandatory dependencies for the core package.** Dashboard requires `[server]` extras (already includes FastAPI which pulls Jinja2). OTel requires new `[otel]` extras.

## Assumptions

1. **Jinja2 is already available** via FastAPI's dependencies when `[server]` extras are installed. FastAPI doesn't require Jinja2, but `python-multipart` and `jinja2` are common additions — we'll add `jinja2>=3.1` to the `[server]` extras explicitly.
2. **Python 3.10+ compatibility** is maintained (no 3.12+ only features).
3. **SQLite json_extract()** is available (Python 3.10+ ships SQLite 3.38+).
4. **CDN loading** for HTMX/Chart.js is acceptable by default; offline mode is a future enhancement noted but not implemented in v0.4.0.
5. **Slack/Discord use webhook URLs** (not Bot APIs). Users create Incoming Webhooks in their workspace/channel settings.
6. **Email uses SMTP with TLS** on port 587 by default. No STARTTLS auto-detection — users must configure the correct port.
7. **Alerting fires from server endpoint only** — SDK users call `dispatch_alert()` manually.
8. **The server verify endpoint currently does NOT save traces.** Task 1 adds trace recording to the server endpoint.
9. **Dashboard is read-only** — no ability to delete traces or modify data from the UI.
10. **Grafana template is a static JSON file** — users import it manually into their Grafana instance.
11. **Mongo/Postgres/Redis backends use the default base-class metrics implementation** (Python-level aggregation). Optimized queries for these backends are deferred to a future release.
12. **The `dashboard/` directory with templates and static files** is committed directly to git — no build step.
13. **HTMX polling interval** for trace list refresh is 30 seconds (configurable via template variable).

## Tasks

> These tasks are ordered by dependency. Each task leaves the codebase in a runnable, testable state. Use the `build` skill to execute them individually or all at once.

---

### Task 1: Metrics Data Layer & Trace Recording

**Goal:** Make trace documents contain verification metrics and expose them via API endpoints. This is the foundation that the dashboard, alerting, and Grafana template all depend on.

**Scope:**
- Modify `longtracer/guard/tracer.py`: add `metrics` parameter to `end_root()`, lift metrics fields into trace document
- Modify `longtracer/guard/cache/backend.py`: add `get_metrics_summary()` and `get_metrics_timeseries()` with default implementations
- Modify `longtracer/guard/cache/sqlite.py`: override metrics methods with optimized `json_extract()` SQL
- Modify `longtracer/guard/cache/memory.py`: override metrics methods with direct iteration
- Modify `longtracer/server.py`: save trace after verification in `/api/v1/verify` endpoint, add `GET /api/v1/metrics/summary` and `GET /api/v1/metrics/timeseries` endpoints
- Add `jinja2>=3.1` to `[server]` optional dependencies in `pyproject.toml`
- Create `tests/test_metrics.py`

**Acceptance Criteria:**
- [ ] After calling `POST /api/v1/verify`, a trace is saved with `trust_score`, `hallucination_count`, `claim_count` as top-level JSON fields
- [ ] `GET /api/v1/metrics/summary?project=X&start=2026-05-01&end=2026-05-14` returns correct `{total_traces, avg_trust_score, min_trust_score, max_trust_score, pass_rate, total_hallucinations, total_claims, hallucination_rate}`
- [ ] `GET /api/v1/metrics/timeseries?range=7d&interval=1d` returns array of `{bucket, trace_count, avg_trust_score, hallucination_count, claim_count, pass_rate}` with one entry per day
- [ ] SQLite backend uses `json_extract()` SQL for metrics queries (not Python iteration)
- [ ] Existing traces without `trust_score` field are gracefully skipped in metrics aggregation (no errors, no contribution to averages)

**Error Handling:**
- Metrics methods in backends wrap all I/O in try/except, return empty results on failure (consistent with existing pattern)
- Server metrics endpoints return 200 with zeroed stats when no traces match (not 404)
- Missing/invalid query params use sensible defaults (start=7d ago, end=now, interval=1d)

**Verify:**
```bash
cd "/home/mohsin/Programming 2025/ENDEVSOLS WORk/Longtracer Forked/LongTracer"
python -m pytest tests/test_metrics.py -v
# Start server, run verification, check metrics
python -c "
from longtracer.server import create_app
app = create_app()
print('Server app created successfully')
"
```

---

### Task 2: Dashboard Foundation & Trace Browser

**Goal:** Build the web dashboard shell with layout, API key authentication, trace list, and trace detail views. Users can browse their traces visually after running `longtracer serve`.

**Scope:**
- Create `longtracer/dashboard/__init__.py`
- Create `longtracer/dashboard/templates/base.html` — layout with sidebar nav, API key check
- Create `longtracer/dashboard/templates/login.html` — API key input form
- Create `longtracer/dashboard/templates/traces.html` — trace list with HTMX pagination, project filter
- Create `longtracer/dashboard/templates/trace_detail.html` — single trace view with spans, claims, evidence map
- Create `longtracer/dashboard/templates/metrics.html` — placeholder page (populated in Task 3)
- Create `longtracer/dashboard/static/css/dashboard.css` — dark theme matching existing HTML report
- Create `longtracer/dashboard/static/js/dashboard.js` — HTMX config, Chart.js placeholder, API key management
- Modify `longtracer/server.py`: add dashboard routes, mount static files, add cookie-based auth for dashboard, add Jinja2Templates setup

**Acceptance Criteria:**
- [ ] `GET /dashboard` redirects to `/dashboard/login` if no valid API key cookie; otherwise shows metrics placeholder page
- [ ] `POST /dashboard/login` with correct API key sets `longtracer_api_key` HttpOnly cookie and redirects to `/dashboard`
- [ ] `GET /dashboard/traces` shows a paginated list of traces with columns: ID, Project, Duration, Created, Verdict — filterable by project dropdown
- [ ] `GET /dashboard/traces/{id}` shows full trace detail: metadata, span timeline, claim verification table with scores
- [ ] Dashboard renders correctly with no API key configured on server (dev mode — no auth required)

**Error Handling:**
- Invalid API key on login → re-render login page with error message
- Trace not found → render 404 template with link back to trace list
- Server not initialized (no LongTracer.init()) → dashboard shows "No data source configured" message instead of crashing

**Verify:**
```bash
cd "/home/mohsin/Programming 2025/ENDEVSOLS WORk/Longtracer Forked/LongTracer"
# Check templates render without errors
python -c "
from longtracer.server import create_app
from fastapi.testclient import TestClient
app = create_app()
client = TestClient(app)
resp = client.get('/dashboard/login')
assert resp.status_code == 200
print('Login page:', resp.status_code)
resp = client.get('/dashboard/traces')
print('Traces page:', resp.status_code)
"
```

---

### Task 3: Dashboard Metrics & Charts

**Goal:** Populate the metrics page with Chart.js visualizations showing trust score trends, hallucination rate, and per-project breakdowns. This makes the dashboard a genuine monitoring tool.

**Scope:**
- Modify `longtracer/dashboard/templates/metrics.html` — add Chart.js canvases for: trust score line chart, hallucination rate bar chart, verdict pie chart, per-project comparison table
- Modify `longtracer/dashboard/static/js/dashboard.js` — add Chart.js initialization, fetch metrics from `/api/v1/metrics/*`, render charts
- Modify `longtracer/dashboard/static/css/dashboard.css` — chart container styling, metric cards

**Acceptance Criteria:**
- [ ] `/dashboard` (metrics page) shows a trust score line chart with daily data points for the last 7 days
- [ ] A hallucination rate bar chart shows daily hallucination counts over the same period
- [ ] A verdict summary card shows pass/fail counts and pass rate percentage
- [ ] A project selector dropdown filters all charts to a specific project (HTMX re-renders charts)
- [ ] Charts load data from `/api/v1/metrics/timeseries` via fetch() with API key from localStorage

**Error Handling:**
- No trace data → show "No data yet" placeholder in each chart area
- API returns error → show toast notification "Failed to load metrics"
- Date range with zero traces → chart renders with empty axes (no crash)

**Verify:**
```bash
cd "/home/mohsin/Programming 2025/ENDEVSOLS WORk/Longtracer Forked/LongTracer"
python -c "
from longtracer.server import create_app
from fastapi.testclient import TestClient
app = create_app()
client = TestClient(app)
# Check metrics API works
resp = client.get('/api/v1/metrics/summary')
assert resp.status_code == 200
data = resp.json()
assert 'total_traces' in data
print('Metrics summary:', data)
resp = client.get('/api/v1/metrics/timeseries')
assert resp.status_code == 200
print('Timeseries points:', len(resp.json().get('data_points', [])))
# Check dashboard page loads
resp = client.get('/dashboard')
assert resp.status_code == 200
assert b'chart' in resp.text.lower()
print('Dashboard metrics page OK')
"
```

---

### Task 4: Alerting System

**Goal:** Implement multi-channel alerting that fires automatically from the server when trust score drops below a configurable threshold, with a public API for SDK users.

**Scope:**
- Create `longtracer/alerts.py` — `dispatch_alert()`, `check_alert_threshold()`, channel-specific senders (`_send_slack()`, `_send_discord()`, `_send_email()`, `_send_webhook_alert()`)
- Modify `longtracer/config.py` — add 10 new alert config keys to `_VALID_KEYS`
- Modify `longtracer/server.py` — call `dispatch_alert()` in the verify endpoint after `dispatch_verification_result()`
- Create `tests/test_alerts.py` — test threshold logic, mock HTTP calls, mock SMTP

**Acceptance Criteria:**
- [ ] When `trust_score < alert_threshold` in a server verify call, alerts are dispatched to all channels in `alert_channels` config
- [ ] Slack alert: HTTP POST to `slack_webhook_url` with JSON `{text, blocks}` — delivers successfully to a test webhook
- [ ] Discord alert: HTTP POST to `discord_webhook_url` with JSON `{content}` — delivers successfully to a test webhook
- [ ] Email alert: SMTP send with TLS, from/to/subject/body correctly formatted — testable with a mock SMTP server
- [ ] `from longtracer.alerts import dispatch_alert` works as a public API for SDK users without LongTracer init

**Error Handling:**
- Each channel dispatch is independent — one channel failure does not block others
- All channel dispatches run in background daemon threads — never block the API response
- Invalid webhook URLs → log warning, no crash
- SMTP auth failure → log warning with hint about app passwords, no crash
- Missing channel config (e.g., `slack_webhook_url` not set but `slack` in channels) → skip that channel with a warning log
- `alert_threshold` not configured → alerting is disabled silently (zero overhead)

**Verify:**
```bash
cd "/home/mohsin/Programming 2025/ENDEVSOLS WORk/Longtracer Forked/LongTracer"
python -m pytest tests/test_alerts.py -v
# Verify alert dispatch works end-to-end
python -c "
from longtracer.alerts import check_alert_threshold
assert check_alert_threshold(0.42, 0.6) == True
assert check_alert_threshold(0.85, 0.6) == False
print('Alert threshold logic OK')
"
```

---

### Task 5: OpenTelemetry Export

**Goal:** Add optional OpenTelemetry span emission for `verify()` calls, exportable via OTLP to any compatible backend (Jaeger, Grafana Tempo, Datadog). This makes LongTracer a first-class citizen in standard observability stacks.

**Scope:**
- Create `longtracer/otel.py` — `setup_otel()`, `verify_span()` context manager, lazy imports with graceful fallback
- Modify `pyproject.toml` — add `[otel]` optional dependency group
- Modify `longtracer/config.py` — add `otel_enabled: bool` to `_VALID_KEYS`
- Modify `longtracer/guard/verifier.py` — when OTel is enabled and configured, wrap the core verify logic in a `verify_span()`
- Create `tests/test_otel.py` — test with OTel SDK in-memory exporter

**Acceptance Criteria:**
- [ ] After `setup_otel(endpoint="http://localhost:4318")`, each `verify_parallel()` call creates an OTel span named `longtracer.verify` with correct attributes
- [ ] Span attributes include: `trust_score`, `claim_count`, `hallucination_count`, `verdict`, `project_id`, `threshold`, `duration_ms`
- [ ] When `opentelemetry-*` packages are not installed, `setup_otel()` logs a warning and all span creation is a no-op
- [ ] `pip install 'longtracer[otel]'` installs all three OTel packages successfully
- [ ] Configuration via env vars works: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`

**Error Handling:**
- OTel import failure → `_OTEL_AVAILABLE = False`, all functions become no-ops
- OTLP export failure (endpoint unreachable) → OTel SDK handles retries internally via BatchSpanProcessor; no impact on verify() calls
- `otel_enabled=True` but no endpoint configured → log warning on setup, spans go to ConsoleSpanExporter as fallback
- verify() exceptions → span records the exception as an OTel event before propagating

**Verify:**
```bash
cd "/home/mohsin/Programming 2025/ENDEVSOLS WORk/Longtracer Forked/LongTracer"
# Install OTel deps
pip install 'longtracer[otel]'
# Run OTel tests
python -m pytest tests/test_otel.py -v
# Verify no-op when OTel not configured
python -c "
from longtracer.guard.verifier import CitationVerifier
v = CitationVerifier(threshold=0.5)
# Should work without any OTel setup
result = v.verify_parallel('Paris is in France.', ['Paris is the capital of France.'])
print(f'Verify OK: {result.verdict}, trust={result.trust_score:.2f}')
"
```

---

### Task 6: Grafana Dashboard Template

**Goal:** Ship a ready-to-import Grafana dashboard JSON file so users can visualize LongTracer OTel data immediately in their Grafana instance.

**Scope:**
- Create `grafana/longtracer.json` — complete Grafana dashboard definition
- Create `grafana/README.md` — import instructions, required data sources, environment variables
- Modify `ROADMAP.md` — check off completed v0.4.0 items

**Acceptance Criteria:**
- [ ] `grafana/longtracer.json` is valid JSON that imports successfully into Grafana 10+
- [ ] Dashboard contains panels for: trust score over time (time series), hallucination rate over time (bar chart), claims per project (pie chart), verification latency p50/p95 (stat panel)
- [ ] `grafana/README.md` documents: how to import the JSON, which data source to configure (Tempo for traces, Prometheus for metrics), required environment variables
- [ ] Dashboard uses variables `${datasource}` and `${project}` so users can select their data source and filter by project

**Error Handling:**
- Not applicable — this is a static JSON file and documentation

**Verify:**
```bash
cd "/home/mohsin/Programming 2025/ENDEVSOLS WORk/Longtracer Forked/LongTracer"
# Validate JSON
python -c "
import json
with open('grafana/longtracer.json') as f:
    dashboard = json.load(f)
panels = dashboard.get('panels', [])
print(f'Dashboard has {len(panels)} panels')
panel_titles = [p.get('title', 'untitled') for p in panels]
print('Panels:', panel_titles)
assert len(panels) >= 4, 'Expected at least 4 panels'
print('Grafana template valid')
"
# Check README exists
test -f grafana/README.md && echo "README exists" || echo "README missing"
```

---

## Full Verification

After all 6 tasks are complete, run this end-to-end verification:

```bash
cd "/home/mohsin/Programming 2025/ENDEVSOLS WORk/Longtracer Forked/LongTracer"

# 1. Run all tests
python -m pytest tests/ -v --tb=short

# 2. Start the server
LONGTRACER_API_KEY=test-key longtracer serve --port 8100 &
SERVER_PID=$!
sleep 3

# 3. Verify API endpoints
curl -s -H "x-api-key: test-key" http://localhost:8100/api/v1/health | python -m json.tool
curl -s -H "x-api-key: test-key" http://localhost:8100/api/v1/metrics/summary | python -m json.tool
curl -s -H "x-api-key: test-key" http://localhost:8100/api/v1/metrics/timeseries | python -m json.tool

# 4. Run a verification through the API
curl -s -X POST -H "x-api-key: test-key" -H "Content-Type: application/json" \
  http://localhost:8100/api/v1/verify \
  -d '{"response": "Paris is in France.", "sources": ["Paris is the capital of France."]}' \
  | python -m json.tool

# 5. Check dashboard loads
curl -s http://localhost:8100/dashboard/login | grep -q "API" && echo "Login page OK"
curl -s -b "longtracer_api_key=test-key" http://localhost:8100/dashboard/traces | grep -q "trace" && echo "Traces page OK"
curl -s -b "longtracer_api_key=test-key" http://localhost:8100/dashboard | grep -q "chart" && echo "Metrics page OK"

# 6. Verify alerting module
python -c "
from longtracer.alerts import dispatch_alert, check_alert_threshold
assert check_alert_threshold(0.3, 0.6) == True
assert check_alert_threshold(0.9, 0.6) == False
print('Alerting module OK')
"

# 7. Verify OTel module loads (with deps installed)
pip install 'longtracer[otel]' 2>/dev/null
python -c "
from longtracer.otel import setup_otel
setup_otel(endpoint='http://localhost:4318')
print('OTel setup OK')
"

# 8. Validate Grafana template
python -c "
import json
with open('grafana/longtracer.json') as f:
    d = json.load(f)
print(f'Grafana dashboard: {len(d.get(\"panels\", []))} panels')
"

# 9. Cleanup
kill $SERVER_PID 2>/dev/null
echo "=== Full verification complete ==="
```
