# LongTracer Grafana Dashboard

Pre-built Grafana dashboard for visualizing LongTracer RAG verification metrics.

## Panels

| Panel | Type | Description |
|-------|------|-------------|
| Trust Score Over Time | Time series | Average trust score with red/yellow/green thresholds at 0.4 and 0.7 |
| Hallucination Rate Over Time | Bar chart | Hallucinated claims per time bucket, stacked by project |
| Claims Per Project | Donut chart | Distribution of total claims across projects |
| Verification Latency | Stat | P50 and P95 verification duration in milliseconds |
| Verdict Summary | Gauge | Overall pass rate percentage |

## Prerequisites

### 1. OpenTelemetry Export

LongTracer must be configured to emit OTel metrics. Install the OTel extras and enable:

```bash
pip install 'longtracer[otel]'
```

Configure in your environment:

```bash
export LONGTRACER_OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=longtracer
```

Or in `pyproject.toml`:

```toml
[tool.longtracer]
otel_enabled = true
```

### 2. OTel Collector

Run an OpenTelemetry Collector that receives OTLP data and exports to Prometheus + Tempo:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
  otlphttp:
    endpoint: "http://tempo:4318"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus]
    traces:
      receivers: [otlp]
      exporters: [otlphttp]
```

### 3. Grafana Data Sources

Add two data sources in Grafana:

- **Prometheus** — for metrics (trust score, hallucination count, latency)
  - URL: `http://prometheus:9090` (or wherever the OTel Collector exports Prometheus metrics)
- **Tempo** (optional) — for trace details
  - URL: `http://tempo:3200`

## Import Instructions

1. Open Grafana → Dashboards → Import
2. Upload the `longtracer.json` file or paste its contents
3. Select your Prometheus data source when prompted
4. Click Import

## Variables

The dashboard uses two template variables:

| Variable | Description |
|----------|-------------|
| `${datasource}` | Prometheus data source selector |
| `${project}` | Project filter (multi-select, defaults to All) |

## Expected Metrics

The dashboard expects these Prometheus metrics (emitted by LongTracer via OTel):

| Metric | Type | Labels |
|--------|------|--------|
| `longtracer_trust_score` | Gauge | `project`, `verdict` |
| `longtracer_hallucination_count` | Counter | `project` |
| `longtracer_claim_count` | Counter | `project` |
| `longtracer_verification_duration_ms` | Histogram | `project` |

## Customization

- **Time range**: Defaults to last 1 hour. Change via Grafana's time picker.
- **Refresh interval**: Set to 30 seconds. Adjust in dashboard settings.
- **Thresholds**: Trust score thresholds are at 0.4 (yellow) and 0.7 (green). Edit individual panels to customize.
- **Alerts**: Add Grafana alert rules on the Trust Score panel to notify when scores drop below your SLA.

## Troubleshooting

**No data showing:**
- Verify `LONGTRACER_OTEL_ENABLED=true`
- Check `OTEL_EXPORTER_OTLP_ENDPOINT` points to your OTel Collector
- Confirm the OTel Collector is scraping Prometheus metrics
- Check Grafana Prometheus data source is connected and healthy

**Missing projects in dropdown:**
- Ensure verifications have been run with different `project` values
- The project variable auto-discovers from `longtracer_trust_score` metric labels
