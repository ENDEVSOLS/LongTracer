# OpenTelemetry Integration

LongTracer seamlessly integrates with [OpenTelemetry (OTel)](https://opentelemetry.io/), allowing you to export your RAG verification spans directly to enterprise monitoring backends like Jaeger, Datadog, Honeycomb, or Grafana Tempo.

## Installation

The OpenTelemetry integration is completely optional. If the packages are not installed, LongTracer gracefully falls back to a zero-overhead no-op execution.

To install the required OTel packages, use the `otel` extra:

```bash
pip install "longtracer[otel]"
```

## Setup

Enable OpenTelemetry tracing by setting the environment variable:

```bash
export LONGTRACER_OTEL_ENABLED="true"
```

You can optionally configure the standard OTel endpoint and service name:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces"
export OTEL_SERVICE_NAME="my-rag-app"
```

## Trace Attributes

Whenever LongTracer verifies a response, it emits an OTLP span named `longtracer.verify`. This span is automatically enriched with the following custom attributes:

- `longtracer.trust_score`: The computed trust score (0.0 - 1.0).
- `longtracer.hallucination_count`: The number of contradictory claims detected.
- `longtracer.claim_count`: The total number of claims evaluated.
- `longtracer.verdict`: The final string verdict (e.g., `"PASS"`, `"FAIL"`).
- `longtracer.duration_ms`: Execution latency in milliseconds.
- `longtracer.project_id`: The project tag.

If an unhandled exception occurs inside the verification pipeline, the span correctly records the exception event.

## Grafana Dashboard Template

We provide a pre-configured Grafana Dashboard JSON template to visualize these exported metrics effortlessly. 

You can find the template inside the repository at `grafana/longtracer.json`. Simply import this JSON into Grafana to instantly unlock panels for:
- Trust Score Over Time
- Hallucination Rate Over Time
- Claims Per Project
- Verification Latency
- Verdict Summary
