import os
import sys
from datetime import datetime, timezone

# Add to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from longtracer.guard.cache import get_default_backend
from longtracer.guard.tracer import Tracer
from longtracer.alerts import dispatch_alert

class MockResult:
    def __init__(self, trust_score, verdict, hallucination_count, claims):
        self.trust_score = trust_score
        self.verdict = verdict
        self.hallucination_count = hallucination_count
        self.claims = claims
        self._trace_id = "test-trace-123"

def main():
    print("--- 1. Testing Metrics ---")
    backend = get_default_backend()
    
    # Insert some dummy metrics data via Tracer
    tracer = Tracer(project_name="test-project", run_name="test-run", backend=backend)
    tracer.start_root(inputs={"response": "test response", "source_count": 1})
    tracer.end_root(
        outputs={"verdict": "FAIL", "trust_score": 0.4, "summary": "Failed validation"},
        metrics={"trust_score": 0.4, "hallucination_count": 2, "claim_count": 3}
    )
    print(f"Inserted trace: {tracer.root_run.get('trace_id')}")

    # Query metrics
    summary = backend.get_metrics_summary(project="test-project")
    print(f"Metrics Summary: {summary}")

    timeseries = backend.get_metrics_timeseries(project="test-project", interval="1d")
    print(f"Timeseries: {timeseries}")

    print("\n--- 2. Testing Alerts ---")
    os.environ["LONGTRACER_ALERT_THRESHOLD"] = "0.5"
    os.environ["LONGTRACER_ALERT_CHANNELS"] = "webhook"
    
    mock_result = MockResult(0.4, "FAIL", 2, [{"claim": "test", "supported": False}])
    dispatch_id = dispatch_alert(mock_result, project="test-project")
    print(f"Dispatched alert, ID: {dispatch_id}")

    print("\n--- 3. Testing Dashboard Traces List ---")
    traces = tracer.list_recent_traces(limit=5)
    print(f"Found {len(traces)} traces")
    for t in traces:
        print(f" - {t.get('trace_id')} | Score: {t.get('trust_score')}")

if __name__ == "__main__":
    main()
