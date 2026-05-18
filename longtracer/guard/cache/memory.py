"""
In-memory cache backend for testing and development.

Fast, no external dependencies. Data is lost when process exits.
"""

from typing import Dict, List, Optional, Any
from collections import OrderedDict
from datetime import datetime

from .backend import TraceCacheBackend


class MemoryBackend(TraceCacheBackend):
    """
    In-memory trace storage using Python dictionaries.
    
    Use cases:
    - Unit testing (fast, no setup)
    - Development without database
    - Temporary traces that don't need persistence
    
    Note: All data is lost when the process exits.
    """
    
    def __init__(self, max_traces: int = 1000):
        """
        Initialize memory backend.
        
        Args:
            max_traces: Maximum traces to keep (LRU eviction when exceeded)
        """
        self._traces: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._max_traces = max_traces
        self._connected = True
    
    def save_run(self, run: Dict[str, Any]) -> str:
        """Save run to memory."""
        run_id = run.get("run_id")
        if not run_id:
            raise ValueError("Run must have run_id")
        
        # Add timestamp if missing
        if "created_at" not in run:
            run["created_at"] = datetime.utcnow()
        
        self._runs[run_id] = run.copy()
        return run_id
    
    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing run in memory."""
        if run_id not in self._runs:
            return False
        
        self._runs[run_id].update(updates)
        self._runs[run_id]["updated_at"] = datetime.utcnow()
        return True
    
    def save_trace(self, trace: Dict[str, Any]) -> str:
        """Save trace to memory with LRU eviction."""
        trace_id = trace.get("trace_id")
        if not trace_id:
            raise ValueError("Trace must have trace_id")
        
        # Add timestamp if missing
        if "created_at" not in trace:
            trace["created_at"] = datetime.utcnow()
        
        # LRU eviction: remove oldest if at capacity
        if len(self._traces) >= self._max_traces:
            self._traces.popitem(last=False)  # Remove oldest
        
        self._traces[trace_id] = trace.copy()
        # Move to end (most recently used)
        self._traces.move_to_end(trace_id)
        
        return trace_id
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trace from memory."""
        trace = self._traces.get(trace_id)
        if trace:
            # Move to end (mark as recently used)
            self._traces.move_to_end(trace_id)
        return trace
    
    def list_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent traces (newest first)."""
        traces = list(self._traces.values())
        # Sort by created_at descending
        traces.sort(key=lambda t: t.get("created_at", datetime.min), reverse=True)
        return traces[:limit]
    
    def get_runs_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all runs for a trace."""
        runs = [r for r in self._runs.values() if r.get("trace_id") == trace_id]
        # Sort by created_at ascending
        runs.sort(key=lambda r: r.get("created_at", datetime.min))
        return runs
    
    def is_connected(self) -> bool:
        """Memory backend is always connected."""
        return self._connected
    
    def clear(self):
        """Clear all stored data (useful for testing)."""
        self._traces.clear()
        self._runs.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        return {
            "traces": len(self._traces),
            "runs": len(self._runs),
            "max_traces": self._max_traces
        }

    # ── Optimized metrics overrides ─────────────────────────────

    def get_metrics_summary(
        self,
        project=None,
        start_time=None,
        end_time=None,
    ):
        """Optimized metrics summary via direct dict iteration."""
        traces = list(self._traces.values())
        return self._aggregate_summary(traces, project, start_time, end_time)

    def get_metrics_timeseries(
        self,
        project=None,
        start_time=None,
        end_time=None,
        interval: str = "1d",
    ):
        """Optimized metrics timeseries via direct dict iteration."""
        traces = list(self._traces.values())
        return self._aggregate_timeseries(
            traces, project, start_time, end_time, interval,
        )
