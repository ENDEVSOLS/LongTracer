"""
Abstract interface for trace cache backends.

All cache backends must implement TraceCacheBackend.
This allows swapping databases (Mongo, SQLite, Memory) without changing application code.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any


class TraceCacheBackend(ABC):
    """
    Abstract base class for trace storage backends.
    
    All concrete backends (MongoDB, SQLite, Memory) must implement these methods.
    This enables database-agnostic trace persistence.
    """
    
    @abstractmethod
    def save_run(self, run: Dict[str, Any]) -> str:
        """
        Save a run document to the cache.
        
        Args:
            run: Run document with run_id, trace_id, name, inputs, outputs, etc.
            
        Returns:
            The run_id of the saved run.
        """
        pass
    
    @abstractmethod
    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing run with new data.
        
        Args:
            run_id: ID of the run to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update succeeded, False otherwise.
        """
        pass
    
    @abstractmethod
    def save_trace(self, trace: Dict[str, Any]) -> str:
        """
        Save an aggregated trace document.
        
        Args:
            trace: Trace document with trace_id, inputs, outputs, claim_evidence_map, etc.
            
        Returns:
            The trace_id of the saved trace.
        """
        pass
    
    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a trace by its ID.
        
        Args:
            trace_id: ID of the trace to retrieve
            
        Returns:
            Trace document or None if not found.
        """
        pass
    
    @abstractmethod
    def list_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent traces, ordered by creation time (newest first).
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of trace documents.
        """
        pass
    
    @abstractmethod
    def get_runs_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Get all runs belonging to a specific trace.
        
        Args:
            trace_id: ID of the trace
            
        Returns:
            List of run documents ordered by creation time.
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the backend is connected and operational.
        
        Returns:
            True if connected, False otherwise.
        """
        pass

    # ── Metrics methods (with default implementations) ──────────

    def get_metrics_summary(
        self,
        project: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Return aggregated metrics from trace data.

        Default implementation iterates all traces and aggregates
        in Python. Subclasses should override with optimized queries
        (e.g., SQL json_extract for SQLite, aggregation pipeline
        for MongoDB).

        Args:
            project: Filter by project name (None = all projects).
            start_time: Only include traces created after this time.
            end_time: Only include traces created before this time.

        Returns:
            Dict with: total_traces, avg_trust_score, min_trust_score,
            max_trust_score, pass_rate, total_hallucinations,
            total_claims, hallucination_rate.
        """
        try:
            traces = self.list_traces(limit=10000)
        except Exception:
            return self._empty_summary(project, start_time, end_time)

        return self._aggregate_summary(
            traces, project, start_time, end_time,
        )

    def get_metrics_timeseries(
        self,
        project: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """Return time-bucketed metrics from trace data.

        Default implementation iterates all traces and aggregates
        in Python. Subclasses should override with optimized queries.

        Args:
            project: Filter by project name (None = all projects).
            start_time: Only include traces after this time.
            end_time: Only include traces before this time.
            interval: Bucket size — "1h", "6h", "1d", "1w".

        Returns:
            Dict with: project, interval, data_points (list of dicts).
        """
        try:
            traces = self.list_traces(limit=10000)
        except Exception:
            return {"project": project, "interval": interval, "data_points": []}

        return self._aggregate_timeseries(
            traces, project, start_time, end_time, interval,
        )

    # ── Shared aggregation helpers ──────────────────────────────

    @staticmethod
    def _empty_summary(
        project: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Dict[str, Any]:
        """Return a zeroed-out summary when no data is available."""
        return {
            "project": project,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "total_traces": 0,
            "avg_trust_score": None,
            "min_trust_score": None,
            "max_trust_score": None,
            "pass_rate": None,
            "total_hallucinations": 0,
            "total_claims": 0,
            "hallucination_rate": None,
        }

    @staticmethod
    def _parse_dt(val: Any) -> Optional[datetime]:
        """Parse a datetime from various formats."""
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None
        return None

    def _aggregate_summary(
        self,
        traces: List[Dict[str, Any]],
        project: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Dict[str, Any]:
        """Aggregate summary metrics from a list of trace dicts."""
        scores = []
        total_hallucinations = 0
        total_claims = 0
        pass_count = 0

        for t in traces:
            ts = t.get("trust_score")
            if ts is None:
                continue  # Skip traces without metrics

            # Filter by project
            if project and t.get("project_name") != project:
                continue

            # Filter by time range
            created = self._parse_dt(t.get("created_at"))
            if start_time and created and created < start_time:
                continue
            if end_time and created and created > end_time:
                continue

            scores.append(float(ts))
            total_hallucinations += int(t.get("hallucination_count", 0) or 0)
            total_claims += int(t.get("claim_count", 0) or 0)
            if float(ts) >= 0.5:
                pass_count += 1

        if not scores:
            return self._empty_summary(project, start_time, end_time)

        return {
            "project": project,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "total_traces": len(scores),
            "avg_trust_score": round(sum(scores) / len(scores), 4),
            "min_trust_score": round(min(scores), 4),
            "max_trust_score": round(max(scores), 4),
            "pass_rate": round(pass_count / len(scores), 4),
            "total_hallucinations": total_hallucinations,
            "total_claims": total_claims,
            "hallucination_rate": round(
                total_hallucinations / total_claims, 4
            ) if total_claims > 0 else None,
        }

    def _aggregate_timeseries(
        self,
        traces: List[Dict[str, Any]],
        project: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """Aggregate time-bucketed metrics from a list of trace dicts."""
        buckets: Dict[str, List[float]] = {}
        bucket_hall: Dict[str, int] = {}
        bucket_claims: Dict[str, int] = {}
        bucket_pass: Dict[str, int] = {}

        for t in traces:
            ts = t.get("trust_score")
            if ts is None:
                continue

            if project and t.get("project_name") != project:
                continue

            created = self._parse_dt(t.get("created_at"))
            if not created:
                continue
            if start_time and created < start_time:
                continue
            if end_time and created > end_time:
                continue

            bucket_key = self._bucket_key(created, interval)
            buckets.setdefault(bucket_key, []).append(float(ts))
            bucket_hall[bucket_key] = bucket_hall.get(bucket_key, 0) + int(
                t.get("hallucination_count", 0) or 0
            )
            bucket_claims[bucket_key] = bucket_claims.get(bucket_key, 0) + int(
                t.get("claim_count", 0) or 0
            )
            if float(ts) >= 0.5:
                bucket_pass[bucket_key] = bucket_pass.get(bucket_key, 0) + 1

        data_points = []
        for key in sorted(buckets.keys()):
            vals = buckets[key]
            data_points.append({
                "bucket": key,
                "trace_count": len(vals),
                "avg_trust_score": round(sum(vals) / len(vals), 4),
                "hallucination_count": bucket_hall.get(key, 0),
                "claim_count": bucket_claims.get(key, 0),
                "pass_rate": round(
                    bucket_pass.get(key, 0) / len(vals), 4
                ),
            })

        return {
            "project": project,
            "interval": interval,
            "data_points": data_points,
        }

    @staticmethod
    def _bucket_key(dt: datetime, interval: str) -> str:
        """Generate a bucket key for the given datetime and interval."""
        if interval == "1h":
            return dt.strftime("%Y-%m-%dT%H:00")
        elif interval == "6h":
            block = dt.hour // 6
            return f"{dt.strftime('%Y-%m-%d')}T{block * 6:02d}:00"
        elif interval == "1w":
            # ISO week number
            return f"{dt.strftime('%G')}-W{dt.strftime('%V')}"
        else:  # default: 1d
            return dt.strftime("%Y-%m-%d")
