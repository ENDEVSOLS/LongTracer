"""
SQLite cache backend for local/offline trace storage.

File-based, no server required. Good for local development and single-machine deployments.
"""

import os
import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime

from .backend import TraceCacheBackend


class SQLiteBackend(TraceCacheBackend):
    """
    SQLite-based trace storage backend.
    
    Use cases:
    - Local development without database server
    - Offline/air-gapped environments
    - Single-user or embedded applications
    - Lightweight persistence
    
    No external dependencies required (sqlite3 is in Python stdlib).
    """
    
    def __init__(self, path: Optional[str] = None):
        """
        Initialize SQLite backend.
        
        Args:
            path: Path to SQLite database file.
                  Default: SQLITE_TRACE_PATH env var, or ~/.longtracer/traces.db
        """
        self._path = path or os.environ.get(
            "SQLITE_TRACE_PATH",
            os.path.join(os.path.expanduser("~"), ".longtracer", "traces.db"),
        )
        self._connected = False
        self._conn = None
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._connect()
    
    def _connect(self):
        """Create database connection and tables."""
        try:
            # check_same_thread=False is safe here because all writes go through
            # a single shared connection protected by the GIL + commit() calls.
            # For high-concurrency production use, prefer the SQLite KV backend
            # (kv_sqlite.py) which uses thread-local connections + WAL mode.
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent read performance
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._create_tables()
            self._connected = True
            print(f"  ✅ SQLite Backend Active [Path: {self._path}]")
        except Exception as e:
            print(f"  ⚠️  SQLite connection failed: {e}")
            self._connected = False
    
    def _create_tables(self):
        """Create traces and runs tables if they don't exist."""
        cursor = self._conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                project_name TEXT,
                run_name TEXT,
                data TEXT,
                created_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                trace_id TEXT,
                parent_id TEXT,
                name TEXT,
                data TEXT,
                created_at TEXT,
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_trace ON runs(trace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_id)")
        
        self._conn.commit()
    
    def save_run(self, run: Dict[str, Any]) -> str:
        """Save run to SQLite."""
        if not self._connected:
            return run.get("run_id", "")
        
        try:
            run_id = run.get("run_id", "")
            trace_id = run.get("trace_id", "")
            parent_id = run.get("parent_id")
            name = run.get("name", "")
            created_at = run.get("created_at", datetime.utcnow())
            
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            
            cursor = self._conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO runs (run_id, trace_id, parent_id, name, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, trace_id, parent_id, name, json.dumps(run, default=str), created_at)
            )
            self._conn.commit()
            return run_id
        except Exception as e:
            print(f"  ⚠️  Failed to save run to SQLite: {e}")
            return run.get("run_id", "")
    
    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update run in SQLite."""
        if not self._connected:
            return False
        
        try:
            cursor = self._conn.cursor()
            # Fetch existing data
            cursor.execute("SELECT data FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            # Merge updates
            data = json.loads(row["data"])
            data.update(updates)
            
            cursor.execute(
                "UPDATE runs SET data = ? WHERE run_id = ?",
                (json.dumps(data, default=str), run_id)
            )
            self._conn.commit()
            return True
        except Exception as e:
            print(f"  ⚠️  Failed to update run in SQLite: {e}")
            return False
    
    def save_trace(self, trace: Dict[str, Any]) -> str:
        """Save trace to SQLite."""
        if not self._connected:
            return trace.get("trace_id", "")
        
        try:
            trace_id = trace.get("trace_id", "")
            project_name = trace.get("project_name", "")
            run_name = trace.get("run_name", "")
            created_at = trace.get("created_at", datetime.utcnow())
            
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            
            cursor = self._conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO traces (trace_id, project_name, run_name, data, created_at) VALUES (?, ?, ?, ?, ?)",
                (trace_id, project_name, run_name, json.dumps(trace, default=str), created_at)
            )
            self._conn.commit()
            return trace_id
        except Exception as e:
            print(f"  ⚠️  Failed to save trace to SQLite: {e}")
            return trace.get("trace_id", "")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trace from SQLite."""
        if not self._connected:
            return None
        
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT data FROM traces WHERE trace_id = ?", (trace_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["data"])
            return None
        except Exception as e:
            print(f"  ⚠️  Failed to get trace from SQLite: {e}")
            return None
    
    def list_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent traces from SQLite."""
        if not self._connected:
            return []
        
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT data FROM traces ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [json.loads(row["data"]) for row in cursor.fetchall()]
        except Exception as e:
            print(f"  ⚠️  Failed to list traces from SQLite: {e}")
            return []
    
    def get_runs_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all runs for a trace from SQLite."""
        if not self._connected:
            return []
        
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT data FROM runs WHERE trace_id = ? ORDER BY created_at ASC",
                (trace_id,)
            )
            return [json.loads(row["data"]) for row in cursor.fetchall()]
        except Exception as e:
            print(f"  ⚠️  Failed to get runs from SQLite: {e}")
            return []
    
    def is_connected(self) -> bool:
        """Check SQLite connection status."""
        return self._connected

    def close(self):
        """Close SQLite connection."""
        if self._conn:
            self._conn.close()
            self._connected = False

    # ── Optimized metrics overrides ─────────────────────────────

    def get_metrics_summary(
        self,
        project: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict:
        """Optimized metrics summary using SQLite json_extract."""
        if not self._connected:
            return self._empty_summary(project, start_time, end_time)

        try:
            cursor = self._conn.cursor()
            conditions = [
                "json_extract(data, '$.trust_score') IS NOT NULL",
            ]
            params: list = []

            if project:
                conditions.append("project_name = ?")
                params.append(project)

            if start_time:
                conditions.append("created_at >= ?")
                params.append(start_time.isoformat())

            if end_time:
                conditions.append("created_at <= ?")
                params.append(end_time.isoformat())

            where = " AND ".join(conditions)
            query = f"""
                SELECT
                    COUNT(*) as total_traces,
                    AVG(CAST(json_extract(data, '$.trust_score') AS REAL)) as avg_trust_score,
                    MIN(CAST(json_extract(data, '$.trust_score') AS REAL)) as min_trust_score,
                    MAX(CAST(json_extract(data, '$.trust_score') AS REAL)) as max_trust_score,
                    SUM(CAST(json_extract(data, '$.hallucination_count') AS INTEGER)) as total_hallucinations,
                    SUM(CAST(json_extract(data, '$.claim_count') AS INTEGER)) as total_claims,
                    SUM(CASE WHEN CAST(json_extract(data, '$.trust_score') AS REAL) >= 0.5 THEN 1 ELSE 0 END) as pass_count
                FROM traces
                WHERE {where}
            """
            cursor.execute(query, params)
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return self._empty_summary(project, start_time, end_time)

            total_traces = row[0]
            total_hall = int(row[4] or 0)
            total_claims = int(row[5] or 0)
            pass_count = int(row[6] or 0)

            return {
                "project": project,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "total_traces": total_traces,
                "avg_trust_score": round(float(row[1]), 4) if row[1] is not None else None,
                "min_trust_score": round(float(row[2]), 4) if row[2] is not None else None,
                "max_trust_score": round(float(row[3]), 4) if row[3] is not None else None,
                "pass_rate": round(pass_count / total_traces, 4),
                "total_hallucinations": total_hall,
                "total_claims": total_claims,
                "hallucination_rate": round(
                    total_hall / total_claims, 4
                ) if total_claims > 0 else None,
            }
        except Exception as e:
            import logging
            logging.getLogger("longtracer").warning(
                "SQLite metrics summary failed: %s", e
            )
            return self._empty_summary(project, start_time, end_time)

    def get_metrics_timeseries(
        self,
        project: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "1d",
    ) -> dict:
        """Optimized metrics timeseries using SQLite json_extract."""
        if not self._connected:
            return {"project": project, "interval": interval, "data_points": []}

        try:
            # Determine bucket format based on interval
            if interval == "1h":
                bucket_expr = "strftime('%Y-%m-%dT%H:00', created_at)"
            elif interval == "6h":
                bucket_expr = "strftime('%Y-%m-%dT', created_at) || ((CAST(strftime('%H', created_at) AS INTEGER) / 6) * 6) || ':00'"
            elif interval == "1w":
                bucket_expr = "strftime('%Y-W%W', created_at)"
            else:  # 1d
                bucket_expr = "strftime('%Y-%m-%d', created_at)"

            conditions = [
                "json_extract(data, '$.trust_score') IS NOT NULL",
            ]
            params: list = []

            if project:
                conditions.append("project_name = ?")
                params.append(project)

            if start_time:
                conditions.append("created_at >= ?")
                params.append(start_time.isoformat())

            if end_time:
                conditions.append("created_at <= ?")
                params.append(end_time.isoformat())

            where = " AND ".join(conditions)
            query = f"""
                SELECT
                    {bucket_expr} as bucket,
                    COUNT(*) as trace_count,
                    AVG(CAST(json_extract(data, '$.trust_score') AS REAL)) as avg_trust_score,
                    SUM(CAST(json_extract(data, '$.hallucination_count') AS INTEGER)) as hallucination_count,
                    SUM(CAST(json_extract(data, '$.claim_count') AS INTEGER)) as claim_count,
                    SUM(CASE WHEN CAST(json_extract(data, '$.trust_score') AS REAL) >= 0.5 THEN 1 ELSE 0 END) as pass_count
                FROM traces
                WHERE {where}
                GROUP BY bucket
                ORDER BY bucket
            """
            cursor = self._conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            data_points = []
            for row in rows:
                tc = int(row[1])
                data_points.append({
                    "bucket": row[0],
                    "trace_count": tc,
                    "avg_trust_score": round(float(row[2]), 4) if row[2] is not None else None,
                    "hallucination_count": int(row[3] or 0),
                    "claim_count": int(row[4] or 0),
                    "pass_rate": round(int(row[5] or 0) / tc, 4) if tc > 0 else None,
                })

            return {
                "project": project,
                "interval": interval,
                "data_points": data_points,
            }
        except Exception as e:
            import logging
            logging.getLogger("longtracer").warning(
                "SQLite metrics timeseries failed: %s", e
            )
            return {"project": project, "interval": interval, "data_points": []}
