# Trace Storage Backends

LongTracer supports pluggable trace storage. Choose the backend that fits your deployment.

## Quick Reference

| Backend | Install | Best for |
|---------|---------|----------|
| SQLite | built-in (default) | Local dev, single-process |
| Memory | built-in | Testing, ephemeral |
| MongoDB | `pip install "longtracer[mongo]"` | Production, distributed |
| PostgreSQL | `pip install "longtracer[postgres]"` | Production, SQL queries |
| Redis | `pip install "longtracer[redis]"` | High-throughput, TTL |

---

## SQLite (Default)

No configuration needed. Traces persist to `~/.longtracer/traces.db`.

```python
LongTracer.init(backend="sqlite")

# Custom path
LongTracer.init(backend="sqlite", path="./my_traces.db")
```

---

## Memory

In-memory only — traces are lost on restart. Useful for testing.

```python
LongTracer.init(backend="memory")
```

---

## MongoDB

```bash
pip install "longtracer[mongo]"
```

```python
LongTracer.init(backend="mongo", uri="mongodb://localhost:27017")

# Or via environment variable
# MONGODB_URI=mongodb://localhost:27017
LongTracer.init(backend="mongo")
```

---

## PostgreSQL

```bash
pip install "longtracer[postgres]"
```

```python
LongTracer.init(
    backend="postgres",
    host="localhost",
    port=5432,
    database="longtracer",
    user="postgres",
    password="secret"
)

# Or via environment variables
# POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
```

---

## Redis

```bash
pip install "longtracer[redis]"
```

```python
LongTracer.init(backend="redis", host="localhost", port=6379)

# Or via environment variable
# REDIS_HOST=localhost
LongTracer.init(backend="redis")
```

---

## Environment Variables

All backends can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACE_CACHE_BACKEND` | `sqlite` | Backend type |
| `MONGODB_URI` | — | MongoDB connection URI |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `longtracer` | PostgreSQL database |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `TRACE_PROJECT` | `longtracer` | Default project name |
