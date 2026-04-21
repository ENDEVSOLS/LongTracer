# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-04-21

### Added
- **OpenAI Assistants API adapter**: `instrument_openai_assistant(client)` — monkey-patches `create_and_poll` to automatically verify assistant responses against `file_search` citations.
- **CrewAI adapter**: `instrument_crewai(crew)` — wraps `kickoff()` to verify each task's output against its context sources. Also provides `verify_crew_output()` for standalone use.
- **AutoGen adapter**: `instrument_autogen(agent)` — wraps AutoGen ≥0.4 `AssistantAgent.on_messages` for automatic response verification. Also provides `verify_autogen_result()` for standalone use.
- **REST API server** (`longtracer serve`): FastAPI-based HTTP server exposing verification as REST endpoints.
  - `POST /api/v1/verify` — verify a single response.
  - `POST /api/v1/verify/batch` — verify multiple responses in one call.
  - `GET /api/v1/health` — health check (no auth required).
  - `GET /api/v1/traces` — list recent traces.
  - `GET /api/v1/traces/{trace_id}` — get a specific trace.
  - API key authentication via `x-api-key` header (LangSmith-standard) with `Authorization: Bearer` fallback.
  - Timing-safe key comparison via `secrets.compare_digest`.
  - CORS middleware with configurable origins (`LONGTRACER_CORS_ORIGINS`).
  - Token bucket rate limiter (60 req/min per IP, configurable via `LONGTRACER_RATE_LIMIT`).
  - Pydantic input validation with max-length and max-items constraints.
- **Webhook support**: HMAC-SHA256 signed HTTP POST dispatch with 5 retries + exponential backoff + jitter (Stripe-style).
  - Configurable via env vars (`LONGTRACER_WEBHOOK_URL`, `LONGTRACER_WEBHOOK_SECRET`) or `[tool.longtracer]` in pyproject.toml.
  - Async dispatch in background thread — never blocks the verification pipeline.
  - Dead-letter logging after max retries.
  - `dispatch_webhook()` and `dispatch_verification_result()` public APIs.
- Webhook config keys added to `[tool.longtracer]`: `webhook_url`, `webhook_secret`, `webhook_events`, `webhook_timeout`.
- Optional dependency groups: `openai`, `crewai`, `autogen`, `server`.

### Changed
- Version bumped to `0.1.6`.
- `[project.optional-dependencies]` `all` extra now includes `openai`, `crewai`, `autogen`, `server`.
- `longtracer.adapters.__init__` now lazily exports all 7 adapter modules.

## [0.1.5] - 2026-04-10

### Added
- `[tool.longtracer]` config support in `pyproject.toml` — set project-level defaults (project name, backend, threshold, verbose, log_level) without code changes.
- Config priority chain: Code args > Environment variables > pyproject.toml > Built-in defaults.
- `verify_batch()` method on `CitationVerifier` — verify multiple response+sources pairs in one call using parallel ThreadPool execution.
- `verify_batch_async()` — async wrapper for `verify_batch()`.
- `check_batch()` top-level convenience function — one-liner batch hallucination check.
- Improved HTML trace report:
  - Per-claim source diff — side-by-side view of LLM claim vs best source evidence, color-coded by status.
  - Summary stats bar — visual pass/fail/hallucination breakdown with colored segments.
  - Inline score bars on each claim row.
  - Click-to-expand claim detail with STS, entailment, and contradiction scores.

### Changed
- `LongTracer.init()` now reads defaults from `[tool.longtracer]` in the nearest `pyproject.toml`.
- `CitationVerifier` reads `threshold` from pyproject.toml config when not explicitly passed.
- `parallel_pipeline.py` now passes `best_source`, `contradiction_score`, and `score` fields to the `eval_claims` trace span for HTML diff rendering.

## [0.1.4] - 2026-04-06

### Added
- Advanced SLM fallback verifier (`Qwen2.5-1.5B-Instruct-GGUF`) to resolve numeric approximations and date contradiction ambiguities.
- Automatic gating logic to dynamically invoke SLM only for tricky numeric/date claims, keeping baseline latency under 150ms. 
- Optional `slm` extra dependencies (`llama-cpp-python` and `huggingface-hub`) in `pyproject.toml`.

### Fixed
- Fatal `TypeError` crash in `LangGraph` integration.
- Extraneous stdout logging from dependencies that was polluting pure JSON output during CLI runs.
- Claim splitter defect preventing sentence arrays < 500 chars from being split, leading to true positives masking false positives during validation.

## [0.1.3] - 2025-04-03

### Added
- LangGraph agent tracing: `instrument_langgraph(graph)` for StateGraph, `create_react_agent`, Functional API
- LangChain agent tracing: `instrument_langchain_agent(executor)` for AgentExecutor, `create_react_agent`, `create_tool_calling_agent`
- `LongTracerAgentHandler` — single callback handler for all LangGraph/LangChain agent patterns
- Accumulates sources across multi-step agent tool calls
- Captures tool calls, LLM responses, and agent actions as spans
- Verification runs once at agent completion (not after every step)
- Thread-safe state via `ContextVar` for concurrent agent invocations
- `langgraph` optional extra in `pyproject.toml`
- Full docs page for LangGraph & LangChain agent integration

## [0.1.2] - 2025-04-03

### Added
- `check()` one-liner function — verify without class instantiation
- `longtracer check` CLI command — zero-config hallucination check from terminal
- `verdict` and `summary` fields on `VerificationResult` (auto-computed)
- Jupyter notebook rich display (`_repr_html_()`) with color-coded claims table
- Input validation on all public `verify*` methods with helpful `TypeError` messages
- `cache=True` option on `CitationVerifier` for in-memory result caching
- `verify_parallel_async()` for async frameworks (FastAPI, LangChain async, etc.)
- Haystack v2 adapter: `LongTracerVerifier` component + `instrument_haystack()`
- `haystack` optional extra in `pyproject.toml`
- MkDocs documentation site with Material theme
- GitHub Pages deployment workflow (`docs.yml`)
- Issue templates: Bug Report, Feature Request, Integration Request
- Release notes categorization (`.github/release.yml`)

## [0.1.1] - 2025-04-03

### Added
- Auto-tag and GitHub Release CI workflow (`auto-tag.yml`)
- Hi-res logo in `assets/` folder

### Changed
- README header updated with centered logo and badge layout
- `pyproject.toml` keywords expanded for better PyPI discoverability
- Documentation URL updated to GitHub Pages site

## [0.1.0] - 2025-04-01

### Added
- Core SDK: `CitationVerifier` with hybrid STS + NLI claim verification
- `LongTracer.init()` one-liner enablement with singleton pattern
- Multi-project support via `LongTracer.get_tracer(project_name)`
- LangChain adapter: `instrument_langchain(chain)`
- LlamaIndex adapter: `instrument_llamaindex(query_engine)`
- Direct API: `CitationVerifier().verify_parallel(response, sources)`
- Parallel batch verification with ThreadPoolExecutor
- Context relevance scoring with bi-encoder cosine similarity
- Claim splitter with meta-statement and hallucination pattern detection
- Pluggable trace storage: Memory, SQLite, MongoDB, PostgreSQL, Redis
- Key-value cache with TTL support (MongoDB + SQLite backends)
- `longtracer` CLI command for viewing and exporting traces
- HTML trace export: self-contained single-file reports
- JSON trace export
- Console trace report with rich formatting
- Verbose logging with per-span summaries
- `py.typed` marker for PEP 561 support

### Fixed
- NLI label order corrected (contradiction=0, neutral=1, entailment=2)
- `store.py` collection_name parameter passthrough
- `context_relevance.py` duplicate chunk ID lookup
- SQLite trace backend thread safety with WAL mode

