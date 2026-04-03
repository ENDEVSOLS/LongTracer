# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
