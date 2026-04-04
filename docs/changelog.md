# Changelog

See the full [CHANGELOG.md](https://github.com/ENDEVSOLS/LongTracer/blob/main/CHANGELOG.md) on GitHub.

## Latest: v0.1.1

- Updated logo and visual assets
- Added auto-tag and GitHub Release CI workflow
- Improved LlamaIndex adapter and trace report HTML

## v0.1.0 — Initial Release

- Core SDK: `CitationVerifier` with hybrid STS + NLI claim verification
- `LongTracer.init()` one-liner enablement with singleton pattern
- Multi-project support via `LongTracer.get_tracer(project_name)`
- LangChain and LlamaIndex adapters
- Pluggable trace storage: Memory, SQLite, MongoDB, PostgreSQL, Redis
- `longtracer` CLI command
- HTML and JSON trace export
