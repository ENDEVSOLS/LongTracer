# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes     |

Older versions are not actively maintained. Please upgrade to the latest release.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues privately via one of the following:

- **GitHub Private Vulnerability Reporting** (preferred): [Security Advisories](https://github.com/ENDEVSOLS/LongTracer/security/advisories/new)
- **Email**: technology@endevsols.com

### What to include

- A clear description of the vulnerability
- Steps to reproduce or a proof-of-concept
- Affected version(s)
- Potential impact assessment

### Response timeline

| Stage | Timeframe |
|-------|-----------|
| Acknowledgement | Within 48 hours |
| Initial assessment | Within 5 business days |
| Fix or mitigation | Within 30 days (critical issues prioritized) |
| Public disclosure | After fix is released |

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). You will be credited in the release notes unless you prefer to remain anonymous.

## Scope

Issues in scope:

- Arbitrary code execution via crafted inputs to `CitationVerifier` or `LongTracer`
- Path traversal or unsafe file writes in trace export functions
- Credential or secret leakage through logs or trace output
- Dependency vulnerabilities with direct exploitability
- **REST API server** (`longtracer serve`): authentication bypass, rate limit evasion, injection via request payloads
- **Webhook system**: HMAC signature bypass, secret leakage, SSRF via webhook URL configuration

Out of scope:

- Vulnerabilities in optional third-party backends (MongoDB, Redis, PostgreSQL) — report those upstream
- Issues requiring physical access to the machine
- Social engineering

## Security Best Practices for Users

- **Secrets**: Never pass API keys or credentials as part of `response` or `sources` strings — they may appear in trace output
- **Trace storage**: Restrict access to `~/.longtracer/traces.db` and any configured database backends
- **HTML reports**: `export_trace_html()` output contains raw LLM response text — treat it as untrusted content before serving in a browser
- **Dependencies**: Pin dependency versions in production and audit with `pip audit` or `safety`
- **REST API server** (`longtracer serve`):
  - Always set `LONGTRACER_API_KEY` in production — without it, the server runs in open/dev mode
  - Use HTTPS (terminate TLS at a reverse proxy) — the server itself does not handle TLS
  - Configure `LONGTRACER_CORS_ORIGINS` to restrict cross-origin access
  - Monitor rate limit logs for abuse patterns
- **Webhooks**:
  - Always set `LONGTRACER_WEBHOOK_SECRET` — without it, payloads are unsigned
  - Verify `X-LongTracer-Signature` on your receiver endpoint using HMAC-SHA256
  - Use HTTPS webhook URLs — HTTP is allowed in dev but discouraged in production

## Dependency Security

LongTracer uses a minimal core dependency set. To audit your installation:

```bash
pip install pip-audit
pip-audit
```
