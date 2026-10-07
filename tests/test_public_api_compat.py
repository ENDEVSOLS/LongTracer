"""
Tests: Public API compatibility for v0.3.0 (handover §2.1).

Snapshots were taken from v0.2.0 / cb3407a. Any change to these names,
parameters, defaults, or return annotations is a public API break.
"""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys

import pytest

import longtracer
from longtracer import CitationVerifier, VerificationResult, check, check_batch

SECTION_2_1_SYMBOLS = [
    "LongTracer",
    "CitationGuard",
    "CitationVerifier",
    "VerificationResult",
    "check",
    "check_batch",
    "instrument_langchain",
    "instrument_langchain_agent",
    "instrument_langgraph",
    "instrument_llamaindex",
    "instrument_haystack",
    "instrument_openai_assistant",
    "instrument_crewai",
    "instrument_autogen",
]

# (parameters, return annotation) exactly as in v0.2.0
VERIFIER_SIGNATURES = {
    "verify": (
        "(self, response: str, sources: List[str], source_metadata: Optional[List[dict]] = None)",
        "VerificationResult",
    ),
    "verify_parallel": (
        "(self, response: str, sources: List[str], source_metadata: Optional[List[dict]] = None)",
        "VerificationResult",
    ),
    "verify_parallel_async": (
        "(self, response: str, sources: List[str], source_metadata: Optional[List[dict]] = None)",
        "VerificationResult",
    ),
    "verify_batch": ("(self, items: List[Dict], max_workers: int = 4)", "List[VerificationResult]"),
    "verify_batch_async": ("(self, items: List[Dict], max_workers: int = 4)", "List[VerificationResult]"),
    "verify_with_rag_result": ("(self, rag_result: dict)", "dict"),
    "cache_stats": ("(self)", "Dict[str, int]"),
}

TOP_LEVEL_SIGNATURES = {
    "check": "(response: str, sources: list[str], source_metadata: list[dict] | None = None, threshold: float = 0.5)",
    "check_batch": "(items: list[dict], threshold: float = 0.5, max_workers: int = 4)",
    "instrument_langchain": "(chain, verbose=None)",
    "instrument_langchain_agent": "(agent_executor, threshold=0.5, verbose=None)",
    "instrument_langgraph": "(graph, threshold=0.5, verbose=None)",
    "instrument_llamaindex": "(query_engine, verbose=None)",
    "instrument_haystack": "(pipeline, verbose=None)",
    "instrument_openai_assistant": "(client, threshold=0.5, verbose=None)",
    "instrument_crewai": "(crew, threshold=0.5, verbose=None)",
    "instrument_autogen": "(agent, threshold=0.5, verbose=None)",
}

LEGACY_RESULT_FIELDS = [
    "trust_score",
    "claims",
    "flagged_claims",
    "hallucinations",
    "all_supported",
    "hallucination_count",
    "verdict",
    "summary",
    "latency_stats",
]


def _params(fn) -> str:
    sig = inspect.signature(fn)
    return str(sig.replace(return_annotation=inspect.Signature.empty))


def _short(annotation) -> str:
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return text.replace("longtracer.guard.verifier.", "").replace("typing.", "")


@pytest.mark.parametrize("name", SECTION_2_1_SYMBOLS)
def test_section_2_1_symbol_exported(name):
    assert name in longtracer.__all__
    assert hasattr(longtracer, name)


def test_citation_guard_alias_preserved():
    assert longtracer.CitationGuard is longtracer.LongTracer


@pytest.mark.parametrize("method, expected", VERIFIER_SIGNATURES.items())
def test_verifier_method_signature_unchanged(method, expected):
    params, ret = expected
    fn = getattr(CitationVerifier, method)
    assert _params(fn) == params
    assert _short(inspect.signature(fn).return_annotation) == ret


def test_async_methods_are_still_coroutines():
    assert inspect.iscoroutinefunction(CitationVerifier.verify_parallel_async)
    assert inspect.iscoroutinefunction(CitationVerifier.verify_batch_async)


def test_verifier_constructor_unchanged():
    params = inspect.signature(CitationVerifier.__init__).parameters
    assert list(params) == ["self", "threshold", "tracer", "cache"]
    assert params["threshold"].default is CitationVerifier._SENTINEL
    assert params["cache"].default is False


@pytest.mark.parametrize("name, expected", TOP_LEVEL_SIGNATURES.items())
def test_top_level_function_signature_unchanged(name, expected):
    assert _params(getattr(longtracer, name)) == expected


def test_check_and_check_batch_return_legacy_type():
    assert _short(inspect.signature(check).return_annotation) == "VerificationResult"
    assert _short(inspect.signature(check_batch).return_annotation) == "list[VerificationResult]"


def test_verification_result_fields_unchanged():
    assert [f.name for f in dataclasses.fields(VerificationResult)] == LEGACY_RESULT_FIELDS


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "longtracer.cli", *args], capture_output=True, text=True)


def test_cli_top_level_commands():
    res = _cli("--help")
    assert res.returncode == 0, res.stderr
    for cmd in ("view", "check", "serve", "doctor", "models"):
        assert cmd in res.stdout


@pytest.mark.parametrize(
    "args, expected_flags",
    [
        (("view", "--help"), ["--id", "--last", "--export", "--html", "--output", "--project", "--limit"]),
        (("check", "--help"), ["response", "sources", "--json", "--threshold"]),
        (("serve", "--help"), ["--host", "--port", "--workers", "--reload"]),
        (("doctor", "--help"), []),
        (("models", "prepare", "--help"), []),
    ],
)
def test_cli_subcommand_shape(args, expected_flags):
    res = _cli(*args)
    assert res.returncode == 0, res.stderr
    for flag in expected_flags:
        assert flag in res.stdout, f"{' '.join(args)} lost {flag}"


@pytest.mark.parametrize(
    "fn",
    [
        "cmd_view",
        "cmd_list",
        "cmd_last",
        "cmd_check",
        "cmd_serve",
        "cmd_doctor",
        "cmd_models_prepare",
        "cmd_export_json",
        "cmd_export_html",
    ],
)
def test_cli_command_handlers_exist(fn):
    """`list`, `last`, `export-json` and `export-html` are view modes backed by these handlers."""
    import longtracer.cli as cli

    assert callable(getattr(cli, fn))


def test_new_api_is_additive_only():
    """v0.3.0 additions exist alongside the legacy surface."""
    assert inspect.iscoroutinefunction(CitationVerifier.verify_case_async)
    assert callable(CitationVerifier.verify_case)
    assert "check_case" in longtracer.__all__
