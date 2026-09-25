"""
Tests for the new `longtracer doctor` and `longtracer models prepare` CLI
commands added in feat/diagnostics-cli (Roadmap Queue 6).

These tests cover:
  - CLI entry-point parsing  (argparse wiring)
  - doctor output structure  (section headers, check lines)
  - doctor exit codes        (0 on healthy, 1 on errors)
  - models prepare parsing   (subcommand wiring)
  - serve port default fix   (must be 8000, not 8100)
  - backward compatibility   (check / view / serve still present)

Model weights are NOT downloaded here — tests use monkeypatching /
subprocess isolation so the CI suite remains fast and offline-safe.
"""

import sys
import types
import importlib
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_cli():
    """Force a fresh import of cli so patched sys.modules take effect."""
    import longtracer.cli as cli_mod
    importlib.reload(cli_mod)
    return cli_mod


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

class TestCliParsing:
    """Verify all subcommands are reachable via argparse."""

    def test_doctor_subcommand_registered(self):
        from longtracer.cli import main
        import argparse
        # Reach into the parser by checking that 'doctor' is a valid command
        # We do this by patching cmd_doctor to avoid real execution.
        with patch("longtracer.cli.cmd_doctor") as mock_doctor:
            sys.argv = ["longtracer", "doctor"]
            try:
                main()
            except SystemExit:
                pass
            mock_doctor.assert_called_once()

    def test_models_prepare_subcommand_registered(self):
        from longtracer.cli import main
        with patch("longtracer.cli.cmd_models_prepare") as mock_prep:
            sys.argv = ["longtracer", "models", "prepare"]
            try:
                main()
            except SystemExit:
                pass
            mock_prep.assert_called_once()

    def test_check_subcommand_still_present(self):
        from longtracer.cli import main
        with patch("longtracer.cli.cmd_check") as mock_check:
            sys.argv = ["longtracer", "check", "hello", "world"]
            try:
                main()
            except SystemExit:
                pass
            mock_check.assert_called_once()

    def test_serve_default_port_is_8000(self):
        """Regression: serve used to default to 8100 instead of 8000."""
        from longtracer.cli import main
        captured_args = {}

        def fake_serve(args):
            captured_args["port"] = args.port

        with patch("longtracer.cli.cmd_serve", side_effect=fake_serve):
            sys.argv = ["longtracer", "serve"]
            try:
                main()
            except SystemExit:
                pass
        assert captured_args.get("port") == 8000, (
            f"serve default port should be 8000, got {captured_args.get('port')}"
        )

    def test_serve_port_override(self):
        from longtracer.cli import main
        captured_args = {}

        def fake_serve(args):
            captured_args["port"] = args.port

        with patch("longtracer.cli.cmd_serve", side_effect=fake_serve):
            sys.argv = ["longtracer", "serve", "--port", "9999"]
            try:
                main()
            except SystemExit:
                pass
        assert captured_args.get("port") == 9999


# ---------------------------------------------------------------------------
# longtracer doctor
# ---------------------------------------------------------------------------

class TestDoctorOutput:
    """Verify doctor prints expected section headers and check icons."""

    def _run_doctor(self, capsys, extra_patches=None):
        """Run cmd_doctor with mocked heavy dependencies."""
        patches = {
            # Mock backend so we don't touch SQLite
            "longtracer.guard.cache.get_default_backend": MagicMock(
                return_value=MagicMock(__class__=type("SQLiteBackend", (), {}))
            ),
        }
        if extra_patches:
            patches.update(extra_patches)

        from longtracer.cli import cmd_doctor
        import argparse
        args = argparse.Namespace()

        with patch("longtracer.cli._is_model_cached", return_value=True), \
             patch("longtracer.guard.cache.get_default_backend",
                   return_value=MagicMock()):
            try:
                cmd_doctor(args)
            except SystemExit:
                pass

        return capsys.readouterr()

    def test_section_headers_present(self, capsys):
        out = self._run_doctor(capsys)
        for header in ("[ Runtime ]", "[ Core dependencies ]", "[ Model cache ]",
                       "[ Optional extras ]", "[ Storage ]", "[ Configuration ]",
                       "[ Backend ]"):
            assert header in out.out, f"Missing section header: {header}"

    def test_summary_line_present(self, capsys):
        out = self._run_doctor(capsys)
        # Should end with a summary line containing ✓, ⚠, or ✗
        # At minimum the equals separator must appear twice
        assert out.out.count("=" * 62) >= 2

    def test_python_version_check_present(self, capsys):
        out = self._run_doctor(capsys)
        assert "Python version" in out.out

    def test_model_cache_lines_present(self, capsys):
        out = self._run_doctor(capsys)
        assert "all-MiniLM-L6-v2" in out.out
        assert "nli-deberta-v3-xsmall" in out.out


class TestDoctorExitCodes:
    """doctor must exit 1 when there are errors, 0 when healthy."""

    def test_exits_0_when_healthy(self):
        from longtracer.cli import cmd_doctor
        import argparse

        with patch("longtracer.cli._is_model_cached", return_value=True), \
             patch("longtracer.guard.cache.get_default_backend",
                   return_value=MagicMock()), \
             patch("importlib.import_module", side_effect=lambda m: types.ModuleType(m)):
            args = argparse.Namespace()
            # If doctor finds no errors it should NOT raise SystemExit(1)
            # (it may or may not call sys.exit(0) — both are acceptable)
            try:
                cmd_doctor(args)
                exited_with = 0
            except SystemExit as exc:
                exited_with = exc.code
        # A healthy run exits with 0 or does not call sys.exit at all
        assert exited_with in (0, None)

    def test_exits_1_on_backend_failure(self, capsys):
        from longtracer.cli import cmd_doctor
        import argparse

        with patch("longtracer.cli._is_model_cached", return_value=True), \
             patch("longtracer.guard.cache.get_default_backend",
                   side_effect=RuntimeError("connection refused")):
            args = argparse.Namespace()
            with pytest.raises(SystemExit) as exc_info:
                cmd_doctor(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# longtracer models prepare
# ---------------------------------------------------------------------------

class TestModelsPrepare:
    """cmd_models_prepare must attempt to load both models and report results."""

    def test_success_path(self, capsys):
        from longtracer.cli import cmd_models_prepare
        import argparse
        args = argparse.Namespace()

        mock_st = MagicMock()
        mock_ce = MagicMock()

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_st), \
             patch("sentence_transformers.CrossEncoder", return_value=mock_ce):
            try:
                cmd_models_prepare(args)
            except SystemExit:
                pass

        out = capsys.readouterr().out
        assert "1/2" in out
        assert "2/2" in out
        # Success line
        assert "Ready" in out or "\u2713" in out

    def test_failure_exits_1(self, capsys):
        from longtracer.cli import cmd_models_prepare
        import argparse
        args = argparse.Namespace()

        with patch("sentence_transformers.SentenceTransformer",
                   side_effect=OSError("no space left on device")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_models_prepare(args)

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "FAILED" in out or "\u2717" in out

    def test_missing_sentence_transformers_exits_1(self, capsys):
        from longtracer.cli import cmd_models_prepare
        import argparse
        args = argparse.Namespace()

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_models_prepare(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _is_model_cached helper
# ---------------------------------------------------------------------------

class TestIsModelCached:
    """Unit-test the cache-detection helper independently."""

    def test_returns_false_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        from longtracer.cli import _is_model_cached  # type: ignore[attr-defined]
        # No model files exist → should return False
        assert _is_model_cached("sentence-transformers/all-MiniLM-L6-v2") is False

    def test_returns_true_when_safetensors_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        from longtracer.cli import _is_model_cached  # type: ignore[attr-defined]
        # Create fake model weight file in the expected path
        model_dir = tmp_path / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "abc123"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"fake")
        assert _is_model_cached("sentence-transformers/all-MiniLM-L6-v2") is True
