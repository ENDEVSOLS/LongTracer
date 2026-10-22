"""Exercise the release checker against wheel and sdist metadata, offline."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_package_metadata.py"
IMAGES = (
    ("LongTracer TUI Demo", "demo-vhs-gifs/tui-demo.gif"),
    ("LongTracer Python API", "demo-vhs-gifs/python-api.gif"),
    ("LongTracer CLI Check Demo", "demo-vhs-gifs/cli-check.gif"),
    ("LongTracer Dashboard", "assets/dashboard.png"),
)
BASE_URL = "https://raw.githubusercontent.com/ENDEVSOLS/LongTracer/main/"


def metadata() -> str:
    # A version different from the real release verifies that the checker reads
    # project metadata rather than hardcoding the current version.
    headers = "Name: longtracer\nVersion: 9.8.7\nRequires-Python: >=3.11\nDescription-Content-Type: text/markdown\n\n"
    return headers + "\n".join(f"![{label}]({BASE_URL}{path})" for label, path in IMAGES)


def write_artifact(directory: Path, kind: str, content: str) -> Path:
    if kind == "wheel":
        path = directory / "longtracer-9.8.7-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("longtracer-9.8.7.dist-info/METADATA", content)
    else:
        path = directory / "longtracer-9.8.7.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            raw = content.encode("utf-8")
            info = tarfile.TarInfo("longtracer-9.8.7/PKG-INFO")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return path


@pytest.fixture
def release(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "longtracer"\nversion = "9.8.7"\nrequires-python = ">=3.11"\n')
    dist = tmp_path / "dist"
    dist.mkdir()
    for kind in ("wheel", "sdist"):
        write_artifact(dist, kind, metadata())
    return project, dist


def run_checker(release):
    project, dist = release
    return subprocess.run(
        [sys.executable, str(CHECKER), "--project", str(project), "--dist-dir", str(dist)],
        capture_output=True, text=True, check=False,
    )


def test_valid_wheel_and_sdist_pass(release):
    result = run_checker(release)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("PASS") == 2


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "before, after, expected_error",
    [
        (BASE_URL + "demo-vhs-gifs/tui-demo.gif", "demo-vhs-gifs/tui-demo.gif", "LongTracer TUI Demo"),
        (f"![LongTracer CLI Check Demo]({BASE_URL}demo-vhs-gifs/cli-check.gif)", "", "LongTracer CLI Check Demo"),
        ("Version: 9.8.7", "Version: 9.8.6", "Version"),
        ("Requires-Python: >=3.11", "Requires-Python: >=3.10", "Requires-Python"),
        ("Description-Content-Type: text/markdown", "Description-Content-Type: text/plain", "Description-Content-Type"),
    ],
)
def test_bad_packaged_metadata_fails(release, kind, before, after, expected_error):
    _, dist = release
    artifact = write_artifact(dist, kind, metadata().replace(before, after))
    result = run_checker(release)
    assert result.returncode == 1
    assert artifact.name in result.stderr
    assert expected_error in result.stderr


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_internal_screenshot_reminder_fails(release, kind):
    _, dist = release
    write_artifact(dist, kind, metadata() + "\nPlease save a screenshot of the dashboard.")
    result = run_checker(release)
    assert result.returncode == 1
    assert "internal screenshot reminder" in result.stderr


@pytest.mark.parametrize("pattern", ["*.whl", "*.tar.gz"])
def test_missing_distribution_fails(release, pattern):
    _, dist = release
    next(dist.glob(pattern)).unlink()
    result = run_checker(release)
    assert result.returncode == 1
    assert "both a wheel and a source distribution are required" in result.stderr
