"""Exercise release preparation with an isolated Git project and offline uv."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="Release preparation uses Python 3.11 or newer.")

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/bump_release.py"
SPEC = importlib.util.spec_from_file_location("bump_release", SCRIPT)
release_version = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_version)


def _run(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a fixture-only command with actionable captured output on failure."""
    return subprocess.run(arguments, cwd=root, check=True, capture_output=True, text=True)


def _commit(root: Path) -> None:
    """Record fixture inputs without configuring the user's Git identity."""
    _run(root, "git", "add", ".")
    _run(
        root,
        "git",
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.invalid",
        "commit",
        "-m",
        "fixture",
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Create a dependency-free editable package using the repository's pinned uv."""
    if shutil.which("uv") is None:
        pytest.skip("Release bump integration checks require uv 0.12.4.")
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_PYTHON", sys.executable)
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "unused-venv"))
    (root / "pyproject.toml").write_text(
        '[project]\nname = "boxmot"\nversion = "1.2.3"\nrequires-python = ">=3.11"\n'
        'dependencies = []\n\n[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n'
        '\n[tool.uv]\nrequired-version = "==0.12.4"\n',
        encoding="utf-8",
    )
    (root / "boxmot").mkdir()
    (root / "boxmot/__init__.py").write_text(
        '"""Fixture package."""\n__version__ = "1.2.3"\nUNCHANGED = "1.2.3"\n', encoding="utf-8"
    )
    (root / "README.md").write_text("Release fixture.\n", encoding="utf-8")
    _run(root, "uv", "lock")
    _run(root, "git", "init")
    _commit(root)
    yield root
    assert not (tmp_path / "unused-venv").exists(), "Version preparation must never sync an environment."
    assert not (root / ".venv").exists()


@pytest.mark.parametrize("kind,expected", (("major", "2.0.0"), ("minor", "1.3.0"), ("patch", "1.2.4")))
def test_bumps_project_runtime_and_lock_without_syncing(project: Path, kind: str, expected: str) -> None:
    before = (project / "boxmot/__init__.py").read_text(encoding="utf-8")

    assert release_version.bump_release(project, kind) == ("1.2.3", expected)

    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == lock["package"][0]["version"] == expected
    assert lock["package"][0]["source"] == {"editable": "."}
    assert (project / "boxmot/__init__.py").read_text(encoding="utf-8") == before.replace(
        '__version__ = "1.2.3"', f'__version__ = "{expected}"'
    )
    assert set(_run(project, "git", "diff", "--name-only").stdout.splitlines()) == release_version.VERSION_FILES


def test_replaces_only_ast_value_with_unicode_and_inline_comment(project: Path) -> None:
    source = '"""Non-ASCII: 🎯."""\nmarker = "🎯"; __version__ = "1.2.3"  # keep this\nOTHER = "1.2.3"\n'
    (project / "boxmot/__init__.py").write_text(source, encoding="utf-8")
    _commit(project)

    release_version.bump_release(project, "patch")

    assert (project / "boxmot/__init__.py").read_text(encoding="utf-8") == source.replace(
        '__version__ = "1.2.3"', '__version__ = "1.2.4"'
    )


@pytest.mark.parametrize("filename", ("pyproject.toml", "boxmot/__init__.py", "uv.lock"))
def test_rejects_disagreeing_versions_before_running_uv(project: Path, filename: str) -> None:
    path = project / filename
    path.write_text(path.read_text(encoding="utf-8").replace('"1.2.3"', '"1.2.2"'), encoding="utf-8")
    _commit(project)

    with pytest.raises(ValueError, match="versions must agree"):
        release_version.bump_release(project, "patch")

    assert not _run(project, "git", "status", "--porcelain").stdout


@pytest.mark.parametrize("version", ("1.2", "1.2.3rc1", "01.2.3"))
def test_rejects_nonstable_or_malformed_versions(project: Path, version: str) -> None:
    for filename in release_version.VERSION_FILES:
        path = project / filename
        path.write_text(path.read_text(encoding="utf-8").replace('"1.2.3"', f'"{version}"'), encoding="utf-8")
    _commit(project)

    with pytest.raises(ValueError, match="stable major.minor.patch"):
        release_version.bump_release(project, "patch")

    assert not _run(project, "git", "status", "--porcelain").stdout


@pytest.mark.parametrize(
    "assignment",
    (
        '__version__ = "1.2.3"\n__version__ = "1.2.3"\n',
        '__version__ = "1.2.3"\n__version__ += ".post1"\n',
        "__version__ = get_version()\n",
    ),
)
def test_rejects_ambiguous_or_computed_runtime_version(project: Path, assignment: str) -> None:
    (project / "boxmot/__init__.py").write_text(assignment, encoding="utf-8")
    _commit(project)

    with pytest.raises(ValueError, match="__version__ assignment"):
        release_version.bump_release(project, "patch")


def test_rejects_a_noneditable_lock_record(project: Path) -> None:
    path = project / "uv.lock"
    path.write_text(path.read_text(encoding="utf-8").replace('editable = "."', 'virtual = "."'), encoding="utf-8")
    _commit(project)

    with pytest.raises(ValueError, match="local editable boxmot"):
        release_version.bump_release(project, "patch")


def test_rejects_unknown_bump_and_dirty_checkout(project: Path) -> None:
    with pytest.raises(ValueError, match="Unknown release bump"):
        release_version.bump_release(project, "rc")
    (project / "README.md").write_text("Uncommitted changes.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean checkout"):
        release_version.bump_release(project, "patch")


def test_rejects_unexpected_files_changed_by_uv(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = subprocess.run

    def unexpected_change(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = run(arguments, **kwargs)
        if arguments[:2] == ["uv", "version"]:
            (project / "unexpected.txt").write_text("Unexpected generated output.\n", encoding="utf-8")
        return result

    monkeypatch.setattr(release_version.subprocess, "run", unexpected_change)
    with pytest.raises(ValueError, match="must change only.*unexpected.txt"):
        release_version.bump_release(project, "patch")


def test_rejects_a_lock_version_left_stale_by_uv(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = subprocess.run
    original_lock = (project / "uv.lock").read_bytes()

    def stale_lock(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = run(arguments, **kwargs)
        if arguments[:2] == ["uv", "version"]:
            (project / "uv.lock").write_bytes(original_lock)
        return result

    monkeypatch.setattr(release_version.subprocess, "run", stale_lock)
    with pytest.raises(ValueError, match="versions must agree"):
        release_version.bump_release(project, "patch")


def test_cli_emits_versions_for_github_actions(project: Path, tmp_path: Path) -> None:
    output = tmp_path / "github-output"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--bump", "minor"],
        cwd=project,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.read_text(encoding="utf-8") == "old_version=1.2.3\nversion=1.3.0\n"
    assert "Prepared BoxMOT 1.2.3 -> 1.3.0" in result.stdout
