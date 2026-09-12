"""Optional dependencies are installed only through an explicit CLI action."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import install as install_command


@pytest.mark.parametrize("quiet", [False, True])
def test_install_command_passes_explicit_requirements_and_indices(monkeypatch, quiet: bool) -> None:
    captured = {}

    def install(extras, **kwargs):
        captured.update(extras=extras, **kwargs)

    monkeypatch.setattr(install_command, "install_extras", install)
    arguments = [
        "install",
        "--extra",
        "onnx",
        "--extra",
        "research",
        "--requirement",
        "demo>=2",
        "--requirement",
        "second[feature]==3",
        "--extra-index-url",
        "https://packages.example/simple",
        "--extra-index-url",
        "https://another.example/simple",
    ]
    if quiet:
        arguments.append("--quiet")

    result = CliRunner().invoke(boxmot, arguments)

    assert result.exit_code == 0, result.output
    assert captured == {
        "extras": ("onnx", "research"),
        "requirements": ("demo>=2", "second[feature]==3"),
        "extra_args": (
            "--extra-index-url",
            "https://packages.example/simple",
            "--extra-index-url",
            "https://another.example/simple",
        ),
        "verbose": not quiet,
    }
    assert ("Requested dependencies are available." in result.output) is not quiet


def test_install_requires_an_explicit_selection(monkeypatch) -> None:
    monkeypatch.setattr(install_command, "install_extras", lambda *args, **kwargs: pytest.fail("Unexpected install"))

    result = CliRunner().invoke(boxmot, ["install"])

    assert result.exit_code == 2
    assert "Provide at least one --extra or --requirement" in result.output


def test_install_errors_are_reported_as_command_errors(monkeypatch) -> None:
    def install(*args, **kwargs):
        raise ValueError("Unknown extra 'invalid'")

    monkeypatch.setattr(install_command, "install_extras", install)

    result = CliRunner().invoke(boxmot, ["install", "--extra", "invalid"])

    assert result.exit_code == 1
    assert "Error: Unknown extra 'invalid'" in result.output


def test_kitti_devkit_install_does_not_install_python_packages(monkeypatch, tmp_path) -> None:
    from boxmot.engine.eval import kitti_object_backend

    calls = []
    binary = tmp_path / "evaluator"
    monkeypatch.setattr(install_command, "install_extras", lambda *args, **kwargs: pytest.fail("Unexpected pip"))
    monkeypatch.setattr(kitti_object_backend, "install_kitti_object_backend", lambda path: calls.append(path) or binary)

    result = CliRunner().invoke(boxmot, ["install", "--kitti-devkit", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls == [tmp_path]
    assert str(binary) in result.output


def test_install_help_does_not_inspect_dependencies_or_start_an_installer() -> None:
    script = """
import subprocess
import sys
from click.testing import CliRunner
from boxmot.engine.cli import boxmot

def fail_if_install_started(*args, **kwargs):
    raise AssertionError("Help must not start an installer")

subprocess.run = fail_if_install_started

result = CliRunner().invoke(boxmot, ["install", "--help"])
assert result.exit_code == 0, result.output
assert "--requirement" in result.output
assert "boxmot.utils.dependencies" not in sys.modules
assert not any(name in sys.modules for name in ("torch", "cv2"))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
