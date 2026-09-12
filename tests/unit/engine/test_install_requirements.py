"""Explicit dependency installation targets and failure handling."""

from __future__ import annotations

import subprocess
import sys

import pytest

from boxmot.engine.commands import install as install_command
from boxmot.utils import dependencies as dependency_utils


@pytest.fixture
def installed_requirements(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Model installed requirements without changing the test environment."""

    installed: set[str] = set()

    def missing(requirements):
        return [requirement for requirement in requirements if requirement not in installed]

    def require(requirements, **kwargs):
        unresolved = missing(requirements)
        if unresolved:
            raise ImportError(f"Still missing: {', '.join(unresolved)}")

    monkeypatch.setattr(dependency_utils, "missing_requirements", missing)
    monkeypatch.setattr(dependency_utils, "require_packages", require)
    return installed


def test_satisfied_requirements_do_not_start_an_installer(monkeypatch, installed_requirements) -> None:
    installed_requirements.add("demo>=2")
    monkeypatch.setattr(install_command.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected install"))

    install_command.install_requirements(["demo>=2"])


@pytest.mark.parametrize("uv_available", [True, False])
def test_installer_targets_the_current_python_and_keeps_normal_caching(
    monkeypatch, installed_requirements, uv_available: bool
) -> None:
    monkeypatch.setattr(install_command.shutil, "which", lambda name: "/usr/bin/uv" if uv_available else None)
    commands = []
    installed_requirements.add("present==1")

    def run(command, **kwargs):
        commands.append(command)
        installed_requirements.add("missing>=2")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(install_command.subprocess, "run", run)

    install_command.install_requirements(
        ["present==1", "missing>=2", "missing>=2"],
        extra_args=("--extra-index-url", "https://packages.example/simple"),
        verbose=False,
    )

    prefix = (
        ["uv", "pip", "install", "--python", sys.executable]
        if uv_available
        else [sys.executable, "-m", "pip", "install"]
    )
    assert commands == [[*prefix, "--extra-index-url", "https://packages.example/simple", "present==1", "missing>=2"]]


def test_resolution_keeps_satisfied_constraints_when_another_range_is_missing(monkeypatch) -> None:
    """An installed lower version must retain the requested upper bound on upgrade."""

    from packaging.requirements import Requirement

    installed_version = "1"
    commands = []

    def missing(requirements):
        return [
            requirement for requirement in requirements if installed_version not in Requirement(requirement).specifier
        ]

    def require(requirements, **kwargs):
        assert missing(requirements) == []

    def run(command, **kwargs):
        nonlocal installed_version
        commands.append(command)
        installed_version = "2" if "demo<3" in command else "4"
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(dependency_utils, "missing_requirements", missing)
    monkeypatch.setattr(dependency_utils, "require_packages", require)
    monkeypatch.setattr(install_command.shutil, "which", lambda name: None)
    monkeypatch.setattr(install_command.subprocess, "run", run)

    install_command.install_requirements(["demo>=2", "demo<3"], verbose=False)

    assert commands == [[sys.executable, "-m", "pip", "install", "demo>=2", "demo<3"]]
    assert installed_version == "2"


@pytest.mark.parametrize("failure", ["exit-error", "missing-executable"])
def test_uv_failure_falls_back_to_the_current_pythons_pip(monkeypatch, installed_requirements, failure: str) -> None:
    monkeypatch.setattr(install_command.shutil, "which", lambda name: "/usr/bin/uv")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "uv":
            if failure == "missing-executable":
                raise FileNotFoundError("uv disappeared")
            raise subprocess.CalledProcessError(1, command, stderr="uv could not resolve the package")
        installed_requirements.add("demo>=2")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(install_command.subprocess, "run", run)

    install_command.install_requirements(["demo>=2"], verbose=False)

    assert commands == [
        ["uv", "pip", "install", "--python", sys.executable, "demo>=2"],
        [sys.executable, "-m", "pip", "install", "demo>=2"],
    ]


def test_all_installer_failures_report_the_cause(monkeypatch, installed_requirements) -> None:
    monkeypatch.setattr(install_command.shutil, "which", lambda name: "/usr/bin/uv")

    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="package index is unavailable")

    monkeypatch.setattr(install_command.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="package index is unavailable"):
        install_command.install_requirements(["demo>=2"], verbose=False)


def test_successful_subprocess_must_actually_satisfy_requirements(monkeypatch, installed_requirements) -> None:
    monkeypatch.setattr(install_command.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        install_command.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    with pytest.raises(ImportError, match="Still missing: demo>=2"):
        install_command.install_requirements(["demo>=2"], verbose=False)


def test_extras_install_missing_dependencies_without_reinstalling_boxmot(monkeypatch, installed_requirements) -> None:
    monkeypatch.setattr(install_command.shutil, "which", lambda name: None)
    monkeypatch.setattr(dependency_utils, "extra_requirements", lambda extra: ("shared>=1", f"{extra}-dependency>=2"))
    installed_requirements.update({"shared>=1", "onnx-dependency>=2"})
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        installed_requirements.update({"openvino-dependency>=2", "custom>=3"})
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(install_command.subprocess, "run", run)

    install_command.install_extras(["onnx", "openvino", "onnx"], requirements=["custom>=3"], verbose=False)

    assert commands == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "shared>=1",
            "onnx-dependency>=2",
            "openvino-dependency>=2",
            "custom>=3",
        ]
    ]


@pytest.mark.parametrize("profile", ["cpu", "cu130", "CPU", " cpu ", " cu130 "])
def test_torch_source_profiles_require_the_documented_sync_workflow(monkeypatch, profile: str) -> None:
    monkeypatch.setattr(install_command.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected install"))

    with pytest.raises(ValueError, match=f"uv sync --extra {profile.strip().lower()}"):
        install_command.install_extras([profile])


@pytest.mark.parametrize("requirement", ["boxmot", "BoxMOT[onnx]==25.0.0"])
def test_dependency_installer_rejects_reinstalling_boxmot(monkeypatch, requirement: str) -> None:
    monkeypatch.setattr(install_command.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected install"))

    with pytest.raises(ValueError, match="Install or upgrade BoxMOT separately"):
        install_command.install_requirements([requirement])


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--python", "/other/python"),
        ("--target", "/other/site-packages"),
        ("--prefix", "/other/environment"),
        ("--extra-index-url",),
        ("--extra-index-url", "--python=/other/python"),
    ],
)
def test_extra_arguments_cannot_redirect_the_installation(monkeypatch, extra_args: tuple[str, ...]) -> None:
    monkeypatch.setattr(install_command.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected install"))

    with pytest.raises(ValueError, match="'--extra-index-url URL' pairs"):
        install_command.install_requirements(["demo>=2"], extra_args=extra_args)
