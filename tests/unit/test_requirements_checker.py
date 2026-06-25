import subprocess
import sys

import boxmot.utils.checks as checks
from boxmot.utils.checks import RequirementsChecker, missing_requirements, requirement_satisfied


def test_install_packages_uses_uv_when_available(monkeypatch):
    commands = []
    monkeypatch.setattr(checks.shutil, "which", lambda executable: "/usr/bin/uv" if executable == "uv" else None)

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    RequirementsChecker()._install_packages(("demo-package>=1",))

    assert commands == [["uv", "pip", "install", "--no-cache-dir", "demo-package>=1"]]


def test_install_packages_falls_back_to_python_pip_when_uv_fails(monkeypatch):
    commands = []
    monkeypatch.setattr(checks.shutil, "which", lambda executable: "/usr/bin/uv" if executable == "uv" else None)

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        if cmd[0] == "uv":
            raise subprocess.CalledProcessError(1, cmd, stderr="uv failed")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    RequirementsChecker()._install_packages(("demo-package>=1",))

    assert commands == [
        ["uv", "pip", "install", "--no-cache-dir", "demo-package>=1"],
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "demo-package>=1"],
    ]


def test_install_packages_falls_back_to_python_pip_when_uv_is_missing(monkeypatch):
    commands = []
    monkeypatch.setattr(checks.shutil, "which", lambda _executable: None)

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    RequirementsChecker()._install_packages(("demo-package>=1",))

    assert commands == [
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "demo-package>=1"],
    ]


def test_install_packages_splits_shell_style_extra_args(monkeypatch):
    commands = []
    monkeypatch.setattr(checks.shutil, "which", lambda executable: "/usr/bin/uv" if executable == "uv" else None)

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    RequirementsChecker()._install_packages(
        ("nvidia-tensorrt",),
        extra_args="--extra-index-url https://pypi.ngc.nvidia.com",
    )

    assert commands == [
        [
            "uv",
            "pip",
            "install",
            "--no-cache-dir",
            "--extra-index-url",
            "https://pypi.ngc.nvidia.com",
            "nvidia-tensorrt",
        ],
    ]


def test_sync_extra_falls_back_to_python_pip_for_source_checkout(monkeypatch):
    commands = []
    monkeypatch.setattr(checks.shutil, "which", lambda _executable: None)
    monkeypatch.setattr(RequirementsChecker, "_missing_packages", lambda _self, _requirements: ["openvino>=2025.2.0"])

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    RequirementsChecker().sync_extra("openvino", verbose=False)

    assert commands == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-e",
            f"{checks.ROOT}[openvino]",
        ],
    ]


def test_requirement_satisfied_ignores_inactive_markers():
    assert requirement_satisfied("definitely-missing-package>=1; python_version < '0'")


def test_missing_requirements_uses_shared_requirement_parser(monkeypatch):
    def fake_version(name):
        if name == "demo-package":
            return "1.2.0"
        raise checks.PackageNotFoundError

    monkeypatch.setattr(checks, "version", fake_version)

    assert missing_requirements(
        (
            "demo-package>=1.0",
            "demo-package>=2.0",
            "missing-package",
        )
    ) == ["demo-package>=2.0", "missing-package"]
