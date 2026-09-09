"""Execute the CI APT setup without modifying the host's package sources."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION = "./.github/actions/install-ubuntu-packages"


@pytest.fixture
def apt_host(tmp_path: Path) -> SimpleNamespace:
    """Record package-manager calls while retaining an unrelated vendor source."""
    sources = tmp_path / "etc with spaces/apt"
    source_parts = sources / "sources.list.d"
    source_parts.mkdir(parents=True)
    (source_parts / "google-chrome.list").write_text("deb https://dl.google.com/linux/chrome/deb/ stable main\n")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    sudo = binary_dir / "sudo"
    sudo.write_text('#!/bin/bash\nexec "$@"\n')
    sudo.chmod(0o755)
    apt = binary_dir / "apt-get"
    apt.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['APT_CALLS']).open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "operation = 'update' if 'update' in sys.argv else 'install'\n"
        "sys.exit(int(os.environ.get('APT_FAIL_' + operation.upper(), '0')))\n"
    )
    apt.chmod(0o755)
    return SimpleNamespace(sources=sources, binary_dir=binary_dir, calls=tmp_path / "calls.jsonl")


def _run_setup(host: SimpleNamespace, packages: str = "cmake g++ libopencv-dev libeigen3-dev", **env: str) -> tuple:
    """Run the actual composite-action script with isolated sources and binaries."""
    action = yaml.safe_load((REPO_ROOT / ACTION / "action.yml").read_text())
    step = action["runs"]["steps"][0]
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": f"{host.binary_dir}{os.pathsep}{os.environ['PATH']}",
            "APT_ETC": str(host.sources),
            "PACKAGES": packages,
            "APT_CALLS": str(host.calls),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [json.loads(line) for line in host.calls.read_text().splitlines()] if host.calls.exists() else []
    return result, calls


@pytest.mark.parametrize("source_file", ("sources.list.d/ubuntu.sources", "sources.list"))
@pytest.mark.parametrize("packages", ("cmake g++ libopencv-dev libeigen3-dev", "jq\ncmake"))
def test_update_and_install_use_only_ubuntu_sources(apt_host: SimpleNamespace, source_file: str, packages: str) -> None:
    ubuntu = apt_host.sources / source_file
    ubuntu.write_text("Ubuntu source fixture\n")
    if source_file.endswith(".sources"):
        # Newer runners must prefer Deb822 even when a legacy file exists.
        (apt_host.sources / "sources.list").write_text("Legacy source fixture\n")
    original_sources = {path: path.read_bytes() for path in apt_host.sources.rglob("*") if path.is_file()}

    result, calls = _run_setup(apt_host, packages)

    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    options = calls[0][:-1]
    assert calls[0][-1] == "update"
    assert calls[1] == [*options, "install", "--yes", "--", *packages.split()]
    assert options[::2] == ["-o"] * 5
    assert set(options[1::2]) == {
        f"Dir::Etc::sourcelist={ubuntu}",
        "Dir::Etc::sourceparts=/dev/null",
        "APT::Get::List-Cleanup=0",
        "APT::Update::Error-Mode=any",
        "Acquire::Retries=3",
    }
    assert {path: path.read_bytes() for path in apt_host.sources.rglob("*") if path.is_file()} == original_sources


@pytest.mark.parametrize(("operation", "call_count"), (("UPDATE", 1), ("INSTALL", 2)))
def test_package_errors_fail_the_step(apt_host: SimpleNamespace, operation: str, call_count: int) -> None:
    (apt_host.sources / "sources.list").write_text("Ubuntu source fixture\n")

    result, calls = _run_setup(apt_host, **{f"APT_FAIL_{operation}": "100"})

    assert result.returncode == 100
    assert len(calls) == call_count


def test_missing_ubuntu_sources_fail_before_invoking_apt(apt_host: SimpleNamespace) -> None:
    (apt_host.sources / "sources.list.d/ubuntu.sources").touch()
    (apt_host.sources / "sources.list").touch()

    result, calls = _run_setup(apt_host)

    assert result.returncode != 0
    assert "No Ubuntu APT source file found" in result.stderr
    assert calls == []


def test_empty_packages_fail_before_invoking_apt(apt_host: SimpleNamespace) -> None:
    (apt_host.sources / "sources.list").write_text("Ubuntu source fixture\n")

    result, calls = _run_setup(apt_host, " \n ")

    assert result.returncode != 0
    assert "At least one Ubuntu package is required" in result.stderr
    assert calls == []


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "packages"),
    (
        ("ci", "native_build", "cmake g++ libopencv-dev libeigen3-dev"),
        ("ci", "cpp_trackers", "cmake g++ libopencv-dev libeigen3-dev"),
        ("ci", "metrics", "jq"),
        ("wheels", "source_gates", "cmake g++ libopencv-dev libeigen3-dev"),
        ("benchmark", "mot-metrics-benchmark", "cmake g++ libopencv-dev libeigen3-dev"),
    ),
)
def test_ubuntu_jobs_use_the_shared_installer(workflow_name: str, job_name: str, packages: str) -> None:
    workflow = yaml.safe_load((REPO_ROOT / f".github/workflows/{workflow_name}.yml").read_text())
    steps = workflow["jobs"][job_name]["steps"]
    installer = next(step for step in steps if step.get("uses") == ACTION)

    assert installer["with"]["packages"] == packages
    assert not any("apt-get" in step.get("run", "") for step in steps)
    if job_name == "cpp_trackers":
        assert installer["if"] == "runner.os == 'Linux'"
    if job_name == "source_gates":
        source_checkout = next(
            step for step in steps if step.get("uses") == "./.github/actions/checkout-release-source"
        )
        assert steps.index(installer) < steps.index(source_checkout)
