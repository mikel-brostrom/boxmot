"""Exercise the webcam launcher without Docker or elevated privileges."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "docker" / "webcam.sh"


@dataclass
class Launcher:
    """A temporary checkout with recording Docker and sudo executables."""

    repo: Path
    environment: dict[str, str]
    settings: Path
    log: Path
    sudo_log: Path

    def run(self, *, docker_access: bool = True, exit_code: int = 0) -> subprocess.CompletedProcess[str]:
        """Launch from outside the checkout with a controlled Docker outcome."""
        self.settings.write_text(json.dumps({"access": docker_access, "exit_code": exit_code}))
        return subprocess.run(
            [str(self.repo / "docker" / "webcam.sh")],
            cwd=self.repo.parent,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    def calls(self) -> list[dict]:
        """Return recorded Docker invocations, including the launched environment."""
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []


@pytest.fixture
def launcher(tmp_path: Path) -> Launcher:
    """Copy only the launcher and supply subprocess fakes on PATH."""
    repo = tmp_path / "checkout with spaces"
    (repo / "docker").mkdir(parents=True)
    shutil.copy2(LAUNCHER, repo / "docker" / "webcam.sh")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    settings, log, sudo_log = (tmp_path / name for name in ("settings.json", "docker.jsonl", "sudo.log"))
    scripts = {
        "docker": f"""import json, os, pathlib, sys
settings = json.loads(pathlib.Path({str(settings)!r}).read_text())
with pathlib.Path({str(log)!r}).open('a') as log:
    log.write(json.dumps({{'args': sys.argv[1:], 'cwd': os.getcwd(), 'env': dict(os.environ)}}) + '\\n')
if sys.argv[1:] == ['info']:
    sys.exit(0 if settings['access'] else 1)
sys.exit(settings['exit_code'] if 'up' in sys.argv[1:] else 0)
""",
        "sudo": f"""import os, pathlib, sys
pathlib.Path({str(sudo_log)!r}).write_text('called')
os.execvpe(sys.argv[1], sys.argv[1:], {{'PATH': os.environ['PATH'], 'HOME': '/root'}})
""",
    }
    for name, source in scripts.items():
        executable = binary_dir / name
        executable.write_text(f"#!{sys.executable}\n{source}")
        executable.chmod(0o755)
    environment = {
        "PATH": f"{binary_dir}:{os.defpath}",
        "HOME": str(tmp_path / "desktop home"),
        "DISPLAY": ":42",
        "VIDEO_DEVICE": "/dev/null",
    }
    return Launcher(repo, environment, settings, log, sudo_log)


@pytest.mark.parametrize("docker_access", (True, False))
def test_default_launch_preserves_desktop_identity_and_cache(launcher: Launcher, docker_access: bool) -> None:
    result = launcher.run(docker_access=docker_access)

    assert result.returncode == 0, result.stderr
    invocation = launcher.calls()[-1]
    assert invocation["cwd"] == str(launcher.repo)
    assert invocation["args"] == [
        "compose", "-f", "docker/compose.attributes.yaml", "-f", "docker/compose.webcam.yaml",
        "up", "--build", "--abort-on-container-exit", "--exit-code-from", "boxmot", "boxmot",
    ]
    cache = Path(launcher.environment["HOME"]) / ".cache" / "huggingface"
    expected = {
        "BOXMOT_UID": str(os.getuid()),
        "BOXMOT_GID": str(os.getgid()),
        "HF_CACHE_DIR": str(cache),
        "DISPLAY": ":42",
        "VIDEO_DEVICE": "/dev/null",
        "VIDEO_GID": str(Path("/dev/null").stat().st_gid),
    }
    assert {name: invocation["env"][name] for name in expected} == expected
    assert invocation["env"]["HOME"] == (launcher.environment["HOME"] if docker_access else "/root")
    assert launcher.sudo_log.exists() is not docker_access
    assert cache.is_dir()
    assert all((launcher.repo / name).is_dir() for name in ("data", "models", "runs"))


def test_sudo_keeps_custom_settings_and_resolves_relative_cache(launcher: Launcher) -> None:
    overrides = {
        "BOXMOT_UID": "1234",
        "BOXMOT_GID": "5678",
        "VIDEO_GID": "999",
        "HF_CACHE_DIR": "model cache $(literal)",
        "BOXMOT_INPUT_DIR": "/inputs with spaces",
        "BOXMOT_SOURCE": "0",
        "BOXMOT_DETECTOR": "custom detector.onnx",
        "BOXMOT_TRACKER": "botsort",
        "BOXMOT_REID": "custom reid.pt",
        "BOXMOT_RUN_NAME": "webcam run",
        "VLLM_CPU_IMAGE": "vllm:test",
        "VLLM_PORT": "8010",
        "OLLAMA_PORT": "11435",
        "VLLM_CPU_OMP_THREADS_BIND": "0-3",
        "VLLM_CPU_NUM_OF_RESERVED_CPU": "1",
    }
    launcher.environment.update(overrides)

    result = launcher.run(docker_access=False)

    assert result.returncode == 0, result.stderr
    expected = {**overrides, "HF_CACHE_DIR": str(launcher.repo / overrides["HF_CACHE_DIR"])}
    environment = launcher.calls()[-1]["env"]
    assert {name: environment[name] for name in expected} == expected
    assert Path(environment["HF_CACHE_DIR"]).is_dir()


@pytest.mark.parametrize("docker_access", (True, False))
def test_launcher_propagates_compose_exit_status(launcher: Launcher, docker_access: bool) -> None:
    assert launcher.run(docker_access=docker_access, exit_code=37).returncode == 37


@pytest.mark.parametrize(
    "name,value,error",
    (("DISPLAY", "", "DISPLAY is not set"), ("VIDEO_DEVICE", "/missing-camera", "Webcam device not found")),
)
def test_invalid_desktop_inputs_do_not_launch_stack(launcher: Launcher, name: str, value: str, error: str) -> None:
    launcher.environment[name] = value

    result = launcher.run(docker_access=False)

    assert result.returncode == 1
    assert error in result.stderr
    assert not any("up" in call["args"] for call in launcher.calls())
    assert not launcher.sudo_log.exists()
    assert not (launcher.repo / "runs").exists()
