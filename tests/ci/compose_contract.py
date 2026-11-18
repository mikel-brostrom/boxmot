"""Validate the real Compose merges without a daemon, models, or physical devices.

Run from the repository root: python -m tests.ci.compose_contract
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = {"cpu": "vllm-cpu", "gpu": "vllm-gpu", "ollama": "ollama"}


def render(provider: str, webcam: bool, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolve authored files in an environment independent of a developer's settings."""
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("BOXMOT_", "VLLM_", "OLLAMA_", "VIDEO_", "COMPOSE_"))
    }
    environment.update(DISPLAY=":99", BOXMOT_UID="12345", BOXMOT_GID="23456", VIDEO_GID="34567")
    environment.update(overrides or {})
    command = ["docker", "compose", "--env-file", os.devnull, "-f", "docker/compose.attributes.yaml"]
    if provider != "cpu":
        command += ["-f", f"docker/compose.attributes.{provider}.yaml"]
    if webcam:
        command += ["-f", "docker/compose.webcam.yaml"]
    command += ["--profile", "track", "config", "--format", "json"]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def check(provider: str, webcam: bool) -> None:
    """Check dependency selection, persistence, hardware requests, and immutable defaults."""
    config = render(provider, webcam)
    services = config["services"]
    server_name = PROVIDERS[provider]
    assert set(services) == {"boxmot", "postgres", server_name}, services.keys()
    worker, server = services["boxmot"], services[server_name]
    assert set(worker["depends_on"]) == {"postgres", server_name}
    assert all(value["condition"] == "service_healthy" for value in worker["depends_on"].values())
    assert worker["build"]["target"] == ("cli-gpu" if provider == "gpu" else "cli-cpu")
    assert worker["user"] == "12345:23456" and worker["init"] is True
    assert worker["stop_signal"] == "SIGINT" and worker["stop_grace_period"] == "30s"
    assert "@sha256:" in server["image"] and re.fullmatch(r".*@sha256:[0-9a-f]{64}", server["image"])
    assert "--exist-ok" not in worker["entrypoint"] + worker["command"]
    assert worker["command"] == (["--source", "0", "--show"] if webcam else ["--source", "/inputs/video.mp4"])
    mounts = {volume["target"]: volume for volume in worker["volumes"]}
    assert mounts["/inputs"]["read_only"] is True
    assert mounts["/opt/boxmot/models"]["source"] == str(ROOT / "models")
    assert mounts["/opt/boxmot/runs"]["source"] == str(ROOT / "runs")
    assert "@postgres:5432/" in worker["environment"]["BOXMOT_ATTRIBUTES_DATABASE_URL"]
    endpoint = "http://ollama:11434" if provider == "ollama" else f"http://{server_name}:8000"
    assert worker["entrypoint"][worker["entrypoint"].index("--attributes-host") + 1] == endpoint
    for service in (server, services["postgres"]):
        assert service["healthcheck"]["test"]
        assert all(port["host_ip"] == "127.0.0.1" for port in service["ports"])
    if provider == "ollama":
        assert re.fullmatch(r"[0-9a-f]{64}", worker["environment"]["OLLAMA_MODEL_DIGEST"])
        assert worker["entrypoint"][:2] == ["python", "-c"]
    else:
        arguments = server["command"]
        assert re.fullmatch(r"[0-9a-f]{40}", arguments[arguments.index("--revision") + 1])
        cache = next(volume for volume in server["volumes"] if volume["target"] == "/root/.cache/huggingface")
        assert cache["type"] == "volume" and cache["source"] == "vllm-models"
    if provider == "gpu":
        for service in (worker, server):
            device = service["deploy"]["resources"]["reservations"]["devices"][0]
            assert device["driver"] == "nvidia" and device["device_ids"] == ["0"]
        assert worker["entrypoint"][worker["entrypoint"].index("--device") + 1] == "0"
    else:
        assert "deploy" not in worker
    if webcam:
        assert mounts["/tmp/.X11-unix"]["read_only"] is True
        assert worker["environment"]["DISPLAY"] == ":99" and worker["group_add"] == ["34567"]
        assert worker["devices"][0]["target"] == "/dev/video0"


def main() -> None:
    """Check every supported provider/input combination."""
    for provider in PROVIDERS:
        for webcam in (False, True):
            check(provider, webcam)
            print(f"PASS: {provider} {'webcam' if webcam else 'video'}")


if __name__ == "__main__":
    main()
