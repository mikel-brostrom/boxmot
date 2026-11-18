"""Exercise merged Compose deployments and the Ollama startup guard."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import urllib.request
from unittest.mock import Mock

import pytest

from tests.ci.compose_contract import PROVIDERS, check, render

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="Requires the Docker Compose CLI")


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("webcam", [False, True])
def test_authored_compose_combinations(provider: str, webcam: bool) -> None:
    """Validate real merges without a daemon or hardware."""
    check(provider, webcam)


def test_image_revision_and_gpu_overrides_survive_merging() -> None:
    """Deliberate operator overrides remain available after pinning defaults."""
    services = render("gpu", True, {
        "VLLM_GPU_IMAGE": "registry.example.com/vllm:test",
        "VLLM_MODEL_REVISION": "b" * 40,
        "BOXMOT_GPU_ID": "2", "VLLM_GPU_ID": "3",
    })["services"]
    assert services["vllm-gpu"]["image"] == "registry.example.com/vllm:test"
    assert "b" * 40 in services["vllm-gpu"]["command"]
    for name, device_id in (("boxmot", "2"), ("vllm-gpu", "3")):
        assert services[name]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == [device_id]


@pytest.mark.parametrize("model,digest,accepted", [
    ("qwen3.5:9b", "a" * 64, True),
    ("qwen3.5:9b", "b" * 64, False),
    ("unrelated:latest", "a" * 64, False),
    (None, None, False),
])
def test_ollama_checks_full_digest_before_replacing_process(
    monkeypatch: pytest.MonkeyPatch, model: str | None, digest: str | None, accepted: bool,
) -> None:
    """Missing, changed, or unrelated models cannot start the tracking process."""
    entrypoint = render("ollama", False)["services"]["boxmot"]["entrypoint"]
    monkeypatch.setenv("OLLAMA_MODEL_DIGEST", "a" * 64)
    models = [] if model is None else [{"name": model, "digest": digest}]
    request = Mock(return_value=io.BytesIO(json.dumps({"models": models}).encode()))
    monkeypatch.setattr(urllib.request, "urlopen", request)
    execute = Mock(side_effect=RuntimeError("replaced process"))
    monkeypatch.setattr(os, "execvp", execute)
    monkeypatch.setattr(sys, "argv", ["-c", *entrypoint[3:], "--source", "/inputs/video.mp4"])
    if accepted:
        with pytest.raises(RuntimeError, match="replaced process"):
            exec(entrypoint[2], {})
        execute.assert_called_once_with("boxmot", ["boxmot", "track", *sys.argv[1:]])
    else:
        with pytest.raises(SystemExit, match="missing or its digest differs"):
            exec(entrypoint[2], {})
        execute.assert_not_called()
    request.assert_called_once_with("http://ollama:11434/api/tags", timeout=10)
