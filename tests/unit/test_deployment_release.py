"""Exercise publication identity, monotonic latest tags and registry failure handling."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
import yaml

from tests.ci.deployment_release import (
    latest_stable_release,
    require_unused_chart_version,
    validate_chart_release,
    validate_promotion,
)


@pytest.mark.parametrize(
    ("candidate", "tags", "expected"),
    [
        ("v26.2.0", ["v26.1.9", "v26.2.0", "v26.3.0rc1"], True),
        ("v26.9.0", ["v26.9.0", "v26.10.0"], False),
        ("v26.2.0rc1", ["v26.1.0", "v26.2.0rc1"], False),
        ("v26.2.0-rc1", ["v26.1.0", "v26.2.0-rc1"], False),
        ("v26.2.0", ["v26.1.0"], False),
        ("v26.2.0", [], False),
        ("v26.1.0", ["v26.1.0", "boxmot-chart-100.0.0", "v026.2.0"], True),
    ],
)
def test_latest_tag_only_advances_to_an_existing_stable_release(
    candidate: str, tags: list[str], expected: bool
) -> None:
    assert latest_stable_release(candidate, tags) is expected


def _git(directory: Path, *arguments: str) -> str:
    """Use temporary local repositories without involving a network remote."""
    return subprocess.check_output(["git", *arguments], cwd=directory, text=True).strip()


def test_chart_release_is_bound_to_tag_source_version_and_default_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(tmp_path, "init", "--initial-branch=master")
    _git(tmp_path, "config", "user.name", "Chart release test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    chart = tmp_path / "chart"
    chart.mkdir()
    (chart / "Chart.yaml").write_text('name: boxmot\nversion: 0.1.0\nappVersion: "26.0.0"\n')
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "Chart release")
    sha = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "update-ref", "refs/remotes/origin/master", sha)
    _git(tmp_path, "tag", "boxmot-chart-0.1.0")
    monkeypatch.chdir(tmp_path)

    assert validate_chart_release("boxmot-chart-0.1.0", sha, "master", chart) == {
        "chart_version": "0.1.0",
        "app_version": "26.0.0",
        "source_sha": sha,
    }
    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        validate_chart_release("boxmot-chart-0.1.0", "HEAD", "master", chart)
    with pytest.raises(ValueError, match="checked-out commit"):
        validate_chart_release("boxmot-chart-0.1.0", "f" * 40, "master", chart)
    (chart / "Chart.yaml").write_text('name: boxmot\nversion: 0.2.0\nappVersion: "26.0.0"\n')
    with pytest.raises(ValueError, match="name/version"):
        validate_chart_release("boxmot-chart-0.1.0", sha, "master", chart)
    _git(tmp_path, "commit", "-am", "Another chart version")
    newer_sha = _git(tmp_path, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="tag does not point"):
        validate_chart_release("boxmot-chart-0.1.0", newer_sha, "master", chart)
    _git(tmp_path, "tag", "boxmot-chart-0.2.0")
    with pytest.raises(subprocess.CalledProcessError):
        validate_chart_release("boxmot-chart-0.2.0", newer_sha, "master", chart)


@pytest.mark.parametrize("status", [200, 404, 401, 403, 429, 500])
def test_chart_registry_check_only_allows_a_confirmed_missing_version(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    requests = []

    def open_registry(request, **kwargs):
        requests.append(request)
        assert kwargs["timeout"] == 30
        if len(requests) == 1:
            return io.BytesIO(json.dumps({"token": "test-token"}).encode())
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "test status", {}, None)
        return io.BytesIO(b"")

    monkeypatch.setattr("tests.ci.deployment_release.urllib.request.urlopen", open_registry)
    if status == 404:
        require_unused_chart_version("test-owner", "0.1.0", "actor", "credential")
    elif status == 200:
        with pytest.raises(ValueError, match="already exists"):
            require_unused_chart_version("test-owner", "0.1.0", "actor", "credential")
    else:
        with pytest.raises(urllib.error.HTTPError) as error:
            require_unused_chart_version("test-owner", "0.1.0", "actor", "credential")
        assert error.value.code == status
    assert requests[1].method == "HEAD"
    assert requests[1].full_url == "https://ghcr.io/v2/test-owner/charts/boxmot/manifests/0.1.0"
    assert requests[1].get_header("Authorization") == "Bearer test-token"


def _evidence() -> dict:
    """Represent the successful harness artifact consumed by promotion."""
    return {
        "schema_version": 1,
        "scenario": "recovery",
        "camera_image": "boxmot/boxmot@sha256:" + "a" * 64,
        "vllm_image": "vllm/vllm-openai-cpu@sha256:" + "b" * 64,
        "model_revision": "c" * 40,
        "chart_version": "0.1.0",
        "app_version": "26.0.0",
        "runtime_app_version": "26.0.0",
        "source_sha": "d" * 40,
        "validation_run_id": "123",
        "validation_run_attempt": "2",
        "external_vllm": False,
    }


def test_promotion_preserves_the_exact_tested_image_identity() -> None:
    evidence = _evidence()
    assert validate_promotion(evidence, run_id="123", run_attempt="2", source_sha="d" * 40) == evidence


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 2),
        ("scenario", "capacity"),
        ("scenario", None),
        ("camera_image", "boxmot/boxmot:latest-cpu"),
        ("camera_image", "someone/boxmot@sha256:" + "a" * 64),
        ("vllm_image", "vllm/vllm-openai-cpu:latest"),
        ("model_revision", "main"),
        ("external_vllm", True),
        ("external_vllm", None),
        ("chart_version", "latest"),
        ("app_version", ""),
        ("runtime_app_version", "25.0.0"),
        ("runtime_app_version", None),
        ("validation_run_id", "999"),
        ("validation_run_attempt", "1"),
        ("source_sha", "e" * 40),
    ],
)
def test_promotion_rejects_mutable_images_and_mismatched_test_evidence(key: str, value: object) -> None:
    evidence = {**_evidence(), key: value}
    with pytest.raises(ValueError):
        validate_promotion(evidence, run_id="123", run_attempt="2", source_sha="d" * 40)


@pytest.mark.parametrize("path", [".github/workflows/helm.yml", ".github/workflows/helm.yml@refs/heads/master"])
def test_promotion_accepts_github_workflow_paths_with_optional_ref(tmp_path: Path, path: str) -> None:
    """Run the API verification shell using a recorded successful workflow response."""
    binary = tmp_path / "gh"
    binary.write_text(f"#!{sys.executable}\nimport os\nprint(os.environ['RUN_RESPONSE'])\n")
    binary.chmod(0o755)
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/workflows/promote-images.yml").read_text()
    )
    script = next(step["run"] for step in workflow["jobs"]["promote"]["steps"] if step.get("id") == "validation")
    output = tmp_path / "output"
    run = {
        "path": path,
        "conclusion": "success",
        "status": "completed",
        "event": "workflow_dispatch",
        "head_repository": {"full_name": "example/boxmot"},
        "head_sha": "d" * 40,
        "run_attempt": 2,
    }
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "VALIDATION_RUN_ID": "123",
            "GITHUB_REPOSITORY": "example/boxmot",
            "GITHUB_OUTPUT": str(output),
            "RUN_RESPONSE": json.dumps(run),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"source_sha={'d' * 40}\nrun_attempt=2\n"
