"""Check immutable image handoff and the release workflow's publication boundary."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.ci.docker_artifact import candidate_identity, record_candidate, verify_candidate

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ID = "sha256:" + "a" * 64
IDENTITY = candidate_identity("b" * 40, "26.0.1", "cli-cpu", "boxmot/boxmot:26.0.1-cpu")


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    """Represent the archived bytes that passed all container smoke checks."""
    (tmp_path / "image.tar.gz").write_bytes(b"saved Docker image bytes")
    record_candidate(tmp_path, IDENTITY, IMAGE_ID)
    return tmp_path


def test_candidate_round_trip_preserves_docker_image_identity(candidate: Path) -> None:
    assert verify_candidate(candidate, IDENTITY) == IMAGE_ID


@pytest.mark.parametrize("key", ("source_sha", "version", "target", "image", "schema_version"))
def test_candidate_cannot_be_promoted_as_another_release(candidate: Path, key: str) -> None:
    with pytest.raises(ValueError, match=key):
        verify_candidate(candidate, {**IDENTITY, key: "different"})


def test_candidate_archive_corruption_is_rejected_before_loading(candidate: Path) -> None:
    (candidate / "image.tar.gz").write_bytes(b"different Docker image")
    with pytest.raises(ValueError, match="checksum"):
        verify_candidate(candidate, IDENTITY)


@pytest.mark.parametrize("image_id", ("", "latest", None, "sha256:wrong"))
def test_invalid_candidate_image_identity_is_rejected(candidate: Path, image_id: object) -> None:
    path = candidate / "candidate.json"
    payload = json.loads(path.read_text())
    payload["image_id"] = image_id
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="image ID"):
        verify_candidate(candidate, IDENTITY)


def _docker_workflow() -> dict:
    return yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text())


def test_publication_restores_gated_images_and_never_rebuilds_them() -> None:
    workflow = _docker_workflow()
    assert "release" not in workflow.get("on", workflow.get(True))
    steps = workflow["jobs"]["build-and-push"]["steps"]
    build = next(step for step in steps if step.get("uses") == "docker/build-push-action@v6")
    save = next(step for step in steps if step.get("name") == "Save the image that passed every smoke test")
    verify = next(step for step in steps if step.get("name") == "Verify and load the exact candidate image")
    push = next(step for step in steps if step.get("name") == "Push ${{ matrix.target }} image tags")
    guard = next(step for step in steps if step.get("name") == "Require tested artifacts for publication")
    assert build["if"] == save["if"] == "inputs.candidate_run_id == ''"
    assert verify["if"] == "inputs.candidate_run_id != ''"
    assert guard["if"] == "inputs.push_images"
    assert steps.index(guard) < steps.index(build)
    assert steps.index(verify) < steps.index(save) < steps.index(push)
    assert steps.index(save) > max(i for i, step in enumerate(steps) if step.get("name", "").startswith("Smoke test"))
    assert 'test "$actual_id" = "$expected_id"' in verify["run"]
    assert 'latest --release-tag "$RELEASE_TAG"' in push["run"]
    publish = yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text())["jobs"]["docker-publish"]
    assert "pypi-upload" in publish["needs"] and "docker-gates" in publish["needs"]
    assert publish["if"] == "inputs.pypi == 'pypi'"
    assert publish["with"]["candidate_run_id"] == "${{ github.run_id }}"
    assert publish["with"]["push_images"] is True


@pytest.mark.parametrize("tamper", ("none", "archive", "loaded-image"))
def test_actual_restore_script_rejects_wrong_bytes_or_image_id(candidate: Path, tamper: str) -> None:
    """Run the real workflow shell with Docker stubbed; bad images never gain tags."""
    directory = candidate / "candidate-image"
    directory.mkdir()
    for name in ("candidate.json", "image.tar.gz"):
        (candidate / name).rename(directory / name)
    if tamper == "archive":
        (directory / "image.tar.gz").write_bytes(b"tampered")
    shutil.copyfile(ROOT / "tests/ci/docker_artifact.py", candidate / "docker_artifact.py")
    binaries = candidate / "bin"
    binaries.mkdir()
    executable = binaries / "docker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['DOCKER_CALLS']).open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    print(os.environ['ACTUAL_IMAGE_ID'])\n"
    )
    executable.chmod(0o755)
    steps = _docker_workflow()["jobs"]["build-and-push"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == "Verify and load the exact candidate image")
    script = script.replace("${{ matrix.repository }}", "boxmot/boxmot").replace("${{ matrix.tag_suffix }}", "-cpu")
    calls = candidate / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(candidate),
        "SOURCE_SHA": str(IDENTITY["source_sha"]),
        "VERSION": str(IDENTITY["version"]),
        "IMAGE_TARGET": str(IDENTITY["target"]),
        "IMAGE": str(IDENTITY["image"]),
        "DOCKER_CALLS": str(calls),
        "ACTUAL_IMAGE_ID": IMAGE_ID if tamper != "loaded-image" else "sha256:" + "c" * 64,
    }
    result = subprocess.run(["bash", "-eo", "pipefail", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0 if tamper == "none" else result.returncode != 0
    invocations = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    if tamper == "archive":
        assert invocations == []
    assert sum(call[0] == "tag" for call in invocations) == (2 if tamper == "none" else 0)


@pytest.mark.skipif(shutil.which("jq") is None, reason="Publication workflow uses jq")
@pytest.mark.parametrize(
    ("run_id", "event", "workflow", "accepted"),
    (
        ("123", "workflow_dispatch", ".github/workflows/publish.yml", True),
        ("123", "workflow_dispatch", ".github/workflows/publish.yml@main", True),
        ("123", "workflow_dispatch", ".github/workflows/docker.yml@refs/heads/master", True),
        ("123", "pull_request", ".github/workflows/docker.yml", False),
        ("123", "workflow_dispatch", ".github/workflows/other.yml", False),
        ("", "workflow_dispatch", ".github/workflows/docker.yml", False),
    ),
)
def test_publication_requires_a_trusted_artifact_workflow(
    tmp_path: Path, run_id: str, event: str, workflow: str, accepted: bool
) -> None:
    """Exercise the actual API gate, including GitHub's optional workflow ref suffix."""
    binary = tmp_path / "gh"
    binary.write_text(f"#!{sys.executable}\nimport os\nprint(os.environ['RUN_RESPONSE'])\n", encoding="utf-8")
    binary.chmod(0o755)
    steps = _docker_workflow()["jobs"]["build-and-push"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == "Require tested artifacts for publication")
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "CANDIDATE_RUN_ID": run_id,
            "GITHUB_REPOSITORY": "example/boxmot",
            "RUN_RESPONSE": json.dumps({"event": event, "path": workflow}),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is accepted, result.stderr
