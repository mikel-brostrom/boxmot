"""Exercise unpublished release checkout using real local Git bundles."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_PATH = REPO_ROOT / ".github/actions/checkout-release-source/action.yml"


def _action() -> dict:
    """Load the same action scripts executed by the reusable release gates."""
    return yaml.safe_load(ACTION_PATH.read_text(encoding="utf-8"))


def _git(path: Path, *arguments: str) -> str:
    """Run Git in an isolated fixture repository."""
    return subprocess.run(["git", *arguments], cwd=path, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def candidate(tmp_path: Path) -> SimpleNamespace:
    """Clone the published base before preparing an unpushed candidate commit."""
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "Release Test")
    _git(source, "config", "user.email", "release-test@example.invalid")
    _git(source, "config", "commit.gpgsign", "false")
    _git(source, "config", "core.hooksPath", "/dev/null")
    (source / "version.txt").write_text("24.0.0\n", encoding="utf-8")
    _git(source, "add", "version.txt")
    _git(source, "commit", "-m", "base")
    base_sha = _git(source, "rev-parse", "HEAD")
    checkout = tmp_path / "checkout with spaces"
    _git(tmp_path, "clone", "--no-local", str(source), str(checkout))

    _git(source, "checkout", "-b", "release-candidate")
    (source / "version.txt").write_text("24.0.1\n", encoding="utf-8")
    _git(source, "commit", "-am", "release candidate")
    candidate_sha = _git(source, "rev-parse", "HEAD")
    bundle = tmp_path / "bundle with spaces"
    bundle.mkdir()
    _git(source, "bundle", "create", str(bundle / "release-source.bundle"), "release-candidate", f"^{base_sha}")
    return SimpleNamespace(source=source, checkout=checkout, base=base_sha, sha=candidate_sha, bundle=bundle)


def _select_source(
    candidate: SimpleNamespace, *, expected: str | None = None, bundle: bool = True
) -> subprocess.CompletedProcess:
    """Execute the composite's actual selection script with GitHub's Bash flags."""
    step = next(step for step in _action()["runs"]["steps"] if step.get("id") == "source")
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=candidate.checkout,
        env={
            **os.environ,
            "CANDIDATE_SHA": (candidate.sha if expected is None else expected) if bundle else "",
            "BUNDLE_DIRECTORY": str(candidate.bundle) if bundle else "",
            "GITHUB_OUTPUT": str(candidate.checkout.parent / "source-output"),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_candidate_bundle_checks_out_the_exact_unpublished_commit(candidate: SimpleNamespace) -> None:
    missing = subprocess.run(
        ["git", "cat-file", "-e", candidate.sha], cwd=candidate.checkout, capture_output=True, check=False
    )
    assert missing.returncode != 0

    result = _select_source(candidate)

    assert result.returncode == 0, result.stderr
    assert _git(candidate.checkout, "rev-parse", "HEAD") == candidate.sha
    assert (candidate.checkout / "version.txt").read_text() == "24.0.1\n"
    assert _git(candidate.checkout, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert (candidate.checkout.parent / "source-output").read_text() == f"sha={candidate.sha}\n"
    assert _git(candidate.checkout, "rev-parse", "origin/main") == candidate.base


def test_standalone_checkout_keeps_the_published_revision(candidate: SimpleNamespace) -> None:
    result = _select_source(candidate, bundle=False)

    assert result.returncode == 0, result.stderr
    assert _git(candidate.checkout, "rev-parse", "HEAD") == candidate.base
    assert (candidate.checkout / "version.txt").read_text() == "24.0.0\n"


def test_candidate_bundle_rejects_a_different_commit_before_checkout(candidate: SimpleNamespace) -> None:
    result = _select_source(candidate, expected=candidate.base)

    assert result.returncode != 0
    assert f"expected {candidate.base}" in result.stderr
    assert _git(candidate.checkout, "rev-parse", "HEAD") == candidate.base


def test_candidate_bundle_rejects_an_unrelated_history(candidate: SimpleNamespace) -> None:
    _git(candidate.source, "checkout", "--orphan", "unrelated")
    _git(candidate.source, "commit", "-m", "unrelated source")
    candidate.sha = _git(candidate.source, "rev-parse", "HEAD")
    _git(candidate.source, "update-ref", "refs/heads/release-candidate", candidate.sha)
    _git(candidate.source, "bundle", "create", str(candidate.bundle / "release-source.bundle"), "release-candidate")

    result = _select_source(candidate)

    assert result.returncode != 0
    assert "does not descend from base" in result.stderr
    assert _git(candidate.checkout, "rev-parse", "HEAD") == candidate.base


@pytest.mark.parametrize(("candidate_sha", "artifact"), (("abc", ""), ("", "release-source")))
def test_candidate_inputs_must_be_supplied_together(candidate_sha: str, artifact: str) -> None:
    step = next(step for step in _action()["runs"]["steps"] if step.get("name") == "Validate candidate inputs")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env={**os.environ, "CANDIDATE_SHA": candidate_sha, "SOURCE_BUNDLE": artifact},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must be supplied together" in result.stderr


def test_checkout_action_keeps_credentials_and_downloads_outside_the_source_tree() -> None:
    action = _action()
    checkout = next(step for step in action["runs"]["steps"] if step.get("uses") == "actions/checkout@v4")
    allocation = next(step for step in action["runs"]["steps"] if step.get("id") == "bundle-directory")
    download = next(step for step in action["runs"]["steps"] if step.get("uses") == "actions/download-artifact@v4")

    assert action["inputs"]["token"]["default"] == "${{ github.token }}"
    assert checkout["with"] == {
        "ref": "${{ inputs.source_sha }}",
        "fetch-depth": 0,
        "path": "${{ inputs.path }}",
        "token": "${{ inputs.token }}",
    }
    assert 'mktemp -d "$RUNNER_TEMP/boxmot-release-source.XXXXXX"' in allocation["run"]
    assert download["with"] == {
        "name": "${{ inputs.source_bundle }}",
        "path": "${{ steps.bundle-directory.outputs.path }}",
    }


def test_reusable_wheel_jobs_all_select_the_same_candidate_source() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/wheels.yml").read_text())
    for job in ("source_gates", "build", "clean_wheel_smoke"):
        steps = workflow["jobs"][job]["steps"]
        assert steps[0]["uses"] == "actions/checkout@v4"
        assert steps[0]["with"]["ref"] == "${{ github.workflow_sha }}"
        selection = steps[1]
        assert selection["uses"] == "./.github/actions/checkout-release-source"
        assert selection["with"]["source_sha"] == "${{ inputs.source_sha || github.sha }}"
        assert selection["with"]["candidate_sha"] == "${{ inputs.candidate_sha }}"
        assert selection["with"]["source_bundle"] == "${{ inputs.source_bundle }}"
    build_script = next(
        step["run"] for step in workflow["jobs"]["build"]["steps"] if step.get("id") == "release-version"
    )
    assert 'metadata["Version"] == version' in build_script
    assert "module_version == version" in build_script
    assert "24.0.0" not in build_script


def test_reusable_docker_job_selects_candidate_or_published_source() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/docker.yml").read_text())
    steps = workflow["jobs"]["build-and-push"]["steps"]
    assert steps[0]["uses"] == "actions/checkout@v4"
    assert steps[0]["with"]["ref"] == "${{ github.workflow_sha }}"
    assert steps[1]["uses"] == "./.github/actions/checkout-release-source"
    assert steps[1]["with"] == {
        "source_sha": "${{ inputs.source_base || inputs.source_ref || github.event.release.tag_name }}",
        "candidate_sha": "${{ inputs.source_bundle && inputs.source_ref || '' }}",
        "source_bundle": "${{ inputs.source_bundle }}",
    }
