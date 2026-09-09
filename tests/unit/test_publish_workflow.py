"""Exercise release publication against local Git remotes and package-upload stubs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/publish.yml"
DESTINATION_STEP = "Check the destination immediately before publication"
PUBLISH_STEP = "Publish the gated distributions"
PUSH_STEP = "Push the successful version commit and tag"
RELEASE_STEP = "Create code release at the published commit"
PYPI_SUCCESS = "${{ success() && inputs.pypi == 'pypi' }}"


def _workflow() -> dict:
    """Load the actual workflow, including shell commands exercised below."""
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _git(directory: Path, *arguments: str) -> str:
    """Execute Git only against repositories prepared inside the test directory."""
    return subprocess.run(
        ["git", *arguments], cwd=directory, capture_output=True, text=True, check=True
    ).stdout.strip()


@dataclass
class ReleaseRepository:
    """A candidate checkout, its local origin, and stubbed external publishers."""

    checkout: Path
    origin: Path
    log: Path
    env: dict[str, str]

    def remote_ref(self, ref: str) -> str:
        """Return an advertised remote ref, or an empty string when it is absent."""
        output = _git(self.checkout, "ls-remote", "origin", ref)
        return output.split()[0] if output else ""

    def events(self) -> list[dict]:
        """Read calls recorded by the publish, push, and release executables."""
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def run_publication(self) -> list[subprocess.CompletedProcess[str]]:
        """Run real workflow shell steps with GitHub's success and target gating."""
        steps = _workflow()["jobs"]["pypi-upload"]["steps"]
        start = next(index for index, step in enumerate(steps) if step["name"] == DESTINATION_STEP)
        results = []
        for step in steps[start:]:
            if condition := step.get("if"):
                assert condition == PYPI_SUCCESS
                if self.env["TARGET_REPOSITORY"] != "pypi":
                    continue
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
                cwd=self.checkout,
                env=self.env,
                capture_output=True,
                text=True,
                check=False,
            )
            results.append(result)
            if result.returncode:
                break
        return results


@pytest.fixture
def release_repo(tmp_path: Path) -> ReleaseRepository:
    """Create an unpublished version commit and intercept all nonlocal actions."""
    origin = tmp_path / "origin.git"
    checkout = tmp_path / "candidate"
    _git(tmp_path, "init", "--bare", "--initial-branch=master", str(origin))
    _git(tmp_path, "clone", str(origin), str(checkout))
    _git(checkout, "config", "user.email", "release-test@example.invalid")
    _git(checkout, "config", "user.name", "Release test")
    _git(checkout, "config", "commit.gpgsign", "false")
    (checkout / "version.txt").write_text("24.0.0\n")
    _git(checkout, "add", "version.txt")
    _git(checkout, "commit", "-m", "base")
    _git(checkout, "push", "origin", "HEAD:refs/heads/master")
    base_sha = _git(checkout, "rev-parse", "HEAD")
    (checkout / "version.txt").write_text("24.0.1\n")
    _git(checkout, "commit", "-am", "candidate")
    candidate_sha = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "checkout", "-b", "concurrent", base_sha)
    (checkout / "concurrent.txt").write_text("An unrelated branch update.\n")
    _git(checkout, "add", "concurrent.txt")
    _git(checkout, "commit", "-m", "concurrent change")
    race_sha = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "checkout", "--detach", candidate_sha)
    (checkout / "dist").mkdir()
    (checkout / "dist/boxmot-24.0.1-py3-none-any.whl").touch()
    (checkout / "dist/boxmot-24.0.1.tar.gz").touch()

    binaries = tmp_path / "bin"
    binaries.mkdir()
    stub = f"#!{sys.executable}\n" + '''\
import json
import os
from pathlib import Path
import subprocess
import sys

command = Path(sys.argv[0]).name
arguments = sys.argv[1:]
real_git = os.environ["REAL_GIT"]
if command != "git" or arguments[0] == "push":
    with open(os.environ["RELEASE_TEST_LOG"], "a") as log:
        log.write(json.dumps({"command": command, "arguments": arguments}) + "\\n")
if command == "git":
    os.execv(real_git, [real_git, *arguments])
elif command == "uv":
    if os.environ.get("FAIL_PUBLICATION"):
        print("Simulated package-upload failure", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("RACE_DURING_PUBLICATION"):
        subprocess.run(
            [real_git, "push", "origin", os.environ["RACE_SHA"] + ":refs/heads/master"],
            check=True,
        )
elif command == "gh":
    tag = "refs/tags/" + arguments[2]
    published_tag = subprocess.check_output([real_git, "ls-remote", "origin", tag], text=True)
    assert published_tag.split()[0] == os.environ["CANDIDATE_SHA"], published_tag
else:
    raise AssertionError(command)
'''
    for name in ("git", "uv", "gh"):
        executable = binaries / name
        executable.write_text(stub, encoding="utf-8")
        executable.chmod(0o755)
    log = tmp_path / "events.jsonl"
    real_git = shutil.which("git")
    assert real_git is not None
    env = {
        **os.environ,
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "REAL_GIT": real_git,
        "RELEASE_TEST_LOG": str(log),
        "BASE_SHA": base_sha,
        "CANDIDATE_SHA": candidate_sha,
        "RACE_SHA": race_sha,
        "SOURCE_BRANCH": "master",
        "RELEASE_VERSION": "24.0.1",
        "TARGET_REPOSITORY": "pypi",
        "RELEASE_TOKEN": "test-release-token",
        "PYPI_TOKEN": "test-pypi-token",
        "TEST_PYPI_TOKEN": "test-testpypi-token",
    }
    return ReleaseRepository(checkout, origin, log, env)


def test_release_dispatch_and_gates_share_one_unpublished_candidate() -> None:
    workflow = _workflow()
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["bump_type"]["options"] == ["patch", "minor", "major"]
    assert inputs["bump_type"]["default"] == "patch"
    assert inputs["pypi"]["options"] == ["pypi", "testpypi"]
    assert workflow["concurrency"] == {
        "group": "pypi-release-${{ inputs.pypi }}",
        "cancel-in-progress": False,
    }

    jobs = workflow["jobs"]
    assert jobs["release-gates"]["needs"] == "prepare-release"
    assert set(jobs["docker-gates"]["needs"]) == {"prepare-release", "release-gates"}
    assert set(jobs["pypi-upload"]["needs"]) == {"prepare-release", "release-gates", "docker-gates"}
    candidate_sha = "${{ needs.prepare-release.outputs.candidate_sha }}"
    assert jobs["release-gates"]["with"]["candidate_sha"] == candidate_sha
    assert jobs["docker-gates"]["with"]["source_ref"] == candidate_sha
    assert jobs["docker-gates"]["with"]["push_images"] is False
    assert jobs["pypi-upload"]["env"]["CANDIDATE_SHA"] == candidate_sha
    steps = jobs["pypi-upload"]["steps"]
    assert [step["name"] for step in steps[-4:]] == [DESTINATION_STEP, PUBLISH_STEP, PUSH_STEP, RELEASE_STEP]
    assert all(step["if"] == PYPI_SUCCESS for step in steps[-2:])
    assert all("git push" not in step.get("run", "") for step in jobs["prepare-release"]["steps"])


def test_successful_pypi_upload_precedes_atomic_commit_tag_push_and_release(release_repo: ReleaseRepository) -> None:
    assert release_repo.remote_ref("refs/heads/master") == release_repo.env["BASE_SHA"]
    results = release_repo.run_publication()

    assert len(results) == 4
    assert all(result.returncode == 0 for result in results), results
    assert release_repo.remote_ref("refs/heads/master") == release_repo.env["CANDIDATE_SHA"]
    assert release_repo.remote_ref("refs/tags/v24.0.1") == release_repo.env["CANDIDATE_SHA"]
    events = release_repo.events()
    assert [event["command"] for event in events] == ["uv", "git", "gh"]
    assert events[0]["arguments"] == [
        "publish", "dist/boxmot-24.0.1-py3-none-any.whl", "dist/boxmot-24.0.1.tar.gz"
    ]
    assert events[1]["arguments"] == [
        "push", "--atomic", "origin", f"{release_repo.env['CANDIDATE_SHA']}:refs/heads/master", "refs/tags/v24.0.1"
    ]
    assert events[2]["arguments"][:4] == ["release", "create", "v24.0.1", "--verify-tag"]


def test_testpypi_upload_does_not_change_branch_tag_or_create_release(release_repo: ReleaseRepository) -> None:
    release_repo.env["TARGET_REPOSITORY"] = "testpypi"
    release_repo.env["RELEASE_TOKEN"] = ""
    results = release_repo.run_publication()

    assert len(results) == 2
    assert all(result.returncode == 0 for result in results), results
    assert release_repo.remote_ref("refs/heads/master") == release_repo.env["BASE_SHA"]
    assert release_repo.remote_ref("refs/tags/v24.0.1") == ""
    assert release_repo.events() == [{
        "command": "uv",
        "arguments": [
            "publish", "--index", "testpypi", "dist/boxmot-24.0.1-py3-none-any.whl", "dist/boxmot-24.0.1.tar.gz"
        ],
    }]


def test_failed_upload_leaves_branch_and_tag_unchanged(release_repo: ReleaseRepository) -> None:
    release_repo.env["FAIL_PUBLICATION"] = "1"
    results = release_repo.run_publication()

    assert len(results) == 2
    assert results[-1].returncode != 0
    assert "Simulated package-upload failure" in results[-1].stderr
    assert [event["command"] for event in release_repo.events()] == ["uv"]
    assert release_repo.remote_ref("refs/heads/master") == release_repo.env["BASE_SHA"]
    assert release_repo.remote_ref("refs/tags/v24.0.1") == ""


@pytest.mark.parametrize("obstruction", ("branch-moved", "existing-tag", "missing-token"))
def test_unready_destination_is_rejected_before_upload(release_repo: ReleaseRepository, obstruction: str) -> None:
    expected_branch = release_repo.env["BASE_SHA"]
    expected_tag = ""
    if obstruction == "branch-moved":
        _git(release_repo.checkout, "push", "origin", f"{release_repo.env['RACE_SHA']}:refs/heads/master")
        expected_branch = release_repo.env["RACE_SHA"]
    elif obstruction == "existing-tag":
        _git(release_repo.checkout, "push", "origin", f"{expected_branch}:refs/tags/v24.0.1")
        expected_tag = expected_branch
    else:
        release_repo.env["RELEASE_TOKEN"] = ""

    results = release_repo.run_publication()

    assert len(results) == 1
    assert results[0].returncode != 0
    assert release_repo.events() == []
    assert release_repo.remote_ref("refs/heads/master") == expected_branch
    assert release_repo.remote_ref("refs/tags/v24.0.1") == expected_tag


@pytest.mark.parametrize("obstruction", ("branch-race", "rejected-push"))
def test_push_failure_after_upload_preserves_remote_and_reports_recovery(
    release_repo: ReleaseRepository, obstruction: str
) -> None:
    expected_branch = release_repo.env["BASE_SHA"]
    if obstruction == "branch-race":
        release_repo.env["RACE_DURING_PUBLICATION"] = "1"
        expected_branch = release_repo.env["RACE_SHA"]
    else:
        hook = release_repo.origin / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\necho 'Repository policy rejected release push' >&2\nexit 1\n")
        hook.chmod(0o755)

    results = release_repo.run_publication()

    assert len(results) == 3
    assert results[-1].returncode != 0
    assert "PyPI publication succeeded" in results[-1].stdout
    assert "Recover the tested candidate from the release-source artifact" in results[-1].stdout
    assert "do not rebuild or force-push" in results[-1].stdout
    assert [event["command"] for event in release_repo.events()] == ["uv", "git"]
    assert release_repo.remote_ref("refs/heads/master") == expected_branch
    assert release_repo.remote_ref("refs/tags/v24.0.1") == ""
