"""Static release-contract tests for the v24 Docker workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.ci import release_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker.yml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
PYPROJECT = REPO_ROOT / "pyproject.toml"
WHEEL_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "wheels.yml"
CHECKOUT_ACTION = REPO_ROOT / ".github" / "actions" / "checkout-release-source" / "action.yml"


def _docker_job() -> dict:
    workflow = yaml.safe_load(DOCKER_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["build-and-push"]


def _docker_stage(name: str) -> str:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    marker = f" AS {name}\n"
    return dockerfile.split(marker, 1)[1].split("\nFROM ", 1)[0]


def _selected_images(tmp_path: Path, enabled: str | None) -> list[dict]:
    """Execute the workflow's matrix selection without scheduling any runners."""
    workflow = yaml.safe_load(DOCKER_WORKFLOW.read_text(encoding="utf-8"))
    selection = workflow["jobs"]["prepare-matrix"]
    step = next(step for step in selection["steps"] if step.get("id") == "images")
    assert selection["runs-on"] == "ubuntu-latest"
    assert step["env"]["ENABLE_GPU_SERVICE"] == "${{ vars.BOXMOT_GPU_SERVICE_CI }}"
    env = {key: value for key, value in os.environ.items() if key != "ENABLE_GPU_SERVICE"}
    if enabled is not None:
        env["ENABLE_GPU_SERVICE"] = enabled
    output = tmp_path / "matrix-output"
    env["GITHUB_OUTPUT"] = str(output)
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    name, _, value = output.read_text(encoding="utf-8").strip().partition("=")
    assert name == "matrix"
    return json.loads(value)["include"]


@pytest.mark.parametrize("enabled", (None, "", "false"))
def test_docker_defaults_schedule_only_available_hosted_runners(tmp_path: Path, enabled: str | None) -> None:
    matrix = _selected_images(tmp_path, enabled)
    assert {entry["target"] for entry in matrix} == {"cli-gpu", "cli-cpu", "service-cpu"}
    assert {entry["runner"] for entry in matrix} == {"ubuntu-latest"}


def test_enabled_gpu_matrix_covers_every_image_with_bounded_jobs(tmp_path: Path) -> None:
    job = _docker_job()
    matrix = _selected_images(tmp_path, "true")

    assert job["needs"] == "prepare-matrix"
    assert job["strategy"]["matrix"] == "${{ fromJSON(needs.prepare-matrix.outputs.matrix) }}"
    assert job["timeout-minutes"] == "${{ matrix.timeout_minutes }}"
    assert {(entry["target"], entry["repository"], entry["tag_suffix"]) for entry in matrix} == {
        ("cli-gpu", "boxmot/boxmot", ""),
        ("cli-cpu", "boxmot/boxmot", "-cpu"),
        ("service-cpu", "boxmot/boxmot-service", ""),
        ("service-gpu", "boxmot/boxmot-service", "-gpu"),
    }
    assert all(isinstance(entry["timeout_minutes"], int) and entry["timeout_minutes"] > 0 for entry in matrix)
    gpu_service = next(entry for entry in matrix if entry["target"] == "service-gpu")
    assert gpu_service["runner"] == "gpu-latest"
    assert gpu_service["timeout_minutes"] >= 120


def test_manual_image_rebuild_is_explicit_and_safe_by_default() -> None:
    workflow = DOCKER_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "description: 'Commit SHA to rebuild and smoke-test'" in workflow
    assert "description: 'Exact v-prefixed release tag to validate'" in workflow
    assert "push_images:\n        description: 'Push validated images to Docker Hub'" in workflow
    assert "type: boolean\n        default: false" in workflow
    action = yaml.safe_load(CHECKOUT_ACTION.read_text(encoding="utf-8"))
    checkout = next(step for step in action["runs"]["steps"] if step.get("uses") == "actions/checkout@v4")
    assert checkout["with"]["fetch-depth"] == 0
    assert "Manual publishing requires $RELEASE_TAG to point at $source_sha" in workflow


def test_every_docker_image_checks_the_exact_v24_surface() -> None:
    job = _docker_job()
    step = next(step for step in job["steps"] if step.get("name") == "Smoke test release package and command surface")
    script = step["run"]

    assert "if:" not in step
    assert 'python "${python_args[@]}" --expected-version "${{ env.VERSION }}" < tests/ci/release_contract.py' in script
    assert "--workdir /tmp" in script


@pytest.mark.parametrize("target", ("cli-cpu", "cli-gpu", "service-cpu", "service-gpu"))
def test_docker_release_smoke_preserves_each_images_import_context(tmp_path: Path, target: str) -> None:
    """Execute the workflow's Python flags against a source-only package fixture."""
    step = next(
        step for step in _docker_job()["steps"] if step.get("name") == "Smoke test release package and command surface"
    )
    script = step["run"]
    for expression, value in (
        ("${{ matrix.target }}", target),
        ("${{ matrix.repository }}", "boxmot/example"),
        ("${{ env.VERSION }}", "24.0.0"),
        ("${{ matrix.tag_suffix }}", ""),
    ):
        script = script.replace(expression, value)

    source = tmp_path / "service-source"
    package = source / "boxmot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("source_only_service = True\n", encoding="utf-8")
    check = tmp_path / "tests" / "ci" / "release_contract.py"
    check.parent.mkdir(parents=True)
    check.write_text(
        "import sys\nimport boxmot\n"
        + (
            "assert sys.flags.isolated == 0\nassert boxmot.source_only_service is True\n"
            "assert '--check-trackers' not in sys.argv\nassert '--check-cli-help' not in sys.argv\n"
            if target.startswith("service-")
            else "assert sys.flags.isolated == 1\nassert not hasattr(boxmot, 'source_only_service')\n"
            "assert '--check-trackers' in sys.argv\nassert '--check-cli-help' in sys.argv\n"
        ),
        encoding="utf-8",
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os\nimport sys\n"
        "arguments = sys.argv[sys.argv.index('python') + 1:]\n"
        "os.execv(sys.executable, [sys.executable, *arguments])\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}", "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_wheel_smoke_uses_shared_checks_without_importing_the_checkout() -> None:
    workflow = yaml.safe_load(WHEEL_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["clean_wheel_smoke"]["steps"]
    checkout = next(step for step in steps if step.get("name") == "Checkout release checks")
    smoke = next(step for step in steps if step.get("name") == "Smoke-test release imports and CLI")

    assert checkout["uses"] == "./.github/actions/checkout-release-source"
    assert checkout["with"] == {
        "source_sha": "${{ inputs.source_sha || github.sha }}",
        "candidate_sha": "${{ inputs.candidate_sha }}",
        "source_bundle": "${{ inputs.source_bundle }}",
        "path": "source",
    }
    assert smoke["working-directory"] == "${{ runner.temp }}"
    assert smoke["env"]["RELEASE_VERSION"] == "${{ needs.build.outputs.version }}"
    assert 'python -I "$GITHUB_WORKSPACE/source/tests/ci/release_contract.py"' in smoke["run"]
    assert '--expected-version "$RELEASE_VERSION" --check-cli-help --check-trackers' in smoke["run"]
    assert "expected_public_api" not in smoke["run"]
    assert "expected_cli_commands" not in smoke["run"]


def test_release_contract_discovers_commands_without_populating_the_lazy_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from boxmot.engine.cli import boxmot as boxmot_cli

    monkeypatch.setattr(boxmot_cli, "commands", {})

    release_contract.check_release_contract()

    assert boxmot_cli.commands == {}


def test_release_contract_rejects_a_missing_time_variant_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from boxmot.engine.cli import boxmot as boxmot_cli

    monkeypatch.setattr(
        boxmot_cli,
        "list_commands",
        lambda _context: [name for name in release_contract.EXPECTED_CLI_COMMANDS if name != "time-variant"],
    )

    with pytest.raises(AssertionError, match="CLI commands:.*time-variant"):
        release_contract.check_release_contract()


def test_release_contract_rejects_a_changed_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import boxmot

    monkeypatch.setattr(boxmot, "__all__", (*boxmot.__all__, "UnexpectedExport"))

    with pytest.raises(AssertionError, match="Public API:.*UnexpectedExport"):
        release_contract.check_release_contract()


def test_release_contract_rejects_an_unexpected_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    import boxmot

    monkeypatch.setattr(boxmot, "__version__", "0.0.0")

    with pytest.raises(AssertionError, match="Package version:.*0.0.0"):
        release_contract.check_release_contract()


@pytest.mark.parametrize("expected_version", ("25.0.0", "24.1.0", "24.0.1"))
def test_release_contract_accepts_requested_version_without_installed_metadata(
    monkeypatch: pytest.MonkeyPatch, expected_version: str
) -> None:
    """Source-only service images validate the independently prepared release."""
    import boxmot

    def missing_metadata(_name: str) -> str:
        raise release_contract.importlib.metadata.PackageNotFoundError("boxmot")

    monkeypatch.setattr(boxmot, "__version__", expected_version)
    monkeypatch.setattr(release_contract.importlib.metadata, "version", missing_metadata)

    release_contract.check_release_contract(expected_version=expected_version)


def test_release_contract_rejects_mismatched_requested_and_installed_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    import boxmot

    monkeypatch.setattr(boxmot, "__version__", "24.0.1")
    monkeypatch.setattr(release_contract.importlib.metadata, "version", lambda _name: "24.0.0")

    with pytest.raises(AssertionError, match="Package version:.*24.0.0.*24.0.1"):
        release_contract.check_release_contract()
    with pytest.raises(AssertionError, match="Package version:.*25.0.0.*24.0.1"):
        release_contract.check_release_contract(expected_version="25.0.0")


def test_release_cli_help_checks_every_command_and_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(release_contract.subprocess, "run", run)
    release_contract.check_cli_help()
    assert calls == [["boxmot", name, "--help"] for name in release_contract.EXPECTED_CLI_COMMANDS]

    monkeypatch.setattr(
        release_contract.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="cannot load adapter"),
    )
    with pytest.raises(AssertionError, match="cannot load adapter"):
        release_contract.check_cli_help()


def test_service_gpu_smoke_still_executes_cuda_reid_request() -> None:
    job = _docker_job()
    step = next(step for step in job["steps"] if step.get("name") == "Smoke test GPU service image")
    script = step["run"]

    assert "--gpus all" in script
    assert "torch.cuda.is_available()" in script
    assert "/v1/streams/smoke-camera/sessions/smoke-run/frames" in script
    assert "active CUDA context after ReID inference" in script


def test_cli_images_smoke_prebuilt_native_tracker_without_toolchain() -> None:
    job = _docker_job()
    step = next(step for step in job["steps"] if step.get("name") == "Smoke test prebuilt native tracker")
    script = step["run"]

    assert step["if"] == "matrix.target == 'cli-cpu' || matrix.target == 'cli-gpu'"
    assert "python -I -" in script
    assert 'shutil.which("cmake") is None' in script
    assert 'shutil.which("g++") is None' in script
    assert 'for component in ("reid", "botsort", "bytetrack", "occluboost", "ocsort", "sfsort")' in script
    assert "ctypes.CDLL(str(path))" in script
    assert "ensure_bytetrack_cpp_library()" in script
    assert 'TrackerSpec(name="bytetrack", backend="cpp"' in script
    assert "tracker.update(detections)" in script


def test_cpu_service_uses_its_locked_cpu_torch_group() -> None:
    service_builder = _docker_stage("service-cpu-builder")
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert "--only-group service-runtime" in service_builder
    assert "--extra cpu" not in service_builder
    assert '{ index = "pytorch-cpu", group = "service-runtime" }' in pyproject
    assert '{ group = "service-runtime" },\n    { extra = "cu130" },' in pyproject

    job = _docker_job()
    step = next(step for step in job["steps"] if step.get("name") == "Smoke test CPU service image")
    assert "torch.version.cuda is None" in step["run"]
    assert 'name.startswith("nvidia-")' in step["run"]
    assert '{"cuda-bindings", "cuda-toolkit", "triton"}' in step["run"]


def test_cli_images_package_native_libraries_without_runtime_build_tools() -> None:
    native_builder = _docker_stage("native-cli-builder")
    cli_runtime = _docker_stage("cli-runtime")
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    for dependency in ("cmake", "libeigen3-dev", "libopencv-dev", "ninja-build"):
        assert dependency in native_builder
        assert dependency not in cli_runtime
    assert "-DBOXMOT_INSTALL_NATIVE=ON" in native_builder
    assert dockerfile.count("COPY --from=native-cli-builder") == 2
    assert dockerfile.count("ctypes.CDLL") == 2
    for runtime_library in (
        "libopencv-calib3d406",
        "libopencv-core406",
        "libopencv-dnn406",
        "libopencv-imgproc406",
        "libopencv-video406",
    ):
        assert runtime_library in cli_runtime


def test_release_contract_rejects_missing_packaged_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    import boxmot.engine.experiment_config as experiment_config

    def missing_config(_name: str) -> dict:
        raise FileNotFoundError("Packaged experiment is missing")

    monkeypatch.setattr(experiment_config, "resolve_experiment_config", missing_config)
    with pytest.raises(FileNotFoundError, match="Packaged experiment"):
        release_contract.check_release_contract()
