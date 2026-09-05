"""Static release-contract tests for the v24 Docker workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker.yml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _docker_job() -> dict:
    workflow = yaml.safe_load(DOCKER_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["build-and-push"]


def _docker_stage(name: str) -> str:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    marker = f" AS {name}\n"
    return dockerfile.split(marker, 1)[1].split("\nFROM ", 1)[0]


def test_docker_matrix_covers_every_v24_image_with_bounded_jobs() -> None:
    job = _docker_job()
    matrix = job["strategy"]["matrix"]["include"]

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
    assert "fetch-depth: 0" in workflow
    assert "Manual publishing requires $RELEASE_TAG to point at $source_sha" in workflow


def test_every_docker_image_checks_the_exact_v24_surface() -> None:
    job = _docker_job()
    step = next(step for step in job["steps"] if step.get("name") == "Smoke test v24 package and command surface")
    script = step["run"]

    assert "if:" not in step
    assert 'boxmot.__version__ == "24.0.0"' in script
    assert "boxmot.__all__ == expected_public_api" in script
    assert "tuple(boxmot_cli.commands) == expected_cli_commands" in script
    for command in (
        "track",
        "materialize",
        "eval",
        "tune",
        "research",
        "train-reid",
        "eval-reid",
        "compare-reid",
        "export",
        "build",
    ):
        assert f'"{command}"' in script
    assert 'find_spec("boxmot.api") is None' in script
    assert 'find_spec("boxmot.data") is None' in script


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


def test_all_images_smoke_packaged_component_configs() -> None:
    job = _docker_job()
    step = next(step for step in job["steps"] if step.get("name") == "Smoke test v24 package and command surface")
    script = step["run"]

    assert "from boxmot.engine.experiment_config import resolve_experiment_config" in script
    assert 'resolve_experiment_config("mot17-ablation-yolox-lmbn")' in script
    assert 'experiment["detector"]["id"] == "yolox-x-mot17"' in script
    assert 'experiment["reid"]["id"] == "lmbn-n-duke"' in script
