"""Static regression tests for v24 CI workflow dependencies and build selection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
BENCHMARK_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "benchmark.yml"


def _job_script(workflow_path: Path, job_name: str) -> str:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    return "\n".join(str(step.get("run", "")) for step in workflow["jobs"][job_name]["steps"])


@pytest.mark.parametrize(
    ("workflow_path", "job_name"),
    (
        (CI_WORKFLOW, "tune"),
        (CI_WORKFLOW, "metrics"),
        (BENCHMARK_WORKFLOW, "mot-metrics-benchmark"),
    ),
)
def test_cached_workflows_select_only_published_materialized_builds(
    workflow_path: Path,
    job_name: str,
) -> None:
    script = _job_script(workflow_path, job_name)

    assert "-type f -name manifest.json -print -quit" in script
    assert 'test -n "$BUILD_MANIFEST"' in script
    assert 'BUILD_PATH=$(dirname "$BUILD_MANIFEST")' in script
    assert "-type d ! -name .staging -print -quit" not in script


def test_macos_native_ci_pins_and_exposes_opencv_four() -> None:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    step = next(
        step
        for step in workflow["jobs"]["cpp_trackers"]["steps"]
        if step.get("name") == "Install native build dependencies (macOS)"
    )
    script = step["run"]

    assert "brew install cmake opencv@4 eigen onnxruntime" in script
    assert "CMAKE_PREFIX_PATH=$(brew --prefix opencv@4):$(brew --prefix eigen):$(brew --prefix onnxruntime)" in script
    assert '>> "$GITHUB_ENV"' in script


@pytest.mark.parametrize("job_name", ("tune", "metrics"))
def test_cached_ci_jobs_use_one_yolo26n_experiment_for_build_and_consumption(job_name: str) -> None:
    script = _job_script(CI_WORKFLOW, job_name)

    assert script.count("--experiment mot17-mini/train-yolo26n-lmbn.yaml") == 2
    assert "mot17-mini/train-yolox-lmbn.yaml" not in script


def test_rtdetr_smoke_installs_its_declared_feature_extra() -> None:
    script = _job_script(CI_WORKFLOW, "yolos")

    assert "--extra cpu --extra yolo --extra rtdetr" in script
    assert "boxmot track --detector rtdetr_v2_r18vd" in script


def test_obb_smoke_selects_obb_geometry_explicitly() -> None:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["obb"]
    step = next(step for step in job["steps"] if step.get("name") == "Run obb tracking method")
    track_args = next(
        line.strip() for line in str(step["run"]).splitlines() if line.strip().startswith("track_args=(boxmot track ")
    )

    assert '--detector "$BOXMOT_CI_OBB_DETECTOR" --geometry obb' in track_args
    assert "--name" not in track_args
    assert "--exist-ok" not in track_args
    assert job["env"]["EXPECTED_OBB_TRACKERS"].split() == job["env"]["TRACKERS"].split()
