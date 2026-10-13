"""Static regression tests for v24 CI workflow dependencies and build selection."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
BENCHMARK_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "benchmark.yml"


def _job_script(workflow_path: Path, job_name: str) -> str:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    return "\n".join(str(step.get("run", "")) for step in workflow["jobs"][job_name]["steps"])


def test_benchmark_workflow_is_manual_only() -> None:
    workflow = yaml.safe_load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))

    assert set(triggers) == {"workflow_dispatch"}


@pytest.mark.parametrize(
    ("workflow_path", "job_name"),
    (
        (CI_WORKFLOW, "tune"),
        (CI_WORKFLOW, "metrics"),
        (CI_WORKFLOW, "materialize"),
        (BENCHMARK_WORKFLOW, "mot-metrics-benchmark"),
    ),
)
def test_cached_workflows_select_only_published_materialized_builds(
    workflow_path: Path,
    job_name: str,
) -> None:
    script = _job_script(workflow_path, job_name)

    assert "-type f -name manifest.json -print" in script
    assert 'test -n "$BUILD_MANIFEST"' in script or 'test "${#BUILD_MANIFESTS[@]}" -eq 1' in script
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


@pytest.mark.parametrize("job_name", ("materialize", "tune", "metrics"))
def test_cached_ci_jobs_use_one_yolo26n_experiment_for_build_and_consumption(job_name: str) -> None:
    script = _job_script(CI_WORKFLOW, job_name)

    assert script.count("--experiment mot17-mini/train-yolo26n-lmbn.yaml") == 1
    assert "mot17-mini/train-yolox-lmbn.yaml" not in script


def test_tune_and_metrics_share_one_cached_materialization() -> None:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    materializing_jobs = [name for name in jobs if "boxmot materialize" in _job_script(CI_WORKFLOW, name)]

    assert materializing_jobs == ["materialize"]

    producer = jobs["materialize"]
    cache = next(step for step in producer["steps"] if step.get("uses") == "actions/cache@v4")
    cache_paths = str(cache["with"]["path"])
    cache_key = str(cache["with"]["key"])
    assert "ci-materialization" in cache_paths
    assert "models/yolo26n.pt" in cache_paths
    assert "models/lmbn_n_duke.pt" in cache_paths
    assert "restore-keys" not in cache["with"]
    for fingerprint_input in (
        "uv.lock",
        "pyproject.toml",
        "boxmot/**",
        "assets/MOT17-mini/**",
        ".github/workflows/ci.yml",
        ".github/actions/setup-ci-python/**",
        ".github/actions/prepare-ci-assets/**",
        ".github/scripts/fetch_ci_asset.sh",
        ".github/scripts/prepare_ci_assets.sh",
    ):
        assert fingerprint_input in cache_key

    materialize_step = next(step for step in producer["steps"] if "boxmot materialize" in str(step.get("run", "")))
    assert "if" not in materialize_step
    validation_step = next(step for step in producer["steps"] if step.get("name") == "Locate validated materialization")
    assert "if" not in validation_step
    assert 'test "${#BUILD_MANIFESTS[@]}" -eq 1' in validation_step["run"]
    assert 'test -f "$BUILD_PATH/_SUCCESS"' in validation_step["run"]

    upload = next(step for step in producer["steps"] if step.get("uses") == "actions/upload-artifact@v4")
    assert upload["with"]["name"] == "mot17-mini-materialization"
    assert set(str(upload["with"]["path"]).splitlines()) == {
        "ci-materialization",
        "models/yolo26n.pt",
        "models/lmbn_n_duke.pt",
    }
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["compression-level"] == 0
    assert upload["with"]["retention-days"] == 1
    assert producer["runs-on"] == "ubuntu-latest"

    for job_name in ("tune", "metrics"):
        job = jobs[job_name]
        assert job["needs"] == "materialize"
        assert job["strategy"]["matrix"]["os"] == ["ubuntu-latest"]
        assert "boxmot materialize" not in _job_script(CI_WORKFLOW, job_name)
        download = next(step for step in job["steps"] if step.get("uses") == "actions/download-artifact@v4")
        assert download["with"] == {
            "name": "mot17-mini-materialization",
            "path": "${{ github.workspace }}",
        }
        script = _job_script(CI_WORKFLOW, job_name)
        assert 'BUILD_ROOT="$GITHUB_WORKSPACE/ci-materialization"' in script
        assert 'test -f "$BUILD_PATH/_SUCCESS"' in script
        assert "BOXMOT_CI_BUILD=$BUILD_PATH" in script
        assert '--build "$BOXMOT_CI_BUILD"' in script

    failure_gate = jobs["check-failures"]
    assert "materialize" in failure_gate["needs"]
    assert "materialize=${{ needs.materialize.result }}" in _job_script(CI_WORKFLOW, "check-failures")


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


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("jq") is None, reason="CI reporting uses bash and jq")
@pytest.mark.parametrize("output", ("valid", "missing", "malformed", "null", "incomplete", "boolean", "failed"))
def test_metrics_workflow_requires_current_numeric_scores(tmp_path: Path, output: str) -> None:
    """Execute the CI shell against evaluator outputs, including a stale artifact."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    scripts = {step.get("name"): step["run"] for step in workflow["jobs"]["metrics"]["steps"] if "run" in step}
    activate = tmp_path / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.touch()
    stub = tmp_path / "evaluate.py"
    stub.write_text(
        textwrap.dedent(
            """\
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            assert args[0] == "eval"
            assert "--exist-ok" in args
            project = Path(args[args.index("--project") + 1])
            name = args[args.index("--name") + 1]
            path = project / "mot17-mini" / name / "metrics.json"
            mode = os.environ["TEST_METRICS_OUTPUT"] if name == "bytetrack" else "valid"
            if mode == "failed":
                sys.exit(1)
            if mode == "missing":
                sys.exit(0)
            path.parent.mkdir(parents=True, exist_ok=True)
            metrics = {"HOTA": 0, "MOTA": -1.5, "IDF1": 20.55}
            if name == "ocsort":
                metrics = {"HOTA": 29.54, "MOTA": 10.61, "IDF1": 19.87}
            if mode == "null":
                metrics["HOTA"] = None
            elif mode == "incomplete":
                del metrics["IDF1"]
            elif mode == "boolean":
                metrics["HOTA"] = True
            payload = "{broken" if mode == "malformed" else json.dumps({"pedestrian": metrics})
            path.write_text(payload)
            """
        ),
        encoding="utf-8",
    )
    env_path = tmp_path / "github-env"
    env = {
        **os.environ,
        "TRACKERS": "bytetrack ocsort",
        "RUNNER_TEMP": str(tmp_path),
        "BOXMOT_CI_BUILD": str(tmp_path / "build"),
        "BOXMOT_CI_DATA_ROOT": str(tmp_path / "assets"),
        "GITHUB_ENV": str(env_path),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "TEST_METRICS_OUTPUT": output,
        "TEST_PYTHON": sys.executable,
        "TEST_EVALUATOR": str(stub),
    }
    stale = tmp_path / "boxmot-ci-metrics/mot17-mini/bytetrack/metrics.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"pedestrian": {"HOTA": 99, "MOTA": 99, "IDF1": 99}}))
    script = 'boxmot() { "$TEST_PYTHON" "$TEST_EVALUATOR" "$@"; }\n' + scripts["Evaluate MOT17-mini metrics"]
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with (tmp_path / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    valid = output == "valid"
    assert rows == [
        {
            "Tracker": "Bytetrack",
            "Status❔": "✅" if valid else "❌",
            "HOTA": "0" if valid else "",
            "MOTA": "-1.5" if valid else "",
            "IDF1": "20.55" if valid else "",
        },
        {"Tracker": "Ocsort", "Status❔": "✅", "HOTA": "29.54", "MOTA": "10.61", "IDF1": "19.87"},
    ]
    assert env_path.read_text().strip() == f"BOXMOT_METRICS_FAILED={0 if valid else 1}"
    gate = subprocess.run(
        ["bash", "-e", "-c", scripts["Enforce successful tracker evaluations"]],
        cwd=tmp_path,
        env={**env, "BOXMOT_METRICS_FAILED": "0" if valid else "1"},
        capture_output=True,
        text=True,
    )
    assert gate.returncode == (0 if valid else 1)
    # The sorting step only reformats rows; use its output names to exercise
    # the actual summary step without requiring the optional `column` binary.
    shutil.copyfile(tmp_path / "metrics.csv", tmp_path / "sorted_results.csv")
    (tmp_path / "pretty_results.txt").write_text("Metrics\n")
    subprocess.run(
        ["bash", "-e", "-c", scripts["Show Results"]], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    summary = (tmp_path / "summary.md").read_text()
    assert ("| Bytetrack | ✅ | 0 | -1.5 | 20.55 |" if valid else "| Bytetrack | ❌ |  |  |  |") in summary
    assert "| Ocsort | ✅ | 29.54 | 10.61 | 19.87 |" in summary
