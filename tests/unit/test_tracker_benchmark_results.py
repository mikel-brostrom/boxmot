"""Tests for the benchmark workflow's result and README contract."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / ".github/scripts/tracker_benchmark_results.py"


@pytest.fixture(scope="module")
def benchmark_helper():
    spec = importlib.util.spec_from_file_location("tracker_benchmark_results", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _complete_records(helper, *, native_delta: float = 0.1):
    records = {}
    for benchmark_index, benchmark in enumerate(helper.BENCHMARKS):
        for tracker_index, tracker in enumerate(helper.TRACKERS):
            python_metrics = {
                metric: 20.0 * (benchmark_index + 1) + tracker_index + metric_index / 10
                for metric_index, metric in enumerate(helper.METRICS)
            }
            records[(benchmark, tracker, "python")] = python_metrics
            if tracker in helper.NATIVE_TRACKERS:
                records[(benchmark, tracker, "cpp")] = {
                    metric: value + native_delta for metric, value in python_metrics.items()
                }
    return records


def _published_readme(helper, *, native_delta: float = 0.1) -> str:
    return helper.update_readme(
        (REPO_ROOT / "README.md").read_text(),
        _complete_records(helper, native_delta=native_delta),
    )


def test_mmot_extracts_class_average_cls(benchmark_helper):
    payload = {
        "cls_comb_cls_av": {"HOTA": 49.84, "MOTA": 39.41, "IDF1": 58.60},
        "cls_comb_det_av": {"HOTA": 99.0, "MOTA": 98.0, "IDF1": 97.0},
    }

    assert benchmark_helper.extract_metrics(payload, "mmot-obb") == {
        "HOTA": 49.84,
        "MOTA": 39.41,
        "IDF1": 58.60,
    }

    with pytest.raises(benchmark_helper.BenchmarkError, match="cls_comb_cls_av"):
        benchmark_helper.extract_metrics({"cls_comb_det_av": payload["cls_comb_det_av"]}, "mmot-obb")


def test_parity_gate_accepts_quarter_point_and_rejects_larger_delta(benchmark_helper):
    records = _complete_records(benchmark_helper, native_delta=0.25)
    comparisons = benchmark_helper.validate_records(records, tolerance=0.25)
    assert len(comparisons) == 2 * 5 * 3

    records[("mmot-obb", "occluboost", "cpp")]["HOTA"] += 0.001
    with pytest.raises(benchmark_helper.BenchmarkError, match=r"exceeded 0\.25.*mmot-obb/occluboost/HOTA"):
        benchmark_helper.validate_records(records, tolerance=0.25)


def test_parity_gate_rejects_missing_backend_result(benchmark_helper):
    records = _complete_records(benchmark_helper)
    del records[("mot17-aabb", "occluboost", "cpp")]

    with pytest.raises(
        benchmark_helper.BenchmarkError,
        match="missing required benchmark result.*mot17-aabb/occluboost/cpp",
    ):
        benchmark_helper.validate_records(records)


def test_published_readme_gate_accepts_complete_native_pairs(benchmark_helper):
    readme = _published_readme(benchmark_helper, native_delta=0.25)

    comparisons = benchmark_helper.validate_published_readme(readme, tolerance=0.25)
    records = benchmark_helper.published_readme_records(readme)

    assert len(comparisons) == 2 * 5 * 3
    assert set(records) == {
        (benchmark, tracker, backend)
        for benchmark in benchmark_helper.BENCHMARKS
        for tracker in benchmark_helper.NATIVE_TRACKERS
        for backend in benchmark_helper.BACKENDS
    }


def test_published_readme_gate_rejects_unpopulated_cpp_metric(benchmark_helper):
    readme = _published_readme(benchmark_helper).replace("20.00<br>(20.10)", "20.00<br>(—)", 1)

    with pytest.raises(
        benchmark_helper.BenchmarkError,
        match=r"mot17-aabb/occluboost/HOTA.*missing its published C\+\+ metric",
    ):
        benchmark_helper.validate_published_readme(readme)


def test_published_readme_gate_rejects_delta_above_tolerance(benchmark_helper):
    readme = _published_readme(benchmark_helper, native_delta=0.26)

    with pytest.raises(
        benchmark_helper.BenchmarkError,
        match=r"Published README.*exceeded 0\.25.*mot17-aabb/occluboost/HOTA: 0\.2600 pp",
    ):
        benchmark_helper.validate_published_readme(readme, tolerance=0.25)


def test_published_readme_gate_rejects_missing_native_tracker_row(benchmark_helper):
    readme = _published_readme(benchmark_helper)
    botsort_row = re.search(r"<tr>(?:(?!</tr>).)*botsort(?:(?!</tr>).)*</tr>", readme, re.DOTALL)
    assert botsort_row is not None
    readme = readme[: botsort_row.start()] + readme[botsort_row.end() :]

    with pytest.raises(benchmark_helper.BenchmarkError, match="native tracker row.*botsort"):
        benchmark_helper.validate_published_readme(readme)


def test_checked_in_readme_published_native_pairs_pass(benchmark_helper):
    comparisons = benchmark_helper.validate_published_readme((REPO_ROOT / "README.md").read_text())

    assert len(comparisons) == 2 * 5 * 3


def test_readme_update_keeps_benchmark_and_backend_dimensions_separate(benchmark_helper):
    records = _complete_records(benchmark_helper)
    readme = (REPO_ROOT / "README.md").read_text()

    updated = benchmark_helper.update_readme(readme, records)
    botsort_row = re.search(r"<tr>(?:(?!</tr>).)*botsort(?:(?!</tr>).)*</tr>", updated, re.DOTALL)
    boosttrack_row = re.search(r"<tr>(?:(?!</tr>).)*boosttrack(?:(?!</tr>).)*</tr>", updated, re.DOTALL)
    assert botsort_row is not None and boosttrack_row is not None

    # botsort is index 1: MOT17 values are 21.x, while MMOT values are 41.x.
    assert "21.00<br>(21.10)" in botsort_row.group(0)
    assert "41.00<br>(41.10)" in botsort_row.group(0)
    # SportsMOT is not part of this workflow and remains untouched.
    assert "76.93" in botsort_row.group(0)
    # Trackers without a native implementation still use the documented pair form.
    assert "22.00<br>(—)" in boosttrack_row.group(0)
    assert "42.00<br>(—)" in boosttrack_row.group(0)


def test_selected_benchmark_validation_rejects_cross_group_artifacts(benchmark_helper):
    records = {key: value for key, value in _complete_records(benchmark_helper).items() if key[0] == "mot17-aabb"}
    benchmark_helper.validate_records(records, benchmarks=("mot17-aabb",))

    records[("mmot-obb", "botsort", "python")] = {metric: 1.0 for metric in benchmark_helper.METRICS}
    with pytest.raises(benchmark_helper.BenchmarkError, match="unexpected benchmark result"):
        benchmark_helper.validate_records(records, benchmarks=("mot17-aabb",))


def test_selected_readme_update_leaves_other_benchmark_untouched(benchmark_helper):
    records = {key: value for key, value in _complete_records(benchmark_helper).items() if key[0] == "mot17-aabb"}
    readme = (REPO_ROOT / "README.md").read_text()
    botsort_before = re.search(r"<tr>(?:(?!</tr>).)*botsort(?:(?!</tr>).)*</tr>", readme, re.DOTALL)
    assert botsort_before is not None

    updated = benchmark_helper.update_readme(readme, records, benchmarks=("mot17-aabb",))
    botsort_after = re.search(r"<tr>(?:(?!</tr>).)*botsort(?:(?!</tr>).)*</tr>", updated, re.DOTALL)
    assert botsort_after is not None

    # The selected MOT17 group changes, while the existing MMOT publication is
    # byte-for-byte preserved by the hosted MOT17-only workflow.
    assert "21.00<br>(21.10)" in botsort_after.group(0)
    cell_pattern = re.compile(r"<td[^>]*><sub>(.*?)</sub></td>", re.DOTALL)
    before_cells = cell_pattern.findall(botsort_before.group(0))
    after_cells = cell_pattern.findall(botsort_after.group(0))
    assert after_cells[8:11] == before_cells[8:11]


def test_workflow_matrix_matches_required_result_keys(benchmark_helper):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/benchmark.yml").read_text())
    matrix = workflow["jobs"]["mot-metrics-benchmark"]["strategy"]["matrix"]
    excluded = {(entry["tracker"], entry["backend"]) for entry in matrix.get("exclude", [])}
    expanded = {
        ("mot17-aabb", tracker, backend)
        for tracker in matrix["tracker"]
        for backend in matrix["backend"]
        if (tracker, backend) not in excluded
    }

    assert expanded == benchmark_helper.expected_keys(("mot17-aabb",))
    assert len(expanded) == 14


def test_workflow_has_download_free_published_readme_gate():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/benchmark.yml").read_text())
    job = workflow["jobs"]["published-metric-parity"]

    assert "needs" not in job
    step_texts = [f"{step.get('uses', '')}\n{step.get('run', '')}" for step in job["steps"]]
    assert any("tracker_benchmark_results.py verify-readme" in step_text for step_text in step_texts)
    assert all("download" not in step_text.lower() for step_text in step_texts)
