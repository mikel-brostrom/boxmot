from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.engine.eval.motmetrics import SequenceData, _combine_bundles, _eval_bundle
from boxmot.engine.eval.trackeval_reference import evaluate_trackeval_motchallenge
from boxmot.engine.workflows.results import ValidationResult

trackeval = pytest.importorskip(
    "trackeval",
    reason="TrackEval is installed by the metrics CI parity step",
)


def test_trackeval_reference_uses_motchallenge_distractor_preprocessing(tmp_path):
    gt_folder = tmp_path / "gt"
    tracker_folder = tmp_path / "tracker"
    gt_path = gt_folder / "seq" / "gt" / "gt_temp.txt"
    gt_path.parent.mkdir(parents=True)
    tracker_folder.mkdir()

    gt_path.write_text(
        "1,1,0,0,10,10,1,1,1\n"
        "1,2,20,20,10,10,1,8,1\n"
    )
    (tracker_folder / "seq.txt").write_text(
        "1,1,0,0,10,10,1,1,-1,-1\n"
        "1,2,20,20,10,10,1,1,-1,-1\n"
    )

    result = evaluate_trackeval_motchallenge(
        gt_folder=gt_folder,
        tracker_folder=tracker_folder,
        seq_info={"seq": 1},
        benchmark="MOT17",
    )

    assert result["HOTA"] == pytest.approx(100.0)
    assert result["MOTA"] == pytest.approx(100.0)
    assert result["IDF1"] == pytest.approx(100.0)
    assert result["Dets"] == 1
    assert result["per_sequence"]["seq"]["Dets"] == 1


def test_validation_result_labels_trackeval_deltas():
    metrics = {
        "HOTA": 70.0,
        "MOTA": 80.0,
        "IDF1": 90.0,
        "AssA": 75.0,
        "AssRe": 85.0,
        "IDSW": 2,
        "IDs": 10,
        "per_sequence": {},
    }
    result = ValidationResult(
        benchmark="mot17",
        raw=metrics,
        summary_label="single_class",
        summary=metrics,
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type="aabb", classes=None),
        reference_raw={**metrics, "HOTA": 69.5},
        reference_name="TrackEval",
    )

    report = result.render(include_sequences=False)

    assert "BOXMOT vs TRACKEVAL" in report
    assert "Δ vs TrackEval" in report
    assert "+0.50" in report

PARITY_TOLERANCE = 1e-12
PARITY_FIELDS = {
    "HOTA": (
        "HOTA",
        "DetA",
        "AssA",
        "DetRe",
        "DetPr",
        "AssRe",
        "AssPr",
        "LocA",
        "OWTA",
    ),
    "CLEAR": (
        "MOTA",
        "MOTP",
        "CLR_Re",
        "CLR_Pr",
        "CLR_TP",
        "CLR_FN",
        "CLR_FP",
        "IDSW",
        "MT",
        "PT",
        "ML",
        "Frag",
    ),
    "Identity": ("IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"),
}


def _sequence(
    name: str,
    *,
    gt_ids: list[list[int]],
    tracker_ids: list[list[int]],
    similarity_scores: list[list[list[float]]],
) -> SequenceData:
    gt_arrays = [np.asarray(ids, dtype=int) for ids in gt_ids]
    tracker_arrays = [np.asarray(ids, dtype=int) for ids in tracker_ids]
    similarity_arrays = [
        np.asarray(scores, dtype=float).reshape(len(gt_frame), len(tracker_frame))
        for scores, gt_frame, tracker_frame in zip(similarity_scores, gt_arrays, tracker_arrays)
    ]
    unique_gt_ids = {int(value) for frame in gt_arrays for value in frame}
    unique_tracker_ids = {int(value) for frame in tracker_arrays for value in frame}
    assert unique_gt_ids == set(range(len(unique_gt_ids)))
    assert unique_tracker_ids == set(range(len(unique_tracker_ids)))
    return SequenceData(
        seq=name,
        gt_ids=gt_arrays,
        tracker_ids=tracker_arrays,
        similarity_scores=similarity_arrays,
        num_timesteps=len(gt_arrays),
        num_gt_dets=sum(len(ids) for ids in gt_arrays),
        num_tracker_dets=sum(len(ids) for ids in tracker_arrays),
        num_gt_ids=len(unique_gt_ids),
        num_tracker_ids=len(unique_tracker_ids),
    )


def _parity_sequences() -> dict[str, SequenceData]:
    return {
        "switches-and-fragments": _sequence(
            "switches-and-fragments",
            gt_ids=[[0, 1], [0, 1], [0, 1], [0], [0, 1]],
            tracker_ids=[[0, 1], [0, 1], [1, 2], [2, 3], [2]],
            similarity_scores=[
                [[0.95, 0.0], [0.0, 0.90]],
                [[0.80, 0.10], [0.10, 0.85]],
                [[0.0, 0.75], [0.90, 0.0]],
                [[0.60, 0.0]],
                [[0.40], [0.0]],
            ],
        ),
        "empty-and-boundary-frames": _sequence(
            "empty-and-boundary-frames",
            gt_ids=[[0], [0], [], [1]],
            tracker_ids=[[], [0], [0], [1]],
            similarity_scores=[[], [[1.0]], [], [[0.5]]],
        ),
    }


def _to_trackeval_data(data: SequenceData) -> dict[str, object]:
    return {
        "num_timesteps": data.num_timesteps,
        "num_gt_ids": data.num_gt_ids,
        "num_tracker_ids": data.num_tracker_ids,
        "num_gt_dets": data.num_gt_dets,
        "num_tracker_dets": data.num_tracker_dets,
        "gt_ids": data.gt_ids,
        "tracker_ids": data.tracker_ids,
        "similarity_scores": data.similarity_scores,
    }


def _evaluate_boxmot(sequences: dict[str, SequenceData]) -> dict[str, dict[str, dict[str, object]]]:
    per_sequence = {name: _eval_bundle(data) for name, data in sequences.items()}
    return {**per_sequence, "OVERALL": _combine_bundles(per_sequence)}


def _evaluate_trackeval(sequences: dict[str, SequenceData]) -> dict[str, dict[str, dict[str, object]]]:
    metrics = {
        "HOTA": trackeval.metrics.HOTA(),
        "CLEAR": trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        "Identity": trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
    }
    per_sequence = {
        name: {family: metric.eval_sequence(_to_trackeval_data(sequence)) for family, metric in metrics.items()}
        for name, sequence in sequences.items()
    }
    overall = {
        family: metric.combine_sequences({name: results[family] for name, results in per_sequence.items()})
        for family, metric in metrics.items()
    }
    return {**per_sequence, "OVERALL": overall}


def _comparison(left: object, right: object) -> tuple[float, float, float]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return float(np.mean(left_array)), float(np.mean(right_array)), float("inf")
    difference = np.abs(left_array - right_array)
    return (
        float(np.mean(left_array)),
        float(np.mean(right_array)),
        float(np.max(difference)) if difference.size else 0.0,
    )


def _render_table(rows: list[tuple[str, str, float, float, float]]) -> str:
    lines = [
        f"TrackEval parity (absolute tolerance: {PARITY_TOLERANCE:.0e})",
        "HOTA-family values are alpha means; max abs diff covers every alpha.",
        "Dataset | Metric | BoxMOT | TrackEval | max abs diff | Status",
        "--- | --- | ---: | ---: | ---: | :---:",
    ]
    for dataset, field, boxmot_value, trackeval_value, max_difference in rows:
        status = "PASS" if max_difference <= PARITY_TOLERANCE else "FAIL"
        lines.append(
            f"{dataset} | {field} | {boxmot_value:.12g} | {trackeval_value:.12g} | {max_difference:.3e} | {status}"
        )
    return "\n".join(lines)


def _write_github_summary(rows: list[tuple[str, str, float, float, float]]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    table = _render_table(rows).splitlines()
    with Path(summary_path).open("a", encoding="utf-8") as summary_file:
        summary_file.write("## BoxMOT ↔ TrackEval metric parity\n\n")
        summary_file.write("\n".join(table[1:]) + "\n")


def test_boxmot_metrics_match_trackeval() -> None:
    sequences = _parity_sequences()
    boxmot_results = _evaluate_boxmot(sequences)
    trackeval_results = _evaluate_trackeval(sequences)

    rows: list[tuple[str, str, float, float, float]] = []
    for dataset in (*sequences, "OVERALL"):
        for family, fields in PARITY_FIELDS.items():
            for field in fields:
                boxmot_value, trackeval_value, max_difference = _comparison(
                    boxmot_results[dataset][family][field],
                    trackeval_results[dataset][family][field],
                )
                rows.append((dataset, field, boxmot_value, trackeval_value, max_difference))

    print("\n" + _render_table(rows))
    _write_github_summary(rows)

    failures = [
        f"{dataset}: {field} ({max_difference:.3e})"
        for dataset, field, _, _, max_difference in rows
        if max_difference > PARITY_TOLERANCE
    ]
    if failures:
        pytest.fail(
            f"TrackEval parity exceeded {PARITY_TOLERANCE:.0e}:\n" + "\n".join(failures),
            pytrace=False,
        )
