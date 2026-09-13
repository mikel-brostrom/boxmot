"""Built-in image-only KITTI scoring and optional installed-reference parity."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from boxmot.engine.config.experiments import resolve_experiment_config
from boxmot.engine.eval import kitti_tracking
from boxmot.engine.eval.evaluator import _target_classes
from boxmot.engine.eval.trackeval_reference import evaluate_trackeval_kitti, validate_trackeval_kitti_dependencies


def _gt(
    frame: int,
    identity: int,
    label: str = "Car",
    bounds: tuple[int, ...] = (0, 0, 50, 50),
    *,
    truncation: int = 0,
    occlusion: int = 0,
) -> str:
    """Use native tracking metadata and explicitly unavailable spatial geometry."""
    return (
        f"{frame} {identity} {label} {truncation} {occlusion} -10 "
        f"{' '.join(map(str, bounds))} -1 -1 -1 -1000 -1000 -1000 -10"
    )


def _prediction(frame: int, identity: int, *, class_id: int = 1, bounds: tuple[int, ...] = (0, 0, 50, 50)) -> str:
    left, top, right, bottom = bounds
    return f"{frame + 1},{identity},{left},{top},{right - left},{bottom - top},0.9,{class_id},-1"


def _args(
    root: Path,
    rows: list[str],
    predictions: list[str],
    *,
    native_count: int = 1,
    frames: list[tuple[int, int]] | None = None,
) -> SimpleNamespace:
    truth = root / "source.txt"
    truth.write_text("\n".join(rows) + "\n", encoding="utf-8")
    output = root / "results"
    output.mkdir()
    (output / "0000.txt").write_text("\n".join(predictions) + "\n", encoding="utf-8")
    return SimpleNamespace(
        exp_dir=output,
        evaluation_config={
            "classes": {
                name: {"id": identity, "evaluation": "target"} for name, identity in (("car", 1), ("pedestrian", 2))
            },
            "kitti_gt_sequences": {
                "0000": {"path": str(truth), "frame_count": native_count, "frames": frames or [(0, 0)]}
            },
        },
    )


@pytest.fixture(autouse=True)
def forbid_implicit_trackeval(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every built-in regression must run without importing the optional evaluator."""
    if "installed_trackeval" in request.fixturenames:
        return
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "trackeval" or name.startswith("trackeval."):
            raise AssertionError("Built-in KITTI evaluation must not import TrackEval")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.fixture
def installed_trackeval() -> None:
    pytest.importorskip("trackeval")
    validate_trackeval_kitti_dependencies()


@pytest.mark.parametrize("cache_inputs", [False, True])
def test_kitti_2d_metrics_use_official_distractor_visibility_and_dontcare_rules(
    tmp_path: Path, cache_inputs: bool
) -> None:
    rows = [
        _gt(0, 1),
        _gt(0, 2, "Van", (60, 0, 110, 50)),
        _gt(0, 3, bounds=(120, 0, 170, 50), truncation=1),
        _gt(0, 4, bounds=(180, 0, 230, 50), occlusion=3),
        _gt(0, -1, "DontCare", (240, 0, 290, 50)),
        _gt(0, 5, "Person_sitting", (0, 60, 50, 110)),
        _gt(0, 6, "Pedestrian", (60, 60, 110, 110)),
    ]
    predictions = [
        _prediction(0, index + 10, class_id=1 if index <= 4 else 2, bounds=tuple(map(int, row.split()[6:10])))
        for index, row in enumerate(rows)
    ]
    args = _args(tmp_path, rows, predictions)
    args.cache_inputs = cache_inputs
    result = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 1})

    for label in ("car", "pedestrian"):
        assert result[label]["HOTA"] == result[label]["MOTA"] == result[label]["IDF1"] == 100
        assert result[label]["Dets"] == result[label]["GT_Dets"] == 1
    exported = (args.exp_dir / "protocol_inputs/tracking/ground_truth/label_02/0000.txt").read_text()
    assert " Person " in exported and " DontCare " in exported and " Van " in exported
    assert len(exported.splitlines()) == len(rows)
    assert json.loads((args.exp_dir / "metrics.json").read_text()) == result
    assert (args.exp_dir / "metrics.csv").is_file()
    protocol = json.loads((args.exp_dir / "evaluation.json").read_text())
    assert protocol["tracking"]["geometry"] == "2d"
    assert "detection" not in protocol
    assert not (args.exp_dir / "detection_metrics.json").exists()
    if cache_inputs:
        repeated = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 1})
        assert repeated == result


def test_kitti_2d_metrics_remap_selected_frames_and_large_identities_losslessly(tmp_path: Path) -> None:
    identity = 2**53
    rows = [
        _gt(frame, identity + offset, bounds=(60 * offset, 0, 60 * offset + 50, 50))
        for frame in (0, 2, 4)
        for offset in (0, 1)
    ]
    predictions = [
        _prediction(frame, identity + offset + 10, bounds=(60 * offset, 0, 60 * offset + 50, 50))
        for frame in (0, 1)
        for offset in (0, 1)
    ]
    args = _args(tmp_path, rows, predictions, native_count=5, frames=[(0, 0), (1, 4)])
    result = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 2})

    assert result["car"]["HOTA"] == result["car"]["IDF1"] == 100
    assert result["car"]["GT_Dets"] == 4
    assert result["car"]["GT_IDs"] == result["car"]["IDs"] == 2
    protocol = json.loads((args.exp_dir / "evaluation.json").read_text())
    assert protocol["tracking_identity_maps"]["0000"]["ground_truth"] == {str(identity): 0, str(identity + 1): 1}
    assert protocol["frames"]["0000"] == [
        {"frame_index": 0, "source_frame_index": 0},
        {"frame_index": 1, "source_frame_index": 4},
    ]
    exported = (args.exp_dir / "protocol_inputs/tracking/predictions/0000.txt").read_text().splitlines()
    assert all(row.split()[10:17] == ["-1", "-1", "-1", "-1000", "-1000", "-1000", "-10"] for row in exported)


def test_kitti_2d_metrics_count_identity_switch_and_empty_frames(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        [_gt(0, 1), _gt(2, 1)],
        [_prediction(0, 3), _prediction(2, 4)],
        native_count=3,
        frames=[(0, 0), (1, 1), (2, 2)],
    )
    result = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 3})["car"]
    assert result["Frames"] == 3
    assert result["IDSW"] == 1
    assert result["MOTA"] == result["IDF1"] == 50


@pytest.mark.parametrize(
    "class_map", [{"car": "car"}, {"pedestrian": "person"}, {"car": "car", "pedestrian": "person"}]
)
def test_kitti_2d_honors_authored_experiment_class_subset(tmp_path: Path, class_map: dict[str, str]) -> None:
    """An omitted class must not lower the selected experiment's objective."""
    experiment_path = tmp_path / "selected-classes.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"ref": "kitti-2d", "split": "train"},
                "detector": {"ref": "yolo26n", "checkpoint": "default"},
                "evaluation": {"class_map": class_map},
            }
        ),
        encoding="utf-8",
    )
    experiment = resolve_experiment_config(experiment_path, mode="eval")
    identities, labels = _target_classes(experiment["dataset"], experiment)
    rows = [_gt(0, 7), _gt(0, 8, "Pedestrian", (60, 0, 110, 50))]
    predictions = [
        _prediction(0, identity, class_id=identity, bounds=(0, 0, 50, 50) if identity == 1 else (60, 0, 110, 50))
        for identity in identities
    ]
    args = _args(tmp_path, rows, predictions)
    args.remapped_class_ids = list(identities)
    args.remapped_class_names = [name for _, name in labels]

    results = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 1})

    assert set(results) == set(class_map) | {"cls_comb_cls_av", "cls_comb_det_av"}
    for name in class_map:
        assert results[name]["GT_Dets"] == results[name]["Dets"] == 1
    for summary in ("cls_comb_cls_av", "cls_comb_det_av"):
        assert results[summary]["HOTA"] == results[summary]["IDF1"] == results[summary]["MOTA"] == 100
        assert results[summary]["GT_Dets"] == len(class_map)
    protocol = json.loads((args.exp_dir / "evaluation.json").read_text())
    assert protocol["tracking"]["classes"] == args.remapped_class_names
    assert len((args.exp_dir / "metrics.csv").read_text().splitlines()) == len(class_map) + 3
    # Keep ignored and distractor GT available for native preprocessing; class
    # selection must not discard those annotations at export.
    exported = (args.exp_dir / "protocol_inputs/tracking/ground_truth/label_02/0000.txt").read_text()
    assert " Car " in exported and " Pedestrian " in exported


@pytest.mark.parametrize(
    ("names", "identities"),
    [(["car"], [2]), (["pedestrian"], [1]), (["car", "car"], [1, 1]), (["car"], []), ([], []), (["car"], None)],
)
def test_kitti_2d_rejects_invalid_experiment_class_bridge(
    tmp_path: Path, names: list[str], identities: list[int]
) -> None:
    args = _args(tmp_path, [_gt(0, 7)], [_prediction(0, 1)])
    args.remapped_class_names, args.remapped_class_ids = names, identities
    with pytest.raises(ValueError, match="class"):
        kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 1})


@pytest.mark.parametrize("names", [(), ("car", "car"), ("cyclist",), "car", None])
def test_trackeval_kitti_rejects_invalid_class_subsets_before_loading_backend(tmp_path: Path, names: object) -> None:
    with pytest.raises(ValueError, match="unique subset"):
        evaluate_trackeval_kitti(gt_folder=tmp_path, tracker_folder=tmp_path, seq_info={"0000": 1}, class_names=names)


@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (0, "0", "frame number"),
        (0, "2", "selected image timeline"),
        (1, "9007199254740993.0", "integer"),
        (1, "-1", "identity"),
        (1, str(2**63), "int64"),
        (4, "0", "positive width"),
        (2, "nan", "finite"),
        (6, "2", "confidence"),
        (7, "3", "classes"),
        (8, "-2", "detection index"),
    ],
)
def test_kitti_2d_rejects_invalid_replay_before_scoring(tmp_path: Path, index: int, value: str, message: str) -> None:
    fields = _prediction(0, 1).split(",")
    fields[index] = value
    args = _args(tmp_path, [_gt(0, 1)], [",".join(fields)])
    with pytest.raises(ValueError, match=message):
        kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 1})


def test_kitti_2d_rejects_duplicate_predictions_before_scoring(tmp_path: Path) -> None:
    args = _args(tmp_path, [_gt(0, 1)], [_prediction(0, 2), _prediction(0, 2)])
    with pytest.raises(ValueError, match="duplicate track identity"):
        kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 1})


@pytest.mark.parametrize("frames", [[(0, 0), (0, 1)], [(0, 0), (1, 0)], [(0, 1), (1, 0)], [(0, 0), (1, 2)]])
def test_kitti_2d_rejects_invalid_frame_mapping(tmp_path: Path, frames: list[tuple[int, int]]) -> None:
    args = _args(tmp_path, [_gt(0, 1)], [], native_count=2, frames=frames)
    with pytest.raises(ValueError, match="(selected frames|chronological)"):
        kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 2})


def _preprocessing_case(name: str) -> tuple[list[str], list[str], int, int]:
    """Cases distinguish KITTI preprocessing from filtering before association."""
    if name == "matched-small-inside-dontcare":
        bounds = (0, 0, 50, 25)
        return [_gt(0, 1, bounds=bounds), _gt(0, -1, "DontCare", bounds)], [_prediction(0, 10, bounds=bounds)], 1, 1
    if name == "unmatched-height-boundary":
        return (
            [_gt(0, 1)],
            [
                _prediction(0, 10),
                _prediction(0, 11, bounds=(60, 0, 110, 25)),
                _prediction(0, 12, bounds=(120, 0, 170, 26)),
            ],
            1,
            2,
        )
    if name == "dontcare-ioa-boundary":
        return (
            [_gt(0, 1), _gt(0, -1, "DontCare", (100, 0, 125, 50)), _gt(0, -1, "DontCare", (200, 0, 226, 50))],
            [
                _prediction(0, 10),
                _prediction(0, 11, bounds=(100, 0, 150, 50)),
                _prediction(0, 12, bounds=(200, 0, 250, 50)),
            ],
            1,
            2,
        )
    if name == "distractor-iou-boundary":
        return (
            [_gt(0, 1), _gt(0, 2, "Van", (100, 0, 130, 60)), _gt(0, 3, "Van", (200, 0, 230, 60))],
            [
                _prediction(0, 10),
                _prediction(0, 11, bounds=(100, 0, 160, 60)),
                _prediction(0, 12, bounds=(200, 0, 261, 60)),
            ],
            1,
            2,
        )
    if name == "distractor-matched-before-removal":
        return (
            [_gt(0, 1, bounds=(0, 0, 60, 60)), _gt(0, 2, "Van", (10, 0, 70, 60))],
            [_prediction(0, 10, bounds=(10, 0, 70, 60))],
            1,
            0,
        )
    if name == "visibility-boundaries":
        return (
            [
                _gt(0, 1, occlusion=2),
                _gt(0, 2, bounds=(60, 0, 110, 50), occlusion=3),
                _gt(0, 3, bounds=(120, 0, 170, 50), truncation=1),
            ],
            [
                _prediction(0, 10),
                _prediction(0, 11, bounds=(60, 0, 110, 50)),
                _prediction(0, 12, bounds=(120, 0, 170, 50)),
            ],
            1,
            1,
        )
    if name == "wrong-class-neighbor":
        return (
            [_gt(0, 1), _gt(0, 2, "Person_sitting", (60, 0, 110, 50))],
            [_prediction(0, 10), _prediction(0, 11, bounds=(60, 0, 110, 50))],
            1,
            2,
        )
    if name == "no-ground-truth":
        return [], [_prediction(0, 10)], 0, 1
    if name == "no-predictions":
        return [_gt(0, 1)], [], 1, 0
    if name == "empty-sequence":
        return [], [], 0, 0
    raise AssertionError(f"Unknown preprocessing case: {name}")


_PREPROCESSING_CASES = (
    "matched-small-inside-dontcare",
    "unmatched-height-boundary",
    "dontcare-ioa-boundary",
    "distractor-iou-boundary",
    "distractor-matched-before-removal",
    "visibility-boundaries",
    "wrong-class-neighbor",
    "no-ground-truth",
    "no-predictions",
    "empty-sequence",
)


@pytest.mark.parametrize("case", _PREPROCESSING_CASES)
def test_builtin_kitti_preprocessing_preserves_matching_order_and_boundaries(tmp_path: Path, case: str) -> None:
    rows, predictions, gt_count, prediction_count = _preprocessing_case(case)
    args = _args(tmp_path, rows, predictions)

    result = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 1})

    assert result["car"]["GT_Dets"] == gt_count
    assert result["car"]["Dets"] == prediction_count
    assert result["car"]["Frames"] == 1
    protocol = json.loads((args.exp_dir / "evaluation.json").read_text())
    assert protocol["tracking"]["evaluator"] == "boxmot"
    assert protocol["tracking"]["geometry"] == "2d"


def _assert_metric_parity(actual: dict, expected: dict) -> None:
    """Compare metric summaries and per-sequence results with float tolerance."""
    assert set(expected) <= set(actual)
    assert set(actual) - set(expected) <= {"IDa", "IDm", "IDt"}
    for name, expected_value in expected.items():
        if isinstance(expected_value, dict):
            _assert_metric_parity(actual[name], expected_value)
        else:
            assert actual[name] == pytest.approx(expected_value, rel=1e-10, abs=1e-10), name


@pytest.mark.usefixtures("installed_trackeval")
@pytest.mark.parametrize("case", _PREPROCESSING_CASES)
def test_builtin_kitti_preprocessing_and_metrics_match_optional_trackeval(tmp_path: Path, case: str) -> None:
    rows, predictions, _, _ = _preprocessing_case(case)
    args = _args(tmp_path, rows, predictions)
    actual = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 1})
    protocol_root = args.exp_dir / "protocol_inputs" / "tracking"

    reference = evaluate_trackeval_kitti(
        gt_folder=protocol_root / "ground_truth",
        tracker_folder=protocol_root / "predictions",
        seq_info={"0000": 1},
    )

    _assert_metric_parity(actual, reference)


@pytest.mark.usefixtures("installed_trackeval")
def test_builtin_kitti_identity_metrics_match_optional_trackeval_on_subsampled_frames(tmp_path: Path) -> None:
    identity = 2**53
    rows = [
        _gt(
            frame,
            identity + offset,
            "Car" if offset == 0 else "Pedestrian",
            bounds=(60 * offset, 0, 60 * offset + 50, 50),
        )
        for frame in (0, 1, 3, 4)
        for offset in (0, 1)
    ]
    predictions = [
        _prediction(
            frame,
            identity + offset + (10 if frame < 2 else 20),
            class_id=offset + 1,
            bounds=(60 * offset, 0, 60 * offset + 50, 50),
        )
        for frame in (0, 2, 3)
        for offset in (0, 1)
    ]
    args = _args(tmp_path, rows, predictions, native_count=5, frames=[(0, 0), (1, 2), (2, 3), (3, 4)])
    actual = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info={"0000": 4})
    protocol_root = args.exp_dir / "protocol_inputs" / "tracking"
    reference = evaluate_trackeval_kitti(
        gt_folder=protocol_root / "ground_truth", tracker_folder=protocol_root / "predictions", seq_info={"0000": 4}
    )

    _assert_metric_parity(actual, reference)
    assert actual["car"]["IDSW"] == actual["pedestrian"]["IDSW"] == 1


def _multi_sequence_args(tmp_path: Path) -> SimpleNamespace:
    """Make unequal class sizes distinguish class and detection weighted metrics."""
    args = _args(
        tmp_path,
        [_gt(0, 1), _gt(1, 1)],
        [_prediction(0, 10), _prediction(1, 10)],
        native_count=2,
        frames=[(0, 0), (1, 1)],
    )
    second_truth = tmp_path / "second-source.txt"
    second_truth.write_text("\n".join(_gt(frame, 1, "Pedestrian") for frame in range(3)) + "\n", encoding="utf-8")
    (args.exp_dir / "0001.txt").write_text(_prediction(0, 10, class_id=2) + "\n", encoding="utf-8")
    args.evaluation_config["kitti_gt_sequences"]["0001"] = {
        "path": str(second_truth),
        "frame_count": 3,
        "frames": [(0, 0), (1, 1), (2, 2)],
    }
    return args


def test_builtin_kitti_combines_sequences_and_weights_classes_without_double_counting_frames(tmp_path: Path) -> None:
    args = _multi_sequence_args(tmp_path)
    result = kitti_tracking.run_kitti_tracking_metrics(
        args, [], args.exp_dir, tmp_path, seq_info={"0000": 2, "0001": 3}
    )

    assert result["car"]["MOTA"] == 100
    assert result["pedestrian"]["MOTA"] == pytest.approx(100 / 3)
    assert result["cls_comb_cls_av"]["MOTA"] == pytest.approx(200 / 3)
    assert result["cls_comb_det_av"]["MOTA"] == 60
    for name in ("cls_comb_cls_av", "cls_comb_det_av"):
        assert result[name]["GT_Dets"] == 5
        assert result[name]["Dets"] == 3
        assert result[name]["GT_IDs"] == result[name]["IDs"] == 2
    for values in result.values():
        assert values["Frames"] == 5


@pytest.mark.usefixtures("installed_trackeval")
def test_builtin_kitti_multiple_sequence_aggregation_matches_optional_trackeval(tmp_path: Path) -> None:
    args = _multi_sequence_args(tmp_path)
    counts = {"0000": 2, "0001": 3}
    actual = kitti_tracking.run_kitti_tracking_metrics(args, [], args.exp_dir, tmp_path, seq_info=counts)
    protocol_root = args.exp_dir / "protocol_inputs" / "tracking"
    reference = evaluate_trackeval_kitti(
        gt_folder=protocol_root / "ground_truth", tracker_folder=protocol_root / "predictions", seq_info=counts
    )

    _assert_metric_parity(actual, reference)
