"""Exercise the KITTI sensor reader, class presets, CLI, and mask evaluation together."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner
from PIL import Image

from boxmot import EagerMot
from boxmot.datasets.sequence import MultimodalSequence
from boxmot.engine.cli import boxmot
from boxmot.engine.eval.eagermot_kitti import KITTI_PROFILES, _track_frame, load_kitti_profiles
from boxmot.engine.eval.mots_io import read_mots_results
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture

mask_utils = pytest.importorskip("pycocotools.mask")


def _fixture(tmp_path: Path) -> SimpleNamespace:
    """Write real KITTI inputs with two objects separated by a complete sensor dropout."""
    data = sensor_dataset_fixture(tmp_path)
    paths = data.reader_paths
    paths["calibration"].write_text("P2: 20 0 24 0 0 20 12 0 0 0 1 0\n")
    np.save(paths["poses"], np.repeat(np.eye(4)[None], 3, axis=0))
    objects = (
        (2, "Pedestrian", "pedestrian_detections_3d", (26, 8, 34, 15), 5, 1),
        (1, "Car", "car_detections_3d", (14, 8, 23, 15), -5, 4),
    )
    detection_rows = []
    for frame in range(3):
        labels = np.zeros((24, 48), dtype=np.uint16)
        Image.fromarray(np.zeros((24, 48, 3), dtype=np.uint8)).save(paths["images"] / f"{frame:06d}.png")
        for class_id, class_name, variant, bounds, x, length in objects:
            directory = paths[variant]
            directory.mkdir(parents=True, exist_ok=True)
            if frame == 1:
                # No 2D rows and no PointGNN file: the reader must retain this timestep.
                continue
            x1, y1, x2, y2 = bounds
            labels[y1:y2, x1:x2] = class_id * 1000 + 1
            encoded = mask_utils.encode(np.asfortranarray(labels == class_id * 1000 + 1, dtype=np.uint8))
            fields = [frame, *bounds, 0.98, class_id, 24, 48, encoded["counts"].decode("ascii"), *([0] * 128)]
            detection_rows.append(" ".join(map(str, fields)))
            row = [class_name, 0, 0, 0, *bounds, 3, 1, length, x, 0, 20, 0, 120]
            (directory / f"{frame:06d}.txt").write_text(" ".join(map(str, row)) + "\n")
        Image.fromarray(labels).save(data.ground_truth / f"{frame:06d}.png")
    paths["detections_2d"].write_text("\n".join(detection_rows) + "\n")
    return data


def _arguments(data: SimpleNamespace) -> list[str]:
    """Invoke the registered command with explicit temporary data paths."""
    return [
        "eval",
        "--tracker",
        "eagermot",
        "--dataset",
        str(data.dataset),
        "--project",
        str(data.project),
        "--sequence",
        "0002",
    ]


def test_cli_evaluates_actual_sensor_inputs_and_preserves_previous_results(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    previous_threads = torch.get_num_threads()
    invocation = CliRunner().invoke(boxmot, _arguments(data))
    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    assert torch.get_num_threads() == previous_threads
    output = data.project / "val"
    metrics = json.loads((output / "metrics.json").read_text())
    assert set(metrics) == {"car", "pedestrian", "cls_comb_cls_av", "cls_comb_det_av"}
    for class_name in ("car", "pedestrian"):
        values = metrics[class_name]
        assert {key: values[key] for key in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")} == dict.fromkeys(
            ("HOTA", "DetA", "AssA", "MOTA", "IDF1"), 100
        )
        assert values["Frames"] == 3
        assert values["Dets"] == values["GT_Dets"] == 2
        assert values["IDs"] == values["GT_IDs"] == 1
        assert set(values["per_sequence"]) == {"0002"}
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["split"] == "val"
    assert manifest["sequences"] == {"0002": 3}
    assert manifest["dataset_config"] == str(data.dataset.resolve())
    assert manifest["dataset_id"] == "kitti-mots-fusion"
    assert manifest["sequence_inputs"]["0002"]["ground_truth"]["paths"] == [str(data.ground_truth.resolve())]
    assert manifest["sequence_inputs"]["0002"]["detections_3d"]["format"] == "kitti-detections"
    with (output / "metrics.csv").open(newline="") as handle:
        assert {row["class"] for row in csv.DictReader(handle)} == set(metrics)

    rows = read_mots_results(output / "mots/0002.txt")
    assert set(rows) == {0, 2}
    first_ids = {row.class_id: row.track_id for row in rows[0]}
    assert len(set(first_ids.values())) == 2
    assert {row.class_id: row.track_id for row in rows[2]} == first_ids
    originals = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    repeated = CliRunner().invoke(boxmot, _arguments(data))
    assert repeated.exit_code == 0, (repeated.output, repeated.exception)
    assert (data.project / "val2/metrics.json").is_file()
    assert all((output / relative).read_bytes() == contents for relative, contents in originals.items())


def test_class_replay_retains_empty_frame_and_original_detection_indices(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    sequence = MultimodalSequence(
        data.sequence_inputs(),
        classes={"car": {"id": 1, "evaluation": "target"}, "pedestrian": {"id": 2, "evaluation": "target"}},
        fps=10.0,
        split="val",
    )
    trackers = {class_id: EagerMot(**profile) for class_id, profile in KITTI_PROFILES.items()}
    first = _track_frame(sequence[0], trackers).image_tracks
    assert first.class_ids.tolist() == [1, 2]
    assert first.detection_indices.tolist() == [1, 0]
    assert len(first.track_ids.unique()) == 2
    empty = sequence[1]
    assert len(empty.detections) == len(empty.detections_3d) == 0
    assert len(_track_frame(empty, trackers)) == 0
    assert all(tracker.frame_count == 2 for tracker in trackers.values())
    recovered = _track_frame(sequence[2], trackers).image_tracks
    torch.testing.assert_close(recovered.track_ids, first.track_ids)
    assert recovered.detection_indices.tolist() == [1, 0]
    assert all(tracker.frame_count == 3 for tracker in trackers.values())


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--sequence", "0001"], "absent from dataset split"),
        (["--sequence", "0002"], "duplicate"),
        (["--split", "test"], "has no split 'test'"),
    ],
)
def test_cli_rejects_split_mixing_duplicates_and_unannotated_test_split(
    tmp_path: Path, options: list[str], message: str
) -> None:
    data = _fixture(tmp_path)
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), *options])
    assert invocation.exit_code != 0
    assert message in invocation.output
    assert not data.project.exists()


def test_missing_ground_truth_fails_before_output_and_restores_threads(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    (data.ground_truth / "000002.png").unlink()
    previous_threads = torch.get_num_threads()
    invocation = CliRunner().invoke(boxmot, _arguments(data))
    assert invocation.exit_code == 1
    assert "Missing KITTI MOTS ground-truth instance PNG" in invocation.output
    assert not data.project.exists()
    assert torch.get_num_threads() == previous_threads


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "exactly 'car' and 'pedestrian'"),
        ({"car": {}}, "exactly 'car' and 'pedestrian'"),
        ({"car": [], "pedestrian": {}}, "must be a mapping"),
        ({"car": {"distance_treshold": 1}, "pedestrian": {}}, "Unknown EagerMOT car options"),
        ({"car": {}, "pedestrian": {"det_thresh": 1.1}}, "Invalid EagerMOT pedestrian configuration"),
        ({"car": {}, "pedestrian": {"min_hits": "2"}}, "Invalid EagerMOT pedestrian configuration"),
        ({"car": {"per_class": True}, "pedestrian": {}}, "replay already separates classes"),
    ],
)
def test_class_config_rejects_invalid_values_before_replay(tmp_path: Path, contents: object, message: str) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_kitti_profiles(path)


def test_class_overrides_preserve_other_class_and_do_not_mutate_presets(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text("car: {max_age: 6}\npedestrian: {distance_threshold: 0.5}\n", encoding="utf-8")
    profiles = load_kitti_profiles(path)
    assert profiles[1] == {**KITTI_PROFILES[1], "max_age": 6}
    assert profiles[2] == {**KITTI_PROFILES[2], "distance_threshold": 0.5}
    profiles[1]["det_thresh"] = 0.7
    assert load_kitti_profiles() == KITTI_PROFILES
    assert KITTI_PROFILES[1]["det_thresh"] == 0.0


def test_invalid_class_config_cli_fails_before_creating_results(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    path = tmp_path / "invalid.yaml"
    path.write_text("car: [\n", encoding="utf-8")
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--class-config", str(path)])
    assert invocation.exit_code == 1
    assert "Invalid EagerMOT class configuration" in invocation.output
    assert not data.project.exists()
