"""Run the public TrackR-CNN replay command with real MAF-HDA and mask metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from PIL import Image

from boxmot.engine.cli import boxmot
from boxmot.engine.eval.mots_io import read_mots_results

mask_utils = pytest.importorskip("pycocotools.mask")


def _fixture(tmp_path: Path) -> SimpleNamespace:
    """Create only RGB images, instance GT, and raw 138-field TrackR-CNN rows."""
    detections, images, instances = (tmp_path / name for name in ("detections", "images", "instances"))
    detections.mkdir()
    (images / "0002").mkdir(parents=True)
    (instances / "0002").mkdir(parents=True)
    height, width = 40, 64
    objects = ((2, (36, 8, 46, 30), 0.98), (1, (8, 8, 26, 25), 0.9))
    texture = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    rows = []
    for frame_index in range(5):
        labels = np.zeros((height, width), dtype=np.uint16)
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_index in (0, 1, 3):
            for class_id, bounds, score in objects:
                x1, y1, x2, y2 = bounds
                labels[y1:y2, x1:x2] = class_id * 1000 + 1
                pixels[y1:y2, x1:x2] = texture[y1:y2, x1:x2]
                mask = np.asfortranarray(labels == class_id * 1000 + 1, dtype=np.uint8)
                encoded = mask_utils.encode(mask)
                fields = [
                    frame_index,
                    *bounds,
                    score,
                    class_id,
                    height,
                    width,
                    encoded["counts"].decode("ascii"),
                    *([0.125] * 128),
                ]
                rows.append(" ".join(map(str, fields)))
        if frame_index == 1:
            empty = mask_utils.encode(np.zeros((height, width), dtype=np.uint8, order="F"))
            fields = [
                frame_index,
                50,
                5,
                60,
                15,
                0.999,
                1,
                height,
                width,
                empty["counts"].decode("ascii"),
                *([0.125] * 128),
            ]
            rows.append(" ".join(map(str, fields)))
        Image.fromarray(pixels).save(images / "0002" / f"{frame_index:06d}.png")
        Image.fromarray(labels).save(instances / "0002" / f"{frame_index:06d}.png")
    (detections / "0002.txt").write_text("\n".join(rows) + "\n")
    return SimpleNamespace(detections=detections, images=images, instances=instances, project=tmp_path / "results")


def _arguments(data: SimpleNamespace, *, tracker: str = "maf_hda") -> list[str]:
    """Use the registered command without any 3D inputs or model weights."""
    return [
        "track",
        "--tracker",
        tracker,
        "--detections",
        str(data.detections),
        "--images",
        str(data.images),
        "--instances",
        str(data.instances),
        "--project",
        str(data.project),
        "--sequence",
        "0002",
    ]


def test_cli_evaluates_masks_and_preserves_class_identities_across_missing_observations(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    previous_threads = torch.get_num_threads()
    invocation = CliRunner().invoke(boxmot, _arguments(data))
    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    assert torch.get_num_threads() == previous_threads
    output = data.project / "val"
    assert f"Results: {output}" in invocation.output
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["tracker"] == "maf_hda"
    assert manifest["tracker_backend"] == "python"
    assert manifest["per_class"] is True
    assert manifest["sequences"] == {"0002": 5}
    assert manifest["dropped_empty_masks"] == {"0002": 1}
    assert manifest["tracker_options"]["det_thresh"] == 0.7

    metrics = json.loads((output / "metrics.json").read_text())
    assert set(metrics) == {"car", "pedestrian", "cls_comb_cls_av", "cls_comb_det_av"}
    for class_name in ("car", "pedestrian"):
        values = metrics[class_name]
        for key in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
            assert values[key] == pytest.approx(100), (class_name, key, values[key])
        assert values["Frames"] == 5
        assert values["Dets"] == values["GT_Dets"] == 3
        assert values["IDs"] == values["GT_IDs"] == 1
        assert set(values["per_sequence"]) == {"0002"}
        assert all(np.isfinite(value) for key, value in values.items() if key != "per_sequence")
    with (output / "metrics.csv").open(newline="") as handle:
        assert {row["class"] for row in csv.DictReader(handle)} == set(metrics)

    predictions = read_mots_results(output / "mots/0002.txt")
    assert set(predictions) == {0, 1, 3}
    identities = {row.class_id: row.track_id for row in predictions[0]}
    assert set(identities) == {1, 2}
    assert len(set(identities.values())) == 2
    for frame_index, rows in predictions.items():
        assert {row.class_id: row.track_id for row in rows} == identities
        labels = np.asarray(Image.open(data.instances / "0002" / f"{frame_index:06d}.png"))
        for row in rows:
            decoded = mask_utils.decode(row.encoded_mask.rle).astype(bool)
            np.testing.assert_array_equal(decoded, labels == row.class_id * 1000 + 1)

    originals = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    repeated = CliRunner().invoke(boxmot, _arguments(data))
    assert repeated.exit_code == 0, (repeated.output, repeated.exception)
    assert (data.project / "val2/metrics.json").is_file()
    assert all((output / relative).read_bytes() == contents for relative, contents in originals.items())


def test_custom_scalar_tracker_config_filters_low_confidence_observations(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    config = tmp_path / "confidence.yaml"
    config.write_text("det_thresh: 0.95\n")
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--tracker-config", str(config)])
    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    output = data.project / "val"
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["tracker_options"]["det_thresh"] == 0.95
    assert manifest["dropped_empty_masks"] == {"0002": 1}
    predictions = read_mots_results(output / "mots/0002.txt")
    assert set(predictions) == {0, 1, 3}
    assert {row.class_id for rows in predictions.values() for row in rows} == {2}
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["car"]["Dets"] == 0
    assert metrics["car"]["GT_Dets"] == 3
    assert metrics["car"]["HOTA"] == 0
    assert metrics["pedestrian"]["HOTA"] == pytest.approx(100)


def test_empty_image_timestep_advances_expiry_even_without_a_detection_row(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    config = tmp_path / "expiry.yaml"
    config.write_text("max_age: 1\n")
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--tracker-config", str(config)])
    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    predictions = read_mots_results(data.project / "val/mots/0002.txt")
    assert set(predictions) == {0, 1, 3}
    first_ids = {row.class_id: row.track_id for row in predictions[0]}
    assert {row.class_id: row.track_id for row in predictions[1]} == first_ids
    recovered_ids = {row.class_id: row.track_id for row in predictions[3]}
    assert all(recovered_ids[class_id] != track_id for class_id, track_id in first_ids.items())
    metrics = json.loads((data.project / "val/metrics.json").read_text())
    assert all(metrics[class_name]["Frames"] == 5 for class_name in ("car", "pedestrian"))


def test_missing_ground_truth_fails_before_output_and_restores_threads(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    (data.instances / "0002/000004.png").unlink()
    previous_threads = torch.get_num_threads()
    invocation = CliRunner().invoke(boxmot, _arguments(data))
    assert invocation.exit_code == 1
    assert "Missing KITTI MOTS ground-truth instance PNG" in invocation.output
    assert not data.project.exists()
    assert torch.get_num_threads() == previous_threads


def test_eagermot_requires_sensor_inputs_that_this_command_cannot_supply(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    invocation = CliRunner().invoke(boxmot, _arguments(data, tracker="eagermot"))
    assert invocation.exit_code == 1
    assert "requires 3D detections and a CameraModel" in invocation.output
    assert not data.project.exists()
