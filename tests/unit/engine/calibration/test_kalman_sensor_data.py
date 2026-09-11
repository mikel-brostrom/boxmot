"""Calibration uses real 3D identities, saved predictions, and runtime ego poses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from boxmot.datasets.inputs import DatasetInputs, ModalityInput, SequenceInputs
from boxmot.engine.calibration import kalman_sensor_data
from boxmot.engine.calibration.kalman_sensor_data import load_sensor_calibration_data
from boxmot.trackers.eagermot.geometry import transform_boxes3d


def _label(frame: int, identity: int, x: float, *, label: str = "Car") -> str:
    """Write true KITTI 3D annotations with identity and no prediction score."""
    return f"{frame} {identity} {label} 0 0 0 0 0 4 3 1.5 2 4 {x} 1.5 20 0.25\n"


def _detection(x: float, *, label: str = "Car", score: float = 3) -> str:
    """Write the independent 16-field detector format using odds scores."""
    return f"{label} -1 -1 0 0 0 4 3 1.5 2 4 {x} 1.5 20 0.25 {score}\n"


def _dataset(tmp_path: Path) -> DatasetInputs:
    """Build moving-camera observations with both annotation and detector gaps."""
    images = tmp_path / "images"
    images.mkdir()
    for index in range(4):
        # Timeline metadata suffices: reading image pixels would fail.
        (images / f"{index:06d}.png").write_bytes(b"not decoded during calibration")
    ground_truth = tmp_path / "annotations.txt"
    ground_truth.write_text(
        _label(0, 0, 10)
        + _label(0, 0, 10, label="Pedestrian")
        + _label(1, 0, 9)
        + _label(3, 0, 7)
        + "0 -1 DontCare -1 -1 -10 0 0 4 3 -1 -1 -1 -1000 -1000 -1000 -10\n"
    )
    projection = tmp_path / "calibration.txt"
    projection.write_text("P2: 100 0 2 0.4 0 100 1.5 0.2 0 0 1 0.0027\n")
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = np.arange(4)
    np.save(tmp_path / "poses.npy", poses)
    predictions = tmp_path / "detections"
    predictions.mkdir()
    (predictions / "000000.txt").write_text(_detection(10.1) + _detection(10.2, label="Pedestrian", score=1))
    (predictions / "000002.txt").write_text(_detection(8.1))
    (predictions / "000003.txt").write_text(_detection(7.1) + _detection(100, label="Pedestrian"))
    modalities = {
        "images": ModalityInput("image-directory", (images,), {}),
        "ground_truth_3d": ModalityInput("kitti-tracking-labels", (ground_truth,), {"ignore_classes": ["DontCare"]}),
        "calibration": ModalityInput("kitti-p2", (projection,), {}),
        "poses": ModalityInput("camera-to-world-npy", (tmp_path / "poses.npy",), {}),
        "detections_3d": ModalityInput("kitti-detections", (predictions,), {"score_transform": "odds"}),
        # Annotation and mask readers must never inspect these missing files.
        "ground_truth": ModalityInput("instance-png", (tmp_path / "mask-ground-truth",), {}),
        "detections_2d": ModalityInput("trackrcnn", (tmp_path / "mask-detections.txt",), {}),
    }
    return DatasetInputs(
        config_path=None,
        id="calibration-drive",
        root=tmp_path,
        split="train",
        sequence_names=("drive",),
        sequences=(SequenceInputs("drive", modalities),),
        fps=20.0,
        classes={"car": {"id": 1, "evaluation": "target"}, "pedestrian": {"id": 2, "evaluation": "target"}},
    )


def test_sensor_calibration_preserves_ego_compensation_identities_and_gaps(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    messages: list[str] = []

    data = load_sensor_calibration_data(dataset, progress=messages.append)

    assert messages == ["KF calibration: matching saved 3D detections to GT for drive…"]
    assert len(data.tracks) == 2
    car, pedestrian = data.tracks
    assert (car.track_id, car.class_id, pedestrian.track_id, pedestrian.class_id) == (0, 1, 0, 2)
    np.testing.assert_array_equal(car.frame_indices, [0, 1, 3])
    np.testing.assert_allclose(car.timestamps_s, [0, 0.05, 0.15])
    np.testing.assert_allclose(car.gt_boxes[:, 0], [10, 10, 10])
    np.testing.assert_allclose(car.detection_boxes[[0, 2], 0], [10.1, 10.1])
    assert np.isnan(car.detection_boxes[1]).all()
    np.testing.assert_allclose(car.scores[[0, 2]], 0.75)
    assert np.isnan(car.scores[1])
    assert pedestrian.detection_boxes[0, 0] == pytest.approx(10.2)
    assert pedestrian.scores[0] == pytest.approx(0.5)
    assert data.statistics == {
        "frames": 4,
        "detections": 5,
        "target_detections": 5,
        "ground_truth_rows": 5,
        "ground_truth": 4,
        "filtered_ground_truth": 1,
        "matched": 3,
        "unmatched_ground_truth": 1,
        "unmatched_detections": 2,
        "trajectories": 2,
    }
    assert data.match_iou == 0.5


def test_camera_matching_precedes_the_same_world_transform_as_eager_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path)
    pose = np.array([[0, 0, 1, 30], [0, 1, 0, 5], [-1, 0, 0, -7], [0, 0, 0, 1]], dtype=np.float32)
    np.save(tmp_path / "poses.npy", np.repeat(pose[None], 4, axis=0))
    original = kalman_sensor_data.iou3d_matrix
    matched_camera_boxes: list[np.ndarray] = []

    def capture_camera_boxes(gt: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        matched_camera_boxes.append(gt.copy())
        return original(gt, predictions)

    monkeypatch.setattr(kalman_sensor_data, "iou3d_matrix", capture_camera_boxes)

    data = load_sensor_calibration_data(dataset)

    assert matched_camera_boxes[0][0, 0] == 10
    assert matched_camera_boxes[0][0, 2] == 20
    camera_gt = np.array([[10, 1.5, 20, 0.25, 4, 2, 1.5]])
    np.testing.assert_allclose(data.tracks[0].gt_boxes[0], transform_boxes3d(camera_gt, pose)[0])
    assert data.tracks[0].gt_boxes[0, 3] == pytest.approx(0.25 + np.pi / 2)


def test_matching_is_one_to_one_class_specific_and_uses_vertical_3d_overlap(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (tmp_path / "annotations.txt").write_text(
        _label(0, 0, 10) + _label(0, 1, 10.1) + _label(0, 0, 10, label="Pedestrian")
    )
    (tmp_path / "detections/000000.txt").write_text(
        _detection(10) + _detection(10, label="Pedestrian").replace("10 1.5 20", "10 10 20")
    )

    data = load_sensor_calibration_data(dataset)

    assert data.statistics["matched"] == 1
    assert sum(np.isfinite(track.scores).sum() for track in data.tracks) == 1
    assert np.isnan(data.tracks[-1].detection_boxes).all()


@pytest.mark.parametrize("missing", ["ground_truth_3d", "poses", "calibration", "detections_3d"])
def test_missing_calibration_inputs_fail_before_any_sensor_payload_read(tmp_path: Path, missing: str) -> None:
    dataset = _dataset(tmp_path)
    modalities = dict(dataset.sequences[0].modalities)
    del modalities[missing]
    dataset = replace(dataset, sequences=(SequenceInputs("drive", modalities),))
    (tmp_path / "poses.npy").write_bytes(b"invalid")

    with pytest.raises(ValueError, match=f"requires {missing}.*drive"):
        load_sensor_calibration_data(dataset)


def test_prediction_rows_cannot_substitute_for_3d_ground_truth(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (tmp_path / "annotations.txt").write_text(_detection(10))

    with pytest.raises(ValueError, match="17 KITTI tracking label fields"):
        load_sensor_calibration_data(dataset)


@pytest.mark.parametrize("role", ["ground_truth_3d", "poses", "calibration", "detections_3d", "metadata"])
def test_provenance_changes_when_a_calibration_source_changes(tmp_path: Path, role: str) -> None:
    dataset = _dataset(tmp_path)
    before = load_sensor_calibration_data(dataset)
    if role == "ground_truth_3d":
        path = tmp_path / "annotations.txt"
        path.write_text(path.read_text().replace("10 1.5 20", "10.1 1.5 20"))
    elif role == "poses":
        poses = np.load(tmp_path / "poses.npy")
        poses[:, 0, 3] += 1
        np.save(tmp_path / "poses.npy", poses)
    elif role == "calibration":
        (tmp_path / "calibration.txt").write_text("P2: 101 0 2 0.4 0 100 1.5 0.2 0 0 1 0.0027\n")
    elif role == "detections_3d":
        (tmp_path / "detections/000002.txt").write_text(_detection(8.2))
    else:
        dataset = replace(dataset, fps=10.0)

    after = load_sensor_calibration_data(dataset)

    if role == "ground_truth_3d":
        assert before.ground_truth_sources[0]["sha256"] != after.ground_truth_sources[0]["sha256"]
    else:
        before_hash = next(source["sha256"] for source in before.input_sources if source["role"] == role)
        after_hash = next(source["sha256"] for source in after.input_sources if source["role"] == role)
        assert before_hash != after_hash


def test_provenance_includes_prediction_presence_even_for_empty_frames(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    before = load_sensor_calibration_data(dataset)
    (tmp_path / "detections/000001.txt").touch()

    after = load_sensor_calibration_data(dataset)

    assert before.statistics == after.statistics
    assert before.input_sources != after.input_sources


def test_calibration_uses_all_cached_spatial_inputs_without_decoding_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached and direct calibration preserve exact samples and source provenance."""
    import cv2

    from boxmot.datasets.sensor_cache import open_sensor_sequence, prepare_sensor_sequence

    dataset = _dataset(tmp_path)
    # Calibration-only dataset: image metadata defines time, spatial annotations
    # and predictions provide all observations; image masks are not required.
    modalities = {
        role: modality
        for role, modality in dataset.sequences[0].modalities.items()
        if role not in {"ground_truth", "detections_2d"}
    }
    dataset = replace(dataset, sequences=(SequenceInputs("drive", modalities),))
    for path in (tmp_path / "images").glob("*.png"):
        assert cv2.imwrite(str(path), np.zeros((3, 4, 3), dtype=np.uint8))
    expected = load_sensor_calibration_data(dataset)
    path = prepare_sensor_sequence(dataset, "drive")
    cached = open_sensor_sequence(path)

    def unexpected(*args, **kwargs):
        pytest.fail("Calibration must reuse cached spatial data and source hashes")

    for reader in ("read_camera_to_world_poses", "read_kitti_projection", "read_kitti_tracking_labels", "_file_digest"):
        monkeypatch.setattr(kalman_sensor_data, reader, unexpected)
    monkeypatch.setattr(kalman_sensor_data.KittiDetections3D, "read", unexpected)
    monkeypatch.setattr(type(cached), "__getitem__", unexpected)
    try:
        actual = load_sensor_calibration_data(dataset, cached_sequences={"drive": cached})
        assert actual.statistics == expected.statistics
        assert actual.ground_truth_sources == expected.ground_truth_sources
        assert actual.input_sources == expected.input_sources
        for direct, mapped in zip(expected.tracks, actual.tracks, strict=True):
            assert (direct.sequence_id, direct.class_id, direct.track_id) == (
                mapped.sequence_id,
                mapped.class_id,
                mapped.track_id,
            )
            for field in ("frame_indices", "timestamps_s", "gt_boxes", "detection_boxes", "scores"):
                np.testing.assert_array_equal(getattr(direct, field), getattr(mapped, field))
        with pytest.raises(ValueError, match="configuration"):
            load_sensor_calibration_data(replace(dataset, fps=2.0), cached_sequences={"drive": cached})
    finally:
        cached.close()


@pytest.mark.parametrize("change", ["add", "remove", "replace", "remove-linked"])
def test_cached_calibration_provenance_uses_its_immutable_source_snapshot(tmp_path: Path, change: str) -> None:
    """Live file changes must not alter provenance of already cached observations."""
    import cv2

    from boxmot.datasets.sensor_cache import open_sensor_sequence, prepare_sensor_sequence

    dataset = _dataset(tmp_path)
    modalities = {
        role: modality
        for role, modality in dataset.sequences[0].modalities.items()
        if role not in {"ground_truth", "detections_2d"}
    }
    dataset = replace(dataset, sequences=(SequenceInputs("drive", modalities),))
    for path in (tmp_path / "images").glob("*.png"):
        assert cv2.imwrite(str(path), np.zeros((3, 4, 3), dtype=np.uint8))
    source = tmp_path / "detections/000000.txt"
    if change == "remove-linked":
        target = tmp_path / "prediction-target.txt"
        source.rename(target)
        source.symlink_to(target)
    expected = load_sensor_calibration_data(dataset)
    cached = open_sensor_sequence(prepare_sensor_sequence(dataset, "drive"))
    try:
        if change == "add":
            (tmp_path / "detections/000001.txt").write_text(_detection(9.5))
        elif change == "replace":
            source.write_text(_detection(500))
        else:
            source.unlink()
        actual = load_sensor_calibration_data(dataset, cached_sequences={"drive": cached})
        assert actual.statistics == expected.statistics
        assert actual.input_sources == expected.input_sources
        assert actual.ground_truth_sources == expected.ground_truth_sources
        for direct, mapped in zip(expected.tracks, actual.tracks, strict=True):
            for field in ("frame_indices", "timestamps_s", "gt_boxes", "detection_boxes", "scores"):
                np.testing.assert_array_equal(getattr(direct, field), getattr(mapped, field))
    finally:
        cached.close()
