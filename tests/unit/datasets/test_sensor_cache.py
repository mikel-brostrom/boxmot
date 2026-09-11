"""Sensor replay caches retain source contracts while avoiding repeat decoding."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from boxmot.datasets.inputs import DatasetInputs, ModalityInput, SequenceInputs
from boxmot.datasets.sensor_cache import SensorReplayCacheError, open_sensor_sequence, prepare_sensor_sequence
from boxmot.datasets.sequence import MultimodalSequence


def _dataset(tmp_path: Path, *, frames: int = 3, annotations: bool = True) -> DatasetInputs:
    """Provide every modality with empty predictions, motion and annotation gaps."""
    images, spatial, masks = (tmp_path / name for name in ("images", "boxes3d", "annotations"))
    for directory in (images, spatial, masks):
        directory.mkdir(parents=True)
    for index in range(frames):
        Image.new("RGB", (4, 3), (index + 5, 10, 20)).save(images / f"{index:06d}.png")
        labels = np.zeros((3, 4), dtype=np.uint16)
        if index != 1:
            labels[0, :2] = 101  # Configured divisor is 100, not KITTI's 1000.
            labels[1, :2] = 205
            labels[2, 0] = 99  # Explicit ignored label.
            labels[2, 1] = 301  # Configured ignored class.
        assert cv2.imwrite(str(masks / f"{index:06d}.png"), labels)
        if index != 1:
            (spatial / f"{index:06d}.txt").write_text("Car -1 -1 0 0 0 4 3 1.5 2 4 5 6 20 0.25 0.9\n")
    detections = tmp_path / "detections.txt"
    detections.write_text(" ".join(["0", "0", "0", "4", "3", "0.9", "1", "3", "4", "0<", *(["0"] * 128)]) + "\n")
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("P2: 100 0 2 0.4 0 100 1.5 0.2 0 0 1 0.0027\n")
    poses = tmp_path / "poses.npy"
    matrices = np.repeat(np.eye(4)[None], frames, axis=0)
    matrices[:, 0, 3] = np.arange(frames)
    np.save(poses, matrices)
    gt3d = tmp_path / "labels.txt"
    gt3d.write_text(
        "0 7 Car 0 0 0 0 0 4 3 1.5 2 4 5 6 20 0.25\n0 -1 DontCare -1 -1 -10 0 0 4 3 -1 -1 -1 -1000 -1000 -1000 -10\n"
    )
    objects = tmp_path / "object_labels"
    objects.mkdir()
    for frame in range(frames):
        (objects / f"{frame:06d}.txt").write_text("" if frame == 1 else "Car 0.25 1 -0.1 0 0 4 3 1.5 2 4 5 6 20 0.25\n")
    modalities = {
        "images": ModalityInput("image-directory", (images,), {}),
        "detections_2d": ModalityInput("trackrcnn", (detections,), {}),
        "detections_3d": ModalityInput("kitti-detections", (spatial,), {}),
        "calibration": ModalityInput("kitti-p2", (calibration,), {}),
        "poses": ModalityInput("camera-to-world-npy", (poses,), {}),
    }
    if annotations:
        modalities.update(
            ground_truth=ModalityInput(
                "instance-png", (masks,), {"class_divisor": 100, "background_id": 0, "ignore_ids": [99]}
            ),
            ground_truth_3d=ModalityInput("kitti-tracking-labels", (gt3d,), {"ignore_classes": ["DontCare"]}),
            ground_truth_objects=ModalityInput("kitti-object-labels", (objects,), {}),
        )
    return DatasetInputs(
        config_path=None,
        id="custom",
        root=tmp_path,
        split="train",
        sequence_names=("drive-a",),
        sequences=(SequenceInputs("drive-a", modalities),),
        classes={
            "car": {"id": 1, "evaluation": "target"},
            "pedestrian": {"id": 2, "evaluation": "target"},
            "cyclist": {"id": 3, "evaluation": "ignore"},
        },
        fps=10.0,
    )


def _prepare(dataset: DatasetInputs, **kwargs) -> Path:
    return prepare_sensor_sequence(dataset, "drive-a", **kwargs)


@pytest.mark.parametrize("load_images", [False, True])
def test_saved_boxes_cache_preserves_absent_masks_and_empty_frames(tmp_path: Path, monkeypatch, load_images) -> None:
    """A boxes-only declaration must never decode, store or invent mask inputs."""
    dataset = _dataset(tmp_path)
    modalities = dataset.sequences[0].modalities
    dataset = replace(
        dataset,
        sequences=(
            SequenceInputs(
                "drive-a",
                {
                    "images": modalities["images"],
                    "detections_2d": replace(modalities["detections_2d"], options={"load_masks": False}),
                },
            ),
        ),
    )

    def forbid_mask_decoding(*args, **kwargs):
        raise AssertionError("Disabled detection masks must not be decoded.")

    monkeypatch.setattr("boxmot.datasets.readers.detections._decode_mask", forbid_mask_decoding)
    raw = MultimodalSequence(dataset.sequences[0], classes=dataset.classes, fps=dataset.fps, split=dataset.split)
    expected = list(raw)
    path = _prepare(dataset, load_images=load_images)
    index = json.loads((path / "index.json").read_text())
    assert "masks2d.bin" not in index["arrays"]
    assert "ground_truth.bin" not in index["arrays"]
    assert ("images.bin" in index["arrays"]) is load_images

    def forbid_parsing(*args, **kwargs):
        raise AssertionError("Warm replay must not parse detections again.")

    monkeypatch.setattr("boxmot.datasets.sensor_cache.MultimodalSequence", forbid_parsing)
    assert _prepare(dataset, load_images=load_images) == path
    with closing(open_sensor_sequence(path)) as cached:
        assert len(cached) == 3
        for actual, original in zip(cached, expected, strict=True):
            assert actual.detections.masks is original.detections.masks is None
            assert actual.detections.sample_id == original.detections.sample_id
            torch.testing.assert_close(actual.detections.geometry.values, original.detections.geometry.values)
            torch.testing.assert_close(actual.detections.scores, original.detections.scores)
            torch.testing.assert_close(actual.detections.class_ids, original.detections.class_ids)
            assert actual.camera is None
            assert len(actual.detections_3d) == 0
        assert [len(frame.detections) for frame in cached] == [1, 0, 0]
        if load_images:
            assert cached.read_image(2)[:, 0, 0].tolist() == [7, 10, 20]


def test_saved_boxes_cache_invalidates_reader_selection_and_detection_rows(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, annotations=False)
    masked_path = _prepare(dataset)
    modalities = dict(dataset.sequences[0].modalities)
    modalities["detections_2d"] = replace(modalities["detections_2d"], options={"load_masks": False})
    boxes_dataset = replace(dataset, sequences=(SequenceInputs("drive-a", modalities),))
    boxes_path = _prepare(boxes_dataset)
    assert boxes_path != masked_path
    source = modalities["detections_2d"].paths[0]
    source.write_text(source.read_text().replace("0.9", "0.8"))
    changed_path = _prepare(boxes_dataset)
    assert changed_path != boxes_path
    with closing(open_sensor_sequence(changed_path)) as cached:
        assert cached[0].detections.masks is None
        assert cached[0].detections.scores.tolist() == pytest.approx([0.8])
    with closing(open_sensor_sequence(masked_path)) as cached:
        assert cached[0].detections.masks is not None
        assert cached[0].detections.scores.tolist() == pytest.approx([0.9])


def test_sensor_cache_preserves_all_modalities_and_empty_frames(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    raw = MultimodalSequence(dataset.sequences[0], classes=dataset.classes, fps=dataset.fps, split=dataset.split)
    cached = open_sensor_sequence(_prepare(dataset, load_images=True))
    try:
        assert len(raw) == len(cached) == 3
        assert cached.frame_paths == raw.frame_paths
        assert cached.image_size == raw.image_size
        assert cached.fps == raw.fps
        assert cached.missing_3d_frames == raw.missing_3d_frames
        for actual, expected in zip(cached, raw, strict=True):
            assert actual.frame_index == expected.frame_index
            assert actual.timestamp_s == expected.timestamp_s
            for field in ("detections", "detections_3d"):
                left, right = getattr(actual, field), getattr(expected, field)
                assert left.sample_id == right.sample_id
                torch.testing.assert_close(left.geometry.values, right.geometry.values)
                torch.testing.assert_close(left.class_ids, right.class_ids)
                torch.testing.assert_close(left.scores, right.scores)
            torch.testing.assert_close(actual.detections.masks.values, expected.detections.masks.values)
            torch.testing.assert_close(actual.camera.projection, expected.camera.projection)
            torch.testing.assert_close(actual.camera.camera_to_world, expected.camera.camera_to_world)
        assert cached[-1].frame_index == 2
        assert [frame.frame_index for frame in cached[1:]] == [1, 2]
        assert cached.read_image(2)[:, 0, 0].tolist() == [7, 10, 20]
        labels = cached.ground_truth_3d()
        assert labels.frame_indices.tolist() == [0]
        assert labels.track_ids.tolist() == [7]
        np.testing.assert_equal(labels.boxes, [[5, 6, 20, 0.25, 4, 2, 1.5]])
        assert labels.source_rows == tuple((tmp_path / "labels.txt").read_text().splitlines())
        assert labels.row_count == 2
        objects = cached.ground_truth_objects()
        assert objects.frame_rows[1] == ()
        assert objects.frame_rows[0][0].split()[1] == "0.25"
        assert len(objects.source_sha256) == 64
        assert cached.source_sha256(tmp_path / "labels.txt") == labels.source_sha256
    finally:
        cached.close()


def test_sensor_ground_truth_preserves_configured_encoding_and_ignored_pixels(tmp_path: Path) -> None:
    cached = open_sensor_sequence(_prepare(_dataset(tmp_path)))
    try:
        ids, classes, encoded, ignore = cached.ground_truth(0)
        assert ids.dtype == classes.dtype == np.int64
        assert ids.tolist() == [101, 205]
        assert classes.tolist() == [1, 2]
        decoded = mask_utils.decode(encoded)
        assert decoded.shape == (3, 4, 2)
        assert decoded[:, :, 0].sum() == decoded[:, :, 1].sum() == 2
        np.testing.assert_equal(mask_utils.decode(ignore)[2], [1, 1, 0, 0])
        ids, classes, encoded, ignore = cached.ground_truth(1)
        assert ids.shape == classes.shape == (0,)
        assert encoded == []
        assert ignore is None
    finally:
        cached.close()


def test_sensor_cache_warm_reuse_and_replay_never_decode_sources(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset(tmp_path)
    path = _prepare(dataset, load_images=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached replay must not decode any source modality")

    for name in (
        "MultimodalSequence",
        "read_instance_png",
        "read_kitti_tracking_labels",
        "read_kitti_object_labels",
        "read_rgb_chw_uint8",
        "_file_digest",
    ):
        monkeypatch.setattr(f"boxmot.datasets.sensor_cache.{name}", forbidden)
    assert _prepare(dataset, load_images=True) == path
    cached = open_sensor_sequence(path)
    try:
        cached.validate(dataset=dataset)
        assert len(list(cached)) == 3
        assert cached.read_image(0).shape == (3, 3, 4)
        assert cached.ground_truth(0)[0].tolist() == [101, 205]
        assert cached.ground_truth_3d().track_ids.tolist() == [7]
        assert cached.ground_truth_3d().source_rows[1].split()[2] == "DontCare"
        assert cached.ground_truth_objects().frame_rows[1] == ()
    finally:
        cached.close()


def test_sensor_cache_supports_images_directly_in_dataset_root(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    for image in (tmp_path / "images").glob("*.png"):
        image.rename(tmp_path / image.name)
    modalities = dict(dataset.sequences[0].modalities)
    modalities["images"] = replace(modalities["images"], paths=(tmp_path,))
    dataset = replace(dataset, sequences=(replace(dataset.sequences[0], modalities=modalities),))

    path = _prepare(dataset, load_images=True)
    assert _prepare(dataset, load_images=True) == path
    cached = open_sensor_sequence(path)
    try:
        assert cached.frame_paths == tuple(tmp_path / f"{frame:06d}.png" for frame in range(3))
        assert cached.read_image(2)[:, 0, 0].tolist() == [7, 10, 20]
    finally:
        cached.close()


def test_sensor_cache_copies_are_owned_and_survive_close(tmp_path: Path) -> None:
    cached = open_sensor_sequence(_prepare(_dataset(tmp_path), load_images=True))
    first = cached[0]
    first.detections.geometry.values[:] = -999
    first.detections.scores[:] = 0
    first.detections.masks.values[:] = False
    first.detections_3d.geometry.values[:] = 999
    first.camera.projection[:] = 0
    first.camera.camera_to_world[:] = 0
    image = cached.read_image(0)
    image[:] = 0
    cached.ground_truth_3d().boxes[:] = 0
    ids, _, masks, ignore = cached.ground_truth(0)
    ids[:] = 0
    masks[0]["counts"] = b"garbage"
    ignore["size"][0] = -1
    second = cached[0]
    assert second.detections.geometry.values[0].tolist() == [0, 0, 4, 3]
    assert second.detections.masks.values.all()
    assert second.detections_3d.geometry.values[0, 0] == 5
    assert second.camera.projection[0, 0] == 100
    assert second.camera.camera_to_world[0, 0] == 1
    assert cached.read_image(0)[0, 0, 0] == 5
    assert cached.ground_truth_3d().boxes[0, 0] == 5
    assert cached.ground_truth(0)[0].tolist() == [101, 205]
    cached.close()
    cached.close()
    assert second.detections.masks.values.all()
    with pytest.raises(SensorReplayCacheError, match="closed"):
        cached[0]
    with pytest.raises(SensorReplayCacheError, match="closed"):
        cached.validate()


def test_sensor_cache_pickles_only_path_and_reopens_independent_maps(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cached = open_sensor_sequence(_prepare(dataset))
    serialized = pickle.dumps(cached)
    assert len(serialized) < 1000
    clone = pickle.loads(serialized)
    cached.close()
    try:
        assert clone[0].detections.masks.values.all()
        clone.validate(dataset=dataset)
    finally:
        clone.close()


@pytest.mark.parametrize(
    "role",
    [
        "images",
        "detections_2d",
        "detections_3d",
        "calibration",
        "poses",
        "ground_truth",
        "ground_truth_3d",
        "ground_truth_objects",
    ],
)
def test_sensor_cache_invalidates_edited_modality(tmp_path: Path, role: str) -> None:
    dataset = _dataset(tmp_path)
    old_path = _prepare(dataset)
    source = dataset.sequences[0].modalities[role].paths[0]
    if source.is_dir():
        source = sorted(source.iterdir())[0]
    # Even a byte-identical rewrite receives a new content mutation epoch.
    source.write_bytes(source.read_bytes())
    assert _prepare(dataset) != old_path


def test_sensor_cache_invalidates_added_and_removed_prediction_frames(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    old_path = _prepare(dataset)
    prediction = tmp_path / "boxes3d/000001.txt"
    prediction.write_text((tmp_path / "boxes3d/000000.txt").read_text())
    added_path = _prepare(dataset)
    assert added_path != old_path
    cached = open_sensor_sequence(added_path)
    assert len(cached[1].detections_3d) == 1
    cached.close()
    prediction.unlink()
    removed_path = _prepare(dataset)
    assert removed_path != added_path
    cached = open_sensor_sequence(removed_path)
    assert len(cached[1].detections_3d) == 0
    cached.close()


@pytest.mark.parametrize("change", ["fps", "split", "classes", "options", "images"])
def test_sensor_cache_authored_context_invalidates_reuse(tmp_path: Path, change: str) -> None:
    dataset = _dataset(tmp_path)
    old_path = _prepare(dataset)
    cached = open_sensor_sequence(old_path)
    if change == "fps":
        updated = replace(dataset, fps=20.0)
    elif change == "split":
        updated = replace(dataset, split="val")
    elif change == "classes":
        updated = replace(
            dataset, classes={**dataset.classes, "car": {**dataset.classes["car"], "display_name": "Cars"}}
        )
    elif change == "options":
        modalities = dict(dataset.sequences[0].modalities)
        modalities["detections_3d"] = replace(modalities["detections_3d"], options={"score_transform": "odds"})
        updated = replace(dataset, sequences=(replace(dataset.sequences[0], modalities=modalities),))
    else:
        updated = dataset
    try:
        if change != "images":
            with pytest.raises(SensorReplayCacheError, match="configuration"):
                cached.validate(dataset=updated)
        assert _prepare(updated, load_images=change == "images") != old_path
    finally:
        cached.close()


@pytest.mark.parametrize(
    "filename",
    [
        "boxes3d.bin",
        "masks2d.bin",
        "ground_truth.bin",
        "gt3d_source_rows.bin",
        "gt_object_rows.bin",
        "index.json",
        "_SUCCESS",
    ],
)
def test_sensor_cache_rebuilds_corrupt_entries(tmp_path: Path, filename: str) -> None:
    dataset = _dataset(tmp_path)
    path = _prepare(dataset)
    corrupted = path / filename
    content = corrupted.read_bytes()
    corrupted.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(SensorReplayCacheError):
        open_sensor_sequence(path)
    assert _prepare(dataset) == path
    cached = open_sensor_sequence(path)
    try:
        assert cached[0].detections_3d.geometry.values[0, 0] == 5
    finally:
        cached.close()


def test_sensor_cache_optional_modalities_and_rgb_selection(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset(tmp_path, annotations=False)
    images = dataset.sequences[0].modalities["images"]
    dataset = replace(dataset, sequences=(SequenceInputs("drive-a", {"images": images}),))

    def forbidden(*args, **kwargs):
        raise AssertionError("RGB was not requested")

    monkeypatch.setattr("boxmot.datasets.sensor_cache.read_rgb_chw_uint8", forbidden)
    cached = open_sensor_sequence(_prepare(dataset))
    try:
        frame = cached[0]
        assert frame.camera is None
        assert len(frame.detections) == len(frame.detections_3d) == 0
        assert cached.ground_truth(0) is None
        assert cached.ground_truth_3d() is None
        assert cached.ground_truth_objects() is None
        with pytest.raises(SensorReplayCacheError, match="load_images=True"):
            cached.read_image(0)
    finally:
        cached.close()


def test_sensor_cache_annotation_failure_never_publishes_partial_entry(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (tmp_path / "annotations/000001.png").unlink()
    with pytest.raises(ValueError, match="Unable to decode instance PNG"):
        _prepare(dataset)
    assert not list((tmp_path / ".boxmot/replay_cache").glob("*/_SUCCESS"))


def test_sensor_cache_requires_object_labels_for_every_image(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (tmp_path / "object_labels/000001.txt").unlink()
    with pytest.raises(ValueError, match="align to every sequence image"):
        _prepare(dataset)
    assert not list((tmp_path / ".boxmot/replay_cache").glob("*/_SUCCESS"))


def test_sensor_cache_preserves_object_metadata_snapshot_across_source_edits(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    old_path = _prepare(dataset)
    cached = open_sensor_sequence(old_path)
    original = cached.ground_truth_objects()
    source = tmp_path / "object_labels/000000.txt"
    source.write_text(source.read_text().replace("Car 0.25", "Car 0.75"))
    new_path = _prepare(dataset)
    refreshed = open_sensor_sequence(new_path)
    try:
        assert new_path != old_path
        assert cached.ground_truth_objects() == original
        assert refreshed.ground_truth_objects().frame_rows[0][0].split()[1] == "0.75"
        assert refreshed.ground_truth_objects().source_sha256 != original.source_sha256
        assert isinstance(original.frame_rows, tuple) and isinstance(original.frame_rows[0], tuple)
    finally:
        cached.close()
        refreshed.close()
    assert original.frame_rows[0][0].split()[1] == "0.25"
    with pytest.raises(SensorReplayCacheError, match="closed"):
        cached.ground_truth_objects()


def test_sensor_cache_concurrent_preparation_publishes_one_complete_entry(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(_prepare, (dataset, dataset)))
    assert paths[0] == paths[1]
    cached = open_sensor_sequence(paths[0])
    try:
        assert len(cached) == 3
        index = json.loads((paths[0] / "index.json").read_text())
        assert index["arrays"]["masks2d.bin"]["shape"] == [1, 2]
    finally:
        cached.close()


def test_sensor_cache_spatial_access_does_not_unpack_2d_masks(tmp_path: Path, monkeypatch) -> None:
    cached = open_sensor_sequence(_prepare(_dataset(tmp_path)))

    def forbidden(*args, **kwargs):
        raise AssertionError("3D calibration must not unpack 2D masks")

    monkeypatch.setattr("boxmot.datasets.sensor_cache.np.unpackbits", forbidden)
    try:
        assert len(cached.read_spatial(0)) == 1
        assert cached.read_camera(2).camera_to_world[0, 3] == 2
    finally:
        cached.close()


def test_sensor_cache_reuses_sequence_when_other_selected_sequences_change(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    path = _prepare(dataset)
    updated = replace(dataset, sequence_names=("drive-a", "drive-b"))
    assert _prepare(updated) == path
    cached = open_sensor_sequence(path)
    try:
        cached.validate(dataset=updated)
    finally:
        cached.close()


@pytest.mark.parametrize("change", ["dtype", "shape", "offset", "frame", "extra_array", "missing_array"])
def test_sensor_cache_validates_array_and_frame_schema(tmp_path: Path, change: str) -> None:
    path = _prepare(_dataset(tmp_path))
    index = json.loads((path / "index.json").read_text())
    if change == "dtype":
        index["arrays"]["boxes3d.bin"]["dtype"] = "<u4"
    elif change == "shape":
        index["arrays"]["boxes3d.bin"]["shape"] = []
    elif change == "offset":
        index["frames"][0]["detections_3d"] = [1, 2]
    elif change == "frame":
        index["frames"][1]["frame_index"] = 0
    elif change == "extra_array":
        index["arrays"]["unpublished.bin"] = index["arrays"]["boxes3d.bin"]
    else:
        del index["arrays"]["masks2d.bin"]
    payload = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    (path / "index.json").write_bytes(payload)
    (path / "_SUCCESS").write_text(
        json.dumps({"schema": index["schema"], "index_sha256": hashlib.sha256(payload).hexdigest()})
    )
    with pytest.raises(SensorReplayCacheError):
        open_sensor_sequence(path)


def test_sensor_cache_entry_cannot_be_substituted_under_another_identity(tmp_path: Path) -> None:
    path = _prepare(_dataset(tmp_path))
    destination = path.with_name("incorrect-identity")
    shutil.copytree(path, destination)
    with pytest.raises(SensorReplayCacheError, match="identity"):
        open_sensor_sequence(destination)


def test_sensor_cache_missing_declared_source_cannot_use_warm_entry(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    _prepare(dataset)
    (tmp_path / "poses.npy").unlink()
    with pytest.raises(SensorReplayCacheError, match="missing"):
        _prepare(dataset)
