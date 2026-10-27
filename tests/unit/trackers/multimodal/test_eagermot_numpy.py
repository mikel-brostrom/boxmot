"""Public NumPy sensor fusion agrees with canonical EagerMOT tracking."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from boxmot import EagerMot, EagerMotConfig
from boxmot.structures import (
    Boxes,
    Boxes3D,
    CameraModel,
    Detections,
    Detections3D,
    Frame,
    MultimodalTracks,
)


def _packed_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Provide a fused observation and a separate off-camera spatial object."""
    detections = np.array([[78, 39, 122, 62, 0.95, 5]], dtype=np.float64)
    spatial = np.array([[40, 1, 10, 0, 4, 2, 2, 0.9, 7], [0, 1, 10, 0, 4, 2, 2, 0.9, 5]], dtype=np.float64)
    projection = np.array([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=np.float64)
    return detections, spatial, projection


def _canonical_inputs(
    detections: np.ndarray, spatial: np.ndarray, projection: np.ndarray
) -> tuple[Detections, Detections3D, CameraModel]:
    """Construct the reference inputs independently of the packed converters."""
    return (
        Detections(
            Boxes(torch.tensor(detections[:, :4], dtype=torch.float32)),
            torch.tensor(detections[:, 4], dtype=torch.float32),
            torch.tensor(detections[:, 5], dtype=torch.int64),
        ),
        Detections3D(
            Boxes3D(torch.tensor(spatial[:, :7], dtype=torch.float32)),
            torch.tensor(spatial[:, 7], dtype=torch.float32),
            torch.tensor(spatial[:, 8], dtype=torch.int64),
        ),
        CameraModel(torch.tensor(projection, dtype=torch.float32), (100, 200)),
    )


def _assert_matches(packed: tuple[np.ndarray, np.ndarray], canonical: MultimodalTracks) -> None:
    """Check every public output column, including independent sensor indices."""
    assert isinstance(packed, tuple) and len(packed) == 2
    for rows, tracks, geometry_columns in zip(packed, (canonical.image_tracks, canonical.spatial_tracks), (4, 7)):
        assert type(rows) is np.ndarray
        assert rows.dtype == np.float64 and rows.flags.c_contiguous
        assert rows.shape == (len(tracks), geometry_columns + 4)
        expected = np.column_stack(
            (
                tracks.geometry.values.numpy(),
                tracks.track_ids.numpy(),
                tracks.scores.numpy(),
                tracks.class_ids.numpy(),
                tracks.detection_indices.numpy(),
            )
        )
        np.testing.assert_allclose(rows, expected)


@pytest.mark.parametrize("per_class", [False, True])
def test_numpy_sequence_matches_canonical_tracking_and_sensor_dropout(per_class: bool) -> None:
    """The wrapper preserves identity, moving boxes, and 2D-only updates."""
    packed_tracker, canonical_tracker = EagerMot(per_class=per_class), EagerMot(per_class=per_class)
    detections, spatial, projection = _packed_inputs()
    first_image_id = None
    for frame_index in range(3):
        if frame_index == 1:
            detections[:, [0, 2]] += 2
            spatial[:, 0] += 0.2
        if frame_index == 2:
            spatial = spatial[:0]
        image_input, spatial_input, camera = _canonical_inputs(detections, spatial, projection)
        canonical = canonical_tracker.update(image_input, detections_3d=spatial_input, camera=camera)
        packed = packed_tracker.update(detections, detections_3d=spatial, camera=projection, image_size=(100, 200))
        _assert_matches(packed, canonical)
        if frame_index == 0:
            assert packed[0].shape == (1, 8) and packed[1].shape == (2, 11)
            first_image_id = packed[0][0, 4]
            fused = packed[1][packed[1][:, 7] == first_image_id]
            assert fused[0, 10] == 1
        else:
            assert packed[0][0, 4] == first_image_id
        if frame_index == 2:
            assert packed[1].shape == (1, 11)
            assert packed[1][0, 10] == -1


@pytest.mark.parametrize("sensor", ["image", "spatial", "both"])
def test_numpy_accepts_independent_empty_sensor_batches(sensor: str) -> None:
    """Empty input and output arrays retain each sensor's documented width."""
    detections, spatial, projection = _packed_inputs()
    if sensor in ("image", "both"):
        detections = detections[:0]
    if sensor in ("spatial", "both"):
        spatial = spatial[:0]
    canonical_inputs = _canonical_inputs(detections, spatial, projection)
    canonical = EagerMot().update(canonical_inputs[0], detections_3d=canonical_inputs[1], camera=canonical_inputs[2])
    packed = EagerMot().update(detections, detections_3d=spatial, camera=projection, image_size=(100, 200))
    _assert_matches(packed, canonical)
    if sensor in ("spatial", "both"):
        assert packed[0].shape == (0, 8) and packed[1].shape == (0, 11)


@pytest.mark.parametrize("frame_kind", ["numpy", "canonical"])
def test_numpy_camera_infers_image_size_from_frame(frame_kind: str) -> None:
    """A supplied image replaces the otherwise required image_size argument."""
    detections, spatial, projection = _packed_inputs()
    frame = (
        np.zeros((100, 200, 3), dtype=np.uint8)
        if frame_kind == "numpy"
        else Frame(torch.zeros((3, 100, 200), dtype=torch.uint8), "camera:0")
    )
    actual = EagerMot().update(detections, frame, detections_3d=spatial, camera=projection)
    expected = EagerMot().update(detections, detections_3d=spatial, camera=projection, image_size=(100, 200))
    for actual_rows, expected_rows in zip(actual, expected):
        np.testing.assert_array_equal(actual_rows, expected_rows)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_readonly_strided_numpy_inputs_are_not_modified(dtype: np.dtype) -> None:
    """External sensor buffers need not be writable or contiguous."""
    arrays = []
    for original in _packed_inputs():
        storage = np.zeros((original.shape[0], original.shape[1] * 2), dtype=dtype)
        view = storage[:, ::2]
        view[:] = original
        view.flags.writeable = False
        arrays.append(view)
    detections, spatial, projection = arrays
    before = [values.copy() for values in arrays]
    image_input, spatial_input, camera = _canonical_inputs(*arrays)
    actual = EagerMot().update(detections, detections_3d=spatial, camera=projection, image_size=(100, 200))
    canonical = EagerMot().update(image_input, detections_3d=spatial_input, camera=camera)
    _assert_matches(actual, canonical)
    for values, saved in zip(arrays, before):
        np.testing.assert_array_equal(values, saved)


def test_numpy_camera_pose_compensates_ego_motion() -> None:
    """A changing ndarray pose preserves spatial identity after camera movement."""
    detections, spatial, projection = _packed_inputs()
    detections, spatial = detections[:0], spatial[1:].copy()
    tracker = EagerMot(config=EagerMotConfig(distance_threshold=1.0))
    pose = np.eye(4)
    _, first = tracker.update(
        detections, detections_3d=spatial, camera=projection, image_size=(100, 200), camera_to_world=pose
    )
    pose[0, 3] = 5
    spatial[:, 0] -= 5
    _, second = tracker.update(
        detections, detections_3d=spatial, camera=projection, image_size=(100, 200), camera_to_world=pose
    )
    assert second[0, 7] == first[0, 7]
    assert second[0, 0] == pytest.approx(-5)


def test_mixed_inputs_keep_canonical_results_and_resolve_sample_ids() -> None:
    """The first argument controls return type; explicit sensor IDs still agree."""
    detections, spatial, projection = _packed_inputs()
    canonical_detections, canonical_spatial, camera = _canonical_inputs(detections, spatial, projection)
    canonical_detections = replace(canonical_detections, sample_id="sequence:7")
    result = EagerMot().update(
        canonical_detections, detections_3d=spatial, camera=projection, image_size=(100, 200)
    )
    assert isinstance(result, MultimodalTracks)
    assert result.sample_id == "sequence:7"
    canonical_spatial = replace(canonical_spatial, sample_id="sequence:7")
    frame = Frame(torch.zeros((3, 100, 200), dtype=torch.uint8), "sequence:7")
    packed = EagerMot().update(detections, frame, detections_3d=canonical_spatial, camera=camera)
    _assert_matches(packed, result)
    tracker = EagerMot()
    with pytest.raises(ValueError, match="same sample"):
        tracker.update(detections, replace(frame, sample_id="other"), detections_3d=canonical_spatial, camera=camera)
    assert tracker.frame_count == tracker._sample_index == 0


@pytest.mark.parametrize("class_id", [2**24 + 1, 2**40 + 1, 2**53])
def test_numpy_outputs_preserve_exact_large_class_ids(class_id: int) -> None:
    """Packing must never route class IDs through float32 model geometry."""
    detections, spatial, projection = _packed_inputs()
    spatial = spatial[1:].copy()
    detections[:, -1] = spatial[:, -1] = class_id
    image_tracks, spatial_tracks = EagerMot().update(
        detections, detections_3d=spatial, camera=projection, image_size=(100, 200)
    )
    assert image_tracks[0, 6] == class_id
    assert spatial_tracks[0, 9] == class_id


def test_mixed_canonical_spatial_class_outside_float64_range_is_rejected_before_tracking() -> None:
    """A canonical side input cannot silently lose integer precision on packing."""
    detections, spatial, projection = _packed_inputs()
    _, canonical_spatial, camera = _canonical_inputs(detections, spatial, projection)
    canonical_spatial = replace(canonical_spatial, class_ids=torch.tensor([2**53 + 1, 5]))
    tracker = EagerMot()
    with pytest.raises(ValueError, match="exact float64 integer range"):
        tracker.update(detections, detections_3d=canonical_spatial, camera=camera)
    assert tracker.frame_count == tracker._sample_index == 0


@pytest.mark.parametrize("sensor", ["image", "spatial"])
@pytest.mark.parametrize("invalid", ["width", "dtype", "nonfinite", "dimensions", "score", "class", "precision"])
def test_invalid_numpy_detections_fail_before_advancing_state(sensor: str, invalid: str) -> None:
    """Bad observations cannot consume a frame or change the tracking sequence."""
    detections, spatial, projection = _packed_inputs()
    tracker = EagerMot()
    tracker.update(detections, detections_3d=spatial, camera=projection, image_size=(100, 200))
    rows = detections if sensor == "image" else spatial
    if invalid == "width":
        rows = rows[:, :-1]
    elif invalid == "dtype":
        rows = rows.astype(complex)
    elif invalid == "nonfinite":
        rows[0, 0] = np.nan
    elif invalid == "dimensions":
        rows[0, 2 if sensor == "image" else 4] = 0
    elif invalid == "score":
        rows[0, -2] = 1.5
    elif invalid == "class":
        rows[0, -1] = 1.5
    elif invalid == "precision":
        rows[0, -1] = 2**53 + 2
    if sensor == "image":
        detections = rows
    else:
        spatial = rows
    with pytest.raises((TypeError, ValueError)):
        tracker.update(detections, detections_3d=spatial, camera=projection, image_size=(100, 200))
    assert tracker.frame_count == tracker._sample_index == 1


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_size", "zero_size", "float_size", "bool_size", "projection_shape", "projection_dtype",
        "projection_nan", "projection_overflow", "pose_shape", "pose_dtype", "pose_nan", "pose_overflow",
        "pose_last_row", "frame_size",
    ],
)
def test_invalid_numpy_calibration_fails_before_advancing_state(invalid: str) -> None:
    """Calibration and image dimensions are validated before the kernel runs."""
    detections, spatial, projection = _packed_inputs()
    kwargs = {"detections_3d": spatial, "camera": projection, "image_size": (100, 200)}
    if invalid == "missing_size":
        del kwargs["image_size"]
    elif invalid == "zero_size":
        kwargs["image_size"] = (0, 200)
    elif invalid == "float_size":
        kwargs["image_size"] = (100.0, 200)
    elif invalid == "bool_size":
        kwargs["image_size"] = (True, 200)
    elif invalid == "projection_shape":
        kwargs["camera"] = np.eye(3)
    elif invalid == "projection_dtype":
        kwargs["camera"] = projection.astype(complex)
    elif invalid == "projection_nan":
        projection[0, 0] = np.nan
    elif invalid == "projection_overflow":
        projection[0, 0] = np.finfo(np.float64).max
    elif invalid == "pose_shape":
        kwargs["camera_to_world"] = np.eye(3)
    elif invalid == "pose_dtype":
        kwargs["camera_to_world"] = np.eye(4, dtype=object)
    elif invalid == "pose_nan":
        pose = np.eye(4)
        pose[0, 3] = np.nan
        kwargs["camera_to_world"] = pose
    elif invalid == "pose_overflow":
        pose = np.eye(4)
        pose[0, 3] = np.finfo(np.float64).max
        kwargs["camera_to_world"] = pose
    elif invalid == "pose_last_row":
        kwargs["camera_to_world"] = np.ones((4, 4))
    elif invalid == "frame_size":
        kwargs["frame"] = np.zeros((99, 200, 3), dtype=np.uint8)
    tracker = EagerMot()
    with pytest.raises((TypeError, ValueError)):
        tracker.update(detections, **kwargs)
    assert tracker.frame_count == tracker._sample_index == 0


@pytest.mark.parametrize("extra", [{"image_size": (100, 200)}, {"camera_to_world": np.eye(4)}])
def test_camera_model_rejects_duplicate_numpy_calibration_options(extra: dict) -> None:
    """Canonical camera settings must not be silently overridden by side inputs."""
    detections, spatial, projection = _packed_inputs()
    _, _, camera = _canonical_inputs(detections, spatial, projection)
    tracker = EagerMot()
    with pytest.raises(ValueError):
        tracker.update(detections, detections_3d=spatial, camera=camera, **extra)
    assert tracker.frame_count == tracker._sample_index == 0
