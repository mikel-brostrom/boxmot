"""Shared motion preparation retains scalar distances and track-local noise."""

from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.trackers.common.association import matching
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.models import MotionModelAdapter, MotionModelKind, create_motion_model


def _tracks(models: list[MotionModelAdapter], dtype=np.float64) -> list[SimpleNamespace]:
    """Build correlated distributions with independent measurement policies."""
    rng = np.random.default_rng(501)
    tracks = []
    for index, model in enumerate(models):
        box = np.array(
            [30.0 + index, 50.0, 20.0, 40.0, 0.23] if model.is_obb else [20.0 + index, 30.0, 40.0 + index, 70.0]
        )
        noise = KalmanNoiseConfig(
            measurement_noise_scale=1.0 + index,
            measurement_noise_diagonal=tuple(np.linspace(0.7, 1.3, model.dim_z)),
            measurement_confidence_floor=tuple(np.linspace(0.1, 0.3, model.dim_z)),
        )
        kalman = model.create_filter(noise_config=noise)
        measurement = model.to_measurement(np.r_[box, 0.9], column=False)
        mean, _ = model.initiate(kalman, measurement, confidence=0.9)
        factors = rng.normal(size=(model.dim_x, model.dim_x)).astype(dtype)
        covariance = factors @ factors.T + np.eye(model.dim_x, dtype=dtype)
        track = SimpleNamespace(motion_model=model, mean=mean.astype(dtype), covariance=covariance)
        if index % 3 == 0:
            track.kalman_filter = kalman
        else:
            track.kf = kalman
        if index % 3 == 2:
            kalman.x, kalman.P = track.mean[:, None], covariance
            del track.mean, track.covariance
        tracks.append(track)
    return tracks


def _boxes(is_obb: bool, dtype=np.float64) -> np.ndarray:
    """Include nearby detections, clear rejections and equivalent OBB forms."""
    if is_obb:
        return np.array(
            [
                [30.2, 50.1, 20.0, 40.0, 0.23],
                [31.2, 49.9, 40.0, 20.0, 0.23 + np.pi / 2],
                [400.0, 500.0, 20.0, 40.0, -np.pi + 0.01],
            ],
            dtype=dtype,
        )
    return np.array([[20.2, 30.1, 40.2, 70.1], [22.0, 29.9, 42.0, 69.9], [400, 500, 420, 540]], dtype=dtype)


def _state(track: SimpleNamespace) -> tuple[np.ndarray, np.ndarray]:
    """Read either externally held or filter-owned state without conversion."""
    if hasattr(track, "mean"):
        return track.mean, track.covariance
    return track.kf.x, track.kf.P


@pytest.mark.parametrize("kind", tuple(MotionModelKind))
@pytest.mark.parametrize("only_position", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_distances_match_scalar_with_correlated_states_and_independent_noise(kind, only_position, dtype):
    """Batching preserves exact distances for every selectable AABB filter."""
    model = create_motion_model(kind, is_obb=False)
    tracks = _tracks([model] * 6, dtype)
    boxes = _boxes(False, dtype)
    original_boxes = boxes.copy()
    original_states = [(mean.copy(), covariance.copy()) for mean, covariance in map(_state, tracks)]
    expected = np.asarray(
        [matching.motion_gating_distance(track, boxes, only_position=only_position) for track in tracks]
    )

    actual = matching.motion_gating_distances(tracks, boxes, only_position=only_position)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(boxes, original_boxes)
    for track, (mean, covariance) in zip(tracks, original_states):
        actual_mean, actual_covariance = _state(track)
        np.testing.assert_array_equal(actual_mean, mean)
        np.testing.assert_array_equal(actual_covariance, covariance)


@pytest.mark.parametrize("only_position", [False, True])
def test_batched_mixed_models_convert_each_shared_adapter_once(only_position, monkeypatch):
    """AABB measurement reuse neither combines model layouts nor caches state."""
    models = [create_motion_model(kind, is_obb=False) for kind in MotionModelKind]
    tracks = _tracks(models * 2)
    boxes = _boxes(False)
    expected = np.asarray(
        [matching.motion_gating_distance(track, boxes, only_position=only_position) for track in tracks]
    )
    calls = []
    convert = MotionModelAdapter.to_measurements

    def count_conversion(model, boxes):
        calls.append(id(model))
        return convert(model, boxes)

    monkeypatch.setattr(MotionModelAdapter, "to_measurements", count_conversion)
    actual = matching.motion_gating_distances(tracks, boxes, only_position=only_position)

    np.testing.assert_array_equal(actual, expected)
    assert calls == [id(model) for model in models]


@pytest.mark.parametrize("kind", tuple(MotionModelKind))
@pytest.mark.parametrize("only_position", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_obb_distances_retain_scalar_alignment(kind, only_position, dtype):
    """Oriented gates keep track-specific equivalent-rectangle selection."""
    model = create_motion_model(kind, is_obb=True)
    tracks = _tracks([model] * 3, dtype)
    boxes = _boxes(True, dtype)
    original_boxes = boxes.copy()
    original_states = [(mean.copy(), covariance.copy()) for mean, covariance in map(_state, tracks)]
    expected = np.asarray(
        [matching.motion_gating_distance(track, boxes, only_position=only_position) for track in tracks]
    )
    actual = matching.motion_gating_distances(tracks, boxes, only_position=only_position)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(boxes, original_boxes)
    for track, (mean, covariance) in zip(tracks, original_states):
        actual_mean, actual_covariance = _state(track)
        np.testing.assert_array_equal(actual_mean, mean)
        np.testing.assert_array_equal(actual_covariance, covariance)


@pytest.mark.parametrize("custom", ["adapter", "conversion"])
def test_custom_adapters_keep_per_track_scalar_preparation(custom, monkeypatch):
    """Custom measurement callbacks remain observable for every track."""
    model = create_motion_model("xyah", is_obb=False)
    tracks = _tracks([model] * 3)
    boxes = _boxes(False)
    convert = model.to_measurements
    calls = []

    def custom_conversion(boxes):
        calls.append(boxes)
        result = convert(boxes)
        result[:, 0] += 0.125
        return result

    if custom == "adapter":
        adapter = SimpleNamespace(
            is_obb=False,
            to_measurements=custom_conversion,
            project=model.project,
            geometry_measurement_indices=model.geometry_measurement_indices,
        )
        for track in tracks:
            track.motion_model = adapter
    else:
        monkeypatch.setitem(model.__dict__, "to_measurements", custom_conversion)
    expected = np.asarray([matching.motion_gating_distance(track, boxes) for track in tracks])
    calls.clear()

    actual = matching.motion_gating_distances(tracks, boxes)

    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == len(tracks)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_batched_projection_uses_each_tracks_filter(mixed_precision, monkeypatch):
    """Projection callbacks and track-local filter changes remain live."""
    model = create_motion_model("xyah", is_obb=False)
    tracks = _tracks([model] * 3)
    boxes = _boxes(False)
    project = MotionModelAdapter.project
    calls = []

    def custom_project(model, kalman, mean, covariance):
        calls.append(kalman)
        projected, uncertainty = project(model, kalman, mean, covariance)
        dtype = np.float32 if mixed_precision and mean[0] % 2 else np.float64
        return (projected + 0.125).astype(dtype), (uncertainty * 1.25).astype(dtype)

    monkeypatch.setattr(MotionModelAdapter, "project", custom_project)
    expected = np.asarray([matching.motion_gating_distance(track, boxes) for track in tracks])
    calls.clear()

    actual = matching.motion_gating_distances(tracks, boxes)

    np.testing.assert_array_equal(actual, expected)
    assert calls == [getattr(track, "kalman_filter", getattr(track, "kf", None)) for track in tracks]


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (1, 0), (3, 0)])
@pytest.mark.parametrize("is_obb", [False, True])
def test_batched_empty_gates_preserve_matrix_shape(shape, is_obb):
    model = create_motion_model("xyah", is_obb=is_obb)
    tracks = _tracks([model] * shape[0])

    actual = matching.motion_gating_distances(tracks, np.empty((shape[1], 5 if is_obb else 4)))

    assert actual.shape == shape
    assert actual.dtype == np.float64
