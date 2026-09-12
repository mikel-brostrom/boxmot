"""Batch motion must preserve independent filters and complete tracker trajectories."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from boxmot import EagerMot
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, MaskBatch
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.eagermot.geometry import project_box3d, transform_boxes3d
from boxmot.trackers.eagermot.motion import Kalman3D


@pytest.mark.parametrize("angular", [False, True, None])
def test_batch_motion_matches_independent_filters_over_long_trajectories(angular: bool | None) -> None:
    """Different priors, Q/R/F, missing observations, and yaw wraps remain independent."""
    rng = np.random.default_rng(321)
    truth = rng.normal(size=(12, 7))
    truth[:, 4:] = rng.uniform(1, 5, size=(12, 3))
    truth[:, 3] += np.arange(12) * 4 * np.pi
    models = [
        Kalman3D(
            box,
            is_angular=bool(index % 2) if angular is None else angular,
            noise_config=KalmanNoiseConfig(
                process_position_scale=1 + index,
                process_velocity_scale=1 + index / 2,
                measurement_noise_scale=0.5 + index,
                initial_position_scale=2 + index,
                initial_velocity_scale=1 + index / 3,
            ),
        )
        for index, box in enumerate(truth)
    ]
    for index, model in enumerate(models):
        dimensions = len(model.state)
        prior = rng.normal(size=(dimensions, dimensions))
        model.covariance = prior @ prior.T + np.eye(dimensions)
        model._transition[0, 7] = 0.5 + index / 12
    scalar = deepcopy(models)
    for frame in range(160):
        truth[:, :3] += rng.normal(scale=0.1, size=(len(models), 3))
        truth[:, 3] += 0.14
        measurements = truth.copy()
        measurements[:, 3] = (measurements[:, 3] + np.pi) % (2 * np.pi) - np.pi
        expected = np.stack([model.predict() for model in scalar])
        prediction = Kalman3D.multi_predict(models)
        np.testing.assert_allclose(prediction, expected, rtol=1e-12, atol=1e-11)
        observed = np.flatnonzero(rng.random(len(models)) > 0.35) if frame % 11 else np.empty(0, dtype=int)
        original = measurements.copy()
        correction = Kalman3D.multi_update([models[index] for index in observed], measurements[observed])
        expected = np.asarray([scalar[index].update(measurements[index]) for index in observed]).reshape(-1, 7)
        np.testing.assert_allclose(correction, expected, rtol=1e-12, atol=1e-11)
        np.testing.assert_array_equal(measurements, original)
        prediction[:] = 0
        correction[:] = 0
        for actual, reference in zip(models, scalar):
            np.testing.assert_allclose(actual.state, reference.state, rtol=1e-12, atol=1e-11)
            np.testing.assert_allclose(actual.covariance, reference.covariance, rtol=1e-12, atol=1e-11)
            if frame % 20 == 0:
                np.testing.assert_allclose(actual.covariance, actual.covariance.T, atol=1e-12)
                assert np.linalg.eigvalsh(actual.covariance).min() > 0


def test_multiple_filters_do_not_dispatch_scalar_numerical_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real batches use matrix operations rather than hiding per-track KF calls."""
    box = np.array([0, 1, 20, 0, 4, 2, 2], dtype=float)
    filters = [Kalman3D(box, is_angular=bool(index % 2)) for index in range(8)]

    def fail(*args: object) -> None:
        raise AssertionError("A batch dispatched a scalar Kalman operation.")

    monkeypatch.setattr(Kalman3D, "predict", fail)
    monkeypatch.setattr(Kalman3D, "update", fail)
    Kalman3D.multi_predict(filters)
    Kalman3D.multi_update(filters, np.tile(box, (len(filters), 1)))


@pytest.mark.parametrize("bad_box", [np.zeros((2, 7)), np.ones((1, 7)), np.full((2, 7), np.nan)])
def test_batch_update_validates_before_mutating_any_filter(bad_box: np.ndarray) -> None:
    filters = [Kalman3D(np.array([0, 1, 20, 0, 4, 2, 2])) for _ in range(2)]
    original = deepcopy(filters)
    with pytest.raises(ValueError, match="finite.*positive"):
        Kalman3D.multi_update(filters, bad_box)
    for actual, reference in zip(filters, original):
        np.testing.assert_array_equal(actual.state, reference.state)
        np.testing.assert_array_equal(actual.covariance, reference.covariance)


def test_empty_motion_batches_have_box_shape() -> None:
    assert Kalman3D.multi_predict([]).shape == (0, 7)
    assert Kalman3D.multi_update([], np.empty((0, 7))).shape == (0, 7)


def _sensor_frame(frame: int, world: bool) -> tuple[Detections, Detections3D, CameraModel]:
    """Generate masked, shuffled observations with births, expiry, and sensor gaps."""
    projection = torch.tensor([[120, 0, 160, 0], [0, 120, 96, 0], [0, 0, 1, 0]], dtype=torch.float32)
    pose = np.eye(4)
    pose[0, 3] = frame * 0.03
    camera = CameraModel(projection, (192, 320), torch.from_numpy(pose).float() if world else None)
    identities = np.arange(7)
    present = ~((identities == 2) & (15 <= frame) & (frame <= 21))
    present &= (identities != 6) | (frame >= 17)
    identities = identities[present]
    boxes = np.column_stack(
        (
            identities * 4 - 10 + frame * 0.04,
            np.ones(len(identities)),
            20 + identities * 3,
            np.full(len(identities), frame * 0.02),
            np.full(len(identities), 3.5),
            np.full(len(identities), 1.8),
            np.full(len(identities), 1.7),
        )
    )
    if world:
        boxes = transform_boxes3d(boxes, pose, inverse=True)
    boxes[:, 3] += np.where((identities + frame) % 3 == 0, np.pi, 0)
    image_rows = np.flatnonzero((identities + frame) % 8 != 3)[::-1]
    spatial_rows = np.flatnonzero((identities + frame) % 6 != 2)
    if frame in (10, 11):
        spatial_rows = spatial_rows[:0]
    if frame in (11, 12):
        image_rows = image_rows[:0]
    image_boxes = np.asarray(
        [project_box3d(boxes[index], projection.numpy(), camera.image_size) for index in image_rows]
    )
    image_boxes = image_boxes.reshape(-1, 4)
    masks = torch.zeros((len(image_rows), *camera.image_size), dtype=torch.bool)
    for index, (x1, y1, x2, y2) in enumerate(image_boxes.astype(int)):
        masks[index, y1:y2, x1:x2] = True
    sample_id = f"batch/{frame}"
    image = Detections(
        Boxes(torch.from_numpy(image_boxes).float()),
        torch.full((len(image_rows),), 0.9),
        torch.from_numpy(identities[image_rows] % 2),
        sample_id=sample_id,
        masks=MaskBatch(masks),
    )
    spatial = Detections3D(
        Boxes3D(torch.from_numpy(boxes[spatial_rows]).float()),
        torch.full((len(spatial_rows),), 0.95),
        torch.from_numpy(identities[spatial_rows] % 2),
        sample_id=sample_id,
    )
    return image, spatial, camera


@pytest.mark.parametrize("angular", [False, True])
@pytest.mark.parametrize("world", [False, True])
def test_tracker_batch_motion_preserves_associations_lifecycle_and_masks(
    angular: bool, world: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare complete multi-object runs with the previous scalar KF execution."""
    batched = EagerMot(is_angular=angular, max_age=4, per_class=True)
    scalar = EagerMot(is_angular=angular, max_age=4, per_class=True)

    def scalar_predict(filters: list[Kalman3D]) -> np.ndarray:
        return np.asarray([model.predict() for model in filters], dtype=float).reshape(-1, 7)

    def scalar_update(filters: list[Kalman3D], boxes: np.ndarray) -> np.ndarray:
        return np.asarray([model.update(box) for model, box in zip(filters, boxes)], dtype=float).reshape(-1, 7)

    for frame in range(45):
        image, spatial, camera = _sensor_frame(frame, world)
        actual = batched.update(image, detections_3d=spatial, camera=camera)
        with monkeypatch.context() as scalar_context:
            scalar_context.setattr(Kalman3D, "multi_predict", scalar_predict)
            scalar_context.setattr(Kalman3D, "multi_update", scalar_update)
            expected = scalar.update(image, detections_3d=spatial, camera=camera)
        for modality in ("image_tracks", "spatial_tracks"):
            output, reference = getattr(actual, modality), getattr(expected, modality)
            torch.testing.assert_close(output.geometry.values, reference.geometry.values, rtol=1e-6, atol=1e-6)
            for field in ("track_ids", "class_ids", "detection_indices", "scores"):
                torch.testing.assert_close(getattr(output, field), getattr(reference, field), rtol=0, atol=0)
        torch.testing.assert_close(actual.image_tracks.masks.values, expected.image_tracks.masks.values)
        assert len(batched._tracks) == len(scalar._tracks)
        for track, reference in zip(batched._tracks, scalar._tracks):
            for field in ("id", "cls", "hits", "age", "time_since_update", "time_since_2d_update"):
                assert getattr(track, field) == getattr(reference, field)
            np.testing.assert_allclose(track.motion.state, reference.motion.state, rtol=1e-12, atol=1e-11)
            np.testing.assert_allclose(track.motion.covariance, reference.motion.covariance, rtol=1e-12, atol=1e-11)
