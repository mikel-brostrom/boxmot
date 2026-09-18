"""Preserve HybridSORT's accumulated motion evidence during camera correction."""

import numpy as np
import pytest

from boxmot.trackers.common.geometry.obb import transform_aabb
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.hybridsort.track import KalmanBoxTracker


def test_projective_camera_correction_retains_perspective() -> None:
    """The affine covariance policy must not truncate a true homography."""
    detection = np.array([100.0, 200.0, 130.0, 270.0, 0.9])
    track = KalmanBoxTracker(detection, np.array([1.0, 0.0]), id_allocator=TrackIdAllocator())
    transform = np.array([[1.0, 0.05, 3.0], [-0.02, 1.0, -2.0], [0.0005, -0.0003, 1.0]])
    expected = transform_aabb(detection[:4], transform)

    track.camera_update(transform)

    np.testing.assert_allclose(track.motion_model.to_box(track.kf.x)[0, :4], expected, atol=1e-6)
    assert track.kf.x[3, 0] == pytest.approx(detection[4])
    np.testing.assert_allclose(track.kf.P, track.kf.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(track.kf.P).min() >= -1e-8


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(np.eye(3), id="identity"),
        pytest.param(np.array([[1.0, 0.0, 9.0], [0.0, 1.0, -4.0]]), id="translation"),
        pytest.param(np.array([[0.8, -0.6, 9.0], [0.6, 0.8, -4.0]]), id="rotation"),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_camera_correction_preserves_accumulated_corner_motion(transform: np.ndarray, batched: bool) -> None:
    """A coordinate change must not discard the strength of multi-frame cues."""
    tracks = []
    for index in range(2 if batched else 1):
        track = KalmanBoxTracker(
            np.array([10.0 + index, 20.0, 30.0 + index, 80.0, 0.9]),
            np.array([1.0, 0.0]),
            delta_t=4,
            id_allocator=TrackIdAllocator(),
        )
        # HybridSORT sums normalized displacement vectors over delta_t frames.
        # These are intentionally not unit vectors: their norms encode the
        # amount and consistency of the available motion evidence.
        for frame in range(1, 6):
            track.predict()
            track.update(
                np.array([10.0 + index + frame, 20.0, 30.0 + index + frame, 80.0, 0.9]),
                np.array([1.0, 0.0]),
            )
        tracks.append(track)

    attributes = ("velocity_lt", "velocity_rt", "velocity_lb", "velocity_rb")
    original = [[getattr(track, name).copy() for name in attributes] for track in tracks]
    assert all(np.linalg.norm(direction) > 3.9 for directions in original for direction in directions)

    if batched:
        KalmanBoxTracker.multi_camera_update(tracks, transform)
    else:
        tracks[0].camera_update(transform)

    for track, directions in zip(tracks, original, strict=True):
        for name, direction in zip(attributes, directions, strict=True):
            expected = (transform[:2, :2] @ direction[::-1])[::-1]
            np.testing.assert_allclose(getattr(track, name), expected, rtol=1e-12, atol=1e-12)


def test_tiny_camera_rotation_keeps_tall_pedestrian_prediction_stable() -> None:
    """Small camera motion must not make area residuals distort fitted aspect."""
    detection = np.array([890.0, 423.0, 1028.0, 943.0, 0.9])
    embedding = np.array([1.0, 0.0])
    track = KalmanBoxTracker(detection, embedding, id_allocator=TrackIdAllocator())
    transform = np.array([[0.99992, 0.00017, 0.09], [-0.00017, 0.99992, 0.14]])

    for frame in range(20):
        track.camera_update(transform)
        track.predict()
        measurement = detection.copy()
        measurement[:4] += frame
        track.update(measurement, embedding)

        # Each camera displacement is below a pixel, and detections retain
        # their aspect. A >60px correction reveals covariance coupling between
        # the area state (tens of thousands) and the dimensionless aspect.
        fitted_box = track.motion_model.to_box(track.kf.x)[0]
        np.testing.assert_allclose(fitted_box, measurement[:4], rtol=0.0, atol=1.0)


@pytest.mark.parametrize("homogeneous", [False, True])
def test_affine_camera_covariance_preserves_size_priors_and_cross_terms(homogeneous: bool) -> None:
    """Center correction must preserve correlated uncertainty and size priors."""
    track = KalmanBoxTracker(
        np.array([890.0, 423.0, 1028.0, 943.0, 0.9]),
        np.array([1.0, 0.0]),
        id_allocator=TrackIdAllocator(),
    )
    means = track.kf.x.T.copy()
    covariance = np.diag([4.0, 9.0, 20.0, 0.1, 10.0, 16.0, 25.0, 100.0, 0.01])[None]
    covariance[:, 0, 5] = covariance[:, 5, 0] = 0.5
    original_covariance = covariance.copy()
    original_means = means.copy()
    matrix = np.array([[0.8, -0.6, 12.0], [0.6, 0.8, -7.0], [0.0, 0.0, 1.0]])
    transform = matrix if homogeneous else matrix[:2]

    _, corrected = KalmanBoxTracker._map_camera_states_and_covariances(
        means, covariance, transform, track.motion_model
    )

    np.testing.assert_allclose(corrected[0, :2, :2], [[5.8, -2.4], [-2.4, 7.2]])
    np.testing.assert_allclose(corrected[0, 5:7, 5:7], [[19.24, -4.32], [-4.32, 21.76]])
    np.testing.assert_allclose(corrected[0, :2, 5:7], [[0.32, 0.24], [0.24, 0.18]])
    np.testing.assert_array_equal(corrected[:, [2, 3, 4, 7, 8]], covariance[:, [2, 3, 4, 7, 8]])
    np.testing.assert_allclose(corrected, corrected.swapaxes(1, 2), atol=1e-15)
    assert np.linalg.eigvalsh(corrected).min() > 0.0
    np.testing.assert_array_equal(covariance, original_covariance)
    np.testing.assert_array_equal(means, original_means)
