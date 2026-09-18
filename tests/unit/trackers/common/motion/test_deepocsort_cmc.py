"""Keep camera compensation consistent with DeepOcSort's fitted motion model."""

from copy import deepcopy

import numpy as np
import pytest

from boxmot.trackers.common.geometry.obb import transform_aabb
from boxmot.trackers.common.motion.cmc.state import transform_xysr_kalman_centers
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.deepocsort.track import DeepOBBKalmanBoxTracker
from boxmot.trackers.deepocsort.track import KalmanBoxTracker as DeepTrack


@pytest.mark.parametrize("column", [False, True])
def test_xysr_camera_correction_preserves_size_and_covariance_terms(column: bool) -> None:
    """Camera rotation must not couple area and aspect uncertainty."""
    means = np.array([[100.0, 200.0, 72000.0, 0.265, 2.0, -3.0, 25.0]])
    covariance = np.diag([4.0, 9.0, 20.0, 10.0, 16.0, 25.0, 100.0])[None]
    covariance[:, 0, 4] = covariance[:, 4, 0] = 0.5
    original_covariance = covariance.copy()
    original_means = means.copy()
    matrix = np.array([[0.8, -0.6, 12.0], [0.6, 0.8, -7.0]])

    transformed, corrected = transform_xysr_kalman_centers(means[..., None] if column else means, covariance, matrix)

    assert transformed.shape == ((1, 7, 1) if column else (1, 7))
    result = transformed.reshape(1, 7)
    np.testing.assert_allclose(result[0, :2], [-28.0, 213.0])
    np.testing.assert_allclose(result[0, 4:6], [3.4, -1.2])
    np.testing.assert_array_equal(result[:, [2, 3, 6]], means[:, [2, 3, 6]])
    np.testing.assert_allclose(corrected[0, :2, :2], [[5.8, -2.4], [-2.4, 7.2]])
    np.testing.assert_allclose(corrected[0, 4:6, 4:6], [[19.24, -4.32], [-4.32, 21.76]])
    np.testing.assert_allclose(corrected[0, :2, 4:6], [[0.32, 0.24], [0.24, 0.18]])
    np.testing.assert_array_equal(corrected[:, [2, 3, 6]], covariance[:, [2, 3, 6]])
    np.testing.assert_array_equal(means, original_means)
    np.testing.assert_array_equal(covariance, original_covariance)


def test_xysr_camera_correction_preserves_positive_covariance() -> None:
    """Rotating only diagonal blocks makes correlated center/velocity indefinite."""
    means = np.array([[100.0, 200.0, 72000.0, 0.265, 2.0, -3.0, 25.0]])
    covariance = np.eye(7)[None]
    covariance[:, 0, 4] = covariance[:, 4, 0] = 0.9999
    transform = np.array([[0.99992, 0.00017, 0.09], [-0.00017, 0.99992, 0.14]])
    _, corrected = transform_xysr_kalman_centers(means, covariance, transform)
    np.testing.assert_allclose(corrected, corrected.swapaxes(1, 2), atol=1e-15)
    assert np.linalg.eigvalsh(corrected).min() > 0.0


def test_deep_camera_rotation_keeps_pedestrian_trajectory_and_recovery_stable() -> None:
    """Repeated tiny rotations must not collapse a tall pedestrian's aspect."""
    detection = np.array([890.0, 423.0, 1028.0, 943.0, 0.9, 1, 0])
    track = DeepTrack(detection, id_allocator=TrackIdAllocator())
    original_shape = track.kf.x[[2, 3]].copy()
    transform = np.array([[0.99992, 0.00017, 0.09], [-0.00017, 0.99992, 0.14]])
    for frame in range(20):
        track.camera_update(transform)
        track.predict()
        measurement = detection.copy()
        measurement[:4] += frame
        track.update(measurement)
        np.testing.assert_allclose(track.kf.x[[2, 3]], original_shape, rtol=1e-10)
        assert np.max(np.abs(track.kf.P)) < 25.0
        np.testing.assert_allclose(track.get_state()[0], measurement[:4], atol=1.0)

    timed_track = deepcopy(track)
    timed_track.predict(dt=0.5)
    timed_track.update(None)
    timed_track.predict(dt=0.5)
    timed_original = deepcopy(timed_track.kf)
    timed_track.camera_update(transform)

    track.predict()
    track.update(None)
    original = deepcopy(track.kf)
    track.camera_update(transform)
    for actual, previous in (
        ((track.kf.x, track.kf.P), (original.x, original.P)),
        (timed_track.kf._prediction_origin, timed_original._prediction_origin),
        ((track.kf.attr_saved["x"], track.kf.attr_saved["P"]), (original.attr_saved["x"], original.attr_saved["P"])),
    ):
        np.testing.assert_array_equal(actual[0][[2, 3, 6]], previous[0][[2, 3, 6]])
        np.testing.assert_array_equal(actual[1][[2, 3, 6]], previous[1][[2, 3, 6]])


def test_deep_projective_camera_correction_retains_perspective() -> None:
    """A projective warp must not be silently truncated to an affine matrix."""
    track = DeepTrack(np.array([100.0, 200.0, 130.0, 270.0, 0.9, 1, 0]), id_allocator=TrackIdAllocator())
    transform = np.array([[1.0, 0.05, 3.0], [-0.02, 1.0, -2.0], [0.0005, -0.0003, 1.0]])
    expected = transform_aabb(track.get_state()[0], transform)
    track.camera_update(transform)
    np.testing.assert_allclose(track.get_state()[0], expected, atol=1e-6)


def test_deep_oriented_camera_correction_still_rotates_and_scales_box() -> None:
    """The OBB adapter retains its full geometric camera transformation."""
    track = DeepOBBKalmanBoxTracker(
        np.array([100.0, 200.0, 30.0, 70.0, 0.2, 0.9, 1, 0]),
        emb=None,
        alpha=0.95,
        delta_t=3,
        max_obs=50,
        id_allocator=TrackIdAllocator(),
    )
    angle, scale = 0.07, 1.1
    linear = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transform = np.column_stack((linear, [12.0, -7.0]))
    track.camera_update(transform)
    expected_center = linear @ np.array([100.0, 200.0]) + [12.0, -7.0]
    np.testing.assert_allclose(track.get_state()[0], [*expected_center, 33.0, 77.0, 0.27], atol=1e-6)
