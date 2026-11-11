"""Timed observation replay uses the compensated camera coordinate frame."""

from copy import deepcopy

import numpy as np
import pytest

from boxmot.trackers.common.motion.cmc.batching import transform_filter_histories
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.kalman_filters.xyscr import KalmanFilterXYSCR
from boxmot.trackers.common.motion.kalman_filters.xysr import KalmanFilterXYSR
from boxmot.trackers.common.motion.models import create_motion_model
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.ocsort.track import KalmanBoxTracker


def _initialized_filter() -> KalmanFilterXYSR:
    """Create moving geometry with unequal coordinate noise and fading memory."""
    filter_ = KalmanFilterXYSR(
        noise_config=KalmanNoiseConfig(
            process_position_diagonal=(4.0, 0.25, 1.0, 1.0),
            process_velocity_diagonal=(0.1, 2.0, 0.5),
        )
    )
    measurement = np.array([20.0, 30.0, 200.0, 2.0])
    filter_.x, filter_.P = filter_.initiate(measurement)
    filter_.update(measurement)
    filter_.x[4:, 0] = [3.0, -2.0, 0.4]
    filter_._alpha_sq = 1.1
    return filter_


def _predict_gap(filter_: KalmanFilterXYSR) -> tuple[np.ndarray, np.ndarray]:
    """Retain independent snapshots of each ordinary prediction endpoint."""
    means, covariances = [filter_.x[:, 0].copy()], [filter_.P.copy()]
    for interval in (0.2, 0.7, 0.1):
        filter_.predict(dt=interval)
        filter_.update(None)
        means.append(filter_.x[:, 0].copy())
        covariances.append(filter_.P.copy())
    return np.asarray(means), np.asarray(covariances)


def _assert_prediction_endpoints(filter_, means: np.ndarray, covariances: np.ndarray) -> None:
    """Recorded predictions must reconstruct every independently warped endpoint."""
    assert len(filter_._prediction_steps) == len(means) - 1
    mean, covariance = filter_._prediction_origin
    mean = mean.reshape(-1).copy()
    np.testing.assert_allclose(mean, means[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(covariance, covariances[0], rtol=1e-9, atol=1e-9)
    for index, (_, transition, noise, control) in enumerate(filter_._prediction_steps, start=1):
        mean = transition @ mean
        if control is not None:
            mean += np.asarray(control).reshape(-1)
        covariance = filter_._alpha_sq * transition @ covariance @ transition.T + noise
        np.testing.assert_allclose(mean, means[index], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(covariance, covariances[index], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(filter_.x[:, 0], means[-1], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(filter_.P, covariances[-1], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("mode", ["xysr", "xysr_obb", "xyscr"])
def test_timed_replay_commutes_with_rigid_camera_transform(mode: str) -> None:
    """Translate all modes and additionally rotate the oriented-box model."""
    if mode == "xyscr":
        filter_ = KalmanFilterXYSCR()
        measurement = np.array([20.0, 30.0, 200.0, 0.8, 2.0])
    elif mode == "xysr_obb":
        filter_ = KalmanFilterXYSR(dim_x=9, dim_z=5)
        measurement = np.array([20.0, 30.0, 200.0, 2.0, 0.2])
    else:
        filter_ = KalmanFilterXYSR()
        measurement = np.array([20.0, 30.0, 200.0, 2.0])
    filter_.x, filter_.P = filter_.initiate(measurement)
    filter_.update(measurement)
    reference = deepcopy(filter_)
    for kf in (filter_, reference):
        kf.predict(dt=0.2)
        kf.update(None)

    rotation_angle = 0.4 if mode == "xysr_obb" else 0.0
    rotation = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    jacobian = np.eye(filter_.dim_x)
    jacobian[:2, :2] = rotation
    velocity = filter_.dim_z
    jacobian[velocity : velocity + 2, velocity : velocity + 2] = rotation
    offset = np.zeros((filter_.dim_x, 1))
    offset[:2, 0] = [100.0, -20.0]
    if mode == "xysr_obb":
        offset[4, 0] = rotation_angle

    def transform_state(state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return jacobian @ state + offset, jacobian @ covariance @ jacobian.T

    def transform_states(means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            means @ jacobian.T + offset[:, 0],
            jacobian @ covariances @ jacobian.T,
            np.broadcast_to(jacobian, covariances.shape),
        )

    def transform_measurements(observations: np.ndarray) -> np.ndarray:
        return observations @ jacobian[: filter_.dim_z, : filter_.dim_z].T + offset[: filter_.dim_z, 0]

    transform_filter_histories([filter_], transform_states, transform_measurements)
    for kf in (filter_, reference):
        kf.predict(dt=0.7)
        kf.update(None)
        kf.predict(dt=0.1)
    recovery = measurement.copy()
    recovery[:2] += [5.0, -2.0]
    if mode == "xysr_obb":
        recovery[4] += 0.1
    reference.update(recovery)
    filter_.update(transform_measurements(recovery[None])[0])
    expected_state, expected_covariance = transform_state(reference.x, reference.P)
    np.testing.assert_allclose(filter_.x, expected_state, atol=1e-10)
    np.testing.assert_allclose(filter_.P, expected_covariance, atol=1e-10)


@pytest.mark.parametrize(
    "warp",
    [
        np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0]]),
        np.array([[0.0, -1.0, 100.0], [1.0, 0.0, -20.0]]),
        np.array([[1.1, -0.03, 4.0], [0.04, 0.95, -3.0], [0.002, -0.001, 1.0]]),
    ],
    ids=["zoom", "rotation-anisotropic-noise", "projective"],
)
def test_camera_warp_preserves_prediction_chain_endpoints(warp: np.ndarray) -> None:
    """History reconstructs transformed means and first-order covariance at each step."""
    filter_ = _initialized_filter()
    means, covariances = _predict_gap(filter_)
    model = create_motion_model("xysr")
    expected_means, expected_covariances = model.transform_states(means, covariances, warp)
    intervals = [step[0] for step in filter_._prediction_steps]

    transform_filter_histories(
        [filter_],
        lambda x, p: model.transform_states(x, p, warp, return_jacobian=True),
        lambda z: model.transform_measurements(z, warp),
    )

    assert [step[0] for step in filter_._prediction_steps] == intervals
    _assert_prediction_endpoints(filter_, expected_means, expected_covariances)


def test_translation_preserves_predictions_with_nontrivial_transition_and_control() -> None:
    """Translation changes the affine prediction offset when F does not commute."""
    filter_ = _initialized_filter()
    means, covariances = [filter_.x[:, 0].copy()], [filter_.P.copy()]
    for interval in (0.2, 0.7):
        transition = filter_.F.copy()
        transition[0, 0], transition[1, 1] = 0.9, 1.05
        transition[0, 4] = interval
        control = np.array([1.0, -0.5, 0.1, 0.0, 0.3, -0.2, 0.0])[:, None]
        filter_.predict(F=transition, Q=np.eye(7) * 0.1, B=np.eye(7), u=control, dt=interval)
        filter_.update(None)
        means.append(filter_.x[:, 0].copy())
        covariances.append(filter_.P.copy())
    offset = np.array([100.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def transform_states(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return x + offset, p.copy(), np.broadcast_to(np.eye(7), p.shape)

    transform_filter_histories([filter_], transform_states, lambda z: z + offset[:4])
    _assert_prediction_endpoints(filter_, np.asarray(means) + offset, np.asarray(covariances))


@pytest.mark.parametrize("scale", [0.0, 1e-15], ids=["singular", "ill-conditioned"])
def test_unstable_camera_map_falls_back_to_direct_recovery(scale: float) -> None:
    """An uninvertible history map keeps live CMC and corrects recovery directly."""
    filter_ = _initialized_filter()
    means, covariances = _predict_gap(filter_)
    jacobian = np.eye(7)
    jacobian[0, 0] = scale
    offset = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def transform_states(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return x @ jacobian.T + offset, jacobian @ p @ jacobian.T, np.broadcast_to(jacobian, p.shape)

    transform_filter_histories([filter_], transform_states, lambda z: z @ jacobian[:4, :4].T + offset[:4])
    np.testing.assert_allclose(filter_.x[:, 0], jacobian @ means[-1] + offset)
    np.testing.assert_allclose(filter_.P, jacobian @ covariances[-1] @ jacobian.T)
    assert filter_._prediction_history_overflowed
    assert filter_._prediction_origin is None
    assert not filter_._prediction_steps

    reference = deepcopy(filter_)
    reference.observed = True
    recovery = filter_.x[:4, 0] + [1.0, -0.5, 2.0, 0.0]
    for kf in (filter_, reference):
        kf.predict(dt=0.1)
        kf.update(recovery.copy())
    np.testing.assert_allclose(filter_.x, reference.x)
    np.testing.assert_allclose(filter_.P, reference.P)
    assert not filter_._prediction_history_overflowed


def test_ocsort_camera_zoom_recovery_matches_scaled_coordinate_system() -> None:
    """The real track CMC path preserves recovery under a consistent pixel rescale."""
    box = np.array([10.0, 20.0, 30.0, 30.0, 0.9])
    track = KalmanBoxTracker(box, cls=1, det_ind=0, id_allocator=TrackIdAllocator())
    track.update(box, cls=1, det_ind=0)
    track.kf.x[4:, 0] = [3.0, -2.0, 0.4]
    reference = deepcopy(track)
    for item in (track, reference):
        item.predict(dt=0.2)
        item.update(None, cls=1, det_ind=0)

    scale, tx, ty = 1.5, 40.0, -12.0
    warp = np.array([[scale, 0.0, tx], [0.0, scale, ty]])
    jacobian = np.diag([scale, scale, scale**2, 1.0, scale, scale, scale**2])
    offset = np.array([tx, ty, 0.0, 0.0, 0.0, 0.0, 0.0])[:, None]
    track.camera_update(warp)
    # Express future process and measurement noise in the same scaled pixels.
    track.kf.Q = jacobian @ reference.kf.Q @ jacobian.T
    track.kf.R = jacobian[:4, :4] @ reference.kf.R @ jacobian[:4, :4].T
    for item in (track, reference):
        item.predict(dt=0.7)
        item.update(None, cls=1, det_ind=0)
        item.predict(dt=0.1)

    recovery = box + [5.0, -2.0, 5.0, -2.0, 0.0]
    reference.update(recovery, cls=1, det_ind=0)
    recovery[:4] = recovery[:4] * scale + [tx, ty, tx, ty]
    track.update(recovery, cls=1, det_ind=0)
    # AABB measurement conversion adds 1e-6 to height when computing aspect ratio.
    np.testing.assert_allclose(track.kf.x, jacobian @ reference.kf.x + offset, rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(track.kf.P, jacobian @ reference.kf.P @ jacobian.T, rtol=1e-9, atol=1e-9)
