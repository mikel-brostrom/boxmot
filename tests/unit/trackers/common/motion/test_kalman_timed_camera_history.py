"""Timed observation replay uses the compensated camera coordinate frame."""

from copy import deepcopy

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters.xyscr import KalmanFilterXYSCR
from boxmot.trackers.common.motion.kalman_filters.xysr import KalmanFilterXYSR


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

    def transform_measurement(observation: np.ndarray) -> np.ndarray:
        return (
            jacobian[: filter_.dim_z, : filter_.dim_z] @ np.asarray(observation).reshape(-1, 1)
            + offset[: filter_.dim_z]
        )

    filter_.x, filter_.P = transform_state(filter_.x, filter_.P)
    filter_.transform_timed_history(transform_state, transform_measurement)
    for kf in (filter_, reference):
        kf.predict(dt=0.7)
        kf.update(None)
        kf.predict(dt=0.1)
    recovery = measurement.copy()
    recovery[:2] += [5.0, -2.0]
    if mode == "xysr_obb":
        recovery[4] += 0.1
    reference.update(recovery)
    filter_.update(transform_measurement(recovery))
    expected_state, expected_covariance = transform_state(reference.x, reference.P)
    np.testing.assert_allclose(filter_.x, expected_state, atol=1e-10)
    np.testing.assert_allclose(filter_.P, expected_covariance, atol=1e-10)
