"""Numerical parity for batched prediction, projection, and correction."""

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters import batch
from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH


@pytest.mark.parametrize("filter_type", [KalmanFilterXYAH, KalmanFilterXYWH])
@pytest.mark.parametrize("ndim", [4, 5])
@pytest.mark.parametrize("count", [0, 1, 17])
@pytest.mark.parametrize("dt,time_unit", [(None, "frames"), (0.7, "frames"), (0.03, "seconds")])
def test_stateless_batch_matches_scalar_over_repeated_steps(filter_type, ndim, count, dt, time_unit):
    """Match both geometries with calibration, confidence, and elapsed time."""
    rng = np.random.default_rng(145)
    kalman = filter_type(
        ndim=ndim,
        noise_config=KalmanNoiseConfig(
            process_position_scale=2.3,
            process_velocity_scale=0.4,
            measurement_noise_scale=1.7,
            initial_position_scale=0.9,
            initial_velocity_scale=1.4,
            time_unit=time_unit,
        ),
    )
    measurements = rng.uniform(1.0, 100.0, (count, ndim))
    if ndim == 5:
        measurements[:, 4] = rng.uniform(-np.pi, np.pi, count)
    states = [kalman.initiate(value) for value in measurements]
    means = np.asarray([value[0] for value in states]).reshape(count, 2 * ndim)
    covariances = np.asarray([value[1] for value in states]).reshape(count, 2 * ndim, 2 * ndim)
    means[:, ndim:] = rng.normal(0.0, 0.5, (count, ndim))
    scalar_means, scalar_covariances = means.copy(), covariances.copy()
    for _ in range(6):
        original_means, original_covariances = means.copy(), covariances.copy()
        means, covariances = kalman.multi_predict(means, covariances, dt=dt)
        for index in range(count):
            scalar_means[index], scalar_covariances[index] = kalman.predict(
                scalar_means[index], scalar_covariances[index], dt=dt
            )
        np.testing.assert_allclose(means, scalar_means, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(covariances, scalar_covariances, rtol=1e-8, atol=1e-8)
        # Input arrays remain independent from returned predictions.
        assert not np.shares_memory(original_means, means)
        assert not np.shares_memory(original_covariances, covariances)
        measurements = scalar_means[:, :ndim] + rng.normal(0.0, 0.2, (count, ndim))
        if ndim == 5:
            measurements[:, 4] += np.pi * 2.0
            if filter_type is KalmanFilterXYWH:
                measurements[::2, 2:4] = measurements[::2, 2:4][:, ::-1]
                measurements[::2, 4] += np.pi / 2.0
        confidence = rng.uniform(0.0, 0.99, count)
        projected_mean, projected_covariance = kalman.multi_project(means, covariances, confidence)
        for index in range(count):
            expected_mean, expected_covariance = kalman.project(means[index], covariances[index], confidence[index])
            np.testing.assert_allclose(projected_mean[index], expected_mean)
            np.testing.assert_allclose(projected_covariance[index], expected_covariance, rtol=1e-10, atol=1e-10)
            scalar_means[index], scalar_covariances[index] = kalman.update(
                scalar_means[index], scalar_covariances[index], measurements[index], confidence[index]
            )
        original_measurements = measurements.copy()
        means, covariances = kalman.multi_update(means, covariances, measurements, confidence)
        np.testing.assert_array_equal(measurements, original_measurements)
        np.testing.assert_allclose(means, scalar_means, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(covariances, scalar_covariances, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("filter_type", [KalmanFilterXYAH, KalmanFilterXYWH])
def test_batch_constraints_and_confidence_broadcast(filter_type):
    kalman = filter_type(ndim=5)
    mean, covariance = kalman.initiate(np.array([10.0, 20.0, 3.0, 4.0, 3.1]))
    mean[2:4] = -1.0
    measurements = np.array([[10.0, 20.0, -2.0, -3.0, -3.1]])
    expected_mean, expected_covariance = kalman.update(mean, covariance, measurements[0], 0.3)
    actual_mean, actual_covariance = kalman.multi_update(mean[None], covariance[None], measurements, 0.3)
    np.testing.assert_allclose(actual_mean[0], expected_mean)
    np.testing.assert_allclose(actual_covariance[0], expected_covariance)


@pytest.mark.parametrize("filter_type", [KalmanFilterXYAH, KalmanFilterXYWH])
@pytest.mark.parametrize("count", [0, 1, 23])
@pytest.mark.parametrize("metric", ["gaussian", "maha"])
def test_vectorized_obb_gating_matches_scalar_alignment(filter_type, count, metric):
    kalman = filter_type(ndim=5)
    mean, covariance = kalman.initiate(np.array([10.0, 20.0, 3.0, 4.0, 3.1]))
    rng = np.random.default_rng(47)
    measurements = rng.uniform(1.0, 40.0, size=(count, 5))
    measurements[:, 4] = rng.uniform(-4.0 * np.pi, 4.0 * np.pi, size=count)
    projected_mean, projected_covariance = kalman.project(mean, covariance)
    aligned = measurements.copy()
    for index in range(count):
        if filter_type is KalmanFilterXYWH:
            aligned[index] = kalman._align_obb_measurement(aligned[index], projected_mean)
        else:
            aligned[index, 4] = kalman._align_angle_to_reference(aligned[index, 4], projected_mean[4])
    expected = kalman._gating_from_residuals(aligned - projected_mean, projected_covariance, metric)
    np.testing.assert_allclose(kalman.gating_distance(mean, covariance, measurements, metric=metric), expected)


def test_numerical_kernels_keep_individual_models_and_fading_memory():
    rng = np.random.default_rng(18)
    means = rng.normal(size=(7, 8))
    roots = rng.normal(size=(7, 8, 8))
    covariances = roots @ roots.transpose(0, 2, 1)
    transition = rng.normal(size=(7, 8, 8))
    noise = batch.diagonal(rng.uniform(0.1, 3.0, size=(7, 8)))
    alpha = rng.uniform(0.7, 1.5, size=7)
    predicted_mean, predicted_covariance = batch.predict(means, covariances, transition, noise, alpha_sq=alpha)
    for index in range(7):
        np.testing.assert_allclose(predicted_mean[index], transition[index] @ means[index])
        np.testing.assert_allclose(
            predicted_covariance[index],
            alpha[index] * transition[index] @ covariances[index] @ transition[index].T + noise[index],
        )


def test_joseph_correction_matches_scalar_adaptive_stabilization():
    rng = np.random.default_rng(38)
    filters = [BaseKalmanFilter(4) for _ in range(4)]
    for index, kalman in enumerate(filters):
        kalman.x[:, 0] = rng.normal(size=8)
        root = rng.normal(size=(8, 8))
        kalman.P = root @ root.T
        kalman.R *= index + 0.3
    # Force one member through the scalar adaptive jitter path.
    filters[-1].P = -1e-13 * np.eye(8)
    filters[-1].R = np.zeros((4, 4))
    measurement = rng.normal(size=(4, 4))
    result = batch.correct(
        np.asarray([kalman.x[:, 0] for kalman in filters]),
        np.asarray([kalman.P for kalman in filters]),
        measurement,
        np.asarray([kalman.H for kalman in filters]),
        np.asarray([kalman.R for kalman in filters]),
        joseph=True,
        stabilize=True,
        return_inverse=True,
    )
    for index, kalman in enumerate(filters):
        kalman.update_state(measurement[index])
        for actual, expected in zip(result, (kalman.x[:, 0], kalman.P, kalman.K, kalman.y[:, 0], kalman.S, kalman.SI)):
            np.testing.assert_allclose(actual[index], expected, rtol=1e-10, atol=1e-10)
