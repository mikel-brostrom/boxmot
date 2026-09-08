"""Adaptive XYHR covariance estimates retain consistent elapsed-time units."""

import numpy as np
import pytest

from boxmot.motion.kalman_filters.xyhr import AdaptiveNoiseXYHR, KalmanFilterXYHR


@pytest.mark.parametrize("dim_z", [4, 5])
def test_timed_adaptation_normalizes_samples_before_averaging(dim_z: int) -> None:
    """Mixed intervals recover the same density from balanced innovations."""
    dim_x = 2 * dim_z
    policy = AdaptiveNoiseXYHR(dim_x, dim_z, window=100, warmup=1)
    filter_ = KalmanFilterXYHR(ndim=dim_x, dim_z=dim_z)
    target = np.diag(np.linspace(0.3, 0.9, dim_x))
    gain = np.vstack((0.2 * np.eye(dim_z), 0.1 * np.eye(dim_z)))
    baseline = policy.get_q().copy()
    for dt in (0.05, 0.8, 0.2, 1.5):
        transition, noise = filter_._elapsed_motion(target, dt)
        for coordinate in range(dim_z):
            for sign in (-1, 1):
                residual = np.zeros(dim_z)
                residual[coordinate] = sign * np.sqrt(dim_z)
                policy.observe_innovation(
                    residual,
                    gain,
                    np.eye(dim_x),
                    transition,
                    dt=dt,
                    process_noise=noise,
                    innovation_covariance=np.eye(dim_z),
                )
    np.testing.assert_allclose(policy.get_q(dt=0.1), 0.7 * target + 0.3 * baseline)
    np.testing.assert_array_equal(policy.get_q(), baseline)
    assert policy._innovations == []


@pytest.mark.parametrize("dim_z", [4, 5])
def test_adaptive_xyhr_timed_single_and_batch_predictions_agree(dim_z: int) -> None:
    """Both prediction forms use learned rates after mixed-interval warmup."""
    measurement = np.array([20.0, 30.0, 10.0, 2.0, 0.1])[:dim_z]
    filter_ = KalmanFilterXYHR(z=measurement, dim_z=dim_z, adaptive_kf=True)
    elapsed = 0.0
    for index in range(40):
        dt = (0.03, 0.2, 0.1)[index % 3]
        elapsed += dt
        filter_.predict(dt=dt)
        current = measurement.copy()
        current[0] += 3.0 * elapsed + 0.02 * np.sin(index)
        filter_.update(current)
    assert filter_.cov_update_policy._q_rate is not None
    assert np.isfinite(filter_.covariance).all()
    assert np.linalg.eigvalsh(filter_.covariance).min() >= -1e-10
    mean, covariance = filter_.x.copy(), filter_.covariance.copy()
    one_mean, one_covariance = filter_.predict(mean, covariance, dt=0.15)
    batch_mean, batch_covariance = filter_.multi_predict(mean[None], covariance[None], dt=0.15)
    np.testing.assert_allclose(batch_mean[0], one_mean)
    np.testing.assert_allclose(batch_covariance[0], one_covariance)
    np.testing.assert_array_equal(filter_.x, mean)
    np.testing.assert_array_equal(filter_.covariance, covariance)
