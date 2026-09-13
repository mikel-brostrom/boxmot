from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH


@pytest.mark.parametrize(("ndim", "measurement"), [(4, [30, 20, 12, 6]), (5, [30, 20, 12, 6, 0.2])])
def test_xywh_initializes_aabb_and_obb_state_dimensions(ndim: int, measurement: list[float]) -> None:
    kalman_filter = KalmanFilterXYWH(ndim=ndim)

    mean, covariance = kalman_filter.initiate(np.asarray(measurement, dtype=float))

    assert mean.shape == (2 * ndim,)
    assert covariance.shape == (2 * ndim, 2 * ndim)
    np.testing.assert_allclose(mean[:ndim], measurement)
    np.testing.assert_allclose(mean[ndim:], 0.0)
    assert np.all(np.diag(covariance) > 0.0)


def test_xywh_rejects_unsupported_measurement_dimensions() -> None:
    with pytest.raises(ValueError, match="ndim must be 4 .* or 5"):
        KalmanFilterXYWH(ndim=6)


def test_xywh_obb_aligns_equivalent_width_height_representation() -> None:
    reference = np.array([30.0, 20.0, 12.0, 6.0, 0.2])
    equivalent = np.array([30.0, 20.0, 6.0, 12.0, 0.2 + (np.pi / 2.0)])

    aligned = KalmanFilterXYWH._align_obb_measurement(equivalent, reference)

    np.testing.assert_allclose(aligned, reference, atol=1e-12)


def test_xywh_obb_update_damps_angular_velocity() -> None:
    kalman_filter = KalmanFilterXYWH(ndim=5)
    mean, covariance = kalman_filter.initiate(np.array([30.0, 20.0, 12.0, 6.0, 0.0]))
    predicted_mean, predicted_covariance = kalman_filter.predict(mean, covariance)

    updated_mean, updated_covariance = kalman_filter.update(
        predicted_mean,
        predicted_covariance,
        np.array([30.0, 20.0, 12.0, 6.0, 0.4]),
    )

    assert 0.0 < updated_mean[4] < 0.4
    undamped_theta_velocity = (updated_mean[4] - predicted_mean[4]) / kalman_filter.dt
    assert 0.0 < updated_mean[-1] < undamped_theta_velocity
    assert np.all(np.isfinite(updated_covariance))


@pytest.mark.parametrize("ndim", [4, 5])
def test_xywh_multi_predict_matches_individual_prediction(ndim: int) -> None:
    kalman_filter = KalmanFilterXYWH(ndim=ndim)
    measurements = np.array(
        [
            [30.0, 20.0, 12.0, 6.0, 0.2],
            [80.0, 50.0, 20.0, 10.0, -0.4],
        ],
        dtype=float,
    )[:, :ndim]
    states = [kalman_filter.initiate(measurement) for measurement in measurements]
    means = np.stack([state[0] for state in states])
    covariances = np.stack([state[1] for state in states])

    batch_mean, batch_covariance = kalman_filter.multi_predict(means.copy(), covariances.copy())
    individual = [kalman_filter.predict(mean.copy(), covariance.copy()) for mean, covariance in states]

    np.testing.assert_allclose(batch_mean, np.stack([state[0] for state in individual]))
    np.testing.assert_allclose(batch_covariance, np.stack([state[1] for state in individual]))


def test_xywh_obb_gating_treats_equivalent_representations_equally() -> None:
    kalman_filter = KalmanFilterXYWH(ndim=5)
    reference = np.array([30.0, 20.0, 12.0, 6.0, 0.2])
    equivalent = np.array([30.0, 20.0, 6.0, 12.0, 0.2 + (np.pi / 2.0)])
    mean, covariance = kalman_filter.initiate(reference)

    distances = kalman_filter.gating_distance(
        mean,
        covariance,
        np.stack([reference, equivalent]),
    )

    np.testing.assert_allclose(distances, [0.0, 0.0], atol=1e-10)
