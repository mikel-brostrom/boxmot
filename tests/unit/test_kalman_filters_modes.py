import numpy as np
import pytest

from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.motion.kalman_filters.xyhr import KalmanFilterXYHR
from boxmot.motion.kalman_filters.xysr import KalmanFilterXYSR
from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


@pytest.mark.parametrize(
    ("kf_cls", "init_measurement", "update_measurement"),
    [
        (
            KalmanFilterXYWH,
            np.array([100.0, 80.0, 40.0, 20.0]),
            np.array([101.0, 79.5, 40.5, 20.5]),
        ),
        (
            KalmanFilterXYAH,
            np.array([100.0, 80.0, 1.6, 60.0]),
            np.array([100.5, 80.2, 1.58, 60.1]),
        ),
    ],
)
def test_xywh_xyah_support_aabb_mode(kf_cls, init_measurement, update_measurement):
    kf = kf_cls(ndim=4)
    mean, covariance = kf.initiate(init_measurement)
    mean, covariance = kf.predict(mean, covariance)
    new_mean, new_cov = kf.update(mean, covariance, update_measurement, confidence=0.9)

    assert new_mean.shape == (8,)
    assert new_cov.shape == (8, 8)
    assert new_mean[2] > 0 and new_mean[3] > 0
    assert np.all(np.isfinite(new_mean))
    assert np.all(np.isfinite(new_cov))
    distance = kf.gating_distance(new_mean, new_cov, update_measurement[None, :])
    assert distance.shape == (1,)
    assert np.isfinite(distance[0])


@pytest.mark.parametrize(
    ("kf_cls", "init_measurement", "update_measurement"),
    [
        (
            KalmanFilterXYWH,
            np.array([100.0, 80.0, 40.0, 20.0, np.pi - 0.01]),
            np.array([101.0, 79.5, 40.5, 20.5, -np.pi + 0.02]),
        ),
        (
            KalmanFilterXYAH,
            np.array([100.0, 80.0, 1.6, 60.0, np.pi - 0.03]),
            np.array([100.5, 80.2, 1.58, 60.1, -np.pi + 0.01]),
        ),
    ],
)
def test_xywh_xyah_support_obb_mode(kf_cls, init_measurement, update_measurement):
    kf = kf_cls(ndim=5)
    mean, covariance = kf.initiate(init_measurement)
    mean, covariance = kf.predict(mean, covariance)
    new_mean, new_cov = kf.update(mean, covariance, update_measurement, confidence=0.9)

    assert new_mean.shape == (10,)
    assert new_cov.shape == (10, 10)
    assert new_mean[2] > 0 and new_mean[3] > 0
    assert -np.pi <= float(new_mean[4]) < np.pi
    assert abs(_angle_diff(float(new_mean[4]), update_measurement[4])) < 0.2
    distance = kf.gating_distance(new_mean, new_cov, update_measurement[None, :])
    assert distance.shape == (1,)
    assert np.isfinite(distance[0])


def test_xysr_supports_aabb_mode():
    kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=50)
    init_measurement = np.array([[300.0], [200.0], [50000.0], [1.5]])
    mean, covariance = kf.initiate(init_measurement)
    kf.x = mean.copy()
    kf.P = covariance.copy()

    kf.predict()
    measurement = np.array([[305.0], [202.0], [50500.0], [1.45]])
    kf.update(measurement)

    assert kf.x.shape == (7, 1)
    assert kf.P.shape == (7, 7)
    assert kf.x[2, 0] > 0 and kf.x[3, 0] > 0
    assert np.all(np.isfinite(kf.x))
    assert np.all(np.isfinite(kf.P))
    assert np.isfinite(kf.md_for_measurement(measurement))


def test_xysr_supports_obb_mode():
    kf = KalmanFilterXYSR(dim_x=9, dim_z=5, max_obs=50)
    init_measurement = np.array([[300.0], [200.0], [50000.0], [1.5], [np.pi - 0.01]])
    mean, covariance = kf.initiate(init_measurement)
    kf.x = mean.copy()
    kf.P = covariance.copy()

    kf.predict()
    measurement = np.array([[305.0], [202.0], [50500.0], [1.45], [-np.pi + 0.02]])
    kf.update(measurement)

    assert kf.x.shape == (9, 1)
    assert kf.P.shape == (9, 9)
    assert kf.x[2, 0] > 0 and kf.x[3, 0] > 0
    assert -np.pi <= float(kf.x[4, 0]) < np.pi
    assert abs(float(kf.y[4, 0])) < 0.2
    assert abs(float(kf.x[8, 0])) < 1e-12
    assert np.isfinite(kf.md_for_measurement(measurement))


def test_xysr_obb_aligns_equivalent_ratio_angle_forms():
    kf = KalmanFilterXYSR(dim_x=9, dim_z=5, max_obs=50)

    theta_ref = 0.35
    init_measurement = np.array([[300.0], [200.0], [50000.0], [2.0], [theta_ref]])
    mean, covariance = kf.initiate(init_measurement)
    kf.x = mean.copy()
    kf.P = covariance.copy()

    kf.predict()

    # Equivalent rectangle representation in XYSR:
    # r -> 1/r and theta -> theta + pi/2
    equivalent_measurement = np.array(
        [[300.5], [199.5], [50050.0], [0.5], [theta_ref + (np.pi / 2.0)]]
    )
    kf.update(equivalent_measurement)

    assert abs(_angle_diff(float(kf.x[4, 0]), theta_ref)) < 0.25
    assert abs(np.log(float(kf.x[3, 0]) / 2.0)) < 0.35
    assert abs(float(kf.x[8, 0])) < 1e-12


def test_xysr_obb_unfreeze_handles_angle_wrap():
    kf = KalmanFilterXYSR(dim_x=9, dim_z=5, max_obs=50)

    obs1 = np.array([[300.0], [200.0], [50000.0], [1.5], [np.pi - 0.05]])
    obs2 = np.array([[320.0], [210.0], [51000.0], [1.4], [-np.pi + 0.04]])
    obs3 = np.array([[350.0], [230.0], [52000.0], [1.3], [np.pi - 0.02]])

    kf.predict()
    kf.update(obs1)
    kf.predict()
    kf.update(obs2)

    for _ in range(5):
        kf.predict()
        kf.update(None)

    kf.predict()
    kf.update(obs3)

    assert np.all(np.isfinite(kf.x))
    assert -np.pi <= float(kf.x[4, 0]) < np.pi
    assert abs(float(kf.y[4, 0])) < 0.3


def test_xysr_unfreeze_with_column_vectors():
    """Regression test for mikel-brostrom/boxmot#2207."""
    kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=50)

    kf.F = np.eye(7)
    kf.F[:4, 4:] = np.pad(np.eye(3), ((0, 1), (0, 0)))
    kf.H = np.zeros((4, 7))
    kf.H[:4, :4] = np.eye(4)
    kf.R *= 10.0

    obs1 = np.array([[300.0], [200.0], [50000.0], [1.5]])
    obs2 = np.array([[320.0], [210.0], [51000.0], [1.4]])

    kf.predict()
    kf.update(obs1)
    kf.predict()
    kf.update(obs2)

    for _ in range(5):
        kf.predict()
        kf.update(None)

    obs3 = np.array([[350.0], [230.0], [52000.0], [1.3]])
    kf.predict()
    kf.update(obs3)

    state = kf.x.flatten()
    assert np.all(np.isfinite(state)), f"State contains non-finite values: {state}"
    assert abs(state[0] - 350.0) < 100
    assert abs(state[1] - 230.0) < 100


def test_xysr_unfreeze_insufficient_history():
    """Guard against missing replay anchors after history truncation."""
    kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=4)

    kf.F = np.eye(7)
    kf.F[:4, 4:] = np.pad(np.eye(3), ((0, 1), (0, 0)))
    kf.H = np.zeros((4, 7))
    kf.H[:4, :4] = np.eye(4)
    kf.R *= 10.0

    obs1 = np.array([[300.0], [200.0], [50000.0], [1.5]])

    kf.predict()
    kf.update(obs1)

    for _ in range(10):
        kf.predict()
        kf.update(None)

    obs2 = np.array([[320.0], [210.0], [51000.0], [1.4]])
    kf.predict()
    kf.update(obs2)

    state = kf.x.flatten()
    assert np.all(np.isfinite(state)), f"State contains non-finite values: {state}"


def test_xyhr_supports_aabb_mode_and_column_measurement():
    kf = KalmanFilterXYHR(np.array([[100.0], [80.0], [40.0], [1.2]]), dim_z=4, ndim=8)
    pred_x, pred_cov = kf.predict()
    upd_x, upd_cov = kf.update(np.array([101.0, 80.5, 39.0, 1.25]))

    assert pred_x.shape == (8,)
    assert pred_cov.shape == (8, 8)
    assert upd_x.shape == (8,)
    assert upd_cov.shape == (8, 8)
    assert upd_x[2] > 0 and upd_x[3] > 0
    assert np.all(np.isfinite(upd_x))
    assert np.all(np.isfinite(upd_cov))


def test_xyhr_supports_obb_mode_and_inference():
    inferred = KalmanFilterXYHR(np.array([10.0, 11.0, 12.0, 1.1, 0.3]))
    assert inferred.dim_z == 5
    assert inferred.dim_x == 10

    kf = KalmanFilterXYHR(
        np.array([100.0, 80.0, 40.0, 1.2, np.pi - 0.02]),
        dim_z=5,
        ndim=10,
    )
    pred_x, pred_cov = kf.predict()
    measurement = np.array([101.0, 80.5, 39.0, 1.25, -np.pi + 0.01])
    upd_x, upd_cov = kf.update(measurement)

    assert pred_x.shape == (10,)
    assert pred_cov.shape == (10, 10)
    assert upd_x.shape == (10,)
    assert upd_cov.shape == (10, 10)
    assert upd_x[2] > 0 and upd_x[3] > 0
    assert -np.pi <= float(upd_x[4]) < np.pi
    assert abs(_angle_diff(float(upd_x[4]), measurement[4])) < 0.2
