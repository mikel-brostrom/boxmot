import numpy as np

from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.motion.kalman_filters.xysr import KalmanFilterXYSR
from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def test_xywh_kf_supports_obb_mode():
    kf = KalmanFilterXYWH(ndim=5)
    mean, covariance = kf.initiate(np.array([100.0, 80.0, 40.0, 20.0, np.pi - 0.01]))
    mean, covariance = kf.predict(mean, covariance)

    measurement = np.array([101.0, 79.5, 40.5, 20.5, -np.pi + 0.02])
    new_mean, _ = kf.update(mean, covariance, measurement, confidence=0.9)

    assert new_mean.shape[0] == 10
    assert new_mean[2] > 0 and new_mean[3] > 0
    assert -np.pi <= float(new_mean[4]) < np.pi
    assert abs(_angle_diff(float(new_mean[4]), measurement[4])) < 0.2


def test_xyah_kf_supports_obb_mode():
    kf = KalmanFilterXYAH(ndim=5)
    mean, covariance = kf.initiate(np.array([100.0, 80.0, 1.6, 60.0, np.pi - 0.03]))
    mean, covariance = kf.predict(mean, covariance)

    measurement = np.array([100.5, 80.2, 1.58, 60.1, -np.pi + 0.01])
    new_mean, _ = kf.update(mean, covariance, measurement, confidence=0.9)

    assert new_mean.shape[0] == 10
    assert new_mean[2] > 0 and new_mean[3] > 0
    assert -np.pi <= float(new_mean[4]) < np.pi
    assert abs(_angle_diff(float(new_mean[4]), measurement[4])) < 0.2


def test_xysr_kf_supports_obb_mode_stateful_update():
    kf = KalmanFilterXYSR(dim_x=9, dim_z=5, max_obs=50)

    init_measurement = np.array([[300.0], [200.0], [50000.0], [1.5], [np.pi - 0.01]])
    mean, covariance = kf.initiate(init_measurement)
    kf.x = mean.copy()
    kf.P = covariance.copy()

    kf.predict()
    measurement = np.array([[305.0], [202.0], [50500.0], [1.45], [-np.pi + 0.02]])
    kf.update(measurement)

    assert np.all(np.isfinite(kf.x))
    assert kf.x.shape == (9, 1)
    assert kf.x[2, 0] > 0 and kf.x[3, 0] > 0
    assert -np.pi <= float(kf.x[4, 0]) < np.pi
    assert abs(float(kf.y[4, 0])) < 0.2
    assert np.isfinite(kf.md_for_measurement(measurement))


def test_xysr_kf_obb_unfreeze_handles_angle_wrap():
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
