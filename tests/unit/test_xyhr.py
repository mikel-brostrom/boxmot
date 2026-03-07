import numpy as np

from boxmot.motion.kalman_filters.xyhr import KalmanFilterXYHR


def test_xyhr_init_accepts_column_measurement():
    measurement = np.array([[100.0], [80.0], [40.0], [1.2]])
    kf = KalmanFilterXYHR(measurement)

    assert kf.x.shape == (8,)
    assert kf.covariance.shape == (8, 8)
    assert np.allclose(kf.x[:4], measurement.reshape(-1))


def test_xyhr_predict_and_update_stateful():
    kf = KalmanFilterXYHR(np.array([100.0, 80.0, 40.0, 1.2]))

    pred_x, pred_cov = kf.predict()
    assert pred_x.shape == (8,)
    assert pred_cov.shape == (8, 8)

    upd_x, upd_cov = kf.update(np.array([101.0, 80.5, 39.0, 1.25]))
    assert upd_x.shape == (8,)
    assert upd_cov.shape == (8, 8)
    assert np.all(np.isfinite(upd_x))
    assert np.all(np.isfinite(upd_cov))
