import numpy as np

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter
from boxmot.motion.kalman_filters.obb.xywha_kf import KalmanFilterXYWHA


def test_xywha_kf_inherits_base_kalman_filter():
    assert issubclass(KalmanFilterXYWHA, BaseKalmanFilter)


def test_xywha_kf_unfreeze_after_missing_observations():
    kf = KalmanFilterXYWHA(dim_x=10, dim_z=5, max_obs=50)

    # Match the OBB constant-velocity model used by KalmanBoxTrackerOBB.
    kf.F = np.array(
        [
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    kf.H = np.zeros((5, 10), dtype=float)
    kf.H[:, :5] = np.eye(5, dtype=float)

    obs1 = np.array([[300.0], [200.0], [120.0], [60.0], [0.1]])
    obs2 = np.array([[320.0], [210.0], [122.0], [61.0], [0.2]])
    obs3 = np.array([[350.0], [230.0], [125.0], [63.0], [0.3]])

    kf.predict()
    kf.update(obs1)
    kf.predict()
    kf.update(obs2)

    for _ in range(5):
        kf.predict()
        kf.update(None)

    kf.predict()
    kf.update(obs3)

    assert np.all(np.isfinite(kf.x)), "State contains non-finite values"
    assert kf.x.shape == (10, 1)
    assert abs(float(kf.x[0, 0]) - 350.0) < 100.0
    assert abs(float(kf.x[1, 0]) - 230.0) < 100.0


def test_xywha_kf_uses_shortest_angular_residual():
    kf = KalmanFilterXYWHA(dim_x=10, dim_z=5, max_obs=50)
    kf.H = np.zeros((5, 10), dtype=float)
    kf.H[:, :5] = np.eye(5, dtype=float)
    kf.P = np.eye(10, dtype=float)
    kf.x[:5] = np.array([[300.0], [200.0], [120.0], [60.0], [np.pi - 0.01]])

    # Equivalent orientation across the wrap boundary.
    z = np.array([[300.0], [200.0], [120.0], [60.0], [-np.pi + 0.02]])
    kf.update(z)

    assert abs(float(kf.y[4, 0])) < 0.1
    assert -np.pi <= float(kf.x[4, 0]) < np.pi


def test_xywha_kf_affine_correction_updates_angle_and_size():
    kf = KalmanFilterXYWHA(dim_x=10, dim_z=5, max_obs=50)
    kf.x[:5] = np.array([[1.0], [2.0], [10.0], [5.0], [0.0]])

    theta = np.pi / 2.0
    m = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    t = np.array([[2.0], [3.0]], dtype=float)

    kf.apply_affine_correction(m, t)

    assert np.allclose(kf.x[0:2], np.array([[0.0], [4.0]]), atol=1e-6)
    assert np.isclose(float(kf.x[2, 0]), 10.0, atol=1e-6)
    assert np.isclose(float(kf.x[3, 0]), 5.0, atol=1e-6)
    assert np.isclose(float(kf.x[4, 0]), np.pi / 2.0, atol=1e-5)


def test_xywha_kf_unfreeze_handles_angle_wrap():
    kf = KalmanFilterXYWHA(dim_x=10, dim_z=5, max_obs=50)
    kf.H = np.zeros((5, 10), dtype=float)
    kf.H[:, :5] = np.eye(5, dtype=float)

    obs1 = np.array([[300.0], [200.0], [120.0], [60.0], [np.pi - 0.05]])
    obs2 = np.array([[320.0], [210.0], [122.0], [61.0], [-np.pi + 0.04]])
    obs3 = np.array([[350.0], [230.0], [125.0], [63.0], [np.pi - 0.02]])

    kf.predict()
    kf.update(obs1)
    kf.predict()
    kf.update(obs2)

    for _ in range(5):
        kf.predict()
        kf.update(None)

    kf.predict()
    kf.update(obs3)

    assert abs(float(kf.y[4, 0])) < 0.25
    assert -np.pi <= float(kf.x[4, 0]) < np.pi
