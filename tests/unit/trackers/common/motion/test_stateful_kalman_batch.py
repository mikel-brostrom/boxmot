"""Scalar parity for batched stateful filters and their observation histories."""

from copy import deepcopy

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.kalman_filters.xyhr import ConstantNoiseXYHR, KalmanFilterXYHR
from boxmot.trackers.common.motion.kalman_filters.xyscr import KalmanFilterXYSCR
from boxmot.trackers.common.motion.kalman_filters.xysr import KalmanFilterXYSR


def _filters(kind: str, *, seconds: bool = False, adaptive: bool = False) -> list:
    """Create distinct states, correlated covariances, and instance-local noise."""
    result = []
    rng = np.random.default_rng(149)
    for index in range(4):
        config = KalmanNoiseConfig(
            process_position_scale=0.7 + index,
            process_velocity_scale=1.3 + index,
            measurement_noise_scale=0.8 + index,
            time_unit="seconds" if seconds else "frames",
        )
        if kind.startswith("xyhr"):
            measurement = np.array([20.0 + index, 30.0, 12.0, 1.5, 3.13])
            measurement = measurement if kind.endswith("obb") else measurement[:4]
            kf = KalmanFilterXYHR(measurement, adaptive_kf=adaptive, cls_id=index, noise_config=config)
            if adaptive:
                kf.cov_update_policy._warmup = 3
                kf.cov_update_policy._window = 4
        else:
            if kind == "xyscr":
                kf = KalmanFilterXYSCR(noise_config=config)
                measurement = np.array([20.0 + index, 30.0, 144.0, 0.7, 1.5])
            else:
                obb = kind.endswith("obb")
                kf = KalmanFilterXYSR(dim_x=9 if obb else 7, dim_z=5 if obb else 4, noise_config=config)
                measurement = np.array([20.0 + index, 30.0, 144.0, 1.5, 3.13])[: kf.dim_z]
            kf.x, kf.P = kf.initiate(measurement)
            kf.F[0, kf.dim_z] *= 1.0 + 0.1 * index
            kf._alpha_sq = 1.0 + 0.1 * index
            matrix = rng.normal(size=(kf.dim_x, kf.dim_x))
            kf.Q = matrix @ matrix.T * 0.01
            matrix = rng.normal(size=(kf.dim_z, kf.dim_z))
            kf.R = matrix @ matrix.T * 0.01 + np.eye(kf.dim_z) * 0.1
        matrix = rng.normal(size=(kf.dim_x, kf.dim_x))
        kf.P = matrix @ matrix.T + np.eye(kf.dim_x)
        kf.x.reshape(-1)[kf.dim_z :] = rng.normal(size=kf.dim_x - kf.dim_z) * 0.05
        result.append(kf)
    return result


def _assert_state(actual, expected) -> None:
    """Compare observable filter state, replay metadata and adaptive estimates."""
    for name in ("x", "P", "x_prior", "P_prior", "x_post", "P_post", "K", "y", "S", "SI"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=2e-9, atol=2e-9, err_msg=name)
    if expected.z.dtype == object:
        np.testing.assert_array_equal(actual.z, expected.z)
    else:
        np.testing.assert_allclose(actual.z, expected.z, rtol=2e-9, atol=2e-9)
    assert actual.observed == expected.observed
    assert actual._time_aware == expected._time_aware
    assert actual._unrecorded_prediction == expected._unrecorded_prediction
    assert actual._prediction_history_overflowed == expected._prediction_history_overflowed
    assert len(actual.history_obs) == len(expected.history_obs)
    for observed, reference in zip(actual.history_obs, expected.history_obs):
        if reference is None:
            assert observed is None
        else:
            np.testing.assert_allclose(observed, reference, rtol=2e-9, atol=2e-9)
    assert len(actual._prediction_steps) == len(expected._prediction_steps)
    for observed, reference in zip(actual._prediction_steps, expected._prediction_steps):
        for value, target in zip(observed, reference):
            if target is None:
                assert value is None
            else:
                np.testing.assert_allclose(value, target, rtol=2e-9, atol=2e-9)
    if hasattr(actual, "cov_update_policy"):
        for name in ("_q_adaptive", "_q_rate", "_innovations", "_rate_samples"):
            observed = getattr(actual.cov_update_policy, name, None)
            reference = getattr(expected.cov_update_policy, name, None)
            if reference is None:
                assert observed is None
            else:
                np.testing.assert_allclose(observed, reference, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("kind", ["xysr", "xysr-obb", "xyscr"])
@pytest.mark.parametrize("timing", ["fixed", "elapsed", "seconds", "switch"])
def test_observation_centric_batch_preserves_replay(kind, timing):
    """Missing/recovered observations, angle crossing and varying dt retain parity."""
    expected = _filters(kind, seconds=timing == "seconds")
    actual = deepcopy(expected)
    factory = type(actual[0])
    for frame in range(9):
        dt = None if timing == "fixed" or (timing == "switch" and frame < 3) else (0.02 + frame * 0.004)
        for kf in expected:
            kf.predict(dt=dt)
        factory.predict_many(actual, dt=dt)
        for observed, reference in zip(actual, expected):
            _assert_state(observed, reference)
        measurements = []
        for index, kf in enumerate(expected):
            measurement = kf.x[: kf.dim_z].copy()
            measurement[0] += 0.4 + index * 0.2
            if kind.endswith("obb"):
                measurement[4] -= np.pi
            measurements.append(None if (index + frame) % 5 in (2, 3) else measurement)
        for kf, z in zip(expected, measurements):
            kf.update(None if z is None else z.copy())
        factory.update_many(actual, deepcopy(measurements))
        for observed, reference in zip(actual, expected):
            _assert_state(observed, reference)


@pytest.mark.parametrize("kind", ["xysr", "xysr-obb", "xyscr"])
def test_matrix_overrides_remain_discrete_and_instance_local(kind):
    expected = _filters(kind, seconds=True)
    actual = deepcopy(expected)
    factory = type(actual[0])
    noises = [None, 0.5, np.eye(actual[0].dim_x) * 0.3, None]
    transitions = [np.eye(actual[0].dim_x) * 0.9, None, None, None]
    for kf, noise, transition in zip(expected, noises, transitions):
        kf.predict(dt=0.3, Q=noise, F=transition)
    factory.predict_many(actual, dt=0.3, Q=noises, F=transitions)
    measurements = [kf.x[: kf.dim_z].copy() for kf in expected]
    for kf in expected + actual:
        kf.observed = True
    noise = [None, 0.3, np.eye(actual[0].dim_z) * 0.2, None]
    observation = [None, None, actual[0].H * 0.95, None]
    for kf, z, r, h in zip(expected, measurements, noise, observation):
        kf.update(z.copy(), R=r, H=h)
    factory.update_many(actual, deepcopy(measurements), R=noise, H=observation)
    for observed, reference in zip(actual, expected):
        _assert_state(observed, reference)


@pytest.mark.parametrize("kind", ["xyhr", "xyhr-obb"])
@pytest.mark.parametrize("seconds", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_xyhr_batch_keeps_class_noise_adaptation_and_gain_suppression(kind, seconds, adaptive, monkeypatch):
    monkeypatch.setattr(
        ConstantNoiseXYHR,
        "_per_class_noise",
        {1: {"q_pos_diag": np.array([0.2, 0.3, 0.1, 0.01]), "r_diag": np.array([0.5, 0.6, 0.3, 0.02])}},
    )
    expected = _filters(kind, seconds=seconds, adaptive=adaptive)
    actual = deepcopy(expected)
    for frame in range(8):
        dt = 0.02 + frame * 0.003 if seconds else None
        for kf in expected:
            kf.predict(dt=dt)
        KalmanFilterXYHR.predict_many(actual, dt=dt)
        measurements = []
        for index, kf in enumerate(expected):
            measurement = kf.x[: kf.dim_z].copy()
            measurement[0] += np.sin(frame) + index * 0.1
            if kind.endswith("obb"):
                measurement[4] += 2.0 * np.pi - 0.2
            measurements.append(measurement)
        alphas = [1.0, 0.8, 0.5, 0.0]
        for kf, z, alpha in zip(expected, measurements, alphas):
            kf.update(z.copy(), alpha=alpha)
        KalmanFilterXYHR.update_many(actual, deepcopy(measurements), alpha=alphas)
        for observed, reference in zip(actual, expected):
            _assert_state(observed, reference)


@pytest.mark.parametrize("factory", [KalmanFilterXYSR, KalmanFilterXYSCR, KalmanFilterXYHR])
def test_empty_batch_and_measurement_count(factory):
    factory.predict_many([])
    factory.update_many([], [])
    with pytest.raises(ValueError, match="one measurement"):
        factory.update_many([factory()], [])


@pytest.mark.parametrize("kind", ["xysr", "xysr-obb", "xyscr", "xyhr", "xyhr-obb"])
def test_stateless_noise_helpers_match_individual_scales(kind):
    """The array API retains the previous scale-based noise convention."""
    models = _filters(kind)
    kf = models[0]
    means = np.stack([model.x.reshape(-1) for model in models])
    observed_pos, observed_vel = kf._get_multi_process_noise_std(means)
    expected_pos, expected_vel = zip(*(kf._get_process_noise_std(mean) for mean in means))
    np.testing.assert_allclose(np.asarray(observed_pos).T, expected_pos)
    np.testing.assert_allclose(np.asarray(observed_vel).T, expected_vel)
    np.testing.assert_allclose(
        kf._get_multi_measurement_noise_std(means),
        [kf._get_measurement_noise_std(mean, confidence=0.0) for mean in means],
    )


@pytest.mark.parametrize("kind", ["xysr", "xyhr"])
def test_mixed_aabb_obb_filters_keep_input_order(kind):
    expected = [_filters(kind)[0], _filters(kind + "-obb")[1], _filters(kind)[2]]
    actual = deepcopy(expected)
    factory = type(actual[0])
    for model in expected + actual:
        model.observed = True
    for model in expected:
        model.predict(dt=0.3)
    factory.predict_many(actual, dt=0.3)
    measurements = [model.x[: model.dim_z].copy() for model in expected]
    for model, measurement in zip(expected, measurements):
        model.update(measurement.copy())
    factory.update_many(actual, deepcopy(measurements))
    for observed, reference in zip(actual, expected):
        _assert_state(observed, reference)
