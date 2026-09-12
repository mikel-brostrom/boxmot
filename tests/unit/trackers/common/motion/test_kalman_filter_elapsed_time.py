"""Elapsed-time prediction contracts shared by the box Kalman filters."""

from copy import deepcopy

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter
from boxmot.trackers.common.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.trackers.common.motion.kalman_filters.xyhr import KalmanFilterXYHR
from boxmot.trackers.common.motion.kalman_filters.xyscr import KalmanFilterXYSCR
from boxmot.trackers.common.motion.kalman_filters.xysr import KalmanFilterXYSR
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH

FILTER_MODES = (
    "xyah",
    "xyah_obb",
    "xywh",
    "xywh_obb",
    "xyhr",
    "xyhr_obb",
    "xysr",
    "xysr_obb",
    "xyscr",
)
BATCH_MODES = FILTER_MODES[:6]


def _make_filter(mode: str) -> BaseKalmanFilter:
    """Initialize a valid box with translation and, where supported, rotation."""
    name = mode.split("_")[0]
    dim_z = 5 if mode.endswith("_obb") else 4
    measurements = {
        "xyah": [100.0, 80.0, 2.0, 20.0],
        "xywh": [100.0, 80.0, 40.0, 20.0],
        "xyhr": [100.0, 80.0, 20.0, 2.0],
        "xysr": [100.0, 80.0, 800.0, 2.0],
        "xyscr": [100.0, 80.0, 800.0, 0.8, 2.0],
    }
    measurement = measurements[name]
    if mode.endswith("_obb"):
        measurement = [*measurement, 0.3]

    if name == "xyah":
        kf = KalmanFilterXYAH(ndim=dim_z)
    elif name == "xywh":
        kf = KalmanFilterXYWH(ndim=dim_z)
    elif name == "xyhr":
        kf = KalmanFilterXYHR(dim_z=dim_z, ndim=2 * dim_z)
    elif name == "xysr":
        kf = KalmanFilterXYSR(dim_z=dim_z, dim_x=2 * dim_z - 1)
    else:
        kf = KalmanFilterXYSCR()
    kf.x, kf.P = kf.initiate(np.asarray(measurement))
    state = kf.x.reshape(-1)
    state[kf.dim_z : kf.dim_z + 2] = [30.0, -6.0]
    if mode.endswith("_obb"):
        state[-1] = 0.2
    elif name == "xyscr":
        state[-1] = 0.03
    return kf


def _predict(kf: BaseKalmanFilter, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
    """Exercise each filter's public prediction interface."""
    if isinstance(kf, (KalmanFilterXYAH, KalmanFilterXYWH)):
        kf.x, kf.P = kf.predict(kf.x, kf.P, **kwargs)
    else:
        kf.predict(**kwargs)
    return kf.x, kf.P


@pytest.mark.parametrize("mode", FILTER_MODES)
@pytest.mark.parametrize("dt", [0.1, 0.5, 2.0])
def test_elapsed_time_moves_position_and_optional_angle_or_score(mode: str, dt: float) -> None:
    kf = _make_filter(mode)
    initial_geometry = kf.x.reshape(-1)[2:4].copy()

    mean, _ = _predict(kf, dt=dt)
    state = mean.reshape(-1)

    np.testing.assert_allclose(state[:2], [100.0 + 30.0 * dt, 80.0 - 6.0 * dt])
    np.testing.assert_allclose(state[kf.dim_z : kf.dim_z + 2], [30.0, -6.0])
    if mode.endswith("_obb"):
        assert state[4] == pytest.approx(0.3 + 0.2 * dt)
    if mode == "xyscr":
        assert state[3] == pytest.approx(0.8 + 0.03 * dt)
        assert state[4] == pytest.approx(2.0)
    else:
        np.testing.assert_allclose(state[2:4], initial_geometry)


@pytest.mark.parametrize("mode", FILTER_MODES)
def test_elapsed_prediction_composes_over_irregular_intervals(mode: str) -> None:
    """A constant-size object has the same forecast across skipped predictions."""
    single = _make_filter(mode)
    split = deepcopy(single)

    for dt in (0.1, 0.7, 0.25):
        _predict(split, dt=dt)
    _predict(single, dt=1.05)

    np.testing.assert_allclose(split.x, single.x, atol=1e-12)
    np.testing.assert_allclose(split.P, single.P, atol=1e-12)


@pytest.mark.parametrize("mode", FILTER_MODES)
def test_longer_elapsed_time_increases_process_uncertainty(mode: str) -> None:
    short = _make_filter(mode)
    short.P = np.zeros_like(short.P)
    long = deepcopy(short)

    _predict(short, dt=0.1)
    _predict(long, dt=2.0)

    np.testing.assert_allclose(long.P, long.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(short.P).min() >= -1e-12
    assert np.linalg.eigvalsh(long.P - short.P).min() >= -1e-12
    assert long.P[0, 0] > short.P[0, 0] > 0.0
    assert long.P[0, long.dim_z] > 0.0


@pytest.mark.parametrize("mode", FILTER_MODES)
def test_explicit_elapsed_time_does_not_change_default_prediction(mode: str) -> None:
    """A per-call interval must not leak into the filter's default matrices."""
    reference = _make_filter(mode)
    kf = deepcopy(reference)
    _predict(kf, dt=2.5)
    kf.x, kf.P = reference.x.copy(), reference.P.copy()

    _predict(kf)
    _predict(reference)

    np.testing.assert_allclose(kf.x, reference.x)
    np.testing.assert_allclose(kf.P, reference.P)
    np.testing.assert_array_equal(kf.F, reference.F)
    np.testing.assert_array_equal(kf.Q, reference.Q)


@pytest.mark.parametrize("mode", FILTER_MODES)
@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf, -np.inf, True, "0.2", [0.2]])
def test_invalid_elapsed_time_is_rejected_before_state_changes(mode: str, dt: object) -> None:
    kf = _make_filter(mode)
    before = deepcopy(kf)

    with pytest.raises(ValueError, match="dt"):
        _predict(kf, dt=dt)

    for name in ("x", "P", "F", "Q", "x_prior", "P_prior"):
        np.testing.assert_array_equal(getattr(kf, name), getattr(before, name))
    assert kf.dt == before.dt
    assert list(kf.history_obs) == list(before.history_obs)


@pytest.mark.parametrize("mode", BATCH_MODES)
def test_elapsed_batch_prediction_matches_individual_tracks(mode: str) -> None:
    first = _make_filter(mode)
    second = _make_filter(mode)
    second.x[:4] *= 1.5
    second.x[second.dim_z] = -12.0
    means = np.stack([first.x, second.x])
    covariances = np.stack([first.P, second.P])
    means_before, covariances_before = means.copy(), covariances.copy()

    batch_mean, batch_covariance = first.multi_predict(means, covariances, dt=0.7)
    _predict(first, dt=0.7)
    _predict(second, dt=0.7)

    np.testing.assert_allclose(batch_mean, np.stack([first.x, second.x]))
    np.testing.assert_allclose(batch_covariance, np.stack([first.P, second.P]))
    np.testing.assert_array_equal(means, means_before)
    np.testing.assert_array_equal(covariances, covariances_before)


@pytest.mark.parametrize("mode", FILTER_MODES)
def test_elapsed_batch_prediction_accepts_empty_tracks(mode: str) -> None:
    kf = _make_filter(mode)
    means = np.empty((0, kf.dim_x))
    covariances = np.empty((0, kf.dim_x, kf.dim_x))

    predicted_mean, predicted_covariance = kf.multi_predict(means, covariances, dt=0.7)

    assert predicted_mean.shape == means.shape
    assert predicted_covariance.shape == covariances.shape
    with pytest.raises(ValueError, match="dt"):
        kf.multi_predict(means, covariances, dt=0.0)


@pytest.mark.parametrize("mode", ["xysr", "xysr_obb", "xyscr"])
def test_scale_noise_batch_prediction_composes_over_elapsed_intervals(mode: str) -> None:
    """The matrix-based filters also expose a batch path with scale-based noise."""
    kf = _make_filter(mode)
    means = kf.x.reshape(1, -1)
    covariances = kf.P[None, ...]

    short_mean, short_covariance = kf.multi_predict(means, covariances, dt=0.2)
    split_mean, split_covariance = kf.multi_predict(short_mean, short_covariance, dt=0.8)
    whole_mean, whole_covariance = kf.multi_predict(means, covariances, dt=1.0)

    np.testing.assert_allclose(split_mean, whole_mean)
    np.testing.assert_allclose(split_covariance, whole_covariance)
    np.testing.assert_allclose(whole_mean[0, :2], [130.0, 74.0])
    assert np.linalg.eigvalsh(whole_covariance[0]).min() >= -1e-12
    if mode == "xysr_obb":
        assert whole_mean[0, 4] == pytest.approx(0.5)
        assert whole_mean[0, 3] == pytest.approx(2.0)
    elif mode == "xyscr":
        assert whole_mean[0, 3] == pytest.approx(0.83)
        assert whole_mean[0, 4] == pytest.approx(2.0)


@pytest.mark.parametrize("mode", ["xysr", "xysr_obb", "xyscr"])
def test_elapsed_time_preserves_discrete_matrix_overrides(mode: str) -> None:
    kf = _make_filter(mode)
    initial_x, initial_p = kf.x.copy(), kf.P.copy()
    transition = np.eye(kf.dim_x)
    transition[0, kf.dim_z] = 0.25
    noise = np.eye(kf.dim_x) * 3.0

    kf.predict(dt=2.0, F=transition, Q=noise)

    np.testing.assert_allclose(kf.x, transition @ initial_x)
    np.testing.assert_allclose(kf.P, transition @ initial_p @ transition.T + noise)


def test_xyhr_explicit_interval_overrides_constructor_interval() -> None:
    kf = KalmanFilterXYHR(np.array([100.0, 80.0, 20.0, 2.0]), dt=0.5)
    kf.x[4] = 30.0

    kf.predict(dt=0.1)
    assert kf.x[0] == pytest.approx(103.0)
    kf.predict()
    assert kf.x[0] == pytest.approx(118.0)


@pytest.mark.parametrize("mode", ["xysr", "xysr_obb", "xyscr"])
def test_irregular_missing_observation_replay_matches_timed_observations(mode: str) -> None:
    """Recovery fills missing observations at capture times and corrects once."""
    recovered = _make_filter(mode)
    anchor = recovered.x[: recovered.dim_z].copy()
    if mode.endswith("_obb"):
        anchor[4, 0] = np.pi - 0.1
    recovered.update(anchor)
    reference = deepcopy(recovered)
    intervals = [0.1, 0.7, 0.2, 0.4]
    duration = sum(intervals)
    elapsed = 0.0

    for index, dt in enumerate(intervals):
        elapsed += dt
        measurement = anchor.copy()
        measurement[0, 0] += 30.0 * elapsed
        measurement[1, 0] -= 6.0 * elapsed
        if mode.endswith("_obb"):
            angle = anchor[4, 0] + 0.3 * elapsed / duration
            measurement[4, 0] = (angle + np.pi) % (2.0 * np.pi) - np.pi
        elif mode == "xyscr":
            measurement[3, 0] += 0.03 * elapsed

        reference.predict(dt=dt)
        reference.update(measurement)
        recovered.predict(dt=dt)
        recovered.update(measurement if index == len(intervals) - 1 else None)

    np.testing.assert_allclose(recovered.x, reference.x, atol=1e-10)
    np.testing.assert_allclose(recovered.P, reference.P, atol=1e-10)
    np.testing.assert_allclose(recovered.x_prior, reference.x_prior, atol=1e-10)
    np.testing.assert_allclose(recovered.P_prior, reference.P_prior, atol=1e-10)


@pytest.mark.parametrize("mode", ["xysr", "xysr_obb", "xyscr"])
def test_fixed_step_predictions_do_not_retain_replay_matrices(mode: str) -> None:
    """Default prediction avoids timed replay allocations even during a gap."""
    kf = _make_filter(mode)
    kf.update(kf.x[: kf.dim_z].copy())
    for _ in range(3):
        kf.predict()
        kf.update(None)
    assert kf._prediction_steps == []
    assert kf._prediction_origin is None
    assert not kf._time_aware


@pytest.mark.parametrize("mode", ["xysr", "xysr_obb", "xyscr"])
def test_enabling_timing_during_unrecorded_gap_corrects_current_prediction(mode: str) -> None:
    """Partial elapsed history must not replay against an older observation."""
    kf = _make_filter(mode)
    anchor = kf.x[: kf.dim_z].copy()
    kf.update(anchor)
    kf.predict()
    kf.update(None)
    kf.predict(dt=0.2)
    kf.update(None)
    kf.predict(dt=0.7)
    assert kf._prediction_steps == []
    reference = deepcopy(kf)
    recovered = anchor.copy()
    recovered[0, 0] += 10.0
    reference._correct_observation(recovered)
    kf.update(recovered)
    np.testing.assert_array_equal(kf.x, reference.x)
    np.testing.assert_array_equal(kf.P, reference.P)
    kf.predict(dt=0.1)
    assert len(kf._prediction_steps) == 1


@pytest.mark.parametrize("mode", FILTER_MODES)
@pytest.mark.parametrize("dt", [1e200, 10**1000], ids=["covariance_overflow", "float_overflow"])
def test_unrepresentable_elapsed_time_is_rejected_without_state_mutation(mode: str, dt: float) -> None:
    """Finite inputs whose model exceeds floating-point range fail explicitly."""
    kf = _make_filter(mode)
    before = deepcopy(kf)
    with pytest.raises(ValueError, match="dt"):
        _predict(kf, dt=dt)
    np.testing.assert_array_equal(kf.x, before.x)
    np.testing.assert_array_equal(kf.P, before.P)
    assert kf._time_aware == before._time_aware
