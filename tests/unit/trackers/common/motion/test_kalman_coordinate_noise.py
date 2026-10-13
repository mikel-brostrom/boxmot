"""Coordinate calibration preserves native layouts, covariance, and score dynamics."""

import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from boxmot.trackers.common.motion.confidence import ConfidencePrediction
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.noise import (
    KALMAN_DIAGONAL_OPTIONS,
    KalmanNoiseConfig,
)
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model, supported_motion_model_kinds

DIAGONALS = tuple(option.removeprefix("kalman.noise.") for option in KALMAN_DIAGONAL_OPTIONS)


def _profile(model, *, seconds=False, factor=1.0):
    """Use distinct multipliers so coordinate permutation and pooling are visible."""
    return KalmanNoiseConfig(
        process_position_scale=2.0,
        process_velocity_scale=3.0,
        measurement_noise_scale=4.0,
        initial_position_scale=5.0,
        initial_velocity_scale=6.0,
        process_position_diagonal=tuple(factor * np.linspace(0.2, 2.0, model.dim_z)),
        process_velocity_diagonal=tuple(factor * np.linspace(0.3, 3.0, model.dim_x - model.dim_z)),
        measurement_noise_diagonal=tuple(factor * np.linspace(0.4, 4.0, model.dim_z)),
        initial_position_diagonal=tuple(factor * np.linspace(0.5, 5.0, model.dim_z)),
        initial_velocity_diagonal=tuple(factor * np.linspace(0.6, 6.0, model.dim_x - model.dim_z)),
        time_unit="seconds" if seconds else "frames",
        reference_dt_s=0.04,
    )


def _measurement(model):
    """Create valid geometry with confidence in the correct measurement slot."""
    box = [20.0, 30.0, 40.0, 20.0, 0.2, 0.8] if model.is_obb else [0.0, 20.0, 40.0, 40.0, 0.8]
    return model.to_measurement(np.array(box), column=False)


@pytest.mark.parametrize(
    "kind,obb",
    [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)],
)
@pytest.mark.parametrize("timing", ["fixed", "elapsed", "seconds"])
def test_native_coordinate_covariances_match_independent_matrix_oracle(kind, obb, timing):
    """P, Q, and R apply each multiplier exactly once, before time integration."""
    model = create_motion_model(kind, is_obb=obb)
    config = _profile(model, seconds=timing == "seconds")
    baseline = model.create_filter()
    actual = model.create_filter(noise_config=config)
    measurement = _measurement(model)
    mean, covariance = model.initiate(baseline, measurement)
    actual_mean, initial = model.initiate(actual, measurement)
    np.testing.assert_array_equal(actual_mean, mean)
    factors = np.sqrt(
        np.r_[5.0 * np.array(config.initial_position_diagonal), 6.0 * np.array(config.initial_velocity_diagonal)]
    )
    if timing == "seconds":
        factors[model.dim_z :] /= 0.04
    np.testing.assert_allclose(initial, covariance * factors[:, None] * factors[None, :])
    zero = np.zeros_like(covariance)
    _, default_q = model.predict(baseline, mean, zero)
    factors = np.sqrt(
        np.r_[2.0 * np.array(config.process_position_diagonal), 3.0 * np.array(config.process_velocity_diagonal)]
    )
    if timing == "seconds":
        factors /= np.sqrt(0.04)
        factors[model.dim_z :] /= 0.04
    expected_q = default_q * factors[:, None] * factors[None, :]
    dt = None if timing == "fixed" else 0.083
    if dt is not None:
        generator = baseline._motion_mat - np.eye(model.dim_x)
        expected_q = (
            dt * expected_q
            + dt**2 / 2.0 * (generator @ expected_q + expected_q @ generator.T)
            + dt**3 / 3.0 * generator @ expected_q @ generator.T
        )
    _, actual_q = model.predict(actual, mean, zero, dt=dt)
    np.testing.assert_allclose(actual_q, expected_q, atol=1e-10)
    _, default_r = model.project(baseline, mean, zero)
    _, actual_r = model.project(actual, mean, zero)
    factors = np.sqrt(4.0 * np.array(config.measurement_noise_diagonal))
    np.testing.assert_allclose(actual_r, default_r * factors[:, None] * factors[None, :])
    assert np.linalg.eigvalsh(actual_q).min() >= -1e-10


@pytest.mark.parametrize(
    "kind,obb",
    [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)],
)
@pytest.mark.parametrize("seconds", [False, True])
@pytest.mark.parametrize("observation_centric", [False, True])
def test_coordinate_profiles_preserve_grouped_and_scalar_trajectories(kind, obb, seconds, observation_centric):
    """Batch prediction/correction honors distinct track-local vector profiles."""
    model = create_motion_model(kind, is_obb=obb, adaptive_kf=kind is MotionModelKind.XYHR)
    expected = []
    for index in range(3):
        kf = model.create_filter(noise_config=_profile(model, seconds=seconds, factor=1.0 + index))
        model.initialize_filter(kf, _measurement(model))
        expected.append(kf)
    actual = deepcopy(expected)
    for frame in range(20):
        dt = 0.025 + frame * 0.003 if seconds else None
        for kf in expected:
            model.predict_filter(kf, dt=dt, observation_centric=observation_centric)
        model.predict_filters(actual, dt=dt, observation_centric=observation_centric)
        observations = []
        for index in range(3):
            observation = _measurement(model)
            observation[0] += 0.4 * frame + index
            observations.append(None if frame in (6, 7) else observation)
        for kf, measurement in zip(expected, observations):
            model.update_filter(kf, measurement, observation_centric=observation_centric)
        model.update_filters(actual, observations, observation_centric=observation_centric)
        for observed, reference in zip(actual, expected):
            np.testing.assert_allclose(observed.x, reference.x, rtol=2e-9, atol=2e-9)
            np.testing.assert_allclose(observed.P, reference.P, rtol=2e-9, atol=2e-9)
            assert np.linalg.eigvalsh(observed.P).min() >= -1e-9




@pytest.mark.parametrize("field", DIAGONALS)
@pytest.mark.parametrize("value", [True, 1.0, "1", np.array([1.0]), [False], [0.0], [-1.0], [np.nan], [np.inf], ["2"]])
def test_coordinate_vectors_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        KalmanNoiseConfig(**{field: value})


@pytest.mark.parametrize(
    "kind,obb",
    [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)],
)
@pytest.mark.parametrize("field", DIAGONALS)
@pytest.mark.parametrize("by_class", [False, True])
def test_filter_creation_rejects_wrong_coordinate_layouts(kind, obb, field, by_class):
    model = create_motion_model(kind, is_obb=obb)
    noise = KalmanNoiseConfig(**{field: (1.0,)})
    if by_class:
        noise = KalmanNoiseConfig(by_class={2: noise})
    with pytest.raises(ValueError, match=f"{field}.*expected"):
        model.create_filter(noise_config=noise)


def test_coordinate_config_round_trips_without_mutable_aliases():
    source = [1.0, 4.0]
    noise = KalmanNoiseConfig(
        measurement_noise_diagonal=source, by_class={2: KalmanNoiseConfig(process_position_diagonal=[9.0, 16.0])}
    )
    source[0] = 100.0
    assert noise.measurement_noise_diagonal == (1.0, 4.0)
    with pytest.raises(FrozenInstanceError):
        noise.measurement_noise_diagonal = (2.0, 2.0)
    assert not any(field in KalmanNoiseConfig().to_dict() for field in DIAGONALS)
    payload = noise.to_dict()
    assert payload["measurement_noise_diagonal"] == [1.0, 4.0]
    assert KalmanNoiseConfig.from_mapping(payload) == pickle.loads(pickle.dumps(noise)) == noise
    nested = {"kalman": {"noise": noise.to_dict(), "variable_dt": True}}
    dotted = {
        "kalman.variable_dt": True,
        "kalman.noise.measurement_noise_diagonal": [1.0, 4.0],
        "kalman.noise.by_class.2.process_position_diagonal": [9.0, 16.0],
    }
    assert normalize_kalman_config(nested).noise == normalize_kalman_config(dotted).noise
    payload["measurement_noise_diagonal"][0] = 999.0
    assert noise.measurement_noise_diagonal == (1.0, 4.0)


@pytest.mark.parametrize("batch", [False, True])
def test_coordinate_covariance_congruence_preserves_off_diagonal_psd(batch):
    covariance = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
    if batch:
        covariance = np.stack([covariance, covariance * 2.0])
    config = KalmanNoiseConfig(
        process_position_diagonal=(4.0, 9.0),
        process_velocity_diagonal=(16.0,),
        initial_position_diagonal=(4.0, 9.0),
        initial_velocity_diagonal=(16.0,),
        measurement_noise_diagonal=(4.0, 9.0, 16.0),
    )
    expected = covariance * np.array([2.0, 3.0, 4.0])[:, None] * np.array([2.0, 3.0, 4.0])[None, :]
    for result in (
        config.process_covariance(covariance, 2),
        config.initial_covariance(covariance, 2),
        config.measurement_covariance(covariance),
    ):
        np.testing.assert_allclose(result, expected)
        assert np.linalg.eigvalsh(result).min() > 0.0


@pytest.mark.parametrize(
    "kind,obb",
    [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)],
)
@pytest.mark.parametrize("seconds", [False, True])
def test_auxiliary_confidence_is_independent_of_geometry_coordinate_calibration(kind, obb, seconds):
    model = create_motion_model(kind, is_obb=obb)
    config = _profile(model, seconds=seconds)
    globals_only = replace(config, **{field: () for field in DIAGONALS})
    actual = ConfidencePrediction(0.8, noise_config=config)
    expected = ConfidencePrediction(0.8, noise_config=globals_only)
    for score in (0.7, 0.9, None, None, 0.6, 0.85):
        dt = 0.041 if seconds else None
        actual.predict(dt=dt)
        expected.predict(dt=dt)
        actual.update(score)
        expected.update(score)
        assert actual.value == pytest.approx(expected.value)
        np.testing.assert_array_equal(actual._filter.P, expected._filter.P)


@pytest.mark.parametrize("obb", [False, True])
@pytest.mark.parametrize("seconds", [False, True])
def test_xyscr_geometric_profiles_leave_native_confidence_block_unchanged(obb, seconds):
    """Confidence slots of one preserve HybridSORT score priors and replay."""
    model = create_motion_model("xyscr", is_obb=obb)
    fields = {}
    for field in DIAGONALS:
        size = model.dim_x - model.dim_z if "velocity" in field else model.dim_z
        values = np.linspace(0.1, 100.0, size)
        values[3] = 1.0
        fields[field] = tuple(values)
    baseline_noise = KalmanNoiseConfig(time_unit="seconds" if seconds else "frames")
    actual = model.create_filter(noise_config=replace(baseline_noise, **fields))
    expected = model.create_filter(noise_config=baseline_noise)
    for kf in (actual, expected):
        model.initialize_filter(kf, _measurement(model))
    indices = [3, model.dim_z + 3]
    for frame, score in enumerate((0.7, 0.9, None, None, 0.6, 0.85)):
        measurement = _measurement(model)
        measurement[0] += frame * 2.0
        measurement[2] *= 1.0 + 0.1 * frame
        measurement[3] = 0.0 if score is None else score
        for kf in (actual, expected):
            model.predict_filter(kf, dt=0.041 if seconds else None, observation_centric=True)
            model.update_filter(kf, None if score is None else measurement.copy(), observation_centric=True)
        np.testing.assert_allclose(actual.x.reshape(-1)[indices], expected.x.reshape(-1)[indices], atol=1e-12)
        np.testing.assert_allclose(actual.P[np.ix_(indices, indices)], expected.P[np.ix_(indices, indices)], atol=1e-12)
