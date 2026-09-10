"""Noise scales alter only their own covariance and remain instance-local."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from boxmot.trackers.common.factory import create_tracker
from boxmot.trackers.common.motion.kalman_filters.noise import (
    KALMAN_NOISE_OPTIONS,
    KalmanNoiseConfig,
    normalize_kalman_options,
)
from boxmot.trackers.common.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.trackers.common.motion.kalman_filters.xyhr import KalmanFilterXYHR
from boxmot.trackers.common.motion.kalman_filters.xyscr import KalmanFilterXYSCR
from boxmot.trackers.common.motion.kalman_filters.xysr import KalmanFilterXYSR
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.native import load_native_tracker_config
from boxmot.trackers.common.specs import TrackerSpec

MODES = ("xyah", "xyah_obb", "xywh", "xywh_obb", "xyhr", "xyhr_obb", "xysr", "xysr_obb", "xyscr")


def _filter(mode: str, config: KalmanNoiseConfig | None = None):
    """Build valid geometry in every supported Kalman representation."""
    name = mode.split("_")[0]
    ndim = 5 if mode.endswith("_obb") else 4
    if name in ("xyah", "xywh"):
        cls = KalmanFilterXYAH if name == "xyah" else KalmanFilterXYWH
        kf = cls(ndim=ndim, noise_config=config)
        measurement = [20.0, 30.0, 2.0 if name == "xyah" else 40.0, 20.0]
    elif name == "xyhr":
        kf = KalmanFilterXYHR(dim_z=ndim, noise_config=config)
        measurement = [20.0, 30.0, 20.0, 2.0]
    elif name == "xysr":
        kf = KalmanFilterXYSR(dim_x=2 * ndim - 1, dim_z=ndim, noise_config=config)
        measurement = [20.0, 30.0, 800.0, 2.0]
    else:
        kf = KalmanFilterXYSCR(noise_config=config)
        measurement = [20.0, 30.0, 800.0, 0.8, 2.0]
    if mode.endswith("_obb"):
        measurement.append(0.2)
    kf.x, kf.P = kf.initiate(np.array(measurement))
    return kf


def _predict(kf, dt: float | None):
    if isinstance(kf, (KalmanFilterXYAH, KalmanFilterXYWH)):
        return kf.predict(kf.x, kf.P, dt=dt)
    kf.predict(dt=dt)
    return kf.x, kf.P


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dt", [None, 0.2])
def test_process_covariance_scale_applies_once_before_time_integration(mode, dt):
    baseline = _filter(mode)
    scaled = _filter(mode, KalmanNoiseConfig(process_position_scale=7.0, process_velocity_scale=7.0))
    baseline.P[:] = 0.0
    scaled.P[:] = 0.0
    base_mean, base_covariance = _predict(baseline, dt)
    scaled_mean, scaled_covariance = _predict(scaled, dt)
    np.testing.assert_array_equal(base_mean, scaled_mean)
    np.testing.assert_allclose(scaled_covariance, base_covariance * 7.0, atol=1e-12)
    assert baseline.noise_config.is_default


@pytest.mark.parametrize("mode", MODES)
def test_measurement_covariance_scale_leaves_initial_covariance_unchanged(mode):
    baseline = _filter(mode)
    scaled = _filter(mode, KalmanNoiseConfig(measurement_noise_scale=5.0))
    np.testing.assert_array_equal(scaled.P, baseline.P)
    baseline.P[:] = 0.0
    scaled.P[:] = 0.0
    if isinstance(baseline, (KalmanFilterXYSR, KalmanFilterXYSCR)):
        _, base_covariance = baseline.project_state()
        _, scaled_covariance = scaled.project_state()
    else:
        _, base_covariance = baseline.project(baseline.x, baseline.P)
        _, scaled_covariance = scaled.project(scaled.x, scaled.P)
    np.testing.assert_allclose(scaled_covariance, base_covariance * 5.0)


@pytest.mark.parametrize("mode", MODES)
def test_initial_velocity_scale_preserves_initial_position_covariance(mode):
    baseline = _filter(mode)
    scaled = _filter(mode, KalmanNoiseConfig(initial_velocity_scale=4.0))
    np.testing.assert_array_equal(
        scaled.P[: baseline.dim_z, : baseline.dim_z], baseline.P[: baseline.dim_z, : baseline.dim_z]
    )
    np.testing.assert_allclose(
        scaled.P[baseline.dim_z :, baseline.dim_z :], 4.0 * baseline.P[baseline.dim_z :, baseline.dim_z :]
    )
    np.testing.assert_array_equal(scaled.x, baseline.x)


def test_initial_velocity_covariance_scales_cross_terms_and_preserves_psd():
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    result = KalmanNoiseConfig(initial_velocity_scale=9.0).initial_covariance(covariance, 1)
    np.testing.assert_array_equal(result, [[4.0, 3.0], [3.0, 18.0]])
    assert np.linalg.eigvalsh(result).min() > 0.0
    np.testing.assert_array_equal(covariance, [[4.0, 1.0], [1.0, 2.0]])


@pytest.mark.parametrize("mode", ["xysr", "xysr_obb", "xyscr"])
def test_stateful_measurement_scale_is_consistent_in_joseph_covariance(mode):
    actual = _filter(mode, KalmanNoiseConfig(measurement_noise_scale=3.0))
    reference = _filter(mode)
    reference.R *= 3.0
    measurement = actual.x[: actual.dim_z].copy()
    measurement[0] += 2.0
    actual.update(measurement.copy())
    reference.update(measurement.copy())
    np.testing.assert_allclose(actual.x, reference.x)
    np.testing.assert_allclose(actual.P, reference.P)
    assert actual.md_for_measurement(measurement) == pytest.approx(reference.md_for_measurement(measurement))


@pytest.mark.parametrize("field", [*[option.removeprefix("kf_") for option in KALMAN_NOISE_OPTIONS], "reference_dt_s"])
@pytest.mark.parametrize("value", [True, False, 0.0, -1.0, np.nan, np.inf, "2", [2]])
def test_noise_config_rejects_malformed_scalars(field, value):
    with pytest.raises(ValueError, match=f"kf_{field}"):
        KalmanNoiseConfig(**{field: value})


def test_noise_config_is_immutable():
    config = KalmanNoiseConfig(process_position_scale=2.0)
    with pytest.raises(FrozenInstanceError):
        config.process_position_scale = 3.0


@pytest.mark.parametrize("name, backend", [("bytetrack", "cpp"), ("sfsort", "python"), ("maf_hda", "python")])
def test_factory_rejects_unsupported_noise_scaling_before_loading_models(name, backend):
    with pytest.raises(ValueError, match="Python Kalman"):
        create_tracker(TrackerSpec(name=name, backend=backend, options=(("kf_process_position_scale", 2.0),)))


def test_native_adapter_accepts_default_scales_and_rejects_custom_scales():
    config = load_native_tracker_config("bytetrack", {"kf_process_position_scale": 1.0, "kf_time_unit": "frames"})
    assert not any(option.startswith("kf_") for option in config)
    with pytest.raises(ValueError, match="Python Kalman"):
        load_native_tracker_config("bytetrack", {"kf_measurement_noise_scale": 0.5})


@pytest.mark.parametrize("dt", [0.013, 0.1, 1.0])
@pytest.mark.parametrize("time_unit", ["frames", "seconds"])
def test_adaptive_xyhr_process_scale_is_not_applied_twice(dt, time_unit):
    """A calibrated adaptive rate is stored before the runtime multiplier."""
    config = KalmanNoiseConfig(
        process_position_scale=4.0, process_velocity_scale=9.0, time_unit=time_unit, reference_dt_s=0.04
    )
    kf = KalmanFilterXYHR(noise_config=config, adaptive_kf=True)
    policy = kf.cov_update_policy
    baseline = policy.get_q().copy()
    transition, scaled_noise = kf._elapsed_motion(baseline, dt)
    for _ in range(15):
        policy.observe_innovation(
            np.zeros(4),
            np.zeros((8, 4)),
            np.eye(8),
            transition,
            dt=dt,
            process_noise=scaled_noise,
            innovation_covariance=np.eye(4),
        )
    np.testing.assert_allclose(policy.get_q(dt=dt), baseline)
    _, learned_noise = kf._elapsed_motion(policy.get_q(dt=dt), dt)
    np.testing.assert_allclose(learned_noise, scaled_noise)


def _integrated_noise(generator, density, dt):
    """Independently integrate the constant-velocity density for a test oracle."""
    return (
        dt * density
        + dt**2 / 2.0 * (generator @ density + density @ generator.T)
        + dt**3 / 3.0 * generator @ density @ generator.T
    )


@pytest.mark.parametrize("mode", MODES)
def test_seconds_initial_covariance_converts_before_first_prediction(mode):
    baseline = _filter(mode)
    config = KalmanNoiseConfig(
        time_unit="seconds", reference_dt_s=0.04, initial_position_scale=2.0, initial_velocity_scale=3.0
    )
    converted = _filter(mode, config)
    factors = np.full(baseline.dim_x, np.sqrt(2.0))
    factors[baseline.dim_z :] = np.sqrt(3.0) / 0.04
    expected = baseline.P * factors[:, None] * factors[None, :]
    np.testing.assert_allclose(converted.P, expected)
    np.testing.assert_array_equal(converted.x, baseline.x)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dt", [0.013, 0.077])
def test_seconds_process_density_uses_fixed_reference_and_separate_scales(mode, dt):
    baseline = _filter(mode)
    converted = _filter(
        mode,
        KalmanNoiseConfig(
            time_unit="seconds", reference_dt_s=0.04, process_position_scale=2.0, process_velocity_scale=7.0
        ),
    )
    baseline.P[:] = 0.0
    converted.P[:] = 0.0
    _, frame_noise = _predict(baseline, None)
    factors = np.full(baseline.dim_x, np.sqrt(2.0))
    factors[baseline.dim_z :] = np.sqrt(7.0) / 0.04
    density = frame_noise * factors[:, None] * factors[None, :] / 0.04
    generator = baseline._motion_mat - np.eye(baseline.dim_x)
    expected = _integrated_noise(generator, density, dt)
    _, covariance = _predict(converted, dt)
    np.testing.assert_allclose(covariance, expected, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    assert converted.noise_config.reference_dt_s == 0.04


@pytest.mark.parametrize("mode", MODES)
def test_seconds_batch_prediction_converts_each_process_covariance(mode):
    baseline = _filter(mode)
    converted = _filter(mode, KalmanNoiseConfig(time_unit="seconds", reference_dt_s=0.05))
    means = np.repeat(baseline.x.reshape(1, -1), 2, axis=0)
    covariances = np.zeros((2, baseline.dim_x, baseline.dim_x))
    _, frame_noise = baseline.multi_predict(means.copy(), covariances.copy())
    factors = np.ones(baseline.dim_x)
    factors[baseline.dim_z :] /= 0.05
    density = frame_noise * factors[:, None] * factors[None, :] / 0.05
    expected = _integrated_noise(baseline._motion_mat - np.eye(baseline.dim_x), density, 0.083)
    _, actual = converted.multi_predict(means.copy(), covariances.copy(), dt=0.083)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_seconds_prediction_requires_measured_interval_without_mutation(mode):
    kf = _filter(mode, KalmanNoiseConfig(time_unit="seconds"))
    mean, covariance = kf.x.copy(), kf.P.copy()
    with pytest.raises(ValueError, match="explicit measured dt"):
        _predict(kf, None)
    np.testing.assert_array_equal(kf.x, mean)
    np.testing.assert_array_equal(kf.P, covariance)
    empty_mean, empty_covariance = kf.multi_predict(np.empty((0, kf.dim_x)), np.empty((0, kf.dim_x, kf.dim_x)))
    assert empty_mean.shape == (0, kf.dim_x)
    assert empty_covariance.shape == (0, kf.dim_x, kf.dim_x)


@pytest.mark.parametrize("unit", ["Frames", "second", None, False, []])
def test_noise_config_rejects_noncanonical_time_units(unit):
    with pytest.raises(ValueError, match="kf_time_unit"):
        KalmanNoiseConfig(time_unit=unit)


@pytest.mark.parametrize("variable_dt,unit", [(False, "seconds"), (True, "frames")])
def test_persisted_units_cannot_silently_change_timing_mode(variable_dt, unit):
    with pytest.raises(ValueError, match="conflicts with variable_dt"):
        normalize_kalman_options({"kf_time_unit": unit}, variable_dt=variable_dt)


@pytest.mark.parametrize("variable_dt,unit", [(False, "frames"), (True, "seconds")])
def test_unspecified_units_derive_from_selected_tracker_mode(variable_dt, unit):
    config = normalize_kalman_options({}, variable_dt=variable_dt)
    assert config.time_unit == unit


def test_removed_process_scale_is_rejected_without_an_alias():
    with pytest.raises(TypeError, match="kf_process_noise_scale"):
        normalize_kalman_options({"kf_process_noise_scale": 1.0}, variable_dt=False)


def test_process_density_congruence_preserves_correlations_and_psd():
    config = KalmanNoiseConfig(
        time_unit="seconds", reference_dt_s=0.1, process_position_scale=4.0, process_velocity_scale=9.0
    )
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    density = config.process_covariance(covariance, 1, continuous=True)
    np.testing.assert_allclose(density, [[160.0, 600.0], [600.0, 18000.0]])
    assert np.linalg.eigvalsh(density).min() > 0.0


def test_reference_step_continuous_noise_does_not_claim_discrete_equivalence():
    config = KalmanNoiseConfig(time_unit="seconds", reference_dt_s=0.1)
    frame_noise = np.diag([1.0, 0.03])
    density = config.process_covariance(frame_noise, 1, continuous=True)
    result = _integrated_noise(np.array([[0.0, 1.0], [0.0, 0.0]]), density, 0.1)
    np.testing.assert_allclose(result, [[1.01, 0.15], [0.15, 3.0]])
