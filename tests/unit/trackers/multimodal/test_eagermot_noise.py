"""Calibrated 3D covariance scales preserve priors and independent filter state."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.eagermot.motion import Kalman3D


def _box() -> np.ndarray:
    """Return one valid camera/world box in EagerMOT's seven-coordinate order."""
    return np.array([0.0, 2.0, 20.0, 0.1, 4.0, 2.0, 2.0])


@pytest.mark.parametrize("angular", [False, True])
@pytest.mark.parametrize(
    "field, covariance, block",
    [
        ("process_position_scale", "_process_noise", slice(0, 7)),
        ("process_velocity_scale", "_process_noise", slice(7, None)),
        ("measurement_noise_scale", "_measurement_noise", slice(None)),
        ("initial_position_scale", "covariance", slice(0, 7)),
        ("initial_velocity_scale", "covariance", slice(7, None)),
    ],
)
def test_noise_scale_changes_only_its_covariance_block(
    angular: bool, field: str, covariance: str, block: slice
) -> None:
    """Position covers xyz/yaw/dimensions, while angular velocity joins derivatives."""
    baseline = Kalman3D(_box(), is_angular=angular)
    scaled = Kalman3D(_box(), is_angular=angular, noise_config=KalmanNoiseConfig(**{field: 9.0}))
    for name in ("covariance", "_process_noise", "_measurement_noise"):
        expected = getattr(baseline, name).copy()
        if name == covariance:
            expected[block, block] *= 9.0
        np.testing.assert_allclose(getattr(scaled, name), expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(scaled.state, baseline.state)
    np.testing.assert_array_equal(scaled._transition, baseline._transition)


@pytest.mark.parametrize("angular", [False, True])
def test_explicit_default_noise_preserves_source_trajectory_exactly(angular: bool) -> None:
    """Default configuration does not introduce rounding changes over yaw wraps."""
    implicit = Kalman3D(_box(), is_angular=angular)
    explicit = Kalman3D(_box(), is_angular=angular, noise_config=KalmanNoiseConfig())
    for frame in range(20):
        box = _box()
        box[0] += 0.2 * frame
        box[3] += 0.1 * frame + (np.pi if frame % 2 else 0)
        np.testing.assert_array_equal(explicit.predict(), implicit.predict())
        if frame % 4:
            np.testing.assert_array_equal(explicit.update(box), implicit.update(box))
        np.testing.assert_array_equal(explicit.state, implicit.state)
        np.testing.assert_array_equal(explicit.covariance, implicit.covariance)


@pytest.mark.parametrize("angular", [False, True])
def test_process_scales_apply_once_on_each_prediction(angular: bool) -> None:
    """A zero prior isolates Q and subsequent predictions must not rescale it."""
    model = Kalman3D(
        _box(),
        is_angular=angular,
        noise_config=KalmanNoiseConfig(process_position_scale=4.0, process_velocity_scale=9.0),
    )
    expected_noise = np.diag([4.0] * 7 + [0.09] * (len(model.state) - 7))
    model.covariance[:] = 0
    model.predict()
    np.testing.assert_allclose(model.covariance, expected_noise, rtol=0, atol=1e-15)
    expected_next = model._transition @ expected_noise @ model._transition.T + expected_noise
    model.predict()
    np.testing.assert_allclose(model.covariance, expected_next, rtol=0, atol=1e-15)


def test_measurement_scale_controls_correction_without_leaking_between_filters() -> None:
    """A noisier detector receives less correction weight, with stable Joseph updates."""
    baseline = Kalman3D(_box())
    calibrated = Kalman3D(_box(), noise_config=KalmanNoiseConfig(measurement_noise_scale=1000.0))
    observation = _box()
    observation[0] += 4.0
    baseline.update(observation)
    calibrated.update(observation)
    assert baseline.box[0] == pytest.approx(4 * 10 / 10.01)
    assert calibrated.box[0] == pytest.approx(2.0)
    np.testing.assert_allclose(calibrated.covariance, calibrated.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(calibrated.covariance).min() > 0
    np.testing.assert_array_equal(baseline._measurement_noise, np.eye(7) * 0.01)


def test_covariance_storage_is_local_even_when_noise_configuration_is_shared() -> None:
    config = KalmanNoiseConfig(initial_position_scale=2.0, process_velocity_scale=3.0)
    first, second = (Kalman3D(_box(), noise_config=config) for _ in range(2))
    assert first.noise_config is second.noise_config is config
    for name in ("covariance", "_process_noise", "_measurement_noise"):
        before = getattr(second, name).copy()
        getattr(first, name)[:] = 0
        np.testing.assert_array_equal(getattr(second, name), before)


def test_3d_filter_rejects_seconds_without_advertising_variable_timing() -> None:
    with pytest.raises(ValueError, match="requires kf_time_unit='frames'"):
        Kalman3D(_box(), noise_config=KalmanNoiseConfig(time_unit="seconds"))
