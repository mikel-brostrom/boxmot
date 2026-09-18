"""Direct covariance fits recover known moments without tuning trials."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters.fitting import (
    MIN_COVARIANCE_SCALE,
    ProcessNoiseMoments,
    ScalarNoiseMoments,
)


def test_scalar_fit_recovers_known_covariance_scale_across_measurement_units() -> None:
    moments = ScalarNoiseMoments()
    reference = np.array([1e-8, 4.0, 1e6])
    expected_scale = 3.75
    residual = np.sqrt(expected_scale * reference)
    # The second event cancels the sample mean while preserving known moments.
    moments.add(residual, reference)
    moments.add(-residual, reference)

    estimate = moments.estimate(baseline=0.4)

    assert estimate["value"] == pytest.approx(expected_scale)
    assert estimate["raw_estimate"] == pytest.approx(expected_scale)
    assert estimate["baseline"] == 0.4
    assert estimate["events"] == 2
    assert estimate["components"] == 6
    assert estimate["status"] == "fitted"


def test_scalar_fit_keeps_detection_bias_in_zero_mean_model_uncertainty() -> None:
    moments = ScalarNoiseMoments()
    for _ in range(2):
        moments.add(np.array([2.0, 2.0]), np.ones(2))

    assert moments.estimate(baseline=1.0)["value"] == pytest.approx(4.0)


@pytest.mark.parametrize("events", [0, 1])
def test_scalar_sparse_evidence_preserves_baseline_even_with_many_coordinates(events: int) -> None:
    moments = ScalarNoiseMoments()
    for _ in range(events):
        moments.add(np.full(100, 10.0), np.ones(100))

    estimate = moments.estimate(baseline=2.5)

    assert estimate["value"] == 2.5
    assert estimate["status"] == "retained"
    assert estimate["events"] == events
    assert "insufficient" in estimate["reason"]


def test_scalar_ignores_nonfinite_errors_and_nonpositive_reference_variances() -> None:
    moments = ScalarNoiseMoments()
    moments.add(np.array([2.0, np.nan, 3.0, 4.0, 5.0]), np.array([1.0, 1.0, 0.0, -1.0, np.inf]))
    moments.add(np.array([np.inf]), np.ones(1))
    moments.add(np.array([-6.0]), np.array([9.0]))

    estimate = moments.estimate(baseline=1.0)

    assert estimate["events"] == 2
    assert estimate["components"] == 2
    assert estimate["value"] == pytest.approx(4.0)


def test_scalar_zero_residuals_use_positive_numerical_floor() -> None:
    moments = ScalarNoiseMoments()
    for _ in range(2):
        moments.add(np.zeros(3), np.ones(3))

    estimate = moments.estimate(baseline=5.0)

    assert estimate["raw_estimate"] == 0.0
    assert estimate["value"] == MIN_COVARIANCE_SCALE
    assert estimate["status"] == "fitted"


def test_scalar_unrepresentable_moments_raise_instead_of_saving_infinite_noise() -> None:
    moments = ScalarNoiseMoments()
    with np.errstate(over="ignore"):
        for _ in range(2):
            moments.add(np.array([1e200]), np.ones(1))

    with pytest.raises(ValueError, match="unrepresentable"):
        moments.estimate(baseline=1.0)


@pytest.mark.parametrize("state_unit_scale", [1e-5, 1.0, 1e5])
def test_process_fit_recovers_two_noise_rates_across_measured_intervals(state_unit_scale: float) -> None:
    moments = ProcessNoiseMoments()
    expected_position, expected_velocity = 2.25, 0.45
    # Exact continuous CV diagonal bases, with independent position diffusion
    # and white velocity diffusion. A coordinate unit change scales both the
    # residual and covariance consistently and leaves fitted rates unchanged.
    for dt in (1.0 / 30.0, 0.1, 0.37, 0.05):
        position = np.array([2.0 * dt, 0.0]) * state_unit_scale**2
        velocity = np.array([0.8 * dt**3 / 3.0, 0.8 * dt]) * state_unit_scale**2
        residual = np.sqrt(expected_position * position + expected_velocity * velocity)
        moments.add(residual, position, velocity)
        moments.add(-residual, position, velocity)

    position_fit, velocity_fit = moments.estimate((1.0, 1.0))

    assert position_fit["value"] == pytest.approx(expected_position)
    assert velocity_fit["value"] == pytest.approx(expected_velocity)
    assert position_fit["events"] == 8
    assert velocity_fit["components"] == 16
    assert position_fit["status"] == velocity_fit["status"] == "fitted"


@pytest.mark.parametrize("inactive_component", [0, 1])
def test_process_fit_solves_nonnegative_boundary_when_unconstrained_solution_is_negative(
    inactive_component: int,
) -> None:
    moments = ProcessNoiseMoments()
    # Normalized observations require a + b = 0 and a = 1. Unconstrained
    # least squares gives (a,b)=(1,-1). The nonnegative optimum is (0.8,0),
    # not merely clipping the unconstrained solution to (1,0).
    position, velocity = np.array([1.0, 1.0]), np.array([1.0, 0.0])
    if inactive_component == 0:
        position, velocity = velocity, position
    for sign in (1.0, -1.0):
        moments.add(np.array([0.0, sign]), position, velocity)

    estimates = moments.estimate((1.0, 1.0))

    assert estimates[inactive_component]["raw_estimate"] == 0.0
    assert estimates[inactive_component]["value"] == MIN_COVARIANCE_SCALE
    assert estimates[1 - inactive_component]["value"] == pytest.approx(0.8)


@pytest.mark.parametrize("events", [0, 1])
def test_process_fit_preserves_both_baselines_with_too_few_transitions(events: int) -> None:
    moments = ProcessNoiseMoments()
    for _ in range(events):
        moments.add(np.ones(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    position, velocity = moments.estimate((0.7, 3.0))

    assert (position["value"], velocity["value"]) == (0.7, 3.0)
    assert position["status"] == velocity["status"] == "retained"
    assert "insufficient" in position["reason"]


def test_process_fit_preserves_both_baselines_when_rates_are_not_identifiable() -> None:
    moments = ProcessNoiseMoments()
    for residual in (np.array([1.0, 2.0]), np.array([3.0, 4.0])):
        moments.add(residual, np.array([1.0, 2.0]), np.array([3.0, 6.0]))

    position, velocity = moments.estimate((0.7, 3.0))

    assert (position["value"], velocity["value"]) == (0.7, 3.0)
    assert position["status"] == velocity["status"] == "retained"
    assert "not identifiable" in position["reason"]


def test_process_zero_residuals_keep_both_covariance_scales_strictly_positive() -> None:
    moments = ProcessNoiseMoments()
    for _ in range(2):
        moments.add(np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    position, velocity = moments.estimate((0.7, 3.0))

    assert position["raw_estimate"] == velocity["raw_estimate"] == 0.0
    assert position["value"] == velocity["value"] == MIN_COVARIANCE_SCALE


def test_process_ignores_invalid_or_unmodeled_components_without_counting_empty_events() -> None:
    moments = ProcessNoiseMoments()
    for _ in range(2):
        moments.add(
            np.array([2.0, 3.0, np.nan, 5.0, 6.0]),
            np.array([1.0, 0.0, 1.0, np.inf, 0.0]),
            np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
        )
    moments.add(np.zeros(1), np.zeros(1), np.zeros(1))

    position, velocity = moments.estimate((1.0, 1.0))

    assert (position["value"], velocity["value"]) == pytest.approx((4.0, 9.0))
    assert position["events"] == 2
    assert position["components"] == 4


def test_process_unrepresentable_moments_raise_instead_of_saving_infinite_noise() -> None:
    moments = ProcessNoiseMoments()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(2):
            moments.add(np.array([1e200, 1e200]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="unrepresentable"):
        moments.estimate((1.0, 1.0))


@pytest.mark.parametrize("intervals", [(0.1, 0.1, 0.1), (1.0 / 30.0, 0.1, 0.37)])
@pytest.mark.parametrize("expected_scales", [(2.25, 0.45), (0.002, 9.0)])
@pytest.mark.parametrize("state_unit_scale", [1e-3, 1.0, 1e3])
def test_variance_and_signed_lag_moments_recover_continuous_cv_rates(
    intervals: tuple[float, float, float], expected_scales: tuple[float, float], state_unit_scale: float
) -> None:
    """Uniform capture times identify both rates when lag covariance is used."""
    a, b, c = intervals
    position_rate, velocity_rate = 2.0 * state_unit_scale**2, 0.8 * state_unit_scale**2
    position_first = position_rate * (b + b**2 / a)
    velocity_first = velocity_rate * b**2 * (a + b) / 3.0
    position_second = position_rate * (c + c**2 / b)
    velocity_second = velocity_rate * c**2 * (b + c) / 3.0
    position_lag = -c * position_rate
    velocity_lag = c * b**2 * velocity_rate / 6.0
    position_scale, velocity_scale = expected_scales
    variance_first = position_scale * position_first + velocity_scale * velocity_first
    variance_second = position_scale * position_second + velocity_scale * velocity_second
    lag = position_scale * position_lag + velocity_scale * velocity_lag
    covariance = np.array([[variance_first, lag], [lag, variance_second]])
    # Four antithetic quadrature points have exactly zero mean and identity
    # covariance. Transform them to prescribed CV residual moments so this
    # verifies covariance identification, without Monte Carlo tolerances.
    quadrature = np.sqrt(2.0) * np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    residuals = quadrature @ np.linalg.cholesky(covariance).T
    moments = ProcessNoiseMoments()
    for first, second in residuals:
        moments.add(np.array([first]), np.array([position_first]), np.array([velocity_first]))
        moments.add(np.array([second]), np.array([position_second]), np.array([velocity_second]))
        moments.add_cross_covariance(
            np.array([first]), np.array([second]), np.array([position_lag]), np.array([velocity_lag])
        )

    position, velocity = moments.estimate((1.0, 1.0))

    assert position["value"] == pytest.approx(position_scale, rel=1e-8)
    assert velocity["value"] == pytest.approx(velocity_scale, rel=1e-8)
    assert position["events"] == 8
    assert position["lag_pairs"] == 4
    assert position["components"] == 12
    assert position["status"] == velocity["status"] == "fitted"


def test_discrete_uniform_cv_uses_negative_lag_covariance_to_identify_position_diffusion() -> None:
    moments = ProcessNoiseMoments()
    expected_position, expected_velocity = 0.6, 2.0
    variance = 2.0 * expected_position + expected_velocity
    covariance = np.array([[variance, -expected_position], [-expected_position, variance]])
    quadrature = np.sqrt(2.0) * np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    for first, second in quadrature @ np.linalg.cholesky(covariance).T:
        moments.add(np.array([first, second]), np.full(2, 2.0), np.ones(2))
        moments.add_cross_covariance(np.array([first]), np.array([second]), np.array([-1.0]), np.array([0.0]))

    position, velocity = moments.estimate((1.0, 1.0))

    assert position["value"] == pytest.approx(expected_position)
    assert velocity["value"] == pytest.approx(expected_velocity)
    assert position["events"] == 4
    assert position["lag_pairs"] == 4


def test_cross_covariance_skips_invalid_pairs_and_does_not_invent_transition_events() -> None:
    moments = ProcessNoiseMoments()
    moments.add_cross_covariance(
        np.array([1.0, np.nan, 3.0, 4.0]),
        np.array([-2.0, 1.0, np.inf, 5.0]),
        np.array([-1.0, -1.0, -1.0, 0.0]),
        np.zeros(4),
    )
    moments.add_cross_covariance(np.array([np.nan]), np.ones(1), -np.ones(1), np.zeros(1))

    position, velocity = moments.estimate((0.7, 3.0))

    assert position["lag_pairs"] == 1
    assert position["events"] == 0
    assert position["components"] == 1
    assert position["value"] == 0.7
    assert velocity["value"] == 3.0
    assert position["status"] == velocity["status"] == "retained"


def test_signed_lag_moments_do_not_allow_negative_covariance_scales() -> None:
    moments = ProcessNoiseMoments()
    for _ in range(2):
        moments.add(np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        # Negative position basis with a positive observed lag pushes the
        # unconstrained position scale negative, but the covariance stays PSD.
        moments.add_cross_covariance(np.ones(1), np.ones(1), -np.ones(1), np.zeros(1))

    position, velocity = moments.estimate((1.0, 1.0))

    assert position["raw_estimate"] == 0.0
    assert position["value"] == MIN_COVARIANCE_SCALE
    assert velocity["value"] == pytest.approx(1.0)
