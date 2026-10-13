"""Coordinate accumulation must retain scalar fitting statistics and event counts."""

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters.fitting import (
    ConfidenceNoiseMoments,
    DiagonalConfidenceNoiseMoments,
    DiagonalNoiseMoments,
    DiagonalProcessNoiseMoments,
    ProcessNoiseMoments,
    ScalarNoiseMoments,
)


@pytest.mark.parametrize("confidence_fit", [False, True])
def test_diagonal_measurement_statistics_match_scalar_events(confidence_fit: bool) -> None:
    """Invalid coordinates and zero variance cannot add calibration evidence."""
    rng = np.random.default_rng(784)
    size = 5
    diagonal = DiagonalConfidenceNoiseMoments(size) if confidence_fit else DiagonalNoiseMoments(size)
    scalar = [ConfidenceNoiseMoments() if confidence_fit else ScalarNoiseMoments() for _ in range(size)]
    for event in range(40):
        residual, variances = rng.normal(size=size), rng.uniform(0.01, 4.0, size=size)
        residual[event % size] = np.nan
        variances[(event + 1) % size] = (0.0, -1.0, np.inf, np.nan)[event % 4]
        if event == 20:
            variances[:] = 0.0
        kwargs = {"confidence": event / 39.0} if confidence_fit else {}
        diagonal.add(residual, variances, **kwargs)
        for index, expected in enumerate(scalar):
            expected.add(residual[index : index + 1], variances[index : index + 1], **kwargs)
            actual = diagonal.moments[index]
            assert (actual.events, actual.components) == (expected.events, expected.components)
            if confidence_fit:
                np.testing.assert_array_equal(actual.normal, expected.normal)
                np.testing.assert_array_equal(actual.target, expected.target)
            else:
                assert actual.total == expected.total
    for actual, expected in zip(diagonal.moments, scalar, strict=True):
        baselines = (1.0, 0.0) if confidence_fit else (1.0,)
        assert actual.estimate(*baselines) == expected.estimate(*baselines)


def test_diagonal_process_statistics_match_scalar_events_and_signed_lags() -> None:
    """Position/velocity equations keep their signs and independent event counts."""
    rng = np.random.default_rng(749)
    size = 5
    diagonal = DiagonalProcessNoiseMoments(size)
    scalar = [ProcessNoiseMoments() for _ in range(size)]
    events = lag_pairs = 0
    for event in range(40):
        first, second = rng.normal(size=(2, size))
        position, velocity = rng.uniform(0.01, 4.0, size=(2, size))
        first[event % size] = np.nan
        position[(event + 1) % size] = np.inf
        position[-1] = velocity[-1] = 0.0
        if event == 20:
            first[:] = np.nan
        before = [item.events for item in scalar]
        diagonal.add(first, position, velocity)
        for index, expected in enumerate(scalar):
            selection = slice(index, index + 1)
            expected.add(first[selection], position[selection], velocity[selection])
        events += any(item.events != count for item, count in zip(scalar, before, strict=True))
        before = [item.lag_pairs for item in scalar]
        diagonal.add_cross_covariance(first, second, -position, velocity)
        for index, expected in enumerate(scalar):
            selection = slice(index, index + 1)
            expected.add_cross_covariance(
                first[selection], second[selection], -position[selection], velocity[selection]
            )
            actual = diagonal.moments[index]
            assert (actual.events, actual.lag_pairs, actual.components) == (
                expected.events,
                expected.lag_pairs,
                expected.components,
            )
            np.testing.assert_array_equal(actual.normal, expected.normal)
            np.testing.assert_array_equal(actual.target, expected.target)
        lag_pairs += any(item.lag_pairs != count for item, count in zip(scalar, before, strict=True))
        assert (diagonal.events, diagonal.lag_pairs) == (events, lag_pairs)
    for actual, expected in zip(diagonal.moments, scalar, strict=True):
        assert actual.estimate((1.0, 1.0)) == expected.estimate((1.0, 1.0))
