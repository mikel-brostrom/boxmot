"""HybridSORT recovery retains its live prediction when bounded history expires."""

from copy import deepcopy

import numpy as np
import pytest

from boxmot.trackers.common.motion.kalman_filters.xyscr import KalmanFilterXYSCR


def _measurement(step: float, offset: float = 0.0) -> np.ndarray:
    """Describe linear position, confidence, width and height changes."""
    width, height = 10.0 + 2.0 * step, 20.0 + 3.0 * step
    return np.array([offset + step, 2.0 * step, width * height, 0.9 - 0.03 * step, width / height])


def _filter(*, max_obs: int = 50, offset: float = 0.0) -> KalmanFilterXYSCR:
    """Create a filter with an observed trajectory before a possible gap."""
    result = KalmanFilterXYSCR(max_obs=max_obs)
    result.x[:5, 0] = _measurement(0.0, offset)
    result.update(_measurement(0.0, offset))
    result.predict()
    result.update(_measurement(1.0, offset))
    return result


@pytest.mark.parametrize("max_obs", [1, 2, 3])
def test_expired_history_corrects_live_prediction_without_rewinding(max_obs: int) -> None:
    """A gap longer than retained history falls back to ordinary correction."""
    actual = _filter(max_obs=max_obs)
    for _ in range(5):
        actual.predict()
        actual.update(None)
    actual.predict()
    expected = deepcopy(actual)
    expected.observed = True
    expected.attr_saved = None
    endpoint = _measurement(7.0)
    expected.update(endpoint)
    actual.update(endpoint)
    np.testing.assert_allclose(actual.x, expected.x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.P, expected.P, rtol=1e-12, atol=1e-12)
    assert actual.attr_saved is None
    assert len(actual.history_obs) == max_obs
    np.testing.assert_allclose(actual.history_obs[-1][:, 0], endpoint)


def test_elapsed_recovery_keeps_actual_prediction_intervals() -> None:
    """Measured-time interpolation still corrects the current endpoint once."""
    actual, expected = _filter(), _filter()
    # Start measured-time bookkeeping at an observed state.
    actual.predict(dt=0.5)
    expected.predict(dt=0.5)
    actual.update(_measurement(1.5))
    expected.update(_measurement(1.5))
    elapsed = 1.5
    for index, interval in enumerate((0.25, 0.75, 0.5)):
        elapsed += interval
        actual.predict(dt=interval)
        expected.predict(dt=interval)
        endpoint = _measurement(elapsed)
        actual.update(endpoint if index == 2 else None)
        expected.update(endpoint)
    np.testing.assert_allclose(actual.x, expected.x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.P, expected.P, rtol=1e-12, atol=1e-12)
