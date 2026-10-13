"""Confidence variance moment fits recover independent localization floors."""

import numpy as np
import pytest
from scipy.optimize import nnls

from boxmot.trackers.common.motion.kalman_filters.fitting import ConfidenceNoiseMoments


@pytest.mark.parametrize("coefficients", [(0.2, 2.0), (3.0, 0.0), (0.0, 5.0)])
@pytest.mark.parametrize("variance", [1e-8, 1.0, 1e8])
def test_affine_confidence_fit_recovers_known_coefficients_in_any_units(coefficients, variance):
    intercept, slope = coefficients
    moments = ConfidenceNoiseMoments()
    for score in (0.0, 0.4, 0.8, 1.0):
        error = np.sqrt(variance * (intercept + slope * (1 - score) ** 2))
        for sign in (-1, 1):
            moments.add(np.array([sign * error]), np.array([variance]), confidence=score)
    scale, floor = moments.estimate(7.0, 0.3)
    assert scale["value"] == pytest.approx(intercept + slope)
    assert floor["value"] == pytest.approx(intercept / (intercept + slope), abs=1e-12)
    assert scale["variance_intercept"] == pytest.approx(intercept, abs=1e-12)
    assert scale["variance_slope"] == pytest.approx(slope, abs=1e-12)
    assert scale["events"] == floor["events"] == 8
    assert scale["status"] == floor["status"] == "fitted"


def test_streaming_confidence_fit_matches_direct_nnls_without_retaining_observations():
    rng = np.random.default_rng(614)
    scores = rng.uniform(0, 1, 200)
    reference = rng.uniform(0.1, 10, 200)
    errors = rng.normal(size=200) * np.sqrt(reference * (0.4 + 2 * (1 - scores) ** 2))
    moments = ConfidenceNoiseMoments()
    for score, variance, residual in zip(scores, reference, errors, strict=True):
        moments.add(np.array([residual]), np.array([variance]), confidence=score)
    expected = nnls(np.column_stack((np.ones(len(scores)), (1 - scores) ** 2)), errors**2 / reference)[0]
    scale, floor = moments.estimate(1, 0)
    assert scale["value"] == pytest.approx(expected.sum())
    assert floor["value"] == pytest.approx(expected[0] / expected.sum())
    assert moments.normal.shape == (2, 2)
    assert moments.target.shape == (2,)


@pytest.mark.parametrize("scores", [[], [0.7], [0.7, 0.7, 0.7], [1.0, 1.0]])
def test_unidentifiable_confidence_curve_keeps_scale_and_floor_together(scores):
    moments = ConfidenceNoiseMoments()
    for score in scores:
        moments.add(np.array([20.0]), np.ones(1), confidence=score)
    scale, floor = moments.estimate(3.0, 0.4)
    assert (scale["value"], floor["value"]) == (3.0, 0.4)
    assert scale["status"] == floor["status"] == "retained"
    assert scale["reason"] == floor["reason"]


def test_confidence_one_localization_error_is_not_dropped():
    moments = ConfidenceNoiseMoments()
    moments.add(np.array([2.0]), np.ones(1), confidence=1.0)
    moments.add(np.array([3.0]), np.ones(1), confidence=0.0)
    scale, floor = moments.estimate(1, 0)
    assert scale["events"] == 2
    assert scale["value"] == pytest.approx(9)
    assert floor["value"] == pytest.approx(4 / 9)


def test_exact_detections_keep_positive_covariance_and_do_not_claim_confidence_shape_fit():
    moments = ConfidenceNoiseMoments()
    for score in (0.0, 0.9):
        moments.add(np.zeros(1), np.ones(1), confidence=score)
    scale, floor = moments.estimate(2, 0.4)
    assert scale["value"] == 1e-6
    assert floor["value"] == 0.4
    assert floor["status"] == "fixed"


@pytest.mark.parametrize("score", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_confidence_rejected(score):
    with pytest.raises(ValueError, match="confidence"):
        ConfidenceNoiseMoments().add(np.ones(1), np.ones(1), confidence=score)
