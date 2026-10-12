"""Numerical contracts for optional confidence floors; no quality benchmark data."""

import pickle
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import yaml

from boxmot import create_tracker
from boxmot.trackers.common.config import flatten_tracker_options, load_tracker_config, nest_tracker_options
from boxmot.trackers.common.motion.confidence import ConfidencePrediction
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.noise import (
    KALMAN_DIAGONAL_OPTIONS,
    KALMAN_VECTOR_OPTIONS,
    KalmanNoiseConfig,
)
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model, supported_motion_model_kinds

FLOOR = "kalman.noise.measurement_confidence_floor"


def _measurement(model):
    """Use valid geometry with confidence in the model's native slot."""
    box = [20.0, 30.0, 40.0, 20.0, 0.2, 0.8] if model.is_obb else [0.0, 20.0, 40.0, 40.0, 0.8]
    return model.to_measurement(np.array(box), column=False)


@pytest.mark.parametrize(
    "kind,obb",
    [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)],
)
@pytest.mark.parametrize("score", [0.0, 0.5, 0.95, 1.0])
def test_all_models_project_and_update_with_independent_confidence_floors(kind, obb, score):
    """Compare the formula to a separate filter with explicit effective R."""
    model = create_motion_model(kind, is_obb=obb)
    floors = np.linspace(0.1, 0.8, model.dim_z)
    diagonals = np.linspace(0.5, 3.0, model.dim_z)
    config = KalmanNoiseConfig(
        measurement_noise_scale=2.0,
        measurement_noise_diagonal=tuple(diagonals),
        measurement_confidence_floor=tuple(floors),
    )
    weights = floors + (1.0 - floors) * (1.0 - score) ** 2
    reference_config = replace(
        config, measurement_confidence_floor=(), measurement_noise_diagonal=tuple(diagonals * weights)
    )
    actual = model.create_filter(noise_config=config)
    expected = model.create_filter(noise_config=reference_config)
    measurement = _measurement(model)
    mean, covariance = model.initiate(actual, measurement)
    model.initiate(expected, measurement)
    observation = measurement.copy()
    observation[0] += 2.0
    actual_projection = model.project(actual, mean, covariance, score)
    expected_projection = model.project(expected, mean, covariance, 0.0)
    for value, reference in zip(actual_projection, expected_projection):
        np.testing.assert_allclose(value, reference, atol=1e-11)
    observed = model.update(actual, mean, covariance, observation, score)
    reference = model.update(expected, mean, covariance, observation, 0.0)
    for value, target in zip(observed, reference):
        np.testing.assert_allclose(value, target, rtol=1e-11, atol=1e-11)
    assert np.linalg.eigvalsh(observed[1]).min() > 0.0


@pytest.mark.parametrize(
    "kind,obb",
    [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)],
)
@pytest.mark.parametrize("count", [0, 1, 3])
def test_all_models_batch_matches_scalar_with_individual_confidences(kind, obb, count):
    model = create_motion_model(kind, is_obb=obb)
    config = KalmanNoiseConfig(
        measurement_noise_diagonal=tuple(np.linspace(0.2, 4.0, model.dim_z)),
        measurement_confidence_floor=tuple(np.linspace(0.0, 1.0, model.dim_z)),
    )
    kf = model.create_filter(noise_config=config)
    mean, covariance = model.initiate(kf, _measurement(model))
    means = np.repeat(mean[None], count, axis=0)
    covariances = np.repeat(covariance[None], count, axis=0)
    observations = np.repeat(_measurement(model)[None], count, axis=0)
    observations[:, 0] += np.arange(count) + 1.0
    scores = np.linspace(0.1, 0.98, count)
    actual_mean, actual_covariance = model.multi_update(kf, means, covariances, observations, scores)
    assert actual_mean.shape == means.shape
    assert actual_covariance.shape == covariances.shape
    for index in range(count):
        oracle = model.create_filter(noise_config=config)
        model.initiate(oracle, observations[index])
        reference_mean, reference_covariance = model.update(
            oracle, means[index], covariances[index], observations[index], scores[index]
        )
        np.testing.assert_allclose(actual_mean[index], reference_mean, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(actual_covariance[index], reference_covariance, rtol=1e-11, atol=1e-11)
    if kind in (MotionModelKind.XYAH, MotionModelKind.XYWH):
        projected_mean, projected_covariance = kf.multi_project(means, covariances, scores)
        for index in range(count):
            expected_mean, expected_covariance = kf.project(means[index], covariances[index], scores[index])
            np.testing.assert_allclose(projected_mean[index], expected_mean)
            np.testing.assert_allclose(projected_covariance[index], expected_covariance)


@pytest.mark.parametrize("score", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("batch", [False, True])
def test_confidence_congruence_preserves_full_covariance_correlations_and_psd(score, batch):
    covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
    noise = KalmanNoiseConfig(
        measurement_noise_scale=2.0, measurement_noise_diagonal=(3.0, 5.0), measurement_confidence_floor=(0.2, 0.8)
    )
    confidence = np.array([score, score / 2.0]) if batch else score
    fractions = np.array([0.2, 0.8]) + np.array([0.8, 0.2]) * np.square(1.0 - np.asarray(confidence)[..., None])
    factors = np.sqrt(2.0 * np.array([3.0, 5.0]) * fractions)
    expected = covariance * factors[..., :, None] * factors[..., None, :]
    observed = noise.measurement_covariance(covariance, confidence=confidence)
    np.testing.assert_allclose(observed, expected)
    assert np.linalg.eigvalsh(observed).min() > 0.0
    weights = np.array([2.0, 4.0])[:, None, None]
    if batch:
        np.testing.assert_allclose(
            noise.measurement_covariance(weights * covariance, confidence=confidence), weights * expected
        )


@pytest.mark.parametrize(
    "kind,obb",
    [
        (kind, obb)
        for obb in (False, True)
        for kind in supported_motion_model_kinds(is_obb=obb)
    ],
)
def test_floor_omission_and_explicit_zero_preserve_original_confidence_behavior(kind, obb):
    model = create_motion_model(kind, is_obb=obb)
    original = model.create_filter()
    zeros = model.create_filter(noise_config=KalmanNoiseConfig(measurement_confidence_floor=(0.0,) * model.dim_z))
    mean, covariance = model.initiate(original, _measurement(model))
    model.initiate(zeros, _measurement(model))
    for score in (0.0, 0.6, 1.0):
        actual = model.project(zeros, mean, covariance, score)
        expected = model.project(original, mean, covariance, score)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
    assert "measurement_confidence_floor" not in KalmanNoiseConfig().to_dict()
    assert KALMAN_VECTOR_OPTIONS == (*KALMAN_DIAGONAL_OPTIONS, FLOOR)


@pytest.mark.parametrize("bad", [True, "1", np.array([0.1]), [False], [-0.01], [1.01], [np.nan], [np.inf], ["0.5"]])
def test_confidence_floor_rejects_invalid_types_and_values(bad):
    with pytest.raises(ValueError, match="measurement_confidence_floor"):
        KalmanNoiseConfig(measurement_confidence_floor=bad)


@pytest.mark.parametrize(
    "kind,obb",
    [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)],
)
@pytest.mark.parametrize("by_class", [False, True])
def test_confidence_floor_validates_geometry_layout_including_classes(kind, obb, by_class):
    noise = KalmanNoiseConfig(measurement_confidence_floor=(0.5,))
    if by_class:
        noise = KalmanNoiseConfig(by_class={2: noise})
    with pytest.raises(ValueError, match="measurement_confidence_floor.*expected"):
        create_motion_model(kind, is_obb=obb).create_filter(noise_config=noise)


def test_floor_round_trips_immutable_config_yaml_and_class_profiles(tmp_path):
    source = [0.0, 0.1, 0.5, 1.0]
    child = KalmanNoiseConfig(measurement_confidence_floor=(0.1, 0.2, 0.3, 0.4))
    noise = KalmanNoiseConfig(measurement_confidence_floor=source, by_class={2: child})
    source[0] = 0.9
    assert noise.measurement_confidence_floor == (0.0, 0.1, 0.5, 1.0)
    with pytest.raises(FrozenInstanceError):
        noise.measurement_confidence_floor = ()
    assert pickle.loads(pickle.dumps(noise)) == noise
    dotted = {FLOOR: [0.0, 0.1, 0.5, 1.0], "kalman.noise.by_class.2.measurement_confidence_floor": [0.1, 0.2, 0.3, 0.4]}
    nested = {"kalman": {"noise": noise.to_dict()}}
    assert (
        normalize_kalman_config({**dotted, "kalman.variable_dt": True}).noise
        == normalize_kalman_config({**nested, "kalman.variable_dt": True}).noise
    )
    flattened = flatten_tracker_options(dotted)
    assert flatten_tracker_options(nest_tracker_options(flattened)) == dotted
    path = tmp_path / "noise.yaml"
    path.write_text(yaml.safe_dump(nest_tracker_options(flattened)))
    tracker = create_tracker("bytetrack", per_class=True, options=load_tracker_config("bytetrack", path))
    assert tracker.kalman_noise_config == noise.resolve(variable_dt=False)
    assert tracker.kalman_noise_config.for_class(2).measurement_confidence_floor == child.measurement_confidence_floor


@pytest.mark.parametrize("obb", [False, True])
def test_auxiliary_confidence_filter_discards_geometric_confidence_floor(obb):
    model = create_motion_model("xywh", is_obb=obb)
    noisy = KalmanNoiseConfig(measurement_noise_scale=3.0, measurement_confidence_floor=(0.7,) * model.dim_z)
    actual = ConfidencePrediction(0.8, noise_config=noisy)
    expected = ConfidencePrediction(0.8, noise_config=replace(noisy, measurement_confidence_floor=()))
    assert actual._filter.noise_config.measurement_confidence_floor == ()
    for score in (0.9, 0.7, None, 0.95):
        actual.predict()
        expected.predict()
        actual.update(score)
        expected.update(score)
        np.testing.assert_array_equal(actual._filter.x, expected._filter.x)
        np.testing.assert_array_equal(actual._filter.P, expected._filter.P)
