"""Batched KF selection forecasts preserve the original scalar replay scores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from boxmot.engine.calibration import kalman_replay
from boxmot.engine.calibration.kalman_data import CalibrationTrack
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_DIAGONAL_OPTIONS, KALMAN_NOISE_OPTIONS
from boxmot.trackers.common.motion.models import MotionModelAdapter, MotionModelKind, create_motion_model


def _track(geometry: str, *, count: int = 22, interrupted: bool = True) -> CalibrationTrack:
    """Make reproducible noisy observations, including gaps and missed detections."""
    index = np.arange(count, dtype=float)
    centers = np.column_stack((80 + 0.8 * index + 0.008 * index**2, 95 + 0.5 * index))
    sizes = np.column_stack((27 + 0.2 * index + np.sin(index / 4), 65 + 0.15 * index + np.cos(index / 5)))
    if geometry == "obb":
        truth = np.column_stack((centers, sizes, 0.22 + 0.23 * np.sin(index / 9)))
    else:
        truth = np.column_stack((centers - sizes / 2, centers + sizes / 2))
    rng = np.random.default_rng(372)
    detections = truth + rng.normal(0, 0.6, truth.shape)
    if geometry == "obb":
        detections[:, 4] = truth[:, 4] + rng.normal(0, 0.015, count)
    frames = np.arange(count)
    if interrupted:
        detections[[0, 5, 6, count - 2]] = np.nan
        frames[9:] += 3
        frames[16:] += 2
    times = np.cumsum(0.025 + 0.018 * (1 + np.sin(index / 3)))
    scores = 0.45 + 0.5 * np.sin(index / 4) ** 2
    return CalibrationTrack("replay", 7, 0, frames, times, truth, detections, scores)


def _config(
    kind: str,
    geometry: str,
    *,
    tracker_name: str = "bytetrack",
    nonunit: bool = False,
    variable_dt: bool = False,
    adaptive: bool = False,
) -> dict[str, Any]:
    """Cover scalar scales and every dimension of fitted coordinate profiles."""
    options: dict[str, Any] = {"kalman.type": kind, "kalman.variable_dt": variable_dt}
    if tracker_name == "boosttrack":
        options["kalman.adaptive_kf"] = adaptive
    if nonunit:
        model = create_motion_model(kind, is_obb=geometry == "obb")
        options.update(dict(zip(KALMAN_NOISE_OPTIONS, (0.7, 1.6, 0.45, 1.3, 0.8), strict=True)))
        for offset, name in enumerate(KALMAN_DIAGONAL_OPTIONS):
            dimensions = model.dim_x - model.dim_z if "velocity" in name else model.dim_z
            options[name] = np.linspace(0.45 + offset / 10, 1.8 + offset / 10, dimensions).tolist()
        if tracker_name == "strongsort":
            options["kalman.noise.measurement_confidence_floor"] = np.linspace(0.05, 0.3, model.dim_z).tolist()
    return load_tracker_config(tracker_name, None, options)


def _scalar_replay(
    track: CalibrationTrack, *, tracker_name: str, geometry: str, config: Mapping[str, Any]
) -> dict[int, dict[str, Any]]:
    """Keep the pre-batching replay as an independent forecast scheduling oracle."""
    settings = normalize_kalman_config(config, tracker_name=tracker_name)
    model = create_motion_model(
        settings.type,
        is_obb=geometry == "obb",
        cls_id=track.class_id,
        adaptive_kf=bool(settings.adaptive_kf),
    )
    noise = settings.noise.for_class(track.class_id)
    kalman = model.create_filter(noise_config=noise)
    mean = covariance = None
    matched = np.all(np.isfinite(track.detection_boxes), axis=1)
    totals = {horizon: np.zeros(3) for horizon in (1, 3, 5)}
    for index, box in enumerate(track.detection_boxes):
        contiguous = index > 0 and track.frame_indices[index] == track.frame_indices[index - 1] + 1
        if not contiguous:
            mean = covariance = None
            kalman = model.create_filter(noise_config=noise)
        if mean is not None:
            dt = float(track.timestamps_s[index] - track.timestamps_s[index - 1]) if settings.variable_dt else None
            mean, covariance = model.predict(kalman, mean, covariance, dt=dt)
            kalman_replay._validate_state(mean, covariance)
        if not matched[index]:
            continue
        values = np.r_[box, track.scores[index]] if model.kind is MotionModelKind.XYSCR else box
        measurement = model.to_measurement(values, column=False)
        if mean is None:
            mean, covariance = model.initiate(kalman, measurement)
        else:
            confidence = float(track.scores[index]) if tracker_name == "strongsort" else 0.0
            mean, covariance = model.update(kalman, mean, covariance, measurement, confidence=confidence)
        kalman_replay._validate_state(mean, covariance)
        forecast, uncertainty = mean.copy(), covariance.copy()
        for horizon in range(1, min(5, len(track.gt_boxes) - index - 1) + 1):
            target = index + horizon
            if track.frame_indices[target] != track.frame_indices[target - 1] + 1:
                break
            dt = float(track.timestamps_s[target] - track.timestamps_s[target - 1]) if settings.variable_dt else None
            forecast, uncertainty = model.predict(kalman, forecast, uncertainty, dt=dt)
            kalman_replay._validate_state(forecast, uncertainty)
            if horizon in totals:
                iou, error = kalman_replay.box_metrics(
                    model.to_box(forecast)[0], track.gt_boxes[target], is_obb=model.is_obb
                )
                totals[horizon] += (iou, error, 1)
    return {
        horizon: {
            "mean_iou": float(iou / count),
            "normalized_center_error": float(error / count),
            "forecasts": int(count),
        }
        for horizon, (iou, error, count) in totals.items()
        if count
    }


def _assert_scores_equal(actual: dict, expected: dict) -> None:
    """Require identical forecast coverage as well as numerical score parity."""
    assert actual.keys() == expected.keys()
    for horizon in expected:
        assert actual[horizon]["forecasts"] == expected[horizon]["forecasts"]
        for metric in ("mean_iou", "normalized_center_error"):
            assert actual[horizon][metric] == pytest.approx(expected[horizon][metric], rel=0, abs=1e-10)


@pytest.mark.parametrize("kind", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("nonunit", [False, True], ids=["default", "calibrated"])
def test_batched_replay_preserves_scalar_scores_with_missing_observations_and_gaps(kind, geometry, nonunit):
    """All representations retain scalar metrics across missed frames and resets."""
    track = _track(geometry)
    options = dict(tracker_name="bytetrack", geometry=geometry, config=_config(kind, geometry, nonunit=nonunit))
    _assert_scores_equal(kalman_replay.replay_track(track, **options), _scalar_replay(track, **options))


@pytest.mark.parametrize("kind", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_variable_dt_replay_preserves_scalar_scores(kind, geometry):
    """Irregular capture intervals retain the original sequential forecasts."""
    track = _track(geometry)
    options = dict(
        tracker_name="bytetrack", geometry=geometry, config=_config(kind, geometry, nonunit=True, variable_dt=True)
    )
    _assert_scores_equal(kalman_replay.replay_track(track, **options), _scalar_replay(track, **options))


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_adaptive_xyhr_replay_preserves_scalar_scores_after_noise_warmup(geometry):
    """Adaptive histories cannot be forecast using the filter's later noise state."""
    track = _track(geometry, count=38, interrupted=False)
    options = dict(
        tracker_name="boosttrack",
        geometry=geometry,
        config=_config("xyhr", geometry, tracker_name="boosttrack", nonunit=True, adaptive=True),
    )
    _assert_scores_equal(kalman_replay.replay_track(track, **options), _scalar_replay(track, **options))


@pytest.mark.parametrize("kind", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_strongsort_replay_preserves_confidence_weighted_scalar_scores(kind, geometry):
    """Confidence-dependent measurement covariance remains in every history update."""
    track = _track(geometry)
    options = dict(
        tracker_name="strongsort",
        geometry=geometry,
        config=_config(kind, geometry, tracker_name="strongsort", nonunit=True),
    )
    _assert_scores_equal(kalman_replay.replay_track(track, **options), _scalar_replay(track, **options))


@pytest.mark.parametrize("kind", ["xyah", "xyscr"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_long_replay_preserves_forecasts_crossing_batch_boundaries(kind, geometry):
    """A chunk boundary must not truncate a still-annotated forecast horizon."""
    track = _track(geometry, count=280, interrupted=False)
    options = dict(tracker_name="bytetrack", geometry=geometry, config=_config(kind, geometry, nonunit=True))
    expected = _scalar_replay(track, **options)
    assert [expected[horizon]["forecasts"] for horizon in (1, 3, 5)] == [279, 277, 275]
    _assert_scores_equal(kalman_replay.replay_track(track, **options), expected)


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_unobserved_final_segment_preserves_configured_forecast_noise(geometry):
    """A reset without initialization must not replace pending origins' Q priors."""
    box = [20.0, 35.0, 40.0, 70.0, 0.2] if geometry == "obb" else [0.0, 0.0, 40.0, 70.0]
    truth = np.tile(box, (8, 1))
    detections = truth.copy()
    detections[1:] = np.nan
    track = CalibrationTrack(
        "unobserved-tail", 1, 0, np.array([0, 1, 2, 3, 4, 5, 20, 21]), None, truth, detections, np.full(8, 0.8)
    )
    config = load_tracker_config(
        "bytetrack", None, {"kalman.type": "xyscr", "kalman.noise.process_velocity_scale": 1e307}
    )
    options = dict(tracker_name="bytetrack", geometry=geometry, config=config)
    # XYSCR's configured velocity priors keep these forecasts finite. Its
    # constructor's unit Q would overflow, despite predicting identical boxes.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        expected = _scalar_replay(track, **options)
        actual = kalman_replay.replay_track(track, **options)
    assert [expected[horizon]["forecasts"] for horizon in (1, 3, 5)] == [1, 1, 1]
    _assert_scores_equal(actual, expected)


@pytest.mark.parametrize("kind", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
def test_fixed_step_forecasts_avoid_repeated_scalar_predictions(monkeypatch, kind):
    """The fast path removes per-origin scalar forecast work while preserving scores."""
    track = _track("aabb", count=40, interrupted=False)
    options = dict(tracker_name="bytetrack", geometry="aabb", config=_config(kind, "aabb"))
    original = MotionModelAdapter.predict
    calls = 0

    def count_prediction(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MotionModelAdapter, "predict", count_prediction)
    expected = _scalar_replay(track, **options)
    scalar_calls = calls
    calls = 0
    actual = kalman_replay.replay_track(track, **options)
    assert calls <= len(track.frame_indices)
    assert calls < scalar_calls / 4
    _assert_scores_equal(actual, expected)


def test_batched_forecast_covariance_divergence_is_rejected(monkeypatch):
    """A finite box must not conceal a divergent forecast covariance."""
    original = MotionModelAdapter.multi_predict

    def divergent_covariance(self, *args, **kwargs):
        means, covariances = original(self, *args, **kwargs)
        covariances[0, 0, 0] = np.inf
        return means, covariances

    monkeypatch.setattr(MotionModelAdapter, "multi_predict", divergent_covariance)
    with pytest.raises(ValueError, match="non-finite"):
        kalman_replay.replay_track(
            _track("aabb"), tracker_name="bytetrack", geometry="aabb", config=_config("xyah", "aabb")
        )
