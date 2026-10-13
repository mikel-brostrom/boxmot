"""Confidence localization curves integrate with geometry and class calibration."""

from dataclasses import replace

import numpy as np
import pytest

from boxmot.engine.calibration.kalman import calibrate_kalman
from boxmot.engine.calibration.kalman_fit import apply_global_noise_fallback, fit_kalman_noise
from boxmot.engine.calibration.kalman_model import CalibrationModel
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_DIAGONAL_OPTIONS
from tests.unit.engine.calibration.test_kalman import _args, _data, _load_fixture
from tests.unit.engine.calibration.test_kalman_coordinate_noise import _trajectory


@pytest.mark.parametrize("kind", ["xyah", "xywh", "xyhr", "xysr", "xyscr"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("timed", [False, True])
def test_strongsort_fits_confidence_floor_in_geometry_coordinates_only(kind, geometry, timed):
    options = load_tracker_config("strongsort", None, {"kalman.type": kind, "kalman.variable_dt": timed})
    model = CalibrationModel("strongsort", geometry, options)
    track = _trajectory(geometry)
    scores = np.linspace(0.1, 1.0, len(track.scores))
    detections = track.gt_boxes.copy()
    x_error = np.sqrt(0.4 + 1.6 * (1 - scores) ** 2)
    if geometry == "aabb":
        detections[:, [0, 2]] += x_error[:, None]
    else:
        detections[:, 0] += x_error
    track = replace(track, detection_boxes=detections, scores=scores)
    parameters, _ = fit_kalman_noise((track, replace(track, track_id=2)), {0: model}, options, variable_dt=timed)
    floor = parameters["kalman.noise.measurement_confidence_floor"]
    assert floor["value"][0] == pytest.approx(0.2)
    assert floor["coordinates"][0]["status"] == "fitted"
    assert floor["coordinates"][0]["events"] == 16  # Includes score-one observations.
    assert all(0 <= value <= 1 for value in floor["value"])
    if kind == "xyscr":
        assert floor["value"][3] == 0
        assert floor["coordinates"][3]["status"] == "excluded"
        assert all(parameters[name]["value"][3] == 1 for name in KALMAN_DIAGONAL_OPTIONS)


def test_class_curve_uses_pooled_scale_and_floor_together_when_scores_do_not_vary():
    options = load_tracker_config("strongsort")
    model = CalibrationModel("strongsort", "aabb", options)
    track = _trajectory("aabb")
    varying = replace(track, scores=np.linspace(0.1, 0.9, len(track.scores)))
    pooled, _ = fit_kalman_noise((varying, replace(varying, track_id=2)), {0: model}, options)
    constant = replace(track, scores=np.full(len(track.scores), 0.7))
    sparse, _ = fit_kalman_noise((constant,), {0: model}, options)
    actual = apply_global_noise_fallback(sparse, pooled)
    for name in ("kalman.noise.measurement_noise_diagonal", "kalman.noise.measurement_confidence_floor"):
        assert sparse[name]["coordinates"][0]["status"] == "retained"
        assert actual[name]["coordinates"][0]["status"] == "global_fallback"
        assert actual[name]["value"][0] == pooled[name]["value"][0]


def test_reference_model_resets_confidence_floor_without_mutating_authored_config():
    options = load_tracker_config(
        "strongsort", None, {"kalman.noise.measurement_confidence_floor": [0.2, 0.3, 0.4, 0.5]}
    )
    model = CalibrationModel("strongsort", "aabb", options)
    baseline = CalibrationModel("strongsort", "aabb", {})
    measurement = model.to_measurement(_trajectory("aabb").gt_boxes[0])
    np.testing.assert_array_equal(
        model.measurement_covariance(measurement, score=0.9), baseline.measurement_covariance(measurement, score=0.9)
    )
    assert not model.noise_config.measurement_confidence_floor
    assert options["kalman.noise.measurement_confidence_floor"] == [0.2, 0.3, 0.4, 0.5]


def test_saved_class_profiles_retain_learned_confidence_curve_and_pooled_fallback(monkeypatch, tmp_path):
    data = _data()
    tracks = tuple(replace(track, scores=np.linspace(0.1, 1.0, len(track.scores))) for track in data.tracks)
    _load_fixture(monkeypatch, replace(data, tracks=tracks))
    result = calibrate_kalman(
        _args(tmp_path, tracker="strongsort", per_class=True, tracker_class_ids=(0, 1)), output_dir=tmp_path
    )
    saved = load_tracker_config("strongsort", result.config_path)
    for option in ("measurement_confidence_floor", "measurement_noise_diagonal"):
        assert saved[f"kalman.noise.by_class.1.{option}"][0] == pytest.approx(saved[f"kalman.noise.{option}"][0])
    assert saved["kalman.noise.measurement_confidence_floor"][0] == pytest.approx(1.0)
    assert any("measurement_confidence_floor[0]" in name for name in result.fitted_parameters)
    assert result.parameter_count == 72
