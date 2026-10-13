"""Geometry coordinate calibration cannot transfer area errors into center noise."""

from dataclasses import replace

import numpy as np
import pytest

from boxmot.engine.calibration.kalman_data import CalibrationTrack
from boxmot.engine.calibration.kalman_fit import apply_global_noise_fallback, fit_kalman_noise
from boxmot.engine.calibration.kalman_model import CalibrationModel
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_DIAGONAL_OPTIONS, KALMAN_NOISE_OPTIONS


def _trajectory(geometry: str, size_error: float = 0.0) -> CalibrationTrack:
    """Use the same center errors with independently controllable area errors."""
    count = 8
    boxes = np.tile([100.0, 120.0, 40.0, 80.0, 0.1], (count, 1))
    boxes[:, 0] += np.arange(count)
    detections = boxes.copy()
    detections[:, 0] += 6.0
    detections[:, 1] -= 10.0
    detections[:, 2] += size_error
    if geometry == "aabb":
        boxes = np.column_stack((boxes[:, :2] - boxes[:, 2:4] / 2, boxes[:, :2] + boxes[:, 2:4] / 2))
        detections = np.column_stack(
            (detections[:, :2] - detections[:, 2:4] / 2, detections[:, :2] + detections[:, 2:4] / 2)
        )
    return CalibrationTrack(
        "fixture", 1, 0, np.arange(count), np.arange(count) * 0.1, boxes, detections, np.full(count, 0.7)
    )


@pytest.mark.parametrize("kind", ["xysr", "xyscr"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("timed", [False, True])
def test_large_area_errors_do_not_inflate_center_or_confidence_noise(kind: str, geometry: str, timed: bool) -> None:
    options = load_tracker_config("bytetrack", None, {"kalman.type": kind, "kalman.variable_dt": timed})
    model = CalibrationModel("bytetrack", geometry, options)
    results = []
    for error in (0.0, 200.0):
        track = _trajectory(geometry, error)
        parameters, _ = fit_kalman_noise((track, replace(track, track_id=2)), {0: model}, options, variable_dt=timed)
        fitted = {**options, **{key: item["value"] for key, item in parameters.items()}}
        noise = normalize_kalman_config(fitted, tracker_name="bytetrack").noise
        measurement = model.to_measurement(track.detection_boxes[0], score=0.7)
        covariance = noise.measurement_covariance(model.measurement_covariance(measurement))
        np.testing.assert_allclose(np.diag(covariance)[:2], [36.0, 100.0])
        assert all(parameters[key]["value"] == 1.0 for key in KALMAN_NOISE_OPTIONS)
        results.append(parameters)
        if kind == "xyscr":
            for key in KALMAN_DIAGONAL_OPTIONS:
                index = 3  # Confidence and vconfidence are fourth in their respective blocks.
                assert parameters[key]["value"][index] == 1.0
                assert parameters[key]["coordinates"][index]["status"] == "excluded"
    key = "kalman.noise.measurement_noise_diagonal"
    assert results[0][key]["value"][:2] == results[1][key]["value"][:2]
    assert results[1][key]["value"][2] > 1e6 * results[0][key]["value"][2]


@pytest.mark.parametrize("kind", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_each_coordinate_has_its_own_evidence_and_static_ratio_is_identifiable(kind: str, geometry: str) -> None:
    options = load_tracker_config("bytetrack", None, {"kalman.type": kind})
    model = CalibrationModel("bytetrack", geometry, options)
    track = _trajectory(geometry)
    parameters, counts = fit_kalman_noise((track, replace(track, track_id=2)), {0: model}, options)
    assert counts == {"gt_transitions": 12, "gt_lag_pairs": 10}
    for key in KALMAN_DIAGONAL_OPTIONS:
        assert len(parameters[key]["value"]) == (model.dim_x - model.dim_z if "velocity" in key else model.dim_z)
        for entry in parameters[key]["coordinates"]:
            if entry["coordinate"] in {"confidence", "v_confidence"}:
                assert entry["status"] == "excluded"
            else:
                assert entry["status"] == "fitted"
                assert entry["events"] >= 2
    if kind in {"xysr", "xyscr"}:
        ratio = next(
            entry
            for entry in parameters["kalman.noise.process_position_diagonal"]["coordinates"]
            if entry["coordinate"] == "aspect_ratio"
        )
        assert ratio["status"] == "fitted"
        assert ratio["value"] == 1e-6


def test_class_fallback_is_per_coordinate_and_does_not_replace_supported_estimates() -> None:
    options = load_tracker_config("bytetrack")
    model = CalibrationModel("bytetrack", "aabb", options)
    tracks = (_trajectory("aabb"), replace(_trajectory("aabb"), track_id=2))
    pooled, _ = fit_kalman_noise(tracks, {0: model}, options)
    sparse, _ = fit_kalman_noise(tracks[:1], {0: model}, options)
    result = apply_global_noise_fallback(sparse, pooled)
    # A single long trajectory supports measurement noise, but only one birth.
    assert result["kalman.noise.measurement_noise_diagonal"]["status"] == "fitted"
    assert result["kalman.noise.initial_position_diagonal"]["status"] == "global_fallback"
    assert (
        result["kalman.noise.initial_position_diagonal"]["value"]
        == pooled["kalman.noise.initial_position_diagonal"]["value"]
    )
    for item in result["kalman.noise.initial_position_diagonal"]["coordinates"]:
        assert item["source"] == "pooled_classes"
        assert item["events"] == 1
        assert item["global_events"] == 2


def test_reference_model_strips_previous_coordinate_calibration_and_preserves_input() -> None:
    options = load_tracker_config(
        "bytetrack",
        None,
        {
            "kalman.noise.measurement_noise_diagonal": [3.0, 4.0, 5.0, 6.0],
            "kalman.noise.process_velocity_diagonal": [7.0, 8.0, 9.0, 10.0],
        },
    )
    model = CalibrationModel("bytetrack", "aabb", options)
    default = CalibrationModel("bytetrack", "aabb", {})
    measurement = model.to_measurement(_trajectory("aabb").gt_boxes[0])
    np.testing.assert_array_equal(
        model.measurement_covariance(measurement), default.measurement_covariance(measurement)
    )
    for actual, expected in zip(
        model.process_covariance_bases(model.initial_state(measurement)[0]),
        default.process_covariance_bases(default.initial_state(measurement)[0]),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert options["kalman.noise.measurement_noise_diagonal"] == [3.0, 4.0, 5.0, 6.0]
