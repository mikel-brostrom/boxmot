"""Calibration covariance bases agree with filters created by live trackers."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.engine.calibration.kalman_model import CalibrationModel
from boxmot.trackers import TrackerSpec, create_tracker
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_profile_signature
from tests.unit.trackers.test_trackers import _aabb_rows, _obb_rows
from tests.unit.trackers.test_variable_frame_time import _state, _update


@pytest.mark.parametrize("tracker_name", sorted(KALMAN_TRACKER_NAMES))
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("timed", [False, True])
def test_covariance_bases_reconstruct_actual_tracker_matrices(tracker_name: str, geometry: str, timed: bool) -> None:
    options = {
        "min_hits": 1,
        "kalman.variable_dt": timed,
        "kalman.noise.reference_dt_s": 0.05,
        "kalman.noise.process_position_scale": 3.0,
        "kalman.noise.process_velocity_scale": 7.0,
        "kalman.noise.measurement_noise_scale": 5.0,
        "kalman.noise.initial_position_scale": 4.0,
        "kalman.noise.initial_velocity_scale": 6.0,
    }
    tracker = create_tracker(TrackerSpec(tracker_name, geometry=geometry, options=tuple(sorted(options.items()))))
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1]
    _update(tracker, rows, index=0, timestamp_s=0.0 if timed else None)
    track = tracker.active_tracks[0]
    actual_mean, actual_filter = _state(track)
    actual_initial_covariance = getattr(track, "covariance", actual_filter.P).copy()
    model = CalibrationModel(tracker_name, geometry, options)
    signature = calibration_profile_signature(tracker_name, geometry, options)
    assert signature["filter"] == model.kind.value
    assert signature["state_dimensions"] == model.dim_x
    assert signature["measurement_dimensions"] == model.dim_z
    geometry_size = 5 if geometry == "obb" else 4
    score = float(rows[0, geometry_size])
    measurement = model.to_measurement(rows[0, :geometry_size], score)
    mean, initial_covariance = model.initial_state(measurement)

    np.testing.assert_allclose(mean, actual_mean)
    np.testing.assert_allclose(
        actual_initial_covariance[: model.dim_z, : model.dim_z],
        4.0 * initial_covariance[: model.dim_z, : model.dim_z],
    )
    np.testing.assert_allclose(
        actual_initial_covariance[model.dim_z :, model.dim_z :],
        6.0 * initial_covariance[model.dim_z :, model.dim_z :],
    )

    zero_covariance = np.zeros_like(initial_covariance)
    if tracker_name in {"boosttrack", "occluboost"}:
        actual_r = actual_filter.project(mean, zero_covariance)[1]
    elif tracker_name in {"ocsort", "deepocsort", "hybridsort"}:
        actual_r = actual_filter.project_state(x=mean, P=zero_covariance)[1]
    else:
        confidence = score if tracker_name == "strongsort" else 0.0
        actual_r = actual_filter.project(mean, zero_covariance, confidence=confidence)[1]
    np.testing.assert_allclose(actual_r, 5.0 * model.measurement_covariance(measurement, score))

    interval = 0.12 if timed else None
    mean[list(model.velocity_indices)] = 0.01
    q_position, q_velocity = model.process_covariance_bases(mean, interval)
    if tracker_name in {"ocsort", "deepocsort", "hybridsort"}:
        actual_filter.x = mean.reshape(-1, 1).copy()
        actual_filter.P = zero_covariance.copy()
        actual_filter.predict(dt=interval)
        predicted_mean, actual_q = actual_filter.x.reshape(-1), actual_filter.P
    else:
        predicted_mean, actual_q = actual_filter.predict(mean.copy(), zero_covariance.copy(), dt=interval)

    np.testing.assert_allclose(predicted_mean, model.transition(interval) @ mean, atol=1e-12)
    np.testing.assert_allclose(actual_q, 3.0 * q_position + 7.0 * q_velocity, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(q_position)) >= -1e-12
    assert np.min(np.linalg.eigvalsh(q_velocity)) >= -1e-12
    assert options["kalman.noise.process_position_scale"] == 3.0


@pytest.mark.parametrize("tracker_name", sorted(KALMAN_TRACKER_NAMES))
def test_obb_measurements_align_equivalent_rectangles_and_angle_wrap(tracker_name: str) -> None:
    model = CalibrationModel(tracker_name, "obb", {})
    reference_box = np.array([60.0, 80.0, 40.0, 20.0, np.pi - 0.02])
    reference = model.to_measurement(reference_box)
    equivalent_box = reference_box.copy()
    equivalent_box[2:4] = equivalent_box[3:1:-1]
    equivalent_box[4] += np.pi / 2.0
    np.testing.assert_allclose(model.to_measurement(equivalent_box, reference=reference), reference, atol=1e-12)

    crossed_box = reference_box.copy()
    crossed_box[4] = -np.pi + 0.03
    crossed = model.to_measurement(crossed_box, reference=reference)
    assert crossed[4] - reference[4] == pytest.approx(0.05)


def test_hybrid_sort_confidence_is_not_supervised_by_box_ground_truth() -> None:
    model = CalibrationModel("hybridsort", "aabb", {})
    assert model.dim_z == 5
    assert model.dim_x == 9
    assert model.measurement_indices == (0, 1, 2, 4)
    assert model.velocity_indices == (5, 6, 7)
    assert model.velocity_measurement_indices == (0, 1, 2)
    assert model.to_measurement(np.array([10.0, 20.0, 40.0, 80.0]), score=0.85)[3] == 0.85


def test_class_reference_model_removes_existing_noise_scales():
    options = {
        "per_class": True,
        "kalman": {
            "noise": {
                "measurement_noise_scale": 5.0,
                "initial_position_scale": 8.0,
                "by_class": {"2": {"measurement_noise_scale": 7.0, "initial_position_scale": 9.0}},
            },
        },
    }
    baseline = CalibrationModel("bytetrack", "aabb", {}, cls_id=2)
    model = CalibrationModel("bytetrack", "aabb", options, cls_id=2)
    measurement = model.to_measurement(np.array([10.0, 20.0, 40.0, 80.0]))
    np.testing.assert_array_equal(model.initial_state(measurement)[1], baseline.initial_state(measurement)[1])
    np.testing.assert_array_equal(
        model.measurement_covariance(measurement), baseline.measurement_covariance(measurement)
    )
    assert not model.noise_config.by_class
    assert options["kalman"]["noise"]["by_class"]["2"]["measurement_noise_scale"] == 7.0


@pytest.mark.parametrize("tracker_name", ["ocsort", "deepocsort", "hybridsort"])
def test_obb_xysr_derivative_mapping_keeps_ratio_static(tracker_name: str) -> None:
    model = CalibrationModel(tracker_name, "obb", {})
    assert model.velocity_indices == (5, 6, 7, 8)
    assert model.velocity_measurement_indices == (0, 1, 2, 4)


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_strongsort_uses_runtime_confidence_uncertainty_without_an_invented_floor(geometry: str) -> None:
    model = CalibrationModel("strongsort", geometry, {})
    box = np.array([30.0, 40.0, 20.0, 10.0, 0.2]) if geometry == "obb" else np.array([20.0, 30.0, 40.0, 80.0])
    measurement = model.to_measurement(box)
    baseline = model.measurement_covariance(measurement, score=0.0)
    np.testing.assert_allclose(model.measurement_covariance(measurement, score=0.7), baseline * 0.09)
    np.testing.assert_array_equal(model.measurement_covariance(measurement, score=1.0), np.zeros_like(baseline))


@pytest.mark.parametrize("tracker_name", sorted(KALMAN_TRACKER_NAMES))
def test_seconds_requires_measured_prediction_interval(tracker_name: str) -> None:
    model = CalibrationModel(tracker_name, "aabb", {"kalman.variable_dt": True})
    measurement = model.to_measurement(np.array([20.0, 30.0, 40.0, 80.0]))
    mean, _ = model.initial_state(measurement)
    with pytest.raises(ValueError, match="explicit measured dt"):
        model.transition()
    with pytest.raises(ValueError, match="explicit measured dt"):
        model.process_covariance_bases(mean)


def test_calibration_rejects_wrong_time_basis_and_trackers_without_a_kalman_filter() -> None:
    with pytest.raises(ValueError, match="conflicts with"):
        CalibrationModel("botsort", "aabb", {"kalman.variable_dt": True, "kalman.noise.time_unit": "frames"})
    with pytest.raises(ValueError, match="supported Kalman"):
        CalibrationModel("sfsort", "aabb", {})


@pytest.mark.parametrize(
    "tracker_name, parameter",
    [
        ("ocsort", "Q_xy_scaling"),
        ("deepocsort", "Q_s_scaling"),
        ("ocsort", "Q_a_scaling"),
        ("deepocsort", "unknown_tracker_parameter"),
    ],
)
def test_calibration_model_rejects_options_runtime_cannot_consume(tracker_name: str, parameter: str) -> None:
    with pytest.raises(ValueError, match=parameter):
        CalibrationModel(tracker_name, "aabb", {parameter: 0.2})


@pytest.mark.parametrize(
    "parameter, value, error",
    [
        ("max_object", 64, TypeError),
        ("max_objects", 0, ValueError),
        ("max_objects", 64.5, TypeError),
        ("min_coverage", 1.1, ValueError),
        ("min_fill", -0.1, ValueError),
        ("prompt_overlap", 0.0, ValueError),
    ],
)
def test_calibration_rejects_invalid_guidance_options(parameter: str, value: float, error: type[Exception]) -> None:
    """Accepting the guidance group must retain the runtime value constraints."""
    with pytest.raises(error, match=parameter):
        CalibrationModel("occluboost", "aabb", {"edgetam": {parameter: value}})
