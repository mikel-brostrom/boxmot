"""3D calibration fits runtime covariance bases and exports reusable profiles."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.datasets.inputs import DatasetInputs
from boxmot.engine.calibration import kalman_sensor
from boxmot.engine.calibration.kalman import calibrate_kalman, fit_kalman_noise, validate_kf_calibration
from boxmot.engine.calibration.kalman_data import CalibrationData, CalibrationTrack
from boxmot.engine.calibration.kalman_model_3d import CalibrationModel3D
from boxmot.engine.eval.eagermot_kitti import load_kitti_profiles
from boxmot.engine.eval.results import ValidationResult
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS, KalmanNoiseConfig
from boxmot.trackers.eagermot.motion import Kalman3D


def _dataset(tmp_path):
    return DatasetInputs(
        config_path=tmp_path / "dataset.yaml",
        id="sensor",
        root=tmp_path,
        split="train",
        sequence_names=("0000",),
        sequences=(),
        classes={"car": {"id": 1, "evaluation": "target"}, "pedestrian": {"id": 2, "evaluation": "target"}},
        fps=10.0,
    )


def _track(class_id=1, track_id=1, offset=0.1, count=6):
    boxes = np.tile([0.0, 0.0, 20.0, 0.0, 4.0, 2.0, 1.6], (count, 1))
    boxes[:, 0] = np.arange(count) * 0.2
    detections = boxes.copy()
    detections[:, 0] += offset
    return CalibrationTrack("0000", track_id, class_id, np.arange(count), None, boxes, detections, np.full(count, 0.8))


def _data():
    tracks = tuple(_track(class_id, track_id, offset=class_id * 0.1) for class_id in (1, 2) for track_id in (1, 2))
    return CalibrationData(
        tracks,
        {"matched": 24, "trajectories": 4},
        ({"sequence_id": "0000", "path": "gt.txt", "sha256": "gt-hash"},),
        input_sources=(
            {"sequence_id": "0000", "role": "poses", "path": "poses.npy", "sha256": "pose-hash"},
            {"sequence_id": "0000", "role": "coordinate_frame", "value": "world", "sha256": "world-hash"},
        ),
    )


@pytest.mark.parametrize("angular", [False, True])
def test_3d_bases_match_runtime_with_all_five_scales(angular):
    scales = dict(zip(KALMAN_NOISE_OPTIONS, (2.0, 3.0, 4.0, 5.0, 6.0), strict=True))
    model = CalibrationModel3D({"is_angular": angular, **scales})
    box = np.array([1.0, 2.0, 20.0, 0.3, 4.0, 2.0, 1.6])
    noise = KalmanNoiseConfig(**{key.removeprefix("kf_"): value for key, value in scales.items()})
    runtime = Kalman3D(box, is_angular=angular, noise_config=noise)
    state, p0 = model.initial_state(box)
    q_position, q_velocity = model.process_covariance_bases(state)
    np.testing.assert_array_equal(state, runtime.state)
    np.testing.assert_array_equal(model.transition(), runtime._transition)
    np.testing.assert_array_equal(noise.initial_covariance(p0, velocity_start=7), runtime.covariance)
    np.testing.assert_allclose(2 * q_position + 3 * q_velocity, runtime._process_noise)
    np.testing.assert_allclose(4 * model.measurement_covariance(box), runtime._measurement_noise)
    assert model.velocity_measurement_indices == ((0, 1, 2, 3) if angular else (0, 1, 2))


def test_yaw_alignment_uses_pi_equivalence_and_keeps_inputs_unchanged():
    model = CalibrationModel3D({"is_angular": True})
    reference = np.array([1.0, 2.0, 20.0, np.pi - 0.03, 4.0, 2.0, 1.6])
    box = reference.copy()
    box[3] = -0.02
    saved = box.copy()
    measurement = model.to_measurement(box, reference=reference)
    assert measurement[3] - reference[3] == pytest.approx(0.01)
    np.testing.assert_array_equal(box, saved)


@pytest.mark.parametrize(
    "options", [{"variable_dt": True}, {"kf_time_unit": "seconds"}, {"kf_measurement_noise_scale": 0}]
)
def test_3d_model_rejects_unusable_timing_or_noise(options):
    with pytest.raises(ValueError):
        CalibrationModel3D(options)


@pytest.mark.parametrize("angular", [False, True])
def test_shared_fit_recovers_simulated_3d_diffusion_and_measurement_noise(angular):
    rng = np.random.default_rng(1709)
    model = CalibrationModel3D({"is_angular": angular})
    base = dict.fromkeys(KALMAN_NOISE_OPTIONS, 1.0)
    position_scale, velocity_scale, measurement_scale = 0.0002, 0.03, 0.09
    q_position, q_velocity = model.process_covariance_bases(np.zeros(model.dim_x))
    q_std = np.sqrt(np.diag(position_scale * q_position + velocity_scale * q_velocity))
    r_std = np.sqrt(0.01 * measurement_scale)
    trajectories = []
    for track_id in range(100):
        states = np.zeros((150, model.dim_x))
        states[0, :7] = [0.0, 0.0, 20.0, 0.0, 4.0, 2.0, 1.6]
        states[0, 7:] = 0.02
        increments = rng.normal(size=(149, model.dim_x)) * q_std
        for index in range(1, len(states)):
            states[index] = model.transition() @ states[index - 1] + increments[index - 1]
        gt = states[:, :7]
        detected = gt + rng.normal(size=gt.shape) * r_std
        trajectories.append(
            CalibrationTrack("simulation", track_id, 1, np.arange(len(gt)), None, gt, detected, np.ones(len(gt)))
        )
    parameters, stats = fit_kalman_noise(trajectories, {1: model}, base)
    assert parameters["kf_process_position_scale"]["value"] == pytest.approx(position_scale, rel=0.15)
    assert parameters["kf_process_velocity_scale"]["value"] == pytest.approx(velocity_scale, rel=0.15)
    assert parameters["kf_measurement_noise_scale"]["value"] == pytest.approx(measurement_scale, rel=0.03)
    assert parameters["kf_initial_position_scale"]["value"] == pytest.approx(0.001 * measurement_scale, rel=0.15)
    assert parameters["kf_initial_velocity_scale"]["value"] > 0
    assert stats == {"gt_transitions": 14800, "gt_lag_pairs": 14700}


def test_class_profiles_fit_separately_preserve_other_settings_and_record_sources(monkeypatch, tmp_path):
    data = _data()
    monkeypatch.setattr(kalman_sensor, "load_sensor_calibration_data", lambda *a, **kw: data)
    profiles = load_kitti_profiles()
    profiles[2]["is_angular"] = True
    progress = []
    result = kalman_sensor.calibrate_sensor_kalman(
        _dataset(tmp_path), profiles, output_dir=tmp_path, progress=progress.append
    )
    saved = load_kitti_profiles(result.config_path)
    assert saved[2]["kf_measurement_noise_scale"] == pytest.approx(4 * saved[1]["kf_measurement_noise_scale"])
    for class_id in profiles:
        assert {key: value for key, value in saved[class_id].items() if key not in KALMAN_NOISE_OPTIONS} == {
            key: value for key, value in profiles[class_id].items() if key not in KALMAN_NOISE_OPTIONS
        }
    assert result.matched_detections == 24
    assert result.gt_transitions == 16
    assert result.parameter_count == 10
    assert len(result.fitted_parameters) == 10
    assert "10/10 covariance scales" in result.description
    report = json.loads(result.report_path.read_text())
    assert report["input_sources"] == list(data.input_sources)
    assert report["ground_truth_sources"] == list(data.ground_truth_sources)
    assert report["classes"]["car"]["state_dimensions"] == 10
    assert report["classes"]["pedestrian"]["state_dimensions"] == 11
    assert report["coordinates"] == "world"
    assert report["coordinate_frames"] == {"0000": "world"}
    assert report["matching"]["coordinates"] == "camera"
    assert report["score_scope"] == "calibrated_on_selected_split"
    assert progress[-1] == "KF calibration complete: 10/10 3D scales fitted."
    result.record_final(ValidationResult("sensor", {"HOTA": 60.0}, "single_class", {"HOTA": 60.0}, tmp_path, None))
    assert json.loads(result.report_path.read_text())["final_summary"] == {"HOTA": 60.0}


def test_sparse_or_absent_class_evidence_retains_supplied_scales(monkeypatch, tmp_path):
    data = CalibrationData((_track(count=1),), {"matched": 1}, ())
    monkeypatch.setattr(kalman_sensor, "load_sensor_calibration_data", lambda *a, **kw: data)
    profiles = load_kitti_profiles()
    for class_id, profile in profiles.items():
        profile.update({key: float(class_id + index) for index, key in enumerate(KALMAN_NOISE_OPTIONS)})
    result = kalman_sensor.calibrate_sensor_kalman(_dataset(tmp_path), profiles, output_dir=tmp_path)
    assert load_kitti_profiles(result.config_path) == profiles
    assert result.fitted_parameters == ()
    report = json.loads(result.report_path.read_text())
    assert report["classes"]["pedestrian"]["statistics"]["ground_truth"] == 0
    assert all(value["status"] == "retained" for value in report["classes"]["pedestrian"]["parameters"].values())


def test_calibration_report_identifies_camera_coordinates_without_ego_poses(monkeypatch, tmp_path):
    data = replace(
        _data(),
        input_sources=(
            {"sequence_id": "0000", "role": "coordinate_frame", "value": "camera", "sha256": "camera-hash"},
        ),
    )
    monkeypatch.setattr(kalman_sensor, "load_sensor_calibration_data", lambda *a, **kw: data)
    result = kalman_sensor.calibrate_sensor_kalman(_dataset(tmp_path), load_kitti_profiles(), output_dir=tmp_path)
    report = json.loads(result.report_path.read_text())
    assert report["coordinates"] == "camera"
    assert report["coordinate_frames"] == {"0000": "camera"}


def test_no_matches_fail_without_publishing_artifacts(monkeypatch, tmp_path):
    data = replace(_data(), statistics={"matched": 0})
    monkeypatch.setattr(kalman_sensor, "load_sensor_calibration_data", lambda *a, **kw: data)
    with pytest.raises(ValueError, match="no detections matched"):
        kalman_sensor.calibrate_sensor_kalman(_dataset(tmp_path), load_kitti_profiles(), output_dir=tmp_path)
    assert not (tmp_path / "kf-tuning").exists()


def test_shared_entrypoint_dispatches_3d_without_build_setup(monkeypatch, tmp_path):
    from boxmot.datasets import inputs
    from boxmot.engine.eval import evaluator

    validate_kf_calibration("eagermot")
    monkeypatch.setattr(inputs, "load_dataset_inputs", lambda *a, **kw: _dataset(tmp_path))
    monkeypatch.setattr(kalman_sensor, "load_sensor_calibration_data", lambda *a, **kw: _data())
    monkeypatch.setattr(
        evaluator, "_ensure_setup", lambda *a: pytest.fail("3D calibration must not prepare a vision build")
    )
    args = SimpleNamespace(tracker="eagermot", dataset=tmp_path, split="train", class_config=None)
    result = calibrate_kalman(args, output_dir=tmp_path)
    assert load_kitti_profiles(result.config_path)[1]["kf_measurement_noise_scale"] != 1
