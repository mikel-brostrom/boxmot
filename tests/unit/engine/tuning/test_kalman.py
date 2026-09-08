"""Supervised calibration, saved configuration reuse, and single evaluation wiring."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from boxmot.engine.eval import evaluator
from boxmot.engine.eval.results import ValidationResult
from boxmot.engine.tuning import kalman_data
from boxmot.engine.tuning.kalman import calibrate_kalman, validate_kf_calibration
from boxmot.engine.tuning.kalman_data import CalibrationData, CalibrationTrack
from boxmot.motion.kalman_filters.fitting import MIN_COVARIANCE_SCALE
from boxmot.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS, KALMAN_TRACKER_NAMES
from boxmot.trackers.config import load_tracker_config


def _args(tmp_path, **overrides):
    return SimpleNamespace(
        **{
            "tracker": "bytetrack",
            "tracker_backend": "python",
            "geometry": "aabb",
            "dataset_id": "fixture",
            "split": "ablation",
            "build_path": tmp_path / "build",
            "sequence_names": ("seq1",),
            "variable_dt": False,
            "_build_validated": True,
            **overrides,
        }
    )


def _data(*, geometry="aabb", count=6, tracks=2, offset=2.0):
    times = np.array([0.0, 0.1, 0.4, 0.45, 0.7, 1.1])[:count]
    trajectories = []
    for track_id in range(tracks):
        boxes = np.tile([10.0, 20.0 + track_id * 200, 30.0, 120.0 + track_id * 200], (count, 1))
        boxes[:, [0, 2]] += 10 * times[:, None]
        if geometry == "obb":
            boxes = np.column_stack(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2], np.zeros(count)))
        detections = boxes.copy()
        detections[:, [0] if geometry == "obb" else [0, 2]] += offset
        trajectories.append(
            CalibrationTrack("seq1", track_id, 0, np.arange(count), times, boxes, detections, np.full(count, 0.8))
        )
    return CalibrationData(tuple(trajectories), {"matched": tracks * count, "trajectories": tracks}, ())


def _load_fixture(monkeypatch, data):
    calls = []

    def load(args, **kwargs):
        calls.append(args)
        return data

    monkeypatch.setattr(kalman_data, "load_calibration_data", load)
    return calls


def _fake_replay(monkeypatch, score=64.0):
    calls = []

    def replay(args, **kwargs):
        config = dict(evaluator._tracker_options(args, kwargs.get("evolve_config")))
        calls.append((args, config, kwargs))
        return ValidationResult("fixture", {"HOTA": score}, "single_class", {"HOTA": score}, kwargs["output_dir"], args)

    monkeypatch.setattr(evaluator, "run_eval", replay)
    return calls


@pytest.mark.parametrize("variable_dt", [False, True])
def test_calibration_fits_errors_without_replay_and_holds_other_settings(monkeypatch, tmp_path, variable_dt):
    args = _args(tmp_path, variable_dt=variable_dt, asso_func="giou", per_class=True, tracker_class_ids=(0,))
    inputs = _load_fixture(monkeypatch, _data())
    replays = _fake_replay(monkeypatch)
    progress = []
    result = calibrate_kalman(args, output_dir=tmp_path, progress=progress.append)
    saved = load_tracker_config("bytetrack", result.config_path)
    assert replays == []
    assert inputs[0] is not args
    assert inputs[0].variable_dt is variable_dt
    assert result.matched_detections == 12
    assert result.gt_transitions == 8
    assert set(result.fitted_parameters) == set(KALMAN_NOISE_OPTIONS)
    assert saved["variable_dt"] is variable_dt
    assert saved["kf_time_unit"] == ("seconds" if variable_dt else "frames")
    assert saved["asso_func"] == "giou"
    assert saved["kf_measurement_noise_scale"] == pytest.approx(0.04)
    assert saved["kf_initial_position_scale"] == pytest.approx(0.01)
    report = json.loads(result.report_path.read_text())
    for key, value in report["baseline_config"].items():
        if key not in KALMAN_NOISE_OPTIONS:
            assert saved[key] == value
    assert report["method"] == "supervised_covariance_moments"
    assert report["score_scope"] == "calibrated_on_selected_split"
    assert report["sequences"] == ["seq1"]
    assert report["per_class"] is True
    assert report["class_ids"] == [0]
    assert "final_summary" not in report
    assert "trials" not in report
    assert result.config_path.name == "calibrated.yaml"
    assert result.report_path.name == "calibration.json"
    assert "12 matched detections" in result.description
    assert progress[0].startswith("KF calibration:")
    assert not hasattr(args, "exp_dir")


def test_irregular_intervals_preserve_constant_velocity_process_residuals(monkeypatch, tmp_path):
    _load_fixture(monkeypatch, _data())
    results = []
    for variable_dt in (False, True):
        result = calibrate_kalman(_args(tmp_path, variable_dt=variable_dt), output_dir=tmp_path / str(variable_dt))
        results.append(load_tracker_config("bytetrack", result.config_path))
    assert results[1]["kf_process_position_scale"] == MIN_COVARIANCE_SCALE
    assert results[1]["kf_process_velocity_scale"] == MIN_COVARIANCE_SCALE
    assert results[0]["kf_process_velocity_scale"] > MIN_COVARIANCE_SCALE
    assert results[0]["kf_measurement_noise_scale"] == results[1]["kf_measurement_noise_scale"]


def test_larger_detector_errors_increase_r_and_initial_position_but_not_q(monkeypatch, tmp_path):
    configurations = []
    for offset in (1.0, 3.0):
        _load_fixture(monkeypatch, _data(offset=offset))
        result = calibrate_kalman(_args(tmp_path), output_dir=tmp_path / str(offset))
        configurations.append(load_tracker_config("bytetrack", result.config_path))
    small, large = configurations
    for key in ("kf_measurement_noise_scale", "kf_initial_position_scale"):
        assert large[key] == pytest.approx(9 * small[key])
    for key in ("kf_process_position_scale", "kf_process_velocity_scale", "kf_initial_velocity_scale"):
        assert large[key] == small[key]


@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_first_detection_after_obb_angle_wrap_has_no_artificial_birth_error(monkeypatch, tmp_path, tracker):
    scales = []
    for crosses_wrap in (False, True):
        data = _data(geometry="obb")
        trajectories = []
        for track in data.tracks:
            truth, detections = track.gt_boxes.copy(), track.detection_boxes.copy()
            angles = np.linspace(-0.1, 0.15, len(truth)) + (np.pi if crosses_wrap else 0.0)
            truth[:, 4] = detections[:, 4] = (angles + np.pi) % (2 * np.pi) - np.pi
            detections[:3] = np.nan
            trajectories.append(replace(track, gt_boxes=truth, detection_boxes=detections))
        _load_fixture(monkeypatch, replace(data, tracks=tuple(trajectories), statistics={"matched": 6}))
        result = calibrate_kalman(
            _args(tmp_path, tracker=tracker, geometry="obb"), output_dir=tmp_path / str(crosses_wrap)
        )
        scales.append(load_tracker_config(tracker, result.config_path)["kf_initial_position_scale"])
    assert scales[0] == pytest.approx(scales[1])


@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_all_filter_families_produce_reusable_calibration(monkeypatch, tmp_path, tracker, geometry):
    _load_fixture(monkeypatch, _data(geometry=geometry))
    result = calibrate_kalman(
        _args(tmp_path, tracker=tracker, geometry=geometry, variable_dt=True), output_dir=tmp_path
    )
    saved = load_tracker_config(tracker, result.config_path)
    assert set(result.fitted_parameters) == set(KALMAN_NOISE_OPTIONS)
    assert all(np.isfinite(saved[key]) and saved[key] > 0 for key in KALMAN_NOISE_OPTIONS)


def test_sparse_evidence_retains_custom_baselines_and_resolves_implicit_timing(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.yaml"
    scales = dict(zip(KALMAN_NOISE_OPTIONS, [0.0001, 3.0, 2.0, 4.0, 900.0], strict=True))
    config_path.write_text(yaml.safe_dump({"tracker": "bytetrack", "variable_dt": True, **scales}))
    calls = _load_fixture(monkeypatch, _data(count=1, tracks=1))
    result = calibrate_kalman(_args(tmp_path, variable_dt=None, tracker_config=config_path), output_dir=tmp_path)
    saved = load_tracker_config("bytetrack", result.config_path)
    assert {key: saved[key] for key in KALMAN_NOISE_OPTIONS} == scales
    assert calls[0].variable_dt is True
    assert saved["kf_time_unit"] == "seconds"
    assert result.fitted_parameters == ()
    report = json.loads(result.report_path.read_text())
    assert all(entry["status"] == "retained" and entry["reason"] for entry in report["parameters"].values())


def test_detection_misses_contribute_process_evidence_but_annotation_gaps_do_not(monkeypatch, tmp_path):
    data = _data(tracks=1)
    track = data.tracks[0]
    detections = track.detection_boxes.copy()
    detections[1:3] = np.nan
    _load_fixture(
        monkeypatch, replace(data, tracks=(replace(track, detection_boxes=detections),), statistics={"matched": 4})
    )
    result = calibrate_kalman(_args(tmp_path), output_dir=tmp_path / "misses")
    assert result.gt_transitions == 4
    frames = np.array([0, 1, 2, 4, 5, 6])
    _load_fixture(monkeypatch, replace(data, tracks=(replace(track, frame_indices=frames),)))
    result = calibrate_kalman(_args(tmp_path), output_dir=tmp_path / "annotation-gap")
    assert result.gt_transitions == 2


@pytest.mark.parametrize("mode,override", [("seconds", False), ("frames", True)])
def test_calibrated_units_cannot_be_overridden(monkeypatch, tmp_path, mode, override):
    path = tmp_path / "calibrated.yaml"
    path.write_text(yaml.safe_dump({"variable_dt": mode == "seconds", "kf_time_unit": mode, "kf_reference_dt_s": 0.04}))
    calls = _load_fixture(monkeypatch, _data())
    with pytest.raises(ValueError, match="conflicts with variable_dt"):
        calibrate_kalman(_args(tmp_path, tracker_config=path, variable_dt=override), output_dir=tmp_path)
    assert calls == []


@pytest.mark.parametrize(
    "tracker, parameter",
    [
        ("ocsort", "Q_xy_scaling"),
        ("deepocsort", "Q_s_scaling"),
        ("ocsort", "Q_a_scaling"),
        ("deepocsort", "unknown_tracker_parameter"),
    ],
)
@pytest.mark.parametrize("source", ["config", "programmatic"])
def test_unsupported_tracker_options_fail_before_calibration_reads_data_or_writes_output(
    monkeypatch, tmp_path, tracker, parameter, source
):
    calls = _load_fixture(monkeypatch, _data())
    args = _args(tmp_path, tracker=tracker)
    options = {parameter: 0.2}
    if source == "config":
        args.tracker_config = tmp_path / "tracker.yaml"
        args.tracker_config.write_text(yaml.safe_dump(options))
        options = None

    with pytest.raises(ValueError, match=parameter):
        calibrate_kalman(args, output_dir=tmp_path, tracker_options=options)

    assert calls == []
    assert not (tmp_path / "kf-tuning").exists()


def test_no_ground_truth_matches_fails_before_writing_output(monkeypatch, tmp_path):
    _load_fixture(monkeypatch, replace(_data(), statistics={"matched": 0}))
    with pytest.raises(ValueError, match="no valid matches"):
        calibrate_kalman(_args(tmp_path), output_dir=tmp_path)
    assert not (tmp_path / "kf-tuning").exists()


def test_unfiltered_calibration_records_all_resolved_sequences(monkeypatch, tmp_path):
    _load_fixture(monkeypatch, _data())
    result = calibrate_kalman(
        _args(tmp_path, sequence_names=None, seq_info={"seq1": 50, "seq2": 100}), output_dir=tmp_path
    )
    assert json.loads(result.report_path.read_text())["sequences"] == ["seq1", "seq2"]


@pytest.mark.parametrize("tracker,backend", [("sam2mot", "python"), ("botsort", "cpp")])
def test_unsupported_calibration_rejected(tracker, backend):
    with pytest.raises(ValueError, match="Python tracker with a Kalman filter"):
        validate_kf_calibration(tracker, backend)


def test_eval_main_calibrates_then_evaluates_once_with_display(monkeypatch, tmp_path):
    _load_fixture(monkeypatch, _data())
    calls = _fake_replay(monkeypatch)
    monkeypatch.setattr(evaluator, "eval_setup", lambda *args, **kwargs: None)
    args = _args(
        tmp_path, calibrate_kf=True, project=tmp_path, name="eval", experiment_id="fixture", show=True, save=True
    )
    result = evaluator.main(args)
    assert len(calls) == 1
    assert "evolve_config" not in calls[0][2]
    assert calls[0][2]["setup"] is False
    assert calls[0][0].show is True and calls[0][0].save is True
    assert result.exp_dir == tmp_path / "fixture" / "eval"
    saved = load_tracker_config("bytetrack", Path(args.tracker_config))
    assert calls[0][1] == saved
    report = json.loads((result.exp_dir / "kf-tuning" / "calibration.json").read_text())
    assert report["final_summary"] == {"HOTA": 64.0}
    assert Path(report["final_output_dir"]) == result.exp_dir


def test_eval_without_calibration_runs_once(monkeypatch, tmp_path):
    calls = []

    def replay(args, **kwargs):
        calls.append(kwargs)
        return ValidationResult("fixture", {}, "", {}, exp_dir=tmp_path, args=args)

    monkeypatch.setattr(evaluator, "run_eval", replay)
    evaluator.main(_args(tmp_path))
    assert len(calls) == 1
    assert not (tmp_path / "kf-tuning").exists()
