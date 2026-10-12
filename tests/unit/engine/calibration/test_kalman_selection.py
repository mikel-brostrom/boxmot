"""Cross-validated KF selection uses shared histories and held-out box metrics."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from boxmot.engine.calibration import kalman_folds, kalman_replay
from boxmot.engine.calibration import kalman_selection as selection
from boxmot.engine.calibration.kalman_data import CalibrationData, CalibrationTrack
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.noise import (
    KALMAN_DIAGONAL_OPTIONS,
    KALMAN_NOISE_OPTIONS,
    KALMAN_VECTOR_OPTIONS,
)
from boxmot.trackers.common.motion.kalman_filters.profile import (
    KALMAN_CALIBRATION_PROFILES,
    apply_calibration_profile,
)
from boxmot.trackers.common.motion.models import MotionModelKind, supported_motion_model_kinds


def _track(sequence="s0", track_id=0, *, geometry="aabb", count=9, cls_id=0):
    times = np.cumsum(np.linspace(0.02, 0.08, count))
    boxes = np.tile([20.0, 30.0, 50.0, 100.0], (count, 1))
    boxes[:, [0, 2]] += 25 * times[:, None]
    boxes[:, 2] += 3 * np.arange(count)
    if geometry == "obb":
        boxes = np.column_stack(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2], times))
    detections = boxes.copy()
    detections[:, [0] if geometry == "obb" else [0, 2]] += np.sin(np.arange(count))[:, None]
    return CalibrationTrack(sequence, track_id, cls_id, np.arange(count), times, boxes, detections, np.full(count, 0.8))


def _data(*tracks):
    return CalibrationData(
        tuple(tracks), {"matched": sum(kalman_folds.matched_detections(track).sum() for track in tracks)}, ()
    )


def _configs(*kinds, **options):
    return {kind: load_tracker_config("bytetrack", None, {"kalman.type": kind, **options}) for kind in kinds}


def _select(data, configs=None, baseline="xyah", **kwargs):
    return selection.select_kalman_model(
        data,
        tracker_name="bytetrack",
        geometry="aabb",
        configs=_configs("xyah") if configs is None else configs,
        baseline_type=baseline,
        **kwargs,
    )


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("timed", [False, True])
def test_all_supported_models_use_all_heldout_forecasts_and_serialize(geometry, timed):
    data = _data(*(_track(f"s{index}", geometry=geometry) for index in range(4)))
    report = selection.select_kalman_model(
        data,
        tracker_name="bytetrack",
        geometry=geometry,
        configs=_configs(
            *(kind.value for kind in supported_motion_model_kinds(is_obb=geometry == "obb")),
            **{"kalman.variable_dt": timed},
        ),
        baseline_type="xyah",
    )
    assert len(report["folds"]) == 3
    assert report["grouping"] == "sequence"
    assert report["version"] == 3
    assert report["method"] == "grouped_cross_validated_box_profiles"
    assert set(report["candidates"]) == {kind.value for kind in supported_motion_model_kinds(is_obb=geometry == "obb")}
    for candidate in report["candidates"].values():
        assert candidate["status"] == "valid", candidate
        assert 0 <= candidate["mean_iou"] <= 1
        assert candidate["normalized_center_error"] >= 0
        assert candidate["forecasts"] == 4 * (8 + 6 + 4)
        assert candidate["trajectories"] == 4
        assert candidate["sequences"] == 4
        assert candidate["horizons"]["5"]["forecasts"] == 16
        assert tuple(candidate["profiles"]) == KALMAN_CALIBRATION_PROFILES
        assert candidate["selected_profile"] in candidate["profiles"]
        for profile in candidate["profiles"].values():
            assert profile["status"] == "valid", profile
            assert profile["forecasts"] == candidate["forecasts"]
            assert profile["trajectories"] == candidate["trajectories"]
            assert profile["sequences"] == candidate["sequences"]
            for horizon in report["horizons"]:
                assert (
                    profile["horizons"][str(horizon)]["forecasts"] == candidate["horizons"][str(horizon)]["forecasts"]
                )
        assert candidate["mean_iou"] >= candidate["profiles"]["default"]["mean_iou"] - report["tie_tolerance"]
    json.dumps(report, allow_nan=False)


def test_folds_hold_out_complete_sequences_and_training_fits_do_not_see_validation(monkeypatch):
    data = _data(*(_track(f"s{seq}", identity) for seq in range(4) for identity in range(2)))
    fit = selection.fit_kalman_noise
    observed = []

    def spy(tracks, *args, **kwargs):
        observed.append({track.sequence_id for track in tracks})
        return fit(tracks, *args, **kwargs)

    monkeypatch.setattr(selection, "fit_kalman_noise", spy)
    report = _select(data)
    for fold, training in zip(report["folds"], observed, strict=True):
        assert training == set(fold["training_sequences"])
        assert training.isdisjoint(fold["validation_sequences"])
        assert len(fold["validation_trajectories"]) == 2 * len(fold["validation_sequences"])
    replayed = {row["sequence_id"] for fold in report["folds"] for row in fold["validation_trajectories"]}
    assert replayed == {"s0", "s1", "s2", "s3"}


def test_single_sequence_splits_identities_and_keeps_class_segments_together():
    data = _data(_track(track_id=0), _track(track_id=0, cls_id=1), _track(track_id=1), _track(track_id=2))
    report = _select(data)
    assert report["grouping"] == "trajectory"
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        train_ids = {row["track_id"] for row in fold["training_trajectories"]}
        validation_ids = {row["track_id"] for row in fold["validation_trajectories"]}
        assert train_ids.isdisjoint(validation_ids)
    with pytest.raises(ValueError, match="independent sequences or trajectory identities"):
        _select(_data(_track(track_id=0), _track(track_id=0, cls_id=1)))


def test_prior_full_data_noise_and_metadata_never_leak_into_fold_fallbacks():
    data = _data(_track("s0", count=2), _track("s1", count=2))
    config = _configs("xyah", per_class=True)["xyah"]
    config.update({name: 9000.0 for name in KALMAN_NOISE_OPTIONS})
    config.update({name: [7000.0] * 4 for name in KALMAN_DIAGONAL_OPTIONS})
    config["kalman.noise.measurement_confidence_floor"] = [0.7] * 4
    config.update(
        {f"kalman.noise.by_class.0.{name.removeprefix('kalman.noise.')}": 8000.0 for name in KALMAN_NOISE_OPTIONS}
    )
    config["calibration.filter"] = "xysr"
    report = _select(data, {"xyah": config})
    for fold in report["candidates"]["xyah"]["profiles"]["calibrated"]["folds"]:
        estimates = fold["calibration"]["parameters"]
        for name in KALMAN_NOISE_OPTIONS:
            assert estimates[name]["baseline"] == 1
        assert estimates["kalman.noise.process_position_scale"]["status"] == "fixed"
        assert estimates["kalman.noise.process_position_scale"]["value"] == 1
        assert fold["calibration"]["classes"]["0"]["parameters"]["kalman.noise.process_position_scale"]["value"] == 1
        for name in KALMAN_DIAGONAL_OPTIONS:
            assert estimates[name]["baseline"] == [1.0] * 4
        assert estimates["kalman.noise.process_position_diagonal"]["value"] == [1.0] * 4
    assert config["kalman.noise.process_position_scale"] == 9000
    assert config["kalman.noise.measurement_noise_diagonal"] == [7000.0] * 4
    clean = selection.selection_baseline_config("bytetrack", "xyah", config)
    assert all(clean.get(name) in (None, [], ()) for name in KALMAN_VECTOR_OPTIONS)
    assert all(not name.startswith(("calibration.", "kalman.noise.by_class.")) for name in clean)


def test_unseen_heldout_class_uses_training_pooled_noise():
    data = _data(_track("s0", cls_id=0), _track("s1", cls_id=1))
    report = _select(data, _configs("xyah", per_class=True))
    for index, fold in enumerate(report["candidates"]["xyah"]["profiles"]["calibrated"]["folds"]):
        heldout = str(index)
        pooled = fold["calibration"]["parameters"]["kalman.noise.measurement_noise_diagonal"]
        class_estimate = fold["calibration"]["classes"][heldout]["parameters"][
            "kalman.noise.measurement_noise_diagonal"
        ]
        assert class_estimate["status"] == "global_fallback"
        assert class_estimate["value"] == pooled["value"]


def _capture_predictions(monkeypatch, track, config=None, *, tracker_name="bytetrack", geometry="aabb"):
    captured = []

    def metrics(predicted, truth, **kwargs):
        captured.append(predicted.copy())
        return 0.5, 0.1

    def batch_metrics(predicted, truth, **kwargs):
        return np.asarray([metrics(box, target) for box, target in zip(predicted, truth, strict=True)])

    monkeypatch.setattr(kalman_replay, "box_metrics", metrics)
    monkeypatch.setattr(kalman_replay, "batch_box_metrics", batch_metrics)
    kalman_replay.replay_track(
        track,
        tracker_name=tracker_name,
        geometry=geometry,
        config=load_tracker_config(tracker_name, None, {"kalman.type": "xywh"}) if config is None else config,
    )
    return np.asarray(captured)


def test_heldout_truth_never_enters_filter_state(monkeypatch):
    track = _track()
    original = _capture_predictions(monkeypatch, track)
    altered = replace(track, gt_boxes=track.gt_boxes + 500)
    np.testing.assert_array_equal(_capture_predictions(monkeypatch, altered), original)


def test_forecasts_withhold_future_detections_but_regular_history_updates(monkeypatch):
    track = _track()
    original = _capture_predictions(monkeypatch, track)
    detections = track.detection_boxes.copy()
    detections[1] += 100
    altered = _capture_predictions(monkeypatch, replace(track, detection_boxes=detections))
    # First origin's three horizons are independent of all subsequent detections.
    np.testing.assert_array_equal(altered[:3], original[:3])
    assert not np.allclose(altered[3:6], original[3:6])


def test_missing_detections_are_scored_but_never_become_forecast_origins():
    track = _track()
    detections = track.detection_boxes.copy()
    detections[1:] = np.nan
    results = kalman_replay.replay_track(
        replace(track, detection_boxes=detections),
        tracker_name="bytetrack",
        geometry="aabb",
        config=_configs("xyah")["xyah"],
    )
    assert set(results) == {1, 3, 5}
    assert all(row["forecasts"] == 1 for row in results.values())


def test_annotation_discontinuity_resets_state_and_stops_forecasts(monkeypatch):
    first, second = _track(count=3), _track(count=3)
    second = replace(second, detection_boxes=second.detection_boxes + 100, gt_boxes=second.gt_boxes + 100)
    combined = replace(
        first,
        frame_indices=np.array([0, 1, 2, 10, 11, 12]),
        timestamps_s=None,
        gt_boxes=np.concatenate((first.gt_boxes, second.gt_boxes)),
        detection_boxes=np.concatenate((first.detection_boxes, second.detection_boxes)),
        scores=np.concatenate((first.scores, second.scores)),
    )
    expected = np.concatenate((_capture_predictions(monkeypatch, first), _capture_predictions(monkeypatch, second)))
    np.testing.assert_array_equal(_capture_predictions(monkeypatch, combined), expected)
    assert len(expected) == 4


def test_measured_timestamps_change_predictions_and_are_required(monkeypatch):
    track = _track()
    config = _configs("xywh", **{"kalman.variable_dt": True})["xywh"]
    original = _capture_predictions(monkeypatch, track, config)
    times = track.timestamps_s.copy()
    times[2:] += 0.5
    altered = _capture_predictions(monkeypatch, replace(track, timestamps_s=times), config)
    assert not np.allclose(altered, original)
    with pytest.raises(ValueError, match="capture timestamps"):
        _select(_data(track, replace(track, sequence_id="s1", timestamps_s=None)), {"xywh": config})


@pytest.mark.parametrize("kind", [kind.value for kind in MotionModelKind])
def test_strongsort_observations_keep_confidence_noise_policy(monkeypatch, kind):
    track = _track()
    config = load_tracker_config("strongsort", None, {"kalman.type": kind})
    uncertain = _capture_predictions(monkeypatch, replace(track, scores=np.zeros(9)), config, tracker_name="strongsort")
    certain = _capture_predictions(monkeypatch, replace(track, scores=np.ones(9)), config, tracker_name="strongsort")
    assert not np.allclose(uncertain[3:], certain[3:])


def test_score_macro_averages_tracks_sequences_then_horizons():
    def metrics(iou, count):
        return {"mean_iou": iou, "normalized_center_error": 1 - iou, "forecasts": count}

    records = [
        (_track("s0", 0), {1: metrics(1, 10000), 3: metrics(0.2, 1)}),
        (_track("s0", 1), {1: metrics(0, 1)}),
        (_track("s1", 0), {1: metrics(0, 1)}),
    ]
    summary = kalman_replay.summarize_forecasts(records)
    assert summary["horizons"]["1"]["mean_iou"] == 0.25
    assert summary["mean_iou"] == pytest.approx((0.25 + 0.2) / 2)
    assert summary["forecasts"] == 10003


def test_ties_prefer_baseline_then_canonical_order_independent_of_input_order(monkeypatch):
    def equal(*args, **kwargs):
        return {1: {"mean_iou": 0.7, "normalized_center_error": 0.1, "forecasts": 1}}

    monkeypatch.setattr(kalman_replay, "replay_track", equal)
    tracks = [_track("s1"), _track("s0")]
    configs = _configs("xyscr", "xysr", "xywh")
    report = _select(_data(*tracks), configs, baseline="xyscr")
    assert report["selected_type"] == "xyscr"
    assert report["selected_profile"] == "default"
    assert all(row["selected_profile"] == "default" for row in report["candidates"].values())
    report = _select(_data(*tracks), configs, baseline="xyah")
    assert report["selected_type"] == "xywh"
    configs.pop("xywh")
    assert _select(_data(*tracks[::-1]), configs)["selected_type"] == "xysr"


def test_invalid_candidate_cannot_win_by_skipping_a_difficult_trajectory(monkeypatch):
    def replay(track, *, config, **kwargs):
        if config["kalman.type"] == "xyah":
            if track.sequence_id == "s1":
                raise FloatingPointError("non-finite state")
            return {1: {"mean_iou": 1.0, "normalized_center_error": 0.0, "forecasts": 1}}
        return {1: {"mean_iou": 0.5, "normalized_center_error": 0.1, "forecasts": 1}}

    monkeypatch.setattr(kalman_replay, "replay_track", replay)
    data = _data(_track("s0"), _track("s1"))
    report = _select(data, _configs("xyah", "xywh"))
    assert report["selected_type"] == "xywh"
    assert report["candidates"]["xyah"]["status"] == "invalid"
    assert report["candidates"]["xyah"]["mean_iou"] is None
    with pytest.raises(ValueError, match="could not evaluate any candidate"):
        _select(data)


@pytest.mark.parametrize(
    "geometry,kind",
    [
        (geometry, kind.value)
        for geometry in ("aabb", "obb")
        for kind in supported_motion_model_kinds(is_obb=geometry == "obb")
    ],
)
def test_harmful_calibration_retains_original_predictions(monkeypatch, geometry, kind):
    def harmful_fit(training, *, baseline, **kwargs):
        # Deliberately exaggerate measurement uncertainty so that the KF ignores
        # observations of the moving and growing object. This must lose to the
        # unchanged default filter, regardless of the family's state units.
        return {**baseline, "kalman.noise.measurement_noise_scale": 1e12}, {"parameters": {}}

    monkeypatch.setattr(selection, "_fit_fold", harmful_fit)
    report = selection.select_kalman_model(
        _data(_track("s0", geometry=geometry, count=25), _track("s1", geometry=geometry, count=25)),
        tracker_name="bytetrack",
        geometry=geometry,
        configs=_configs(kind),
        baseline_type=kind,
    )
    candidate = report["candidates"][kind]
    assert candidate["profiles"]["default"]["mean_iou"] > candidate["profiles"]["calibrated"]["mean_iou"] + 0.01
    assert candidate["selected_profile"] == report["selected_profile"] == "default"
    assert candidate["mean_iou"] == candidate["profiles"]["default"]["mean_iou"]
    assert candidate["profiles"]["default"]["forecasts"] == candidate["profiles"]["calibrated"]["forecasts"]


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_fitted_noise_improves_smooth_motion_with_noisy_observations(geometry):
    rng = np.random.default_rng(417)
    tracks = []
    for sequence in range(3):
        for identity in range(2):
            count = 30
            boxes = np.tile([100.0, 100.0, 130.0, 160.0], (count, 1))
            boxes[:, [0, 2]] += np.arange(count)[:, None] * 0.5
            detections = boxes.copy()
            center_noise = rng.normal(0, 6, (count, 2))
            detections[:, :2] += center_noise
            detections[:, 2:] += center_noise
            if geometry == "obb":
                boxes = np.column_stack(
                    ((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2], np.full(count, 0.2))
                )
                detections = np.column_stack(
                    (
                        (detections[:, :2] + detections[:, 2:]) / 2,
                        detections[:, 2:] - detections[:, :2],
                        np.full(count, 0.2),
                    )
                )
            tracks.append(
                replace(
                    _track(str(sequence), identity, geometry=geometry, count=count),
                    gt_boxes=boxes,
                    detection_boxes=detections,
                )
            )
    report = selection.select_kalman_model(
        _data(*tracks),
        tracker_name="bytetrack",
        geometry=geometry,
        configs=_configs(*(kind.value for kind in supported_motion_model_kinds(is_obb=geometry == "obb"))),
        baseline_type="xyah",
    )
    assert report["selected_profile"] != "default"
    for candidate in report["candidates"].values():
        assert candidate["selected_profile"] != "default"
        assert candidate["mean_iou"] > candidate["profiles"]["default"]["mean_iou"] + 0.05


def test_failed_calibration_does_not_disqualify_original_filter(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("unidentifiable calibration")

    monkeypatch.setattr(selection, "_fit_fold", fail)
    report = _select(_data(_track("s0"), _track("s1")))
    candidate = report["candidates"]["xyah"]
    assert candidate["status"] == "valid"
    assert candidate["selected_profile"] == report["selected_profile"] == "default"
    assert candidate["profiles"]["calibrated"]["status"] == "invalid"
    assert candidate["profiles"]["default"]["forecasts"] == 2 * (8 + 6 + 4)


@pytest.mark.parametrize("winner", ["motion", "measurement"])
def test_component_selection_reuses_one_training_fit_per_fold(monkeypatch, winner):
    fitted_groups, observed = [], []

    def fit(training, *, baseline, **kwargs):
        fitted_groups.append({track.sequence_id for track in training})
        return {
            **baseline,
            "kalman.noise.process_position_scale": 2.0,
            "kalman.noise.initial_position_scale": 3.0,
            "kalman.noise.measurement_noise_scale": 4.0,
            "kalman.noise.measurement_confidence_floor": [0.6] * 4,
        }, {"training_sequences": sorted(fitted_groups[-1])}

    def replay(track, *, config, **kwargs):
        motion = config["kalman.noise.process_position_scale"] == 2.0
        measurement = config["kalman.noise.measurement_noise_scale"] == 4.0
        assert config["kalman.noise.initial_position_scale"] == (3.0 if motion else 1.0)
        assert ("kalman.noise.measurement_confidence_floor" in config) == measurement
        profile = {
            (False, False): "default",
            (True, False): "motion",
            (False, True): "measurement",
            (True, True): "calibrated",
        }[motion, measurement]
        observed.append((profile, track.sequence_id))
        score = 0.9 if profile == winner else 0.6 if profile == "default" else 0.5
        return {1: {"mean_iou": score, "normalized_center_error": 1 - score, "forecasts": 1}}

    monkeypatch.setattr(selection, "_fit_fold", fit)
    monkeypatch.setattr(kalman_replay, "replay_track", replay)
    report = _select(_data(*(_track(f"s{index}") for index in range(4))))
    assert report["selected_profile"] == winner
    assert len(fitted_groups) == len(report["folds"]) == 3
    assert len(observed) == 4 * len(KALMAN_CALIBRATION_PROFILES)
    for fold, training in zip(report["folds"], fitted_groups, strict=True):
        assert training.isdisjoint(fold["validation_sequences"])
    profiles = report["candidates"]["xyah"]["profiles"]
    for index in range(3):
        assert (
            profiles["motion"]["folds"][index]["calibration"] == profiles["measurement"]["folds"][index]["calibration"]
        )
        assert (
            profiles["measurement"]["folds"][index]["calibration"]
            == profiles["calibrated"]["folds"][index]["calibration"]
        )


@pytest.mark.parametrize("profile", KALMAN_CALIBRATION_PROFILES)
def test_profile_recipe_preserves_other_group_and_class_fallback(profile):
    baseline = {
        **_configs("xyah")["xyah"],
        "kalman.noise.process_position_scale": 2.0,
        "kalman.noise.measurement_noise_scale": 3.0,
        "kalman.noise.measurement_confidence_floor": [0.1] * 4,
        "kalman.noise.by_class.1.process_position_scale": 11.0,
        "kalman.noise.by_class.1.measurement_noise_scale": 7.0,
        "calibration.source": "old-report",
        "track_thresh": 0.71,
    }
    fitted = {
        **_configs("xyah")["xyah"],
        "kalman.noise.process_position_scale": 13.0,
        "kalman.noise.measurement_noise_scale": 19.0,
        "kalman.noise.measurement_confidence_floor": [0.6] * 4,
        "kalman.noise.by_class.2.process_position_scale": 23.0,
        "kalman.noise.by_class.2.measurement_noise_scale": 29.0,
        "calibration.source": "fitted-report",
        "track_thresh": 0.99,
    }
    result = apply_calibration_profile(baseline, fitted, profile)
    motion, measurement = profile in ("motion", "calibrated"), profile in ("measurement", "calibrated")
    assert result["track_thresh"] == 0.71
    assert not any(key.startswith("calibration.") for key in result)
    assert result["kalman.noise.process_position_scale"] == (13.0 if motion else 2.0)
    assert result["kalman.noise.measurement_noise_scale"] == (19.0 if measurement else 3.0)
    assert result["kalman.noise.measurement_confidence_floor"] == [0.6 if measurement else 0.1] * 4
    assert result["kalman.noise.by_class.1.process_position_scale"] == (13.0 if motion else 11.0)
    assert result["kalman.noise.by_class.1.measurement_noise_scale"] == (19.0 if measurement else 7.0)
    if profile != "default":
        assert result["kalman.noise.by_class.2.process_position_scale"] == (23.0 if motion else 2.0)
        assert result["kalman.noise.by_class.2.measurement_noise_scale"] == (29.0 if measurement else 3.0)
    result["kalman.noise.measurement_confidence_floor"][0] = 0.9
    assert baseline["kalman.noise.measurement_confidence_floor"][0] == 0.1
    assert fitted["kalman.noise.measurement_confidence_floor"][0] == 0.6


def test_measurement_recipe_removes_a_stale_confidence_curve_when_fit_omits_it():
    baseline = _configs("xyah")["xyah"]
    baseline["kalman.noise.measurement_confidence_floor"] = [0.5] * 4
    fitted = _configs("xyah")["xyah"]
    result = apply_calibration_profile(baseline, fitted, "measurement")
    assert "kalman.noise.measurement_confidence_floor" not in result
    with pytest.raises(ValueError, match="Unknown Kalman calibration profile"):
        apply_calibration_profile(baseline, fitted, "unknown")
    with pytest.raises(ValueError, match="must preserve"):
        apply_calibration_profile(baseline, {**fitted, "kalman.type": "xysr"}, "calibrated")


def test_calibration_divergence_invalidates_entire_profile_not_difficult_targets(monkeypatch):
    def fit(training, *, baseline, **kwargs):
        return {**baseline, "kalman.noise.measurement_noise_scale": 2.0}, {}

    def replay(track, *, config, **kwargs):
        calibrated = config["kalman.noise.measurement_noise_scale"] == 2.0
        if calibrated and track.sequence_id == "s1":
            raise FloatingPointError("divergent difficult trajectory")
        return {1: {"mean_iou": 1.0 if calibrated else 0.5, "normalized_center_error": 0.1, "forecasts": 1}}

    monkeypatch.setattr(selection, "_fit_fold", fit)
    monkeypatch.setattr(kalman_replay, "replay_track", replay)
    report = _select(_data(_track("s0"), _track("s1")))
    assert report["selected_profile"] == "default"
    calibrated = report["candidates"]["xyah"]["profiles"]["calibrated"]
    assert calibrated["status"] == "invalid"
    assert calibrated["mean_iou"] is None
    assert len(calibrated["folds"]) == 1


def test_cross_model_ties_prefer_original_type_then_default_profile(monkeypatch):
    def fit(training, *, baseline, **kwargs):
        return {**baseline, "kalman.noise.measurement_noise_scale": 2.0}, {}

    def replay(track, *, config, **kwargs):
        calibrated = config["kalman.noise.measurement_noise_scale"] == 2.0
        score = 0.8 if calibrated or config["kalman.type"] == "xywh" else 0.5
        return {1: {"mean_iou": score, "normalized_center_error": 0.1, "forecasts": 1}}

    monkeypatch.setattr(selection, "_fit_fold", fit)
    monkeypatch.setattr(kalman_replay, "replay_track", replay)
    data, configs = _data(_track("s0"), _track("s1")), _configs("xyah", "xywh")
    original = _select(data, configs)
    assert (original["selected_type"], original["selected_profile"]) == ("xyah", "measurement")
    without_original = _select(data, configs, baseline="xysr")
    assert (without_original["selected_type"], without_original["selected_profile"]) == ("xywh", "default")


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_xyscr_geometric_profiles_match_xysr_without_calibrating_confidence(geometry):
    report = selection.select_kalman_model(
        _data(_track("s0", geometry=geometry), _track("s1", geometry=geometry)),
        tracker_name="bytetrack",
        geometry=geometry,
        configs=_configs("xysr", "xyscr"),
        baseline_type="xyscr",
    )
    candidates = report["candidates"]
    for profile in KALMAN_CALIBRATION_PROFILES:
        assert candidates["xyscr"]["profiles"][profile]["mean_iou"] == pytest.approx(
            candidates["xysr"]["profiles"][profile]["mean_iou"], abs=1e-10
        )
    for fold in candidates["xyscr"]["profiles"]["calibrated"]["folds"]:
        for name in KALMAN_DIAGONAL_OPTIONS:
            coordinates = fold["calibration"]["parameters"][name]["coordinates"]
            confidence = [item for item in coordinates if "confidence" in item["coordinate"]]
            assert confidence
            assert all(item["value"] == 1.0 and item["status"] == "excluded" for item in confidence)


def test_heldout_size_motion_can_select_a_better_filter_than_the_tracker_default():
    tracks = []
    for sequence in range(3):
        track = _track(str(sequence), count=25)
        boxes = np.tile([20.0, 30.0, 60.0, 80.0], (25, 1))
        boxes[:, 2] += 4 * np.arange(25)
        detections = boxes.copy()
        detections[:, [0, 2]] += 0.1 * np.sin(np.arange(25))[:, None]
        tracks.append(replace(track, gt_boxes=boxes, detection_boxes=detections))
    report = _select(_data(*tracks), _configs("xyah", "xywh", "xysr", "xyhr", "xyscr"), baseline="xysr")
    assert report["selected_type"] in {"xyah", "xywh", "xyhr"}
    assert report["candidates"][report["selected_type"]]["mean_iou"] > report["candidates"]["xysr"]["mean_iou"] + 0.02


def test_candidate_timing_bases_must_be_identical():
    configs = _configs("xyah", "xywh")
    configs["xywh"]["kalman.noise.reference_dt_s"] = 0.2
    with pytest.raises(ValueError, match="same timing mode and reference interval"):
        _select(_data(_track("s0"), _track("s1")), configs)


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_box_metrics_compare_common_geometry_including_equivalent_obbs(geometry):
    truth = np.array([20.0, 30.0, 40.0, 80.0])
    if geometry == "obb":
        truth = np.array([20.0, 30.0, 40.0, 80.0, 0.3])
        predicted = truth.copy()
        predicted[2:4] = predicted[3:1:-1]
        predicted[4] += np.pi / 2
    else:
        predicted = truth.copy()
    iou, error = kalman_replay.box_metrics(predicted, truth, is_obb=geometry == "obb")
    assert iou == pytest.approx(1, abs=1e-6)
    assert error == 0
    with pytest.raises(ValueError, match="non-finite"):
        kalman_replay.box_metrics(np.full_like(predicted, np.nan), truth, is_obb=geometry == "obb")


@pytest.mark.parametrize("tracker,backend", [("sfsort", "python"), ("eagermot", "python"), ("bytetrack", "cpp")])
def test_selection_rejects_unsupported_trackers(tracker, backend):
    with pytest.raises(ValueError, match="Python 2D tracker"):
        selection.validate_kf_selection(tracker, backend)


def test_insufficient_matched_prediction_support_is_actionable():
    with pytest.raises(ValueError, match="at least two usable trajectories"):
        _select(_data(_track()))
    first, second = _track("s0"), _track("s1")
    detections = second.detection_boxes.copy()
    detections[:] = np.nan
    with pytest.raises(ValueError, match="at least two usable trajectories"):
        _select(_data(first, replace(second, detection_boxes=detections)))
