"""Joint policy selection fits and scores each KF under its actual camera motion."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from boxmot.engine.calibration import kalman_replay
from boxmot.engine.calibration import kalman_selection as selection
from boxmot.engine.calibration.kalman_data import CalibrationData, CalibrationTrack
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS
from boxmot.trackers.common.motion.models import MotionModelKind, supported_motion_model_kinds


def _track(sequence: str = "s0", *, geometry: str = "aabb") -> CalibrationTrack:
    """Render a fixed object viewed by an irregularly translating camera."""
    translations = np.array([0, 4, -3, 7, -2, 9, -5, 3, 1], dtype=float)
    positions = np.cumsum(translations)
    boxes = np.tile([20.0, 30.0, 50.0, 100.0], (len(positions), 1))
    boxes[:, [0, 2]] += positions[:, None]
    if geometry == "obb":
        boxes = np.column_stack(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2], positions * 0))
    return CalibrationTrack(
        sequence,
        1,
        0,
        np.arange(len(positions)),
        np.cumsum(np.linspace(0.02, 0.08, len(positions))),
        boxes,
        boxes.copy(),
        np.full(len(positions), 0.8),
    )


def _warps(track: CalibrationTrack, *, moving: bool = False) -> np.ndarray:
    """Make previous-to-current transforms indexed by delivered frame."""
    warps = np.tile(np.eye(2, 3), (int(track.frame_indices[-1]) + 1, 1, 1))
    if moving:
        warps[1:, 0, 2] = np.diff(track.gt_boxes[:, 0])
    return warps


def _config(kind: str = "xyah", *, timed: bool = False) -> dict:
    return load_tracker_config("bytetrack", None, {"kalman.type": kind, "kalman.variable_dt": timed})


def _select(tracks: tuple[CalibrationTrack, ...], cameras: dict, *, configs: dict | None = None) -> dict:
    return selection.select_kalman_model(
        CalibrationData(tracks, {}, ()),
        tracker_name="bytetrack",
        geometry="aabb",
        configs={"xyah": _config()} if configs is None else configs,
        baseline_type="xyah",
        camera_candidates=cameras,
    )


@pytest.mark.parametrize(
    "geometry,kind",
    [
        (geometry, kind.value)
        for geometry in ("aabb", "obb")
        for kind in supported_motion_model_kinds(is_obb=geometry == "obb")
    ],
)
@pytest.mark.parametrize("timed", [False, True])
def test_identity_camera_replay_preserves_existing_scores(kind, geometry, timed):
    track = _track(geometry=geometry)
    kwargs = {"tracker_name": "bytetrack", "geometry": geometry, "config": _config(kind, timed=timed)}
    off = kalman_replay.replay_track(track, **kwargs)
    identity = kalman_replay.replay_track(track, **kwargs, camera_warps=_warps(track))
    for horizon in off:
        assert identity[horizon] == pytest.approx(off[horizon], abs=1e-12)


@pytest.mark.parametrize(
    "geometry,kind",
    [
        (geometry, kind.value)
        for geometry in ("aabb", "obb")
        for kind in supported_motion_model_kinds(is_obb=geometry == "obb")
    ],
)
def test_irregular_camera_motion_is_applied_at_every_detection_gap_step(kind, geometry):
    track = _track(geometry=geometry)
    kwargs = {"tracker_name": "bytetrack", "geometry": geometry, "config": _config(kind)}
    off = kalman_replay.replay_track(track, **kwargs)
    compensated = kalman_replay.replay_track(track, **kwargs, camera_warps=_warps(track, moving=True))
    for horizon in (1, 3, 5):
        # Size conversion and CMC's numerical Jacobian introduce small roundoff.
        assert compensated[horizon]["mean_iou"] == pytest.approx(1.0, abs=1e-6)
        assert compensated[horizon]["normalized_center_error"] == pytest.approx(0.0, abs=1e-8)
        assert off[horizon]["mean_iou"] < 0.9


def test_camera_replay_never_uses_ground_truth_or_future_detections_for_state(monkeypatch):
    track = _track()
    warps = _warps(track, moving=True)

    def capture(current: CalibrationTrack) -> np.ndarray:
        predicted = []

        def metrics(boxes, truth, **kwargs):
            predicted.extend(boxes.copy())
            return np.zeros((len(boxes), 2))

        monkeypatch.setattr(kalman_replay, "batch_box_metrics", metrics)
        kalman_replay.replay_track(
            current, tracker_name="bytetrack", geometry="aabb", config=_config(), camera_warps=warps
        )
        return np.asarray(predicted)

    original = capture(track)
    np.testing.assert_array_equal(capture(replace(track, gt_boxes=track.gt_boxes + 100)), original)
    changed = track.detection_boxes.copy()
    changed[1] += 10
    altered = capture(replace(track, detection_boxes=changed))
    # Each observation origin emits horizons 1, 3 and 5 consecutively.
    np.testing.assert_array_equal(altered[:3], original[:3])
    assert not np.allclose(altered[3:6], original[3:6])


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("missing", "no transforms for sequence"),
        ("short", "cover delivered frame"),
        ("shape", "shape"),
        ("nan", "finite"),
        ("inf", "finite"),
        ("singular", "nonsingular"),
    ],
)
def test_invalid_camera_inputs_are_rejected_before_candidate_selection(corruption, error):
    tracks = (_track("s0"), _track("s1"))
    warps = {track.sequence_id: _warps(track) for track in tracks}
    if corruption == "missing":
        del warps["s1"]
    elif corruption == "short":
        warps["s1"] = warps["s1"][:-1]
    elif corruption == "shape":
        warps["s1"] = np.zeros((len(warps["s1"]), 3, 3))
    elif corruption == "singular":
        warps["s1"][1] = 0
    else:
        warps["s1"][1, 0, 2] = np.nan if corruption == "nan" else np.inf
    with pytest.raises(ValueError, match=error):
        _select(tracks, {"off": None, "auto": warps})


def test_joint_selection_ranks_only_one_frame_priors_and_prefers_off_on_ties(monkeypatch):
    tracks = (_track("s0"), _track("s1"))
    warps = {track.sequence_id: _warps(track, moving=True) for track in tracks}

    def fit(training, *, baseline, **kwargs):
        return dict(baseline), {}

    def replay(track, *, camera_warps=None, config, horizons, **kwargs):
        one_frame = 0.8 if config["kalman.type"] == "xyah" else 0.7
        longer = 0.1 if camera_warps is None else 0.99
        return {
            horizon: {"mean_iou": one_frame if horizon == 1 else longer, "normalized_center_error": 0, "forecasts": 1}
            for horizon in horizons
        }

    monkeypatch.setattr(selection, "_fit_fold", fit)
    monkeypatch.setattr(kalman_replay, "replay_track", replay)
    report = _select(tracks, {"auto": warps, "off": None}, configs={"xywh": _config("xywh"), "xyah": _config()})
    assert report["selected_cmc"] == "off"
    assert report["selected_type"] == "xyah"
    assert report["selected_profile"] == "default"
    assert report["objective"] == "maximize_one_frame_mean_iou"
    assert report["primary_horizon"] == 1
    assert report["diagnostic_horizons"] == [3, 5]
    assert report["candidates"]["xyah"]["mean_iou"] == 0.8
    assert report["horizons"] == [1]
    assert set(report["candidates"]["xyah"]["horizons"]) == {"1"}
    assert report["cmc_candidates"]["auto"]["diagnostics"]["horizons"]["5"]["mean_iou"] == 0.99
    assert report["cmc_candidates"]["auto"]["folds"] == report["cmc_candidates"]["off"]["folds"]
    assert "not future-image forecasts" in report["horizon_semantics"]
    json.dumps(report, allow_nan=False)


def test_training_fits_receive_actual_policy_and_only_training_identities(monkeypatch):
    tracks = tuple(_track(f"s{index}") for index in range(3))
    warps = {track.sequence_id: _warps(track, moving=True) for track in tracks}
    observed = []

    def fit(training, models, baseline, *, camera_warps=None, **kwargs):
        observed.append(({track.sequence_id for track in training}, camera_warps))
        return {name: {"value": baseline[name]} for name in KALMAN_NOISE_OPTIONS}, {}

    monkeypatch.setattr(selection, "fit_kalman_noise", fit)
    report = _select(tracks, {"off": None, "auto": warps})
    assert report["selected_cmc"] == "auto"
    assert len(observed) == 6
    for number, (training, camera_warps) in enumerate(observed):
        fold = report["folds"][number % 3]
        assert training == set(fold["training_sequences"])
        assert training.isdisjoint(fold["validation_sequences"])
        if number < 3:
            assert camera_warps is None
        else:
            for sequence_id in training:
                np.testing.assert_array_equal(camera_warps[sequence_id], warps[sequence_id])


def test_joint_selection_can_choose_partial_noise_recipe_under_camera_policy(monkeypatch):
    tracks = (_track("s0"), _track("s1"))
    warps = {track.sequence_id: _warps(track, moving=True) for track in tracks}

    def fit(training, *, baseline, **kwargs):
        return {
            **baseline,
            "kalman.noise.process_position_scale": 0.2,
            "kalman.noise.measurement_noise_scale": 0.3,
        }, {}

    def replay(track, *, camera_warps=None, config, horizons, **kwargs):
        motion = config["kalman.noise.process_position_scale"] == 0.2
        measurement = config["kalman.noise.measurement_noise_scale"] == 0.3
        primary = 0.9 if camera_warps is not None and motion and not measurement else 0.7
        longer = 1.0 if motion and measurement else 0.5
        return {
            horizon: {"mean_iou": primary if horizon == 1 else longer, "normalized_center_error": 0, "forecasts": 1}
            for horizon in horizons
        }

    monkeypatch.setattr(selection, "_fit_fold", fit)
    monkeypatch.setattr(kalman_replay, "replay_track", replay)
    report = _select(tracks, {"off": None, "auto": warps})
    assert report["selected_cmc"] == "auto"
    assert report["selected_profile"] == "motion"
    assert report["candidates"]["xyah"]["mean_iou"] == 0.9


def test_joint_identity_policy_has_same_fitted_noise_as_off():
    tracks = (_track("s0"), _track("s1"))
    report = _select(tracks, {"identity": {track.sequence_id: _warps(track) for track in tracks}, "off": None})
    assert report["selected_cmc"] == "off"
    off = report["cmc_candidates"]["off"]["candidates"]["xyah"]
    identity = report["cmc_candidates"]["identity"]["candidates"]["xyah"]
    assert identity["mean_iou"] == pytest.approx(off["mean_iou"], abs=1e-12)
    for identity_fold, off_fold in zip(
        identity["profiles"]["calibrated"]["folds"], off["profiles"]["calibrated"]["folds"], strict=True
    ):
        for name, estimate in off_fold["calibration"]["parameters"].items():
            assert identity_fold["calibration"]["parameters"][name]["value"] == pytest.approx(estimate["value"])


def test_all_joint_candidates_fit_and_replay_under_actual_camera_behavior(monkeypatch):
    tracks = tuple(_track(f"s{index}") for index in range(3))
    calls, fits = [], []
    original_replay, original_fit = kalman_replay.replay_track, selection._fit_fold

    def replay(track, **kwargs):
        calls.append((kwargs["config"]["kalman.type"], kwargs["horizons"]))
        return original_replay(track, **kwargs)

    def fit(*args, **kwargs):
        fits.append(kwargs["baseline"]["kalman.type"])
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(kalman_replay, "replay_track", replay)
    monkeypatch.setattr(selection, "_fit_fold", fit)
    report = _select(
        tracks,
        {"off": None, "auto": {track.sequence_id: _warps(track, moving=True) for track in tracks}},
        configs={kind.value: _config(kind.value) for kind in MotionModelKind},
    )
    assert report["selected_cmc"] == "auto"
    assert report["candidates"][report["selected_type"]]["mean_iou"] == pytest.approx(1.0, abs=1e-6)
    for policy in report["cmc_candidates"].values():
        assert set(policy["candidates"]) == {kind.value for kind in MotionModelKind}
        for candidate in policy["candidates"].values():
            assert candidate["status"] == "valid", candidate
            assert len(candidate["profiles"]) == 4
            for profile in candidate["profiles"].values():
                assert profile["status"] == "valid", profile
                assert set(profile["horizons"]) == {"1"}
                assert profile["mean_iou"] == profile["horizons"]["1"]["mean_iou"]
                assert len(profile["folds"]) == 3
        diagnostics = policy["diagnostics"]
        assert diagnostics["status"] == "valid"
        assert diagnostics["selected_type"] == policy["selected_type"]
        assert diagnostics["selected_profile"] == policy["selected_profile"]
        assert diagnostics["affects_selection"] is False
        assert diagnostics["requested_horizons"] == [1, 3, 5]
        assert set(diagnostics["horizons"]) == {"1", "3", "5"}
        assert diagnostics["horizons"]["1"]["mean_iou"] == pytest.approx(
            policy["candidates"][policy["selected_type"]]["mean_iou"]
        )
    assert sum(horizons == (1,) for _, horizons in calls) == 2 * len(MotionModelKind) * 4 * len(tracks)
    assert sum(horizons == (1, 3, 5) for _, horizons in calls) == 2 * len(tracks)
    assert len(fits) == 2 * len(MotionModelKind) * 3
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("kind", [kind.value for kind in MotionModelKind])
@pytest.mark.parametrize("timed", [False, True])
@pytest.mark.parametrize("camera", [False, True])
def test_one_frame_only_replay_matches_full_replay_primary_scores(kind, timed, camera):
    track = _track()
    kwargs = {
        "tracker_name": "bytetrack",
        "geometry": "aabb",
        "config": _config(kind, timed=timed),
        "camera_warps": _warps(track, moving=True) if camera else None,
    }
    full = kalman_replay.replay_track(track, **kwargs)
    primary = kalman_replay.replay_track(track, **kwargs, horizons=(1,))
    assert set(primary) == {1}
    assert primary[1] == pytest.approx(full[1], abs=1e-12)


def test_long_horizon_divergence_cannot_invalidate_winning_one_frame_candidate(monkeypatch):
    tracks = (_track("s0"), _track("s1"))
    warps = {track.sequence_id: _warps(track, moving=True) for track in tracks}
    original = kalman_replay.iter_forecasts

    def fail_long_only(**kwargs):
        for record in original(**kwargs):
            if kwargs["camera_warps"] is not None and record[0] == 3:
                raise FloatingPointError("Synthetic long-horizon divergence")
            yield record

    monkeypatch.setattr(kalman_replay, "iter_forecasts", fail_long_only)
    report = _select(tracks, {"off": None, "auto": warps})
    assert report["selected_cmc"] == "auto"
    assert report["status"] == "valid"
    assert report["candidates"]["xyah"]["status"] == "valid"
    assert report["candidates"]["xyah"]["mean_iou"] == pytest.approx(1.0)
    assert all(row["status"] == "valid" for row in report["candidates"]["xyah"]["profiles"].values())
    assert report["diagnostics"]["status"] == "invalid"
    assert report["diagnostics"]["affects_selection"] is False
    assert report["diagnostics"]["horizons"] == {}
    assert "Synthetic long-horizon divergence" in report["diagnostics"]["error"]
    assert report["cmc_candidates"]["off"]["diagnostics"]["status"] == "valid"
    json.dumps(report, allow_nan=False)
