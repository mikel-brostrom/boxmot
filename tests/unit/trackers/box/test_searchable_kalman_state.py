"""Selectable motion representations preserve tracker-specific behavior."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from boxmot import KalmanConfig
from boxmot.trackers.common.motion.models import MotionModelKind, supported_motion_model_kinds
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.hybridsort.track import KalmanBoxTracker as HybridTrack
from boxmot.trackers.ocsort.track import KalmanBoxTracker as OcTrack
from tests.unit.trackers.test_association_switcher import TRACKER_FACTORIES
from tests.unit.trackers.test_variable_frame_time import _update

TRACKERS = ("ocsort", "deepocsort", "hybridsort", "boosttrack", "occluboost")
MODELS = tuple(kind.value for kind in MotionModelKind)
GEOMETRY_MODELS = [
    (is_obb, kind.value)
    for is_obb in (False, True)
    for kind in supported_motion_model_kinds(is_obb=is_obb)
]


def _tracks(tracker):
    """Include retained missing tracks when a tracker exposes emitted tracks separately."""
    return tracker.trackers if hasattr(tracker, "trackers") else tracker.active_tracks


def _rows(frame: int, *, is_obb: bool) -> np.ndarray:
    """Move two separated detections without making association ambiguous."""
    centers = np.array([[60.0, 60.0], [190.0, 100.0]]) + [0.3 * frame, 0.1 * frame]
    sizes = np.tile([20.0, 40.0], (2, 1))
    if is_obb:
        geometry = np.column_stack((centers, sizes, np.full(2, 0.12 + frame * 0.005)))
    else:
        geometry = np.column_stack((centers - sizes / 2.0, centers + sizes / 2.0))
    return np.column_stack((geometry, np.full(2, 0.9), np.zeros(2))).astype(np.float32)


@pytest.mark.parametrize("name", TRACKERS)
@pytest.mark.parametrize("is_obb,model", GEOMETRY_MODELS)
@pytest.mark.parametrize("timed", [False, True])
def test_selected_filter_tracks_multiple_objects_through_gap(name: str, is_obb: bool, model: str, timed: bool) -> None:
    """Exercise public canonical/packed updates, batching, OBB, and timed replay."""
    tracker = TRACKER_FACTORIES[name](is_obb=is_obb, kalman=KalmanConfig(type=model, variable_dt=timed))
    timestamp = 10.0
    initial_ids = None
    for frame in range(9):
        timestamp += (0.03, 0.08, 0.04)[frame % 3]
        rows = _rows(frame, is_obb=is_obb)
        if frame in (4, 5):
            rows = rows[:0]
        packed = frame % 2 == 0
        output = _update(tracker, rows, index=frame, packed=packed, timestamp_s=timestamp if timed else None)
        tracks = _tracks(tracker)
        assert len(tracks) == 2
        assert all(track.motion_model.kind.value == model for track in tracks)
        current_ids = sorted(track.id for track in tracks)
        if initial_ids is None:
            initial_ids = current_ids
        assert current_ids == initial_ids
        for track in tracks:
            assert np.isfinite(track.kf.x).all()
            assert np.isfinite(track.kf.P).all()
            assert np.linalg.eigvalsh(track.kf.P).min() >= -1e-7
            assert track.history_observations[-1].size == (8 if is_obb else 4)
        if packed:
            assert output.shape[1] == (9 if is_obb else 8)
            assert output.dtype == np.float64
            assert output.flags.c_contiguous
        else:
            assert output.is_obb is is_obb
        if frame == 8:
            assert len(output) == 2


@pytest.mark.parametrize("name", TRACKERS)
@pytest.mark.parametrize("is_obb", [False, True])
def test_explicit_original_filter_matches_tracker_default(name: str, is_obb: bool) -> None:
    """Authored native selections must retain the original priors and trajectory."""
    native = "xyhr" if name in ("boosttrack", "occluboost") else "xysr"
    if name == "hybridsort" and not is_obb:
        native = "xyscr"
    implicit = TRACKER_FACTORIES[name](is_obb=is_obb)
    explicit = TRACKER_FACTORIES[name](is_obb=is_obb, kalman=KalmanConfig(type=native))
    for frame in range(8):
        rows = _rows(frame, is_obb=is_obb)
        if frame in (4, 5):
            rows = rows[:0]
        actual = _update(explicit, rows, index=frame, packed=True)
        expected = _update(implicit, rows, index=frame, packed=True)
        np.testing.assert_array_equal(actual, expected)
        for track, reference in zip(_tracks(explicit), _tracks(implicit), strict=True):
            np.testing.assert_array_equal(track.kf.x, reference.kf.x)
            np.testing.assert_array_equal(track.kf.P, reference.kf.P)


@pytest.mark.parametrize("name", TRACKERS)
def test_selected_oriented_score_filter_receives_detection_confidence(name: str) -> None:
    """OBB conversion must pass confidence as well as the five geometry values."""
    tracker = TRACKER_FACTORIES[name](is_obb=True, kalman=KalmanConfig(type="xyscr"))
    _update(tracker, _rows(0, is_obb=True), index=0)
    original_ids = [track.id for track in _tracks(tracker)]
    for track in _tracks(tracker):
        assert track.kf.x[3, 0] == pytest.approx(0.9)
    rows = _rows(1, is_obb=True)
    rows[:, 5] = 0.7
    _update(tracker, rows, index=1)
    assert [track.id for track in _tracks(tracker)] == original_ids
    for track in _tracks(tracker):
        assert track.kf.x[3, 0] == pytest.approx(0.7, abs=1e-3)


@pytest.mark.parametrize("is_obb,model", [(obb, model) for obb, model in GEOMETRY_MODELS if model != "xyscr"])
@pytest.mark.parametrize("timed", [False, True])
def test_hybrid_confidence_prediction_is_independent_of_box_model(model: str, timed: bool, is_obb: bool) -> None:
    """Changing box coordinates must retain HybridSORT's confidence association cue."""
    box = np.array([60.0, 50.0, 40.0, 40.0, 0.2, 0.9] if is_obb else [40.0, 30.0, 80.0, 70.0, 0.9])
    reference = HybridTrack(box, np.ones(4), id_allocator=TrackIdAllocator(), is_obb=is_obb)
    alternate = HybridTrack(box, np.ones(4), id_allocator=TrackIdAllocator(), kalman_type=model, is_obb=is_obb)
    for frame, score in enumerate((0.85, 0.8, None, None, 0.7, 0.92, 0.75)):
        interval = (0.5, 1.3, 0.8)[frame % 3] if timed else None
        _, expected_score, expected_simple = reference.predict(dt=interval)
        _, actual_score, actual_simple = alternate.predict(dt=interval)
        assert actual_score == pytest.approx(expected_score, rel=1e-12, abs=1e-12)
        assert actual_simple == expected_simple
        detection = None if score is None else np.r_[box[:-1], score]
        reference.update(detection, np.ones(4))
        alternate.update(detection, np.ones(4))


@pytest.mark.parametrize("model", MODELS)
def test_occluboost_suppresses_selected_filter_correction(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """AMS suppresses the chosen model's gain in scalar and grouped updates."""
    tracker = TRACKER_FACTORIES["occluboost"](kalman=KalmanConfig(type=model))
    _update(tracker, _rows(0, is_obb=False), index=0)
    tracks = _tracks(tracker)
    references = deepcopy(tracks)
    observations = np.column_stack((_rows(1, is_obb=False), np.arange(2)))
    for track, reference in zip(tracks, references, strict=True):
        track.predict()
        reference.predict()
    before = [track.kf.x.copy() for track in tracks]
    monkeypatch.setattr(tracker, "_compute_ams_alpha", lambda *_: 0.0)
    tracker._ams_multi_update(tracks, observations)
    for track, reference, observation, predicted in zip(tracks, references, observations, before, strict=True):
        tracker._ams_update(reference, observation)
        np.testing.assert_array_equal(track.kf.x, predicted)
        np.testing.assert_allclose(track.kf.x, reference.kf.x, atol=1e-10)
        np.testing.assert_allclose(track.kf.P, reference.kf.P, atol=1e-10)
        assert track.time_since_update == 0


@pytest.mark.parametrize("name", ["deepocsort", "hybridsort", "boosttrack", "occluboost"])
@pytest.mark.parametrize("is_obb,model", GEOMETRY_MODELS)
def test_camera_translation_preserves_selected_box_geometry(name: str, model: str, is_obb: bool) -> None:
    """CMC transforms live and missing-frame state in the selected representation."""
    tracker = TRACKER_FACTORIES[name](is_obb=is_obb, kalman=KalmanConfig(type=model))
    for frame in range(4):
        _update(tracker, _rows(frame, is_obb=is_obb), index=frame)
    _update(tracker, _rows(4, is_obb=is_obb)[:0], index=4)
    transform = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, -3.0]])
    for track in _tracks(tracker):
        before = track.motion_model.to_box(track.kf.x)[0].copy()
        track.camera_update(transform)
        displacement = [8, -3, 0, 0, 0] if is_obb else [8, -3, 8, -3]
        np.testing.assert_allclose(track.motion_model.to_box(track.kf.x)[0], before + displacement, atol=1e-5)
        assert np.isfinite(track.kf.P).all()
        assert np.linalg.eigvalsh(track.kf.P).min() >= -1e-7


@pytest.mark.parametrize("name", ["ocsort", "hybridsort"])
@pytest.mark.parametrize("is_obb,model", GEOMETRY_MODELS)
def test_selected_filter_respects_observation_history_bound(name: str, model: str, is_obb: bool) -> None:
    """Alternate representations keep the configured bounded recovery memory."""
    box = np.array([60.0, 50.0, 40.0, 40.0, 0.2, 0.9] if is_obb else [40.0, 30.0, 80.0, 70.0, 0.9])
    kwargs = dict(max_obs=3, id_allocator=TrackIdAllocator(), kalman_type=model, is_obb=is_obb)
    track = OcTrack(box, 0, 0, **kwargs) if name == "ocsort" else HybridTrack(box, np.ones(4), **kwargs)
    for frame in range(7):
        track.predict()
        observation = box if frame < 3 else None
        if name == "ocsort":
            track.update(observation, 0, 0)
        else:
            track.update(observation, np.ones(4))
    assert track.kf.max_obs == 3
    assert track.kf.history_obs.maxlen == 3
    assert len(track.kf.history_obs) <= 3
    assert len(track.kf._prediction_steps) <= 3
