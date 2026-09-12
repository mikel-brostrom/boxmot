"""Compare real tracker association/lifecycle trajectories to scalar KF calls."""

from collections import deque
from importlib import import_module

import numpy as np
import pytest

from boxmot.trackers.common.track_state import BoxTrack
from boxmot.trackers.strongsort.track import Track as StrongTrack
from tests.unit.trackers.test_association_switcher import TRACKER_FACTORIES
from tests.unit.trackers.test_variable_frame_time import KALMAN_TRACKERS, _state, _update


def _predict_scalar(tracks, *, dt=None):
    return [track.predict(dt=dt) for track in tracks]


def _update_scalar(tracks, *arguments, **keywords):
    for index, track in enumerate(tracks):
        track.update(
            *(column[index] for column in arguments), **{key: column[index] for key, column in keywords.items()}
        )


def _box_update_scalar(cls, pairs, frame_id, *, reactivate=False, new_id=False):
    for track, detection in pairs:
        if reactivate:
            track.re_activate(detection, frame_id, new_id=new_id)
        else:
            track.update(detection, frame_id)


def _use_scalar_kalman(monkeypatch):
    """Switch only numerical batching off, retaining the same tracker code."""
    for name in ("ocsort", "deepocsort", "hybridsort", "boosttrack", "occluboost"):
        module = import_module(f"boxmot.trackers.{name}.tracker")
        monkeypatch.setattr(module, "predict_tracks", _predict_scalar)
        if hasattr(module, "update_tracks"):
            monkeypatch.setattr(module, "update_tracks", _update_scalar)
    monkeypatch.setattr(
        BoxTrack, "multi_predict", classmethod(lambda cls, tracks, *, dt=None: _predict_scalar(tracks, dt=dt))
    )
    monkeypatch.setattr(BoxTrack, "multi_update", classmethod(_box_update_scalar))
    monkeypatch.setattr(
        StrongTrack, "multi_predict", classmethod(lambda cls, tracks, *, dt=None: _predict_scalar(tracks, dt=dt))
    )
    monkeypatch.setattr(
        StrongTrack,
        "multi_update",
        classmethod(lambda cls, pairs: [track.update(detection) for track, detection in pairs]),
    )


def _assert_matching_state(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_matching_state(actual[key], expected[key])
    elif isinstance(actual, (list, tuple, deque)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_matching_state(left, right)
    elif isinstance(actual, str):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def _trajectory(is_obb):
    """Seeded motion includes gaps, low scores, new tracks, and OBB swaps."""
    rng = np.random.default_rng(834)
    for frame_index in range(24):
        rows = []
        for index in range(6):
            if frame_index in (13, 14) or (index == 0 and frame_index in (7, 8, 9)):
                continue
            if index == 5 and 10 <= frame_index < 18:
                continue
            center = np.array([25.0 + 40.0 * index, 40.0 + 45.0 * (index % 2)])
            center += np.array([0.55, 0.13 * (index + 1)]) * frame_index + rng.normal(0.0, 0.05, 2)
            if index == 5 and frame_index >= 18:
                center[1] += 50.0
            size = np.array([24.0, 16.0]) + rng.normal(0.0, 0.04, 2)
            score = 0.15 if index == 1 and frame_index in (5, 11, 16) else 0.85 + rng.uniform(0.0, 0.1)
            if is_obb:
                angle = 3.1 + frame_index * 0.015
                if frame_index % 3 == 2:
                    size = size[::-1]
                    angle += np.pi / 2.0
                geometry = [*center, *size, (angle + np.pi) % (2.0 * np.pi) - np.pi]
            else:
                geometry = [*(center - size / 2.0), *(center + size / 2.0)]
            rows.append([*geometry, score, index % 2])
        yield np.asarray(rows, dtype=np.float32).reshape(-1, 7 if is_obb else 6)


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("timed", [False, True])
def test_batch_tracker_trajectory_matches_scalar(name, is_obb, timed, monkeypatch):
    options = dict(
        is_obb=is_obb,
        variable_dt=timed,
        max_age=6,
        kf_process_position_scale=1.3,
        kf_process_velocity_scale=0.7,
        kf_measurement_noise_scale=1.8,
    )
    if name in ("boosttrack", "occluboost"):
        options["adaptive_kf"] = True
    actual, expected = [TRACKER_FACTORIES[name](**options) for _ in range(2)]
    # Camera estimation and embedding inference are outside this comparison.
    for tracker in (actual, expected):
        monkeypatch.setattr(tracker, "apply_cmc", lambda *args, **kwargs: None)
    timestamp = 0.0
    for frame_index, rows in enumerate(_trajectory(is_obb)):
        timestamp += (0.03, 0.05, 0.08)[frame_index % 3]
        kwargs = dict(index=frame_index, timestamp_s=timestamp if timed else None)
        output = _update(actual, rows.copy(), **kwargs)
        with monkeypatch.context() as scalar_patch:
            _use_scalar_kalman(scalar_patch)
            reference = _update(expected, rows.copy(), **kwargs)
        for attribute in ("track_ids", "scores", "class_ids", "detection_indices"):
            _assert_matching_state(getattr(output, attribute), getattr(reference, attribute))
        _assert_matching_state(output.geometry.values, reference.geometry.values)
        actual_tracks = sorted(actual.active_tracks, key=lambda track: track.id)
        expected_tracks = sorted(expected.active_tracks, key=lambda track: track.id)
        assert [track.id for track in actual_tracks] == [track.id for track in expected_tracks]
        for track, scalar_track in zip(actual_tracks, expected_tracks):
            mean, kalman = _state(track)
            scalar_mean, scalar_kalman = _state(scalar_track)
            _assert_matching_state(mean, scalar_mean)
            _assert_matching_state(
                getattr(track, "covariance", kalman.P), getattr(scalar_track, "covariance", scalar_kalman.P)
            )
            for attribute in (
                "age",
                "hits",
                "hit_streak",
                "time_since_update",
                "conf",
                "cls",
                "det_ind",
                "state",
                "frame_id",
                "tracklet_len",
                "is_activated",
                "history_observations",
                "last_observation",
                "observations",
                "velocity",
                "features",
                "smooth_feat",
            ):
                if hasattr(track, attribute):
                    _assert_matching_state(getattr(track, attribute), getattr(scalar_track, attribute))
            for attribute in ("x_prior", "P_prior", "x_post", "P_post", "R", "Q"):
                if hasattr(kalman, attribute):
                    _assert_matching_state(getattr(kalman, attribute), getattr(scalar_kalman, attribute))
