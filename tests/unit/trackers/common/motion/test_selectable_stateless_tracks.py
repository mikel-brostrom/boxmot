"""Selectable motion representations preserve stateless track orchestration."""

from copy import deepcopy

import numpy as np
import pytest

from boxmot.trackers.botsort.track import STrack as BotTrack
from boxmot.trackers.bytetrack.track import STrack as ByteTrack
from boxmot.trackers.common.association.strongsort import gate_cost_matrix
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model, supported_motion_model_kinds
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.strongsort.track import Track as StrongTrack
from boxmot.trackers.strongsort.tracker import _Detection

MODEL_MODES = [(kind, is_obb) for is_obb in (False, True) for kind in supported_motion_model_kinds(is_obb=is_obb)]


def _geometry(is_obb: bool, displacement: float) -> np.ndarray:
    """Produce one box translated rightward without changing its shape."""
    if is_obb:
        return np.array([60.0 + displacement, 40.0, 30.0, 20.0, 0.2])
    return np.array([45.0 + displacement, 30.0, 75.0 + displacement, 50.0])


def _box_detection(track_type, model, allocator, displacement=0.0):
    return track_type(
        np.r_[_geometry(model.is_obb, displacement), 0.9, 1, 0],
        max_obs=10,
        id_allocator=allocator,
        is_obb=model.is_obb,
        motion_model=model,
    )


def _strong_detection(is_obb: bool, displacement: float = 0.0) -> _Detection:
    geometry = _geometry(is_obb, displacement)
    if not is_obb:
        geometry[2:] -= geometry[:2]
    return _Detection(geometry, 0.9, 1, 0, np.array([1.0, 0.0]), is_obb=is_obb)


@pytest.mark.parametrize("kind,is_obb", MODEL_MODES)
@pytest.mark.parametrize("track_type", [ByteTrack, BotTrack])
def test_box_tracks_batch_selected_filter_matches_scalar_through_reactivation(kind, is_obb, track_type):
    model = create_motion_model(kind, is_obb=is_obb)
    kalman = model.create_filter()
    allocator = TrackIdAllocator()
    tracks = [_box_detection(track_type, model, allocator, displacement=index * 80.0) for index in range(3)]
    for track in tracks:
        track.activate(kalman, frame_id=1)
        if kind is MotionModelKind.XYSCR:
            assert track.mean[3] == pytest.approx(0.9)
            if is_obb:
                assert track.mean.shape == (11,)
                assert track.mean[5] == pytest.approx(0.2)
    scalar = deepcopy(tracks)
    for frame_index in range(2, 6):
        detections = [
            _box_detection(track_type, model, allocator, displacement=index * 80.0 + frame_index) for index in range(3)
        ]
        if frame_index == 4:
            for track in tracks + scalar:
                track.mark_lost()
        track_type.multi_predict(tracks, dt=0.3)
        for track in scalar:
            track.predict(dt=0.3)
        track_type.multi_update(zip(tracks, detections), frame_index, reactivate=frame_index == 4)
        for expected, detection in zip(scalar, detections):
            if frame_index == 4:
                expected.re_activate(detection, frame_index)
            else:
                expected.update(detection, frame_index)
        for actual, expected in zip(tracks, scalar):
            assert actual.mean.shape == (model.dim_x,)
            assert actual.covariance.shape == (model.dim_x, model.dim_x)
            np.testing.assert_allclose(actual.mean, expected.mean, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(actual.covariance, expected.covariance, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(actual.history_observations, expected.history_observations)
            assert (actual.id, actual.state, actual.frame_id) == (expected.id, expected.state, expected.frame_id)


@pytest.mark.parametrize("kind,is_obb", MODEL_MODES)
def test_strongsort_selected_filter_batches_gates_and_compensates_camera(kind, is_obb):
    model = create_motion_model(kind, is_obb=is_obb)
    tracks = [
        StrongTrack(
            _strong_detection(is_obb, index * 80.0),
            index + 1,
            n_init=1,
            max_age=30,
            max_obs=10,
            is_obb=is_obb,
            motion_model=model,
        )
        for index in range(3)
    ]
    scalar = deepcopy(tracks)
    observations = [_strong_detection(is_obb, index * 80.0 + 2.0) for index in range(3)]
    for _ in range(3):
        StrongTrack.multi_predict(tracks, dt=0.2)
        StrongTrack.multi_update(zip(tracks, observations))
        for actual, expected, detection in zip(tracks, scalar, observations):
            expected.predict(dt=0.2)
            expected.update(detection)
            np.testing.assert_allclose(actual.mean, expected.mean, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(actual.covariance, expected.covariance, rtol=1e-10, atol=1e-10)
            assert (actual.age, actual.hits) == (expected.age, expected.hits)
    blocked = np.zeros((3, 3), dtype=bool)
    costs = gate_cost_matrix(
        np.zeros((3, 3)), tracks, observations, list(range(3)), list(range(3)), 0.5, gated_mask=blocked
    )
    assert np.isfinite(costs).all()
    assert not np.diag(blocked).any()
    assert blocked[0, 2]
    warp = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]])
    before = [track.mean.copy() for track in tracks]
    StrongTrack.multi_camera_update(tracks, warp)
    for track, original in zip(tracks, before):
        np.testing.assert_allclose(track.mean[:2], original[:2] + [7.0, -3.0], atol=1e-5)
        np.testing.assert_allclose(track.mean[2:], original[2:], atol=1e-5)
        assert np.linalg.eigvalsh(track.covariance).min() >= -1e-7


@pytest.mark.parametrize("kind,is_obb", MODEL_MODES)
def test_botsort_selected_filter_camera_translation_preserves_size_and_score(kind, is_obb):
    model = create_motion_model(kind, is_obb=is_obb)
    track = _box_detection(BotTrack, model, TrackIdAllocator())
    track.activate(model.create_filter(), frame_id=1)
    before = track.mean.copy()
    transform = BotTrack.multi_gmc_obb if is_obb else BotTrack.multi_gmc
    transform([track], np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]]))
    np.testing.assert_allclose(track.mean[:2], before[:2] + [7.0, -3.0], atol=1e-5)
    np.testing.assert_allclose(track.mean[2:], before[2:], atol=1e-5)
    assert np.linalg.eigvalsh(track.covariance).min() >= -1e-7


@pytest.mark.parametrize("kind,is_obb", MODEL_MODES)
@pytest.mark.parametrize("name", ["bytetrack", "botsort", "strongsort"])
@pytest.mark.parametrize("per_class", [False, True])
def test_public_trackers_select_filter_and_preserve_identity_through_gap(kind, is_obb, name, per_class):
    """Registry options reach tracks, canonical output, timing and class splits."""
    from boxmot.structures import Tracks
    from boxmot.trackers import TrackerSpec, create_tracker
    from tests.unit.trackers.test_variable_frame_time import _update

    tracker = create_tracker(
        TrackerSpec(
            name,
            geometry="obb" if is_obb else "aabb",
            per_class=per_class,
            options=(("kalman.type", kind.value), ("kalman.variable_dt", True)),
        )
    )
    # Supply embeddings through canonical detections and isolate KF selection
    # from estimation of camera motion on the synthetic blank test frame.
    if hasattr(tracker, "cmc"):
        tracker.cmc = None
    rows = np.array([np.r_[_geometry(is_obb, index * 80.0), 0.95, index] for index in range(2)])
    for index in range(4):
        result = _update(tracker, rows, index=index, timestamp_s=10.0 + index * 0.1)
    assert isinstance(result, Tracks)
    assert result.is_obb is is_obb
    assert len(result) == 2
    track_ids = result.track_ids.clone()
    assert all(track.motion_model.kind is kind for track in tracker.active_tracks)
    assert all(track.mean.shape == (track.motion_model.dim_x,) for track in tracker.active_tracks)
    if is_obb:
        np.testing.assert_allclose(result.geometry.values[:, 4], rows[:, 4], atol=1e-5)
    if kind is MotionModelKind.XYSCR:
        for track in tracker.active_tracks:
            assert track.mean[3] == pytest.approx(0.95)
            if is_obb:
                assert track.mean.shape == (11,)
                assert track.mean[5] == pytest.approx(0.2)
    _update(tracker, np.empty((0, 7 if is_obb else 6)), index=4, timestamp_s=10.55)
    result = _update(tracker, rows, index=5, timestamp_s=10.6)
    np.testing.assert_array_equal(result.track_ids, track_ids)
    tracker.reset()
    assert tracker.motion_model.kind is kind


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("name", ["bytetrack", "botsort", "strongsort"])
def test_public_tracker_defaults_match_explicit_filter(name, is_obb):
    """Omitted, automatic and explicit defaults retain geometry and identity."""
    from boxmot.trackers import TrackerSpec, create_tracker
    from tests.unit.trackers.test_variable_frame_time import _update

    default_kind = "xywh" if name == "botsort" or is_obb else "xyah"
    trackers = [
        create_tracker(TrackerSpec(name, geometry="obb" if is_obb else "aabb", options=options))
        for options in ((), (("kalman.type", None),), (("kalman.type", default_kind),))
    ]
    for tracker in trackers:
        if hasattr(tracker, "cmc"):
            tracker.cmc = None
    for index in range(8):
        geometry = _geometry(is_obb, index * 0.25)
        if is_obb and index % 2:
            geometry[2:4] = geometry[2:4][::-1]
            geometry[4] += np.pi / 2.0
        rows = np.empty((0, 7 if is_obb else 6)) if index in (4, 5) else np.array([np.r_[geometry, 0.9, 1]])
        outputs = [_update(tracker, rows, index=index) for tracker in trackers]
        actual = outputs[0]
        for tracker, expected in zip(trackers[1:], outputs[1:]):
            np.testing.assert_array_equal(
                actual.to_obb_rows() if is_obb else actual.to_aabb_rows(),
                expected.to_obb_rows() if is_obb else expected.to_aabb_rows(),
            )
            for left, right in zip(trackers[0].active_tracks, tracker.active_tracks):
                np.testing.assert_array_equal(left.mean, right.mean)
                np.testing.assert_array_equal(left.covariance, right.covariance)
        if index == 0 and name in ("bytetrack", "strongsort") and is_obb:
            for tracker in trackers:
                track = tracker.active_tracks[0]
                assert track.motion_model.kind is MotionModelKind.XYWH
                assert track.mean.shape == (10,)
                assert track.covariance.shape == (10, 10)
                np.testing.assert_allclose(track.mean[:5], [60.0, 40.0, 30.0, 20.0, 0.2])
                np.testing.assert_array_equal(track.mean[5:], np.zeros(5))
        if index == 3:
            confirmed_ids = actual.track_ids.clone()
            assert len(confirmed_ids) == 1
        if index >= 6:
            np.testing.assert_array_equal(actual.track_ids, confirmed_ids)
        if is_obb and len(actual):
            np.testing.assert_allclose(actual.geometry.values[:, 4], 0.2, atol=1e-5)


@pytest.mark.parametrize("kind,is_obb", MODEL_MODES)
@pytest.mark.parametrize("name", ["bytetrack", "botsort", "strongsort"])
def test_selected_filter_preserves_class_specific_noise_owners(kind, is_obb, name):
    """Matrix-state choices must use their factories for per-class filters."""
    from boxmot import KalmanConfig, KalmanNoiseConfig
    from tests.unit.trackers.test_kalman_noise_config import _tracker
    from tests.unit.trackers.test_variable_frame_time import _state, _update

    tracker = _tracker(
        name,
        "obb" if is_obb else "aabb",
        per_class=True,
        kalman=KalmanConfig(
            type=kind.value,
            variable_dt=True,
            noise=KalmanNoiseConfig(by_class={0: KalmanNoiseConfig(measurement_noise_scale=5.0)}),
        ),
    )
    if hasattr(tracker, "cmc"):
        tracker.cmc = None
    pooled = tracker.kalman_noise_config
    pooled_filter = getattr(tracker, "kalman_filter", None)
    rows = np.array([np.r_[_geometry(is_obb, index * 80.0), 0.95, index] for index in range(2)])
    for index in range(2):
        _update(tracker, rows, index=index, timestamp_s=index * 0.1)
        filters = []
        for class_id in (0, 1):
            track = tracker.get_class_tracks(class_id)[0]
            kalman = _state(track)[1]
            filters.append(kalman)
            assert track.motion_model.kind is kind
            assert kalman.dim_x == track.motion_model.dim_x
            assert kalman.noise_config is pooled.for_class(class_id)
        assert filters[0] is not filters[1]
        assert tracker.kalman_noise_config is pooled
        assert getattr(tracker, "kalman_filter", None) is pooled_filter
