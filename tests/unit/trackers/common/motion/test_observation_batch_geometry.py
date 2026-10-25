"""Observation trackers retain scalar geometry, history, and ownership policies."""

from collections import deque
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from boxmot.trackers.common.motion import batching
from boxmot.trackers.common.motion.cmc.batching import transform_prediction_histories
from boxmot.trackers.common.motion.models import create_motion_model, supported_motion_model_kinds
from boxmot.trackers.common.tracking.observations import k_previous_obs
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.deepocsort.track import DeepOBBKalmanBoxTracker
from boxmot.trackers.deepocsort.track import KalmanBoxTracker as DeepTrack
from boxmot.trackers.hybridsort.track import KalmanBoxTracker as HybridTrack
from boxmot.trackers.ocsort.track import KalmanBoxTracker as OcTrack


def _track(family: str, kind: str, is_obb: bool, index: int = 0):
    """Build independent tracks without requiring images or an encoder."""
    box = np.array([40.0 + index, 50.0, 20.0, 30.0, 3.1, 0.9] if is_obb else [30.0 + index, 35, 50, 65, 0.9])
    options = dict(kalman_type=kind, max_obs=5, id_allocator=TrackIdAllocator())
    if family == "deep":
        detection = np.r_[box, 1, index]
        if is_obb:
            return DeepOBBKalmanBoxTracker(detection, emb=np.ones(4), alpha=0.9, delta_t=3, **options)
        return DeepTrack(detection, emb=np.ones(4), **options)
    if family == "hybrid":
        return HybridTrack(box, np.ones(4), cls=1, det_ind=index, is_obb=is_obb, **options)
    return OcTrack(box, cls=1, det_ind=index, is_obb=is_obb, **options)


@pytest.mark.parametrize("family", ["oc", "deep", "hybrid"])
@pytest.mark.parametrize(
    "kind,is_obb", [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)]
)
def test_grouped_prediction_geometry_preserves_state_and_history(family, kind, is_obb, monkeypatch):
    """Only box conversion changes; grouped KF arithmetic and counters stay exact."""
    actual = [_track(family, kind, is_obb, index) for index in range(3)]
    expected = deepcopy(actual)
    for frame in range(4):
        predictions = batching.predict_tracks(actual, dt=0.5 + frame * 0.1)
        with monkeypatch.context() as scalar:
            scalar.setattr(batching, "_batch_prediction_hook", lambda track: False)
            reference = batching.predict_tracks(expected, dt=0.5 + frame * 0.1)
        for prediction, target, track, original in zip(predictions, reference, actual, expected, strict=True):
            if family == "hybrid":
                np.testing.assert_array_equal(prediction[1:], target[1:])
                prediction, target = prediction[0], target[0]
            np.testing.assert_allclose(prediction, target, rtol=1e-13, atol=1e-13)
            np.testing.assert_array_equal(track.kf.x, original.kf.x)
            np.testing.assert_array_equal(track.kf.P, original.kf.P)
            np.testing.assert_allclose(track.history, original.history, rtol=1e-13, atol=1e-13)
            assert (track.age, track.time_since_update, track.hit_streak) == (
                original.age, original.time_since_update, original.hit_streak
            )
        assert not np.shares_memory(actual[0].history[-1], actual[1].history[-1])


def test_custom_prediction_finish_hooks_keep_their_call_contract():
    """An inherited batch opt-in must not pass new arguments to custom hooks."""
    class CustomTrack(OcTrack):
        def _finish_prediction(self):
            self.finished = True
            return super()._finish_prediction()

    custom = CustomTrack(np.array([10, 20, 30, 40, 0.9]), cls=0, det_ind=0, id_allocator=TrackIdAllocator())
    modified = _track("oc", "xysr", False)
    original = modified._finish_prediction

    def finish(self):
        self.finished = True
        return original()

    modified._finish_prediction = MethodType(finish, modified)
    batching.predict_tracks([custom, modified])
    assert custom.finished and modified.finished


def test_custom_state_getters_and_deep_converters_are_not_bypassed():
    """Batch prediction/output must retain instance and subclass geometry hooks."""
    class CustomTrack(OcTrack):
        def get_state(self):
            return super().get_state() + 3.0

    custom = CustomTrack(np.array([10, 20, 30, 40, 0.9]), cls=0, det_ind=0, id_allocator=TrackIdAllocator())
    modified = _track("oc", "xysr", False)
    original = modified.get_state
    modified.get_state = lambda: original() + 4.0
    deep = _track("deep", "xysr", False)
    converter = deep.x_to_bbox_func
    deep.x_to_bbox_func = lambda state: converter(state) + 5.0
    tracks = [custom, modified, deep]
    expected = np.concatenate([track.get_state() for track in tracks])
    np.testing.assert_array_equal(batching.current_track_boxes(tracks, is_obb=False), expected)
    np.testing.assert_array_equal(batching.observation_output_boxes(tracks, is_obb=False), expected)
    predictions = batching.predict_tracks(tracks)
    np.testing.assert_array_equal(predictions[-1], deep.get_state())


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("kind", ["xysr", "xywh"])
@pytest.mark.parametrize("aliases", [False, True])
def test_prediction_history_cmc_retains_in_place_rows_and_alias_order(is_obb, kind, aliases):
    """Disjoint histories batch; repeated/overlapping arrays retain scalar order."""
    model = create_motion_model(kind, is_obb=is_obb)
    box = [40, 50, 20, 30, 0.3] if is_obb else [30, 35, 50, 65, 0.9]
    values = np.tile(np.asarray(box, dtype=np.float32), (4, 1))
    first = SimpleNamespace(history=deque([values[0:1], values[1:2]], maxlen=5))
    second = SimpleNamespace(history=deque([values[2:3], values[3:4]], maxlen=5))
    if aliases:
        second.history.append(first.history[0])
        second.history.append(values[1].reshape(1, -1))
    expected_values = values.copy()
    expected_rows = [expected_values[0:1], expected_values[1:2], expected_values[2:3], expected_values[3:4]]
    if aliases:
        expected_rows.extend([expected_values[0:1], expected_values[1:2]])
    transform = np.array([[1.01, 0.02, 3.1], [-0.02, 1.01, -2.7], [0.0001, -0.0002, 1.0]])
    for row in expected_rows:
        row[:] = model.transform_boxes(row, transform)
    owners = [id(row) for track in (first, second) for row in track.history]
    transform_prediction_histories([first, second], transform, model=model)
    np.testing.assert_array_equal(values, expected_values)
    assert owners == [id(row) for track in (first, second) for row in track.history]
    assert first.history.maxlen == second.history.maxlen == 5


@pytest.mark.parametrize("max_obs,delta_t", [(3, 2), (3, 5)])
def test_deep_observation_retention_preserves_velocity_recovery_and_cmc(max_obs, delta_t):
    """Long-running AABB tracks stop accumulating discarded observations."""
    class UnboundedDeepTrack(DeepTrack):
        def _prepare_update(self, det):
            retained = self.observations.copy()
            measurement = super()._prepare_update(det)
            retained.update(self.observations)
            self.observations = retained
            return measurement

    detection = np.array([30.0, 40, 50, 80, 0.9, 0, 0])
    actual = DeepTrack(detection, max_obs=max_obs, delta_t=delta_t, id_allocator=TrackIdAllocator())
    expected = UnboundedDeepTrack(detection, max_obs=max_obs, delta_t=delta_t, id_allocator=TrackIdAllocator())
    transform = np.array([[1.0, -0.001, 0.1], [0.001, 1.0, 0.2]])
    for frame in range(24):
        for track in (actual, expected):
            track.predict()
            track.camera_update(transform)
        np.testing.assert_array_equal(
            k_previous_obs(actual.observations, actual.age, delta_t),
            k_previous_obs(expected.observations, expected.age, delta_t),
        )
        observed = None if 12 <= frame <= 18 else detection + [frame * 0.1, 0, frame * 0.1, 0, 0, 0, 0]
        actual.update(observed)
        expected.update(observed)
        np.testing.assert_array_equal(actual.kf.x, expected.kf.x)
        np.testing.assert_array_equal(actual.kf.P, expected.kf.P)
        if actual.velocity is not None:
            np.testing.assert_array_equal(actual.velocity, expected.velocity)
        assert len(actual.observations) <= max(max_obs, delta_t)
    assert len(expected.observations) > max(max_obs, delta_t)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hybrid_feature_mean_cache_preserves_order_eviction_and_rounding(dtype):
    """Cache unchanged banks without accumulating running-sum roundoff."""
    rng = np.random.default_rng(17)
    track = HybridTrack(
        np.array([10, 20, 30, 50, 0.9]),
        np.ones(8, dtype=dtype),
        longterm_bank_length=3,
        id_allocator=TrackIdAllocator(),
    )
    for _ in range(12):
        expected = np.vstack(list(track.features)).mean(axis=0)
        cached = track.get_longterm_feature()
        np.testing.assert_array_equal(cached, expected)
        assert track.get_longterm_feature() is cached
        track.update_features(rng.normal(size=8).astype(dtype))
        assert track.get_longterm_feature() is not cached
        np.testing.assert_array_equal(track.get_longterm_feature(), np.vstack(list(track.features)).mean(axis=0))
    track.features.clear()
    assert track.get_longterm_feature() is track.smooth_feat
