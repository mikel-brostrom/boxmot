"""Regression coverage for batched CMC state ownership and motion adapters."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.trackers.boosttrack.track import KalmanBoxTracker as BoostTrack
from boxmot.trackers.common.detections.layout import AABB_DETECTIONS
from boxmot.trackers.common.geometry.obb import transform_aabb, transform_obb
from boxmot.trackers.common.motion.cmc.batching import transform_observation_histories
from boxmot.trackers.common.motion.cmc.integration import apply_cmc_to_tracks
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.deepocsort.track import KalmanBoxTracker as DeepTrack
from boxmot.trackers.hybridsort.track import KalmanBoxTracker as HybridTrack
from boxmot.trackers.ocsort.track import KalmanBoxTracker as OrientedTrack


@pytest.mark.parametrize(
    "kind,is_obb",
    [(kind, oriented) for kind in MotionModelKind for oriented in (False, True) if kind != "xyscr" or not oriented],
)
def test_batched_motion_conversion_matches_scalar_policy(kind, is_obb):
    """Retain the scalar models' different clamp, ratio and score conventions."""
    model = create_motion_model(kind, is_obb=is_obb)
    boxes = np.array([[20.0, 30.0, 14.0, 8.0, 3.4], [40.0, 60.0, 1e-9, 1e-10, -3.5], [1.0, 2.0, -3.0, 0.0, 0.7]])
    if not is_obb:
        boxes[:, 2:4] += boxes[:, :2]
    measurements = model.to_measurements(boxes)
    expected = np.asarray([model.to_measurement(row, column=False) for row in boxes])
    np.testing.assert_allclose(measurements, expected, rtol=1e-13, atol=1e-13)
    for include_score in (False, True):
        actual = model.to_boxes(measurements, include_score=include_score)
        score = 0.5 if include_score and kind == "xyscr" else None
        expected_boxes = np.concatenate([model.to_box(row, score=score) for row in measurements])
        np.testing.assert_allclose(actual, expected_boxes, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(actual, model.to_boxes(measurements[..., None], include_score=include_score))
    assert model.to_measurements(boxes[:0]).shape == (0, model.dim_z)
    assert model.to_boxes(measurements[:0]).shape == (0, 5 if is_obb else 4)


def test_cmc_dispatch_batches_concrete_track_types_and_accepts_external_tracks():
    calls = []

    class BatchTrack:
        @classmethod
        def multi_camera_update(cls, tracks, warp):
            calls.append((cls, list(tracks), warp))

        def camera_update(self, warp):
            pytest.fail("Internal track CMC must use the batch entrypoint")

    class OtherBatchTrack(BatchTrack):
        pass

    class ExternalTrack:
        def camera_update(self, warp):
            calls.append((type(self), [self], warp))

    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -3.0]])
    tracks = [BatchTrack(), OtherBatchTrack(), BatchTrack(), ExternalTrack()]
    cmc = SimpleNamespace(apply=lambda image, boxes: matrix)
    result = apply_cmc_to_tracks(cmc, np.zeros((4, 4, 3)), np.empty((0, 6)), AABB_DETECTIONS, tracks)
    assert result is matrix
    assert [(kind, owners) for kind, owners, _ in calls] == [
        (BatchTrack, [tracks[0], tracks[2]]),
        (OtherBatchTrack, [tracks[1]]),
        (ExternalTrack, [tracks[3]]),
    ]


@pytest.mark.parametrize("is_obb", [False, True])
def test_observation_aliases_keep_in_place_dtype_and_per_track_warp_count(is_obb):
    row = np.array([20, 30, 14, 8, 0.3, 0.9] if is_obb else [10, 20, 30, 50, 0.9], dtype=np.float32)
    first = SimpleNamespace(last_observation=row, observations={1: row, 2: row})
    second = SimpleNamespace(last_observation=row, observations={1: row})
    transform = np.array([[1.01, 0.02, 3.1], [-0.02, 1.01, -2.7]])
    expected = row.copy()
    width = 5 if is_obb else 4
    for _ in range(2):
        expected[:width] = transform_obb(expected[:5], transform) if is_obb else transform_aabb(expected, transform)[:4]
    transform_observation_histories([first, second], transform, is_obb=is_obb)
    assert first.last_observation is first.observations[1] is second.last_observation is row
    assert row.dtype == np.float32
    np.testing.assert_allclose(row, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("family", ["deep", "oriented", "hybrid"])
def test_observation_cmc_warps_frozen_and_timed_measurements_together(family):
    is_obb = family == "oriented"
    tracks = []
    originals = []
    for index in range(3):
        box = (
            np.array([30.0 + index, 40.0, 20.0, 12.0, 0.2, 0.9])
            if is_obb
            else np.array([10.0 + index, 20.0, 30.0, 50.0, 0.9])
        )
        track = (
            OrientedTrack(box, cls=1, det_ind=index, is_obb=True, id_allocator=TrackIdAllocator())
            if is_obb
            else DeepTrack(np.r_[box, 1, index], id_allocator=TrackIdAllocator())
        )
        if family == "hybrid":
            track = HybridTrack(box, np.ones(4), id_allocator=TrackIdAllocator())
            track.last_observation_save = box.copy()
            track.history = deque([box.reshape(1, -1).copy()], maxlen=11)
            track.velocity_lt = np.array([0.6, 0.8])
        model, kalman = track.motion_model, track.kf
        measurement = model.to_measurement(box)
        kalman.update(measurement)
        kalman.predict(dt=0.2)
        kalman.update(None)
        kalman.predict(dt=0.4)
        kalman.history_obs = deque([measurement.copy(), None], maxlen=7)
        kalman.observed = False
        kalman.attr_saved = {
            "x": kalman.x.copy(),
            "P": kalman.P.copy(),
            "history_obs": deque([None, measurement.copy()], maxlen=9),
            "last_measurement": measurement.copy(),
        }
        kalman.last_measurement = measurement.copy()
        track.last_observation = box.copy()
        track.observations = {1: track.last_observation}
        track.velocity = np.array([0.6, 0.8])
        tracks.append(track)
        originals.append(deepcopy(track))
    transform = np.array([[1.02, 0.05, 4.0], [-0.01, 0.97, -2.0]])
    type(tracks[0]).multi_camera_update(tracks, transform)
    warp = transform_obb if is_obb else transform_aabb
    for track, original in zip(tracks, originals):

        def expected(measurement):
            score = float(np.asarray(measurement).reshape(-1)[3]) if family == "hybrid" else None
            box = original.motion_model.to_box(measurement, score=score)[0]
            return original.motion_model.to_measurement(warp(box, transform))

        for name in ("last_measurement", "_last_observed_measurement"):
            np.testing.assert_allclose(getattr(track.kf, name), expected(getattr(original.kf, name)))
        np.testing.assert_allclose(track.kf.history_obs[0], expected(original.kf.history_obs[0]))
        np.testing.assert_allclose(
            track.kf.attr_saved["history_obs"][1], expected(original.kf.attr_saved["history_obs"][1])
        )
        np.testing.assert_allclose(
            track.kf.attr_saved["last_measurement"], expected(original.kf.attr_saved["last_measurement"])
        )
        assert track.kf.history_obs[1] is None and track.kf.history_obs.maxlen == 7
        assert track.kf.attr_saved["history_obs"][0] is None and track.kf.attr_saved["history_obs"].maxlen == 9
        if family == "hybrid":
            assert track.last_observation is not track.observations[1]
            assert track.history.maxlen == 11 and track.history[0].shape == original.history[0].shape
            np.testing.assert_array_equal(track.history[0][..., 4:], original.history[0][..., 4:])
            np.testing.assert_allclose(
                track.last_observation_save, transform_aabb(original.last_observation_save, transform)
            )
            np.testing.assert_array_equal(track.kf.x[[3, 8]], original.kf.x[[3, 8]])
        else:
            assert track.last_observation is track.observations[1]
        assert len(track.kf._prediction_steps) == len(original.kf._prediction_steps)
        for current, previous in zip(track.kf._prediction_steps, original.kf._prediction_steps):
            assert current[0] == previous[0]
            np.testing.assert_array_equal(current[1], previous[1])
            np.testing.assert_array_equal(current[2], previous[2])
        assert track.kf._prediction_origin[0].shape == original.kf._prediction_origin[0].shape
        assert track.kf.x.shape == original.kf.x.shape
        assert track.age == original.age


@pytest.mark.parametrize("projective", [False, True])
def test_boost_aabb_batch_preserves_diagonal_measurement_only_contract(projective):
    tracks = [
        BoostTrack(np.array([10.0 + i, 20.0, 30.0, 50.0, 0.9, 1, i]), max_obs=5, id_allocator=TrackIdAllocator())
        for i in range(3)
    ]
    matrix = np.array([[1.1, -0.03, 4.0], [0.04, 0.95, -3.0], [0.0002, 0.0003, 1.0]])
    transform = matrix if projective else matrix[:2]
    originals = [deepcopy(track) for track in tracks]
    BoostTrack.multi_camera_update(tracks, transform)
    for track, original in zip(tracks, originals):
        box = original.get_state()[0]
        first = matrix @ np.r_[box[:2], 1.0]
        last = matrix @ np.r_[box[2:], 1.0]
        width, height = last[:2] - first[:2]
        expected = [first[0] + width / 2, first[1] + height / 2, height, width / height]
        np.testing.assert_allclose(track.kf.x[:4], expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(track.kf.x[4:], original.kf.x[4:])
        np.testing.assert_array_equal(track.kf.covariance, original.kf.covariance)
