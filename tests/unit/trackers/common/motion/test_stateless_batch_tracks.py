"""Track lifecycle and calibration stay identical when KFs run in batches."""

from copy import deepcopy

import numpy as np
import pytest

from boxmot.trackers.botsort.track import STrack as BotTrack
from boxmot.trackers.bytetrack.track import STrack as ByteTrack
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.strongsort.track import Track as StrongTrack
from boxmot.trackers.strongsort.tracker import _Detection


@pytest.mark.parametrize("is_obb", [False, True])
def test_strongsort_batch_preserves_independent_filters_and_history(is_obb):
    tracks, detections = [], []
    for index in range(7):
        geometry = [60.0 + index, 45.0, 30.0, 20.0]
        if is_obb:
            geometry.append(3.1)
        detection = _Detection(np.asarray(geometry), 0.8, 0, index, np.array([1.0, 0.0]), is_obb=is_obb)
        track = StrongTrack(
            detection,
            index + 1,
            n_init=3,
            max_age=30,
            max_obs=50,
            ema_alpha=0.9,
            is_obb=is_obb,
            noise_config=KalmanNoiseConfig(measurement_noise_scale=2.0 if index % 2 else 1.0),
        )
        tracks.append(track)
        geometry[0] += 0.8
        if is_obb:
            geometry[2:4] = geometry[2:4][::-1]
            geometry[4] += np.pi / 2.0
        detections.append(
            _Detection(np.asarray(geometry), 0.1 + index / 10.0, 1, index + 7, np.array([0.8, 0.2]), is_obb=is_obb)
        )
    scalar_tracks = deepcopy(tracks)
    for _ in range(3):
        StrongTrack.multi_predict(tracks, dt=0.7)
        StrongTrack.multi_update(zip(tracks, detections))
        for actual, expected, detection in zip(tracks, scalar_tracks, detections):
            expected.predict(dt=0.7)
            expected.update(detection)
            np.testing.assert_allclose(actual.mean, expected.mean, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(actual.covariance, expected.covariance, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(actual.features, expected.features)
            np.testing.assert_allclose(actual.history_observations, expected.history_observations, rtol=1e-6, atol=1e-5)
            for name in ("id", "bbox", "age", "hits", "state", "time_since_update", "conf", "cls", "det_ind"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("track_type", [ByteTrack, BotTrack])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("reactivate", [False, True])
def test_boxtrack_batch_correction_preserves_metadata_and_history(track_type, is_obb, reactivate):
    allocator = TrackIdAllocator()
    filters = [
        (KalmanFilterXYWH if is_obb or track_type is BotTrack else KalmanFilterXYAH)(
            ndim=5 if is_obb else 4, noise_config=KalmanNoiseConfig(measurement_noise_scale=scale)
        )
        for scale in (1.0, 3.0)
    ]
    tracks, detections = [], []
    for index in range(7):
        geometry = [50.0 + index, 30.0, 80.0 + index, 60.0]
        if is_obb:
            geometry = [65.0 + index, 45.0, 30.0, 20.0, 3.1]
        track = track_type(np.asarray([*geometry, 0.8, 0, index]), max_obs=10, id_allocator=allocator, is_obb=is_obb)
        track.activate(filters[index % 2], frame_id=1)
        if reactivate:
            track.mark_lost()
        tracks.append(track)
        geometry[0] += 0.5
        detections.append(
            track_type(np.asarray([*geometry, 0.9, 1, index + 7]), max_obs=10, id_allocator=allocator, is_obb=is_obb)
        )
    scalar_tracks = deepcopy(tracks)
    track_type.multi_predict(tracks, dt=0.8)
    track_type.multi_update(zip(tracks, detections), frame_id=5, reactivate=reactivate)
    for actual, expected, detection in zip(tracks, scalar_tracks, detections):
        expected.predict(dt=0.8)
        if reactivate:
            expected.re_activate(detection, frame_id=5)
        else:
            expected.update(detection, frame_id=5)
        np.testing.assert_allclose(actual.mean, expected.mean, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual.covariance, expected.covariance, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual.history_observations, expected.history_observations, rtol=1e-6, atol=1e-5)
        for name in (
            "id",
            "state",
            "is_activated",
            "frame_id",
            "start_frame",
            "tracklet_len",
            "conf",
            "cls",
            "det_ind",
        ):
            assert getattr(actual, name) == getattr(expected, name)
        if track_type is BotTrack:
            assert actual.cls_hist == expected.cls_hist
