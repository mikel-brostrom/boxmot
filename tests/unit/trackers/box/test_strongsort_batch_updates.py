"""Batched StrongSORT observations preserve geometry, histories, and hooks."""

from types import MethodType

import numpy as np
import pytest

from boxmot.trackers.common.geometry.obb import smooth_obb_corners
from boxmot.trackers.strongsort.track import Track, _Detection


def make_tracks(is_obb: bool, count: int = 6) -> tuple[list[Track], list[_Detection]]:
    """Build independent tracks with nontrivial measurements and embeddings."""
    tracks, detections = [], []
    for index in range(count):
        box = [31.25 + index, 52.125, 19.1 + index, 34.7 - index]
        if is_obb:
            box.append((-1) ** index * (np.pi - 0.02 * index))
        detection = _Detection(np.asarray(box), 0.7, index % 2, index, np.array([0.3, 0.7]), is_obb=is_obb)
        detections.append(detection)
        tracks.append(Track(detection, index + 1, n_init=3, max_age=10, max_obs=4, is_obb=is_obb))
    return tracks, detections


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_batched_measurements_preserve_scalar_values_and_input_ownership(is_obb, mixed_dtype):
    tracks, detections = make_tracks(is_obb)
    if mixed_dtype:
        detections[-1].tlwh = detections[-1].tlwh.astype(np.float64)
    original = [detection.tlwh.copy() for detection in detections]
    expected = np.asarray([track.measurement_for_detection(detection) for track, detection in zip(tracks, detections)])

    Track.multi_update(zip(tracks, detections))

    actual = np.asarray([track.bbox for track in tracks])
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    for detection, previous in zip(detections, original):
        assert detection.tlwh.dtype == previous.dtype
        np.testing.assert_array_equal(detection.tlwh, previous)
    assert all(not np.shares_memory(track.bbox, detection.tlwh) for track, detection in zip(tracks, detections))


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("hook", ["track", "detection"])
def test_batched_measurements_retain_instance_conversion_hooks(is_obb, hook):
    tracks, detections = make_tracks(is_obb)
    expected = tracks[0].measurement_for_detection(detections[0])
    expected[:2] += 2.0
    calls = []
    if hook == "track":

        def measurement(self, detection):
            calls.append(detection)
            return expected.copy()

        tracks[0].measurement_for_detection = MethodType(measurement, tracks[0])
    else:

        def measurement(self):
            calls.append(self)
            return expected.copy()

        detections[0].to_measurement = MethodType(measurement, detections[0])

    Track.multi_update(zip(tracks, detections))

    assert calls == [detections[0]]
    np.testing.assert_array_equal(tracks[0].bbox, expected)


def test_batched_obb_history_matches_scalar_corners_and_continuity():
    tracks, detections = make_tracks(True)
    tracks[0]._plot_angle = None
    for frame in range(3):
        previous_angles = [track._plot_angle for track in tracks]
        for index, detection in enumerate(detections):
            detection.tlwh[:2] += 0.13
            detection.tlwh[4] += (-1) ** index * 0.17
            if frame == 1:
                detection.tlwh[2:4] = detection.tlwh[2:4][::-1]
                detection.tlwh[4] += np.pi / 2.0

        Track.multi_predict(tracks, dt=0.7)
        Track.multi_update(zip(tracks, detections))

        for track, previous_angle in zip(tracks, previous_angles):
            corners, angle = smooth_obb_corners(track.xywha, previous_angle)
            assert track._plot_angle == angle
            assert track.history_observations[-1].dtype == np.float32
            np.testing.assert_array_equal(track.history_observations[-1], corners)
        assert not np.shares_memory(tracks[0].history_observations[-1], tracks[1].history_observations[-1])
    assert all(len(track.history_observations) == 4 for track in tracks)


@pytest.mark.parametrize("hook", ["_append_current_history", "_finish_update"])
def test_batched_obb_history_retains_instance_update_hooks(hook):
    tracks, detections = make_tracks(True)
    calls = []
    original = getattr(tracks[0], hook)

    def called(self, *args):
        calls.append(self)
        return original(*args)

    setattr(tracks[0], hook, MethodType(called, tracks[0]))

    Track.multi_update(zip(tracks, detections))

    assert calls == [tracks[0]]
    assert all(len(track.history_observations) == 2 for track in tracks)


def test_batched_update_retains_track_subclass_history():
    class CustomTrack(Track):
        def _append_current_history(self) -> None:
            """Record a custom history row instead of native box geometry."""
            self.history_observations.append(np.array([self.hits]))

    _, detections = make_tracks(True)
    tracks = [
        CustomTrack(detection, index + 1, n_init=3, max_age=10, max_obs=4, is_obb=True)
        for index, detection in enumerate(detections)
    ]

    Track.multi_update(zip(tracks, detections))

    assert all(len(track.history_observations) == 2 for track in tracks)
    assert all(track.history_observations[-1].tolist() == [1] for track in tracks)
