"""Exercise long-running ByteTrack identity churn and lifecycle retention."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from boxmot.trackers.bytetrack.track import STrack, TrackState
from boxmot.trackers.bytetrack.tracker import ByteTrack


def _detections() -> np.ndarray:
    """Return one stationary object whose identity can expire between visits."""
    return np.array([[10, 10, 30, 60, 0.95, 0]], dtype=np.float32)


@pytest.mark.parametrize("display_frames", [0, 10])
def test_retired_tracks_are_collectible_during_long_identity_churn(display_frames: int) -> None:
    """Old trajectories must die while outputs match the original archive behavior."""
    tracker = ByteTrack(track_buffer=2)
    tracker.removed_display_frames = display_frames
    reference = ByteTrack(track_buffer=2)
    reference.removed_display_frames = display_frames
    archive: dict[int, STrack] = {}
    observed: dict[int, weakref.ReferenceType[STrack]] = {}
    empty = np.empty((0, 6), dtype=np.float32)
    peaks = []
    for frame_index in range(1_400):
        detections = _detections() if frame_index % 7 < 2 else empty
        actual = tracker.update(detections)
        # Restore every former object before each update to reproduce the old
        # lifetime-long removal archive independently of the new retention.
        reference.removed_stracks = list(archive.values())
        expected = reference.update(detections)
        archive.update((track.id, track) for track in reference.removed_stracks)
        np.testing.assert_array_equal(actual, expected)
        for track in tracker.active_tracks:
            observed.setdefault(track.id, weakref.ref(track))
        peaks.append(len(tracker.removed_stracks))
        tracker._get_removed_tracks_for_display(now=frame_index, ttl=display_frames)
        assert len(tracker._removed_first_seen) <= 2
        assert len(tracker._removed_expired) <= 2
    del track
    gc.collect()

    assert len(observed) == 200
    assert len(archive) == 200
    assert max(peaks) <= 4
    assert sum(reference() is not None for reference in observed.values()) <= 2
    assert all(reference() is None for identity, reference in observed.items() if identity < 190)

    tracker.reset()
    gc.collect()
    assert all(reference() is None for reference in observed.values())


def test_removal_preserves_delayed_association_and_display_lifetime() -> None:
    """A just-expired lost candidate can still recover on the next frame."""
    tracker = ByteTrack(track_buffer=2)
    detections = _detections()
    empty = np.empty((0, 6), dtype=np.float32)
    identity = int(tracker.update(detections)[0, 4])
    for _ in range(3):
        tracker.update(empty)
    assert tracker.lost_stracks[0].state == TrackState.Removed
    np.testing.assert_array_equal(tracker.update(detections)[:, 4], [identity])

    # Exercise the existing archive effect after that boundary reactivation:
    # the next missed observation leaves the association pool immediately.
    tracker.update(empty)
    assert tracker.lost_stracks == []
    assert tracker.update(detections).shape == (0, 8)
    replacement = tracker.update(detections)
    assert int(replacement[0, 4]) != identity

    display_tracker = ByteTrack(track_buffer=2)
    display_tracker.update(detections)
    for _ in range(3):
        display_tracker.update(empty)
    for offset in range(display_tracker.removed_display_frames):
        visible = display_tracker._get_removed_tracks_for_display(now=offset, ttl=10)
        assert {track.id for track in visible} == {identity}
        display_tracker.update(empty)
    assert display_tracker.removed_stracks == []
