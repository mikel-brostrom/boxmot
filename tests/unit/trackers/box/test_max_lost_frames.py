"""ByteTrack-family lifetime settings count updates directly in both geometries."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot import BotSort, BotSortConfig, ByteTrack, ByteTrackConfig
from boxmot.trackers.botsort.track import TrackState as BotSortState
from boxmot.trackers.bytetrack.track import TrackState as ByteTrackState
from boxmot.trackers.common.box.base import BoxTracker


def _tracker(name: str, max_lost_frames: int, is_obb: bool) -> BoxTracker:
    """Disable optional inference so lifecycle checks exercise only tracking."""
    if name == "botsort":
        config = BotSortConfig(max_lost_frames=max_lost_frames, use_cmc=False, use_embeddings=False)
        return BotSort(config=config, is_obb=is_obb)
    return ByteTrack(config=ByteTrackConfig(max_lost_frames=max_lost_frames), is_obb=is_obb)


def _detections(is_obb: bool) -> np.ndarray:
    """Return one high-confidence stationary detection in the selected geometry."""
    row = [20, 35, 20, 50, 0.3, 0.95, 0] if is_obb else [10, 10, 30, 60, 0.95, 0]
    return np.array([row], dtype=np.float32)


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("max_lost_frames", [0, 1, 3, 45])
def test_max_lost_frames_controls_expiry_and_survives_reset(name: str, is_obb: bool, max_lost_frames: int) -> None:
    """Expiry keeps the existing association-first ordering, including a zero limit."""
    tracker = _tracker(name, max_lost_frames, is_obb)
    detections = _detections(is_obb)
    empty = np.empty((0, detections.shape[1]), dtype=np.float32)
    state = BotSortState if name == "botsort" else ByteTrackState

    for _ in range(2):
        assert tracker.max_lost_frames == max_lost_frames
        assert tracker.config.max_lost_frames == max_lost_frames
        initial = tracker.update(detections)
        identity = int(initial[0, 5 if is_obb else 4])
        track = tracker.active_tracks[0]

        # Newly lost tracks enter the expiry pool after the current check,
        # so even a zero limit is first applied on the second missed update.
        for missed_updates in range(1, max(2, max_lost_frames + 1)):
            assert tracker.update(empty).shape == (0, 9 if is_obb else 8)
            assert track.state == state.Lost
            assert tracker.frame_count - track.end_frame == missed_updates
        tracker.update(empty)
        assert track.state == state.Removed
        assert {removed.id for removed in tracker.removed_stracks} == {identity}

        # Existing cleanup removes the expired candidate one update later.
        tracker.update(empty)
        assert tracker.lost_stracks == []
        tracker.reset()
        assert tracker.frame_count == 0
        assert tracker.active_tracks == []
        assert tracker.lost_stracks == []
        assert not tracker.removed_stracks


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
@pytest.mark.parametrize("is_obb", [False, True])
def test_last_retained_candidate_can_recover_before_expiry_cleanup(name: str, is_obb: bool) -> None:
    """Renaming the limit preserves the existing final association opportunity."""
    tracker = _tracker(name, 2, is_obb)
    detections = _detections(is_obb)
    empty = np.empty((0, detections.shape[1]), dtype=np.float32)
    id_column = 5 if is_obb else 4
    state = BotSortState if name == "botsort" else ByteTrackState
    identity = int(tracker.update(detections)[0, id_column])

    for _ in range(3):
        tracker.update(empty)
    assert tracker.lost_stracks[0].state == state.Removed
    np.testing.assert_array_equal(tracker.update(detections)[:, id_column], [identity])
