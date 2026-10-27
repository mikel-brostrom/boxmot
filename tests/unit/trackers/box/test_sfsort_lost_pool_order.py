"""Stable lost-pool filtering retains SFSORT recovery and expiry semantics."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot import SFSORT, SFSORTConfig


def _tracker(is_obb: bool) -> SFSORT:
    """Keep each region alive long enough for both association passes."""
    return SFSORT(SFSORTConfig(
        high_th=0.5, new_track_th=0.5, low_th=0.1, frame_width=400, frame_height=200,
        horizontal_margin=40, vertical_margin=40, central_timeout=5, marginal_timeout=3,
    ), is_obb=is_obb)


def _rows(is_obb: bool, confidence: float = 0.95) -> np.ndarray:
    """Place one marginal identity before two central identities."""
    centers = np.array([20.0, 100.0, 220.0])
    if is_obb:
        boxes = np.column_stack((centers, np.full(3, 100), np.full(3, 16), np.full(3, 32), np.zeros(3)))
    else:
        boxes = np.column_stack((centers - 8, np.full(3, 84), centers + 8, np.full(3, 116)))
    return np.column_stack((boxes, np.full(3, confidence), np.zeros(3)))


@pytest.mark.parametrize("is_obb", [False, True])
def test_first_and_second_pass_recovery_preserve_lost_pool_order(is_obb: bool) -> None:
    """Recovering from either pass removes just that identity and preserves survivors."""
    tracker = _tracker(is_obb)
    tracker.update(_rows(is_obb))
    identities = [track.id for track in tracker.active_tracks]
    empty = _rows(is_obb)[:0]
    tracker.update(empty)
    assert [track.id for track in tracker.lost_tracks] == identities
    assert [track.lost_region for track in tracker.lost_tracks] == ["marginal", "central", "central"]
    tracker.update(_rows(is_obb)[1:2])
    assert [track.id for track in tracker.active_tracks] == identities[1:2]
    assert [track.id for track in tracker.lost_tracks] == [identities[0], identities[2]]
    tracker.update(_rows(is_obb, confidence=0.3)[2:3])
    assert [track.id for track in tracker.active_tracks] == identities[2:3]
    assert [track.id for track in tracker.lost_tracks] == identities[:2]
    assert [track.time_since_update for track in tracker.lost_tracks] == [3, 1]


@pytest.mark.parametrize("is_obb", [False, True])
def test_timeout_boundaries_and_repeated_lost_input_keep_identity_and_order(is_obb: bool) -> None:
    """Expiry remains strictly greater than the timeout; duplicate inputs append once."""
    tracker = _tracker(is_obb)
    tracker.update(_rows(is_obb))
    tracks = tracker.active_tracks.copy()
    tracker.frame_count = 2
    tracker._update_lost_tracks([tracks[1], tracks[0], tracks[1], tracks[2]])
    assert tracker.lost_tracks == [tracks[1], tracks[0], tracks[2]]
    tracker.frame_count = 4
    tracker._purge_stale_lost_tracks()
    assert tracker.lost_tracks == [tracks[1], tracks[0], tracks[2]]
    tracker.frame_count = 5
    tracker._purge_stale_lost_tracks()
    assert tracker.lost_tracks == [tracks[1], tracks[2]]
    tracker.frame_count = 6
    tracker._purge_stale_lost_tracks()
    assert tracker.lost_tracks == [tracks[1], tracks[2]]
    tracker.frame_count = 7
    tracker._purge_stale_lost_tracks()
    assert tracker.lost_tracks == []
