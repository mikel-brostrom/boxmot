from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from boxmot.trackers.common.association.matching import iou_distance

TrackT = TypeVar("TrackT")


def track_id(track) -> int:
    """Return a tracker object's canonical integer ID."""
    if hasattr(track, "id"):
        return int(getattr(track, "id"))
    return int(getattr(track, "track_id"))


def joint_stracks(tlista: Sequence[TrackT], tlistb: Sequence[TrackT]) -> list[TrackT]:
    """Join track lists, keeping the first occurrence of each track ID."""
    exists: set[int] = set()
    result: list[TrackT] = []
    for track in tlista:
        exists.add(track_id(track))
        result.append(track)
    for track in tlistb:
        tid = track_id(track)
        if tid not in exists:
            exists.add(tid)
            result.append(track)
    return result


def sub_stracks(tlista: Sequence[TrackT], tlistb: Sequence[TrackT]) -> list[TrackT]:
    """Return tracks from ``tlista`` whose IDs are not present in ``tlistb``."""
    remove_ids = {track_id(track) for track in tlistb}
    return [track for track in tlista if track_id(track) not in remove_ids]


def track_duration(track) -> int:
    """Return how long a track has been alive according to frame metadata."""
    return int(getattr(track, "frame_id", 0)) - int(getattr(track, "start_frame", 0))


def remove_duplicate_stracks(
    stracksa: Sequence[TrackT],
    stracksb: Sequence[TrackT],
    *,
    distance: Callable[[Sequence[TrackT], Sequence[TrackT]], np.ndarray] | None = None,
    duplicate_threshold: float = 0.15,
) -> tuple[list[TrackT], list[TrackT]]:
    """Remove duplicate tracks, keeping the longer-lived track in each pair."""
    if distance is None:
        is_obb = any(getattr(track, "is_obb", False) for track in [*stracksa, *stracksb])
        pdist = iou_distance(stracksa, stracksb, is_obb=is_obb)
    else:
        pdist = distance(stracksa, stracksb)

    pairs = np.where(pdist < duplicate_threshold)
    dupa: list[int] = []
    dupb: list[int] = []
    for p, q in zip(*pairs):
        if track_duration(stracksa[p]) > track_duration(stracksb[q]):
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [track for i, track in enumerate(stracksa) if i not in dupa]
    resb = [track for i, track in enumerate(stracksb) if i not in dupb]
    return resa, resb
