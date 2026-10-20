"""Batched ByteTrack geometry preserves scalar conversion and input ownership."""

import numpy as np
import pytest

from boxmot.trackers.bytetrack.track import STrack
from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("kind", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("initialized", [(False, False), (True, True), (False, True)])
def test_association_boxes_match_scalar_geometry(is_obb: bool, kind: str, initialized: tuple[bool, bool]) -> None:
    """Match scalar rounding for detections, live states, and mixed precision."""
    tracker = ByteTrack(kalman=KalmanConfig(type=kind), is_obb=is_obb)
    boxes = (
        [[103.12345, 57.09876, 28.56789, 42.34567, 0.3], [201.56789, 93.12345, 19.87654, 36.23456, -0.7]]
        if is_obb
        else [[103.12345, 57.09876, 131.69134, 99.44443], [201.56789, 93.12345, 221.44443, 129.35801]]
    )
    tracks = []
    for index, (box, live) in enumerate(zip(boxes, initialized, strict=True)):
        track = STrack(
            np.asarray([*box, 0.9, 0, index], dtype=np.float32),
            max_obs=50,
            id_allocator=tracker.id_allocator,
            is_obb=is_obb,
            motion_model=tracker.motion_model,
        )
        if live:
            track.activate(tracker.kalman_filter, 1)
        tracks.append(track)
    expected = np.asarray([track.xywha if is_obb else track.xyxy for track in tracks])
    result = tracker._association_boxes(tracks)
    np.testing.assert_array_equal(result, expected)
    result[:] = 0
    np.testing.assert_array_equal(np.asarray([track.xywha if is_obb else track.xyxy for track in tracks]), expected)


def test_initialized_states_with_mixed_dtypes_preserve_scalar_rounding() -> None:
    """Do not promote float32 state arithmetic before converting its geometry."""
    tracker = ByteTrack()
    tracks = []
    for index, dtype in enumerate((np.float32, np.float64)):
        track = STrack(
            np.array([103.12345, 57.09876, 131.69134, 99.44443, 0.9, 0, index], dtype=np.float32),
            max_obs=50,
            id_allocator=tracker.id_allocator,
            motion_model=tracker.motion_model,
        )
        track.activate(tracker.kalman_filter, 1)
        track.mean = track.mean.astype(dtype)
        tracks.append(track)
    np.testing.assert_array_equal(tracker._association_boxes(tracks), np.asarray([track.xyxy for track in tracks]))
