"""IoU duplicate suppression retains geometry and lifecycle decisions."""

from types import MethodType

import numpy as np
import pytest

from boxmot.trackers.botsort.track import STrack
from boxmot.trackers.common.association.matching import iou_distance
from boxmot.trackers.common.motion.models import create_motion_model
from boxmot.trackers.common.tracking.lifecycle import remove_duplicate_stracks
from boxmot.trackers.common.tracking.track import TrackIdAllocator


def _track(kind: str, is_obb: bool, allocator: TrackIdAllocator, index: int = 0) -> STrack:
    """Create a real track with fractional geometry and an independent filter."""
    model = create_motion_model(kind, is_obb=is_obb)
    box = [40.1234567 + index * 0.25, 60.7654321, 20.3456789, 30.9876543, 0.23456789]
    if not is_obb:
        box = [30.1234567 + index * 0.25, 45.7654321, 50.3456789 + index * 0.25, 75.9876543]
    track = STrack(
        np.asarray([*box, 0.9, 0, index]),
        id_allocator=allocator,
        is_obb=is_obb,
        motion_model=model,
    )
    track.activate(model.create_filter(), 1)
    track.predict()
    return track


def _scalar_distance(left, right, *, is_obb: bool) -> np.ndarray:
    """Reference the scalar property access used before geometry batching."""
    attribute = "xywha" if is_obb else "xyxy"
    width = 5 if is_obb else 4
    left_boxes = np.asarray([getattr(track, attribute) for track in left]).reshape(-1, width)
    right_boxes = np.asarray([getattr(track, attribute) for track in right]).reshape(-1, width)
    return iou_distance(left_boxes, right_boxes, is_obb=is_obb)


@pytest.mark.parametrize("kind", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("is_obb", [False, True])
def test_duplicate_suppression_matches_scalar_geometry_for_every_filter(kind: str, is_obb: bool) -> None:
    """Batching must preserve exact costs and the identity retained in a tie."""
    allocator = TrackIdAllocator()
    tracks = [_track(kind, is_obb, allocator, index) for index in range(4)]
    for track, duration in zip(tracks, (8, 3, 4, 8), strict=True):
        track.frame_id = 10
        track.start_frame = 10 - duration
    left, right = tracks[:2], tracks[2:]
    expected = _scalar_distance(left, right, is_obb=is_obb)
    np.testing.assert_array_equal(iou_distance(left, right), expected)
    actual = remove_duplicate_stracks(left, right)
    reference = remove_duplicate_stracks(left, right, distance=lambda _left, _right: expected)
    assert actual == reference
    assert actual == ([], [tracks[3]])


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("variation", ["dtype", "uninitialized", "instance", "subclass", "mixed_geometry"])
def test_iou_geometry_fallback_preserves_custom_and_mixed_tracks(is_obb: bool, variation: str) -> None:
    """Native geometry batching may decline states that need scalar hooks."""
    allocator = TrackIdAllocator()
    tracks = [_track("xywh", is_obb, allocator, index) for index in range(3)]
    if variation == "dtype":
        tracks[1].mean = tracks[1].mean.astype(np.float32)
    elif variation == "uninitialized":
        tracks[1].mean = None
    elif variation == "instance":
        if is_obb:
            # BoxTrack's properties are data descriptors. An explicit geometry
            # hook still makes the native batch adapter decline this track.
            tracks[1].__dict__["xywha"] = tracks[1].xywha.copy()
        else:
            original = tracks[1]._current_aabb_xywh
            tracks[1]._current_aabb_xywh = MethodType(lambda self: original() + 1.25, tracks[1])
    elif variation == "subclass":
        class CustomTrack(STrack):
            @property
            def xyxy(self) -> np.ndarray:
                return super().xyxy + 1.25

            @property
            def xywha(self) -> np.ndarray:
                box = super().xywha
                box[:2] += 1.25
                return box

        tracks[1].__class__ = CustomTrack
    else:
        tracks[1] = _track("xywh", not is_obb, allocator, 1)
    oriented = is_obb or variation == "mixed_geometry"
    np.testing.assert_array_equal(
        iou_distance(tracks[:2], tracks[2:]),
        _scalar_distance(tracks[:2], tracks[2:], is_obb=oriented),
    )


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("left_size,right_size", [(0, 0), (1, 0), (0, 1)])
def test_iou_empty_track_collections_keep_matrix_shape(is_obb: bool, left_size: int, right_size: int) -> None:
    """Duplicate bookkeeping accepts empty live or lost collections."""
    track = _track("xywh", is_obb, TrackIdAllocator())
    result = iou_distance([track] * left_size, [track] * right_size, is_obb=is_obb)
    assert result.shape == (left_size, right_size)
    assert result.dtype == np.float32
