"""Shared output packing preserves scalar rounding and user geometry hooks."""

from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.tracking.formatting import TrackFormattingMixin


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("dtype", [None, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("precomputed", [False, True])
def test_shared_output_matches_scalar_rows(is_obb, dtype, precomputed):
    """Packing retains float32 geometry and float64 angle-wrap arithmetic."""
    tracker = ByteTrack(is_obb=is_obb)
    angles = [np.pi, -np.pi, 13 * np.pi, np.nextafter(np.float32(np.pi), np.float32(0))]
    boxes = np.array([[103.1234567, 57.1234567, 28.5678912, 42.3456789, angle] for angle in angles])
    if not is_obb:
        boxes = boxes[:, :4]
    tracks = [
        SimpleNamespace(
            output_box=box, xywha=box, id=index + 1, conf=0.9123456789, cls=2, det_ind=index
        )
        for index, box in enumerate(boxes)
    ]
    expected = np.asarray([
        tracker.format_output_row(
            tracker._track_box_for_output(track), track.id, track.conf, track.cls, track.det_ind, dtype=dtype
        )
        for track in tracks
    ], dtype=dtype)
    actual = TrackFormattingMixin.format_outputs(
        tracker, iter(tracks), dtype=dtype, boxes=boxes if precomputed else None
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype(dtype)
    assert actual.flags.c_contiguous


def test_precomputed_output_retains_custom_row_hook():
    """Supplied geometry still passes through overridden scalar row formatting."""
    class CustomTracker(ByteTrack):
        def format_output_row(self, *args, **kwargs):
            row = super().format_output_row(*args, **kwargs)
            row[-1] = -2
            return row

    tracker = CustomTracker()
    tracks = [SimpleNamespace(id=3, conf=0.5, cls=1, det_ind=7)]
    actual = TrackFormattingMixin.format_outputs(tracker, tracks, boxes=np.array([[1, 2, 3, 4]]))
    np.testing.assert_array_equal(actual, [[1, 2, 3, 4, 3, 0.5, 1, -2]])


def test_precomputed_output_checks_alignment_and_finiteness():
    """An incorrect snapshot cannot silently pair geometry with another identity."""
    tracker = ByteTrack()
    tracks = [SimpleNamespace(id=3)]
    with pytest.raises(ValueError, match="shape"):
        TrackFormattingMixin.format_outputs(tracker, tracks, boxes=np.zeros((2, 4)))
    with pytest.raises(ValueError, match="finite"):
        TrackFormattingMixin.format_outputs(tracker, tracks, boxes=np.full((1, 4), np.nan))
    empty = TrackFormattingMixin.format_outputs(tracker, [], boxes=np.empty((0, 4)))
    assert empty.shape == (0, 8)
