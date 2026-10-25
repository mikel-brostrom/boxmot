"""ByteTrack batch formatting retains generic geometry and metadata semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.trackers.bytetrack.track import STrack
from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig
from boxmot.trackers.common.tracking.display import TrackDisplayMixin


def _tracks(tracker: ByteTrack) -> list[STrack]:
    """Build mixed live and detection states with nontrivial float rounding."""
    tracks = []
    for index in range(3):
        box = (
            [103.12345 + index, 57.09876, 28.56789, 42.34567, (index + 1) * np.pi]
            if tracker.is_obb
            else [103.12345 + index, 57.09876, 131.69134, 99.44443]
        )
        track = STrack(
            np.array([*box, 0.912345, 2, index], dtype=np.float32),
            max_obs=50,
            id_allocator=tracker.id_allocator,
            is_obb=tracker.is_obb,
            motion_model=tracker.motion_model,
        )
        if index:
            track.activate(tracker.kalman_filter, 1)
        tracks.append(track)
    return tracks


def _scalar_outputs(tracker: ByteTrack, tracks: list[object], dtype=np.float32) -> np.ndarray:
    """Independent row-by-row reference for geometry and metadata rounding."""
    rows = [
        tracker.format_output_row(
            tracker._track_box_for_output(track),
            tracker._track_id(track),
            float(track.conf),
            int(track.cls),
            int(track.det_ind),
            dtype=dtype,
        )
        for track in tracks
    ]
    return tracker.format_output_rows(rows, dtype=dtype)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("kind", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("dtype", [None, np.float16, np.float32, np.float64])
def test_native_output_matches_generic_formatting(is_obb: bool, kind: str, dtype) -> None:
    """Preserve float32 geometry rounding, requested dtype, and scalar angle normalization."""
    tracker = ByteTrack(kalman=KalmanConfig(type=kind), is_obb=is_obb)
    tracks = _tracks(tracker)
    for selected in (tracks, tracks[1:], []):
        expected = _scalar_outputs(tracker, selected, dtype=dtype)
        actual = tracker.format_outputs(iter(selected), dtype=dtype)
        assert actual.dtype == np.dtype(dtype)
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)


def test_foreign_and_subclass_track_output_uses_generic_geometry() -> None:
    """Foreign output boxes and subclass overrides take precedence over association boxes."""
    tracker = ByteTrack()

    class CustomTrack(STrack):
        @property
        def output_box(self) -> np.ndarray:
            return np.array([1, 2, 3, 4], dtype=np.float32)

    custom = CustomTrack(
        np.array([10, 20, 30, 40, 0.9, 2, 0], dtype=np.float32),
        max_obs=50,
        id_allocator=tracker.id_allocator,
        motion_model=tracker.motion_model,
    )
    foreign = SimpleNamespace(output_box=np.array([5, 6, 7, 8]), id=9, conf=0.7, cls=2, det_ind=1)
    actual = tracker.format_outputs([custom, foreign])
    np.testing.assert_array_equal(actual, _scalar_outputs(tracker, [custom, foreign]))
    np.testing.assert_array_equal(actual[:, :4], [[1, 2, 3, 4], [5, 6, 7, 8]])


def test_tracker_formatting_override_is_preserved() -> None:
    """A tracker subclass can still customize individual output rows."""

    class CustomTracker(ByteTrack):
        def format_output_row(self, *args, **kwargs) -> np.ndarray:
            row = super().format_output_row(*args, **kwargs)
            row[-1] = -1
            return row

    tracker = CustomTracker()
    rows = tracker.format_outputs(_tracks(tracker))
    np.testing.assert_array_equal(rows[:, -1], [-1, -1, -1])


def test_instance_formatting_override_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instance hooks must not be bypassed by native batch formatting."""
    tracker = ByteTrack()
    monkeypatch.setattr(tracker, "_track_box_for_output", lambda track: np.array([1, 2, 3, 4]))
    rows = tracker.format_outputs(_tracks(tracker))
    np.testing.assert_array_equal(rows[:, :4], [[1, 2, 3, 4]] * 3)


def test_output_matrix_and_empty_hooks_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consolidation retains ByteTrack's final matrix and empty output hooks."""
    tracker = ByteTrack()
    format_rows = tracker.format_output_rows

    def custom_rows(rows, dtype=np.float32):
        result = format_rows(rows, dtype=dtype)
        result[:, -1] = -2
        return result

    monkeypatch.setattr(tracker, "format_output_rows", custom_rows)
    np.testing.assert_array_equal(tracker.format_outputs(_tracks(tracker))[:, -1], [-2, -2, -2])
    empty = np.empty((0, 8), dtype=np.float32)
    monkeypatch.setattr(tracker, "_empty_output", lambda dtype: empty)
    assert tracker.format_outputs([]) is empty


def test_native_output_rejects_nonfinite_metadata() -> None:
    """Preallocated rows retain the common output finiteness check."""
    tracker = ByteTrack()
    tracks = _tracks(tracker)
    tracks[0].conf = np.nan
    with pytest.raises(ValueError, match="finite values"):
        tracker.format_outputs(tracks)


def test_native_output_keeps_batched_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sharing the packer must not reintroduce one conversion per native track."""
    tracker = ByteTrack()
    tracks = _tracks(tracker)[1:]
    expected = _scalar_outputs(tracker, tracks)
    original = STrack._current_aabb_xywh
    conversions = []

    def counted(track):
        conversions.append(track)
        return original(track)

    monkeypatch.setattr(STrack, "_current_aabb_xywh", counted)
    np.testing.assert_array_equal(tracker.format_outputs(tracks), expected)
    assert not conversions


def test_geometry_property_is_resolved_once() -> None:
    """Display/output adapters must not evaluate a computed box property twice."""

    class Track:
        calls = 0

        @property
        def xyxy(self) -> np.ndarray:
            self.calls += 1
            return np.array([[1, 2, 3, 4]])

    track = Track()
    np.testing.assert_array_equal(TrackDisplayMixin._resolve_track_box_attr(track, "xyxy"), [1, 2, 3, 4])
    assert track.calls == 1
    assert TrackDisplayMixin._resolve_track_box_attr(track, "missing") is None
