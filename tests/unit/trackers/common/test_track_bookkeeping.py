"""Metadata lookups and history snapshots retain their public ownership semantics."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.trackers.common.track_state import BoxTrack
from boxmot.trackers.common.tracking.lifecycle import track_id
from boxmot.trackers.common.tracking.track import TrackMeta, TrackState, sync_track_meta


def test_metadata_refresh_reads_present_properties_once() -> None:
    """Refresh explicit None values and retain absent fields without double property reads."""

    class Track:
        def __init__(self) -> None:
            self.meta = TrackMeta(id=1, age=7, conf=0.2, lost_region="central")
            self.calls = {"id": 0, "conf": 0, "age": 0}
            self.lost_region = None

        @property
        def id(self) -> int:
            self.calls["id"] += 1
            return 3

        @property
        def conf(self) -> float:
            self.calls["conf"] += 1
            return 0.8

        @property
        def age(self) -> int:
            self.calls["age"] += 1
            raise AttributeError("No age available")

    track = Track()
    meta = sync_track_meta(track, TrackState.LOST)
    assert meta is track.meta
    assert (meta.id, meta.conf, meta.age, meta.lost_region, meta.state) == (3, 0.8, 7, None, TrackState.LOST)
    assert track.calls == {"id": 1, "conf": 1, "age": 1}


@pytest.mark.parametrize("attribute", ["id", "track_id"])
def test_identity_helpers_read_properties_once(attribute: str) -> None:
    """Primary and fallback IDs retain integer coercion with one lookup per helper."""

    class Track:
        def __init__(self) -> None:
            self.calls = 0

        def identity(self) -> np.int64:
            self.calls += 1
            return np.int64(9)

    setattr(Track, attribute, property(Track.identity))
    track = Track()
    assert track_id(track) == 9
    assert track.calls == 1
    track.calls = 0
    meta = sync_track_meta(track)
    assert meta.id == 9
    # Creation resolves its initial identity, then refreshes any explicit id.
    assert track.calls == (2 if attribute == "id" else 1)


def test_identity_fallback_and_missing_identity_errors() -> None:
    """Missing primary IDs fall back, while a present None is still invalid."""

    class Track:
        track_id = 4

        @property
        def id(self) -> int:
            raise AttributeError("No primary identity")

    assert track_id(Track()) == 4
    assert sync_track_meta(Track()).id == 4
    with pytest.raises(AttributeError, match="does not expose an id"):
        sync_track_meta(SimpleNamespace())
    with pytest.raises(TypeError):
        track_id(SimpleNamespace(id=None, track_id=4))


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_history_snapshots_own_float32_geometry(is_obb: bool, dtype) -> None:
    """Snapshots stay independent of borrowed, strided geometry and later history entries."""

    class BorrowedGeometryTrack(BoxTrack):
        def __init__(self, geometry: np.ndarray) -> None:
            self.geometry = geometry
            self.is_obb = is_obb
            self.history_observations = deque(maxlen=2)

        @property
        def xyxy(self) -> np.ndarray:
            return self.geometry

        def _state_obb_for_plot(self) -> np.ndarray:
            return self.geometry

    geometry = (np.arange(16 if is_obb else 8, dtype=dtype) + dtype(0.123456789))[::2]
    track = BorrowedGeometryTrack(geometry)
    expected = geometry.astype(np.float32)
    track._append_current_history()
    snapshot = track.history_observations[0]
    assert snapshot.dtype == np.float32
    assert snapshot.flags.owndata
    assert not np.shares_memory(snapshot, geometry)
    np.testing.assert_array_equal(snapshot, expected)
    geometry[:] = -1
    track._append_current_history()
    np.testing.assert_array_equal(snapshot, expected)
    np.testing.assert_array_equal(track.history_observations[-1], np.full(len(geometry), -1, dtype=np.float32))
