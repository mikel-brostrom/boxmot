from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot.native.trackers import bytetrack as native_binding
from boxmot.structures import Detections, OrientedBoxes, Tracks
from boxmot.trackers.bytetrack import native as native_module

from ._helpers import detections_from_rows, empty_native_batch, update_rows, update_tracks


class _FakeLibrary:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create(self, cfg):
        self.calls.append(("create", cfg["frame_rate"], cfg["track_thresh"]))
        return "handle"

    def reset(self, handle):
        self.calls.append(("reset", handle))

    def update(
        self,
        handle,
        *,
        geometry,
        scores,
        class_ids,
        detection_indices,
        embeddings,
        image,
    ):
        self.calls.append(("update", handle, len(geometry), image, geometry.shape[1], embeddings))
        return empty_native_batch(geometry.shape[1])

    def destroy(self, handle):
        self.calls.append(("destroy", handle))


def test_native_bytetrack_uses_structured_live_library_wrapper():
    library = _FakeLibrary()
    tracker = native_module.NativeByteTrackTracker(
        {"frame_rate": 15, "track_thresh": 0.5},
        geometry="aabb",
        library=library,
    )
    detections = detections_from_rows(
        np.array(
            [[1, 1, 4, 5, 0.9, 0], [2, 2, 6, 7, 0.8, 0]],
            dtype=np.float32,
        )
    )

    output = tracker.update(detections)
    tracker.reset()
    tracker.close()

    assert isinstance(output, Tracks)
    assert output.is_obb is False
    assert output.sample_id == detections.sample_id
    assert library.calls == [
        ("create", 15, 0.5),
        ("update", "handle", 2, None, 4, None),
        ("reset", "handle"),
        ("destroy", "handle"),
    ]


def test_native_bytetrack_accepts_numpy_aabb6_and_rejects_mode_mismatch():
    library = _FakeLibrary()
    tracker = native_module.NativeByteTrackTracker(geometry="aabb", library=library)
    try:
        output = tracker.update(np.array([[1, 1, 4, 5, 0.9, 3]], dtype=np.float64))
        assert type(output) is np.ndarray
        assert output.shape == (0, 8)
        assert library.calls[1] == ("update", "handle", 1, None, 4, None)

        with pytest.raises(ValueError, match=r"AABB detection rows must have shape \[N, 6\]"):
            tracker.update(np.empty((0, 7), dtype=np.float32))
        with pytest.raises(ValueError, match="fixed to AABB"):
            tracker.update(
                Detections(
                    geometry=OrientedBoxes(torch.empty((0, 5), dtype=torch.float32)),
                    scores=torch.empty((0,), dtype=torch.float32),
                    class_ids=torch.empty((0,), dtype=torch.int64),
                    sample_id="sample",
                )
            )
    finally:
        tracker.close()


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_native_bytetrack_live_v2_returns_packed_numpy_tracks(geometry: str):
    library = native_binding.ByteTrackLibrary(native_binding.ensure_bytetrack_cpp_library())
    tracker = native_module.NativeByteTrackTracker(
        {"min_conf": 0.01, "track_thresh": 0.1, "match_thresh": 0.8},
        geometry=geometry,
        library=library,
    )
    rows = (
        np.array([[10, 10, 30, 40, 0.95, 2**31 + 17]], dtype=np.float64)
        if geometry == "aabb"
        else np.array([[20, 25, 20, 30, 0.3, 0.95, 2**31 + 17]], dtype=np.float64)
    )

    try:
        tracks = tracker.update(rows)
    finally:
        tracker.close()

    geometry_columns = 5 if geometry == "obb" else 4
    assert type(tracks) is np.ndarray
    assert tracks.dtype == np.float64
    assert tracks.shape == (1, geometry_columns + 4)
    assert tracks[0, geometry_columns + 2] == 2**31 + 17
    assert tracks[0, geometry_columns + 3] == 0
    assert len(rows) == len(tracks)


def test_native_bytetrack_empty_update_preserves_obb_mode():
    library = native_binding.ByteTrackLibrary(native_binding.ensure_bytetrack_cpp_library())
    tracker = native_module.NativeByteTrackTracker(geometry="obb", library=library)
    try:
        tracks = update_tracks(tracker, np.empty((0, 7), dtype=np.float32))
    finally:
        tracker.close()
    assert tracks.is_obb is True
    assert tracks.geometry.values.shape == (0, 5)
    assert tracks.track_ids.dtype == torch.int64


def test_native_bytetrack_reset_starts_fresh_sequence_ids():
    library = native_binding.ByteTrackLibrary(native_binding.ensure_bytetrack_cpp_library())
    tracker = native_module.NativeByteTrackTracker(
        {"min_conf": 0.01, "track_thresh": 0.1},
        library=library,
    )
    rows = np.array([[10, 10, 20, 20, 0.95, 0]], dtype=np.float32)
    try:
        first = update_rows(tracker, rows)
        tracker.reset()
        second = update_rows(tracker, rows)
    finally:
        tracker.close()
    assert first.shape == second.shape == (1, 8)
    assert first[0, 4] == second[0, 4] == 1
