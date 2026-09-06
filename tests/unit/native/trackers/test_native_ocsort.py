from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot.native.trackers import ocsort as native_binding
from boxmot.structures import Boxes, Detections, OrientedBoxes, Tracks
from boxmot.trackers.box.ocsort import native as native_module

from ._helpers import empty_native_batch


class _FakeLibrary:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create(self, cfg):
        self.calls.append(("create", cfg["det_thresh"], cfg["iou_threshold"], cfg["use_byte"]))
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


def _detections(*, obb: bool = False) -> Detections:
    geometry = (
        OrientedBoxes(torch.tensor([[3.0, 4.0, 2.0, 3.0, 0.2]], dtype=torch.float32))
        if obb
        else Boxes(torch.tensor([[1.0, 1.0, 4.0, 5.0]], dtype=torch.float32))
    )
    return Detections(
        geometry=geometry,
        scores=torch.tensor([0.9], dtype=torch.float32),
        class_ids=torch.tensor([2**31 + 9], dtype=torch.int64),
        sample_id="sample",
    )


def test_native_ocsort_is_internal_and_declares_structured_requirements() -> None:
    assert native_module.NativeOcSortTracker.__name__ == "NativeOcSortTracker"
    assert not hasattr(native_binding, "NativeOcSortTracker")
    tracker = native_module.NativeOcSortTracker(library=_FakeLibrary())
    assert tracker.supports_masks is False
    assert tracker.requirements.embeddings is False
    assert tracker.requirements.frame is False
    tracker.close()


def test_native_ocsort_uses_structured_live_library_wrapper() -> None:
    library = _FakeLibrary()
    tracker = native_module.NativeOcSortTracker(
        {"det_thresh": 0.55, "iou_threshold": 0.27, "use_byte": True},
        geometry="aabb",
        library=library,
    )
    output = tracker.update(_detections())
    tracker.reset()
    tracker.close()
    assert isinstance(output, Tracks)
    assert len(output) == 0
    assert output.to_aabb_rows().shape == (0, 8)
    assert library.calls == [
        ("create", 0.55, 0.27, True),
        ("update", "handle", 1, None, 4, None),
        ("reset", "handle"),
        ("destroy", "handle"),
    ]


def test_native_ocsort_accepts_numpy_aabb6_and_rejects_geometry_mismatch() -> None:
    library = _FakeLibrary()
    tracker = native_module.NativeOcSortTracker(geometry="aabb", library=library)
    try:
        output = tracker.update(np.array([[1, 1, 4, 5, 0.9, 3]], dtype=np.float64))
        assert output.sample_id == "numpy:000000"
        assert library.calls[1] == ("update", "handle", 1, None, 4, None)

        with pytest.raises(ValueError, match=r"AABB detection rows must have shape \[N, 6\]"):
            tracker.update(np.empty((0, 7), dtype=np.float32))
        with pytest.raises(ValueError, match="fixed to AABB"):
            tracker.update(_detections(obb=True))
    finally:
        tracker.close()


def test_native_ocsort_centroid_association_requires_frame() -> None:
    tracker = native_module.NativeOcSortTracker(
        {"asso_func": "centroid"},
        library=_FakeLibrary(),
    )
    assert tracker.requirements.frame is True
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(_detections())
    tracker.close()


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_native_ocsort_v2_emits_canonical_tracks(geometry: str) -> None:
    library = native_binding.OcSortLibrary(native_binding.ensure_ocsort_cpp_library())
    tracker = native_module.NativeOcSortTracker(
        {"min_hits": 1, "det_thresh": 0.1, "iou_threshold": 0.3},
        geometry=geometry,
        library=library,
    )
    rows = (
        np.array([[1, 1, 4, 5, 0.9, 2**31 + 9]], dtype=np.float64)
        if geometry == "aabb"
        else np.array([[3, 4, 2, 3, 0.2, 0.9, 2**31 + 9]], dtype=np.float64)
    )
    try:
        tracks = tracker.update(rows)
    finally:
        tracker.close()
    assert tracks.is_obb is (geometry == "obb")
    assert tracks.class_ids.tolist() == [2**31 + 9]
    assert tracks.detection_indices.tolist() == [0]
    assert tracks.sample_id == "numpy:000000"
