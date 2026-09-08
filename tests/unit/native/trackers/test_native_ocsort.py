from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.native.trackers import ocsort as native_binding
from boxmot.structures import Boxes, Detections, OrientedBoxes, Tracks
from boxmot.trackers.box.ocsort import native as native_module
from boxmot.trackers.box.ocsort.tracker import OcSort
from boxmot.trackers.config import load_tracker_config

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
        assert type(output) is np.ndarray
        assert output.shape == (0, 8)
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


@pytest.mark.parametrize("option", ("Q_xy_scaling", "Q_s_scaling", "Q_a_scaling"))
def test_native_ocsort_rejects_removed_noise_options(option: str) -> None:
    library = _FakeLibrary()
    with pytest.raises(TypeError, match=option):
        native_module.NativeOcSortTracker({option: 0.01}, library=library)
    assert library.calls == []


def test_native_ocsort_rejects_stale_config_abi_before_creating_a_tracker(monkeypatch) -> None:
    calls = []
    stale_library = SimpleNamespace(boxmot_ocsort_create=lambda _config: calls.append("unsafe create"))
    monkeypatch.setattr(native_binding.ctypes, "CDLL", lambda _path: stale_library)
    with pytest.raises(RuntimeError, match="incompatible configuration ABI"):
        native_binding.OcSortLibrary(Path("stale-native-library"))
    assert calls == []


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_native_ocsort_v2_emits_packed_numpy_tracks(geometry: str) -> None:
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
    geometry_columns = 5 if geometry == "obb" else 4
    assert type(tracks) is np.ndarray
    assert tracks.dtype == np.float64
    assert tracks.shape == (1, geometry_columns + 4)
    assert tracks[0, geometry_columns + 2] == 2**31 + 9
    assert tracks[0, geometry_columns + 3] == 0


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_native_ocsort_fixed_process_noise_preserves_python_parity_through_misses(geometry: str) -> None:
    """Changing box size and missing updates exercise center/area/angle dynamics."""
    options = {"min_hits": 1, "det_thresh": 0.1, "iou_threshold": 0.1}
    python_tracker = OcSort(**load_tracker_config("ocsort", None, options), is_obb=geometry == "obb")
    library = native_binding.OcSortLibrary(native_binding.ensure_ocsort_cpp_library())
    tracker = native_module.NativeOcSortTracker(options, geometry=geometry, library=library)
    geometry_columns = 5 if geometry == "obb" else 4
    try:
        for index in range(12):
            box = (
                [10 + index * 0.4, 20 + index * 0.2, 40 + index * 0.6, 70 + index * 0.5]
                if geometry == "aabb"
                else [30 + index * 0.4, 45 + index * 0.2, 30 + index * 0.2, 50 + index * 0.3, 0.2 + index * 0.01]
            )
            detections = (
                np.empty((0, geometry_columns + 2), dtype=np.float64)
                if index in (4, 5)
                else np.asarray([[*box, 0.9, 1]], dtype=np.float64)
            )
            expected = python_tracker.update(detections)
            actual = tracker.update(detections)
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual[:, :geometry_columns], expected[:, :geometry_columns], atol=1e-5)
            # Native IDs have their own allocator; geometry, score, class and
            # detection index must agree without depending on initial ID offset.
            np.testing.assert_allclose(actual[:, geometry_columns + 1 :], expected[:, geometry_columns + 1 :])
    finally:
        tracker.close()
