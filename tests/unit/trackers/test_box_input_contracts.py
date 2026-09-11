"""Resolved input consumption for geometric, appearance, and dimension-only trackers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from boxmot.native.trackers._common import NativeTrackBatch
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes
from boxmot.trackers.bytetrack.native import NativeByteTrackTracker
from boxmot.trackers.common import native
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.registry import get_tracker_class
from boxmot.trackers.ocsort.native import NativeOcSortTracker
from boxmot.trackers.sfsort.native import NativeSFSORTTracker


def _tracker(name: str, **overrides: Any):
    """Exercise authored defaults while disabling no features implicitly."""
    return get_tracker_class(name)(**{**load_tracker_defaults(name), **overrides})


def _detections(*, is_obb: bool = False, embeddings: bool = False) -> Detections:
    geometry = (
        OrientedBoxes(torch.tensor([[20.0, 24.0, 12.0, 20.0, 0.2]]))
        if is_obb
        else Boxes(torch.tensor([[14.0, 14.0, 26.0, 34.0]]))
    )
    return Detections(
        geometry,
        scores=torch.tensor([0.95]),
        class_ids=torch.tensor([1]),
        sample_id="camera:0",
        embeddings=torch.tensor([[1.0, 0.0]]) if embeddings else None,
    )


def _frame() -> Frame:
    return Frame(torch.full((3, 48, 64), 127, dtype=torch.uint8), sample_id="camera:0")


@pytest.mark.parametrize("name", ("bytetrack", "ocsort"))
@pytest.mark.parametrize("is_obb", (False, True))
def test_geometric_defaults_track_boxes_without_frames_or_appearance(name: str, is_obb: bool) -> None:
    tracker = _tracker(name, is_obb=is_obb)
    assert not tracker.requirements.frame
    assert not tracker.requirements.embeddings
    assert not tracker.generates_embeddings
    for _ in range(2):
        tracker.update(_detections(is_obb=is_obb))
    assert tracker.frame_count == 2


@pytest.mark.parametrize("name", ("bytetrack", "ocsort", "sfsort", "hybridsort"))
@pytest.mark.parametrize("is_obb", (False, True))
def test_dimension_only_association_does_not_convert_image_pixels(
    name: str, is_obb: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = {"use_embeddings": False, "cmc_method": None} if name == "hybridsort" else {}
    tracker = _tracker(name, asso_func="centroid", is_obb=is_obb, **options)
    assert tracker.requirements.frame_dimensions_only
    assert not tracker.requirements.frame_pixels

    def forbidden(frame: Frame) -> np.ndarray:
        pytest.fail("Dimension-only association must not convert source-image pixels")

    monkeypatch.setattr(tracker, "_frame_to_bgr", forbidden)
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(_detections(is_obb=is_obb))
    tracker.update(_detections(is_obb=is_obb), _frame())
    tracker.update(_detections(is_obb=is_obb), _frame())
    assert (tracker.w, tracker.h) == (64, 48)


@pytest.mark.parametrize("asso_func", ("iou", "centroid"))
def test_sfsort_configured_dimensions_remove_frame_requirement(asso_func: str) -> None:
    tracker = _tracker("sfsort", frame_width=64, frame_height=48, asso_func=asso_func)
    assert not tracker.requirements.frame
    assert not tracker.requirements.frame_dimensions_only
    tracker.update(_detections())
    assert (tracker.r_margin, tracker.b_margin) == (64, 48)
    tracker.reset()
    tracker.update(_detections())
    assert tracker.frame_count == 1


@pytest.mark.parametrize("is_obb", (False, True))
@pytest.mark.parametrize("use_embeddings", (False, True))
def test_hybrid_without_cmc_accepts_boxes_and_only_enabled_appearance(is_obb: bool, use_embeddings: bool) -> None:
    tracker = _tracker("hybridsort", cmc_method=None, use_embeddings=use_embeddings, is_obb=is_obb)
    assert not tracker.requirements.frame
    assert tracker.requirements.embeddings is use_embeddings
    for _ in range(2):
        tracker.update(_detections(is_obb=is_obb, embeddings=use_embeddings))
    assert tracker.frame_count == 2


@pytest.mark.parametrize("name", ("hybridsort", "strongsort"))
@pytest.mark.parametrize("asso_func", ("iou", "centroid"))
def test_ecc_consumes_pixels_even_with_cached_embeddings_and_centroid(
    name: str, asso_func: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _tracker(name, asso_func=asso_func)
    assert tracker.requirements.embeddings
    assert tracker.requirements.frame_pixels
    calls: list[np.ndarray] = []

    def apply(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        calls.append(image.copy())
        return np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(tracker.cmc, "apply", apply)
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(_detections(embeddings=True))
    tracker.update(_detections(embeddings=True), _frame())
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], np.full((48, 64, 3), 127, dtype=np.uint8))


def test_hybrid_live_reid_requires_pixels_only_when_cached_embeddings_are_absent() -> None:
    calls: list[np.ndarray] = []

    class Encoder:
        def get_features(self, boxes: np.ndarray, image: np.ndarray) -> np.ndarray:
            calls.append(image.copy())
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(boxes), 1))

    tracker = _tracker("hybridsort", cmc_method=None, reid_model=Encoder())
    tracker.update(_detections(embeddings=True))
    assert not calls
    with pytest.raises(ValueError, match="requires a frame to generate"):
        tracker.update(_detections())
    tracker.update(_detections(), _frame())
    assert len(calls) == 1


@pytest.mark.parametrize("name", ("bytetrack", "ocsort", "sfsort", "hybridsort", "strongsort"))
def test_box_trackers_reject_unused_mask_payloads(name: str) -> None:
    tracker = _tracker(name)
    detections = replace(
        _detections(embeddings=tracker.requirements.embeddings),
        masks=MaskBatch(torch.ones((1, 48, 64), dtype=torch.bool)),
    )
    with pytest.raises(ValueError, match="masks"):
        tracker.update(detections, _frame())
    assert tracker.frame_count == 0


@pytest.mark.parametrize("name", ("bytetrack", "ocsort", "sfsort", "hybridsort"))
def test_appearance_disabled_trackers_reject_unused_embeddings(name: str) -> None:
    options = {"use_embeddings": False, "cmc_method": None} if name == "hybridsort" else {}
    tracker = _tracker(name, **options)
    with pytest.raises(ValueError, match="embeddings"):
        tracker.update(_detections(embeddings=True), _frame())
    assert tracker.frame_count == 0


class _NativeLibrary:
    """Capture adapter inputs without compiling or downloading native libraries."""

    def __init__(self) -> None:
        self.images: list[np.ndarray | None] = []

    def create(self, config: dict[str, Any]) -> int:
        return 1

    def destroy(self, handle: int) -> None:
        pass

    def reset(self, handle: int) -> None:
        pass

    def update(self, handle: int, **batch: Any) -> NativeTrackBatch:
        self.images.append(batch["image"])
        return NativeTrackBatch(
            geometry=np.empty((0, batch["geometry"].shape[1]), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            track_ids=np.empty(0, dtype=np.int64),
            class_ids=np.empty(0, dtype=np.int64),
            detection_indices=np.empty(0, dtype=np.int64),
        )


@pytest.mark.parametrize("tracker_class", (NativeByteTrackTracker, NativeOcSortTracker, NativeSFSORTTracker))
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_native_centroid_adapters_request_dimensions_without_source_pixels(
    tracker_class: type, geometry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _NativeLibrary()
    tracker = tracker_class(options={"asso_func": "centroid"}, geometry=geometry, library=library)
    assert tracker.requirements.frame_dimensions_only
    assert not tracker.requirements.frame_pixels
    original = native._frame_to_bgr

    def convert(frame: Frame, *, dimensions_only: bool = False, placeholders=None) -> np.ndarray:
        assert dimensions_only
        return original(frame, dimensions_only=dimensions_only, placeholders=placeholders)

    monkeypatch.setattr(native, "_frame_to_bgr", convert)
    tracker.update(_detections(is_obb=geometry == "obb"), _frame())
    assert library.images[-1].shape == (48, 64, 3)


@pytest.mark.parametrize("tracker_class", (NativeByteTrackTracker, NativeOcSortTracker, NativeSFSORTTracker))
@pytest.mark.parametrize("channel", ("masks", "embeddings"))
def test_native_geometric_adapters_reject_unused_detection_channels(tracker_class: type, channel: str) -> None:
    library = _NativeLibrary()
    tracker = tracker_class(library=library)
    detections = _detections(embeddings=channel == "embeddings")
    if channel == "masks":
        detections = replace(detections, masks=MaskBatch(torch.ones((1, 48, 64), dtype=torch.bool)))
    with pytest.raises(ValueError, match=channel):
        tracker.update(detections, _frame())
    assert library.images == []
