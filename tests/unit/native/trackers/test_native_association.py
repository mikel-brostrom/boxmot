from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from boxmot.native.trackers import botsort as botsort_binding
from boxmot.native.trackers import bytetrack as bytetrack_binding
from boxmot.native.trackers import occluboost as occluboost_binding
from boxmot.native.trackers import ocsort as ocsort_binding
from boxmot.native.trackers import sfsort as sfsort_binding
from boxmot.structures import MaskBatch
from boxmot.trackers.botsort import native as botsort
from boxmot.trackers.bytetrack import native as bytetrack
from boxmot.trackers.occluboost import native as occluboost
from boxmot.trackers.ocsort import native as ocsort
from boxmot.trackers.sfsort import native as sfsort

from ._helpers import detections_from_rows, empty_native_batch, frame_from_bgr, update_rows


class _FakeLibrary:
    def __init__(self) -> None:
        self.cfg = None
        self.calls: list[tuple[np.ndarray, np.ndarray | None, int, bool]] = []

    def create(self, cfg):
        self.cfg = cfg
        return "handle"

    def update(
        self,
        _handle,
        *,
        geometry,
        scores,
        class_ids,
        detection_indices,
        embeddings,
        image,
    ):
        self.calls.append((geometry, image, geometry.shape[1], embeddings is not None))
        return empty_native_batch(geometry.shape[1])

    def reset(self, _handle):
        return None

    def destroy(self, _handle):
        return None


TRACKERS = (
    (bytetrack.NativeByteTrackTracker, {}),
    (botsort.NativeBotSortTracker, {"use_cmc": False, "use_embeddings": False}),
    (ocsort.NativeOcSortTracker, {}),
    (sfsort.NativeSFSORTTracker, {}),
    (occluboost.NativeOccluBoostTracker, {"use_cmc": False, "use_embeddings": False}),
)


def test_native_adapter_revalidates_mutable_canonical_tensors() -> None:
    library = _FakeLibrary()
    tracker = bytetrack.NativeByteTrackTracker(geometry="aabb", library=library)
    detections = detections_from_rows(np.array([[10, 10, 20, 20, 0.95, 0]], dtype=np.float32))
    detections.scores[0] = 1.5

    with pytest.raises(ValueError, match="inclusive range"):
        tracker.update(detections)

    assert library.calls == []
    tracker.close()


@pytest.mark.parametrize("mask_shape", ((40, 50), (80, 100)))
def test_native_adapter_rejects_unused_masks_before_spatial_alignment(mask_shape: tuple[int, int]) -> None:
    library = _FakeLibrary()
    tracker = bytetrack.NativeByteTrackTracker(geometry="aabb", library=library)
    detections = detections_from_rows(np.array([[10, 10, 20, 20, 0.95, 0]], dtype=np.float32)).with_masks(
        MaskBatch(torch.zeros((1, *mask_shape), dtype=torch.bool))
    )
    frame = frame_from_bgr(np.zeros((80, 100, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="does not use detection masks"):
        tracker.update(detections, frame)

    assert library.calls == []
    tracker.close()


RESOLVERS = (
    bytetrack._resolve_tracker_config,
    botsort._resolve_tracker_config,
    ocsort._resolve_tracker_config,
    sfsort._resolve_tracker_config,
    occluboost._resolve_tracker_config,
)


@pytest.mark.parametrize(("tracker_cls", "options"), TRACKERS)
def test_centroid_requirement_is_frozen_and_routes_canonical_frame(tracker_cls, options):
    library = _FakeLibrary()
    tracker = tracker_cls({**options, "asso_func": "centroid"}, library=library)
    detections = detections_from_rows(np.array([[10, 10, 20, 20, 0.95, 0]], dtype=np.float32))
    frame = frame_from_bgr(np.zeros((80, 100, 3), dtype=np.uint8))

    assert tracker.requirements.frame is True
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(detections)
    tracker.update(detections, frame)

    assert len(library.calls) == 1
    np.testing.assert_array_equal(library.calls[0][0], detections.geometry.values.numpy())
    routed_image = library.calls[0][1]
    if tracker.requirements.frame_dimensions_only:
        assert routed_image.shape == (80, 100, 3)
        assert routed_image.dtype == np.uint8
    else:
        np.testing.assert_array_equal(routed_image, np.zeros((80, 100, 3), dtype=np.uint8))
    tracker.close()


@pytest.mark.parametrize("resolver", RESOLVERS)
def test_native_trackers_reject_unknown_association_function(resolver: Callable):
    with pytest.raises(ValueError, match="Unknown association function"):
        resolver({"asso_func": "made-up"})


@pytest.mark.parametrize("resolver", RESOLVERS)
def test_native_trackers_reject_noncanonical_association_casing(resolver: Callable):
    with pytest.raises(ValueError, match="canonical lowercase identifier"):
        resolver({"asso_func": "IoU"})


@pytest.mark.parametrize("resolver", RESOLVERS)
def test_native_trackers_reject_unknown_config_keys(resolver: Callable):
    with pytest.raises(TypeError, match="unexpected option 'legacy_option'"):
        resolver({"legacy_option": True})


@pytest.mark.parametrize(("tracker_cls", "options"), TRACKERS)
@pytest.mark.parametrize("asso_func", ["iou", "giou", "diou", "ciou", "hmiou", "centroid"])
def test_native_trackers_accept_all_obb_association_functions(tracker_cls, options, asso_func):
    library = _FakeLibrary()
    tracker = tracker_cls(
        {**options, "asso_func": asso_func},
        geometry="obb",
        library=library,
    )
    rows = np.array([[10, 10, 20, 10, 0.2, 0.95, 0]], dtype=np.float32)
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    tracks = update_rows(tracker, rows, image)

    assert tracks.shape == (0, 9)
    assert tracker.cfg["asso_func"] == asso_func
    tracker.close()


NATIVE_AABB_CASES = (
    (
        bytetrack.NativeByteTrackTracker,
        bytetrack_binding.ByteTrackLibrary,
        bytetrack_binding.ensure_bytetrack_cpp_library,
        {"min_conf": 0.01, "track_thresh": 0.1, "match_thresh": 0.2},
    ),
    (
        botsort.NativeBotSortTracker,
        botsort_binding.BotSortLibrary,
        botsort_binding.ensure_botsort_cpp_library,
        {
            "use_embeddings": False,
            "use_cmc": False,
            "track_high_thresh": 0.1,
            "track_low_thresh": 0.01,
            "new_track_thresh": 0.1,
            "match_thresh": 0.2,
            "fuse_first_associate": False,
        },
    ),
    (
        ocsort.NativeOcSortTracker,
        ocsort_binding.OcSortLibrary,
        ocsort_binding.ensure_ocsort_cpp_library,
        {
            "min_conf": 0.01,
            "det_thresh": 0.1,
            "iou_threshold": 0.8,
            "min_hits": 0,
            "inertia": 0.0,
        },
    ),
    (
        sfsort.NativeSFSORTTracker,
        sfsort_binding.SFSORTLibrary,
        sfsort_binding.ensure_sfsort_cpp_library,
        {
            "high_th": 0.1,
            "new_track_th": 0.1,
            "low_th": 0.01,
            "match_th_first": 0.15,
            "dynamic_tuning": False,
        },
    ),
    (
        occluboost.NativeOccluBoostTracker,
        occluboost_binding.OccluBoostLibrary,
        occluboost_binding.ensure_occluboost_cpp_library,
        {
            "use_embeddings": False,
            "use_cmc": False,
            "use_dlo_boost": False,
            "use_duo_boost": False,
            "use_second_pass": False,
            "det_thresh": 0.1,
            "iou_threshold": 0.8,
            "new_track_thresh": 0.1,
            "instant_confirm_thresh": 0.1,
            "confirm_hits": 1,
            "min_hits": 0,
            "min_box_area": 1,
            "aspect_ratio_thresh": 20,
        },
    ),
)


@pytest.mark.parametrize(
    ("tracker_cls", "library_cls", "ensure_library", "options"),
    NATIVE_AABB_CASES,
    ids=("bytetrack", "botsort", "ocsort", "sfsort", "occluboost"),
)
def test_live_native_tracker_uses_selected_aabb_association(
    tracker_cls,
    library_cls,
    ensure_library,
    options,
):
    library_path = ensure_library()
    first_detections = np.array([[5, 5, 15, 15, 0.95, 0]], dtype=np.float32)
    shifted_detections = np.array([[10, 5, 20, 15, 0.95, 0]], dtype=np.float32)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    outputs = {}

    for mode in ("iou", "centroid"):
        tracker = tracker_cls(
            {**options, "asso_func": mode},
            geometry="aabb",
            library=library_cls(library_path),
        )
        try:
            first = update_rows(tracker, first_detections, image)
            second = update_rows(tracker, shifted_detections, image)
        finally:
            tracker.close()
        outputs[mode] = (first, second)

    centroid_first, centroid_second = outputs["centroid"]
    iou_first, iou_second = outputs["iou"]
    assert centroid_first.shape == (1, 8)
    assert centroid_second.shape == (1, 8)
    assert centroid_second[0, 4] == centroid_first[0, 4]
    assert iou_first.shape == (1, 8)
    assert iou_second.shape[0] == 0 or iou_second[0, 4] != iou_first[0, 4]
