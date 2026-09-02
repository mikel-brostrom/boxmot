from __future__ import annotations

import numpy as np
import pytest

from boxmot.native.trackers import botsort, bytetrack, occluboost, ocsort, sfsort


class _FakeLibrary:
    def __init__(self) -> None:
        self.cfg = None
        self.images = []

    def create(self, cfg):
        self.cfg = cfg
        return "handle"

    def update(self, handle, dets, img, *args):
        del handle, args
        self.images.append(img)
        columns = 9 if dets.shape[1] == 7 else 8
        return np.empty((0, columns), dtype=np.float32)

    def reset(self, handle):
        del handle

    def destroy(self, handle):
        del handle


@pytest.mark.parametrize(
    ("tracker_cls", "cfg"),
    [
        (bytetrack.NativeByteTrackTracker, {}),
        (botsort.NativeBotSortTracker, {"use_cmc": False, "with_reid": False}),
        (ocsort.NativeOCSORTTracker, {}),
        (sfsort.NativeSFSORTTracker, {}),
        (occluboost.NativeOccluBoostTracker, {"use_cmc": False, "with_reid": False}),
    ],
)
def test_native_trackers_normalize_and_route_centroid_image_once(tracker_cls, cfg):
    library = _FakeLibrary()
    tracker = tracker_cls({**cfg, "asso_func": " Centroid "}, library=library)
    dets = np.array([[10, 10, 20, 20, 0.95, 0]], dtype=np.float32)
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    assert tracker.cfg["asso_func"] == "centroid"
    assert tracker.requires_image(dets)
    with pytest.raises(ValueError, match="requires img"):
        tracker.update(dets)

    tracker.update(dets, image)
    assert not tracker.requires_image(dets)
    tracker.update(dets)
    assert library.images[0] is image
    assert library.images[1] is None

    tracker.reset()
    assert tracker.requires_image(dets)
    tracker.close()


@pytest.mark.parametrize(
    "tracker_cls",
    [
        bytetrack.NativeByteTrackTracker,
        botsort.NativeBotSortTracker,
        ocsort.NativeOCSORTTracker,
        occluboost.NativeOccluBoostTracker,
    ],
)
def test_only_sfsort_accepts_configured_centroid_frame_dimensions(tracker_cls):
    tracker = tracker_cls(
        {
            "asso_func": "centroid",
            "frame_width": 100,
            "frame_height": 80,
            "use_cmc": False,
            "with_reid": False,
        },
        library=_FakeLibrary(),
    )
    dets = np.array([[10, 10, 20, 20, 0.95, 0]], dtype=np.float32)

    assert tracker.requires_image(dets)
    tracker.close()


def test_sfsort_accepts_configured_centroid_frame_dimensions():
    library = _FakeLibrary()
    tracker = sfsort.NativeSFSORTTracker(
        {
            "asso_func": "centroid",
            "frame_width": 100,
            "frame_height": 80,
        },
        library=library,
    )
    dets = np.array([[10, 10, 20, 20, 0.95, 0]], dtype=np.float32)

    assert not tracker.requires_image(dets)
    tracker.update(dets)
    assert library.images == [None]
    tracker.close()


@pytest.mark.parametrize(
    "resolver",
    [
        bytetrack._resolve_tracker_cfg,
        botsort._resolve_tracker_cfg,
        ocsort._resolve_tracker_cfg,
        sfsort._resolve_tracker_cfg,
        occluboost._resolve_tracker_cfg,
    ],
)
def test_native_trackers_reject_unknown_association_function(resolver):
    with pytest.raises(ValueError, match="Unknown association function"):
        resolver({"asso_func": "made-up"})


@pytest.mark.parametrize(
    "tracker_cls",
    [
        bytetrack.NativeByteTrackTracker,
        botsort.NativeBotSortTracker,
        ocsort.NativeOCSORTTracker,
        sfsort.NativeSFSORTTracker,
        occluboost.NativeOccluBoostTracker,
    ],
)
@pytest.mark.parametrize("asso_func", ["iou", "giou", "diou", "ciou", "hmiou", "centroid"])
def test_native_trackers_accept_all_obb_association_functions(tracker_cls, asso_func):
    library = _FakeLibrary()
    cfg = {"asso_func": asso_func, "use_cmc": False, "with_reid": False}
    tracker = tracker_cls(cfg, library=library)
    dets = np.array([[10, 10, 20, 10, 0.2, 0.95, 0]], dtype=np.float32)
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    tracks = tracker.update(dets, image)

    assert tracks.shape == (0, 9)
    assert tracker.cfg["asso_func"] == asso_func
    tracker.close()


@pytest.mark.parametrize(
    ("tracker_cls", "library_cls", "ensure_library", "cfg"),
    [
        (
            bytetrack.NativeByteTrackTracker,
            bytetrack._ByteTrackLiveLibrary,
            bytetrack.ensure_bytetrack_cpp_library,
            {},
        ),
        (
            botsort.NativeBotSortTracker,
            botsort._BotSortLiveLibrary,
            botsort.ensure_botsort_cpp_library,
            {"use_cmc": False, "with_reid": False},
        ),
        (
            ocsort.NativeOCSORTTracker,
            ocsort._OCSORTLiveLibrary,
            ocsort.ensure_ocsort_cpp_library,
            {},
        ),
        (
            sfsort.NativeSFSORTTracker,
            sfsort._SFSORTLiveLibrary,
            sfsort.ensure_sfsort_cpp_library,
            {},
        ),
        (
            occluboost.NativeOccluBoostTracker,
            occluboost._OccluBoostLiveLibrary,
            occluboost.ensure_occluboost_cpp_library,
            {"use_cmc": False, "with_reid": False},
        ),
    ],
)
def test_live_native_centroid_receives_and_caches_initial_frame_dimensions(
    tracker_cls,
    library_cls,
    ensure_library,
    cfg,
):
    library = library_cls(ensure_library())
    tracker = tracker_cls({**cfg, "asso_func": "centroid"}, library=library)
    dets = np.array([[10, 10, 20, 20, 0.95, 0]], dtype=np.float32)
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    try:
        first = tracker.update(dets, image)
        second = tracker.update(dets)
    finally:
        tracker.close()

    assert first.shape[1] == 8
    assert second.shape[1] == 8


@pytest.mark.parametrize(
    ("tracker_cls", "library_cls", "ensure_library", "cfg"),
    [
        (
            bytetrack.NativeByteTrackTracker,
            bytetrack._ByteTrackLiveLibrary,
            bytetrack.ensure_bytetrack_cpp_library,
            {"min_conf": 0.01, "track_thresh": 0.1, "match_thresh": 0.2},
        ),
        (
            botsort.NativeBotSortTracker,
            botsort._BotSortLiveLibrary,
            botsort.ensure_botsort_cpp_library,
            {
                "with_reid": False,
                "use_cmc": False,
                "track_high_thresh": 0.1,
                "track_low_thresh": 0.01,
                "new_track_thresh": 0.1,
                "match_thresh": 0.2,
                "fuse_first_associate": False,
            },
        ),
        (
            ocsort.NativeOCSORTTracker,
            ocsort._OCSORTLiveLibrary,
            ocsort.ensure_ocsort_cpp_library,
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
            sfsort._SFSORTLiveLibrary,
            sfsort.ensure_sfsort_cpp_library,
            {
                "high_th": 0.1,
                "new_track_th": 0.1,
                "low_th": 0.01,
                "match_th_first": 0.15,
                "dynamic_tuning": False,
                "frame_width": 100,
                "frame_height": 100,
            },
        ),
        (
            occluboost.NativeOccluBoostTracker,
            occluboost._OccluBoostLiveLibrary,
            occluboost.ensure_occluboost_cpp_library,
            {
                "with_reid": False,
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
    ],
)
def test_live_native_tracker_uses_selected_geometry_for_matching(
    tracker_cls,
    library_cls,
    ensure_library,
    cfg,
):
    library_path = ensure_library()
    first_dets = np.array([[5, 5, 15, 15, 0.95, 0]], dtype=np.float32)
    shifted_dets = np.array([[10, 5, 20, 15, 0.95, 0]], dtype=np.float32)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    outputs = {}

    for mode in ("iou", "centroid"):
        tracker = tracker_cls({**cfg, "asso_func": mode}, library=library_cls(library_path))
        try:
            first = tracker.update(first_dets, image)
            second = tracker.update(shifted_dets)
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


@pytest.mark.parametrize(
    ("tracker_cls", "library_cls", "ensure_library", "cfg"),
    [
        (
            bytetrack.NativeByteTrackTracker,
            bytetrack._ByteTrackLiveLibrary,
            bytetrack.ensure_bytetrack_cpp_library,
            {"min_conf": 0.01, "track_thresh": 0.1, "match_thresh": 0.5},
        ),
        (
            botsort.NativeBotSortTracker,
            botsort._BotSortLiveLibrary,
            botsort.ensure_botsort_cpp_library,
            {
                "with_reid": False,
                "use_cmc": False,
                "track_high_thresh": 0.1,
                "track_low_thresh": 0.01,
                "new_track_thresh": 0.1,
                "match_thresh": 0.5,
                "fuse_first_associate": False,
            },
        ),
        (
            ocsort.NativeOCSORTTracker,
            ocsort._OCSORTLiveLibrary,
            ocsort.ensure_ocsort_cpp_library,
            {
                "min_conf": 0.01,
                "det_thresh": 0.1,
                "iou_threshold": 0.5,
                "min_hits": 0,
                "inertia": 0.0,
            },
        ),
        (
            sfsort.NativeSFSORTTracker,
            sfsort._SFSORTLiveLibrary,
            sfsort.ensure_sfsort_cpp_library,
            {
                "high_th": 0.1,
                "new_track_th": 0.1,
                "low_th": 0.01,
                "match_th_first": 0.2,
                "dynamic_tuning": False,
                "frame_width": 100,
                "frame_height": 100,
            },
        ),
        (
            occluboost.NativeOccluBoostTracker,
            occluboost._OccluBoostLiveLibrary,
            occluboost.ensure_occluboost_cpp_library,
            {
                "with_reid": False,
                "use_cmc": False,
                "use_dlo_boost": False,
                "use_duo_boost": False,
                "use_second_pass": False,
                "obb_det_thresh": 0.1,
                "obb_iou_threshold": 0.5,
                "obb_new_track_thresh": 0.1,
                "obb_instant_confirm_thresh": 0.1,
                "confirm_hits": 1,
                "min_hits": 0,
                "min_box_area": 1,
                "aspect_ratio_thresh": 20,
            },
        ),
    ],
)
def test_live_native_tracker_uses_selected_obb_geometry_for_matching(
    tracker_cls,
    library_cls,
    ensure_library,
    cfg,
):
    library_path = ensure_library()
    first_dets = np.array([[10, 10, 10, 10, 0, 0.95, 0]], dtype=np.float32)
    shifted_dets = np.array([[15, 10, 10, 10, 0, 0.95, 0]], dtype=np.float32)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    outputs = {}

    for mode in ("iou", "diou"):
        tracker = tracker_cls({**cfg, "asso_func": mode}, library=library_cls(library_path))
        try:
            first = tracker.update(first_dets, image)
            second = tracker.update(shifted_dets)
        finally:
            tracker.close()
        outputs[mode] = (first, second)

    diou_first, diou_second = outputs["diou"]
    iou_first, iou_second = outputs["iou"]
    assert diou_first.shape == (1, 9)
    assert diou_second.shape == (1, 9)
    assert diou_second[0, 5] == diou_first[0, 5]
    assert iou_first.shape == (1, 9)
    assert iou_second.shape[0] == 0 or iou_second[0, 5] != iou_first[0, 5]
