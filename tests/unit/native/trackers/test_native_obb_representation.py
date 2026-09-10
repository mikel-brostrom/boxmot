# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from boxmot.native.trackers import botsort as botsort_binding
from boxmot.native.trackers import bytetrack as bytetrack_binding
from boxmot.native.trackers import occluboost as occluboost_binding
from boxmot.native.trackers import ocsort as ocsort_binding
from boxmot.native.trackers import sfsort as sfsort_binding
from boxmot.trackers.botsort import native as botsort
from boxmot.trackers.bytetrack import native as bytetrack
from boxmot.trackers.occluboost import native as occluboost
from boxmot.trackers.ocsort import native as ocsort
from boxmot.trackers.sfsort import native as sfsort

from ._helpers import update_rows


def _native_bytetrack(match_thresh: float = 0.01):
    library = bytetrack_binding.ByteTrackLibrary(bytetrack_binding.ensure_bytetrack_cpp_library())
    return bytetrack.NativeByteTrackTracker(
        {
            "min_conf": 0.01,
            "track_thresh": 0.1,
            "match_thresh": match_thresh,
        },
        geometry="obb",
        library=library,
    )


def _native_botsort(match_thresh: float = 0.01):
    library = botsort_binding.BotSortLibrary(botsort_binding.ensure_botsort_cpp_library())
    return botsort.NativeBotSortTracker(
        {
            "use_embeddings": False,
            "use_cmc": False,
            "track_high_thresh": 0.1,
            "track_low_thresh": 0.01,
            "new_track_thresh": 0.1,
            "match_thresh": match_thresh,
            "fuse_first_associate": False,
        },
        geometry="obb",
        library=library,
    )


def _native_ocsort(iou_threshold: float = 0.99):
    library = ocsort_binding.OcSortLibrary(ocsort_binding.ensure_ocsort_cpp_library())
    return ocsort.NativeOcSortTracker(
        {
            "min_conf": 0.01,
            "det_thresh": 0.1,
            "iou_threshold": iou_threshold,
            "min_hits": 3,
            "inertia": 0.0,
        },
        geometry="obb",
        library=library,
    )


def _native_sfsort(match_th_first: float = 0.01):
    library = sfsort_binding.SFSORTLibrary(sfsort_binding.ensure_sfsort_cpp_library())
    return sfsort.NativeSFSORTTracker(
        {
            "high_th": 0.1,
            "new_track_th": 0.1,
            "low_th": 0.01,
            "match_th_first": match_th_first,
            "dynamic_tuning": False,
            "frame_width": 1800,
            "frame_height": 512,
        },
        geometry="obb",
        library=library,
    )


def _native_occluboost(obb_iou_threshold: float = 0.99):
    library = occluboost_binding.OccluBoostLibrary(
        occluboost_binding.ensure_occluboost_cpp_library()
    )
    return occluboost.NativeOccluBoostTracker(
        {
            "use_embeddings": False,
            "use_cmc": False,
            "use_dlo_boost": False,
            "use_duo_boost": False,
            "use_second_pass": False,
            "obb_det_thresh": 0.1,
            "obb_iou_threshold": obb_iou_threshold,
            "obb_new_track_thresh": 0.1,
            "obb_instant_confirm_thresh": 0.1,
            "confirm_hits": 1,
            "min_hits": 0,
            "min_box_area": 1,
            "aspect_ratio_thresh": 10.0,
        },
        geometry="obb",
        library=library,
    )


TRACKER_FACTORIES: tuple[tuple[str, Callable], ...] = (
    ("bytetrack", _native_bytetrack),
    ("botsort", _native_botsort),
    ("ocsort", _native_ocsort),
    ("sfsort", _native_sfsort),
    ("occluboost", _native_occluboost),
)

STRICT_TRACKER_FACTORIES: tuple[tuple[str, Callable], ...] = (
    ("bytetrack", lambda: _native_bytetrack(match_thresh=5.0e-7)),
    ("botsort", lambda: _native_botsort(match_thresh=5.0e-7)),
    ("ocsort", lambda: _native_ocsort(iou_threshold=0.999999)),
    ("sfsort", lambda: _native_sfsort(match_th_first=3.0e-7)),
    ("occluboost", lambda: _native_occluboost(obb_iou_threshold=0.999999)),
)


EQUIVALENT_OBB_PAIRS = (
    (
        "theta_plus_pi",
        [1497.5447, 131.3379, 109.4145, 49.4650, -0.4048741, 0.999, 0],
        [1497.5447, 131.3379, 109.4145, 49.4650, -0.4048741 + np.pi, 0.999, 0],
    ),
    (
        "swapped_width_height",
        [50.0, 50.0, 50.0, 140.0, 0.4, 0.999, 0],
        [50.0, 50.0, 140.0, 50.0, 0.4 + (np.pi / 2.0), 0.999, 0],
    ),
    (
        "square_theta_plus_half_pi",
        [150.0, 100.0, 100.0, 100.0, 0.3, 0.999, 0],
        [150.0, 100.0, 100.0, 100.0, 0.3 + (np.pi / 2.0), 0.999, 0],
    ),
)


@pytest.mark.parametrize(
    ("tracker_name", "tracker_factory"),
    TRACKER_FACTORIES,
    ids=[item[0] for item in TRACKER_FACTORIES],
)
@pytest.mark.parametrize(
    ("representation_name", "first_row", "equivalent_row"),
    EQUIVALENT_OBB_PAIRS,
    ids=[item[0] for item in EQUIVALENT_OBB_PAIRS],
)
def test_native_obb_equivalent_forms_preserve_track_id(
    tracker_name: str,
    tracker_factory: Callable,
    representation_name: str,
    first_row: list[float],
    equivalent_row: list[float],
):
    """Equivalent OBB parameterizations must have perfect overlap for association."""
    del tracker_name, representation_name
    tracker = tracker_factory()
    image = np.zeros((512, 1800, 3), dtype=np.uint8)
    first = np.asarray([first_row], dtype=np.float32)
    equivalent = np.asarray([equivalent_row], dtype=np.float32)

    try:
        first_output = update_rows(tracker, first, image)
        equivalent_output = update_rows(tracker, equivalent, image)
    finally:
        tracker.close()

    assert first_output.shape == (1, 9)
    assert equivalent_output.shape == (1, 9)
    assert equivalent_output[0, 5] == first_output[0, 5]


@pytest.mark.parametrize(
    ("tracker_name", "tracker_factory"),
    STRICT_TRACKER_FACTORIES,
    ids=[item[0] for item in STRICT_TRACKER_FACTORIES],
)
def test_native_obb_near_square_does_not_gain_square_periodicity(
    tracker_name: str,
    tracker_factory: Callable,
):
    """A near-square rotated by pi/2 is similar, but not equivalent, geometry."""
    del tracker_name
    tracker = tracker_factory()
    image = np.zeros((512, 1800, 3), dtype=np.uint8)
    first = np.asarray([[150.0, 100.0, 100.0, 100.00009, 0.3, 1.0, 0]], dtype=np.float32)
    rotated = first.copy()
    rotated[0, 4] += np.pi / 2.0

    try:
        first_output = update_rows(tracker, first, image)
        rotated_output = update_rows(tracker, rotated, image)
    finally:
        tracker.close()

    assert first_output.shape == (1, 9)
    assert rotated_output.shape[1] == 9
    assert len(rotated_output) == 0 or rotated_output[0, 5] != first_output[0, 5]


def test_native_sfsort_obb_center_penalty_uses_oriented_support():
    """A perpendicular move beyond a thin OBB's support must not match through its large AABB envelope."""
    library = sfsort_binding.SFSORTLibrary(sfsort_binding.ensure_sfsort_cpp_library())
    tracker = sfsort.NativeSFSORTTracker(
        {
            "high_th": 0.1,
            "new_track_th": 0.1,
            "low_th": 0.01,
            "match_th_first": 0.55,
            "dynamic_tuning": False,
            "frame_width": 320,
            "frame_height": 240,
        },
        geometry="obb",
        library=library,
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    angle = np.pi / 4.0
    first = np.asarray([[150.0, 100.0, 100.0, 10.0, angle, 0.999, 0]], dtype=np.float32)
    # Move 50 pixels along the rectangle's short-axis direction. The oriented
    # support is only 5 + 5 pixels, while the enclosing AABBs still overlap.
    moved = first.copy()
    moved[0, 0] -= 50.0 / np.sqrt(2.0)
    moved[0, 1] += 50.0 / np.sqrt(2.0)

    try:
        first_output = update_rows(tracker, first, image)
        moved_output = update_rows(tracker, moved, image)
    finally:
        tracker.close()

    assert first_output.shape == (1, 9)
    assert moved_output.shape == (1, 9)
    assert moved_output[0, 5] != first_output[0, 5]
