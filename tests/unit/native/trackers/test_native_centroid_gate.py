"""Exercise centroid radii and stage gates through the compiled C ABI."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.native.trackers.bytetrack import ByteTrackLibrary, ensure_bytetrack_cpp_library
from boxmot.trackers.bytetrack.native import NativeByteTrackTracker

from ._helpers import update_rows
from .test_native_association import NATIVE_AABB_CASES, RESOLVERS


@pytest.mark.parametrize("resolver", RESOLVERS)
@pytest.mark.parametrize("value", [0.0, -0.01, 1.01, np.inf, np.nan])
def test_native_centroid_radius_rejects_invalid_values(resolver, value: float) -> None:
    """Reject invalid radius configuration before constructing a C++ handle."""
    with pytest.raises(ValueError, match="centroid_max_distance"):
        resolver({"centroid_max_distance": value})


def _row(shift: float, score: float, *, obb: bool) -> np.ndarray:
    """Keep shape constant while moving the center by a known pixel distance."""
    geometry = [200 + shift, 200, 200, 200, 0.2] if obb else [100 + shift, 100, 300 + shift, 300]
    return np.array([[*geometry, score, 0]], dtype=np.float32)


@pytest.mark.parametrize(
    ("tracker_cls", "library_cls", "ensure_library", "options"),
    NATIVE_AABB_CASES,
    ids=("bytetrack", "botsort", "ocsort", "sfsort", "occluboost"),
)
@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
@pytest.mark.parametrize(
    "fraction,shift,keeps_id", [(0.05, 2.0, True), (0.05, 50.0, True), (0.05, 51.0, False), (1.0, 60.0, True)]
)
def test_native_centroid_radius_controls_actual_association(
    tracker_cls, library_cls, ensure_library, options, obb: bool, fraction: float, shift: float, keeps_id: bool
) -> None:
    """Radius is measured against the 1,000-pixel frame diagonal in both geometries."""
    tracker = tracker_cls(
        {**options, "asso_func": "centroid", "centroid_max_distance": fraction},
        geometry="obb" if obb else "aabb",
        library=library_cls(ensure_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        first = update_rows(tracker, _row(0, 0.95, obb=obb), image)
        second = update_rows(tracker, _row(shift, 0.95, obb=obb), image)
    finally:
        tracker.close()
    id_column = 5 if obb else 4
    assert len(first) == 1
    matched_original = len(second) == 1 and second[0, id_column] == first[0, id_column]
    assert matched_original == keeps_id


@pytest.mark.parametrize(
    ("tracker_cls", "library_cls", "ensure_library", "options"),
    NATIVE_AABB_CASES,
    ids=("bytetrack", "botsort", "ocsort", "sfsort", "occluboost"),
)
@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
def test_native_centroid_hard_gate_survives_permissive_thresholds_and_fusion(
    tracker_cls, library_cls, ensure_library, options, obb: bool
) -> None:
    """Score/shape/appearance terms cannot revive a pair outside the radius."""
    name = tracker_cls.__name__
    extra = {
        "NativeByteTrackTracker": {"match_thresh": 2.0, "second_match_thresh": 2.0},
        "NativeBotSortTracker": {
            "match_thresh": 2.0,
            "proximity_thresh": 2.0,
            "use_embeddings": True,
            "fuse_first_associate": True,
        },
        "NativeOcSortTracker": {"iou_threshold": -1.0, "inertia": 20.0},
        "NativeSFSORTTracker": {"match_th_first": 0.67},
        "NativeOccluBoostTracker": {
            "iou_threshold": -1.0,
            "recovery_iou_thresh": -1.0,
            "recovery_appearance_thresh": 0.0,
            "use_embeddings": True,
            "lambda_emb_multiplier": 20.0,
        },
    }[name]
    tracker = tracker_cls(
        {**options, **extra, "asso_func": "centroid"},
        geometry="obb" if obb else "aabb",
        library=library_cls(ensure_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32) if extra.get("use_embeddings") else None
    try:
        first = update_rows(tracker, _row(0, 0.95, obb=obb), image, embeddings=embeddings)
        second = update_rows(tracker, _row(60, 0.95, obb=obb), image, embeddings=embeddings)
    finally:
        tracker.close()
    id_column = 5 if obb else 4
    assert len(first) == 1
    assert not len(second) or second[0, id_column] != first[0, id_column]


@pytest.mark.parametrize(
    ("tracker_cls", "library_cls", "ensure_library", "options"),
    NATIVE_AABB_CASES,
    ids=("bytetrack", "botsort", "ocsort", "sfsort", "occluboost"),
)
@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
def test_native_centroid_ignores_strict_geometry_thresholds(
    tracker_cls, library_cls, ensure_library, options, obb: bool
) -> None:
    """The radius alone admits geometry, regardless of another mode's thresholds."""
    extra = {
        "NativeByteTrackTracker": {"match_thresh": 0.0},
        "NativeBotSortTracker": {"match_thresh": 0.0, "proximity_thresh": 0.0},
        "NativeOcSortTracker": {"iou_threshold": 1.0},
        "NativeSFSORTTracker": {"match_th_first": 0.0, "dynamic_tuning": True},
        "NativeOccluBoostTracker": {"iou_threshold": 1.0},
    }[tracker_cls.__name__]
    tracker = tracker_cls(
        {**options, **extra, "asso_func": "centroid"},
        geometry="obb" if obb else "aabb",
        library=library_cls(ensure_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        first = update_rows(tracker, _row(0, 0.95, obb=obb), image)
        second = update_rows(tracker, _row(50, 0.95, obb=obb), image)
    finally:
        tracker.close()
    id_column = 5 if obb else 4
    assert len(first) == len(second) == 1
    assert second[0, id_column] == first[0, id_column]


@pytest.mark.parametrize("tracker_index", [0, 1], ids=("bytetrack", "botsort"))
@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
@pytest.mark.parametrize("stage", ["secondary", "unconfirmed"])
def test_native_centroid_ignores_secondary_and_confirmation_cost_limits(
    tracker_index: int, obb: bool, stage: str
) -> None:
    """Every cost-based stage includes the full centroid radius."""
    tracker_cls, library_cls, ensure_library, options = NATIVE_AABB_CASES[tracker_index]
    confidence_option = "track_thresh" if tracker_index == 0 else "track_high_thresh"
    tracker = tracker_cls(
        {
            **options,
            "asso_func": "centroid",
            "second_match_thresh": 0.0,
            "unconfirmed_match_thresh": 0.0,
            confidence_option: 0.6,
        },
        geometry="obb" if obb else "aabb",
        library=library_cls(ensure_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        if stage == "unconfirmed":
            update_rows(tracker, np.empty((0, 7 if obb else 6), dtype=np.float32), image)
        update_rows(tracker, _row(0, 0.95, obb=obb), image)
        second = update_rows(tracker, _row(50, 0.4 if stage == "secondary" else 0.95, obb=obb), image)
    finally:
        tracker.close()
    assert len(second) == 1


@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
def test_native_sfsort_fused_geometry_ranking_cannot_narrow_centroid_radius(obb: bool) -> None:
    """A finite fused shape/center cost above one still represents eligible geometry."""
    tracker_cls, library_cls, ensure_library, options = NATIVE_AABB_CASES[3]
    tracker = tracker_cls(
        {**options, "asso_func": "centroid", "match_th_first": 0.0, "dynamic_tuning": True},
        geometry="obb" if obb else "aabb",
        library=library_cls(ensure_library()),
    )
    first_geometry = [20, 20, 2, 2, 0.2] if obb else [19, 19, 21, 21]
    second_geometry = [50, 50, 0.02, 0.02, 0.2] if obb else [49.99, 49.99, 50.01, 50.01]
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        first = update_rows(tracker, np.array([[*first_geometry, 0.95, 0]], dtype=np.float32), image)
        second = update_rows(tracker, np.array([[*second_geometry, 0.95, 0]], dtype=np.float32), image)
    finally:
        tracker.close()
    id_column = 5 if obb else 4
    assert len(first) == len(second) == 1
    assert second[0, id_column] == first[0, id_column]


@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
def test_native_sfsort_secondary_cost_limit_cannot_narrow_centroid_radius(obb: bool) -> None:
    """Low-confidence SFSORT matching uses the same inclusive centroid radius."""
    tracker_cls, library_cls, ensure_library, options = NATIVE_AABB_CASES[3]
    tracker = tracker_cls(
        {**options, "asso_func": "centroid", "high_th": 0.6, "match_th_second": 0.0},
        geometry="obb" if obb else "aabb",
        library=library_cls(ensure_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        first = update_rows(tracker, _row(0, 0.95, obb=obb), image)
        second = update_rows(tracker, _row(50, 0.4, obb=obb), image)
    finally:
        tracker.close()
    id_column = 5 if obb else 4
    assert len(first) == len(second) == 1
    assert second[0, id_column] == first[0, id_column]


@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
def test_native_occluboost_scaled_fused_cost_is_not_an_extra_centroid_cutoff(obb: bool) -> None:
    """Appearance weighting ranks eligible pairs without an implicit cost sentinel."""
    tracker_cls, library_cls, ensure_library, options = NATIVE_AABB_CASES[4]
    tracker = tracker_cls(
        {**options, "asso_func": "centroid", "use_embeddings": True, "lambda_iou": 1.0e6},
        geometry="obb" if obb else "aabb",
        library=library_cls(ensure_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        first = update_rows(tracker, _row(0, 0.95, obb=obb), image, embeddings=np.array([[1.0, 0.0]], dtype=np.float32))
        second = update_rows(
            tracker, _row(50, 0.95, obb=obb), image, embeddings=np.array([[-1.0, 0.0]], dtype=np.float32)
        )
    finally:
        tracker.close()
    id_column = 5 if obb else 4
    assert len(first) == len(second) == 1
    assert second[0, id_column] == first[0, id_column]


@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
@pytest.mark.parametrize("threshold,matched", [(0.1, False), (0.5, True)])
def test_native_bytetrack_secondary_threshold_is_independent(obb: bool, threshold: float, matched: bool) -> None:
    """Low-confidence detections use the configurable second-pass cost limit."""
    tracker = NativeByteTrackTracker(
        {"asso_func": "iou", "second_match_thresh": threshold},
        geometry="obb" if obb else "aabb",
        library=ByteTrackLibrary(ensure_bytetrack_cpp_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        first = update_rows(tracker, _row(0, 0.95, obb=obb), image)
        second = update_rows(tracker, _row(20, 0.4, obb=obb), image)
    finally:
        tracker.close()
    assert len(first) == 1
    assert bool(len(second)) == matched
    if matched:
        id_column = 5 if obb else 4
        assert second[0, id_column] == first[0, id_column]


@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
@pytest.mark.parametrize("threshold,matched", [(0.1, False), (0.7, True)])
def test_native_bytetrack_unconfirmed_threshold_is_independent(obb: bool, threshold: float, matched: bool) -> None:
    """New tracks after frame one use their own confirmation cost limit."""
    tracker = NativeByteTrackTracker(
        {"asso_func": "iou", "unconfirmed_match_thresh": threshold},
        geometry="obb" if obb else "aabb",
        library=ByteTrackLibrary(ensure_bytetrack_cpp_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        update_rows(tracker, np.empty((0, 7 if obb else 6), dtype=np.float32), image)
        assert len(update_rows(tracker, _row(0, 0.95, obb=obb), image)) == 0
        confirmed = update_rows(tracker, _row(20, 0.95, obb=obb), image)
    finally:
        tracker.close()
    assert bool(len(confirmed)) == matched


@pytest.mark.parametrize("tracker_index", [2, 4], ids=("ocsort", "occluboost"))
@pytest.mark.parametrize("obb", [False, True], ids=("aabb", "obb"))
@pytest.mark.parametrize("shift,matched", [(49.0, True), (50.0, True), (51.0, False)])
def test_native_centroid_second_pass_includes_radius_boundary(
    tracker_index: int, obb: bool, shift: float, matched: bool
) -> None:
    """Old stage thresholds cannot narrow the inclusive centroid radius."""
    tracker_cls, library_cls, ensure_library, options = NATIVE_AABB_CASES[tracker_index]
    extra = (
        {"use_byte": True, "det_thresh": 0.6, "iou_threshold": 0.99}
        if tracker_index == 2
        else {
            "use_second_pass": True,
            "second_pass_min_hits": 0,
            "det_thresh": 0.6,
            "second_iou_thresh": 0.99,
        }
    )
    tracker = tracker_cls(
        {**options, **extra, "asso_func": "centroid"},
        geometry="obb" if obb else "aabb",
        library=library_cls(ensure_library()),
    )
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    try:
        first = update_rows(tracker, _row(0, 0.95, obb=obb), image)
        second = update_rows(tracker, _row(shift, 0.4, obb=obb), image)
    finally:
        tracker.close()
    assert len(first) == 1
    assert bool(len(second)) == matched
    if matched:
        id_column = 5 if obb else 4
        assert second[0, id_column] == first[0, id_column]
