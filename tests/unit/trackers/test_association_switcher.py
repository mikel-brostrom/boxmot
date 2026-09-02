from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.trackers.base import BaseTracker
from boxmot.trackers.bbox.boosttrack import BoostTrack
from boxmot.trackers.bbox.botsort import BotSort
from boxmot.trackers.bbox.bytetrack import ByteTrack
from boxmot.trackers.bbox.deepocsort import DeepOcSort
from boxmot.trackers.bbox.hybridsort import HybridSort
from boxmot.trackers.bbox.occluboost import OccluBoost
from boxmot.trackers.bbox.ocsort import OcSort
from boxmot.trackers.bbox.sfsort import SFSORT
from boxmot.trackers.bbox.strongsort import StrongSort
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.hybrid.sam2mot.sam2mot import Sam2Mot
from boxmot.trackers.registry import TRACKER_DEFINITIONS

TrackerFactory = Callable[..., BaseTracker]


def _bytetrack(**kwargs) -> ByteTrack:
    return ByteTrack(
        min_hits=1,
        min_conf=0.05,
        track_thresh=0.2,
        iou_threshold=0.1,
        **kwargs,
    )


def _botsort(**kwargs) -> BotSort:
    return BotSort(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        min_hits=1,
        track_high_thresh=0.2,
        track_low_thresh=0.05,
        new_track_thresh=0.2,
        match_thresh=0.9,
        fuse_first_associate=False,
        **kwargs,
    )


def _ocsort(**kwargs) -> OcSort:
    return OcSort(
        min_hits=1,
        min_conf=0.05,
        det_thresh=0.2,
        iou_threshold=0.1,
        **kwargs,
    )


def _deepocsort(**kwargs) -> DeepOcSort:
    return DeepOcSort(
        reid_model=None,
        embedding_off=True,
        cmc_off=True,
        min_hits=1,
        det_thresh=0.2,
        iou_threshold=0.1,
        **kwargs,
    )


def _hybridsort(**kwargs) -> HybridSort:
    return HybridSort(
        reid_model=None,
        with_reid=False,
        cmc_method=None,
        min_hits=1,
        det_thresh=0.2,
        track_thresh=0.2,
        iou_threshold=0.1,
        **kwargs,
    )


def _boosttrack(**kwargs) -> BoostTrack:
    return BoostTrack(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        min_hits=1,
        det_thresh=0.2,
        iou_threshold=0.1,
        **kwargs,
    )


def _occluboost(**kwargs) -> OccluBoost:
    return OccluBoost(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        min_hits=1,
        det_thresh=0.2,
        new_track_thresh=0.2,
        instant_confirm_thresh=0.2,
        iou_threshold=0.1,
        gta_enabled=False,
        **kwargs,
    )


def _sfsort(**kwargs) -> SFSORT:
    return SFSORT(
        high_th=0.2,
        low_th=0.05,
        new_track_th=0.2,
        match_th_first=0.67,
        min_hits=1,
        **kwargs,
    )


def _strongsort(**kwargs) -> StrongSort:
    return StrongSort(reid_model=None, min_hits=1, **kwargs)


def _sam2mot(**kwargs) -> Sam2Mot:
    return Sam2Mot(det_thresh=0.2, new_track_thresh=0.2, min_hits=1, **kwargs)


TRACKER_FACTORIES: dict[str, TrackerFactory] = {
    "strongsort": _strongsort,
    "ocsort": _ocsort,
    "bytetrack": _bytetrack,
    "sfsort": _sfsort,
    "botsort": _botsort,
    "deepocsort": _deepocsort,
    "hybridsort": _hybridsort,
    "boosttrack": _boosttrack,
    "occluboost": _occluboost,
    "sam2mot": _sam2mot,
}


class _SimilaritySpy:
    def __init__(self, value: float = 0.91) -> None:
        self.value = value
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        boxes_a = np.asarray(boxes_a)
        boxes_b = np.asarray(boxes_b)
        self.calls.append((boxes_a.copy(), boxes_b.copy()))
        return np.full((len(boxes_a), len(boxes_b)), self.value, dtype=np.float32)


def _box(value: np.ndarray, **attrs) -> SimpleNamespace:
    return SimpleNamespace(xyxy=np.asarray(value, dtype=np.float32), **attrs)


def _aabb_dets() -> np.ndarray:
    return np.array([[10, 10, 30, 30, 0.95, 0]], dtype=np.float32)


def _img() -> np.ndarray:
    return np.zeros((80, 120, 3), dtype=np.uint8)


def _exercise_core_association(name: str, tracker: BaseTracker, spy: _SimilaritySpy) -> None:
    tracks = [_box(np.array([10, 10, 30, 30], dtype=np.float32), time_since_update=1)]
    detections = [_box(np.array([11, 11, 31, 31], dtype=np.float32), conf=0.95)]

    if name == "bytetrack":
        tracker.asso_func = spy
        tracker._fused_association_cost(tracks, detections)
        return
    if name == "botsort":
        tracker.asso_func = spy
        tracker._first_association_cost(tracks, detections)
        return
    if name == "strongsort":
        tracker.asso_func = spy
        tracker._association_cost(tracks, detections)
        return
    if name == "sam2mot":
        tracker.asso_func = spy
        track = SimpleNamespace(
            bbox=np.array([10, 10, 30, 30], dtype=np.float32),
            velocity=np.zeros(4, dtype=np.float32),
            last_matched_bbox=np.array([10, 10, 30, 30], dtype=np.float32),
            mask=None,
        )
        tracker._association_similarity(
            _aabb_dets()[:, :4],
            [track],
            [0],
            [0],
            det_masks=None,
            det_obbs=None,
        )
        return

    # Prime the track lifecycle before replacing the configured metric with a
    # spy. The second frame must invoke the same callable slot in core matching.
    tracker.update(_aabb_dets())
    tracker.asso_func = spy
    tracker.update(_aabb_dets())


@pytest.mark.parametrize("name", tuple(TRACKER_FACTORIES))
def test_every_registered_python_tracker_uses_selected_geometry_in_core_matching(name: str) -> None:
    assert set(TRACKER_FACTORIES) == set(TRACKER_DEFINITIONS)
    tracker = TRACKER_FACTORIES[name](asso_func="giou")
    expected = AssociationFunction.giou_batch(_aabb_dets()[:, :4], _aabb_dets()[:, :4])
    np.testing.assert_allclose(tracker.association_similarity(_aabb_dets(), _aabb_dets()), expected)

    spy = _SimilaritySpy()
    _exercise_core_association(name, tracker, spy)

    assert spy.calls, f"{name} bypassed the configured association function"
    assert any(len(left) and len(right) for left, right in spy.calls), name
    # OC-SORT-family matchers intentionally append confidence/motion score
    # columns; every shared metric consumes the leading four geometry columns.
    assert all(left.shape[1] >= 4 and right.shape[1] >= 4 for left, right in spy.calls), name


def test_association_distance_uses_selected_similarity_for_arrays_and_objects() -> None:
    tracker = ByteTrack(asso_func="giou")
    boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    boxes_b = np.array([[2, 2, 12, 12], [20, 20, 30, 30]], dtype=np.float32)
    expected = 1.0 - AssociationFunction.giou_batch(boxes_a, boxes_b)

    np.testing.assert_allclose(tracker.association_distance(boxes_a, boxes_b), expected)
    np.testing.assert_allclose(
        tracker.association_distance([_box(boxes_a[0])], [_box(box) for box in boxes_b]),
        expected,
    )
    assert tracker.association_distance([], [_box(boxes_b[0])]).shape == (0, 1)
    assert tracker.association_distance([_box(boxes_a[0])], []).shape == (1, 0)
    assert tracker.association_distance(np.empty((0, 0)), boxes_b).shape == (0, 2)


def test_botsort_preserves_geometric_distance_before_reid_fusion() -> None:
    tracker = BotSort(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        asso_func="diou",
    )
    tracks = [_box(np.array([0, 0, 10, 10], dtype=np.float32))]
    detections = [_box(np.array([2, 2, 12, 12], dtype=np.float32), conf=0.9)]

    expected = tracker.association_distance(tracks, detections)
    np.testing.assert_allclose(tracker._first_association_cost(tracks, detections), expected)


def test_strongsort_fallback_uses_selected_geometry_and_keeps_stale_gate() -> None:
    tracker = StrongSort(reid_model=None, asso_func="hmiou")
    tracks = [
        _box(np.array([0, 0, 10, 10], dtype=np.float32), time_since_update=1),
        _box(np.array([1, 1, 11, 11], dtype=np.float32), time_since_update=2),
    ]
    detections = [_box(np.array([2, 2, 12, 12], dtype=np.float32))]

    cost = tracker._association_cost(tracks, detections)

    np.testing.assert_allclose(cost[0], tracker.association_distance(tracks[:1], detections)[0])
    assert cost[1, 0] > tracker.max_iou_dist


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou", "centroid"))
def test_sam2mot_canonicalizes_inverted_extrapolated_boxes(mode: str) -> None:
    tracker = _sam2mot(asso_func=mode)
    if mode == "centroid":
        tracker._initialize_frame_context(_img())
    track = SimpleNamespace(
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        velocity=np.array([0, 0, -20, 0], dtype=np.float32),
        last_matched_bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        mask=None,
    )

    similarity = tracker._association_similarity(
        np.array([[0, 0, 10, 10]], dtype=np.float32),
        [track],
        [0],
        [0],
        det_masks=None,
        det_obbs=None,
    )

    assert similarity.shape == (1, 1)
    assert np.isfinite(similarity).all()


@pytest.mark.parametrize(
    "name",
    (
        "bytetrack",
        "botsort",
        "ocsort",
        "deepocsort",
        "hybridsort",
        "boosttrack",
        "occluboost",
        "sfsort",
    ),
)
def test_centroid_only_needs_the_initial_image_for_image_optional_trackers(name: str) -> None:
    tracker = TRACKER_FACTORIES[name](asso_func="centroid")
    dets = _aabb_dets()

    assert tracker.requires_image(dets)
    with pytest.raises(ValueError, match="requires img when using 'centroid' association"):
        tracker.update(dets)

    tracker.update(dets, img=_img())
    assert not tracker.requires_image(dets)
    tracker.update(dets)


@pytest.mark.parametrize("name", tuple(TRACKER_FACTORIES))
@pytest.mark.parametrize("mode", ("iou", "diou", "centroid"))
def test_every_registered_tracker_accepts_supported_obb_association_modes(name: str, mode: str) -> None:
    tracker = TRACKER_FACTORIES[name](asso_func=mode, is_obb=True)

    assert tracker.is_obb
    assert tracker.asso_func_name == f"{mode}_obb"


@pytest.mark.parametrize("name", tuple(TRACKER_FACTORIES))
@pytest.mark.parametrize("mode", ("giou", "ciou", "hmiou"))
def test_every_registered_tracker_rejects_unimplemented_obb_association_modes(name: str, mode: str) -> None:
    with pytest.raises(ValueError, match="has no oriented-box implementation"):
        TRACKER_FACTORIES[name](asso_func=mode, is_obb=True)
