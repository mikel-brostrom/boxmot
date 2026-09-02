from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from boxmot.detectors.base import Detections
from boxmot.trackers.bbox.boosttrack import BoostTrack
from boxmot.trackers.bbox.botsort import BotSort
from boxmot.trackers.bbox.bytetrack import ByteTrack
from boxmot.trackers.bbox.deepocsort import DeepOcSort
from boxmot.trackers.bbox.hybridsort import HybridSort
from boxmot.trackers.bbox.occluboost import OccluBoost
from boxmot.trackers.bbox.ocsort import OcSort
from boxmot.trackers.bbox.sfsort import SFSORT
from boxmot.trackers.bbox.sfsort import TrackState as SFSortTrackState
from boxmot.trackers.bbox.strongsort import StrongSort
from boxmot.trackers.common.detections import DetectionBatch
from boxmot.trackers.common.detections.layout import AABB_DETECTIONS, OBB_DETECTIONS
from boxmot.trackers.common.track_models import strongsort as strongsort_track_model
from boxmot.trackers.common.track_models.base import BoxTrack, SortBoxTrack
from boxmot.trackers.common.track_models.boosttrack import KalmanBoxTracker as BoostTrackBoxTrack
from boxmot.trackers.common.track_models.botsort import BaseTrack as BotSortBaseTrack
from boxmot.trackers.common.track_models.botsort import STrack as BotSortTrack
from boxmot.trackers.common.track_models.botsort import TrackState as BotSortTrackState
from boxmot.trackers.common.track_models.bytetrack import BaseTrack as ByteTrackBaseTrack
from boxmot.trackers.common.track_models.bytetrack import STrack as ByteTrackTrack
from boxmot.trackers.common.track_models.bytetrack import TrackState as ByteTrackState
from boxmot.trackers.common.track_models.deepocsort import KalmanBoxTracker as DeepOCSortBoxTrack
from boxmot.trackers.common.track_models.hybridsort import KalmanBoxTracker as HybridSortBoxTrack
from boxmot.trackers.common.track_models.ocsort import KalmanBoxTracker as OCSortBoxTrack
from boxmot.trackers.common.tracking.lifecycle import joint_stracks, remove_duplicate_stracks, sub_stracks
from boxmot.trackers.common.tracking.protocol import TrackerProtocol
from boxmot.trackers.common.tracking.track import (
    TrackIdAllocator,
    TrackLifecycleMixin,
    TrackMeta,
    TrackState,
    sync_track_meta,
)
from boxmot.trackers.hybrid.sam2mot.sam2mot import Sam2Mot
from boxmot.trackers.registry import create_tracker, get_tracker_config


class DummyCMC:
    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
        return np.eye(2, 3, dtype=np.float32)


class DummyReID:
    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        return np.ones((len(boxes), 4), dtype=np.float32)


def _hybridsort(**kwargs):
    tracker = HybridSort(with_reid=False, **kwargs)
    tracker.cmc = DummyCMC()
    return tracker


def _strongsort(**kwargs):
    tracker = StrongSort(reid_model=DummyReID(), n_init=1, **kwargs)
    tracker.cmc = DummyCMC()
    return tracker


AABB_TRACKERS: tuple[tuple[str, Callable[..., object]], ...] = (
    (
        "botsort",
        lambda **kwargs: BotSort(reid_model=None, with_reid=False, use_cmc=False, **kwargs),
    ),
    ("bytetrack", lambda **kwargs: ByteTrack(**kwargs)),
    ("ocsort", lambda **kwargs: OcSort(**kwargs)),
    ("sfsort", lambda **kwargs: SFSORT(**kwargs)),
    (
        "boosttrack",
        lambda **kwargs: BoostTrack(
            reid_model=None,
            with_reid=False,
            use_cmc=False,
            use_dlo_boost=False,
            use_duo_boost=False,
            **kwargs,
        ),
    ),
    (
        "occluboost",
        lambda **kwargs: OccluBoost(
            reid_model=None,
            use_cmc=False,
            use_dlo_boost=False,
            use_duo_boost=False,
            instant_confirm_thresh=0.0,
            **kwargs,
        ),
    ),
    (
        "deepocsort",
        lambda **kwargs: DeepOcSort(reid_model=None, embedding_off=True, cmc_off=True, **kwargs),
    ),
    ("hybridsort", _hybridsort),
    ("strongsort", _strongsort),
)

INSTANCE_LOCAL_ID_TRACKERS = AABB_TRACKERS

OBB_TRACKERS: tuple[tuple[str, Callable[..., object]], ...] = (
    (
        "botsort",
        lambda **kwargs: BotSort(reid_model=None, with_reid=False, use_cmc=False, **kwargs),
    ),
    ("bytetrack", lambda **kwargs: ByteTrack(**kwargs)),
    ("ocsort", lambda **kwargs: OcSort(**kwargs)),
    ("sfsort", lambda **kwargs: SFSORT(**kwargs)),
    (
        "boosttrack",
        lambda **kwargs: BoostTrack(
            reid_model=None,
            with_reid=False,
            use_cmc=False,
            use_dlo_boost=False,
            use_duo_boost=False,
            aspect_ratio_thresh=10.0,
            **kwargs,
        ),
    ),
    (
        "occluboost",
        lambda **kwargs: OccluBoost(
            reid_model=None,
            use_cmc=False,
            use_dlo_boost=False,
            use_duo_boost=False,
            instant_confirm_thresh=0.0,
            aspect_ratio_thresh=10.0,
            **kwargs,
        ),
    ),
    (
        "deepocsort",
        lambda **kwargs: DeepOcSort(reid_model=None, embedding_off=True, cmc_off=True, **kwargs),
    ),
    ("hybridsort", _hybridsort),
    ("strongsort", _strongsort),
)


def _img() -> np.ndarray:
    return np.zeros((96, 128, 3), dtype=np.uint8)


def _aabb_dets() -> np.ndarray:
    return np.array(
        [
            [10, 10, 30, 60, 0.95, 3],
            [70, 15, 90, 65, 0.90, 5],
        ],
        dtype=np.float32,
    )


def _obb_dets(angle: float = 0.2) -> np.ndarray:
    return np.array(
        [
            [20, 35, 20, 40, angle, 0.95, 3],
            [80, 40, 20, 40, angle, 0.90, 5],
        ],
        dtype=np.float32,
    )


def _embs(n: int) -> np.ndarray:
    return np.arange(n * 4, dtype=np.float32).reshape(n, 4) + 1.0


@dataclass
class DummyTrack:
    id: int
    start_frame: int = 0
    frame_id: int = 1


def _run_until_output(
    tracker,
    dets: np.ndarray,
    embs: np.ndarray | None = None,
    n: int = 4,
) -> np.ndarray:
    out = tracker.empty_output(dtype=np.float32)
    for _ in range(n):
        frame_embs = None if embs is None else embs.copy()
        out = tracker.update(dets.copy(), _img(), frame_embs)
        if out.shape[0] > 0:
            return np.asarray(out)
    return np.asarray(out)


@pytest.mark.parametrize(
    ("name", "factory"),
    (
        (
            "ocsort",
            lambda: OcSort(min_hits=1, det_thresh=0.1, min_conf=0.05, iou_threshold=0.1),
        ),
        (
            "deepocsort",
            lambda: DeepOcSort(
                reid_model=None,
                embedding_off=True,
                cmc_off=True,
                min_hits=1,
                det_thresh=0.1,
                iou_threshold=0.1,
            ),
        ),
        (
            "hybridsort",
            lambda: _hybridsort(min_hits=1, det_thresh=0.1, iou_threshold=0.1),
        ),
    ),
)
def test_ocsort_family_obb_outputs_latest_matched_observation(name, factory):
    del name
    tracker = factory()
    first = np.array([[64, 48, 42, 16, -0.35, 0.95, 0]], dtype=np.float32)
    second = np.array([[68, 50, 46, 14, -0.2, 0.95, 0]], dtype=np.float32)

    tracker.update(first, _img())
    output = np.asarray(tracker.update(second, _img()))

    assert output.shape == (1, 9)
    np.testing.assert_allclose(output[0, :5], second[0, :5], atol=1e-5)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: DeepOcSort(
            reid_model=None,
            embedding_off=True,
            cmc_off=False,
            min_hits=1,
            det_thresh=0.1,
            iou_threshold=0.1,
        ),
        lambda: HybridSort(
            reid_model=None,
            with_reid=False,
            min_hits=1,
            det_thresh=0.1,
            iou_threshold=0.1,
            use_byte=False,
        ),
    ),
)
def test_obb_identity_cmc_preserves_ocsort_family_state(factory):
    tracker = factory()
    tracker.cmc = DummyCMC()
    track_ids = []

    for frame in range(12):
        detection = np.array(
            [[64 + frame, 48, 42, 16, -0.35 + (0.01 * frame), 0.95, 0]],
            dtype=np.float32,
        )
        output = np.asarray(tracker.update(detection, _img()))
        assert output.shape == (1, 9)
        track_ids.append(int(output[0, 5]))

    assert len(set(track_ids)) == 1
    state = tracker.active_tracks[0].get_state()[0]
    assert np.isfinite(state).all()
    np.testing.assert_allclose(state[2:4], [42, 16], rtol=0.1)


def test_obb_observation_validity_uses_confidence_not_geometry_sum():
    track = OCSortBoxTrack(
        np.array([0.1, 0.1, 0.2, 0.1, -3.0, 0.95], dtype=np.float32),
        cls=0,
        det_ind=0,
        is_obb=True,
        id_allocator=TrackIdAllocator(),
    )
    first = np.array([0.1, 0.1, 0.2, 0.1, -3.0, 0.95], dtype=np.float32)
    second = np.array([0.2, 0.1, 0.2, 0.1, -3.0, 0.95], dtype=np.float32)

    track.update(first, cls=0, det_ind=0)
    track.update(second, cls=0, det_ind=0)

    assert track.velocity is not None
    assert track.velocity[1] > 0


def test_hybridsort_obb_respects_disabled_byte_association():
    tracker = _hybridsort(
        min_hits=1,
        det_thresh=0.6,
        low_thresh=0.1,
        iou_threshold=0.1,
        use_byte=False,
    )
    high = np.array([[64, 48, 42, 16, -0.35, 0.95, 0]], dtype=np.float32)
    low = np.array([[65, 48, 42, 16, -0.35, 0.4, 0]], dtype=np.float32)

    first = np.asarray(tracker.update(high, _img()))
    second = np.asarray(tracker.update(low, _img()))

    assert first.shape == (1, 9)
    assert second.shape == (0, 9)
    assert tracker.active_tracks[0].time_since_update == 1


def test_detection_layout_roundtrip_aabb():
    dets = _aabb_dets()
    embs = _embs(2)

    batch = DetectionBatch.from_layout(dets, AABB_DETECTIONS, embs=embs)
    boosted = batch.with_confs(np.array([0.7, 0.8], dtype=np.float32))
    replaced = batch.with_embs(embs + 10)
    high, second = batch.split_by_confidence(high_thresh=0.92, low_thresh=0.5)

    np.testing.assert_allclose(batch.boxes, dets[:, :4])
    np.testing.assert_allclose(batch.confs, dets[:, 4])
    np.testing.assert_allclose(batch.clss, dets[:, 5])
    np.testing.assert_array_equal(batch.det_inds, np.array([0, 1], dtype=np.int32))
    np.testing.assert_allclose(batch.embs, embs)
    np.testing.assert_allclose(boosted.confs, np.array([0.7, 0.8], dtype=np.float32))
    np.testing.assert_allclose(batch.confs, dets[:, 4])
    np.testing.assert_allclose(boosted.as_indexed_detections()[:, 4], boosted.confs)
    np.testing.assert_allclose(replaced.embs, embs + 10)
    np.testing.assert_allclose(batch.embs, embs)
    np.testing.assert_array_equal(high.det_inds, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(second.det_inds, np.array([1], dtype=np.int32))
    np.testing.assert_allclose(batch.as_box_conf_detections(), dets[:, :5])
    np.testing.assert_allclose(
        batch.as_indexed_detections(),
        np.column_stack((dets, np.array([0, 1], dtype=np.float32))),
    )


def test_detection_layout_roundtrip_obb():
    dets = _obb_dets()

    batch = DetectionBatch.from_layout(dets, OBB_DETECTIONS)
    selected = batch.select(np.array([False, True]))

    np.testing.assert_allclose(batch.boxes, dets[:, :5])
    np.testing.assert_allclose(batch.confs, dets[:, 5])
    np.testing.assert_allclose(batch.clss, dets[:, 6])
    np.testing.assert_array_equal(selected.det_inds, np.array([1], dtype=np.int32))
    np.testing.assert_allclose(batch.as_box_conf_detections(), dets[:, :6])
    assert batch.as_indexed_detections().shape == (2, 8)


def test_detection_batch_rejects_fractional_class_and_detection_indices():
    fractional_class = _aabb_dets()
    fractional_class[0, 5] = 0.5
    with pytest.raises(ValueError, match="class IDs must be integers"):
        DetectionBatch.from_layout(fractional_class, AABB_DETECTIONS)

    indexed = np.column_stack((_aabb_dets(), np.array([0.5, 1.0], dtype=np.float32)))
    with pytest.raises(ValueError, match="Detection indices must be integers"):
        DetectionBatch.from_layout(indexed, AABB_DETECTIONS)


@pytest.mark.parametrize(
    "detection, message",
    [
        (np.array([[20, 20, np.nan, 10, 0.1, 0.9, 0]], dtype=np.float32), "finite"),
        (np.array([[20, 20, -5, 10, 0.1, 0.9, 0]], dtype=np.float32), "positive width"),
        (np.array([[20, 20, 5, 10, 0.1, 0.9, 1.5]], dtype=np.float32), "integers"),
    ],
)
def test_direct_tracker_rejects_invalid_obb_rows(detection: np.ndarray, message: str):
    tracker = ByteTrack(min_hits=1)
    with pytest.raises(ValueError, match=message):
        tracker.update(detection, _img())


def test_track_id_allocator_is_instance_local():
    first = TrackIdAllocator()
    second = TrackIdAllocator()

    assert first.alloc() == 0
    assert first.alloc() == 1
    assert second.alloc() == 0
    first.reset()
    assert first.alloc() == 0


def test_sync_track_meta_creates_and_refreshes_metadata():
    track = DummyTrack(id=7, start_frame=3, frame_id=9)
    track.age = 4
    track.hits = 2
    track.hit_streak = 2
    track.time_since_update = 1
    track.conf = 0.75
    track.cls = 5
    track.det_ind = 11

    meta = sync_track_meta(track, TrackState.TRACKED)

    assert isinstance(meta, TrackMeta)
    assert track.meta is meta
    assert meta.id == 7
    assert meta.state is TrackState.TRACKED
    assert meta.age == 4
    assert meta.hits == 2
    assert meta.hit_streak == 2
    assert meta.time_since_update == 1
    assert meta.start_frame == 3
    assert meta.frame_id == 9
    assert meta.conf == 0.75
    assert meta.cls == 5
    assert meta.det_ind == 11


def test_bytetrack_and_botsort_base_tracks_share_lifecycle_mixin():
    bytetrack = ByteTrackBaseTrack()
    bytetrack.track_id = 11

    assert isinstance(bytetrack, TrackLifecycleMixin)
    assert bytetrack.end_frame == 0

    bytetrack.mark_lost()
    assert bytetrack.state == ByteTrackState.Lost
    assert bytetrack.meta.id == 11
    assert bytetrack.meta.state is TrackState.LOST

    bytetrack.mark_removed()
    assert bytetrack.state == ByteTrackState.Removed
    assert bytetrack.meta.state is TrackState.REMOVED

    botsort = BotSortBaseTrack()
    botsort.track_id = 12

    assert isinstance(botsort, TrackLifecycleMixin)
    botsort.mark_long_lost()
    assert botsort.state == BotSortTrackState.LongLost
    assert botsort.meta.id == 12
    assert botsort.meta.state is TrackState.LOST

    botsort.mark_removed()
    assert botsort.state == BotSortTrackState.Removed
    assert botsort.meta.state is TrackState.REMOVED


def test_bytetrack_and_botsort_stracks_share_box_track_base():
    det = np.array([10, 10, 30, 60, 0.95, 3, 0], dtype=np.float32)

    assert isinstance(
        ByteTrackTrack(det, max_obs=3, id_allocator=TrackIdAllocator()),
        BoxTrack,
    )
    assert isinstance(
        BotSortTrack(det, max_obs=3, id_allocator=TrackIdAllocator()),
        BoxTrack,
    )


def test_kalman_box_trackers_share_sort_box_track_base():
    assert issubclass(BoostTrackBoxTrack, SortBoxTrack)
    assert issubclass(DeepOCSortBoxTrack, SortBoxTrack)
    assert issubclass(HybridSortBoxTrack, SortBoxTrack)
    assert issubclass(OCSortBoxTrack, SortBoxTrack)


@pytest.mark.parametrize(
    ("box", "expected_region"),
    (
        (np.array([[40, 40, 60, 60, 0.95, 5]], dtype=np.float32), "central"),
        (np.array([[0, 0, 10, 10, 0.95, 5]], dtype=np.float32), "marginal"),
    ),
)
def test_sfsort_lost_region_is_track_metadata(
    box: np.ndarray,
    expected_region: str,
):
    tracker = SFSORT(
        high_th=0.5,
        low_th=0.1,
        new_track_th=0.5,
        match_th_first=0.5,
        central_timeout=10,
        marginal_timeout=10,
        horizontal_margin=20,
        vertical_margin=20,
    )

    tracker.update(box, _img())
    tracker.update(np.empty((0, 6), dtype=np.float32), _img())

    assert len(tracker.lost_tracks) == 1
    lost_track = tracker.lost_tracks[0]
    assert lost_track.state == SFSortTrackState.Lost
    assert lost_track.lost_region == expected_region
    assert lost_track.meta.state is TrackState.LOST
    assert lost_track.meta.lost_region == expected_region


def test_lifecycle_joint_and_subtract_by_track_id():
    first = DummyTrack(id=1)
    duplicate = DummyTrack(id=1)
    second = DummyTrack(id=2)

    assert joint_stracks([first], [duplicate, second]) == [first, second]
    assert sub_stracks([first, second], [duplicate]) == [second]


def test_duplicate_removal_keeps_older_track():
    older = DummyTrack(id=1, start_frame=0, frame_id=10)
    younger = DummyTrack(id=2, start_frame=8, frame_id=10)

    remaining_a, remaining_b = remove_duplicate_stracks(
        [older],
        [younger],
        distance=lambda _a, _b: np.array([[0.01]], dtype=np.float32),
    )

    assert remaining_a == [older]
    assert remaining_b == []


@pytest.mark.parametrize(("name", "factory"), AABB_TRACKERS)
def test_tracker_protocol_surface(name: str, factory: Callable[..., object]):
    tracker = factory(min_hits=1)

    assert isinstance(tracker, TrackerProtocol), name
    assert isinstance(tracker.name, str)
    assert tracker.name
    assert isinstance(tracker.supports_obb, bool)
    assert isinstance(tracker.uses_img, bool)
    assert isinstance(tracker.uses_embs, bool)
    assert isinstance(tracker.supports_masks, bool)


@pytest.mark.parametrize(
    ("name", "factory"),
    (
        ("bytetrack", ByteTrack),
        ("ocsort", OcSort),
        ("sfsort", SFSORT),
        ("botsort", lambda: BotSort(reid_model=None, with_reid=False, use_cmc=False)),
        (
            "boosttrack",
            lambda: BoostTrack(reid_model=None, with_reid=False, use_cmc=False),
        ),
        (
            "occluboost",
            lambda: OccluBoost(reid_model=None, with_reid=False, use_cmc=False),
        ),
        (
            "deepocsort",
            lambda: DeepOcSort(reid_model=None, embedding_off=True, cmc_off=True),
        ),
        (
            "hybridsort",
            lambda: HybridSort(reid_model=None, with_reid=False, cmc_method=None),
        ),
    ),
)
def test_trackers_without_active_pixel_paths_accept_detections_only(name, factory) -> None:
    tracker = factory()

    assert tracker.uses_img is False, name
    assert tracker.requires_image(_aabb_dets()) is False, name
    output = tracker.update(_aabb_dets())

    assert output.ndim == 2, name


@pytest.mark.parametrize(
    ("name", "factory"),
    (
        ("bytetrack", lambda: ByteTrack(asso_func="centroid")),
        (
            "botsort",
            lambda: BotSort(reid_model=None, with_reid=False, use_cmc=False, asso_func="centroid"),
        ),
        (
            "boosttrack",
            lambda: BoostTrack(
                reid_model=None,
                with_reid=False,
                use_cmc=False,
                use_dlo_boost=False,
                use_duo_boost=False,
                asso_func="centroid",
            ),
        ),
        (
            "occluboost",
            lambda: OccluBoost(
                reid_model=None,
                with_reid=False,
                use_cmc=False,
                use_dlo_boost=False,
                use_duo_boost=False,
                asso_func="centroid",
            ),
        ),
        ("sfsort", lambda: SFSORT(asso_func="centroid")),
    ),
)
def test_unused_association_setting_does_not_require_image(name, factory) -> None:
    tracker = factory()

    assert tracker.uses_img is False, name
    assert tracker.requires_image(_aabb_dets()) is False, name
    assert tracker.update(_aabb_dets()).ndim == 2, name
    assert tracker.uses_img is False, name


@pytest.mark.parametrize(
    ("name", "factory"),
    (
        ("ocsort", lambda: OcSort(asso_func="centroid")),
        (
            "deepocsort",
            lambda: DeepOcSort(
                reid_model=None,
                embedding_off=True,
                cmc_off=True,
                asso_func="centroid",
            ),
        ),
        (
            "hybridsort",
            lambda: HybridSort(
                reid_model=None,
                with_reid=False,
                cmc_method=None,
                asso_func="centroid",
            ),
        ),
    ),
)
def test_centroid_association_only_requires_initial_image(name, factory) -> None:
    tracker = factory()
    dets = _aabb_dets()

    assert tracker.uses_img is True, name
    assert tracker.requires_image(dets) is True, name
    with pytest.raises(ValueError, match="requires img when using 'centroid' association"):
        tracker.update(dets)

    assert tracker.update(dets, img=_img()).ndim == 2, name
    assert tracker.requires_image(dets) is False, name
    assert tracker.update(dets).ndim == 2, name


@pytest.mark.parametrize(
    ("name", "factory"),
    (
        ("strongsort", lambda: StrongSort(reid_model=None, asso_func="centroid")),
        ("sam2mot", lambda: Sam2Mot(asso_func="centroid")),
    ),
)
def test_independent_image_requirements_do_not_report_centroid(name, factory) -> None:
    tracker = factory()

    with pytest.raises(ValueError, match="requires img for the current tracker configuration"):
        tracker.update(_aabb_dets())


@pytest.mark.parametrize(
    ("name", "factory"),
    (
        (
            "botsort",
            lambda: BotSort(reid_model=DummyReID(), with_reid=True, use_cmc=False),
        ),
        (
            "boosttrack",
            lambda: BoostTrack(reid_model=DummyReID(), with_reid=True, use_cmc=False),
        ),
        (
            "occluboost",
            lambda: OccluBoost(reid_model=DummyReID(), with_reid=True, use_cmc=False),
        ),
        (
            "deepocsort",
            lambda: DeepOcSort(reid_model=DummyReID(), embedding_off=False, cmc_off=True),
        ),
        (
            "hybridsort",
            lambda: HybridSort(reid_model=DummyReID(), with_reid=True, cmc_method=None),
        ),
    ),
)
def test_precomputed_embeddings_remove_live_reid_image_requirement(name, factory) -> None:
    tracker = factory()
    dets = _aabb_dets()
    embs = _embs(len(dets))

    assert tracker.uses_embs is True, name
    assert tracker.requires_image(dets, embs=embs) is False, name
    output = tracker.update(dets, embs=embs)

    assert output.ndim == 2, name


def test_cmc_enabled_tracker_requires_image() -> None:
    tracker = BoostTrack(reid_model=None, with_reid=False, use_cmc=True)

    with pytest.raises(ValueError, match="requires img"):
        tracker.update(_aabb_dets())


def test_live_reid_requires_image_when_embeddings_are_missing() -> None:
    tracker = BotSort(reid_model=DummyReID(), with_reid=True, use_cmc=False)

    with pytest.raises(ValueError, match="requires img"):
        tracker.update(_aabb_dets())


def test_sam2mot_requires_image_for_mask_coordinate_scaling() -> None:
    tracker = Sam2Mot()

    assert tracker.uses_img is True
    assert tracker.uses_embs is False
    assert tracker.supports_masks is True
    with pytest.raises(ValueError, match="requires img"):
        tracker.update(_aabb_dets())


def test_sfsort_only_requires_image_to_initialize_distinct_region_timeouts() -> None:
    dets = _aabb_dets()
    unresolved = SFSORT(central_timeout=10, marginal_timeout=1)

    assert unresolved.uses_img is True
    with pytest.raises(ValueError, match="requires img"):
        unresolved.update(dets)

    configured = SFSORT(
        central_timeout=10,
        marginal_timeout=1,
        frame_width=128,
        frame_height=96,
    )
    assert configured.uses_img is False
    assert configured.update(dets).ndim == 2


def test_optional_inputs_are_validated_centrally() -> None:
    tracker = ByteTrack()
    dets = _aabb_dets()

    with pytest.raises(TypeError, match="Unsupported image type"):
        tracker.update(dets, img="not-an-array")
    with pytest.raises(ValueError, match="same number of rows"):
        tracker.update(dets, embs=np.ones((1, 4), dtype=np.float32))


@pytest.mark.parametrize(("name", "factory"), AABB_TRACKERS)
def test_empty_input_shape_aabb(name: str, factory: Callable[..., object]):
    tracker = factory(min_hits=1)
    out = tracker.update(np.empty((0, 6), dtype=np.float32), _img(), _embs(0))

    assert out.shape == (0, 8), name


@pytest.mark.parametrize(("name", "factory"), AABB_TRACKERS)
def test_detector_result_input_shape_aabb(name: str, factory: Callable[..., object]):
    tracker = factory(min_hits=1)
    result = Detections(dets=np.empty((0, 6), dtype=np.float32), orig_img=_img())

    out = tracker.update(result, None, _embs(0))

    assert out.shape == (0, 8), name


@pytest.mark.parametrize(("name", "factory"), AABB_TRACKERS)
def test_output_contract_aabb(name: str, factory: Callable[..., object]):
    tracker = factory(min_hits=1)
    out = _run_until_output(tracker, _aabb_dets(), _embs(2))

    assert out.shape == (2, 8), name
    assert set(out[:, 4].astype(int)) == {0, 1}
    assert set(out[:, 6].astype(int)) == {3, 5}
    assert set(out[:, 7].astype(int)) == {0, 1}
    assert np.isfinite(out).all()


@pytest.mark.parametrize(("name", "factory"), AABB_TRACKERS)
def test_reset_clears_tracks_and_ids(name: str, factory: Callable[..., object]):
    tracker = factory(min_hits=1)
    first = _run_until_output(tracker, _aabb_dets(), _embs(2))
    assert first.shape == (2, 8), name

    tracker.reset()

    assert tracker.frame_count == 0
    assert tracker.get_active_tracks_for_display() == []
    second = _run_until_output(tracker, _aabb_dets(), _embs(2))
    assert set(second[:, 4].astype(int)) == {0, 1}, name


@pytest.mark.parametrize(("name", "factory"), INSTANCE_LOCAL_ID_TRACKERS)
def test_tracker_ids_are_instance_local_when_instances_are_interleaved(
    name: str,
    factory: Callable[..., object],
):
    first = factory(min_hits=1)
    second = factory(min_hits=1)

    first_out = _run_until_output(first, _aabb_dets(), _embs(2))
    second_out = _run_until_output(second, _aabb_dets(), _embs(2))

    assert set(first_out[:, 4].astype(int)) == {0, 1}, name
    assert set(second_out[:, 4].astype(int)) == {0, 1}, name


@pytest.mark.parametrize(("name", "factory"), OBB_TRACKERS)
def test_empty_input_shape_obb(name: str, factory: Callable[..., object]):
    tracker = factory(min_hits=1)
    out = tracker.update(np.empty((0, 7), dtype=np.float32), _img(), _embs(0))

    assert out.shape == (0, 9), name


@pytest.mark.parametrize(("name", "factory"), OBB_TRACKERS)
def test_output_contract_obb(name: str, factory: Callable[..., object]):
    tracker = factory(min_hits=1)
    out = _run_until_output(tracker, _obb_dets(), _embs(2))

    assert out.shape == (2, 9), name
    assert set(out[:, 5].astype(int)) == {0, 1}
    assert set(out[:, 7].astype(int)) == {3, 5}
    assert set(out[:, 8].astype(int)) == {0, 1}
    assert np.isfinite(out).all()


def test_per_class_obb_outputs_preserve_frame_global_detection_indices():
    tracker = ByteTrack(per_class=True, min_hits=1, track_thresh=0.5, min_conf=0.1)
    dets = np.array(
        [
            [20, 35, 20, 40, 0.2, 0.95, 0],
            [60, 35, 20, 40, 0.2, 0.95, 1],
            [100, 35, 20, 40, 0.2, 0.95, 0],
        ],
        dtype=np.float32,
    )

    out = _run_until_output(tracker, dets, _embs(3))

    assert {(int(row[7]), int(row[8])) for row in out} == {(0, 0), (0, 2), (1, 1)}
    assert {(int(track.cls), int(track.det_ind)) for track in tracker.active_tracks} == {
        (0, 0),
        (0, 2),
        (1, 1),
    }


@pytest.mark.parametrize(
    ("detections", "class_col", "det_ind_col"),
    (
        (
            np.array(
                [
                    [10, 10, 30, 60, 0.95, 0],
                    [50, 10, 70, 60, 0.95, 1],
                    [90, 10, 110, 60, 0.95, 0],
                ],
                dtype=np.float32,
            ),
            6,
            7,
        ),
        (
            np.array(
                [
                    [20, 35, 20, 40, 0.2, 0.95, 0],
                    [60, 35, 20, 40, 0.2, 0.95, 1],
                    [100, 35, 20, 40, 0.2, 0.95, 0],
                ],
                dtype=np.float32,
            ),
            7,
            8,
        ),
    ),
)
def test_hybridsort_per_class_preserves_frame_global_detection_indices(
    detections: np.ndarray,
    class_col: int,
    det_ind_col: int,
):
    tracker = _hybridsort(
        per_class=True,
        min_hits=1,
        det_thresh=0.1,
        iou_threshold=0.1,
    )

    output = _run_until_output(tracker, detections, _embs(3))

    assert {(int(row[class_col]), int(row[det_ind_col])) for row in output} == {
        (0, 0),
        (1, 1),
        (0, 2),
    }


@pytest.mark.parametrize(
    "factory",
    (
        lambda: OcSort(
            min_hits=1,
            det_thresh=0.1,
            min_conf=0.05,
            iou_threshold=0.1,
            asso_func="diou",
        ),
        lambda: DeepOcSort(
            reid_model=None,
            embedding_off=True,
            cmc_off=True,
            min_hits=1,
            det_thresh=0.1,
            iou_threshold=0.1,
            asso_func="diou",
        ),
    ),
)
def test_ocsort_family_obb_diou_accepts_rows_with_confidence(factory):
    tracker = factory()
    first = np.array([[64, 48, 42, 16, -0.35, 0.95, 0]], dtype=np.float32)
    second = np.array([[66, 49, 42, 16, -0.30, 0.95, 0]], dtype=np.float32)

    tracker.update(first, _img())
    output = tracker.update(second, _img())

    assert output.shape == (1, 9)


@pytest.mark.parametrize(
    "detection",
    (
        np.array([[10, 10, 11, 11, 0.95, 0]], dtype=np.float32),
        np.array([[20, 20, 1, 1, 0.2, 0.95, 0]], dtype=np.float32),
    ),
)
def test_occluboost_geometry_filter_has_aabb_obb_parity(detection: np.ndarray):
    tracker = OccluBoost(
        reid_model=None,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        min_hits=1,
        det_thresh=0.1,
        new_track_thresh=0.1,
        instant_confirm_thresh=0.0,
        obb_det_thresh=0.1,
        obb_new_track_thresh=0.1,
        obb_instant_confirm_thresh=0.0,
        min_box_area=10.0,
        aspect_ratio_thresh=10.0,
    )

    output = tracker.update(detection, _img())

    assert output.shape == (0, 9 if detection.shape[1] == 7 else 8)


def test_strongsort_per_class_obb_keeps_independent_track_pools():
    tracker = create_tracker(
        "strongsort",
        get_tracker_config("strongsort"),
        per_class=True,
        precomputed_reid=True,
        tracker_kwargs={"n_init": 1, "min_conf": 0.1},
    )
    tracker.cmc = DummyCMC()
    detections = np.array(
        [
            [48, 48, 28, 10, 0.25, 0.95, 0],
            [48, 48, 28, 10, 0.25, 0.90, 1],
        ],
        dtype=np.float32,
    )
    embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    output = tracker.update(detections, _img(), embeddings)

    assert output.shape == (2, 9)
    assert len(set(output[:, 5].astype(int))) == 2
    assert {(int(row[7]), int(row[8])) for row in output} == {(0, 0), (1, 1)}
    assert len(tracker.get_class_tracks(0, "pool")) == 1
    assert len(tracker.get_class_tracks(1, "pool")) == 1


def test_strongsort_per_class_obb_keeps_independent_appearance_galleries_across_frames():
    tracker = create_tracker(
        "strongsort",
        get_tracker_config("strongsort"),
        per_class=True,
        precomputed_reid=True,
        tracker_kwargs={"n_init": 1, "min_conf": 0.1},
    )
    tracker.cmc = DummyCMC()
    detections = np.array(
        [
            [48, 48, 28, 10, 0.25, 0.95, 0],
            [48, 48, 28, 10, 0.25, 0.90, 1],
        ],
        dtype=np.float32,
    )
    embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    first = tracker.update(detections, _img(), embeddings)
    second = tracker.update(detections, _img(), embeddings)

    first_ids = {int(row[7]): int(row[5]) for row in first}
    second_ids = {int(row[7]): int(row[5]) for row in second}
    assert second_ids == first_ids
    metric0 = tracker.get_class_track_state(0).attrs["strongsort_metric"]
    metric1 = tracker.get_class_track_state(1).attrs["strongsort_metric"]
    assert metric0 is not metric1
    assert set(metric0.samples) == {first_ids[0]}
    assert set(metric1.samples) == {first_ids[1]}


@pytest.mark.parametrize(("name", "factory"), OBB_TRACKERS)
def test_obb_angle_normalization(name: str, factory: Callable[..., object]):
    tracker = factory(min_hits=1)
    out = _run_until_output(tracker, _obb_dets(angle=(4 * np.pi) + 0.2), _embs(2))

    assert out.shape[1] == 9, name
    assert np.all(out[:, 4] >= -np.pi)
    assert np.all(out[:, 4] < np.pi)


def test_boosttrack_score_filter_preserves_embedding_alignment():
    tracker = BoostTrack(
        reid_model=DummyReID(),
        with_reid=True,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        min_hits=1,
        det_thresh=0.5,
    )
    dets = np.array(
        [
            [10, 10, 30, 60, 0.40, 3],
            [70, 15, 90, 65, 0.95, 5],
        ],
        dtype=np.float32,
    )
    embs = _embs(2)

    tracker.update(dets, _img(), embs)

    assert len(tracker.trackers) == 1
    np.testing.assert_allclose(tracker.trackers[0].get_emb(), embs[1])
    assert tracker.trackers[0].det_ind == 1


def test_boosttrack_duo_obb_treats_equivalent_forms_as_one_candidate_group(monkeypatch):
    tracker = BoostTrack(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        det_thresh=0.6,
    )
    tracker._set_detection_mode(True)
    detections = np.array(
        [
            [40, 40, 24, 8, 0.0, 0.40, 0, 0],
            [40, 40, 8, 24, np.pi / 2, 0.30, 0, 1],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        tracker,
        "get_mh_dist_matrix",
        lambda rows: np.full((len(rows), 1), 20.0, dtype=np.float32),
    )

    boosted = tracker.duo_confidence_boost_obb(detections)

    np.testing.assert_allclose(boosted[:, 5], np.array([0.6001, 0.30]), atol=1e-6)


def test_boosttrack_obb_routes_duo_boost(monkeypatch):
    tracker = BoostTrack(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=True,
        min_hits=1,
        det_thresh=0.5,
    )
    called = False

    def _duo(rows, *, threshold=None):
        nonlocal called
        called = True
        return rows

    monkeypatch.setattr(tracker, "duo_confidence_boost_obb", _duo)

    tracker.update(_obb_dets(), _img(), _embs(2))

    assert called


def test_boosttrack_obb_visual_tracking_dlo_handles_multiple_tracks():
    tracker = BoostTrack(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        use_dlo_boost=True,
        use_duo_boost=False,
        use_rich_s=False,
        use_sb=False,
        use_vt=True,
        det_thresh=0.6,
        min_hits=1,
    )
    tracker.update(_obb_dets(), _img(), _embs(2))
    for track in tracker.trackers:
        track.time_since_update = 1
    detections = np.array([[20, 35, 20, 40, 0.2, 0.3, 3, 0]], dtype=np.float32)

    boosted = tracker.dlo_confidence_boost_obb(detections)

    assert boosted.shape == detections.shape
    assert boosted[0, 5] >= tracker.det_thresh


def test_obb_output_aspect_filter_is_equivalent_form_invariant():
    tracker = BoostTrack(reid_model=None, with_reid=False, use_cmc=False)
    tracker._set_detection_mode(True)
    horizontal = np.array(
        [
            [40, 40, 30, 10, 0.0, 0, 0.9, 0, 0],
            [40, 40, 10, 30, np.pi / 2, 1, 0.9, 0, 1],
        ],
        dtype=np.float32,
    )
    vertical = horizontal.copy()
    vertical[:, 4] += np.pi / 2

    assert len(tracker.filter_outputs_by_geometry(horizontal, max_aspect_ratio=1.6)) == 0
    assert len(tracker.filter_outputs_by_geometry(vertical, max_aspect_ratio=1.6)) == 2


def test_strongsort_score_filter_preserves_embedding_alignment():
    tracker = StrongSort(reid_model=None, min_conf=0.5, n_init=1)
    dets = np.array(
        [
            [10, 10, 30, 60, 0.40, 3],
            [70, 15, 90, 65, 0.95, 5],
        ],
        dtype=np.float32,
    )
    embs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    before = embs.copy()

    tracker.update(dets, _img(), embs)

    assert len(tracker.tracks) == 1
    np.testing.assert_allclose(tracker.tracks[0].features[0], before[1] / 2.0)
    np.testing.assert_allclose(embs, before)
    assert tracker.tracks[0].det_ind == 1
    assert not hasattr(tracker, "tracker")


def test_strongsort_track_model_module_only_owns_single_track_state():
    assert set(strongsort_track_model.__all__) == {"Track", "TrackState"}
    assert hasattr(strongsort_track_model, "Track")
    assert hasattr(strongsort_track_model, "TrackState")
    assert not hasattr(strongsort_track_model, "Detection")
    assert not hasattr(strongsort_track_model, "Tracker")


def test_deepocsort_score_filter_preserves_embedding_alignment():
    tracker = DeepOcSort(
        reid_model=None,
        embedding_off=False,
        cmc_off=True,
        min_hits=1,
        det_thresh=0.5,
    )
    dets = np.array(
        [
            [10, 10, 30, 60, 0.40, 3],
            [70, 15, 90, 65, 0.95, 5],
        ],
        dtype=np.float32,
    )
    embs = _embs(2)

    tracker.update(dets, _img(), embs)

    assert len(tracker.active_tracks) == 1
    np.testing.assert_allclose(tracker.active_tracks[0].get_emb(), embs[1])
    assert tracker.active_tracks[0].det_ind == 1


def test_ocsort_score_filter_preserves_detection_index():
    tracker = OcSort(min_hits=1, det_thresh=0.5, min_conf=0.1)
    dets = np.array(
        [
            [10, 10, 30, 60, 0.40, 3],
            [70, 15, 90, 65, 0.95, 5],
        ],
        dtype=np.float32,
    )

    out = tracker.update(dets, _img())

    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].det_ind == 1
    assert out.shape == (1, 8)
    assert int(out[0, 7]) == 1


def test_bytetrack_score_filter_preserves_detection_index():
    tracker = ByteTrack(track_thresh=0.5, min_conf=0.1)
    dets = np.array(
        [
            [10, 10, 30, 60, 0.40, 3],
            [70, 15, 90, 65, 0.95, 5],
        ],
        dtype=np.float32,
    )

    out = tracker.update(dets, _img())

    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].det_ind == 1
    assert out.shape == (1, 8)
    assert int(out[0, 7]) == 1


def test_sfsort_score_filter_preserves_detection_index():
    tracker = SFSORT(high_th=0.5, low_th=0.1, new_track_th=0.5, match_th_first=0.5)
    dets = np.array(
        [
            [10, 10, 30, 60, 0.40, 3],
            [70, 15, 90, 65, 0.95, 5],
        ],
        dtype=np.float32,
    )

    out = tracker.update(dets, _img())

    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].det_ind == 1
    assert out.shape == (1, 8)
    assert int(out[0, 7]) == 1


def test_occluboost_score_filter_preserves_embedding_alignment():
    tracker = OccluBoost(
        reid_model=DummyReID(),
        with_reid=True,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        min_hits=1,
        det_thresh=0.5,
        new_track_thresh=0.5,
        instant_confirm_thresh=0.0,
    )
    dets = np.array(
        [
            [10, 10, 30, 60, 0.40, 3],
            [70, 15, 90, 65, 0.95, 5],
        ],
        dtype=np.float32,
    )
    embs = _embs(2)

    tracker.update(dets, _img(), embs)

    assert len(tracker.trackers) == 1
    np.testing.assert_allclose(tracker.trackers[0].get_emb(), embs[1])
    assert tracker.trackers[0].det_ind == 1


def test_occluboost_gta_aabb_gap_rows_keep_canonical_tracker_geometry():
    tracker = OccluBoost(
        reid_model=DummyReID(),
        with_reid=True,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        new_track_thresh=0.3,
        gta_smooth_tau=0.0,
    )
    tracker.frame_count = 3
    tracker._gta_graveyard[7] = {
        "emb": np.array([1.0, 0.0], dtype=np.float32),
        "last_box": np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32),
        "frame": 1,
        "conf": 0.8,
        "cls": 2.0,
        "is_obb": False,
    }
    detections = np.array([[20, 10, 40, 40, 0.5, 2, 0]], dtype=np.float32)

    unmatched = tracker._gta_resurrect(
        detections,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([0], dtype=int),
        is_obb=False,
    )
    gap_rows = tracker.flush_gta()

    assert unmatched.size == 0
    assert gap_rows.shape == (1, 9)
    # [frame, x1, y1, x2, y2, id, conf, cls, det_ind]
    np.testing.assert_allclose(
        gap_rows[0],
        np.array([2, 10, 5, 25, 30, 7, 0.8, 2, -1], dtype=np.float32),
        atol=1e-4,
    )


def test_occluboost_gta_obb_uses_obb_birth_threshold_and_preserves_angle():
    tracker = OccluBoost(
        reid_model=DummyReID(),
        with_reid=True,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        new_track_thresh=0.95,
        obb_new_track_thresh=0.3,
        gta_smooth_tau=0.0,
    )
    tracker._set_detection_mode(True)
    tracker.frame_count = 3
    tracker._gta_graveyard[11] = {
        "emb": np.array([1.0, 0.0], dtype=np.float32),
        "last_box": np.array([40.0, 40.0, 20.0, 10.0, 3.1], dtype=np.float32),
        "frame": 1,
        "conf": 0.8,
        "cls": 2.0,
        "is_obb": True,
    }
    detections = np.array([[60, 40, 20, 10, -3.1, 0.5, 2, 0]], dtype=np.float32)

    unmatched = tracker._gta_resurrect(
        detections,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([0], dtype=int),
        is_obb=True,
    )
    gap_rows = tracker.flush_gta()

    assert unmatched.size == 0
    assert gap_rows.shape == (1, 10)
    # [frame, cx, cy, w, h, angle, id, conf, cls, det_ind]
    assert int(gap_rows[0, 6]) == 11
    assert int(gap_rows[0, 8]) == 2
    assert int(gap_rows[0, 9]) == -1
    assert abs(abs(float(gap_rows[0, 5])) - np.pi) < 0.1


def test_occluboost_gta_never_resurrects_another_class_id():
    tracker = OccluBoost(
        reid_model=DummyReID(),
        with_reid=True,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        per_class=True,
        obb_new_track_thresh=0.3,
    )
    tracker._set_detection_mode(True)
    tracker.frame_count = 3
    tracker._gta_graveyard[42] = {
        "emb": np.array([1.0, 0.0], dtype=np.float32),
        "last_box": np.array([40.0, 40.0, 20.0, 10.0, 0.2], dtype=np.float32),
        "frame": 1,
        "conf": 0.8,
        "cls": 0.0,
        "is_obb": True,
    }
    detections = np.array([[42, 40, 20, 10, 0.2, 0.9, 1, 0]], dtype=np.float32)

    unmatched = tracker._gta_resurrect(
        detections,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([0], dtype=int),
        is_obb=True,
    )

    np.testing.assert_array_equal(unmatched, np.array([0]))
    assert 42 in tracker._gta_graveyard
    assert not tracker.trackers


@pytest.mark.parametrize(
    ("name", "factory", "tracks_attr"),
    (
        (
            "botsort",
            lambda: BotSort(
                reid_model=None,
                with_reid=False,
                use_cmc=False,
                min_hits=1,
            ),
            "active_tracks",
        ),
        (
            "bytetrack",
            lambda: ByteTrack(
                min_hits=1,
                track_thresh=0.5,
                min_conf=0.1,
            ),
            "active_tracks",
        ),
        (
            "ocsort",
            lambda: OcSort(
                min_hits=1,
                det_thresh=0.5,
                min_conf=0.1,
            ),
            "active_tracks",
        ),
        (
            "boosttrack",
            lambda: BoostTrack(
                reid_model=None,
                with_reid=False,
                use_cmc=False,
                use_dlo_boost=False,
                use_duo_boost=False,
                min_hits=1,
                det_thresh=0.5,
            ),
            "trackers",
        ),
        (
            "occluboost",
            lambda: OccluBoost(
                reid_model=None,
                with_reid=False,
                use_cmc=False,
                use_dlo_boost=False,
                use_duo_boost=False,
                min_hits=1,
                det_thresh=0.5,
                new_track_thresh=0.5,
                instant_confirm_thresh=0.0,
            ),
            "trackers",
        ),
        (
            "deepocsort",
            lambda: DeepOcSort(
                reid_model=None,
                embedding_off=True,
                cmc_off=True,
                min_hits=1,
                det_thresh=0.5,
            ),
            "active_tracks",
        ),
        (
            "hybridsort",
            lambda: _hybridsort(
                min_hits=1,
                det_thresh=0.5,
                track_thresh=0.5,
            ),
            "active_tracks",
        ),
        (
            "sfsort",
            lambda: SFSORT(
                high_th=0.5,
                low_th=0.1,
                new_track_th=0.5,
                match_th_first=0.5,
            ),
            "active_tracks",
        ),
    ),
)
def test_kalman_track_meta_mirrors_public_fields(name: str, factory: Callable[[], object], tracks_attr: str):
    tracker = factory()
    dets = np.array([[70, 15, 90, 65, 0.95, 5]], dtype=np.float32)

    tracker.update(dets, _img(), _embs(1))
    track = getattr(tracker, tracks_attr)[0]

    assert isinstance(track.meta, TrackMeta), name
    for attr_name in ("id", "age", "hit_streak", "time_since_update", "conf", "cls", "det_ind"):
        if hasattr(track, attr_name):
            assert getattr(track.meta, attr_name) == getattr(track, attr_name), name


def test_boosttrack_confidence_boost_does_not_mutate_input_dets():
    tracker = BoostTrack(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        use_dlo_boost=True,
        use_duo_boost=True,
        min_hits=1,
        det_thresh=0.5,
    )
    tracker.update(np.array([[10, 10, 30, 60, 0.95, 3]], dtype=np.float32), _img())

    dets = np.array([[11, 11, 31, 61, 0.30, 3]], dtype=np.float32)
    before = dets.copy()
    tracker.update(dets, _img())

    np.testing.assert_allclose(dets, before)
