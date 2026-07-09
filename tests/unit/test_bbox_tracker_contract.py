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

    assert len(tracker.tracker.tracks) == 1
    np.testing.assert_allclose(tracker.tracker.tracks[0].features[0], before[1] / 2.0)
    np.testing.assert_allclose(embs, before)
    assert tracker.tracker.tracks[0].det_ind == 1


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
