"""Shared algorithm contracts exercised through the v24 structured boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes, Tracks
from boxmot.trackers import Tracker, TrackerSpec, create_tracker
from boxmot.trackers.box.boosttrack.track import KalmanBoxTracker as BoostTrackBoxTrack
from boxmot.trackers.box.botsort.track import BaseTrack as BotSortBaseTrack
from boxmot.trackers.box.botsort.track import STrack as BotSortTrack
from boxmot.trackers.box.botsort.track import TrackState as BotSortTrackState
from boxmot.trackers.box.bytetrack.track import BaseTrack as ByteTrackBaseTrack
from boxmot.trackers.box.bytetrack.track import STrack as ByteTrackTrack
from boxmot.trackers.box.bytetrack.track import TrackState as ByteTrackState
from boxmot.trackers.box.bytetrack.tracker import ByteTrack
from boxmot.trackers.box.deepocsort.track import KalmanBoxTracker as DeepOCSortBoxTrack
from boxmot.trackers.box.hybridsort.track import KalmanBoxTracker as HybridSortBoxTrack
from boxmot.trackers.box.ocsort.track import KalmanBoxTracker as OCSortBoxTrack
from boxmot.trackers.box.sfsort.tracker import SFSORT
from boxmot.trackers.box.sfsort.tracker import TrackState as SFSortTrackState
from boxmot.trackers.common.detections import _DetectionBatch
from boxmot.trackers.common.detections.layout import AABB_DETECTIONS, OBB_DETECTIONS
from boxmot.trackers.common.track_state import BoxTrack, SortBoxTrack
from boxmot.trackers.common.tracking.lifecycle import joint_stracks, remove_duplicate_stracks, sub_stracks
from boxmot.trackers.common.tracking.track import (
    TrackIdAllocator,
    TrackLifecycleMixin,
    TrackMeta,
    TrackState,
    sync_track_meta,
)
from boxmot.trackers.registry import TRACKER_DEFINITIONS, get_tracker_class

TRACKER_NAMES = tuple(TRACKER_DEFINITIONS)
BOX_TRACKER_NAMES = tuple(name for name in TRACKER_NAMES if name != "sam2mot")


def _frame(sample_id: str, frame_index: int = 0) -> Frame:
    return Frame(
        image=torch.zeros((3, 96, 128), dtype=torch.uint8),
        sample_id=sample_id,
        sequence_id="sequence",
        frame_index=frame_index,
    )


def _detections(
    rows: np.ndarray,
    *,
    sample_id: str,
    embeddings: np.ndarray | None = None,
) -> Detections:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in (6, 7):
        raise ValueError("test detections must be AABB6 or OBB7 rows")
    is_obb = values.shape[1] == 7
    box_columns = 5 if is_obb else 4
    geometry_values = torch.from_numpy(np.ascontiguousarray(values[:, :box_columns]))
    geometry = OrientedBoxes(geometry_values) if is_obb else Boxes(geometry_values)
    return Detections(
        geometry=geometry,
        scores=torch.from_numpy(np.ascontiguousarray(values[:, box_columns], dtype=np.float32)),
        class_ids=torch.from_numpy(np.ascontiguousarray(values[:, box_columns + 1], dtype=np.int64)),
        sample_id=sample_id,
        embeddings=(
            None
            if embeddings is None
            else torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
        ),
    )


def _update(
    tracker: Tracker,
    rows: np.ndarray,
    *,
    frame_index: int = 0,
    embeddings: np.ndarray | None = None,
) -> Tracks:
    sample_id = f"sequence/{frame_index:06d}"
    if tracker.requirements.embeddings and embeddings is None:
        embeddings = np.ones((len(rows), 4), dtype=np.float32)
    detections = _detections(rows, sample_id=sample_id, embeddings=embeddings)
    frame = _frame(sample_id, frame_index) if tracker.requirements.frame else None
    return tracker.update(detections, frame)


def _aabb_dets() -> np.ndarray:
    return np.array(
        [[10, 10, 30, 60, 0.95, 3], [70, 15, 90, 65, 0.90, 5]],
        dtype=np.float32,
    )


def _obb_dets(angle: float = 0.2) -> np.ndarray:
    return np.array(
        [[20, 35, 20, 40, angle, 0.95, 3], [80, 40, 20, 40, angle, 0.90, 5]],
        dtype=np.float32,
    )


def _empty_rows(*, is_obb: bool) -> np.ndarray:
    return np.empty((0, 7 if is_obb else 6), dtype=np.float32)


@pytest.mark.parametrize("per_class", (False, True))
def test_python_kernel_adapter_preserves_large_int64_class_ids(per_class: bool) -> None:
    class_ids = torch.tensor([2**40 + 123, 2**40 + 125], dtype=torch.int64)
    detections = Detections(
        geometry=Boxes(
            torch.tensor(
                [[1.0, 1.0, 10.0, 20.0], [30.0, 1.0, 40.0, 20.0]],
                dtype=torch.float32,
            )
        ),
        scores=torch.tensor([0.9, 0.91], dtype=torch.float32),
        class_ids=class_ids,
        sample_id="large-classes",
    )
    tracker = ByteTrack(min_hits=1, per_class=per_class)

    tracks = tracker.update(detections)

    assert sorted(tracks.class_ids.tolist()) == sorted(class_ids.tolist())


def _run_until_output(tracker: Tracker, rows: np.ndarray, *, limit: int = 5) -> Tracks:
    result = _update(tracker, rows)
    for frame_index in range(1, limit):
        if len(result):
            break
        result = _update(tracker, rows, frame_index=frame_index)
    return result


@pytest.mark.parametrize("tracker_name", BOX_TRACKER_NAMES)
def test_tracker_constructors_reject_unknown_keywords(tracker_name: str) -> None:
    tracker_class = get_tracker_class(tracker_name)

    with pytest.raises(TypeError, match="unexpected keyword argument 'legacy_option'"):
        tracker_class(legacy_option=True)


@pytest.mark.parametrize("tracker_name", BOX_TRACKER_NAMES)
def test_tracker_constructors_reject_noncanonical_association_casing(tracker_name: str) -> None:
    tracker_class = get_tracker_class(tracker_name)

    with pytest.raises(ValueError, match="canonical lowercase identifier"):
        tracker_class(asso_func="IoU")


@pytest.mark.parametrize(
    ("factory", "canonical_parameter"),
    ((ByteTrack, "track_thresh"), (SFSORT, "high_th")),
)
def test_tracker_specific_thresholds_reject_base_alias(factory, canonical_parameter: str) -> None:
    with pytest.raises(TypeError, match=rf"unexpected keyword argument 'det_thresh'.*{canonical_parameter}"):
        factory(det_thresh=0.5)


@pytest.mark.parametrize("tracker_name", BOX_TRACKER_NAMES)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_all_box_trackers_return_canonical_geometry_and_serializers(
    tracker_name: str,
    geometry: str,
) -> None:
    tracker = create_tracker(
        TrackerSpec(tracker_name, geometry=geometry, options=(("min_hits", 1),))
    )
    rows = _obb_dets() if geometry == "obb" else _aabb_dets()

    output = _run_until_output(tracker, rows)

    assert isinstance(output, Tracks)
    assert output.is_obb is (geometry == "obb")
    serialized = output.to_obb_rows() if geometry == "obb" else output.to_aabb_rows()
    assert serialized.shape == (len(output), 9 if geometry == "obb" else 8)
    assert output.track_ids.dtype is torch.int64
    assert output.class_ids.dtype is torch.int64
    assert output.detection_indices.dtype is torch.int64


@pytest.mark.parametrize("tracker_name", BOX_TRACKER_NAMES)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_empty_batches_preserve_tracker_geometry(tracker_name: str, geometry: str) -> None:
    tracker = create_tracker(TrackerSpec(tracker_name, geometry=geometry))

    output = _update(tracker, _empty_rows(is_obb=geometry == "obb"))

    assert len(output) == 0
    assert output.geometry.values.shape == (0, 5 if geometry == "obb" else 4)


@pytest.mark.parametrize("tracker_name", BOX_TRACKER_NAMES)
def test_reset_clears_tracks_and_restarts_instance_local_ids(tracker_name: str) -> None:
    tracker = create_tracker(
        TrackerSpec(tracker_name, options=(("min_hits", 1),))
    )
    first = _run_until_output(tracker, _aabb_dets()[:1])
    tracker.reset()
    second = _run_until_output(tracker, _aabb_dets()[:1])

    assert len(first) == len(second) == 1
    assert first.track_ids.tolist() == second.track_ids.tolist()


def test_per_class_obb_preserves_frame_global_detection_indices() -> None:
    tracker = create_tracker(
        TrackerSpec("bytetrack", geometry="obb", per_class=True, options=(("min_hits", 1),))
    )

    output = _run_until_output(tracker, _obb_dets())

    assert sorted(output.detection_indices.tolist()) == [0, 1]
    by_detection = dict(zip(output.detection_indices.tolist(), output.class_ids.tolist()))
    assert by_detection == {0: 3, 1: 5}


def test_private_detection_batch_keeps_embedding_and_index_alignment() -> None:
    rows = _aabb_dets()
    embeddings = np.arange(8, dtype=np.float32).reshape(2, 4) + 1.0

    batch = _DetectionBatch.from_layout(rows, AABB_DETECTIONS, embs=embeddings)
    selected = batch.select(np.array([False, True]))
    replaced = batch.with_embs(embeddings + 10)

    np.testing.assert_allclose(batch.boxes, rows[:, :4])
    np.testing.assert_array_equal(batch.det_inds, [0, 1])
    np.testing.assert_array_equal(selected.det_inds, [1])
    np.testing.assert_allclose(selected.embs, embeddings[1:])
    np.testing.assert_allclose(replaced.embs, embeddings + 10)
    np.testing.assert_allclose(batch.embs, embeddings)


def test_private_detection_batch_supports_obb_without_losing_indices() -> None:
    rows = _obb_dets()

    batch = _DetectionBatch.from_layout(rows, OBB_DETECTIONS)
    high, low = batch.split_by_confidence(high_thresh=0.92, low_thresh=0.5)

    np.testing.assert_allclose(batch.boxes, rows[:, :5])
    np.testing.assert_array_equal(high.det_inds, [0])
    np.testing.assert_array_equal(low.det_inds, [1])


def test_detection_batch_rejects_fractional_ids() -> None:
    fractional_class = _aabb_dets()
    fractional_class[0, 5] = 0.5
    with pytest.raises(ValueError, match="class IDs must be integers"):
        _DetectionBatch.from_layout(fractional_class, AABB_DETECTIONS)

    indexed = np.column_stack((_aabb_dets(), np.array([0.5, 1.0], dtype=np.float32)))
    with pytest.raises(ValueError, match="Detection indices must be integers"):
        _DetectionBatch.from_layout(indexed, AABB_DETECTIONS)


def test_canonical_wrapper_does_not_mutate_detection_values() -> None:
    tracker = create_tracker(
        TrackerSpec(
            "boosttrack",
            options=(
                ("min_hits", 1),
                ("use_cmc", False),
                ("use_embeddings", False),
            ),
        )
    )
    rows = _aabb_dets()
    detections = _detections(rows, sample_id="sequence/000000")
    before = detections.geometry.values.clone(), detections.scores.clone(), detections.class_ids.clone()

    tracker.update(detections)

    torch.testing.assert_close(detections.geometry.values, before[0])
    torch.testing.assert_close(detections.scores, before[1])
    torch.testing.assert_close(detections.class_ids, before[2])


def test_obb_observation_validity_uses_confidence_not_geometry_sum() -> None:
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


def test_track_id_allocator_is_instance_local() -> None:
    first = TrackIdAllocator()
    second = TrackIdAllocator()

    assert (first.alloc(), first.alloc(), second.alloc()) == (0, 1, 0)
    first.reset()
    assert first.alloc() == 0


@dataclass
class _DummyTrack:
    id: int
    start_frame: int = 0
    frame_id: int = 1


def test_sync_track_meta_creates_and_refreshes_metadata() -> None:
    track = _DummyTrack(id=7, start_frame=3, frame_id=9)
    track.age = 4
    track.hits = 2
    track.hit_streak = 2
    track.time_since_update = 1
    track.conf = 0.75
    track.cls = 5
    track.det_ind = 11

    meta = sync_track_meta(track, TrackState.TRACKED)

    assert isinstance(meta, TrackMeta)
    assert meta.id == 7
    assert meta.state is TrackState.TRACKED
    assert meta.age == 4
    assert meta.conf == 0.75
    assert meta.cls == 5
    assert meta.det_ind == 11


def test_bytetrack_and_botsort_share_lifecycle_and_box_bases() -> None:
    bytetrack_base = ByteTrackBaseTrack()
    bytetrack_base.track_id = 11
    assert isinstance(bytetrack_base, TrackLifecycleMixin)
    bytetrack_base.mark_lost()
    assert bytetrack_base.state == ByteTrackState.Lost

    botsort_base = BotSortBaseTrack()
    botsort_base.track_id = 12
    assert isinstance(botsort_base, TrackLifecycleMixin)
    botsort_base.mark_long_lost()
    assert botsort_base.state == BotSortTrackState.LongLost

    row = np.array([10, 10, 30, 60, 0.95, 3, 0], dtype=np.float32)
    assert isinstance(ByteTrackTrack(row, max_obs=3, id_allocator=TrackIdAllocator()), BoxTrack)
    assert isinstance(BotSortTrack(row, max_obs=3, id_allocator=TrackIdAllocator()), BoxTrack)


def test_kalman_track_models_share_sort_base() -> None:
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
def test_sfsort_lost_region_remains_track_metadata(box: np.ndarray, expected_region: str) -> None:
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

    _update(tracker, box)
    _update(tracker, _empty_rows(is_obb=False), frame_index=1)

    assert len(tracker.lost_tracks) == 1
    lost_track = tracker.lost_tracks[0]
    assert lost_track.state == SFSortTrackState.Lost
    assert lost_track.lost_region == expected_region
    assert lost_track.meta.lost_region == expected_region


def test_lifecycle_collection_helpers_compare_track_ids() -> None:
    first = _DummyTrack(id=1)
    duplicate = _DummyTrack(id=1)
    second = _DummyTrack(id=2)

    assert joint_stracks([first], [duplicate, second]) == [first, second]
    assert sub_stracks([first, second], [duplicate]) == [second]

    older = _DummyTrack(id=1, start_frame=0, frame_id=10)
    younger = _DummyTrack(id=2, start_frame=8, frame_id=10)
    remaining_a, remaining_b = remove_duplicate_stracks(
        [older],
        [younger],
        distance=lambda _a, _b: np.array([[0.01]], dtype=np.float32),
    )
    assert remaining_a == [older]
    assert remaining_b == []
