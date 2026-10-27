"""Boost geometry reuse preserves confidence cues, output selection, and ordering."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import boxmot.trackers.boosttrack.tracker as boost_module
import boxmot.trackers.common.association.boost as association_module
from boxmot import BoostTrackConfig, OccluBoostConfig
from boxmot.trackers.boosttrack.track import KalmanBoxTracker
from boxmot.trackers.boosttrack.tracker import BoostTrack
from boxmot.trackers.common.association.boost import MhDist_similarity, shape_similarity, shape_similarity_obb
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.motion.models import supported_motion_model_kinds
from boxmot.trackers.occluboost.tracker import OccluBoost


def _tracker(tracker_type: type[BoostTrack] = BoostTrack, *, is_obb: bool = False, **settings: object) -> BoostTrack:
    """Exercise geometry without image models or optional camera estimation."""
    config_type = OccluBoostConfig if tracker_type is OccluBoost else BoostTrackConfig
    config = config_type.from_mapping({
        "use_cmc": False, "use_embeddings": False, "use_dlo_boost": True, "use_duo_boost": True,
        "min_hits": 1, "min_box_area": 0, "aspect_ratio_thresh": 100.0, "det_thresh": 0.5,
        **settings,
    })
    return tracker_type(config, is_obb=is_obb)


def _rows(is_obb: bool) -> np.ndarray:
    """Separate two fractional boxes with different angles and confidence."""
    if is_obb:
        return np.array([[20.25, 30.5, 12.5, 25.75, 0.2, 0.95, 0], [80.25, 90.5, 20.5, 41.75, -0.3, 0.9, 0]])
    return np.array([[10.25, 10.5, 30.75, 50.25, 0.95, 0], [70.25, 70.5, 90.75, 110.25, 0.9, 0]])


@pytest.mark.parametrize("is_obb", [False, True])
def test_decoded_batch_matches_scalar_for_every_supported_filter(is_obb: bool) -> None:
    """The shared geometry snapshot keeps the selected motion adapter's arithmetic."""
    rows = np.column_stack((_rows(is_obb), [0, 1]))
    tracker = _tracker(is_obb=is_obb)
    for kind in supported_motion_model_kinds(is_obb=is_obb):
        tracks = [KalmanBoxTracker(row, 20, is_obb=is_obb, kalman_type=kind, track_id=i) for i, row in enumerate(rows)]
        for track in tracks:
            track.predict()
        expected = np.concatenate([track.get_state() for track in tracks])
        np.testing.assert_array_equal(tracker._track_boxes(tracks), expected)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("rich", [False, True])
def test_dlo_computes_only_consumed_geometry_and_preserves_scores(
    is_obb: bool, rich: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary and rich boosting retain scalar snapshots and avoid unused IoU."""
    tracker = _tracker(is_obb=is_obb, use_rich_s=rich, use_sb=False, use_vt=False)
    tracker.update(_rows(is_obb))
    for track in tracker.trackers:
        track.predict()
    detections = _rows(is_obb)
    detections[:, -2] = [0.2, 0.3]
    detections[:, 0] += 1.25
    rows = np.asarray([np.r_[track.get_state()[0], track.get_confidence()] for track in tracker.trackers])
    motion = tracker.get_mh_dist_matrix(detections)
    if is_obb:
        rows = rows.astype(np.float32)
        if rich:
            geometry = boost_module.soft_biou_batch_obb(detections[:, :6], rows)
            similarity = (
                MhDist_similarity(motion) + shape_similarity_obb(detections[:, :5], rows[:, :5]) + geometry
            ) / 3
        else:
            similarity = AssociationFunction.iou_batch_obb(detections[:, :5], rows[:, :5])
    elif rich:
        geometry = boost_module.soft_biou_batch(detections, rows)
        similarity = (MhDist_similarity(motion) + shape_similarity(detections, rows, tracker.s_sim_corr) + geometry) / 3
    else:
        similarity = AssociationFunction.iou_batch(detections, rows)
    expected = np.clip(np.maximum(detections[:, -2], similarity.max(axis=1) * tracker.dlo_boost_coef), 0, 1)

    calls: list[str] = []
    name = "iou_batch_obb" if is_obb else "iou_batch"
    original = getattr(AssociationFunction, name)

    def record_iou(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(AssociationFunction, name, staticmethod(record_iou))
    if not is_obb and not rich:
        monkeypatch.setattr(boost_module, "soft_biou_batch", lambda *_: pytest.fail("unused buffered IoU"))
    method = tracker.dlo_confidence_boost_obb if is_obb else tracker.dlo_confidence_boost
    actual = method(detections.copy(), tracker_rows=rows, motion_distances=motion)
    np.testing.assert_array_equal(actual[:, -2], expected)
    assert len(calls) == int(is_obb or not rich)


@pytest.mark.parametrize("tracker_type", [BoostTrack, OccluBoost])
@pytest.mark.parametrize("is_obb", [False, True])
def test_motion_cue_is_reused_and_missing_tracks_are_not_decoded_for_output(
    tracker_type: type[BoostTrack], is_obb: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame shares one raw cue across boosting and converts only emitted tracks."""
    tracker = _tracker(tracker_type, is_obb=is_obb, use_rich_s=True)
    tracker.update(_rows(is_obb))
    cue_calls: list[int] = []
    decoded: list[list[int]] = []
    motion = tracker.get_mh_dist_matrix
    boxes = tracker._track_boxes

    def record_motion(detections: np.ndarray, n_dims: int | None = None) -> np.ndarray:
        cue_calls.append(len(detections))
        return motion(detections, n_dims)

    def record_boxes(tracks: list[KalmanBoxTracker]) -> np.ndarray:
        decoded.append([track.id for track in tracks])
        return boxes(tracks)

    monkeypatch.setattr(tracker, "get_mh_dist_matrix", record_motion)
    monkeypatch.setattr(tracker, "_track_boxes", record_boxes)
    tracker.update(_rows(is_obb)[:1])
    assert cue_calls == [1]
    assert decoded and all(set(group) <= {track.id for track in tracker.active_tracks} for group in decoded)
    assert len(tracker.active_tracks) == 1


@pytest.mark.parametrize("is_obb", [False, True])
def test_duplicate_candidates_preserve_ordered_suppression(is_obb: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """A younger row can still suppress later tracks after losing its own pair."""
    tracker = _tracker(OccluBoost, is_obb=is_obb)
    tracker.duplicate_iou_thresh = 0.5
    rng = np.random.default_rng(8)
    for count in (0, 1, 3, 25):
        tracks = [SimpleNamespace(id=i, age=int(age)) for i, age in enumerate(rng.integers(0, 10, size=count))]
        overlaps = rng.random((count, count))
        overlaps = np.maximum(overlaps, overlaps.T)
        if count == 3:
            tracks[0].age, tracks[1].age, tracks[2].age = 1, 3, 0
            overlaps[:] = 1
            overlaps[1, 2] = overlaps[2, 1] = 0
        expected_drop: set[int] = set()
        for i in range(count):
            if i in expected_drop:
                continue
            for j in range(i + 1, count):
                if j not in expected_drop and overlaps[i, j] >= tracker.duplicate_iou_thresh:
                    expected_drop.add(j if tracks[i].age >= tracks[j].age else i)
        tracker.trackers = tracks.copy()
        boxes = np.zeros((count, 5 if is_obb else 4))
        name = "iou_batch_obb" if is_obb else "iou_batch"
        monkeypatch.setattr(AssociationFunction, name, staticmethod(lambda *_: overlaps.copy()))
        # Public callers skip empty/singleton batches; exercise the same boundary.
        emitted = list(zip(tracks, boxes, strict=True))
        actual = tracker._suppress_duplicate_emissions(emitted) if count > 1 else emitted
        expected_ids = [i for i in range(count) if i not in expected_drop]
        assert [track.id for track, _ in actual] == expected_ids
        assert [track.id for track in tracker.trackers] == expected_ids


def test_unmatched_indices_keep_original_then_rejected_match_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linear marking must retain the previous association-stage index ordering."""
    pairs = np.array([[2, 1], [0, 3], [3, 0]])
    monkeypatch.setattr(association_module, "match", lambda *_: pairs)
    similarity = np.zeros((5, 5))
    similarity[2, 1] = 1
    matches, unmatched_dets, unmatched_tracks, _ = association_module.linear_assignment(
        np.zeros((5, 5)), np.zeros((5, 5)), similarity, np.ones((5, 5)), 0.5,
    )
    np.testing.assert_array_equal(matches, [[2, 1]])
    np.testing.assert_array_equal(unmatched_dets, [1, 4, 0, 3])
    np.testing.assert_array_equal(unmatched_tracks, [2, 4, 3, 0])


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
def test_empty_assignment_has_ordered_unmatched_indices(shape: tuple[int, int]) -> None:
    """Empty solver results require no integer indexing into their float array."""
    matches, unmatched_dets, unmatched_tracks, _ = association_module.linear_assignment(
        np.zeros((shape[0], 5)), np.zeros((shape[1], 5)), np.zeros(shape), np.zeros(shape), 0.5,
    )
    assert matches.shape == (0, 2)
    np.testing.assert_array_equal(unmatched_dets, np.arange(shape[0]))
    np.testing.assert_array_equal(unmatched_tracks, np.arange(shape[1]))
