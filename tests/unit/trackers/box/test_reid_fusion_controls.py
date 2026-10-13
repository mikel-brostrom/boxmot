"""Appearance trackers fuse cues only at their applicable association stages."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from boxmot import BotSort, BotSortConfig, StrongSort, StrongSortConfig
from boxmot.trackers.botsort.track import STrack
from boxmot.trackers.common.association.matching import chi2inv95, embedding_distance, fuse_score
from boxmot.trackers.common.association.strongsort import gate_cost_matrix
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig
from boxmot.trackers.strongsort.tracker import _Detection


def _box(is_obb: bool, offset: float = 0.0) -> np.ndarray:
    """Use the same twenty-pixel square in either box representation."""
    return np.array([20 + offset, 20, 20, 20, 0] if is_obb else [10 + offset, 10, 30 + offset, 30], dtype=float)


def _bot_pair(tracker: BotSort, *, offset: float = 0.0) -> tuple[STrack, STrack]:
    """Construct a predicted track and an indexed high-confidence observation."""
    allocator = tracker.id_allocator
    track = STrack(
        np.r_[_box(tracker.is_obb), 0.9, 0, 0],
        np.array([1.0, 0.0]),
        id_allocator=allocator,
        is_obb=tracker.is_obb,
        motion_model=tracker.motion_model,
    )
    track.activate(tracker.kalman_filter, frame_id=1)
    STrack.multi_predict([track])
    detection = STrack(
        np.r_[_box(tracker.is_obb, offset), 0.9, 0, 0],
        np.array([1.0, 0.0]),
        id_allocator=allocator,
        is_obb=tracker.is_obb,
        motion_model=tracker.motion_model,
    )
    return track, detection


def _strong_detection(is_obb: bool, *, offset: float = 0.0, confidence: float = 0.9) -> _Detection:
    """Supply StrongSORT's internal TLWH/XYWHA measurement contract."""
    box = _box(is_obb, offset)
    if not is_obb:
        box[2:] -= box[:2]
    return _Detection(box, confidence, 0, 0, np.array([1.0, 0.0]), is_obb=is_obb)


def _strong_pair(tracker: StrongSort, *, offset: float = 0.0) -> _Detection:
    """Seed one confirmed identity and its appearance gallery without model inference."""
    initial = _strong_detection(tracker.is_obb)
    tracker._initiate_track(initial)
    tracker._predict_tracks()
    tracker.metric.partial_fit(np.array([initial.feat]), np.array([tracker.tracks[0].id]), [tracker.tracks[0].id])
    return _strong_detection(tracker.is_obb, offset=offset)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("mode", ["off", "gate", "blend"])
@pytest.mark.parametrize("guided", [False, True])
def test_botsort_motion_gate_survives_masks(is_obb: bool, mode: str, guided: bool) -> None:
    """An auxiliary cue cannot recover a pair rejected by optional motion fusion."""
    tracker = BotSort(BotSortConfig(use_cmc=False, use_embeddings=False, motion_fusion=mode), is_obb=is_obb)
    track, detection = _bot_pair(tracker, offset=80.0)
    if guided:
        tracker._mask_guidance = SimpleNamespace(condition=lambda costs, *args, **kwargs: np.zeros_like(costs))
    costs = tracker._first_association_cost([track], [detection])
    assert bool(np.isfinite(costs[0, 0])) == (mode == "off")
    # The low-confidence recovery stage remains independent of first-pass fusion.
    assert np.isfinite(tracker._second_association_cost([track], [detection])).all()


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("embeddings", [False, True])
def test_botsort_iou_fusion_changes_only_appearance_costs(is_obb: bool, enabled: bool, embeddings: bool) -> None:
    """IoU penalizes a perfect ReID match only when appearance is actually used."""
    tracker = BotSort(
        BotSortConfig(use_cmc=False, use_embeddings=embeddings, fuse_iou=enabled, proximity_thresh=1.0),
        is_obb=is_obb,
    )
    track, detection = _bot_pair(tracker, offset=5.0)
    costs = tracker._first_association_cost([track], [detection])
    if embeddings:
        assert (costs[0, 0] > 0) == enabled
    else:
        expected = fuse_score(tracker.association_distance([track], [detection]), [detection])
        np.testing.assert_array_equal(costs, expected)


@pytest.mark.parametrize("is_obb", [False, True])
def test_botsort_default_cost_matches_existing_formula(is_obb: bool) -> None:
    """New optional cues leave the established geometry/ReID minimum unchanged."""
    tracker = BotSort(BotSortConfig(use_cmc=False), is_obb=is_obb)
    track, detection = _bot_pair(tracker, offset=5.0)
    geometry = tracker.association_distance([track], [detection])
    appearances = embedding_distance([track], [detection])
    appearances[appearances > tracker.appearance_thresh] = 1.0
    appearances[geometry > tracker.proximity_thresh] = 1.0
    expected = np.minimum(fuse_score(geometry, [detection]), appearances)
    np.testing.assert_array_equal(tracker._first_association_cost([track], [detection]), expected)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("mode", ["off", "gate", "blend"])
@pytest.mark.parametrize("guided", [False, True])
def test_strongsort_motion_modes_preserve_hard_gates(is_obb: bool, mode: str, guided: bool) -> None:
    """Turning motion off permits distant appearance matches; masks cannot undo gates."""
    tracker = StrongSort(StrongSortConfig(use_cmc=False, n_init=1, motion_fusion=mode), is_obb=is_obb)
    detection = _strong_pair(tracker, offset=80.0)
    if guided:
        tracker._mask_guidance = SimpleNamespace(condition=lambda costs, *args, **kwargs: np.zeros_like(costs))
    costs = tracker._appearance_cost(tracker.tracks, [detection], [0], [0])
    assert (costs[0, 0] <= tracker.metric.matching_threshold) == (mode == "off")


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("cue", ["fuse_iou", "fuse_detection_score"])
@pytest.mark.parametrize("enabled", [False, True])
def test_strongsort_optional_appearance_cues_change_assignment(is_obb: bool, cue: str, enabled: bool) -> None:
    """A formerly perfect gallery match can fail a strict gate after cue fusion."""
    tracker = StrongSort(
        StrongSortConfig(
            use_cmc=False, n_init=1, motion_fusion="off", max_cos_dist=0.05, max_iou_dist=0.0,
            **{cue: enabled},
        ),
        is_obb=is_obb,
    )
    detection = _strong_pair(tracker, offset=5.0)
    matches, unmatched_tracks, unmatched_detections = tracker._match([detection])
    assert matches == ([] if enabled else [(0, 0)])
    assert unmatched_tracks == ([0] if enabled else [])
    assert unmatched_detections == ([0] if enabled else [])


@pytest.mark.parametrize("is_obb", [False, True])
def test_strongsort_default_motion_cost_matches_existing_formula(is_obb: bool) -> None:
    """Eligible default costs retain the raw-Mahalanobis blend and rejected pairs stay blocked."""
    tracker = StrongSort(StrongSortConfig(use_cmc=False, n_init=1), is_obb=is_obb)
    detections = [_strong_pair(tracker, offset=1.0), _strong_detection(is_obb, offset=80.0)]
    costs = tracker.metric.distance(np.array([detection.feat for detection in detections]), [tracker.tracks[0].id])
    track = tracker.tracks[0]
    measurements = np.asarray([track.measurement_for_detection(detection) for detection in detections])
    distances = track.motion_model.gating_distance(track.kf, track.mean, track.covariance, measurements)
    blocked = distances > chi2inv95[5 if is_obb else 4]
    expected = tracker.mc_lambda * costs + (1.0 - tracker.mc_lambda) * distances
    actual = tracker._appearance_cost(tracker.tracks, detections, [0], [0, 1])
    np.testing.assert_allclose(actual[:, ~blocked], expected[:, ~blocked], rtol=1e-6)
    assert np.isinf(actual[:, blocked]).all()


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("weight", [0.0, 0.98, 1.0])
def test_strongsort_finite_helper_gates_remain_forbidden_to_assignment(is_obb: bool, weight: float) -> None:
    """Even a permissive cost threshold cannot admit a rejected motion edge."""
    tracker = StrongSort(
        StrongSortConfig(use_cmc=False, n_init=1, mc_lambda=weight, max_cos_dist=1e6, max_iou_dist=0.0),
        is_obb=is_obb,
    )
    detection = _strong_pair(tracker, offset=80.0)
    blocked = np.zeros((1, 1), dtype=bool)
    helper_cost = gate_cost_matrix(
        np.zeros((1, 1)), tracker.tracks, [detection], [0], [0], weight, gated_mask=blocked,
    )
    assert np.isfinite(helper_cost).all()
    assert blocked.all()
    assert tracker._match([detection]) == ([], [0], [0])


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("mode", ["gate", "blend"])
def test_strongsort_xyscr_motion_gate_excludes_confidence(is_obb: bool, mode: str) -> None:
    """Score-state discrepancies do not enter the four/five-coordinate geometry gate."""
    tracker = StrongSort(
        StrongSortConfig(use_cmc=False, n_init=1, motion_fusion=mode),
        kalman=KalmanConfig(type="xyscr"),
        is_obb=is_obb,
    )
    detection = _strong_pair(tracker)
    low_confidence = _strong_detection(is_obb, confidence=0.01)
    track = tracker.tracks[0]
    # Make score discrepancies large enough to fail the old full-state gate.
    track.covariance[3, :] = 0.0
    track.covariance[:, 3] = 0.0
    track.covariance[3, 3] = 1e-9
    track.kf.R[3, 3] = 1e-9
    expected = tracker._appearance_cost(tracker.tracks, [detection], [0], [0])
    actual = tracker._appearance_cost(tracker.tracks, [low_confidence], [0], [0])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("name", ["botsort", "strongsort"])
@pytest.mark.parametrize("mode", ["off", "gate", "blend"])
def test_reid_fusion_cannot_bypass_centroid_radius(name: str, mode: str) -> None:
    """Appearance, confidence, and motion ranking retain the outer radius gate."""
    config = dict(use_cmc=False, motion_fusion=mode, fuse_iou=True, asso_func="centroid", centroid_max_distance=0.01)
    if name == "botsort":
        tracker = BotSort(BotSortConfig(**config))
        track, detection = _bot_pair(tracker, offset=5.0)
        tracker.update(np.empty((0, 6)), np.zeros((100, 100, 3), dtype=np.uint8))
        costs = tracker._first_association_cost([track], [detection])
    else:
        tracker = StrongSort(StrongSortConfig(**config, n_init=1, fuse_detection_score=True))
        # Centroid normalization is configured by the public frame-input preparation.
        tracker.update(np.empty((0, 6)), np.zeros((100, 100, 3), dtype=np.uint8))
        detection = _strong_pair(tracker, offset=5.0)
        costs = tracker._appearance_cost(tracker.tracks, [detection], [0], [0])
    assert np.isinf(costs[0, 0])
