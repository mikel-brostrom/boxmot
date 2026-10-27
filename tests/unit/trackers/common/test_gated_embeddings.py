"""Geometric candidate gates skip appearance work without changing distances."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from boxmot import BotSort, BotSortConfig
from boxmot.trackers.botsort import tracker as botsort_module
from boxmot.trackers.botsort.track import STrack
from boxmot.trackers.common.association import matching


@pytest.mark.parametrize("width", [8, 512, 3584])
@pytest.mark.parametrize("density", [0.0, 0.04, 0.5, 1.0])
@pytest.mark.parametrize("metric", ["cosine", "euclidean", "seuclidean"])
def test_gated_features_match_dense_distances_exactly(width: int, density: float, metric: str) -> None:
    """Dense and sparse dispatch retain the same per-pair arithmetic and gates."""
    rng = np.random.default_rng(42)
    tracks = rng.normal(size=(9, width)).astype(np.float32)
    detections = rng.normal(size=(32, width)).astype(np.float32)
    mask = rng.random((len(tracks), len(detections))) < density
    expected = matching.feature_distance(tracks, detections, metric)
    expected[~mask] = np.inf

    actual = matching.feature_distance(tracks, detections, metric, candidate_mask=mask)

    np.testing.assert_array_equal(actual, expected)


def test_sparse_cosine_evaluates_only_geometrically_possible_pairs(monkeypatch) -> None:
    """Wide embeddings never reach cdist for pairs the geometry gate excludes."""
    rng = np.random.default_rng(7)
    tracks = rng.normal(size=(6, 3584)).astype(np.float32)
    detections = rng.normal(size=(32, 3584)).astype(np.float32)
    tracks[0] = 0  # Preserve the dense path's NaN for a zero-norm candidate.
    mask = np.zeros((len(tracks), len(detections)), dtype=bool)
    mask[np.arange(5), [0, 3, 7, 1, 11]] = True
    expected = matching.feature_distance(tracks, detections)
    expected[~mask] = np.inf
    evaluated_pairs = []
    cdist = matching.cdist

    def observe(left, right, metric):
        evaluated_pairs.append(len(left) * len(right))
        return cdist(left, right, metric)

    monkeypatch.setattr(matching, "cdist", observe)
    actual = matching.embedding_distance(
        [SimpleNamespace(smooth_feat=feature) for feature in tracks],
        [SimpleNamespace(curr_feat=feature) for feature in detections],
        candidate_mask=mask,
    )

    np.testing.assert_array_equal(actual, expected)
    assert sum(evaluated_pairs) == np.count_nonzero(mask)


def test_all_rejected_candidates_do_not_compute_distances(monkeypatch) -> None:
    """An empty geometric gate requires no appearance comparisons."""
    monkeypatch.setattr(matching, "cdist", lambda *args: pytest.fail("No candidate needs comparison"))
    actual = matching.feature_distance(np.ones((3, 8)), np.ones((2, 8)), candidate_mask=np.zeros((3, 2), bool))
    assert np.isinf(actual).all()


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0)])
def test_gated_empty_feature_banks_preserve_shapes(shape: tuple[int, int]) -> None:
    """Empty sides retain the existing float32 empty-matrix contract."""
    actual = matching.feature_distance(
        np.zeros((shape[0], 8)), np.zeros((shape[1], 8)), candidate_mask=np.zeros(shape, bool),
    )
    assert actual.shape == shape
    assert actual.dtype == np.float32


@pytest.mark.parametrize("mask", [np.zeros((2, 2), bool), np.zeros((3, 2), np.int64)])
def test_invalid_candidate_masks_are_rejected(mask: np.ndarray) -> None:
    """The caller's mask must align with the feature banks without broadcasting."""
    with pytest.raises(ValueError, match="boolean matrix"):
        matching.feature_distance(np.ones((3, 8)), np.ones((2, 8)), candidate_mask=mask)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("stage", ["first", "unconfirmed"])
@pytest.mark.parametrize("fuse_iou,motion_fusion", [(False, "off"), (True, "gate"), (True, "blend")])
def test_botsort_sparse_costs_preserve_dense_fusion(
    is_obb: bool, stage: str, fuse_iou: bool, motion_fusion: str, monkeypatch,
) -> None:
    """Sparse matching keeps both association stages and optional cues exact."""
    tracker = BotSort(
        BotSortConfig(use_cmc=False, fuse_iou=fuse_iou, motion_fusion=motion_fusion),
        is_obb=is_obb,
    )
    rng = np.random.default_rng(42)
    features = rng.normal(size=(32, 3584)).astype(np.float32)
    features[0] = 0.0  # Keep the dense formula's handling of an undefined cosine.
    x_centers = 30.0 + np.arange(len(features)) * 100.0
    if is_obb:
        geometry = np.column_stack((
            x_centers, np.full(32, 40.0), np.full(32, 20.0), np.full(32, 30.0), np.full(32, 0.2),
        ))
    else:
        geometry = np.column_stack((x_centers - 10.0, np.full(32, 25.0), x_centers + 10.0, np.full(32, 55.0)))
    indexed_rows = np.column_stack((geometry, np.full(32, 0.9), np.zeros(32), np.arange(32)))
    tracks = [
        STrack(
            row, feature, id_allocator=tracker.id_allocator, is_obb=is_obb, motion_model=tracker.motion_model,
        )
        for row, feature in zip(indexed_rows[:6], features[:6], strict=True)
    ]
    for track in tracks:
        track.activate(tracker.kalman_filter, frame_id=1)
    STrack.multi_predict(tracks)
    indexed_rows[:, 0] += 3.0
    if not is_obb:
        indexed_rows[:, 2] += 3.0
    detections = tracker._create_detections(indexed_rows, features)
    cost_builder = tracker._first_association_cost if stage == "first" else tracker._unconfirmed_association_cost

    def dense_distances(left, right, *, candidate_mask=None):
        return matching.embedding_distance(left, right)

    with monkeypatch.context() as dense:
        dense.setattr(botsort_module, "embedding_distance", dense_distances)
        expected = cost_builder(tracks, detections)

    evaluated_pairs = []
    cdist = matching.cdist

    def observe(left, right, metric):
        evaluated_pairs.append(len(left) * len(right))
        return cdist(left, right, metric)

    monkeypatch.setattr(matching, "cdist", observe)
    actual = cost_builder(tracks, detections)

    np.testing.assert_array_equal(actual, expected)
    assert sum(evaluated_pairs) == len(tracks)
