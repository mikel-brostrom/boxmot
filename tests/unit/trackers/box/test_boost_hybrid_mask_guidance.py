"""Mask bonuses preserve multi-cue ranking, stage subsets, and appearance gates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from boxmot.structures import Boxes, Detections
from boxmot.trackers.boosttrack.tracker import BoostTrack
from boxmot.trackers.common.association.boost import associate as associate_boost
from boxmot.trackers.common.association.hybrid import associate_hybrid, associate_hybrid_with_reid
from boxmot.trackers.common.association.masks import apply_mask_guidance
from boxmot.trackers.hybridsort.tracker import HybridSort
from boxmot.trackers.occluboost.tracker import OccluBoost


def _mask(box: np.ndarray) -> np.ndarray:
    foreground = np.zeros((100, 160), dtype=bool)
    x1, y1, x2, y2 = box.astype(int)
    foreground[y1:y2, x1:x2] = True
    return foreground


def _conditioner(
    boxes: np.ndarray, masks: list[np.ndarray | None], threshold: float
) -> Callable[[np.ndarray], np.ndarray]:
    def condition(similarity: np.ndarray) -> np.ndarray:
        return 1.0 - apply_mask_guidance(1.0 - similarity.T, boxes, masks, threshold=1.0 - threshold).T

    return condition


def _associate(
    kind: str,
    boxes: np.ndarray,
    geometry: np.ndarray,
    threshold: float,
    conditioner: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, ...]:
    detections = np.column_stack((boxes, np.full(len(boxes), 0.9)))
    tracks = np.column_stack((boxes[: geometry.shape[1]], np.full(geometry.shape[1], 0.9)))
    if kind == "boost":
        return associate_boost(
            detections,
            tracks,
            threshold,
            geometry_matrix=geometry,
            detection_confidence=detections[:, 4],
            track_confidence=tracks[:, 4],
            emb_cost=np.full(geometry.shape, 0.2),
            geometry_conditioner=conditioner,
        )
    arguments = (
        detections,
        tracks,
        threshold,
        (np.zeros((len(tracks), 2)),) * 4,
        np.full_like(tracks, -1.0),
        0.1,
        lambda left, right: geometry,
    )
    if kind == "hybrid":
        return associate_hybrid(*arguments, geometry_conditioner=conditioner)
    return associate_hybrid_with_reid(
        *arguments,
        embedding_cost=np.full(geometry.shape, 0.2),
        embedding_weight=2.0,
        geometry_weight=0.5,
        geometry_conditioner=conditioner,
    )


@pytest.mark.parametrize("kind", ["boost", "hybrid", "hybrid_reid"])
def test_guidance_resolves_ambiguous_multi_cue_assignment(kind: str) -> None:
    boxes = np.array([[10, 10, 30, 40], [14, 10, 34, 40]], dtype=float)
    geometry = np.array([[0.8, 0.6], [0.6, 0.8]])
    original = geometry.copy()
    baseline = _associate(kind, boxes, geometry, 0.5)
    conditioner = _conditioner(boxes, [_mask(boxes[1]), _mask(boxes[0])], 0.5)

    guided = _associate(kind, boxes, geometry, 0.5, conditioner)

    np.testing.assert_array_equal(baseline[0], [[0, 0], [1, 1]])
    np.testing.assert_array_equal(guided[0], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(geometry, original)
    if kind == "boost":
        np.testing.assert_allclose(guided[3] - baseline[3], conditioner(geometry) - geometry)


@pytest.mark.parametrize("kind", ["boost", "hybrid", "hybrid_reid"])
def test_guidance_preserves_clear_matches_and_missing_masks(kind: str) -> None:
    boxes = np.array([[10, 10, 30, 40], [60, 10, 80, 40]], dtype=float)
    for geometry, masks in (
        (np.array([[0.8, 0.1], [0.1, 0.8]]), [_mask(boxes[1]), _mask(boxes[0])]),
        (np.array([[0.8, 0.6], [0.6, 0.8]]), [None, None]),
    ):
        baseline = _associate(kind, boxes, geometry, 0.5)
        guided = _associate(kind, boxes, geometry, 0.5, _conditioner(boxes, masks, 0.5))
        for original, actual in zip(baseline, guided, strict=True):
            np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("kind", ["boost", "hybrid", "hybrid_reid"])
@pytest.mark.parametrize("threshold, accepted", [(0.65, True), (0.75, False)])
def test_isolation_recovery_keeps_configured_threshold(kind: str, threshold: float, accepted: bool) -> None:
    boxes = np.array([[10, 10, 30, 40]], dtype=float)
    foreground = _mask(np.array([10, 10, 22, 40]))  # 60% box fill.
    geometry = np.array([[0.1]])

    result = _associate(kind, boxes, geometry, threshold, _conditioner(boxes, [foreground], threshold))

    assert len(result[0]) == int(accepted)


def _detections(rows: np.ndarray, embeddings: np.ndarray | None = None) -> Detections:
    return Detections(
        Boxes(torch.as_tensor(rows[:, :4], dtype=torch.float32)),
        torch.as_tensor(rows[:, 4], dtype=torch.float32),
        torch.zeros(len(rows), dtype=torch.int64),
        sample_id="guidance-test",
        embeddings=None if embeddings is None else torch.as_tensor(embeddings, dtype=torch.float32),
    )


def _tracker(kind: str, **kwargs: object) -> BoostTrack | OccluBoost | HybridSort:
    common = dict(use_embeddings=False, min_hits=1, det_thresh=0.5, iou_threshold=0.4)
    common.update(kwargs)
    if kind == "hybrid":
        common.setdefault("tcm_byte_step", False)
        return HybridSort(cmc_method=None, **common)
    common.update(use_cmc=False, use_dlo_boost=False, use_duo_boost=False)
    if kind == "occluboost":
        return OccluBoost(use_second_pass=True, second_pass_min_hits=0, second_iou_thresh=0.55, **common)
    return BoostTrack(**common)


@pytest.mark.parametrize("kind", ["boost", "occluboost", "hybrid"])
def test_tracker_wires_detection_order_and_actual_track_ids(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    tracker = _tracker(kind)
    boxes = np.array([[10, 10, 30, 40], [14, 10, 34, 40]], dtype=float)
    rows = np.column_stack((boxes, [0.95, 0.95]))
    masks: dict[int, np.ndarray] = {}
    calls = []

    def condition(similarity, tracks, detections, *, threshold):
        identities = [track.id for track in tracks]
        calls.append((identities, np.asarray(detections).copy(), threshold))
        return _conditioner(detections, [masks.get(identity) for identity in identities], threshold)(similarity)

    monkeypatch.setattr(tracker, "_condition_similarity", condition, raising=False)
    first = tracker.update(_detections(rows)).to_aabb_rows().numpy()
    identities = {int(row[7]): int(row[4]) for row in first}
    for detection_index, identity in identities.items():
        masks[identity] = _mask(boxes[1 - detection_index])

    result = tracker.update(_detections(rows)).to_aabb_rows().numpy()

    assert {int(row[7]): int(row[4]) for row in result} == {0: identities[1], 1: identities[0]}
    assert calls[-1][0] == [identities[0], identities[1]]
    np.testing.assert_array_equal(calls[-1][1], boxes)
    assert calls[-1][2] == 0.4


@pytest.mark.parametrize("kind", ["occluboost", "hybrid"])
@pytest.mark.parametrize("appearance_ok", [False, True])
def test_low_stage_uses_remaining_track_subset_and_keeps_appearance_gate(
    monkeypatch: pytest.MonkeyPatch, kind: str, appearance_ok: bool
) -> None:
    tracker = _tracker(kind, use_embeddings=True)
    initial = np.array([[10, 10, 30, 40, 0.95], [60, 10, 80, 40, 0.95], [100, 10, 120, 40, 0.95]])
    features = np.eye(3)
    calls = []
    masks: dict[int, np.ndarray] = {}

    def condition(similarity, tracks, detections, *, threshold):
        identities = [track.id for track in tracks]
        calls.append((identities, np.asarray(detections).copy(), threshold))
        return _conditioner(detections, [masks.get(identity) for identity in identities], threshold)(similarity)

    monkeypatch.setattr(tracker, "_condition_similarity", condition, raising=False)
    first = tracker.update(_detections(initial, features)).to_aabb_rows().numpy()
    identities = {int(row[7]): int(row[4]) for row in first}
    current = np.array([[60, 50, 80, 80, 0.3], [10, 10, 30, 40, 0.95]])
    masks[identities[1]] = _mask(current[0, :4])
    current_features = features[[1 if appearance_ok else 2, 0]]

    result = tracker.update(_detections(current, current_features)).to_aabb_rows().numpy()

    assert len(result) == 1 + int(appearance_ok)
    low_calls = [call for call in calls if call[1].shape == (1, 4) and call[1][0, 1] == 50]
    assert len(low_calls) == 1
    assert low_calls[0][0] == [identities[1], identities[2]]
    assert low_calls[0][2] == (0.55 if kind == "occluboost" else 0.4)
    if appearance_ok:
        assert identities[1] in result[:, 4]


@pytest.mark.parametrize("appearance_ok", [False, True])
def test_occluboost_recovery_keeps_appearance_gate(monkeypatch: pytest.MonkeyPatch, appearance_ok: bool) -> None:
    tracker = _tracker("occluboost", use_embeddings=True, recovery_iou_thresh=0.12)
    initial = np.array([[10, 10, 30, 40, 0.95], [60, 10, 80, 40, 0.95]])
    features = np.eye(2)
    masks: dict[int, np.ndarray] = {}
    calls = []

    def condition(similarity, tracks, detections, *, threshold):
        if threshold != 0.12:
            return similarity
        identities = [track.id for track in tracks]
        calls.append((identities, threshold))
        return _conditioner(detections, [masks.get(identity) for identity in identities], threshold)(similarity)

    monkeypatch.setattr(tracker, "_condition_similarity", condition, raising=False)
    first = tracker.update(_detections(initial, features)).to_aabb_rows().numpy()
    identities = {int(row[7]): int(row[4]) for row in first}
    current = initial.copy()
    current[1, 1:4:2] += 40
    masks[identities[1]] = _mask(current[1, :4])

    tracker.update(_detections(current, features[[0, 1 if appearance_ok else 0]]))

    assert calls == [([identities[1]], 0.12)]
    original_track = next(track for track in tracker.trackers if track.id == identities[1])
    assert original_track.time_since_update == int(not appearance_ok)


def test_hybrid_final_rematch_conditions_remaining_subsets(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _tracker("hybrid", tcm_first_step=False, use_byte=False)
    initial = np.array([[10, 10, 30, 40, 0.95], [60, 10, 80, 40, 0.95]])
    masks: dict[int, np.ndarray] = {}
    calls = []

    def condition(similarity, tracks, detections, *, threshold):
        identities = [track.id for track in tracks]
        calls.append((identities, np.asarray(detections).copy(), threshold))
        return _conditioner(detections, [masks.get(identity) for identity in identities], threshold)(similarity)

    monkeypatch.setattr(tracker, "_condition_similarity", condition, raising=False)
    first = tracker.update(_detections(initial)).to_aabb_rows().numpy()
    identities = {int(row[7]): int(row[4]) for row in first}
    current = initial[[1]].copy()
    current[:, 1:4:2] += 40
    masks[identities[1]] = _mask(current[0, :4])

    result = tracker.update(_detections(current)).to_aabb_rows().numpy()

    assert result[:, 4].tolist() == [identities[1]]
    assert calls[-1][0] == [identities[0], identities[1]]
    np.testing.assert_array_equal(calls[-1][1], current[:, :4])
    assert calls[-1][2] == 0.4


def test_hybrid_reid_guidance_retains_confidence_consistency_gate() -> None:
    detections = np.array([[10, 10, 30, 40, 0.2]])
    tracks = np.array([[60, 10, 80, 40, 0.95]])
    foreground = _mask(detections[0, :4])

    matches, _, _ = associate_hybrid_with_reid(
        detections,
        tracks,
        0.4,
        (np.zeros((1, 2)),) * 4,
        np.full_like(tracks, -1.0),
        0.1,
        lambda left, right: np.zeros((1, 1)),
        embedding_cost=np.zeros((1, 1)),
        geometry_conditioner=_conditioner(detections[:, :4], [foreground], 0.4),
    )

    # A full mask raises geometry to one, while the original confidence
    # difference still leaves its acceptance similarity below 0.4.
    assert matches.shape == (0, 2)


@pytest.mark.parametrize("confidence_penalty", [False, True])
def test_hybrid_low_stage_preserves_configured_confidence_penalty(
    monkeypatch: pytest.MonkeyPatch, confidence_penalty: bool
) -> None:
    tracker = _tracker("hybrid", tcm_byte_step=confidence_penalty, tcm_byte_step_weight=4.0)
    initial = np.array([[10, 10, 30, 40, 0.95]])
    current = np.array([[60, 10, 80, 40, 0.3]])
    masks = {}

    def condition(similarity, tracks, detections, *, threshold):
        return _conditioner(detections, [masks.get(track.id) for track in tracks], threshold)(similarity)

    monkeypatch.setattr(tracker, "_condition_similarity", condition)
    first = tracker.update(_detections(initial)).to_aabb_rows().numpy()
    masks[int(first[0, 4])] = _mask(current[0, :4])

    result = tracker.update(_detections(current))

    assert len(result) == int(not confidence_penalty)
