"""Check mask adjustments at observation-centric association boundaries."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from boxmot.trackers.common.association.masks import apply_mask_guidance
from boxmot.trackers.common.association.velocity import associate
from boxmot.trackers.deepocsort.tracker import DeepOcSort
from boxmot.trackers.ocsort.tracker import OcSort

FRAME = np.zeros((96, 128, 3), dtype=np.uint8)
Tracker = OcSort | DeepOcSort


def _tracker(kind: str, **kwargs: object) -> Tracker:
    if kind == "deepocsort":
        return DeepOcSort(use_embeddings=False, cmc_off=True, min_hits=1, **kwargs)
    return OcSort(min_hits=1, **kwargs)


def _mask(box: np.ndarray) -> np.ndarray:
    mask = np.zeros(FRAME.shape[:2], dtype=bool)
    x1, y1, x2, y2 = box[:4].astype(int)
    mask[y1:y2, x1:x2] = True
    return mask


def _conditioner(
    masks: dict[int, np.ndarray], calls: list[list[int]] | None = None, *, skip_first: bool = False
) -> Callable:
    def condition(similarity, tracks, detections, *, threshold):
        if calls is not None:
            calls.append([track.id for track in tracks])
            if skip_first and len(calls) == 1:
                return similarity
        if not masks:
            return similarity
        return 1.0 - apply_mask_guidance(
            1.0 - similarity.T,
            np.asarray(detections)[:, :4],
            [masks.get(track.id) for track in tracks],
            threshold=1.0 - threshold,
        ).T

    return condition


@pytest.mark.parametrize("kind", ["ocsort", "deepocsort"])
def test_first_association_masks_resolve_ambiguity(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    tracker = _tracker(kind)
    masks = {}
    monkeypatch.setattr(tracker, "_condition_similarity", _conditioner(masks), raising=False)
    boxes = np.array([[10, 10, 30, 40, 0.95, 0], [14, 10, 34, 40, 0.95, 0]])
    first = tracker.update(boxes, FRAME)
    original_assignments = {int(row[7]): int(row[4]) for row in first}
    masks[original_assignments[0]] = _mask(boxes[1])
    masks[original_assignments[1]] = _mask(boxes[0])

    result = tracker.update(boxes, FRAME)

    assert {int(row[7]): int(row[4]) for row in result} == {
        0: original_assignments[1],
        1: original_assignments[0],
    }


@pytest.mark.parametrize("kind", ["ocsort", "deepocsort"])
def test_isolation_bonus_reaches_unique_fast_path_and_final_acceptance(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    tracker = _tracker(kind)
    masks = {}
    monkeypatch.setattr(tracker, "_condition_similarity", _conditioner(masks), raising=False)
    first = tracker.update(np.array([[10, 10, 30, 40, 0.95, 0]]), FRAME)
    identity = int(first[0, 4])
    moved = np.array([[70, 10, 90, 40, 0.95, 0]])
    masks[identity] = _mask(moved[0])

    result = tracker.update(moved, FRAME)

    assert result[:, 4].tolist() == [identity]
    assert len(tracker.active_tracks) == 1
    np.testing.assert_allclose(result[0, :4], moved[0, :4])


@pytest.mark.parametrize("kind", ["ocsort", "deepocsort"])
def test_empty_masks_preserve_outputs_exactly(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    ordinary = _tracker(kind)
    guided = _tracker(kind)
    monkeypatch.setattr(ordinary, "_condition_similarity", lambda similarity, *a, **k: similarity, raising=False)
    monkeypatch.setattr(guided, "_condition_similarity", _conditioner({}), raising=False)
    boxes = np.array([[10, 10, 30, 40, 0.95, 0], [14, 10, 34, 40, 0.95, 0]])

    for rows in [boxes, boxes[::-1], np.empty((0, 6)), boxes]:
        np.testing.assert_array_equal(guided.update(rows, FRAME), ordinary.update(rows, FRAME))


def test_ocsort_low_stage_uses_remaining_track_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _tracker("ocsort", use_byte=True)
    masks = {}
    calls = []
    monkeypatch.setattr(tracker, "_condition_similarity", _conditioner(masks, calls), raising=False)
    initial = np.array(
        [[10, 10, 30, 40, 0.95, 0], [45, 10, 65, 40, 0.95, 0], [80, 10, 100, 40, 0.95, 0]]
    )
    first = tracker.update(initial, FRAME)
    identities = {int(row[7]): int(row[4]) for row in first}
    current = np.array([[10, 10, 30, 40, 0.95, 0], [80, 50, 100, 80, 0.2, 0]])
    masks[identities[2]] = _mask(current[1])
    calls.clear()

    result = tracker.update(current, FRAME)

    assert calls == [[identities[0], identities[1], identities[2]], [identities[1], identities[2]]]
    assert {int(row[7]): int(row[4]) for row in result} == {0: identities[0], 1: identities[2]}


@pytest.mark.parametrize("kind", ["ocsort", "deepocsort"])
def test_ocr_stage_tracks_stay_aligned_after_invalid_observation_filter(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    tracker = _tracker(kind)
    masks = {}
    calls = []
    monkeypatch.setattr(tracker, "_condition_similarity", _conditioner(masks, calls, skip_first=True), raising=False)
    initial = np.array(
        [[10, 10, 30, 40, 0.95, 0], [45, 10, 65, 40, 0.95, 0], [80, 10, 100, 40, 0.95, 0]]
    )
    tracker.update(initial, FRAME)
    tracker.update(initial, FRAME)
    identities = [track.id for track in tracker.active_tracks]
    tracker.active_tracks[0].last_observation[-1] = -1
    moved = np.array([[80, 50, 100, 80, 0.95, 0]])
    masks[identities[2]] = _mask(moved[0])
    calls.clear()

    result = tracker.update(moved, FRAME)

    assert calls == [identities, identities[1:]]
    assert result[:, 4].tolist() == [identities[2]]
    assert len(tracker.active_tracks) == 3


@pytest.mark.parametrize("conditioner", [None, lambda similarity: similarity], ids=["disabled", "empty_masks"])
def test_velocity_conditioning_preserves_geometry_threshold_and_appearance(conditioner: Callable | None) -> None:
    detections = np.array([[0, 0, 10, 10, 0.9], [20, 0, 30, 10, 0.9]])
    similarity = np.full((2, 2), 0.4)
    embeddings = np.array([[0.1, 1.0], [1.0, 0.1]])

    result = associate(
        detections,
        detections,
        lambda *_: similarity,
        0.3,
        np.zeros((2, 2)),
        detections,
        0.2,
        emb_cost=embeddings,
        w_assoc_emb=0.5,
        aw_off=True,
        similarity_conditioner=conditioner,
    )

    assert sorted(result.matches.tolist()) == [[0, 1], [1, 0]]
    np.testing.assert_array_equal(result.cost_matrix, 1.0 - similarity.T)


def test_mask_conditioning_keeps_original_appearance_geometric_gate() -> None:
    detections = np.array([[0, 0, 10, 10, 0.9], [20, 0, 30, 10, 0.9]])
    similarity = np.zeros((2, 2))
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    adjusted = np.array([[0.4, 0.6], [0.6, 0.4]])

    result = associate(
        detections,
        detections,
        lambda *_: similarity,
        0.3,
        np.zeros((2, 2)),
        detections,
        0.2,
        emb_cost=embeddings,
        w_assoc_emb=10,
        aw_off=True,
        similarity_conditioner=lambda _: adjusted,
    )

    assert sorted(result.matches.tolist()) == [[0, 1], [1, 0]]
    np.testing.assert_array_equal(embeddings, np.zeros((2, 2)))
