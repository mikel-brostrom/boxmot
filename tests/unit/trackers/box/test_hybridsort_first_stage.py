"""Disabling HybridSORT motion cues must preserve ordinary association."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import HybridSortConfig
from boxmot.structures import Boxes, Detections
from boxmot.trackers.common.association.hybrid import associate_hybrid_with_reid
from boxmot.trackers.hybridsort.tracker import HybridSort


def _detections(rows: np.ndarray, features: np.ndarray | None = None) -> Detections:
    """Build canonical detections without starting an appearance encoder."""
    return Detections(
        geometry=Boxes(torch.as_tensor(rows[:, :4], dtype=torch.float32)),
        scores=torch.as_tensor(rows[:, 4], dtype=torch.float32),
        class_ids=torch.zeros(len(rows), dtype=torch.int64),
        sample_id="sequence/frame",
        embeddings=None if features is None else torch.as_tensor(features, dtype=torch.float32),
    )


def _tracker(*, use_embeddings: bool, min_hits: int = 1) -> HybridSort:
    """Disable TCM and BYTE while retaining configured appearance matching."""
    return HybridSort(
        config=HybridSortConfig(
            cmc_method=None,
            use_embeddings=use_embeddings,
            tcm_first_step=False,
            use_byte=False,
            min_hits=min_hits,
            det_thresh=0.5,
            iou_threshold=0.3,
            asso_func="iou",
            max_age=20,
            max_obs=30,
        )
    )


@pytest.mark.parametrize("use_embeddings", [False, True])
@pytest.mark.parametrize("min_hits", [1, 3])
def test_disabled_tcm_confirms_and_retains_new_track(use_embeddings: bool, min_hits: int) -> None:
    """A stationary detection remains tracked after the initial warmup frames."""
    tracker = _tracker(use_embeddings=use_embeddings, min_hits=min_hits)
    rows = np.array([[10, 10, 30, 40, 0.95]], dtype=float)
    features = np.array([[1.0, 0.0]]) if use_embeddings else None
    detections = _detections(rows, features)

    first = tracker.update(detections)
    identity = first.track_ids.item()
    for _ in range(6):
        output = tracker.update(detections)
        assert output.track_ids.tolist() == [identity]
        assert len(tracker.active_tracks) == 1


def test_disabled_tcm_uses_appearance_when_geometry_is_ambiguous() -> None:
    """Appearance follows identities when identical boxes change detection order."""
    tracker = _tracker(use_embeddings=True)
    rows = np.array([[10, 10, 30, 40, 0.95], [10, 10, 30, 40, 0.95]], dtype=float)
    features = np.eye(2)
    first = tracker.update(_detections(rows, features))
    first_ids = dict(zip(first.detection_indices.tolist(), first.track_ids.tolist(), strict=True))

    second = tracker.update(_detections(rows, features[::-1].copy()))
    second_ids = dict(zip(second.detection_indices.tolist(), second.track_ids.tolist(), strict=True))

    assert second_ids == {0: first_ids[1], 1: first_ids[0]}
    assert len(tracker.active_tracks) == 2


def test_disabled_tcm_does_not_apply_confidence_acceptance_penalty() -> None:
    """Geometry-only acceptance is independent of predicted detector confidence."""
    detections = np.array([[10, 10, 30, 40, 0.2]], dtype=float)
    tracks = np.array([[10, 10, 30, 40, 0.95]], dtype=float)
    arguments = (
        detections,
        tracks,
        0.4,
        (np.zeros((1, 2)),) * 4,
        np.full_like(tracks, -1),
        0.1,
        lambda left, right: np.full((1, 1), 0.8),
    )
    enabled_matches, _, _ = associate_hybrid_with_reid(
        *arguments,
        embedding_cost=np.zeros((1, 1)),
        use_motion_confidence=True,
    )
    disabled_matches, _, _ = associate_hybrid_with_reid(
        *arguments,
        embedding_cost=np.zeros((1, 1)),
        use_motion_confidence=False,
    )

    assert enabled_matches.shape == (0, 2)
    np.testing.assert_array_equal(disabled_matches, [[0, 0]])
