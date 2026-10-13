"""Behavioral coverage for configurable motion gates and confidence prediction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import HybridSort, HybridSortConfig, StrongSort, StrongSortConfig
from boxmot.structures import Boxes, Detections, OrientedBoxes, Tracks


def _update_strongsort(tracker: StrongSort, rows: np.ndarray, embeddings: np.ndarray) -> Tracks:
    """Supply appearance through the canonical public detection contract."""
    geometry = OrientedBoxes if tracker.is_obb else Boxes
    return tracker.update(Detections(
        geometry=geometry(torch.tensor(rows[:, :-2], dtype=torch.float32)),
        scores=torch.tensor(rows[:, -2], dtype=torch.float32),
        class_ids=torch.tensor(rows[:, -1], dtype=torch.int64),
        embeddings=torch.from_numpy(embeddings),
        sample_id="sequence/frame",
    ))


@pytest.mark.parametrize("is_obb", (False, True))
@pytest.mark.parametrize("only_position", (False, True))
def test_strongsort_position_gate_can_accept_size_changes(is_obb: bool, only_position: bool) -> None:
    """Center-only gating retains an identity when its size fails the full gate."""
    tracker = StrongSort(
        config=StrongSortConfig(
            use_cmc=False, n_init=1, mc_lambda=1.0, max_iou_dist=0.0,
            gating_only_position=only_position,
        ),
        is_obb=is_obb,
    )
    first = [20, 20, 20, 20, 0, 0.9, 0] if is_obb else [10, 10, 30, 30, 0.9, 0]
    changed = [20, 20, 60, 60, 0, 0.9, 0] if is_obb else [-10, -10, 50, 50, 0.9, 0]
    embedding = np.array([[1.0, 0.0]], dtype=np.float32)
    initial = _update_strongsort(tracker, np.array([first]), embedding)
    result = _update_strongsort(tracker, np.array([changed]), embedding)
    assert (initial.track_ids.item() == result.track_ids.item()) == only_position


@pytest.mark.parametrize("recovery_age", (1, 2))
def test_strongsort_geometry_can_recover_after_a_missing_frame(recovery_age: int) -> None:
    """The configured age controls fallback when the appearance gallery disagrees."""
    tracker = StrongSort(
        config=StrongSortConfig(use_cmc=False, n_init=1, geometric_recovery_max_age=recovery_age)
    )
    detection = np.array([[10, 10, 30, 30, 0.9, 0]])
    first = _update_strongsort(tracker, detection, np.array([[1.0, 0.0]], dtype=np.float32))
    _update_strongsort(tracker, np.empty((0, 6)), np.empty((0, 2), dtype=np.float32))
    recovered = _update_strongsort(tracker, detection, np.array([[0.0, 1.0]], dtype=np.float32))
    assert (first.track_ids.item() == recovered.track_ids.item()) == (recovery_age == 2)


@pytest.mark.parametrize("confidence_floor", (0.1, 0.2))
@pytest.mark.parametrize("previous_confidence", (None, 0.9))
def test_hybridsort_clamps_predicted_confidence(confidence_floor: float, previous_confidence: float | None) -> None:
    """Both low-confidence prediction paths use the authored lower clamp."""
    tracker = HybridSort(
        config=HybridSortConfig(
            use_cmc=False, use_embeddings=False, det_thresh=0.0,
            confidence_floor=confidence_floor,
        )
    )
    tracker.update(np.array([[10, 10, 30, 30, 0.05, 0]]))
    track = tracker.active_tracks[0]
    track.confidence_pre = previous_confidence
    _, _, simple_score = track.predict()
    assert simple_score == pytest.approx(confidence_floor)


def test_hybridsort_rejects_an_inverted_confidence_interval() -> None:
    """A floor above the upper clamp would otherwise be silently ignored."""
    with pytest.raises(ValueError, match="confidence_floor must not exceed track_thresh"):
        HybridSortConfig(confidence_floor=0.8, track_thresh=0.3)
