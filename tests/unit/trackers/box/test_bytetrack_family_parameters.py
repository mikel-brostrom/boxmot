"""New confidence, appearance, and duplicate controls affect actual tracking."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import BotSort, BotSortConfig, ByteTrack, ByteTrackConfig
from boxmot.structures import Boxes, Detections, OrientedBoxes
from boxmot.trackers.common.box.base import BoxTracker


def _tracker(name: str, is_obb: bool, **options: object) -> BoxTracker:
    """Keep model inference disabled for deterministic tracking sequences."""
    if name == "botsort":
        return BotSort(
            BotSortConfig(use_cmc=False, use_embeddings=False, **options),
            is_obb=is_obb,
        )
    return ByteTrack(ByteTrackConfig(**options), is_obb=is_obb)


def _rows(is_obb: bool, *, confidence: float = 0.99, offset: float = 0.0) -> np.ndarray:
    """Use equivalent boxes in both geometry representations."""
    box = [20 + offset, 20, 20, 20, 0.0] if is_obb else [10 + offset, 10, 30 + offset, 30]
    return np.array([[*box, confidence, 0]], dtype=np.float32)


@pytest.mark.parametrize("name", ["botsort", "bytetrack"])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("threshold, remaining", [(0.1, 1), (0.3, 0)])
def test_duplicate_cutoff_controls_active_lost_suppression(
    name: str, is_obb: bool, threshold: float, remaining: int
) -> None:
    """Nearby unmatched detections survive only below the configured IoU-distance cutoff."""
    tracker = _tracker(name, is_obb, match_thresh=0.0, duplicate_distance_thresh=threshold)
    tracker.update(_rows(is_obb))
    tracker.update(_rows(is_obb, offset=2.0))

    assert len(tracker.active_tracks) == remaining
    assert len(tracker.lost_stracks) == 1


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
def test_bytetrack_first_pass_score_fusion_changes_assignment(is_obb: bool, fuse: bool) -> None:
    """Identical geometry can fail the cost gate when confidence is fused."""
    tracker = _tracker(
        "bytetrack", is_obb, track_thresh=0.5, match_thresh=0.3,
        fuse_first_associate=fuse, duplicate_distance_thresh=0.0,
    )
    initial = tracker.update(_rows(is_obb))
    result = tracker.update(_rows(is_obb, confidence=0.65))

    assert len(result) == (0 if fuse else 1)
    if not fuse:
        id_column = 5 if is_obb else 4
        assert result[0, id_column] == initial[0, id_column]


@pytest.mark.parametrize("name", ["botsort", "bytetrack"])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
def test_tentative_score_fusion_controls_confirmation(name: str, is_obb: bool, fuse: bool) -> None:
    """A later-born track requires a tentative match before it is emitted."""
    thresholds = {"track_high_thresh": 0.5, "new_track_thresh": 0.5} if name == "botsort" else {"track_thresh": 0.5}
    tracker = _tracker(
        name, is_obb, **thresholds, unconfirmed_match_thresh=0.3, fuse_unconfirmed_associate=fuse,
    )
    tracker.update(np.empty((0, 7 if is_obb else 6), dtype=np.float32))
    assert len(tracker.update(_rows(is_obb))) == 0
    result = tracker.update(_rows(is_obb, confidence=0.65))
    assert len(result) == (0 if fuse else 1)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_botsort_appearance_alpha_controls_embedding_updates(is_obb: bool, alpha: float) -> None:
    """The configured weight reaches tracks created from canonical detections."""
    tracker = BotSort(BotSortConfig(use_cmc=False, alpha_fixed_emb=alpha), is_obb=is_obb)
    values = torch.from_numpy(_rows(is_obb)[:, :-2].copy())
    for index, feature in enumerate(([1.0, 0.0], [0.0, 1.0])):
        detections = Detections(
            geometry=OrientedBoxes(values) if is_obb else Boxes(values),
            scores=torch.tensor([0.99]),
            class_ids=torch.tensor([0]),
            embeddings=torch.tensor([feature]),
            sample_id=str(index),
        )
        assert len(tracker.update(detections)) == 1

    expected = np.array([alpha, 1.0 - alpha])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(tracker.active_tracks[0].smooth_feat, expected, atol=1e-6)
