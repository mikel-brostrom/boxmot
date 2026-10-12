"""Shared OccluBoost settings control OBB births and every association stage."""

from __future__ import annotations

import pytest
import torch

from boxmot import OccluBoostConfig
from boxmot.structures import Detections, OrientedBoxes
from boxmot.trackers.occluboost.tracker import OccluBoost


def _tracker(**overrides: object) -> OccluBoost:
    """Disable confidence boosts so configured stage thresholds stay observable."""
    values = {
        "use_embeddings": False,
        "use_cmc": False,
        "use_dlo_boost": False,
        "use_duo_boost": False,
        "min_hits": 1,
        "det_thresh": 0.5,
        "new_track_thresh": 0.5,
        "instant_confirm_thresh": 0.5,
        "iou_threshold": 0.1,
        "second_pass_min_hits": 0,
        **overrides,
    }
    return OccluBoost(config=OccluBoostConfig.from_mapping(values), is_obb=True)


def _observation(
    frame: int,
    *,
    confidence: float = 0.95,
    shift: float = 0.0,
    present: bool = True,
    embeddings: bool = False,
) -> Detections:
    """Provide a fixed-size oriented box and matching appearance without inference."""
    count = int(present)
    return Detections(
        OrientedBoxes(torch.tensor([[40.0 + shift, 40.0, 20.0, 40.0, 0.2]])[:count]),
        torch.full((count,), confidence),
        torch.zeros(count, dtype=torch.int64),
        sample_id=f"shared/{frame}",
        embeddings=torch.tensor([[1.0, 0.0]])[:count] if embeddings else None,
    )


@pytest.mark.parametrize("parameter", ["det_thresh", "new_track_thresh", "instant_confirm_thresh"])
@pytest.mark.parametrize("threshold,emitted", [(0.4, True), (0.7, False)])
def test_shared_confidence_thresholds_control_obb_births(parameter: str, threshold: float, emitted: bool) -> None:
    """A single confidence knob can suppress detection, creation, or activation."""
    tracker = _tracker(**{parameter: threshold})

    tracks = tracker.update(_observation(0, confidence=0.6))

    assert bool(len(tracks)) is emitted
    if not emitted:
        assert len(tracker.trackers) == int(parameter == "instant_confirm_thresh")


@pytest.mark.parametrize("stage", ["first", "second"])
@pytest.mark.parametrize("threshold,matched", [(0.1, True), (0.95, False)])
def test_shared_matching_thresholds_control_obb_association(stage: str, threshold: float, matched: bool) -> None:
    """First-pass and low-confidence matching both respect shared geometric gates."""
    parameter = "iou_threshold" if stage == "first" else "second_iou_thresh"
    tracker = _tracker(**{parameter: threshold})
    identity = tracker.update(_observation(0)).track_ids.item()

    tracks = tracker.update(_observation(1, confidence=0.95 if stage == "first" else 0.3, shift=5.0))

    assert (identity in tracks.track_ids.tolist()) is matched


@pytest.mark.parametrize("recovery_max_age,recovered", [(1, False), (3, True)])
def test_shared_recovery_age_controls_obb_identity_recovery(recovery_max_age: int, recovered: bool) -> None:
    """Identical appearance cannot recover an identity beyond the configured age."""
    tracker = _tracker(
        use_embeddings=True,
        iou_threshold=0.95,
        recovery_iou_thresh=0.1,
        recovery_max_age=recovery_max_age,
    )
    identity = tracker.update(_observation(0, embeddings=True)).track_ids.item()
    tracker.update(_observation(1, present=False, embeddings=True))

    tracks = tracker.update(_observation(2, shift=5.0, embeddings=True))

    assert (identity in tracks.track_ids.tolist()) is recovered
