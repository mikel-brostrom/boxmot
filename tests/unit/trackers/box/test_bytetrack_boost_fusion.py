"""Association fusion changes first-pass assignments without adding appearance requirements."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot import BoostTrack, BoostTrackConfig, ByteTrack, ByteTrackConfig, OccluBoost, OccluBoostConfig
from boxmot.trackers.common.association.boost import associate
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig


def _tracker(name: str, is_obb: bool, model: str, mode: str):
    """Use a permissive centroid radius to isolate uncertainty gating."""
    options = dict(asso_func="centroid", centroid_max_distance=1.0, motion_fusion=mode)
    if name == "bytetrack":
        return ByteTrack(ByteTrackConfig(**options), is_obb=is_obb, kalman=KalmanConfig(type=model))
    options.update(
        use_embeddings=False, use_cmc=False, use_dlo_boost=False, use_duo_boost=False,
        min_hits=1, min_box_area=0,
    )
    cls, cfg = (BoostTrack, BoostTrackConfig) if name == "boosttrack" else (OccluBoost, OccluBoostConfig)
    return cls(cfg(**options), is_obb=is_obb, kalman=KalmanConfig(type=model))


def _detection(is_obb: bool, offset: float = 0.0) -> np.ndarray:
    """Return a tall box, keeping both geometries equivalent."""
    box = [40 + offset, 50, 20, 40, 0.0] if is_obb else [30 + offset, 30, 50 + offset, 70]
    return np.array([[*box, 0.99, 0]], dtype=np.float32)


@pytest.mark.parametrize("name", ["bytetrack", "boosttrack", "occluboost"])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("model", ["xyah", "xywh", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("mode", ["off", "gate", "blend"])
def test_first_pass_motion_gate_rejects_implausible_displacement(name, is_obb, model, mode) -> None:
    """A radius-eligible jump keeps an ID only when the new motion gate is off."""
    tracker = _tracker(name, is_obb, model, mode)
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    first = tracker.update(_detection(is_obb), image)
    # Newly initialized filters deliberately have very uncertain velocity.
    # Establish stationary motion before testing an implausible displacement.
    for _ in range(8):
        tracker.update(_detection(is_obb), image)
    second = tracker.update(_detection(is_obb, 150), image)
    id_column = 5 if is_obb else 4
    assert len(first) == 1
    assert (first[0, id_column] in second[:, id_column]) is (mode == "off")


def test_bytetrack_mask_conditioning_cannot_reopen_motion_gate(monkeypatch) -> None:
    """Motion gates run after mask adjustments, even with perfect mask costs."""
    tracker = _tracker("bytetrack", False, "xyah", "gate")
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    first = tracker.update(_detection(False), image)
    monkeypatch.setattr(tracker, "_condition_association", lambda costs, *args, **kwargs: np.zeros_like(costs))
    second = tracker.update(_detection(False, 150), image)
    assert first[0, 4] not in second[:, 4]


def test_boost_motion_gate_overrides_perfect_appearance_and_unique_match() -> None:
    """A single confident pair cannot bypass the optional final-cost hard gate."""
    matches, dets, tracks, costs = associate(
        np.array([[0, 0, 10, 10, 1.0]]),
        np.array([[0, 0, 10, 10, 1.0]]),
        0.1,
        geometry_matrix=np.ones((1, 1)),
        track_confidence=np.ones((1, 1)),
        detection_confidence=np.ones(1),
        emb_cost=np.ones((1, 1)),
        cost_conditioner=lambda costs: np.full_like(costs, np.inf),
    )
    assert not len(matches)
    assert dets.tolist() == tracks.tolist() == [0]
    assert np.isneginf(costs).all()


def test_boost_motion_blend_changes_ranking_before_assignment() -> None:
    """The final-cost hook affects ranking even when geometry has unique pairs."""
    matches, _, _, _ = associate(
        np.array([[0, 0, 10, 10, 1.0], [1, 0, 11, 10, 1.0]]),
        np.array([[0, 0, 10, 10, 1.0], [1, 0, 11, 10, 1.0]]),
        0.5,
        geometry_matrix=np.array([[0.9, 0.7], [0.7, 0.9]]),
        track_confidence=np.ones((2, 1)),
        detection_confidence=np.ones(2),
        cost_conditioner=lambda costs: np.array([[0.9, 0.1], [0.1, 0.9]]),
    )
    assert matches.tolist() == [[0, 1], [1, 0]]
