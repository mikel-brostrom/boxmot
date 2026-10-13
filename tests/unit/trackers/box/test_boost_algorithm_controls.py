"""Previously fixed boost policies respond to their canonical tracker settings."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import BoostTrackConfig, OccluBoostConfig
from boxmot.structures import Boxes, Detections, OrientedBoxes
from boxmot.trackers.boosttrack.tracker import BoostTrack
from boxmot.trackers.common.association.boost import associate, linear_assignment
from boxmot.trackers.occluboost.tracker import OccluBoost


@pytest.mark.parametrize("config_type", [BoostTrackConfig, OccluBoostConfig])
def test_boost_config_rejects_an_inverted_visibility_interval(config_type: type[BoostTrackConfig]) -> None:
    """An increasing interval would silently hide the configured starting threshold."""
    with pytest.raises(ValueError, match="dlo_threshold_end must not exceed dlo_threshold_start"):
        config_type(dlo_threshold_start=0.4, dlo_threshold_end=0.8)


def _tracker(tracker_type: type[BoostTrack], is_obb: bool = False, **overrides: object) -> BoostTrack:
    """Keep only the policy being tested active without loading models."""
    config_type = OccluBoostConfig if tracker_type is OccluBoost else BoostTrackConfig
    return tracker_type(
        config=config_type.from_mapping(
            {
                "use_cmc": False,
                "use_embeddings": False,
                "use_dlo_boost": False,
                "use_duo_boost": False,
                "min_hits": 1,
                "min_box_area": 0,
                "aspect_ratio_thresh": 10.0,
                "det_thresh": 0.5,
                **overrides,
            }
        ),
        is_obb=is_obb,
    )


def _rows(is_obb: bool, confidence: float = 1.0, shift: float = 0.0) -> np.ndarray:
    """Represent one 20-by-40 box in either public geometry layout."""
    box = [20 + shift, 30, 20, 40, 0] if is_obb else [10 + shift, 10, 30 + shift, 50]
    return np.asarray([[*box, confidence, 0]], dtype=np.float32)


def _ready_tracker(tracker_type: type[BoostTrack], is_obb: bool, **overrides: object) -> BoostTrack:
    """Create and predict one identity before exercising confidence boosting."""
    tracker = _tracker(tracker_type, is_obb, **overrides)
    tracker.update(_rows(is_obb))
    tracker.trackers[0].predict()
    return tracker


@pytest.mark.parametrize("tracker_type", [BoostTrack, OccluBoost])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("alpha", [0.2, 0.8])
def test_first_pass_embedding_ema_uses_configured_baseline(
    tracker_type: type[BoostTrack], is_obb: bool, alpha: float
) -> None:
    """High-confidence accepted detections use the configured EMA endpoint."""
    tracker = _tracker(tracker_type, is_obb, use_embeddings=True, alpha_fixed_emb=alpha)
    geometry_type = OrientedBoxes if is_obb else Boxes
    first_detection = Detections(
        geometry=geometry_type(torch.from_numpy(_rows(is_obb)[:, :-2])),
        scores=torch.ones(1),
        class_ids=torch.zeros(1, dtype=torch.int64),
        embeddings=torch.tensor([[1.0, 0.0]]),
        sample_id="ema/0",
    )
    second_detection = Detections(
        geometry=first_detection.geometry,
        scores=torch.ones(1),
        class_ids=torch.zeros(1, dtype=torch.int64),
        embeddings=torch.tensor([[0.0, 1.0]]),
        sample_id="ema/1",
    )
    first = tracker.update(first_detection)
    second = tracker.update(second_detection)

    assert first.track_ids.item() == second.track_ids.item()
    expected = np.asarray([alpha, 1.0 - alpha])
    np.testing.assert_allclose(tracker.trackers[0].get_emb(), expected / np.linalg.norm(expected), atol=1e-6)


@pytest.mark.parametrize("threshold,fraction,matched", [(0.7, 0.5, True), (0.9, 0.5, False), (0.7, 0.9, False)])
@pytest.mark.parametrize("conditioned", [False, True])
def test_appearance_rescue_respects_both_configurable_gates(
    threshold: float, fraction: float, matched: bool, conditioned: bool
) -> None:
    """Appearance can rescue weak geometry only when both cutoffs permit it."""
    geometry = np.asarray([[0.3, -np.inf]]) if conditioned else np.asarray([[0.3]])
    pairs, _, _, _ = linear_assignment(
        np.zeros((1, 5)),
        np.zeros((geometry.shape[1], 5)),
        geometry,
        np.ones_like(geometry),
        0.5,
        emb_cost=np.full_like(geometry, 0.8),
        appearance_thresh=threshold,
        appearance_iou_fraction=fraction,
    )
    assert bool(len(pairs)) is matched


@pytest.mark.parametrize("multiplier,selected", [(0.0, 0), (2.0, 1)])
def test_appearance_multiplier_changes_first_pass_ranking(multiplier: float, selected: int) -> None:
    """Appearance weight can choose a different eligible identity."""
    pairs, _, _, _ = associate(
        np.zeros((1, 5)),
        np.zeros((2, 5)),
        0.1,
        track_confidence=np.ones(2),
        detection_confidence=np.ones(1),
        geometry_matrix=np.asarray([[0.9, 0.5]]),
        emb_cost=np.asarray([[0.0, 1.0]]),
        lambda_iou=0,
        lambda_shape=0,
        lambda_mhd=0,
        lambda_emb_multiplier=multiplier,
    )
    assert pairs.tolist() == [[0, selected]]


@pytest.mark.parametrize("tracker_type", [BoostTrack, OccluBoost])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("alpha,power,expected", [(0.2, 1.0, 0.52), (0.2, 2.0, 0.328), (0.8, 1.0, 0.28)])
def test_soft_dlo_uses_configured_blend_and_similarity_power(
    tracker_type: type[BoostTrack], is_obb: bool, alpha: float, power: float, expected: float
) -> None:
    """A box with 0.6 IoU receives the requested soft confidence correction."""
    tracker = _ready_tracker(
        tracker_type, is_obb, use_rich_s=False, use_sb=True, use_vt=False,
        dlo_soft_alpha=alpha, dlo_soft_power=power,
    )
    method = tracker.dlo_confidence_boost_obb if is_obb else tracker.dlo_confidence_boost
    boosted = method(_rows(is_obb, confidence=0.2, shift=5.0))
    assert boosted[0, -2] == pytest.approx(expected)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("missing_age,parameter", [(1, "dlo_threshold_start"), (2, "dlo_threshold_end")])
@pytest.mark.parametrize("threshold,boosted", [(0.5, True), (0.7, False)])
def test_varying_dlo_thresholds_control_visibility_boost(
    is_obb: bool, missing_age: int, parameter: str, threshold: float, boosted: bool
) -> None:
    """The initial gate and missing-track floor independently affect boosting."""
    options = {"dlo_threshold_start": 0.95, "dlo_threshold_end": 0.4, parameter: threshold}
    tracker = _ready_tracker(BoostTrack, is_obb, use_rich_s=False, use_sb=False, use_vt=True, **options)
    if missing_age == 2:
        tracker.trackers[0].predict()
    method = tracker.dlo_confidence_boost_obb if is_obb else tracker.dlo_confidence_boost
    result = method(_rows(is_obb, confidence=0.2, shift=5.0))
    assert bool(result[0, -2] > tracker.det_thresh) is boosted


@pytest.mark.parametrize("tracker_type", [BoostTrack, OccluBoost])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("motion_gate,overlap_gate,count", [(100.0, 0.3, 0), (1.0, 0.3, 1), (1.0, 0.8, 2)])
def test_duo_motion_and_overlap_gates_control_candidate_promotion(
    monkeypatch: pytest.MonkeyPatch, tracker_type: type[BoostTrack], is_obb: bool,
    motion_gate: float, overlap_gate: float, count: int,
) -> None:
    """Only distant candidates survive DUO, with duplicates resolved by confidence."""
    tracker = _ready_tracker(
        tracker_type, is_obb, duo_mahalanobis_thresh=motion_gate, duo_iou_thresh=overlap_gate
    )
    monkeypatch.setattr(tracker, "get_mh_dist_matrix", lambda detections, *args: np.full((len(detections), 1), 20.0))
    candidates = np.concatenate((_rows(is_obb, 0.2), _rows(is_obb, 0.3, shift=5.0)))
    method = tracker.duo_confidence_boost_obb if is_obb else tracker.duo_confidence_boost
    result = method(candidates)
    assert np.count_nonzero(result[:, -2] > tracker.det_thresh) == count
    if count == 1:
        assert result[1, -2] > tracker.det_thresh


@pytest.mark.parametrize("tracker_type", [BoostTrack, OccluBoost])
@pytest.mark.parametrize("is_obb", [False, True])
def test_track_confidence_policy_reaches_new_tracks(tracker_type: type[BoostTrack], is_obb: bool) -> None:
    """Young and mature missing identities use the configured confidence decay."""
    tracker = _tracker(tracker_type, is_obb, track_confidence_decay=0.5, track_confidence_warmup=3)
    tracker.update(_rows(is_obb))
    track = tracker.trackers[0]
    assert track.get_confidence() == pytest.approx(0.125)
    track.predict()
    assert track.get_confidence() == pytest.approx(0.25)
    track.predict()
    track.predict()
    assert track.get_confidence() == pytest.approx(0.25)
