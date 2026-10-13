"""First-pass fusion preserves observation-centric matching and recovery."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest
import torch

from boxmot.structures import Boxes, Detections, OrientedBoxes
from boxmot.trackers.common.association.boost import associate as associate_boost
from boxmot.trackers.common.association.hybrid import associate_hybrid, associate_hybrid_with_reid
from boxmot.trackers.common.association.velocity import associate
from boxmot.trackers.common.config import get_tracker_config_class, load_tracker_schema
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig
from boxmot.trackers.common.registry import get_tracker_class


def test_velocity_motion_gate_precedes_unique_pair_shortcut() -> None:
    """A geometrically unique pair must still obey an enabled motion gate."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])
    conditioned = []

    def reject(costs: np.ndarray) -> np.ndarray:
        conditioned.append(costs.copy())
        return np.full_like(costs, np.inf)

    result = associate(
        boxes, boxes, lambda *_: np.ones((1, 1)), 0.3, np.zeros((1, 2)), boxes, 0.2,
        cost_conditioner=reject,
    )
    assert len(conditioned) == 1
    assert result.matches.shape == (0, 2)
    np.testing.assert_array_equal(result.unmatched_tracks, [0])
    np.testing.assert_array_equal(result.unmatched_dets, [0])


def test_velocity_fusion_preserves_existing_appearance_weight() -> None:
    """The fusion hook receives the appearance-weighted cost, not geometry alone."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0, 0.9], [1.0, 0.0, 11.0, 10.0, 0.9]])
    geometry = np.array([[0.8, 0.5], [0.6, 0.9]])
    appearance = np.array([[0.7, 0.2], [0.1, 0.8]])
    captured = []

    def record(costs: np.ndarray) -> np.ndarray:
        captured.append(costs.copy())
        return costs

    associate(
        boxes, boxes, lambda *_: geometry, 0.3, np.zeros((2, 2)), boxes, 0.2,
        emb_cost=appearance.copy(), w_assoc_emb=0.7, aw_off=True, cost_conditioner=record,
    )
    np.testing.assert_allclose(captured[0], -(geometry + 0.7 * appearance))


def test_velocity_all_admitted_motion_gate_does_not_lose_geometric_matches() -> None:
    """Appearance-favored inadmissible pairs cannot steal unique valid assignments."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0, 0.9], [1.0, 0.0, 11.0, 10.0, 0.9]])
    result = associate(
        boxes, boxes, lambda *_: np.array([[0.6, 0.1], [0.1, 0.6]]),
        0.3, np.zeros((2, 2)), boxes, 0.0,
        emb_cost=np.array([[0.0, 1.0], [1.0, 0.0]]), w_assoc_emb=1.0, aw_off=True,
        cost_conditioner=lambda costs: costs,
    )
    np.testing.assert_array_equal(result.matches, [[0, 0], [1, 1]])


def test_hybrid_all_admitted_motion_gate_preserves_geometric_matches() -> None:
    """Confidence-ranked forbidden pairs cannot displace geometry-valid matches."""
    detections = np.array([[0.0, 0.0, 10.0, 10.0, 0.9], [1.0, 0.0, 11.0, 10.0, 0.1]])
    tracks = detections.copy()
    tracks[:, 4] = tracks[::-1, 4]
    matches, _, _ = associate_hybrid(
        detections, tracks, 0.3, (np.zeros((2, 2)),) * 4, detections, 0.0,
        lambda *_: np.array([[0.6, 0.1], [0.1, 0.6]]), cost_conditioner=lambda costs: costs,
    )
    np.testing.assert_array_equal(matches, [[0, 0], [1, 1]])


@pytest.mark.parametrize("appearance_rescue", (False, True))
def test_hybrid_fusion_preassignment_gate_preserves_appearance_rescue(appearance_rescue: bool) -> None:
    """Only pairs satisfying the existing geometry-or-appearance rule enter assignment."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0, 0.9], [1.0, 0.0, 11.0, 10.0, 0.9]])
    appearance = np.array([[1.0, 0.0 if appearance_rescue else 0.2], [0.0 if appearance_rescue else 0.2, 1.0]])
    matches, _, _ = associate_hybrid_with_reid(
        boxes, boxes, 0.3, (np.zeros((2, 2)),) * 4, boxes, 0.0,
        lambda *_: np.array([[0.6, 0.1], [0.1, 0.6]]), embedding_cost=appearance,
        embedding_weight=3.0, correct_with_appearance=True, appearance_threshold=0.1,
        cost_conditioner=lambda costs: costs,
    )
    np.testing.assert_array_equal(matches, [[0, 1], [1, 0]] if appearance_rescue else [[0, 0], [1, 1]])


@pytest.mark.parametrize("appearance_rescue", (False, True))
def test_boost_all_admitted_motion_gate_preserves_acceptance_rules(appearance_rescue: bool) -> None:
    """Forced assignment keeps valid geometry and the configured appearance rescue."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0, 0.9], [1.0, 0.0, 11.0, 10.0, 0.9]])
    overlap = 0.2 if appearance_rescue else 0.1
    matches, _, _, _ = associate_boost(
        boxes, boxes, 0.3, geometry_matrix=np.array([[0.6, overlap], [overlap, 0.6]]),
        track_confidence=np.ones(2), detection_confidence=np.ones(2),
        emb_cost=np.array([[0.0, 1.0], [1.0, 0.0]]), cost_conditioner=lambda costs: costs,
    )
    np.testing.assert_array_equal(matches, [[0, 1], [1, 0]] if appearance_rescue else [[0, 0], [1, 1]])


def test_hybrid_obb_all_admitted_motion_gate_cannot_lose_valid_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """OBB appearance ranking must not occupy assignments rejected after solving."""
    module = import_module("boxmot.trackers.hybridsort.tracker")
    config = get_tracker_config_class("hybridsort")(
        min_hits=1, use_cmc=False, asso_func="iou", motion_fusion="gate",
        iou_threshold=0.3, eg_weight_high_score=4.0, longterm_reid_correction_thresh=0.1,
    )
    tracker = get_tracker_class("hybridsort")(config=config, is_obb=True)
    detections = Detections(
        geometry=OrientedBoxes(torch.tensor([[20.0, 20.0, 10.0, 10.0, 0.0], [60.0, 20.0, 10.0, 10.0, 0.0]])),
        scores=torch.tensor([0.95, 0.95]),
        class_ids=torch.tensor([0, 0]),
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        sample_id="fusion",
    )
    first = tracker.update(detections)
    monkeypatch.setattr(module, "feature_distance", lambda *_: np.array([[0.8, 0.2], [0.2, 0.8]]))
    monkeypatch.setattr(tracker, "_first_pass_cost", lambda costs, boxes: costs)
    second = tracker.update(detections)
    assert first.track_ids.tolist() == second.track_ids.tolist()
    assert len(second) == 2


@pytest.mark.parametrize("appearance", (False, True))
def test_hybrid_motion_gate_cannot_be_bypassed_or_rescued(appearance: bool) -> None:
    """Unique geometry and good appearance cannot override the final motion gate."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])
    options = {}
    matcher = associate_hybrid
    if appearance:
        matcher = associate_hybrid_with_reid
        options.update(embedding_cost=np.zeros((1, 1)), correct_with_appearance=True, appearance_threshold=0.2)
    matches, unmatched_dets, unmatched_tracks = matcher(
        boxes, boxes, 0.3, (np.zeros((1, 2)),) * 4, boxes, 0.2, lambda *_: np.ones((1, 1)),
        cost_conditioner=lambda costs: np.full_like(costs, np.inf), **options,
    )
    assert matches.shape == (0, 2)
    np.testing.assert_array_equal(unmatched_tracks, [0])
    np.testing.assert_array_equal(unmatched_dets, [0])


@pytest.mark.parametrize("name", ("ocsort", "deepocsort", "hybridsort"))
def test_observation_score_fusion_weights_geometry_only_when_enabled(name: str) -> None:
    """Score weighting affects first-pass geometry without changing defaults."""
    detections = np.array([[0.0, 0.0, 10.0, 10.0, 0.5, 0.0]])
    similarity = np.array([[0.8]])
    for enabled in (False, True):
        options = {"use_cmc": False, "asso_func": "iou", "fuse_detection_score": enabled}
        if name != "ocsort":
            options["use_embeddings"] = False
        tracker = get_tracker_class(name)(config=get_tracker_config_class(name)(**options))
        if name == "hybridsort":
            result = tracker._first_pass_similarity(similarity, detections[:, :4], detections[:, 4])
        else:
            result = tracker._first_pass_similarity(similarity, [], detections, threshold=0.3)
        np.testing.assert_allclose(result, [[0.4 if enabled else 0.8]])


@pytest.mark.parametrize("name", ("ocsort", "deepocsort", "hybridsort"))
@pytest.mark.parametrize("kind", ("xyah", "xywh", "xysr", "xyhr", "xyscr"))
@pytest.mark.parametrize("is_obb", (False, True))
@pytest.mark.parametrize("mode", ("gate", "blend"))
def test_observation_trackers_fuse_selected_filter_and_geometry(name: str, kind: str, is_obb: bool, mode: str) -> None:
    """Fusion works with every selectable filter and preserves an unchanged identity."""
    settings = {"min_hits": 1, "use_cmc": False, "motion_fusion": mode, "fuse_detection_score": True}
    if name != "ocsort":
        settings["use_embeddings"] = False
    config = get_tracker_config_class(name)(**settings)
    tracker = get_tracker_class(name)(config=config, kalman=KalmanConfig(type=kind), is_obb=is_obb)
    box = [50.0, 50.0, 20.0, 30.0, 0.2] if is_obb else [40.0, 35.0, 60.0, 65.0]
    detections = np.array([[*box, 0.95, 0.0]])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    first = tracker.update(detections, image)
    for _ in range(2):
        result = tracker.update(detections, image)
        assert result.shape == first.shape == (1, 9 if is_obb else 8)
        assert result[0, 5 if is_obb else 4] == first[0, 5 if is_obb else 4]


@pytest.mark.parametrize("name", ("deepocsort", "hybridsort"))
@pytest.mark.parametrize("is_obb", (False, True))
@pytest.mark.parametrize("enabled", (False, True))
def test_observation_appearance_iou_fusion_is_optional(
    monkeypatch: pytest.MonkeyPatch, name: str, is_obb: bool, enabled: bool,
) -> None:
    """Appearance fusion runs only when requested, with the canonical box geometry."""
    module = import_module(f"boxmot.trackers.{name}.tracker")
    original = module.fuse_iou
    calls = []

    def record(costs, tracks, detections, **options):
        calls.append((costs.shape, tracks.shape, detections.shape, options.get("is_obb", False)))
        return original(costs, tracks, detections, **options)

    monkeypatch.setattr(module, "fuse_iou", record)
    config = get_tracker_config_class(name)(min_hits=1, use_cmc=False, fuse_iou=enabled)
    tracker = get_tracker_class(name)(config=config, is_obb=is_obb)
    box = [50.0, 50.0, 20.0, 30.0, 0.2] if is_obb else [40.0, 35.0, 60.0, 65.0]
    detections = Detections(
        geometry=(OrientedBoxes if is_obb else Boxes)(torch.tensor([box])),
        scores=torch.tensor([0.95]),
        class_ids=torch.tensor([0]),
        embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        sample_id="fusion",
    )
    for _ in range(2):
        tracker.update(detections)
    assert bool(calls) is enabled
    for cost_shape, track_shape, detection_shape, oriented in calls:
        assert cost_shape == (1, 1)
        assert track_shape == detection_shape == (1, 5 if is_obb else 4)
        assert oriented is is_obb


@pytest.mark.parametrize("name", ("ocsort", "deepocsort", "hybridsort"))
def test_observation_fusion_search_parameters_follow_enabled_modes(name: str) -> None:
    """Do not spend trials varying inactive blend weights or appearance fusion."""
    from boxmot.engine.tuning.conditions import resolve_fixed_cases

    schema = load_tracker_schema(name)
    schema["motion_fusion"] = {"default": "off"}
    if name != "ocsort":
        schema["use_embeddings"] = {"default": False}
    disabled = resolve_fixed_cases(schema)
    assert disabled["motion_fusion_lambda"] == {"default": 0.98}
    assert disabled["motion_fusion_only_position"] == {"default": False}
    if name != "ocsort":
        assert disabled["fuse_iou"] == {"default": False}
