"""Distance limits remain independent of tracker score and appearance gates."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot import ByteTrackConfig, MafHdaConfig
from boxmot.structures import Boxes, Detections, Frame, GeometryKind, MaskBatch, OrientedBoxes
from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.association.boost import linear_assignment as boost_assignment
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.association.matching import linear_assignment, solve_assignment
from boxmot.trackers.common.registry import TRACKER_DEFINITIONS
from boxmot.trackers.maf_hda.association import fusion_cost
from boxmot.trackers.maf_hda.tracker import MafHda
from tests.unit.trackers.test_association_switcher import TRACKER_FACTORIES


def _update(tracker, shift: float, index: int, *, confidence: float = 0.99):
    """Supply image dimensions, native geometry, and any required modality."""
    sample = f"frame-{index}"
    frame = Frame(
        torch.zeros((3, 200, 200), dtype=torch.uint8), sample_id=sample, sequence_id="centroid", frame_index=index
    )
    if tracker.is_obb:
        geometry = OrientedBoxes(torch.tensor([[90.0 + shift, 90.0, 80.0, 80.0, 0.2]]))
    else:
        geometry = Boxes(torch.tensor([[50.0 + shift, 50.0, 130.0 + shift, 130.0]]))
    masks = None
    if tracker.requirements.masks:
        values = torch.zeros((1, 200, 200), dtype=torch.bool)
        values[:, 50:130, int(50 + shift) : int(130 + shift)] = True
        masks = MaskBatch(values)
    detections = Detections(
        geometry=geometry,
        scores=torch.tensor([confidence]),
        class_ids=torch.tensor([0]),
        sample_id=sample,
        masks=masks,
        embeddings=torch.ones((1, 4)) if tracker.requirements.embeddings else None,
    )
    return tracker.update(detections, frame)


TRACKER_GEOMETRIES = [
    (name, is_obb)
    for name in TRACKER_FACTORIES
    for is_obb in (False, True)
    if not is_obb or GeometryKind.OBB in TRACKER_DEFINITIONS[name].capabilities.geometry_kinds
]


def _radius_tracker(name: str, is_obb: bool, *, strict: bool):
    """Exercise direct typed configs with conflicting overlap-stage thresholds."""
    options = {"asso_func": "centroid"}
    if name != "maf_hda":
        options["is_obb"] = is_obb
    if name == "strongsort":
        options["use_cmc"] = False
    original = TRACKER_FACTORIES[name](**options)
    config = original.config
    changes = {"centroid_max_distance": 10.0 / np.hypot(200, 200)}
    for field in (
        "iou_threshold", "second_iou_thresh", "recovery_iou_thresh",
    ):
        if hasattr(config, field):
            changes[field] = 0.99 if strict else 0.01
    for field in (
        "match_thresh", "second_match_thresh", "unconfirmed_match_thresh", "proximity_thresh", "max_iou_dist",
        "match_th_first", "match_th_second",
    ):
        if hasattr(config, field):
            changes[field] = 0.01 if strict else 1.0
    if name == "occluboost":
        changes.update(second_pass_min_hits=0, track_low_thresh=0.01)
    if name in {"ocsort", "hybridsort"}:
        changes["use_byte"] = True
    if name == "sfsort":
        changes.update(dynamic_tuning=True, match_th_first_m=0.08 if strict else 0.02)
    return type(original)(config=replace(config, **changes), **({} if name == "maf_hda" else {"is_obb": is_obb}))


@pytest.mark.parametrize(("name", "is_obb"), TRACKER_GEOMETRIES)
@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize("distance,expected_match", ((9.0, True), (10.0, True), (11.0, False)))
def test_centroid_radius_is_the_only_geometric_cutoff(name, is_obb, strict, distance, expected_match):
    tracker = _radius_tracker(name, is_obb, strict=strict)
    for index in range(4):
        initial = _update(tracker, 0, index)
    result = _update(tracker, distance, 4)
    assert (int(initial.track_ids[0]) in result.track_ids.tolist()) is expected_match


@pytest.mark.parametrize("name", ("bytetrack", "botsort", "ocsort", "hybridsort", "occluboost", "sfsort"))
@pytest.mark.parametrize("is_obb", (False, True))
@pytest.mark.parametrize("distance,expected_match", ((10.0, True), (11.0, False)))
def test_low_confidence_stages_use_the_full_centroid_radius(name, is_obb, distance, expected_match):
    tracker = _radius_tracker(name, is_obb, strict=True)
    for index in range(4):
        initial = _update(tracker, 0, index)
    result = _update(tracker, distance, 4, confidence=0.15)
    assert (int(initial.track_ids[0]) in result.track_ids.tolist()) is expected_match


@pytest.mark.parametrize("is_obb", (False, True))
def test_sfsort_shape_penalties_cannot_reduce_centroid_radius(is_obb):
    tracker = _radius_tracker("sfsort", is_obb, strict=True)
    initial_box = [20, 20, 1, 1, 0.0] if is_obb else [20, 20, 21, 21]
    next_box = [30, 20, 0.1, 0.1, 0.0] if is_obb else [30.45, 20.45, 30.55, 20.55]
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    initial = tracker.update(np.array([[*initial_box, 0.99, 0]]), image)
    costs = tracker.calculate_cost(
        tracker.active_tracks, np.array([next_box]), is_obb=is_obb, association_function=tracker.asso_func
    )
    assert costs[0, 0] > 1.0
    result = tracker.update(np.array([[*next_box, 0.99, 0]]), image)
    assert result[0, 5 if is_obb else 4] == initial[0, 5 if is_obb else 4]


@pytest.mark.parametrize("is_obb", (False, True))
@pytest.mark.parametrize("confidence", (0.15, 0.5))
def test_hybrid_confidence_fusion_only_ranks_centroid_candidates(is_obb, confidence):
    original = _radius_tracker("hybridsort", is_obb, strict=True)
    config = replace(original.config, use_embeddings=True, eg_weight_high_score=1.0, eg_weight_low_score=1.0)
    tracker = type(original)(config=config, is_obb=is_obb)
    for index in range(4):
        initial = _update(tracker, 0, index)
    result = _update(tracker, 10, 4, confidence=confidence)
    assert int(initial.track_ids[0]) in result.track_ids.tolist()


def test_boost_fused_negative_scores_still_rank_radius_eligible_pairs():
    similarity = np.array([[0.0, -np.inf], [-np.inf, 0.0]])
    ranking = np.array([[-1.0, -np.inf], [-np.inf, 1.0]])
    matches, _, _, _ = boost_assignment(
        np.zeros((2, 4)), np.zeros((2, 4)), similarity, ranking, 0.0, gate_before_assignment=True
    )
    np.testing.assert_array_equal(matches, [[0, 0], [1, 1]])


@pytest.mark.parametrize("name", ("boosttrack", "occluboost"))
@pytest.mark.parametrize("is_obb", (False, True))
def test_large_appearance_penalties_do_not_shrink_centroid_radius(name, is_obb):
    original = _radius_tracker(name, is_obb, strict=True)
    tracker = type(original)(config=replace(original.config, use_embeddings=True, lambda_iou=1e6), is_obb=is_obb)
    for index in range(4):
        initial = _update(tracker, 0, index)
    tracker.trackers[0].emb *= -1
    result = _update(tracker, 10, 4)
    assert int(initial.track_ids[0]) in result.track_ids.tolist()


@pytest.mark.parametrize("distance,expected_match", ((300.0, True), (600.0, True), (601.0, False)))
def test_maf_motion_uses_radius_without_overlap_or_likelihood_cutoffs(distance, expected_match):
    tracker = MafHda(config=MafHdaConfig(
        asso_func="centroid", centroid_max_distance=0.6, s2ta_mode="motion", t2ta_mode="motion"
    ))
    initial_id = None
    for index, shift in enumerate((0.0, distance)):
        x = int(10 + shift)
        mask = torch.zeros((1, 600, 800), dtype=torch.bool)
        mask[:, 20:30, x:x + 10] = True
        detections = Detections(
            geometry=Boxes(torch.tensor([[x, 20, x + 10, 30]], dtype=torch.float32)),
            scores=torch.tensor([0.99]), class_ids=torch.tensor([0]), masks=MaskBatch(mask),
            sample_id=f"large-radius/{index}",
        )
        result = tracker.update(detections, np.zeros((600, 800, 3), dtype=np.uint8))
        if index == 0:
            initial_id = int(result.track_ids[0])
    assert (initial_id in result.track_ids.tolist()) is expected_match


def test_maf_centroid_keeps_finite_motion_cost_ordering_beyond_overlap_cutoffs():
    expected_costs = np.array([[20000.0, 15000.0], [15000.0, 20000.0]])
    costs = fusion_cost(
        np.exp(-expected_costs / 100), np.zeros((2, 2)), np.zeros((2, 2)),
        mode="motion", recovery=False, appearance_lower=0.1, appearance_upper=0.7,
        overlap_lower=0, gate_geometry=False,
    )
    np.testing.assert_allclose(costs, expected_costs)
    np.testing.assert_array_equal(solve_assignment(costs), [[0, 1], [1, 0]])


def test_maf_centroid_ignores_upper_appearance_threshold_below_or_above_lower():
    appearance = np.array([[0.2, 0.4, 0.6]])
    costs = []
    for upper in (0.1, 0.9):
        config = MafHdaConfig(asso_func="centroid", appearance_lower=0.4, appearance_upper=upper)
        assert config.appearance_upper == upper
        costs.append(fusion_cost(
            np.ones((1, 3)), appearance, np.array([[1.0, 0.0, 0.8]]),
            mode="appearance", recovery=False, appearance_lower=config.appearance_lower,
            appearance_upper=config.appearance_upper, overlap_lower=0, gate_geometry=False,
        ))
    np.testing.assert_array_equal(costs[0], costs[1])
    assert np.isinf(costs[0][0, 0])
    np.testing.assert_allclose(costs[0][0, 1:], -100 * np.log(appearance[0, 1:]))
    with pytest.raises(ValueError, match="appearance_lower must not exceed appearance_upper"):
        MafHdaConfig(asso_func="iou", appearance_lower=0.4, appearance_upper=0.1)


@pytest.mark.parametrize(("name", "is_obb"), TRACKER_GEOMETRIES)
def test_centroid_tracks_nearby_but_rejects_distant_overlapping_boxes(name, is_obb):
    """Every image tracker retains nearby IDs and rejects motion beyond 5%."""
    options = {"asso_func": "centroid", "centroid_max_distance": 0.05}
    if name != "maf_hda":
        options["is_obb"] = is_obb
    if name == "strongsort":
        options["use_cmc"] = False
    tracker = TRACKER_FACTORIES[name](**options)
    for index in range(4):
        initial = _update(tracker, 0.0, index)
    identity = int(initial.track_ids[0])
    nearby = _update(tracker, 2.0, 4)
    assert identity in nearby.track_ids.tolist()
    distant = _update(tracker, 35.0, 5)
    assert identity not in distant.track_ids.tolist()


@pytest.mark.parametrize("is_obb", (False, True))
def test_centroid_radius_changes_scores_independently_of_geometry(is_obb):
    """Identical centers, the boundary, and beyond use an exact pixel budget."""
    mode = "centroid_obb" if is_obb else "centroid"
    boxes = np.array([[100, 100, 20, 20, 0.3]]) if is_obb else np.array([[90, 90, 110, 110]])
    candidates = np.repeat(boxes.astype(float), 3, axis=0)
    candidates[:, 0] += [0, 50, 60]
    if not is_obb:
        candidates[:, 2] += [0, 50, 60]
    narrow = AssociationFunction(600, 800, mode, centroid_max_distance=0.05).asso_func
    wide = AssociationFunction(600, 800, mode, centroid_max_distance=0.1).asso_func
    np.testing.assert_allclose(narrow(boxes, candidates), [[1.0, 0.0, -0.2]])
    np.testing.assert_allclose(wide(boxes, candidates), [[1.0, 0.5, 0.4]])


@pytest.mark.parametrize("value", (0.0, -0.1, 1.01, float("nan"), float("inf")))
def test_centroid_radius_validation(value):
    with pytest.raises(ValueError, match="centroid_max_distance"):
        ByteTrackConfig(centroid_max_distance=value)


@pytest.mark.parametrize("dimensions", ((None, 10), (0, 10), (-1, 10), (float("nan"), 10), (10, float("inf"))))
def test_centroid_requires_finite_positive_dimensions(dimensions):
    metric = AssociationFunction(*dimensions, "centroid")
    with pytest.raises(ValueError, match="finite positive"):
        metric.asso_func(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]]))


@pytest.mark.parametrize("second_threshold", (0.5, 0.01))
@pytest.mark.parametrize("metric", ("iou", "centroid"))
@pytest.mark.parametrize("is_obb", (False, True))
def test_bytetrack_stage_thresholds_only_control_overlap_metrics(second_threshold, metric, is_obb):
    tracker = ByteTrack(
        config=ByteTrackConfig(asso_func=metric, second_match_thresh=second_threshold), is_obb=is_obb
    )
    initial = _update(tracker, 0.0, 0)
    result = _update(tracker, 2.0, 1, confidence=0.15)
    assert (int(initial.track_ids[0]) in result.track_ids.tolist()) is (metric == "centroid" or second_threshold == 0.5)


@pytest.mark.parametrize(
    "cost_method", ("_fused_association_cost", "_low_association_cost", "_unconfirmed_association_cost")
)
def test_bytetrack_threshold_one_accepts_boundary_but_never_beyond(cost_method):
    tracker = ByteTrack(
        config=ByteTrackConfig(asso_func="centroid", match_thresh=1, second_match_thresh=1, unconfirmed_match_thresh=1)
    )
    tracker._initialize_frame_dimensions(width=600, height=800)
    tracks = [SimpleNamespace(xyxy=np.array([0, 0, 10, 10]))]
    boundary = SimpleNamespace(xyxy=np.array([50, 0, 60, 10]), conf=0.1)
    beyond = SimpleNamespace(xyxy=np.array([51, 0, 61, 10]), conf=0.01)
    costs = getattr(tracker, cost_method)(tracks, [boundary, beyond])
    assert costs[0, 0] == pytest.approx(1)
    assert np.isinf(costs[0, 1])
    matches, _, _ = linear_assignment(costs, 1.0)
    np.testing.assert_array_equal(matches, [[0, 0]])


def test_forbidden_assignment_edges_cannot_steal_valid_match():
    costs = np.array([[np.inf, np.inf], [0.1, 0.2]])
    np.testing.assert_array_equal(solve_assignment(costs), [[1, 0]])


@pytest.mark.parametrize("threshold", (0.7, 0.01))
@pytest.mark.parametrize("metric", ("iou", "centroid"))
def test_bytetrack_tentative_threshold_only_controls_overlap_metrics(threshold, metric):
    tracker = ByteTrack(config=ByteTrackConfig(asso_func=metric, unconfirmed_match_thresh=threshold))
    tracker.update(np.empty((0, 6)), np.zeros((200, 200, 3), dtype=np.uint8))
    assert len(_update(tracker, 0, 1)) == 0
    identity = tracker.active_tracks[0].id
    result = _update(tracker, 2, 2)
    assert (identity in result.track_ids.tolist()) is (metric == "centroid" or threshold == 0.7)


def test_auxiliary_mask_score_cannot_revive_a_distant_pair():
    tracker = ByteTrack(config=ByteTrackConfig(asso_func="centroid", match_thresh=3))
    tracker._initialize_frame_dimensions(width=600, height=800)
    tracker._mask_guidance = SimpleNamespace(condition=lambda costs, *args, **kwargs: np.zeros_like(costs))
    tracks = [SimpleNamespace(id=1, xyxy=np.array([0, 0, 10, 10]))]
    detections = [SimpleNamespace(xyxy=np.array([51, 0, 61, 10]), conf=0.01)]
    assert np.isinf(tracker._fused_association_cost(tracks, detections)[0, 0])


@pytest.mark.parametrize("name", ("botsort", "boosttrack", "hybridsort", "deepocsort", "occluboost", "strongsort"))
@pytest.mark.parametrize("is_obb", (False, True))
def test_identical_appearance_cannot_revive_distant_pairs(name, is_obb):
    options = {"asso_func": "centroid", "is_obb": is_obb}
    if name == "strongsort":
        options["use_cmc"] = False
    original = TRACKER_FACTORIES[name](**options)
    config = original.config
    if hasattr(config, "use_embeddings"):
        config = replace(config, use_embeddings=True)
    if name == "hybridsort":
        config = replace(config, with_longterm_reid_correction=True, eg_weight_high_score=1.0)
    tracker = type(original)(config=config, is_obb=is_obb)
    for index in range(4):
        initial = _update(tracker, 0, index)
    identity = int(initial.track_ids[0])
    result = _update(tracker, 35, 4)
    assert identity not in result.track_ids.tolist()


@pytest.mark.parametrize("is_obb", (False, True))
def test_occluboost_low_confidence_accepts_centroid_boundary(is_obb):
    tracker = TRACKER_FACTORIES["occluboost"](asso_func="centroid", is_obb=is_obb)
    config = replace(
        tracker.config,
        centroid_max_distance=10.0 / np.hypot(200, 200),
        second_iou_thresh=0,
        second_pass_min_hits=0,
        track_low_thresh=0.01,
    )
    tracker = type(tracker)(config=config, is_obb=is_obb)
    # Choose the fraction that gives this frame an exact ten-pixel radius.
    for index in range(4):
        initial = _update(tracker, 0, index)
    identity = int(initial.track_ids[0])
    result = _update(tracker, 10, 4, confidence=0.1)
    assert identity in result.track_ids.tolist()


@pytest.mark.parametrize("name", ("ocsort", "deepocsort", "hybridsort", "boosttrack", "occluboost"))
@pytest.mark.parametrize("is_obb", (False, True))
def test_negative_similarity_threshold_cannot_disable_centroid_limit(name, is_obb):
    original = TRACKER_FACTORIES[name](asso_func="centroid", is_obb=is_obb)
    changes = {"iou_threshold": -1.0}
    tracker = type(original)(config=replace(original.config, **changes), is_obb=is_obb)
    for index in range(4):
        initial = _update(tracker, 0, index)
    result = _update(tracker, 35, 4)
    assert int(initial.track_ids[0]) not in result.track_ids.tolist()


def test_centroid_reid_fusion_preserves_distance_gate_with_permissive_cost_threshold():
    original = TRACKER_FACTORIES["botsort"](asso_func="centroid")
    tracker = type(original)(config=replace(original.config, use_embeddings=True, proximity_thresh=10, match_thresh=3))
    tracker._initialize_frame_dimensions(width=600, height=800)
    tracks = [SimpleNamespace(xyxy=np.array([0, 0, 10, 10]), smooth_feat=np.ones(4))]
    detections = [SimpleNamespace(xyxy=np.array([51, 0, 61, 10]), conf=0.01, curr_feat=np.ones(4))]
    assert np.isinf(tracker._first_association_cost(tracks, detections)[0, 0])


def test_rejected_score_candidates_do_not_steal_eligible_assignment():
    geometry = np.array([[0.8, 0.1], [0.1, 0.1]])
    ranking = np.array([[0.8, 100.0], [100.0, 0.1]])
    matches, _, _, _ = boost_assignment(
        np.zeros((2, 4)), np.zeros((2, 4)), geometry, ranking, 0.5, gate_before_assignment=True
    )
    np.testing.assert_array_equal(matches, [[0, 0]])
