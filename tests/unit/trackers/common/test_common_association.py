from __future__ import annotations

import math

import numpy as np
import pytest

from boxmot.trackers.common.association import (
    AssociationFunction,
    AssociationStage,
    detection_track_similarity_assignment,
    iou_distance,
    run_association_stage,
)
from boxmot.trackers.common.association.boost import associate as boost_associate
from boxmot.trackers.common.association.boost import shape_similarity_obb, soft_biou_batch_obb
from boxmot.trackers.common.association.velocity import associate as velocity_associate
from boxmot.trackers.common.detections import OBB_DETECTIONS


def test_obb_velocity_association_uses_negative_angle_observation():
    detections = np.array(
        [
            [-10, 0, 4, 2, -0.5, 0.9],
            [10, 0, 4, 2, -0.5, 0.9],
        ],
        dtype=np.float32,
    )
    tracks = np.array([[0, 0, 4, 2, -0.5, 0.0]], dtype=np.float32)
    previous = np.array([[0, 0, 4, 2, -0.5, 0.9]], dtype=np.float32)

    result = velocity_associate(
        detections,
        tracks,
        lambda left, right: np.full((len(left), len(right)), 0.5, dtype=np.float32),
        0.3,
        velocities=np.array([[0.0, 1.0]], dtype=np.float32),
        previous_obs=previous,
        vdc_weight=0.2,
        is_obb=True,
    )

    np.testing.assert_array_equal(result.matches, np.array([[0, 1]]))


def test_association_stage_solves_matches_and_unmatched_indices():
    cost_matrix = np.array(
        [
            [0.1, 0.9, 0.8],
            [0.7, 0.2, 0.6],
        ],
        dtype=np.float32,
    )
    stage = AssociationStage(
        name="unit",
        cost=lambda tracks, detections: cost_matrix,
        threshold=0.5,
    )

    result = run_association_stage(stage, tracks=["t0", "t1"], detections=["d0", "d1", "d2"])

    np.testing.assert_array_equal(result.matches, np.array([[0, 0], [1, 1]]))
    np.testing.assert_array_equal(result.unmatched_tracks, np.empty((0,), dtype=int))
    np.testing.assert_array_equal(result.unmatched_dets, np.array([2]))
    np.testing.assert_array_equal(result.absolute_matches(), np.array([[0, 0], [1, 1]]))
    np.testing.assert_array_equal(result.absolute_unmatched_dets(), np.array([2]))
    assert result.stage.name == "unit"
    np.testing.assert_allclose(result.cost_matrix, cost_matrix)


def test_association_stage_maps_selected_indices_back_to_source_collections():
    def cost(tracks, detections):
        assert tracks == ["t1", "t3"]
        assert detections == ["d0", "d2"]
        return np.array([[0.8, 0.1], [0.2, 0.9]], dtype=np.float32)

    stage = AssociationStage(
        name="selected",
        cost=cost,
        threshold=0.5,
        track_selector=lambda tracks: np.array([1, 3]),
        detection_selector=lambda detections: np.array([0, 2]),
    )

    result = run_association_stage(
        stage,
        tracks=["t0", "t1", "t2", "t3"],
        detections=["d0", "d1", "d2"],
    )

    np.testing.assert_array_equal(result.matches, np.array([[0, 1], [1, 0]]))
    np.testing.assert_array_equal(result.absolute_matches(), np.array([[1, 2], [3, 0]]))


def test_association_stage_handles_empty_cost_matrix():
    stage = AssociationStage(
        name="empty",
        cost=lambda tracks, detections: np.empty((len(tracks), len(detections))),
        threshold=0.5,
    )

    result = run_association_stage(stage, tracks=["t0", "t1"], detections=[])

    assert result.matches.shape == (0, 2)
    np.testing.assert_array_equal(result.unmatched_tracks, np.array([0, 1]))
    np.testing.assert_array_equal(result.unmatched_dets, np.empty((0,), dtype=int))
    np.testing.assert_array_equal(result.absolute_unmatched_tracks(), np.array([0, 1]))


def test_association_stage_accepts_custom_matcher():
    stage = AssociationStage(
        name="matcher",
        threshold=0.5,
        matcher=lambda tracks, detections: (
            np.array([[1, 0]], dtype=int),
            np.array([0], dtype=int),
            np.array([1], dtype=int),
        ),
    )

    result = run_association_stage(stage, tracks=["t0", "t1"], detections=["d0", "d1"])

    np.testing.assert_array_equal(result.matches, np.array([[1, 0]]))
    np.testing.assert_array_equal(result.unmatched_tracks, np.array([0]))
    np.testing.assert_array_equal(result.unmatched_dets, np.array([1]))


def test_detection_track_similarity_assignment_returns_canonical_orientation():
    similarities = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.1, 0.2],
        ],
        dtype=np.float32,
    )

    result = detection_track_similarity_assignment(
        similarities,
        threshold=0.5,
        assignment_solver=lambda cost: np.array([[0, 1], [1, 0], [2, 1]]),
    )

    np.testing.assert_array_equal(result.matches, np.array([[1, 0], [0, 1]]))
    np.testing.assert_array_equal(result.unmatched_tracks, np.empty((0,), dtype=int))
    np.testing.assert_array_equal(result.unmatched_dets, np.array([2]))
    np.testing.assert_allclose(result.cost_matrix, 1.0 - similarities.T)


def test_boost_association_defaults_to_shared_iou():
    detections = np.array([[0, 0, 10, 10, 0.95]], dtype=np.float32)
    trackers = np.array([[0, 0, 10, 10, 0.90]], dtype=np.float32)

    matches, unmatched_dets, unmatched_trackers, cost_matrix = boost_associate(
        detections,
        trackers,
        0.5,
        track_confidence=np.array([0.90]),
        detection_confidence=np.array([0.95]),
    )

    np.testing.assert_array_equal(matches, np.array([[0, 0]]))
    np.testing.assert_array_equal(unmatched_dets, np.empty((0,), dtype=int))
    np.testing.assert_array_equal(unmatched_trackers, np.empty((0,), dtype=int))
    assert cost_matrix.shape == (1, 1)


def test_generic_association_primitives_live_under_common_root():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

    np.testing.assert_allclose(AssociationFunction.iou_batch(boxes, boxes), np.array([[1.0]]))
    np.testing.assert_allclose(iou_distance(boxes, boxes), np.array([[0.0]], dtype=np.float32))


def test_obb_diou_is_one_for_identical_boxes():
    box = np.array([[20.0, 30.0, 12.0, 6.0, 0.4]], dtype=np.float32)
    np.testing.assert_allclose(AssociationFunction.diou_batch_obb(box, box), np.array([[1.0]]), atol=1e-6)


def test_obb_diou_uses_rotated_overlap_and_center_distance():
    reference = np.array([[20.0, 30.0, 16.0, 4.0, 0.0]], dtype=np.float32)
    rotated = np.array([[20.0, 30.0, 16.0, 4.0, np.pi / 2.0]], dtype=np.float32)
    displaced = np.array([[28.0, 30.0, 16.0, 4.0, 0.0]], dtype=np.float32)
    rotated_iou = AssociationFunction.iou_batch_obb(reference, rotated)[0, 0]
    rotated_diou = AssociationFunction.diou_batch_obb(reference, rotated)[0, 0]
    displaced_diou = AssociationFunction.diou_batch_obb(reference, displaced)[0, 0]

    np.testing.assert_allclose(rotated_diou, (rotated_iou + 1.0) / 2.0, atol=1e-6)
    assert displaced_diou < 1.0


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_obb_overlap_metrics_are_one_for_identical_boxes(mode):
    box = np.array([[20.0, 30.0, 12.0, 6.0, 0.4, 0.95]], dtype=np.float32)
    similarity = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func(box, box)

    np.testing.assert_allclose(similarity, np.array([[1.0]]), atol=1e-6)
    assert np.all((0.0 <= similarity) & (similarity <= 1.0))


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_obb_overlap_metrics_accept_identical_ultra_thin_boxes(mode):
    box = np.array([[20.0, 30.0, 1e-9, 2e-8, 0.4]], dtype=np.float64)
    original = box.copy()
    metric = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func

    similarity = metric(box, box)

    np.testing.assert_allclose(similarity, np.array([[1.0]]), atol=1e-6)
    assert np.all((0.0 <= similarity) & (similarity <= 1.0))
    np.testing.assert_array_equal(box, original)


@pytest.mark.parametrize(
    "invalid_box",
    (
        [[0.0, 0.0, 0.0, 2.0, 0.0]],
        [[0.0, 0.0, 2.0, -1.0, 0.0]],
        [[0.0, 0.0, 2.0, 1.0, np.nan]],
        [[np.inf, 0.0, 2.0, 1.0, 0.0]],
    ),
)
def test_obb_association_rejects_nonpositive_or_nonfinite_geometry(invalid_box):
    valid_box = np.array([[0.0, 0.0, 2.0, 1.0, 0.0]])

    with pytest.raises(ValueError):
        AssociationFunction.iou_batch_obb(np.asarray(invalid_box), valid_box)


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_obb_overlap_metrics_return_exact_identity_for_swapped_ultra_thin_representation(mode):
    box = np.array([[1e6, -1e6, 5e-5, 2e-4, 0.4]], dtype=np.float32)
    equivalent = box.copy()
    equivalent[:, 2:4] = equivalent[:, [3, 2]]
    equivalent[:, 4] += np.pi / 2.0
    metric = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func

    assert metric(box, equivalent).item() == 1.0


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou", "centroid"))
def test_obb_metrics_are_invariant_to_equivalent_width_height_representation(mode):
    box = np.array([[20.0, 30.0, 12.0, 6.0, 0.4]], dtype=np.float32)
    equivalent = box.copy()
    equivalent[:, 2:4] = equivalent[:, [3, 2]]
    equivalent[:, 4] += np.pi / 2.0
    metric = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func

    np.testing.assert_allclose(metric(box, equivalent), np.array([[1.0]]), atol=1e-6)


def test_obb_giou_penalizes_empty_space_in_convex_enclosure():
    reference = np.array([[0.0, 0.0, 10.0, 4.0, 0.0]], dtype=np.float32)
    nearby = np.array([[8.0, 0.0, 10.0, 4.0, 0.0]], dtype=np.float32)
    distant = np.array([[20.0, 0.0, 10.0, 4.0, 0.0]], dtype=np.float32)

    nearby_giou = AssociationFunction.giou_batch_obb(reference, nearby)[0, 0]
    distant_giou = AssociationFunction.giou_batch_obb(reference, distant)[0, 0]

    assert distant_giou < nearby_giou < 1.0


def test_obb_iou_resolves_small_angle_difference_for_extreme_aspect_ratio():
    reference = np.array([[0.0, 0.0, 1e6, 1.0, 0.0]])
    slightly_rotated = np.array([[0.0, 0.0, 1e6, 1.0, 9e-7]])

    similarity = AssociationFunction.iou_batch_obb(reference, slightly_rotated)[0, 0]

    assert similarity == pytest.approx(0.6326530612, abs=1e-9)
    assert similarity < 1.0


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    (
        (
            [0.0, 0.0, 300.0, 1.0, 0.0],
            [3.0, 1.0, 230.0, 1.25, 0.001],
            0.05145403049909077,
        ),
        (
            [0.0, 0.0, 300.0, 1.0, 0.7],
            [3.0, 1.0, 230.0, 1.25, 0.7005],
            0.0003673997508027863,
        ),
    ),
    ids=("partial-overlap", "thin-overlap"),
)
def test_obb_iou_resolves_moderately_thin_overlap(reference, candidate, expected):
    similarity = AssociationFunction.iou_batch_obb(np.asarray([reference]), np.asarray([candidate]))[0, 0]

    assert similarity == pytest.approx(expected, abs=1e-14)


def test_obb_iou_prefilter_preserves_small_boxes_at_large_centers():
    reference = np.array([[1e16, -1e16, 1e-3, 2e-3, 0.0]])
    rotated = np.array([[1e16, -1e16, 1e-3, 2e-3, 0.2]])
    local_reference = reference.copy()
    local_rotated = rotated.copy()
    local_reference[:, :2] = 0.0
    local_rotated[:, :2] = 0.0

    np.testing.assert_allclose(
        AssociationFunction.iou_batch_obb(reference, rotated),
        AssociationFunction.iou_batch_obb(local_reference, local_rotated),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_obb_metrics_are_scale_invariant_below_former_side_floor(mode):
    reference = np.array([[0.0, 0.0, 4.0, 2.0, 0.3]])
    candidate = np.array([[1.2, 0.4, 3.0, 1.5, -0.2]])
    scaled_reference = reference.copy()
    scaled_candidate = candidate.copy()
    scaled_reference[:, :4] *= 1e-5
    scaled_candidate[:, :4] *= 1e-5
    metric = getattr(AssociationFunction, f"{mode}_batch_obb")

    np.testing.assert_allclose(
        metric(reference, candidate),
        metric(scaled_reference, scaled_candidate),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
@pytest.mark.parametrize("scale", (1e-200, 1e200), ids=("subnormal-area", "overflow-area"))
def test_obb_metrics_keep_dimensionless_terms_across_float64_scales(mode, scale):
    reference = np.array([[0.0, 0.0, 4.0, 2.0, 0.3]])
    candidate = np.array([[1.2, 0.4, 3.0, 1.5, -0.2]])
    scaled_reference = reference.copy()
    scaled_candidate = candidate.copy()
    scaled_reference[:, :4] *= scale
    scaled_candidate[:, :4] *= scale
    metric = getattr(AssociationFunction, f"{mode}_batch_obb")

    np.testing.assert_allclose(
        metric(scaled_reference, scaled_candidate),
        metric(reference, candidate),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_obb_metrics_canonicalize_huge_finite_angles(mode):
    reference = np.array([[0.0, 0.0, 4.0, 2.0, 1e20]])
    candidate = np.array([[0.5, 0.2, 3.0, 1.5, 1e20]])
    canonical_reference = reference.copy()
    canonical_candidate = candidate.copy()
    canonical_reference[:, 4] = math.remainder(reference[0, 4], math.pi)
    canonical_candidate[:, 4] = math.remainder(candidate[0, 4], math.pi)
    metric = getattr(AssociationFunction, f"{mode}_batch_obb")

    np.testing.assert_array_equal(
        metric(reference, candidate),
        metric(canonical_reference, canonical_candidate),
    )


def test_obb_metrics_preserve_overlap_near_float64_limit():
    reference = np.array([[-0.9, 0.0, 1.79, 1.79, np.pi / 4.0]])
    candidate = np.array([[0.9, 0.0, 1.79, 1.79, np.pi / 4.0]])
    scaled_reference = reference.copy()
    scaled_candidate = candidate.copy()
    scaled_reference[:, :4] *= 1e308
    scaled_candidate[:, :4] *= 1e308

    expected_iou = AssociationFunction.iou_batch_obb(reference, candidate)
    np.testing.assert_allclose(
        AssociationFunction.iou_batch_obb(scaled_reference, scaled_candidate),
        expected_iou,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        AssociationFunction.hmiou_batch_obb(scaled_reference, scaled_candidate),
        expected_iou,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_obb_metrics_do_not_treat_distinct_subnanometric_swapped_boxes_as_equivalent(mode):
    reference = np.array([[0.0, 0.0, 2.0, 1.0, 0.0]])
    candidate = np.array([[0.0, 0.0, 1.0, 2.0, 0.0]])
    scaled_reference = reference.copy()
    scaled_candidate = candidate.copy()
    scaled_reference[:, 2:4] *= 1e-12
    scaled_candidate[:, 2:4] *= 1e-12
    metric = getattr(AssociationFunction, f"{mode}_batch_obb")

    baseline = metric(reference, candidate)
    scaled = metric(scaled_reference, scaled_candidate)

    np.testing.assert_allclose(scaled, baseline, rtol=1e-12, atol=1e-12)
    assert scaled.item() < 1.0


@pytest.mark.parametrize("mode", ("giou", "diou", "ciou"))
def test_obb_enclosure_metrics_are_invariant_to_rigid_rotation(mode):
    reference = np.array([[0.0, 0.0, 4.0, 1.0, 0.3]])
    candidate = np.array([[1.2, 0.4, 3.0, 2.0, -0.2]])
    rotation_angle = 0.73
    rotation = np.array(
        (
            (np.cos(rotation_angle), -np.sin(rotation_angle)),
            (np.sin(rotation_angle), np.cos(rotation_angle)),
        )
    )
    rotated_reference = reference.copy()
    rotated_candidate = candidate.copy()
    rotated_reference[:, :2] = rotated_reference[:, :2] @ rotation.T
    rotated_candidate[:, :2] = rotated_candidate[:, :2] @ rotation.T
    rotated_reference[:, 4] += rotation_angle
    rotated_candidate[:, 4] += rotation_angle
    metric = getattr(AssociationFunction, f"{mode}_batch_obb")

    np.testing.assert_allclose(
        metric(reference, candidate),
        metric(rotated_reference, rotated_candidate),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_obb_overlap_metrics_are_exactly_symmetric(mode):
    rng = np.random.default_rng(20260902)

    def random_boxes(count):
        return np.column_stack(
            (
                rng.uniform(-5.0, 5.0, count),
                rng.uniform(-5.0, 5.0, count),
                rng.uniform(2.0, 12.0, count),
                rng.uniform(2.0, 12.0, count),
                rng.uniform(-4.0 * np.pi, 4.0 * np.pi, count),
            )
        )

    reference = random_boxes(16)
    candidates = random_boxes(15)
    reference[0, :2] = 0.0
    candidates[0, :2] = 0.0
    metric = getattr(AssociationFunction, f"{mode}_batch_obb")

    np.testing.assert_array_equal(metric(reference, candidates), metric(candidates, reference).T)


def test_obb_ciou_adds_representation_invariant_aspect_ratio_penalty():
    horizontal = np.array([[20.0, 30.0, 12.0, 4.0, 0.2]], dtype=np.float32)
    square = np.array([[20.0, 30.0, 8.0, 8.0, 0.2]], dtype=np.float32)

    ciou = AssociationFunction.ciou_batch_obb(horizontal, square)[0, 0]
    diou = AssociationFunction.diou_batch_obb(horizontal, square)[0, 0]

    assert ciou < diou


def test_ciou_similarities_are_bounded_for_extreme_valid_boxes():
    aabb_reference = np.array([[0.0, 0.0, 1e6, 1.0]])
    aabb_candidate = np.array([[1e9, 0.0, 1e9 + 1.0, 1e6]])
    obb_reference = np.array([[0.0, 0.0, 1e6, 1.0, 0.4]])
    obb_candidate = np.array([[1e9, 0.0, 1.0, 1e6, -0.7]])

    aabb_ciou = AssociationFunction.ciou_batch(aabb_reference, aabb_candidate)
    obb_ciou = AssociationFunction.ciou_batch_obb(obb_reference, obb_candidate)

    assert np.all((0.0 <= aabb_ciou) & (aabb_ciou <= 1.0))
    assert np.all((0.0 <= obb_ciou) & (obb_ciou <= 1.0))


def test_obb_hmiou_modulates_rotated_iou_by_global_vertical_overlap():
    reference = np.array([[0.0, 0.0, 10.0, 10.0, 0.0]], dtype=np.float32)
    horizontal_offset = np.array([[5.0, 0.0, 10.0, 10.0, 0.0]], dtype=np.float32)
    vertical_offset = np.array([[0.0, 5.0, 10.0, 10.0, 0.0]], dtype=np.float32)

    horizontal_hmiou = AssociationFunction.hmiou_batch_obb(reference, horizontal_offset)[0, 0]
    vertical_hmiou = AssociationFunction.hmiou_batch_obb(reference, vertical_offset)[0, 0]

    np.testing.assert_allclose(horizontal_hmiou, 1.0 / 3.0, atol=1e-6)
    np.testing.assert_allclose(vertical_hmiou, 1.0 / 9.0, atol=1e-6)


def test_obb_hmiou_uses_nonzero_global_y_overlap_for_rotated_boxes():
    reference = np.array([[0.0, 0.0, 4.0, 2.0, np.pi / 2.0]])
    vertically_offset = np.array([[0.0, 1.0, 4.0, 2.0, np.pi / 2.0]])

    similarity = AssociationFunction.hmiou_batch_obb(reference, vertically_offset)[0, 0]

    assert similarity == pytest.approx(0.36, abs=1e-12)
    assert similarity > 0.0


@pytest.mark.parametrize("mode", ("giou", "ciou", "hmiou"))
def test_new_obb_metrics_preserve_empty_pairwise_shape(mode):
    empty = np.empty((0, 5), dtype=np.float32)
    boxes = np.array([[20.0, 30.0, 12.0, 6.0, 0.4]], dtype=np.float32)
    metric = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func

    assert metric(empty, boxes).shape == (0, 1)
    assert metric(boxes, empty).shape == (1, 0)


def test_ciou_penalizes_aspect_ratio_more_than_diou():
    horizontal = np.array([[0.0, 2.0, 10.0, 8.0]], dtype=np.float32)
    vertical = np.array([[2.0, 0.0, 8.0, 10.0]], dtype=np.float32)

    ciou = AssociationFunction.ciou_batch(horizontal, vertical)[0, 0]
    diou = AssociationFunction.diou_batch(horizontal, vertical)[0, 0]

    assert ciou < diou


@pytest.mark.parametrize(
    ("mode", "oriented_mode"),
    (
        ("iou", "iou_obb"),
        ("giou", "giou_obb"),
        ("diou", "diou_obb"),
        ("ciou", "ciou_obb"),
        ("hmiou", "hmiou_obb"),
        ("centroid", "centroid_obb"),
    ),
)
def test_obb_detection_layout_routes_supported_association_modes(mode, oriented_mode):
    assert OBB_DETECTIONS.association_mode_name(mode) == oriented_mode


def test_boost_association_uses_oriented_overlap_for_ambiguous_enclosing_boxes():
    detections = np.array(
        [[50, 50, 60, 8, np.pi / 4, 0.95], [50, 50, 60, 8, -np.pi / 4, 0.95]],
        dtype=np.float32,
    )
    trackers = detections[[1, 0]].copy()
    oriented_iou = AssociationFunction.iou_batch_obb(detections[:, :5], trackers[:, :5])

    matches, unmatched_dets, unmatched_tracks, _ = boost_associate(
        detections,
        trackers,
        0.5,
        track_confidence=trackers[:, 5],
        detection_confidence=detections[:, 5],
        lambda_iou=1.0,
        lambda_mhd=0.0,
        lambda_shape=0.0,
        geometry_matrix=oriented_iou,
        shape_matrix=shape_similarity_obb(detections, trackers),
    )

    np.testing.assert_array_equal(matches, np.array([[0, 1], [1, 0]]))
    assert unmatched_dets.size == 0
    assert unmatched_tracks.size == 0


def test_obb_boost_shape_and_buffering_are_representation_and_confidence_aware():
    detection = np.array([[45, 50, 8, 24, np.pi / 2, 0.95]], dtype=np.float32)
    equivalent_track = np.array([[60, 50, 24, 8, 0.0, 0.2]], dtype=np.float32)
    high_conf_track = equivalent_track.copy()
    high_conf_track[:, 5] = 0.95

    np.testing.assert_allclose(shape_similarity_obb(detection, equivalent_track), 1.0, atol=1e-6)
    assert (
        soft_biou_batch_obb(detection, equivalent_track)[0, 0] > soft_biou_batch_obb(detection, high_conf_track)[0, 0]
    )
