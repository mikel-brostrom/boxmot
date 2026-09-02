from __future__ import annotations

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


def test_ciou_penalizes_aspect_ratio_more_than_diou():
    horizontal = np.array([[0.0, 2.0, 10.0, 8.0]], dtype=np.float32)
    vertical = np.array([[2.0, 0.0, 8.0, 10.0]], dtype=np.float32)

    ciou = AssociationFunction.ciou_batch(horizontal, vertical)[0, 0]
    diou = AssociationFunction.diou_batch(horizontal, vertical)[0, 0]

    assert ciou < diou


@pytest.mark.parametrize(
    ("mode", "oriented_mode"),
    (("iou", "iou_obb"), ("diou", "diou_obb"), ("centroid", "centroid_obb")),
)
def test_obb_detection_layout_routes_supported_association_modes(mode, oriented_mode):
    assert OBB_DETECTIONS.association_mode_name(mode) == oriented_mode


@pytest.mark.parametrize("mode", ("giou", "ciou", "hmiou"))
def test_obb_detection_layout_rejects_unimplemented_geometry_metrics(mode):
    with pytest.raises(ValueError, match="no oriented-box implementation"):
        OBB_DETECTIONS.association_mode_name(mode)


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
