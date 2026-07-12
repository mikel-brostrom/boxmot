from __future__ import annotations

import numpy as np

from boxmot.trackers.common.association import (
    AssociationFunction,
    AssociationStage,
    detection_track_iou_assignment,
    detection_track_tuple_to_association_result,
    iou_distance,
    run_association_stage,
)
from boxmot.trackers.common.association.boost import associate as boost_associate
from boxmot.trackers.common.association.boost import iou_batch as boost_iou_batch
from boxmot.trackers.common.association.hybrid import iou_batch as hybrid_iou_batch
from boxmot.trackers.common.detections import OBB_DETECTIONS


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


def test_detection_track_iou_assignment_returns_canonical_orientation():
    ious = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.1, 0.2],
        ],
        dtype=np.float32,
    )

    result = detection_track_iou_assignment(
        ious,
        threshold=0.5,
        assignment_solver=lambda cost: np.array([[0, 1], [1, 0], [2, 1]]),
    )

    np.testing.assert_array_equal(result.matches, np.array([[1, 0], [0, 1]]))
    np.testing.assert_array_equal(result.unmatched_tracks, np.empty((0,), dtype=int))
    np.testing.assert_array_equal(result.unmatched_dets, np.array([2]))
    np.testing.assert_allclose(result.cost_matrix, 1.0 - ious.T)


def test_detection_track_tuple_to_association_result_flips_match_columns():
    result = detection_track_tuple_to_association_result(
        (
            np.array([[0, 2], [1, 3]], dtype=int),
            np.array([4], dtype=int),
            np.array([5], dtype=int),
        )
    )

    np.testing.assert_array_equal(result.matches, np.array([[2, 0], [3, 1]]))
    np.testing.assert_array_equal(result.unmatched_tracks, np.array([5]))
    np.testing.assert_array_equal(result.unmatched_dets, np.array([4]))


def test_algorithm_specific_association_modules_live_under_common():
    detections = np.array([[0, 0, 10, 10, 0.95]], dtype=np.float32)
    trackers = np.array([[0, 0, 10, 10, 0.90]], dtype=np.float32)

    np.testing.assert_allclose(boost_iou_batch(detections, trackers), np.array([[1.0]]))
    np.testing.assert_allclose(hybrid_iou_batch(detections, trackers), np.array([[1.0]]))

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


def test_obb_detection_layout_routes_diou_to_rotated_diou():
    assert OBB_DETECTIONS.association_mode_name("diou") == "diou_obb"
