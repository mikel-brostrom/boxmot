from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.create_market1501_pav_metadata import (
    _normalized_record,
    _pose_only_has_multiple_people,
    pose_mask_agreement,
    select_primary_pose,
    select_primary_segmentation,
)


def _boxes(boxes, confidences, classes=None):
    if classes is None:
        classes = [0] * len(boxes)
    return SimpleNamespace(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        conf=torch.tensor(confidences, dtype=torch.float32),
        cls=torch.tensor(classes, dtype=torch.float32),
    )


def test_pose_selector_prefers_the_centered_market_subject():
    keypoints = torch.zeros(2, 17, 2)
    keypoints[0, :, 0] = 6
    keypoints[0, :, 1] = 10
    keypoints[1, :, 0] = 50
    keypoints[1, :, 1] = 10
    result = SimpleNamespace(
        boxes=_boxes(((2, 1, 10, 19), (45, 1, 55, 19)), (0.8, 0.99)),
        keypoints=SimpleNamespace(
            xy=keypoints,
            conf=torch.full((2, 17), 0.9),
        ),
    )

    selected = select_primary_pose(result, (20, 20))

    assert selected is not None
    np.testing.assert_array_equal(
        selected.box,
        np.array((2, 1, 10, 19), dtype=np.float32),
    )
    assert selected.keypoints.shape == (17, 3)
    assert selected.confidence == pytest.approx(0.8)
    assert selected.candidate_count == 2
    assert selected.score_margin is not None
    assert selected.area_fraction == pytest.approx(0.36)


def test_segmentation_selector_returns_pose_matched_person_and_nearby_bag():
    masks = torch.zeros(3, 20, 20)
    masks[0, 2:18, 4:11] = 1
    masks[1, 2:18, 14:19] = 1
    masks[2, 8:14, 10:14] = 1
    result = SimpleNamespace(
        boxes=_boxes(
            ((4, 2, 11, 18), (14, 2, 19, 18), (10, 8, 14, 14)),
            (0.8, 0.99, 0.9),
            (0, 0, 24),
        ),
        masks=SimpleNamespace(data=masks),
    )

    selected = select_primary_segmentation(
        result,
        (20, 20),
        pose_box=np.array((4, 2, 11, 18), dtype=np.float32),
        person_class=0,
        bag_classes=(24,),
        bag_proximity=0.1,
        mask_threshold=0.5,
    )

    assert selected.person_mask is not None
    assert selected.person_mask[5, 5]
    assert not selected.person_mask[5, 17]
    assert selected.bag_mask[10, 12]
    assert selected.person_candidate_count == 2
    assert selected.bag_candidate_count == 1
    assert selected.matched_iou == pytest.approx(1.0)


def test_pose_selector_reports_ambiguous_similar_primary_candidates():
    keypoints = torch.zeros(2, 17, 2)
    keypoints[0, :, 0] = 9
    keypoints[0, :, 1] = 10
    keypoints[1, :, 0] = 11
    keypoints[1, :, 1] = 10
    result = SimpleNamespace(
        boxes=_boxes(
            ((4, 1, 14, 19), (6, 1, 16, 19)),
            (0.9, 0.9),
        ),
        keypoints=SimpleNamespace(
            xy=keypoints,
            conf=torch.full((2, 17), 0.9),
        ),
    )

    selected = select_primary_pose(result, (20, 20))

    assert selected is not None
    assert selected.candidate_count == 2
    assert selected.score_margin is not None
    assert selected.score_margin < 1.25


def test_segmentation_selector_uses_overlap_before_confidence():
    masks = torch.zeros(2, 20, 20)
    masks[0, 2:18, 3:11] = 1
    masks[1, 2:18, 12:19] = 1
    result = SimpleNamespace(
        boxes=_boxes(
            ((3, 2, 11, 18), (12, 2, 19, 18)),
            (0.6, 0.99),
            (0, 0),
        ),
        masks=SimpleNamespace(data=masks),
    )

    selected = select_primary_segmentation(
        result,
        (20, 20),
        pose_box=np.array((3, 2, 11, 18), dtype=np.float32),
        person_class=0,
        bag_classes=(),
        bag_proximity=0.0,
        mask_threshold=0.5,
    )

    assert selected.person_mask is not None
    assert selected.person_mask[5, 5]
    assert not selected.person_mask[5, 15]
    assert selected.matched_confidence == pytest.approx(0.6)
    assert selected.matched_iou == pytest.approx(1.0)


def test_pose_mask_agreement_exposes_mismatched_people():
    points = np.zeros((17, 3), dtype=np.float32)
    points[:12, :2] = (5, 5)
    points[:12, 2] = 0.9
    points[12:, :2] = (15, 15)
    points[12:, 2] = 0.9
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:10, 2:10] = True

    agreement = pose_mask_agreement(points, mask, 0.5)

    assert agreement == pytest.approx(12 / 17)


def test_pose_mask_agreement_excludes_out_of_bounds_keypoints():
    points = np.zeros((17, 3), dtype=np.float32)
    points[:4] = (
        (5, 5, 0.9),
        (6, 5, 0.9),
        (15, 15, 0.9),
        (-1, 5, 0.9),
    )
    mask = np.zeros((20, 20), dtype=bool)
    mask[5, 0] = True
    mask[5, 5:7] = True

    agreement = pose_mask_agreement(points, mask, 0.5)

    assert agreement == pytest.approx(2 / 3)


def test_pose_mask_agreement_returns_zero_without_in_bounds_keypoints():
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 0] = -1
    points[:, 1] = 5
    points[:, 2] = 0.9

    agreement = pose_mask_agreement(
        points,
        np.ones((20, 20), dtype=bool),
        0.5,
    )

    assert agreement == 0.0


@pytest.mark.parametrize(
    ("pose_candidates", "segmentation_candidates", "expected"),
    (
        (1, 0, False),
        (1, 1, False),
        (2, 0, True),
        (1, 2, True),
    ),
)
def test_pose_only_rejects_multiple_people_from_either_detector(
    pose_candidates,
    segmentation_candidates,
    expected,
):
    assert (
        _pose_only_has_multiple_people(
            pose_candidates,
            segmentation_candidates,
        )
        is expected
    )


def test_normalized_pose_record_is_resolution_independent():
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 0] = 10
    points[:, 1] = 20
    points[:, 2] = 0.9

    record = _normalized_record(
        points,
        np.array((0, 0, 20, 40), dtype=np.float32),
        0.8,
        (40, 20),
    )

    assert record["keypoints"][0] == pytest.approx([0.5, 0.5, 0.9])
    assert record["bbox"] == [0.0, 0.0, 1.0, 1.0]
