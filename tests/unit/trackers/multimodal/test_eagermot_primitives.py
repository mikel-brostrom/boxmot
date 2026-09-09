"""Numerical contracts for EagerMOT's 3D motion, projection, and association."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import EagerMot
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D
from boxmot.trackers.multimodal.eagermot.association import greedy_association, similarity_3d
from boxmot.trackers.multimodal.eagermot.geometry import (
    boxes3d_corners,
    iou2d_matrix,
    iou3d_matrix,
    project_box3d,
    transform_boxes3d,
    yaw_difference,
)
from boxmot.trackers.multimodal.eagermot.motion import Kalman3D


def _box() -> np.ndarray:
    """An upright 4x2x2 object in front of a camera, using a bottom center."""
    return np.array([0.0, 2.0, 10.0, 0.0, 4.0, 2.0, 2.0])


@pytest.mark.parametrize("angular", [False, True])
def test_kalman_uses_source_initial_uncertainty_and_learns_velocity(angular: bool) -> None:
    """The released source's large velocity prior learns displacement in one step."""
    model = Kalman3D(_box(), is_angular=angular)
    assert len(model.state) == (11 if angular else 10)
    np.testing.assert_array_equal(np.diag(model.covariance)[:7], 10)
    np.testing.assert_array_equal(np.diag(model.covariance)[7:], 10000)
    model.predict()
    assert model.covariance[0, 0] == 10011
    assert model.covariance[0, 7] == 10000
    observation = _box()
    observation[0] = 4
    model.update(observation)
    assert model.box[0] == pytest.approx(4 * 10011 / 10011.01)
    assert model.state[7] == pytest.approx(4 * 10000 / 10011.01)
    last_box, velocity = model.box, model.state[7:10].copy()
    prediction = model.predict()
    np.testing.assert_allclose(prediction[:3], last_box[:3] + velocity)
    np.testing.assert_allclose(model.covariance, model.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(model.covariance).min() > 0


def test_kalman_yaw_alignment_handles_full_turns_without_mutating_detections() -> None:
    """Pi-equivalent observations cannot pull an unwrapped state through a full turn."""
    box = _box()
    box[3] = 8 * np.pi + 0.1
    model = Kalman3D(box, is_angular=True)
    model.predict()
    observation = _box()
    observation[3] = np.pi - 0.1
    original = observation.copy()
    model.update(observation)
    assert abs(model.box[3] - (8 * np.pi - 0.1)) < 1e-5
    assert -0.21 < model.state[10] < -0.19
    np.testing.assert_array_equal(observation, original)
    returned_box = model.box
    returned_box[:] = 0
    assert model.box[4] == 4


def test_boxes_use_bottom_center_and_yaw_about_downward_y() -> None:
    """The source's bottom face lies at y, its top face at y-h."""
    first = _box()
    rotated = first.copy()
    rotated[3] = np.pi / 2
    corners = boxes3d_corners(np.stack((first, rotated)))
    np.testing.assert_allclose(corners[0, 0], [2, 2, 11])
    np.testing.assert_allclose(corners[0, 4], [2, 0, 11])
    np.testing.assert_allclose(corners[1, 0], [1, 2, 8])
    assert boxes3d_corners(np.empty((0, 7))).shape == (0, 8, 3)


def test_volumetric_iou_includes_height_and_oriented_ground_footprint() -> None:
    """Boxes with identical projections or centers can still have distinct 3D IoU."""
    base = _box()
    shifted_x, shifted_y, no_height, rotated = (base.copy() for _ in range(4))
    shifted_x[0] += 2
    shifted_y[1] += 1
    no_height[1] += 2
    rotated[3] = np.pi / 2
    others = np.stack((base, shifted_x, shifted_y, no_height, rotated))
    np.testing.assert_allclose(iou3d_matrix(base[None], others), [[1, 1 / 3, 1 / 3, 0, 1 / 3]], atol=1e-10)
    np.testing.assert_allclose(iou3d_matrix(others, base[None]).T, iou3d_matrix(base[None], others))
    assert iou3d_matrix(np.empty((0, 7)), others).shape == (0, 5)


def test_upright_pose_preserves_corners_and_relative_iou() -> None:
    """A common world-frame pose changes coordinates while preserving physical boxes."""
    angle = 0.4
    pose = np.array(
        [[np.cos(angle), 0, np.sin(angle), 100], [0, 1, 0, -4], [-np.sin(angle), 0, np.cos(angle), 200], [0, 0, 0, 1]]
    )
    original = np.stack((_box(), _box() + [1, 0, 0, 0.3, 0, 0, 0]))
    transformed = transform_boxes3d(original, pose)
    expected_corners = boxes3d_corners(original) @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(boxes3d_corners(transformed), expected_corners, atol=1e-12)
    np.testing.assert_allclose(transform_boxes3d(transformed, np.linalg.inv(pose)), original, atol=1e-12)
    np.testing.assert_allclose(iou3d_matrix(original, original), iou3d_matrix(transformed, transformed), atol=1e-4)
    with pytest.raises(ValueError, match="y axis"):
        transform_boxes3d(original, np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))


def test_projection_rounds_pixels_and_rejects_behind_camera_boxes() -> None:
    """The source calibration convention projects x/z and y/z into image xyxy."""
    projection = np.array([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=float)
    np.testing.assert_array_equal(project_box3d(_box(), projection, (100, 200)), [78, 50, 122, 72])
    behind = _box()
    behind[2] = -10
    assert project_box3d(behind, projection, (100, 200)) is None
    outside = _box()
    outside[0] = 100
    assert project_box3d(outside, projection, (100, 200)) is None


def test_projection_handles_camera_extrinsics_in_the_combined_matrix() -> None:
    """Tracking-frame boxes project consistently after a translated camera pose."""
    projection = np.array([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=float)
    pose = np.eye(4)
    pose[:3, 3] = [5, 0, 2]
    moved = transform_boxes3d(_box()[None], pose)[0]
    np.testing.assert_array_equal(
        project_box3d(moved, projection @ np.linalg.inv(pose), (100, 200)),
        project_box3d(_box(), projection, (100, 200)),
    )


def test_camera_accepted_float32_yaw_poses_work_through_tracker_updates() -> None:
    """Canonical pose precision must not trigger a stricter check inside tracking."""
    tracker = EagerMot(distance_threshold=0.01)
    projection = torch.tensor([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=torch.float32)
    world_box = np.array([[2.0, 1.0, 20.0, 0.1, 4.0, 2.0, 2.0]])
    for frame, angle in enumerate((0.23, 0.47)):
        pose = np.array(
            [
                [np.cos(angle), 0, np.sin(angle), frame],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), frame * 0.2],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        # Rounded calibration estimates may be nearly, rather than exactly,
        # orthonormal. CameraModel accepts this error under its 1e-5 contract.
        pose[0, 2] += np.float32(3e-6)
        camera = CameraModel(projection, (100, 200), torch.from_numpy(pose))
        camera_boxes = transform_boxes3d(world_box, np.linalg.inv(pose))
        image_box = project_box3d(camera_boxes[0], projection.numpy(), camera.image_size)
        assert image_box is not None
        sample_id = f"rotating-camera/{frame}"
        image = Detections(
            geometry=Boxes(torch.tensor(image_box[None], dtype=torch.float32)),
            scores=torch.tensor([0.95]),
            class_ids=torch.tensor([0]),
            sample_id=sample_id,
        )
        spatial = Detections3D(
            geometry=Boxes3D(torch.tensor(camera_boxes, dtype=torch.float32)),
            scores=torch.tensor([0.95]),
            class_ids=torch.tensor([0]),
            sample_id=sample_id,
        )
        result = tracker.update(image, detections_3d=spatial, camera=camera)
        assert result.image_tracks.track_ids.tolist() == result.spatial_tracks.track_ids.tolist() == [0]
        np.testing.assert_allclose(result.spatial_tracks.geometry.values.numpy(), camera_boxes, atol=3e-5, rtol=0)
        np.testing.assert_allclose(tracker._tracks[0].motion.box, world_box[0], atol=3e-5, rtol=0)


def test_projection_preserves_depth_translation_and_forward_corner_visibility() -> None:
    """A full calibration matrix controls depth before visibility and pixel division."""
    projection = np.array([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 2]], dtype=float)
    np.testing.assert_array_equal(project_box3d(_box(), projection, (100, 200)), [64, 41, 100, 59])
    crossing_camera = _box()
    crossing_camera[2] = -2
    # Four corners remain at positive projected depth; the visible face is valid.
    assert project_box3d(crossing_camera, projection, (100, 200)) is not None
    crossing_camera[2] = -3
    assert project_box3d(crossing_camera, projection, (100, 200)) is None


def test_source_distance_variants_keep_their_actual_coordinate_dimensions() -> None:
    """Despite its name, dist_2d_full includes y and object dimensions."""
    base = _box()
    changed = base.copy()
    changed[1] += 3
    changed[4] += 4
    changed[3] = np.pi / 2
    assert similarity_3d(base[None], changed[None], "dist_2d")[0, 0] == 0
    assert similarity_3d(base[None], changed[None], "dist_2d_dims")[0, 0] == -5
    assert similarity_3d(base[None], changed[None], "dist_2d_full")[0, 0] == pytest.approx(-10)
    changed[3] = np.pi
    assert similarity_3d(base[None], changed[None], "dist_2d_full")[0, 0] == pytest.approx(-5)
    assert abs(float(yaw_difference(100 * np.pi, 0.1))) == pytest.approx(0.1)
    np.testing.assert_array_equal(similarity_3d(base[None], base[None], "iou_3d"), [[1]])


def test_greedy_assignment_keeps_source_order_and_does_not_become_hungarian() -> None:
    """The globally strongest pair wins even when another assignment sums higher."""
    matches, unused_first, unused_second = greedy_association(np.array([[0.9, 0.8], [0.85, 0.1]]), 0.2)
    np.testing.assert_array_equal(matches, [[0, 0]])
    np.testing.assert_array_equal(unused_first, [1])
    np.testing.assert_array_equal(unused_second, [1])
    tied, _, _ = greedy_association(np.ones((2, 2)), 0.2)
    np.testing.assert_array_equal(tied, [[0, 0], [1, 1]])


def test_association_respects_class_gates_and_negative_distance_thresholds() -> None:
    """Disallowed high-scoring edges cannot consume a valid same-class match."""
    similarities = np.array([[-1.0, -0.1], [-0.2, -2.0]])
    matches, _, _ = greedy_association(similarities, -2.0, allowed=np.eye(2, dtype=bool))
    np.testing.assert_array_equal(matches, [[0, 0], [1, 1]])
    matches, rows, cols = greedy_association(np.empty((0, 3)), -2.0)
    assert matches.shape == (0, 2)
    assert not len(rows)
    np.testing.assert_array_equal(cols, [0, 1, 2])


def test_image_iou_treats_invalid_projections_as_unobservable() -> None:
    """Missing camera projections contribute no overlap to fusion or recovery."""
    detections = np.array([[0, 0, 10, 10], [np.nan] * 4, [0, 0, 0, 0]])
    projections = np.array([[5, 0, 15, 10]])
    np.testing.assert_allclose(iou2d_matrix(detections, projections), [[1 / 3], [0], [0]])


@pytest.mark.parametrize("dimensions", [[0, 2, 2], [4, -2, 2], [4, 2, np.nan]])
def test_invalid_3d_dimensions_fail_before_motion_or_geometry(dimensions: list[float]) -> None:
    """A malformed cuboid must not enter the Kalman state or volumetric geometry."""
    box = _box()
    box[4:] = dimensions
    with pytest.raises(ValueError, match="positive"):
        Kalman3D(box)
    with pytest.raises(ValueError, match="positive"):
        boxes3d_corners(box[None])
