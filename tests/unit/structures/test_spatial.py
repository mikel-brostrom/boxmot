"""Validation and identity contracts for independent 2D and 3D observations."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections3D, MultimodalTracks, Tracks, Tracks3D


def _boxes() -> Boxes3D:
    """Include an off-camera object without inventing image-space geometry."""
    return Boxes3D(torch.tensor([[0.0, 1.5, 10.0, 0.0, 4.0, 2.0, 1.5], [1.0, 1.5, -5.0, 8.0, 3.0, 2.0, 1.0]]))


def _camera() -> CameraModel:
    return CameraModel(
        torch.tensor([[100.0, 0.0, 32.0, 0.0], [0.0, 100.0, 24.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        (48, 64),
    )


def _spatial_tracks() -> Tracks3D:
    return Tracks3D(
        geometry=_boxes(),
        track_ids=torch.tensor([11, 12]),
        scores=torch.tensor([0.9, 0.8]),
        class_ids=torch.tensor([3, 4]),
        detection_indices=torch.tensor([0, -1]),
        sample_id="camera:0",
    )


def _image_tracks() -> Tracks:
    return Tracks(
        geometry=Boxes(torch.tensor([[10.0, 10.0, 30.0, 40.0]])),
        track_ids=torch.tensor([11]),
        scores=torch.tensor([0.9]),
        class_ids=torch.tensor([3]),
        detection_indices=torch.tensor([2]),
        sample_id="camera:0",
    )


def test_spatial_detections_are_frozen_independent_rows_with_preserved_metadata() -> None:
    boxes = _boxes()
    detections = Detections3D(boxes, torch.tensor([0.9, 0.8]), torch.tensor([3, 4]), "camera:0")
    reordered = detections.select(torch.tensor([1, 0]))
    empty = detections.select(torch.tensor([False, False]))

    assert detections.geometry is boxes
    assert not hasattr(detections, "__dict__")
    with pytest.raises(FrozenInstanceError):
        detections.sample_id = "other"  # type: ignore[misc]
    torch.testing.assert_close(reordered.geometry.values, boxes.values.flip(0))
    assert reordered.class_ids.tolist() == [4, 3]
    assert reordered.sample_id == detections.sample_id
    assert empty.geometry.values.shape == (0, 7)
    assert empty.scores.shape == (0,)
    assert boxes.values[1, 2] < 0  # Being behind the camera does not invalidate 3D geometry.


@pytest.mark.parametrize("column", (4, 5, 6))
def test_spatial_boxes_reject_nonpositive_dimensions_even_after_tensor_mutation(column: int) -> None:
    boxes = _boxes()
    boxes.values[0, column] = 0.0
    with pytest.raises(ValueError, match="positive"):
        boxes.validate()


@pytest.mark.parametrize(
    "value,error,match",
    (
        (torch.ones((1, 6)), ValueError, "shape"),
        (torch.ones((1, 7), dtype=torch.float64), TypeError, "dtype"),
        (torch.ones((1, 7), device="meta"), ValueError, "CPU"),
        (torch.ones((7, 2)).T, ValueError, "contiguous"),
        (torch.full((1, 7), float("nan")), ValueError, "finite"),
    ),
)
def test_spatial_boxes_reject_noncanonical_values(value: torch.Tensor, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        Boxes3D(value)


def test_spatial_metadata_and_selection_are_validated() -> None:
    with pytest.raises(ValueError, match="aligned"):
        Detections3D(_boxes(), torch.tensor([0.9]), torch.tensor([3, 4]), "sample")
    with pytest.raises(ValueError, match="range"):
        Detections3D(_boxes(), torch.tensor([1.1, 0.9]), torch.tensor([3, 4]), "sample")
    with pytest.raises(ValueError, match="non-negative"):
        Detections3D(_boxes(), torch.tensor([0.9, 0.9]), torch.tensor([3, -1]), "sample")
    with pytest.raises(IndexError):
        _boxes().select(torch.tensor([2]))
    with pytest.raises(ValueError, match="unique"):
        replace(_spatial_tracks(), track_ids=torch.tensor([11, 11]))
    with pytest.raises(ValueError, match="unmatched"):
        replace(_spatial_tracks(), detection_indices=torch.tensor([-2, -1]))
    assert _spatial_tracks().select(torch.tensor([1])).detection_indices.tolist() == [-1]


def test_camera_supports_general_rigid_poses_and_rejects_nonrigid_transforms() -> None:
    camera = _camera()
    pose = torch.tensor([[0.0, 0.0, 1.0, 4.0], [0.0, 1.0, 0.0, 2.0], [-1.0, 0.0, 0.0, -3.0], [0.0, 0.0, 0.0, 1.0]])
    moved = replace(camera, camera_to_world=pose)
    assert moved.projection is camera.projection
    assert moved.camera_to_world is pose
    assert camera.camera_to_world is None

    roll = torch.tensor([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert replace(camera, camera_to_world=roll).camera_to_world is roll
    pitch = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert replace(camera, camera_to_world=pitch).camera_to_world is pitch
    with pytest.raises(ValueError, match="reflection"):
        replace(camera, camera_to_world=torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])))
    with pytest.raises(ValueError, match="orthonormal"):
        replace(camera, camera_to_world=torch.diag(torch.tensor([2.0, 1.0, 1.0, 1.0])))
    pose[3, 0] = 1.0
    with pytest.raises(ValueError, match="homogeneous"):
        moved.validate()


def test_camera_rejects_invalid_projection_and_image_dimensions() -> None:
    camera = _camera()
    with pytest.raises(ValueError, match="positive integers"):
        replace(camera, image_size=(0, 64))
    with pytest.raises(ValueError, match="shape"):
        replace(camera, projection=torch.eye(3))
    with pytest.raises(ValueError, match="nonsingular"):
        replace(camera, projection=torch.zeros((3, 4)))
    camera.projection[2, 3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        camera.validate()


def test_camera_preserves_general_full_projection_matrices() -> None:
    """Calibration may include a depth translation or a tilted image plane."""
    camera = _camera()
    projection = camera.projection.clone()
    projection[2, 3] = 0.0027
    translated = replace(camera, projection=projection)
    assert translated.projection is projection
    projection[2, 0] = 0.1
    projection[2, 1] = -0.05
    translated.validate()


def test_multimodal_outputs_keep_off_camera_tracks_without_fake_image_rows() -> None:
    output = MultimodalTracks(_image_tracks(), _spatial_tracks())
    assert len(output.image_tracks) == 1
    assert len(output.spatial_tracks) == len(output) == 2
    assert output.sample_id == "camera:0"
    empty_image = output.image_tracks.select(torch.tensor([False]))
    assert len(MultimodalTracks(empty_image, output.spatial_tracks)) == 2
    with pytest.raises(ValueError, match="same sample"):
        MultimodalTracks(replace(_image_tracks(), sample_id="other"), _spatial_tracks())
    with pytest.raises(ValueError, match="same class"):
        MultimodalTracks(replace(_image_tracks(), class_ids=torch.tensor([4])), _spatial_tracks())
