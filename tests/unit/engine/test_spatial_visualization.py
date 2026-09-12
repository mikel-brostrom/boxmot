"""Project true camera-space cuboids without leaking invalid pixels into drawing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.engine.tracking import spatial_visualization
from boxmot.engine.tracking.sinks import _track_color
from boxmot.engine.tracking.spatial_visualization import draw_spatial_tracks
from boxmot.structures import Boxes3D, CameraModel, Tracks3D

_SHAPE = (240, 320)
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def _camera(translation: tuple[float, float, float] = (0, 0, 0), *, pose: torch.Tensor | None = None) -> CameraModel:
    """Use a pinhole camera with an independently adjustable P2 translation column."""
    return CameraModel(
        torch.tensor(
            [[80, 0, 160, translation[0]], [0, 80, 120, translation[1]], [0, 0, 1, translation[2]]], dtype=torch.float32
        ),
        _SHAPE,
        camera_to_world=pose,
    )


def _tracks(boxes: list[list[float]], *, ids: list[int] | None = None, classes: list[int] | None = None) -> Tracks3D:
    """Construct canonical camera-space boxes without any image observations."""
    return Tracks3D(
        geometry=Boxes3D(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7)),
        track_ids=torch.tensor(list(range(len(boxes))) if ids is None else ids, dtype=torch.int64),
        scores=torch.full((len(boxes),), 0.95, dtype=torch.float32),
        class_ids=torch.tensor([1] * len(boxes) if classes is None else classes, dtype=torch.int64),
        detection_indices=torch.full((len(boxes),), -1, dtype=torch.int64),
        sample_id="val:0002:0",
    )


@pytest.fixture
def drawing(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Capture OpenCV boundaries, rejecting any unbounded integer coordinates."""
    lines, labels = [], []

    def line(
        image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], *args: object
    ) -> np.ndarray:
        for x, y in (start, end):
            assert isinstance(x, int) and isinstance(y, int)
            assert 0 <= x < image.shape[1] and 0 <= y < image.shape[0]
        lines.append((start, end, color))
        return image

    def label(
        image: np.ndarray,
        text: str,
        anchor: tuple[int, int],
        font: int,
        scale: float,
        color: tuple[int, int, int],
        *args: object,
    ) -> np.ndarray:
        labels.append((text, anchor, color))
        return image

    monkeypatch.setattr(spatial_visualization.cv2, "line", line)
    monkeypatch.setattr(spatial_visualization.cv2, "putText", label)
    return SimpleNamespace(lines=lines, labels=labels)


@pytest.mark.parametrize("yaw", (0.0, 0.37))
def test_eight_corners_and_twelve_edges_use_full_projection_translation(drawing: SimpleNamespace, yaw: float) -> None:
    tracks = _tracks([[0, 1, 10, yaw, 4, 2, 2]], ids=[7])
    camera = _camera((80, -40, 1.0))
    # Construct corners independently from the helper under test: local bottom
    # and top faces, then the conventional y-axis rotation and bottom center.
    local = np.array(
        [[2, 0, 1], [2, 0, -1], [-2, 0, -1], [-2, 0, 1], [2, -2, 1], [2, -2, -1], [-2, -2, -1], [-2, -2, 1]]
    )
    angle = float(tracks.geometry.values[0, 3])
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    corners = local @ rotation.T + np.array([0, 1, 10])
    homogeneous = np.column_stack((corners, np.ones(8))) @ camera.projection.numpy().T
    pixels = [tuple(map(int, point)) for point in np.rint(homogeneous[:, :2] / homogeneous[:, 2:])]
    draw_spatial_tracks(np.zeros((*_SHAPE, 3), np.uint8), tracks, camera, image_track_ids=frozenset({7}))
    assert len(drawing.lines) == 12
    assert len(set(pixels)) == 8
    assert [(start, end) for start, end, _ in drawing.lines] == [
        (pixels[first], pixels[second]) for first, second in _EDGES
    ]
    assert all(color == _track_color(7) for _, _, color in drawing.lines)
    assert not drawing.labels


@pytest.mark.parametrize("center", ((0.0, 0.05), (0.05, 0.1)))
def test_edges_crossing_near_plane_clip_before_division(drawing: SimpleNamespace, center: tuple[float, float]) -> None:
    tracks = _tracks([[*center, 0.15, 0, 0.1, 0.3, 0.1]])
    with np.errstate(all="raise"):
        draw_spatial_tracks(np.zeros((*_SHAPE, 3), np.uint8), tracks, _camera())
    # The rear four edges lie at zero depth; the front face and four crossing
    # edges survive. One fixture also places a corner exactly at the camera origin.
    assert len(drawing.lines) == 8
    assert len(drawing.labels) == 1


@pytest.mark.parametrize(
    "box", ([0, 1, -10, 0, 4, 2, 2], [1000, 1, 10, 0, 4, 2, 2], [1e30, 1, 10, 0, 4, 2, 2], [0, -1000, 10, 0, 4, 2, 2])
)
def test_behind_camera_or_offscreen_boxes_do_not_draw_spurious_edges(
    drawing: SimpleNamespace, box: list[float]
) -> None:
    image = np.full((*_SHAPE, 3), 37, np.uint8)
    with np.errstate(all="raise"):
        assert draw_spatial_tracks(image, _tracks([box]), _camera()) is image
    assert not drawing.lines and not drawing.labels
    assert np.all(image == 37)


def test_enormous_projected_coordinates_are_clipped_in_float(drawing: SimpleNamespace) -> None:
    with np.errstate(all="raise"):
        draw_spatial_tracks(np.zeros((*_SHAPE, 3), np.uint8), _tracks([[0, 1, 10, 0, 1e30, 2, 2]]), _camera())
    assert len(drawing.lines) == 4
    for start, end, _ in drawing.lines:
        assert {start[0], end[0]} == {0, _SHAPE[1] - 1}
        assert start[1] == end[1]


def test_camera_to_world_is_not_applied_to_camera_space_tracks(drawing: SimpleNamespace) -> None:
    tracks = _tracks([[0, 1, 10, 0.37, 4, 2, 2]])
    image = np.zeros((*_SHAPE, 3), np.uint8)
    draw_spatial_tracks(image, tracks, _camera())
    original_lines, original_labels = list(drawing.lines), list(drawing.labels)
    drawing.lines.clear()
    drawing.labels.clear()
    pose = torch.tensor([[0, -1, 0, 100], [1, 0, 0, 50], [0, 0, 1, 7], [0, 0, 0, 1]], dtype=torch.float32)
    draw_spatial_tracks(image, tracks, _camera(pose=pose))
    assert drawing.lines == original_lines
    assert drawing.labels == original_labels


def test_spatial_only_labels_and_colors_follow_identity_after_reordering(drawing: SimpleNamespace) -> None:
    tracks = _tracks([[-2, 1, 10, 0, 2, 2, 2], [2, 1, 10, 0, 2, 2, 2]], ids=[7, 11], classes=[1, 2])
    options = {"class_names": {1: "car", 2: "pedestrian"}, "image_track_ids": frozenset({7})}
    draw_spatial_tracks(np.zeros((*_SHAPE, 3), np.uint8), tracks, _camera(), **options)
    assert [label for label, _, _ in drawing.labels] == ["pedestrian #11 3D"]
    assert [color for _, _, color in drawing.lines] == [_track_color(7)] * 12 + [_track_color(11)] * 12
    drawing.lines.clear()
    drawing.labels.clear()
    draw_spatial_tracks(np.zeros((*_SHAPE, 3), np.uint8), tracks.select(torch.tensor([1, 0])), _camera(), **options)
    assert [label for label, _, _ in drawing.labels] == ["pedestrian #11 3D"]
    assert [color for _, _, color in drawing.lines] == [_track_color(11)] * 12 + [_track_color(7)] * 12


def test_empty_tracks_leave_the_same_image_untouched(drawing: SimpleNamespace) -> None:
    image = np.full((*_SHAPE, 3), 37, np.uint8)
    assert draw_spatial_tracks(image, _tracks([]), _camera()) is image
    assert np.all(image == 37)
    assert not drawing.lines and not drawing.labels


def test_real_drawing_overlays_wireframe_without_filling_existing_mask_pixels() -> None:
    image = np.full((*_SHAPE, 3), (21, 32, 43), np.uint8)
    original_image = image.copy()
    tracks = _tracks([[0, 1, 10, 0, 4, 2, 2]], ids=[7])
    original_boxes = tracks.geometry.values.clone()
    camera = _camera()
    original_projection = camera.projection.clone()
    assert draw_spatial_tracks(image, tracks, camera, image_track_ids=frozenset({7})) is image
    assert np.any(image != original_image)
    np.testing.assert_array_equal(image[120, 160], original_image[120, 160])
    np.testing.assert_array_equal(image[0], original_image[0])
    torch.testing.assert_close(tracks.geometry.values, original_boxes)
    torch.testing.assert_close(camera.projection, original_projection)
