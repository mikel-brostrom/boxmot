"""Clean-install smoke tests for the public package and command interface."""

from __future__ import annotations

import numpy as np
import torch
from click.testing import CliRunner

from boxmot import EagerMot, create_tracker
from boxmot.engine.cli import boxmot as boxmot_cli
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, MultimodalTracks
from boxmot.trackers import TrackerSpec
from tests.ci.release_contract import EXPECTED_CLI_COMMANDS, check_release_contract, check_tracker_api


def test_eagermot_sensor_fusion_python_api_smoke() -> None:
    """Exercise the installed factory and canonical 2D/3D update without SDKs."""
    tracker = create_tracker(TrackerSpec("eagermot"))
    detections = Detections(
        geometry=Boxes(torch.tensor([[78, 39, 122, 62]], dtype=torch.float32)),
        scores=torch.tensor([0.95]),
        class_ids=torch.tensor([0]),
    )
    spatial = Detections3D(
        geometry=Boxes3D(torch.tensor([[0, 1, 10, 0, 4, 2, 2]], dtype=torch.float32)),
        scores=torch.tensor([0.9]),
        class_ids=torch.tensor([0]),
    )
    camera = CameraModel(
        projection=torch.tensor([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=torch.float32),
        image_size=(100, 200),
    )

    result = tracker.update(detections, detections_3d=spatial, camera=camera)

    assert isinstance(result, MultimodalTracks)
    assert len(result.image_tracks) == len(result.spatial_tracks) == 1
    torch.testing.assert_close(result.image_tracks.track_ids, result.spatial_tracks.track_ids)
    torch.testing.assert_close(result.spatial_tracks.geometry.values, spatial.geometry.values)


def test_eagermot_numpy_sensor_fusion_python_api_smoke() -> None:
    """Exercise public EagerMot construction and fusion with only NumPy inputs."""
    tracker = EagerMot()
    detections = np.array([[78, 39, 122, 62, 0.95, 0]])
    spatial = np.array([[0, 1, 10, 0, 4, 2, 2, 0.9, 0]])
    camera = np.array([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=float)

    image_tracks, spatial_tracks = tracker.update(
        detections, detections_3d=spatial, camera=camera, image_size=(100, 200)
    )

    assert image_tracks.shape == (1, 8)
    assert spatial_tracks.shape == (1, 11)
    assert image_tracks.dtype == spatial_tracks.dtype == np.float64
    assert image_tracks.flags.c_contiguous and spatial_tracks.flags.c_contiguous
    np.testing.assert_array_equal(image_tracks[:, 4], spatial_tracks[:, 7])
    np.testing.assert_allclose(spatial_tracks[:, :7], spatial[:, :7])
    np.testing.assert_array_equal(image_tracks[:, 7], [0])
    np.testing.assert_array_equal(spatial_tracks[:, 10], [0])


def test_python_api_smoke() -> None:
    """Exercise all public tracker imports and synthetic tracking on CPU."""

    assert torch.version.cuda is None, f"Expected CPU-only PyTorch, got torch {torch.__version__}"
    assert not torch.cuda.is_available()
    # Compare runtime to installed metadata; refresh editable installs after a bump.
    check_release_contract()
    check_tracker_api()


def test_cli_command_surface_smoke() -> None:
    result = CliRunner().invoke(boxmot_cli, ["--help"])

    assert result.exit_code == 0, result.output
    for command in EXPECTED_CLI_COMMANDS:
        assert command in result.output
    assert "generate" not in result.output
