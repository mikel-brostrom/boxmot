"""Clean-install smoke tests for the coordinated v24 public cutover."""

from __future__ import annotations

import click
import numpy as np
import torch
from click.testing import CliRunner

import boxmot
from boxmot import ByteTrack, create_tracker
from boxmot.engine.cli import boxmot as boxmot_cli
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, MultimodalTracks
from boxmot.trackers import TrackerSpec


def test_eagermot_sensor_fusion_python_api_smoke() -> None:
    """Exercise the installed factory and canonical 2D/3D update without SDKs."""
    tracker = create_tracker(TrackerSpec("eagermot"))
    detections = Detections(
        geometry=Boxes(torch.tensor([[78, 39, 122, 62]], dtype=torch.float32)),
        scores=torch.tensor([0.95]),
        class_ids=torch.tensor([0]),
        sample_id="smoke:0",
    )
    spatial = Detections3D(
        geometry=Boxes3D(torch.tensor([[0, 1, 10, 0, 4, 2, 2]], dtype=torch.float32)),
        scores=torch.tensor([0.9]),
        class_ids=torch.tensor([0]),
        sample_id=detections.sample_id,
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


def test_python_api_smoke() -> None:
    """Exercise direct NumPy tracking through the public API on CPU."""

    assert torch.version.cuda is None, f"Expected CPU-only PyTorch, got torch {torch.__version__}"
    assert not torch.cuda.is_available()
    assert boxmot.__version__ == "24.0.0"
    assert boxmot.__all__ == (
        "__version__",
        "create_tracker",
        "BoostTrack",
        "BotSort",
        "ByteTrack",
        "DeepOcSort",
        "EagerMot",
        "HybridSort",
        "MafHda",
        "OccluBoost",
        "OcSort",
        "SFSORT",
        "StrongSort",
    )
    assert not hasattr(boxmot, "BoxMOT")
    assert not hasattr(boxmot, "Detector")
    assert not hasattr(boxmot, "ReIDModel")

    tracker = create_tracker(TrackerSpec("bytetrack"))
    assert isinstance(tracker, ByteTrack)

    dets = np.array([[100, 200, 300, 400, 0.9, 0]], dtype=np.float32)
    tracks = tracker.update(dets)
    next_tracks = tracker.update(dets)
    assert type(tracks) is np.ndarray
    assert type(next_tracks) is np.ndarray
    assert tracks.shape == (1, 8)
    assert next_tracks.shape == (1, 8)
    assert next_tracks[:, 4].tolist() == tracks[:, 4].tolist()


def test_cli_command_surface_smoke() -> None:
    result = CliRunner().invoke(boxmot_cli, ["--help"])

    assert result.exit_code == 0, result.output
    expected_commands = (
        "track",
        "materialize",
        "time-variant",
        "eval",
        "eval-trackrcnn",
        "eval-eagermot",
        "tune",
        "tune-eagermot",
        "research",
        "train-reid",
        "eval-reid",
        "compare-reid",
        "export",
        "build",
    )
    assert tuple(boxmot_cli.list_commands(click.Context(boxmot_cli))) == expected_commands
    for command in expected_commands:
        assert command in result.output
    assert "generate" not in result.output
