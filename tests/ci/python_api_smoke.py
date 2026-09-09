"""Clean-install smoke tests for the coordinated v24 public cutover."""

from __future__ import annotations

import numpy as np
import torch
from click.testing import CliRunner

import boxmot
from boxmot import ByteTrack, create_tracker
from boxmot.engine.cli import boxmot as boxmot_cli
from boxmot.trackers import TrackerSpec


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
        "HybridSort",
        "MafHda",
        "OccluBoost",
        "OcSort",
        "Sam2Mot",
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
    for command in (
        "track",
        "materialize",
        "eval",
        "tune",
        "research",
        "train-reid",
        "eval-reid",
        "compare-reid",
        "export",
        "build",
    ):
        assert command in result.output
    assert "generate" not in result.output
