"""Clean-install smoke tests for the coordinated v24 public cutover."""

from __future__ import annotations

import torch
from click.testing import CliRunner

import boxmot
from boxmot import ByteTrack, create_tracker
from boxmot.engine.cli import boxmot as boxmot_cli
from boxmot.structures import Boxes, Detections
from boxmot.trackers import TrackerSpec


def _detections(sample_id: str) -> Detections:
    return Detections(
        geometry=Boxes(torch.tensor([[10.0, 12.0, 30.0, 52.0]], dtype=torch.float32)),
        scores=torch.tensor([0.95], dtype=torch.float32),
        class_ids=torch.tensor([0], dtype=torch.int64),
        sample_id=sample_id,
    )


def test_python_api_smoke() -> None:
    """Exercise the strict public tracker and structure boundary on CPU."""

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
        "OccluBoost",
        "OcSort",
        "Sam2Mot",
        "SFSORT",
        "StrongSort",
    )
    assert not hasattr(boxmot, "BoxMOT")
    assert not hasattr(boxmot, "Detector")
    assert not hasattr(boxmot, "ReIDModel")

    spec = TrackerSpec(
        name="bytetrack",
        options=(("min_hits", 1), ("track_thresh", 0.2)),
    )
    tracker = create_tracker(spec)
    assert isinstance(tracker, ByteTrack)

    first = tracker.update(_detections("frame-0"))
    second = tracker.update(_detections("frame-1"))
    assert first.to_aabb_rows().shape == (1, 8)
    assert second.to_aabb_rows().shape == (1, 8)
    assert second.track_ids.tolist() == first.track_ids.tolist()


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
