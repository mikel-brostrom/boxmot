"""Image-only workflows reject spatial trackers before starting perception."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from boxmot import EagerMot
from boxmot.engine.cli import boxmot
from boxmot.engine.eval.evaluator import eval_setup
from boxmot.engine.eval.replay import replay_build
from boxmot.engine.research.runner import TrackerResearcher
from boxmot.engine.tracking.workflow import run_track
from boxmot.pipelines import TrackingPipeline
from boxmot.trackers import TrackerSpec


@pytest.mark.parametrize(
    "arguments",
    (
        ["track", "--source", "missing-video.mp4", "--tracker", "eagermot"],
        ["eval", "--experiment", "missing-experiment.yaml", "--tracker", "eagermot"],
        ["tune", "--experiment", "missing-experiment.yaml", "--tracker", "eagermot"],
        ["research", "--experiment", "missing-experiment.yaml", "--build", "missing-build", "--tracker", "eagermot"],
    ),
)
def test_image_cli_rejects_spatial_tracker_before_loading_assets(arguments: list[str]) -> None:
    """Sensor requirements are reported even when source assets are unavailable."""
    result = CliRunner().invoke(boxmot, arguments)

    assert result.exit_code == 2, result.output
    assert "requires 3D detections and a CameraModel" in result.output
    assert "Python update() API" in result.output


def test_direct_image_workflow_rejects_spatial_tracker_before_loading_detector() -> None:
    """Programmatic engine calls share the same early capability check."""
    with pytest.raises(ValueError, match="requires 3D detections and a CameraModel"):
        run_track(SimpleNamespace(tracker="eagermot"))


@pytest.mark.parametrize("workflow", (eval_setup, TrackerResearcher))
def test_direct_replay_workflow_rejects_spatial_tracker_before_loading_build(
    workflow: Callable[[SimpleNamespace], object],
) -> None:
    """Replay cannot read unrelated dataset files before validating sensor support."""
    with pytest.raises(ValueError, match="requires 3D detections and a CameraModel"):
        workflow(SimpleNamespace(tracker="eagermot"))


def test_image_pipeline_rejects_spatial_tracker_at_construction() -> None:
    """The image pipeline has no path for independent spatial observations."""
    with pytest.raises(ValueError, match="TrackingPipeline cannot supply 3D detections"):
        TrackingPipeline(detector=None, tracker=EagerMot())


def test_replay_rejects_spatial_tracker_before_creating_output(tmp_path: Path) -> None:
    """Direct replay validates the tracker before resolving a build or writing files."""
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="requires 3D detections and a CameraModel"):
        replay_build("missing-build", TrackerSpec("eagermot"), output_dir=output)
    assert not output.exists()
