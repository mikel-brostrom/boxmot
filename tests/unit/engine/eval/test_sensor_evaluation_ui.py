"""Exercise sensor evaluation through the real Rich terminal workflow."""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import torch
from click.testing import CliRunner
from rich.console import Console

from boxmot.engine.cli import boxmot
from boxmot.engine.eval import eagermot_kitti as sensor
from boxmot.engine.eval.evaluator import run_eval
from boxmot.engine.ui.core import ui
from tests.unit.engine.eval.test_eagermot_kitti import _arguments, _fixture


def _terminal(monkeypatch: pytest.MonkeyPatch) -> tuple[Console, StringIO]:
    """Capture one real terminal Live, including intermediate progress frames."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=130, height=45, theme=ui.BOXMOT_THEME, record=True)
    monkeypatch.setattr(ui, "_stderr_console", console)
    return console, output


@pytest.mark.parametrize("verbose", (False, True))
def test_eval_cli_displays_live_sequence_progress_and_final_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verbose: bool
) -> None:
    """Actual sensor frames must run inside the shared Evaluation panel."""
    data = _fixture(tmp_path)
    console, _output = _terminal(monkeypatch)
    track = sensor._track_frame
    score = sensor.evaluate_kitti_mots
    frames: list[str] = []
    logger = logging.getLogger("boxmot")
    previous_level = logger.level

    def observe_frame(*args: Any, **kwargs: Any) -> Any:
        assert len(console._live_stack) == 1
        frames.append(ui.capture_renderable(console._live_stack[0].renderable, width=130))
        assert logger.isEnabledFor(logging.INFO) is verbose
        return track(*args, **kwargs)

    def observe_scoring(*args: Any, **kwargs: Any) -> Any:
        assert len(console._live_stack) == 1
        frame = ui.capture_renderable(console._live_stack[0].renderable, width=130)
        assert "Computing KITTI mask evaluation metrics" in frame
        return score(*args, **kwargs)

    monkeypatch.setattr(sensor, "_track_frame", observe_frame)
    monkeypatch.setattr(sensor, "evaluate_kitti_mots", observe_scoring)
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), *(["--verbose"] if verbose else [])])

    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    assert len(frames) == 3
    for completed, frame in enumerate(frames):
        assert "Evaluation" in frame
        assert "Tracking: 0/1 sequences done" in frame
        assert "0002" in frame
        assert f"{completed}/3 frames" in frame
    final = console.export_text()
    assert "HOTA" in final and "MOTA" in final and "IDF1" in final
    assert "Class Avg" in final
    assert "Results:" in final
    assert "det_thresh=" not in final if not verbose else "det_thresh=" in final
    assert console._live_stack == []
    assert logger.level == previous_level
    assert json.loads((data.project / "val/metrics.json").read_text())["cls_comb_cls_av"]["HOTA"] == 100
    tracking_summary = data.project / "val/pipeline_steps/run_tracker.txt"
    assert "1/1 sequences done" in tracking_summary.read_text()
    assert "3/3 frames" in tracking_summary.read_text()


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_eval_terminal_failure_restores_logging_threads_and_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    """Abort in replay and verify the existing workflow handles its own error once."""
    data = _fixture(tmp_path)
    console, _output = _terminal(monkeypatch)
    previous_threads = torch.get_num_threads()
    logger = logging.getLogger("boxmot")
    previous_level = logger.level
    error = error_type("sensor replay stopped")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        assert len(console._live_stack) == 1
        raise error

    monkeypatch.setattr(sensor, "_track_frame", fail)
    invocation = CliRunner().invoke(boxmot, _arguments(data))

    assert invocation.exit_code != 0
    assert getattr(error, "_workflow_rendered_error", False)
    assert console._live_stack == []
    assert logger.level == previous_level
    assert torch.get_num_threads() == previous_threads
    assert "Run tracker failed" in console.export_text()
    manifest = json.loads((data.project / "val/run.json").read_text())
    assert manifest["status"] == ("interrupted" if error_type is KeyboardInterrupt else "failed")


def test_python_eval_can_request_shared_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Progress is opt-in for Python callers, and rendered results avoid duplicate printing."""
    from types import SimpleNamespace

    data = _fixture(tmp_path)
    console, _output = _terminal(monkeypatch)
    args = SimpleNamespace(dataset=data.dataset, tracker="eagermot", project=data.project)
    result = run_eval(args, verbose=False, show_progress=True)

    assert result.workflow_rendered is True
    assert str(result) == ""
    assert result.timings["frames"] == 3
    assert console._live_stack == []
    assert "HOTA" in console.export_text()
