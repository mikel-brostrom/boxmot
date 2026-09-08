"""Dataset FPS controls reach both perception preparation and cached replay."""

import click
import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support


@pytest.fixture(autouse=True)
def _load_commands_before_workflow_patching() -> None:
    """Keep lazy command imports from retaining a patched workflow function."""

    for command in ("materialize", "eval", "tune", "track"):
        assert boxmot.get_command(click.Context(boxmot), command) is not None


@pytest.mark.parametrize("command", ("materialize", "eval", "tune"))
@pytest.mark.parametrize("fps", (None, 5.0, 2.5))
def test_dataset_fps_reaches_workflow_namespace(monkeypatch, command: str, fps: float | None) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    argv = [command, "--experiment", "fixture-experiment"]
    if command != "materialize":
        argv += ["--build", "fixture-build"]
    if fps is not None:
        argv += ["--fps", str(fps)]

    result = CliRunner().invoke(boxmot, argv)

    assert result.exit_code == 0, result.output
    assert captured["args"].fps == fps
    if command == "materialize":
        assert ("fps" in captured["args"].materialize_explicit_keys) is (fps is not None)


@pytest.mark.parametrize("command", ("materialize", "eval", "tune"))
@pytest.mark.parametrize("value", ("0", "-5", "nan", "inf", "-inf"))
def test_dataset_fps_rejects_invalid_target_before_workflow(monkeypatch, command: str, value: str) -> None:
    calls = []
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda *args: calls.append(args))
    argv = [command, "--experiment", "fixture-experiment", "--fps", value]
    if command != "materialize":
        argv += ["--build", "fixture-build"]

    result = CliRunner().invoke(boxmot, argv)

    assert result.exit_code == 2
    assert "Invalid value for '--fps'" in result.output
    assert "finite number greater than zero" in result.output
    assert calls == []


def test_track_fps_still_controls_saved_video(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))

    result = CliRunner().invoke(boxmot, ["track", "--source", "video.mp4", "--fps", "5"])

    assert result.exit_code == 0, result.output
    assert captured["args"].fps == 5
    help_result = CliRunner().invoke(boxmot, ["track", "--help"])
    assert "Frame rate of saved tracking video" in help_result.output
