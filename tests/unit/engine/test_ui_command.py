"""The interactive launcher dispatches through the existing Click boundary."""

from __future__ import annotations

import click
from click.testing import CliRunner

from boxmot.engine.cli import boxmot


def test_launcher_requires_terminal_but_help_does_not() -> None:
    """Piped commands must never block waiting for a keyboard UI."""
    help_result = CliRunner().invoke(boxmot, ["ui", "--help"])
    assert help_result.exit_code == 0
    assert "--experiment" in help_result.output
    result = CliRunner().invoke(boxmot, ["ui"])
    assert result.exit_code == 2
    assert "requires an interactive terminal" in result.output


def test_launcher_rejects_experiment_for_a_command_that_cannot_use_it() -> None:
    """An explicit experiment must not be silently discarded by the launcher."""
    result = CliRunner().invoke(boxmot, ["ui", "--command", "track", "--experiment", "mot17/example.yaml"])
    assert result.exit_code == 2
    assert "Experiments apply to materialize, eval, and tune" in result.output


def test_launcher_dispatches_literal_arguments_and_preserves_sources(monkeypatch) -> None:
    """Use Click parsing, never a shell or ctx.invoke with all defaults explicit."""
    from boxmot.engine.ui import keyboard
    from boxmot.engine.ui.launcher import app as launcher

    seen: dict[str, object] = {}

    @click.command("track")
    @click.option("--source", default="0")
    @click.option("--tracker", default="bytetrack")
    @click.pass_context
    def track(ctx: click.Context, source: str, tracker: str) -> None:
        seen.update(
            source=source,
            tracker=tracker,
            source_origin=ctx.get_parameter_source("source"),
            tracker_origin=ctx.get_parameter_source("tracker"),
        )

    source = "a video; $(no-shell).mp4"
    monkeypatch.setitem(boxmot.commands, "track", track)
    monkeypatch.setattr(keyboard, "is_interactive", lambda console: True)
    monkeypatch.setattr(launcher, "run_launcher", lambda *a, **kw: ["track", "--source", source])
    result = CliRunner().invoke(boxmot, ["ui"])

    assert result.exit_code == 0, result.output
    assert seen == {
        "source": source,
        "tracker": "bytetrack",
        "source_origin": click.core.ParameterSource.COMMANDLINE,
        "tracker_origin": click.core.ParameterSource.DEFAULT,
    }


def test_launcher_cancel_does_not_invoke_command(monkeypatch) -> None:
    """Leaving the form must not launch its preselected workflow."""
    from boxmot.engine.ui import keyboard
    from boxmot.engine.ui.launcher import app as launcher

    monkeypatch.setattr(keyboard, "is_interactive", lambda console: True)
    monkeypatch.setattr(launcher, "run_launcher", lambda *a, **kw: None)
    result = CliRunner().invoke(boxmot, ["ui", "--command", "track"])
    assert result.exit_code == 0
    assert "Running:" not in result.output
