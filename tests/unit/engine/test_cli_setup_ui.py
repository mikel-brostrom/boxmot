"""Setup must be visible before CLI work starts loading runtime dependencies."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from boxmot.engine import experiment_config
from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.engine.ui.core import ui
from boxmot.utils.config import ConfigurationError


def _console(monkeypatch, *, terminal: bool = True) -> tuple[Console, StringIO]:
    """Capture the actual Rich display, including its active Live ownership."""
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=terminal,
        width=100,
        height=40,
        theme=ui.BOXMOT_THEME,
        _environ={"TERM": "xterm-256color"},
    )
    monkeypatch.setattr(ui, "_stderr_console", console)
    return console, output


@pytest.mark.parametrize(
    ("module_name", "title"),
    (
        ("boxmot.engine.materialization.workflow", "Dataset Materialization"),
        ("boxmot.engine.eval.evaluator", "Evaluation"),
    ),
)
def test_setup_is_visible_during_import_and_releases_live_before_main(monkeypatch, module_name, title) -> None:
    console, output = _console(monkeypatch)
    args = SimpleNamespace()

    def main(received):
        assert received is args
        assert console._live_stack == []
        return "result"

    def import_workflow(name):
        assert name == module_name
        assert "Setup" in output.getvalue()
        assert title in output.getvalue()
        assert len(console._live_stack) == 1
        return SimpleNamespace(main=main)

    monkeypatch.setattr(_support.importlib, "import_module", import_workflow)

    assert _support._run_engine_workflow(module_name, args) == "result"
    assert console._live_stack == []


@pytest.mark.parametrize("error", (ImportError("missing runtime"), KeyboardInterrupt()))
def test_setup_releases_terminal_on_import_failure(monkeypatch, error) -> None:
    console, output = _console(monkeypatch)

    def fail(_name):
        assert "Setup" in output.getvalue()
        raise error

    monkeypatch.setattr(_support.importlib, "import_module", fail)
    expected = click.ClickException if isinstance(error, ImportError) else KeyboardInterrupt
    with pytest.raises(expected):
        _support._run_engine_workflow("boxmot.engine.eval.evaluator", SimpleNamespace())

    assert console._live_stack == []
    assert output.getvalue().endswith("\x1b[?25h")


def test_setup_does_not_add_output_to_non_terminal_commands(monkeypatch) -> None:
    console, output = _console(monkeypatch, terminal=False)

    def main(_args):
        assert console._live_stack == []
        return "result"

    monkeypatch.setattr(_support.importlib, "import_module", lambda _name: SimpleNamespace(main=main))

    with _support._workflow_setup("Evaluation", "Resolving experiment…"):
        assert output.getvalue() == ""
    assert _support._run_engine_workflow("boxmot.engine.eval.evaluator", SimpleNamespace()) == "result"
    assert output.getvalue() == ""


@pytest.mark.parametrize(("command", "title"), (("eval", "Evaluation"), ("tune", "Tuning")))
def test_replay_shows_setup_before_resolving_components_and_preserves_usage_errors(
    monkeypatch, command: str, title: str
) -> None:
    console, output = _console(monkeypatch)

    def resolve(**_kwargs):
        assert "Resolving experiment" in output.getvalue()
        assert "Setup" in output.getvalue()
        assert title in output.getvalue()
        assert len(console._live_stack) == 1
        raise ConfigurationError("No matching authored experiment")

    monkeypatch.setattr(experiment_config, "resolve_matching_experiment_path", resolve)
    result = CliRunner().invoke(boxmot, [command, "--dataset", "mot17", "--detector", "missing"])

    assert result.exit_code == 2, result.output
    assert "No matching authored experiment" in result.output
    assert console._live_stack == []


def test_eval_help_does_not_start_setup(monkeypatch) -> None:
    console, output = _console(monkeypatch)

    result = CliRunner().invoke(boxmot, ["eval", "--help"])

    assert result.exit_code == 0, result.output
    assert "Resolving experiment" not in output.getvalue()
    assert console._live_stack == []
