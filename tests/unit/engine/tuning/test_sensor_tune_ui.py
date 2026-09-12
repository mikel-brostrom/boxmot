"""Sensor tuning uses the shared Rich workflow without leaking trial log streams."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from click.testing import CliRunner
from rich.console import Console
from rich.text import Text

from boxmot.engine.cli import boxmot
from boxmot.engine.eval import eagermot_kitti as evaluation
from boxmot.engine.tuning import tuner
from boxmot.engine.ui.core import ui
from tests.unit.engine.eval.test_eagermot_kitti import _fixture

optuna = pytest.importorskip("optuna")


def _terminal(monkeypatch: pytest.MonkeyPatch) -> tuple[Console, StringIO]:
    """Capture real Rich Live updates at a useful terminal size."""
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        width=160,
        height=60,
        theme=ui.BOXMOT_THEME,
        _environ={"TERM": "xterm-256color"},
    )
    monkeypatch.setattr(ui, "_stderr_console", console)
    return console, output


def _args(data: SimpleNamespace, *, verbose: bool = False, n_trials: int = 2) -> SimpleNamespace:
    """Select the tiny real sensor fixture without image workflow defaults."""
    return SimpleNamespace(
        dataset=data.dataset,
        tracker="eagermot",
        project=data.project,
        n_trials=n_trials,
        seed=0,
        verbose=verbose,
    )


def _plain(output: StringIO) -> str:
    return Text.from_ansi(output.getvalue()).plain


@pytest.fixture
def log_output(monkeypatch: pytest.MonkeyPatch) -> Iterator[StringIO]:
    """Observe emitted tracker and optimizer records independently of Rich rendering."""
    output = StringIO()
    handler = logging.StreamHandler(output)
    loggers = (logging.getLogger("boxmot"), logging.getLogger("optuna"))
    for logger in loggers:
        monkeypatch.setattr(logger, "level", logging.INFO)
        logger.addHandler(handler)
    try:
        yield output
    finally:
        for logger in loggers:
            logger.removeHandler(handler)


def test_sensor_tune_cli_displays_real_trials_frames_best_metrics_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_output: StringIO
) -> None:
    data = _fixture(tmp_path)
    console, output = _terminal(monkeypatch)
    original_threads = torch.get_num_threads()
    command = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--dataset",
            str(data.dataset),
            "--tracker",
            "eagermot",
            "--project",
            str(data.project),
            "--n-trials",
            "2",
        ],
    )

    assert command.exit_code == 0, (command.output, command.exception, _plain(output))
    rendered = _plain(output)
    assert "Tuning" in rendered
    assert "trial 1/2" in rendered
    assert "trial 2/2" in rendered
    assert "2/2 trials done" in rendered
    assert "0002" in rendered
    assert "3/3 frames" in rendered
    assert "BEST TRIAL SUMMARY" in rendered
    assert "Saved Artifacts" in rendered
    assert "Best config (trial 1)" in rendered
    assert "best.yaml" in rendered
    assert "study.sqlite3" in rendered
    assert "run.json" in rendered
    assert "HOTA" in rendered and "100.00" in rendered
    assert "EagerMOT tuning: trial" not in log_output.getvalue()
    assert "EagerMOT 0002: tracking" not in log_output.getvalue()
    assert "A new study created" not in log_output.getvalue()
    assert "Trial 0 finished" not in log_output.getvalue()
    assert "Best profiles:" not in command.output
    assert console._live_stack == []
    assert torch.get_num_threads() == original_threads
    assert logging.getLogger("boxmot").level == logging.INFO
    assert optuna.logging.get_verbosity() == logging.INFO
    assert (data.project / "val/pipeline_steps/optimize_trials.txt").is_file()


@pytest.mark.parametrize(
    "failure", (RuntimeError("sensor trial failed"), KeyboardInterrupt("sensor trial interrupted"))
)
def test_sensor_tune_failure_restores_live_logging_and_completed_profile_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log_output: StringIO,
    failure: BaseException,
) -> None:
    data = _fixture(tmp_path)
    console, output = _terminal(monkeypatch)
    original_threads = torch.get_num_threads()
    replay = evaluation._replay
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        assert len(console._live_stack) == 1
        assert logging.getLogger("boxmot").level == logging.ERROR
        assert optuna.logging.get_verbosity() == logging.ERROR
        if calls == 2:
            raise failure
        return replay(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_replay", fail_second)

    with pytest.raises(type(failure), match=str(failure)) as raised:
        tuner.main(_args(data))

    assert raised.value._workflow_rendered_error is True
    assert console._live_stack == []
    assert "\x1b[?25h" in output.getvalue()
    assert str(failure) in _plain(output)
    assert "trial 2/2" in _plain(output)
    assert torch.get_num_threads() == original_threads
    assert logging.getLogger("boxmot").level == logging.INFO
    assert optuna.logging.get_verbosity() == logging.INFO
    assert "Trial 1 failed" not in log_output.getvalue()
    manifest = json.loads((data.project / "val/run.json").read_text())
    assert manifest["status"] == ("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")
    assert manifest["completed_trials"] == 1
    assert manifest["best_hota"] == pytest.approx(100)
    assert (data.project / "val/best.yaml").is_file()


def test_verbose_sensor_tune_retains_diagnostics_and_restores_log_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_output: StringIO
) -> None:
    data = _fixture(tmp_path)
    console, output = _terminal(monkeypatch)

    result = tuner.main(_args(data, verbose=True, n_trials=1))

    assert result.workflow_rendered is True
    assert str(result) == ""
    assert "BEST TRIAL SUMMARY" in _plain(output)
    assert "EagerMOT tuning: trial 1/1" in log_output.getvalue()
    assert "A new study created" in log_output.getvalue()
    assert "Trial 0 finished" in log_output.getvalue()
    assert logging.getLogger("boxmot").level == logging.INFO
    assert optuna.logging.get_verbosity() == logging.INFO
    assert console._live_stack == []


def test_python_sensor_tune_does_not_create_cli_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_output: StringIO
) -> None:
    data = _fixture(tmp_path)
    console, output = _terminal(monkeypatch)

    result = tuner.run_tune(_args(data, n_trials=1))

    assert result.workflow_rendered is False
    assert "HOTA" in result.render()
    assert output.getvalue() == ""
    assert console._live_stack == []
    assert logging.getLogger("boxmot").level == logging.INFO
    assert optuna.logging.get_verbosity() == logging.INFO
