"""Single-device CLI selection and import-light command help."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot


@pytest.mark.parametrize("command", ["track", "materialize", "eval", "tune", "export"])
@pytest.mark.parametrize("device", ["0,1", "cuda:0,cuda:1", "auto", ""])
def test_runtime_commands_reject_non_single_device_selectors(command: str, device: str) -> None:
    arguments = [command, "--device", device]
    if command in {"materialize", "eval", "tune"}:
        arguments.extend(["--experiment", "fixture"])

    result = CliRunner().invoke(boxmot, arguments)

    assert result.exit_code == 2, result.output
    assert "expected a single device" in result.output
    assert "GPU lists are not supported" in result.output


@pytest.mark.parametrize("command", ["track", "materialize", "eval", "tune", "export"])
def test_device_option_help_advertises_one_logical_device(command: str) -> None:
    selected = boxmot.get_command(click.Context(boxmot), command)
    option = next(parameter for parameter in selected.params if parameter.name == "device")

    assert "One" in option.help
    assert "cpu, mps, cuda:N, or N" in option.help
    assert "0,1" not in option.help


@pytest.mark.parametrize(("value", "expected"), [("2", "cuda:2"), ("cuda", "cuda:0"), ("MPS", "mps")])
def test_materialize_cli_normalizes_devices_without_probing_hardware(monkeypatch, value: str, expected: str) -> None:
    from boxmot.utils import devices

    monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: pytest.fail("Unexpected CUDA probe"))
    monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: pytest.fail("Unexpected MPS probe"))
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda args: captured.setdefault("args", args)),
    )

    result = CliRunner().invoke(boxmot, ["materialize", "--experiment", "fixture", "--device", value])

    assert result.exit_code == 0, result.output
    assert captured["args"].device == expected
    assert "device" in captured["args"].materialize_explicit_keys


def test_device_option_help_keeps_torch_and_workflow_imports_lazy() -> None:
    script = """
import sys
from click.testing import CliRunner
from boxmot.engine.cli import boxmot

for command in ("track", "materialize", "eval", "tune", "export"):
    result = CliRunner().invoke(boxmot, [command, "--help"])
    assert result.exit_code == 0, result.output
assert "torch" not in sys.modules
assert "boxmot.utils.devices" not in sys.modules
assert "boxmot.engine.materialization.workflow" not in sys.modules
assert "boxmot.engine.tracking.workflow" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
