"""Clean-install smoke tests for the public package and command interface."""

from __future__ import annotations

import torch
from click.testing import CliRunner

from boxmot.engine.cli import boxmot as boxmot_cli
from tests.ci.release_contract import EXPECTED_CLI_COMMANDS, check_release_contract, check_tracker_api


def test_python_api_smoke() -> None:
    """Exercise all public tracker imports and synthetic tracking on CPU."""

    assert torch.version.cuda is None, f"Expected CPU-only PyTorch, got torch {torch.__version__}"
    assert not torch.cuda.is_available()
    # Compare runtime to installed metadata; refresh editable installs after a bump.
    check_release_contract()
    check_tracker_api()


def test_cli_command_surface_smoke() -> None:
    result = CliRunner().invoke(boxmot_cli, ["--help"])

    assert result.exit_code == 0, result.output
    for command in EXPECTED_CLI_COMMANDS:
        assert command in result.output
    assert "generate" not in result.output
