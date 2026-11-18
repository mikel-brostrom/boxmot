"""Compose harness cleanup must remain useful when Docker or output fails."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.e2e.compose_attributes import ComposeTest


@pytest.fixture
def harness(tmp_path: Path) -> ComposeTest:
    """Create cleanup state without selecting ports or contacting Docker."""
    test = object.__new__(ComposeTest)
    test.directory = tmp_path
    test.project = "boxmot-cleanup-test"
    test.profile = "cpu"
    test.passed = ["fixture"]
    test.commands = []
    test.started = True
    test.dc = Mock(return_value=subprocess.CompletedProcess([], 0))
    return test


@pytest.mark.parametrize("primary_error", [None, "AssertionError: original pipeline failure"])
def test_cleanup_attempts_every_step_and_records_failures(
    harness: ComposeTest, primary_error: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Timeouts, execution errors, and failed teardown all remain observable."""
    harness.dc.side_effect = [
        subprocess.TimeoutExpired("docker logs", 120),
        OSError("cannot inspect containers"),
        subprocess.CompletedProcess([], 1),
    ]
    if primary_error is None:
        with pytest.raises(RuntimeError, match="stop isolated project: exit status 1"):
            harness.finish(primary_error)
    else:
        harness.finish(primary_error)
        assert "Cleanup failed:" in capsys.readouterr().err
    assert [call.args[0] for call in harness.dc.call_args_list] == ["logs", "ps", "down"]
    report = json.loads((harness.directory / "result.json").read_text())
    assert len(report["cleanup_errors"]) == 3
    assert "TimeoutExpired" in report["cleanup_errors"][0]
    assert "OSError" in report["cleanup_errors"][1]
    assert report["error"] == (primary_error or "Cleanup failed: " + "; ".join(report["cleanup_errors"]))


@pytest.mark.parametrize("primary_error", [None, "AssertionError: original pipeline failure"])
def test_result_write_failure_does_not_mask_primary_error(
    harness: ComposeTest, primary_error: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed report write is visible after teardown, preserving prior errors."""
    (harness.directory / "result.json").mkdir()
    if primary_error is None:
        with pytest.raises(RuntimeError, match="write result.json: IsADirectoryError"):
            harness.finish(primary_error)
    else:
        harness.finish(primary_error)
        assert "write result.json: IsADirectoryError" in capsys.readouterr().err
    assert [call.args[0] for call in harness.dc.call_args_list] == ["logs", "ps", "down"]


def test_successful_cleanup_preserves_success(harness: ComposeTest) -> None:
    """Successful cleanup does not turn a passing pipeline into a failure."""
    harness.finish(None)
    report = json.loads((harness.directory / "result.json").read_text())
    assert report["error"] is None and report["cleanup_errors"] == []
    assert [call.args[0] for call in harness.dc.call_args_list] == ["logs", "ps", "down"]
