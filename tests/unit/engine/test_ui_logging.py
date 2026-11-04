"""Logging integration at the engine-owned Rich UI boundary."""

from __future__ import annotations

import logging
from io import StringIO

from click.testing import CliRunner
from rich.logging import RichHandler

import boxmot.engine.ui.logging as engine_logging
from boxmot.engine.cli import boxmot
from boxmot.engine.ui.core.ui import get_console


def test_engine_logging_reuses_the_live_ui_console() -> None:
    logger = logging.getLogger("boxmot")
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    previous_propagate = logger.propagate
    try:
        configured = engine_logging.configure_engine_logging()
        rich_handlers = [handler for handler in configured.handlers if isinstance(handler, RichHandler)]

        assert len(rich_handlers) == 1
        assert rich_handlers[0].console is get_console(stderr=True)
    finally:
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def test_cli_configures_engine_logging_before_command_dispatch(monkeypatch) -> None:
    calls: list[None] = []
    monkeypatch.setattr(engine_logging, "configure_engine_logging", lambda: calls.append(None))

    result = CliRunner().invoke(boxmot, ["materialize"])

    assert result.exit_code != 0
    assert calls == [None]


def test_active_dashboard_captures_logs_and_normal_output_resumes(monkeypatch) -> None:
    """Suppress duplicate terminal records only while a dashboard accepts them."""
    from boxmot.engine.ui import dashboard

    messages: list[tuple[str, str]] = []
    active = True

    def capture(message: str, level: str) -> bool:
        if active:
            messages.append((message, level))
        return active

    monkeypatch.setattr(dashboard, "capture_dashboard_log", capture)
    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.addFilter(engine_logging._DashboardLogFilter())
    logger = logging.Logger("dashboard-log-test")
    logger.addHandler(handler)

    logger.warning("Sequence %s is missing", "MOT17-04")
    assert messages == [("Sequence MOT17-04 is missing", "WARNING")]
    assert output.getvalue() == ""

    active = False
    logger.warning("Normal output restored")
    assert output.getvalue() == "Normal output restored\n"
