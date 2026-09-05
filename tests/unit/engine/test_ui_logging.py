"""Logging integration at the engine-owned Rich UI boundary."""

from __future__ import annotations

import logging

from click.testing import CliRunner
from rich.logging import RichHandler

import boxmot.engine.logging as engine_logging
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
