"""Exercise real terminal input framing and restoration without launching models."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import IO

import pytest

from boxmot.engine.ui.keyboard import TerminalInput

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Uses a Unix pseudoterminal")


@pytest.fixture
def terminal() -> Iterator[tuple[int, IO[str]]]:
    """Provide an isolated terminal whose input can be supplied as bytes."""
    master, slave = os.openpty()
    with os.fdopen(slave, "r") as stream:
        try:
            yield master, stream
        finally:
            os.close(master)


def test_keyboard_bounds_idle_and_standalone_escape_reads(terminal) -> None:
    """An Escape key must not wait forever for an arrow-key suffix."""
    master, stream = terminal
    with TerminalInput(stream) as keyboard:
        assert keyboard.read_key(0) is None
        os.write(master, b"\x1b")
        assert keyboard.read_key(0.1) == "escape"
        assert keyboard.read_key(0) is None


def test_keyboard_preserves_fragmented_sequences_and_unicode(terminal) -> None:
    """Key and UTF-8 sequences may arrive across separate reads."""
    master, stream = terminal
    with TerminalInput(stream) as keyboard:
        os.write(master, b"\x1b[")
        assert keyboard.read_key(0) is None
        os.write(master, b"B\t\r")
        assert keyboard.read_key() == "down"
        assert keyboard.read_key() == "tab"
        assert keyboard.read_key() == "enter"
        os.write(master, b"\xc3")
        assert keyboard.read_key(0) is None
        os.write(master, b"\xa5")
        assert keyboard.read_key() == "å"


def test_keyboard_restores_terminal_after_exception(terminal) -> None:
    """Exceptions cannot leave the user's terminal without normal echo."""
    import termios

    _, stream = terminal
    previous = termios.tcgetattr(stream.fileno())
    with pytest.raises(RuntimeError, match="workflow failed"):
        with TerminalInput(stream):
            active = termios.tcgetattr(stream.fileno())
            assert not active[3] & termios.ICANON
            assert not active[3] & termios.ECHO
            assert active[3] & termios.ISIG
            raise RuntimeError("workflow failed")
    assert termios.tcgetattr(stream.fileno()) == previous
