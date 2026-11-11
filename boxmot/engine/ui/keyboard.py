"""Small, bounded terminal input reader shared by the Rich interfaces."""

from __future__ import annotations

import codecs
import os
import sys
import time
from types import TracebackType
from typing import IO, Any

from rich.console import Console

_ESCAPE_KEYS = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1b[H": "home",
    "\x1b[F": "end",
    "\x1bOH": "home",
    "\x1bOF": "end",
    "\x1b[1~": "home",
    "\x1b[4~": "end",
    "\x1b[3~": "delete",
    "\x1b[5~": "pageup",
    "\x1b[6~": "pagedown",
    "\x1b[Z": "shift+tab",
}
_CHAR_KEYS = {"\r": "enter", "\n": "enter", "\t": "tab", "\x7f": "backspace", "\b": "backspace"}
_WINDOWS_KEYS = {
    "H": "up", "P": "down", "M": "right", "K": "left", "G": "home",
    "O": "end", "S": "delete", "I": "pageup", "Q": "pagedown", "\x0f": "shift+tab",
}


def is_interactive(console: Console) -> bool:
    """Return whether input and output both support a terminal interface."""
    return bool(sys.stdin.isatty() and console.is_terminal and not console.is_dumb_terminal)


class TerminalInput:
    """Read keys without blocking execution indefinitely or consuming shell input.

    The context owns terminal mode and always restores it, including on errors.
    On Unix, cbreak mode retains normal Ctrl-C signal handling. Platform imports
    are conditional because Windows does not provide termios or select on stdin.
    """

    def __init__(self, stream: IO[str] | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self._fd: int | None = None
        self._saved: Any = None
        self._buffer = ""
        self._escape_since: float | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def __enter__(self) -> TerminalInput:
        """Enter cbreak mode only for a real input terminal."""
        if not self.stream.isatty():
            raise OSError("Interactive input requires a terminal.")
        self._fd = self.stream.fileno()
        if os.name != "nt":
            import termios
            import tty

            self._saved = termios.tcgetattr(self._fd)
            try:
                tty.setcbreak(self._fd, termios.TCSANOW)
            except BaseException:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._saved)
                self._saved = None
                self._fd = None
                raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the terminal attributes saved by this reader."""
        if self._saved is not None and self._fd is not None:
            import termios

            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._saved)
            finally:
                self._saved = None
                self._fd = None
        self._fd = None

    def _pop_key(self) -> str | None:
        """Decode buffered characters, allowing fragmented escape sequences."""
        if not self._buffer:
            self._escape_since = None
            return None
        if self._buffer.startswith("\x1b"):
            for sequence, key in _ESCAPE_KEYS.items():
                if self._buffer.startswith(sequence):
                    self._buffer = self._buffer[len(sequence):]
                    self._escape_since = None
                    return key
            now = time.monotonic()
            if self._escape_since is None:
                self._escape_since = now
            if any(sequence.startswith(self._buffer) for sequence in _ESCAPE_KEYS):
                if now - self._escape_since < 0.04:
                    return None
            self._buffer = self._buffer[1:]
            self._escape_since = None
            return "escape"
        char, self._buffer = self._buffer[0], self._buffer[1:]
        if char == "\x03":
            return "ctrl+c"
        return _CHAR_KEYS.get(char, char)

    def read_key(self, timeout: float = 0.05) -> str | None:
        """Read one key within *timeout* seconds; return None when idle.

        A standalone Escape is recognized after a short timeout, never by an
        unconditional read of additional bytes. EOF is returned as ``eof`` so
        callers can close cleanly when an attached terminal disappears.
        """
        if self._fd is None:
            raise RuntimeError("Use TerminalInput as a context manager.")
        if os.name == "nt":
            return self._read_windows(timeout)

        import select

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            key = self._pop_key()
            if key is not None:
                return key
            remaining = max(0.0, deadline - time.monotonic())
            wait = min(remaining, 0.01) if self._escape_since is not None else remaining
            ready, _, _ = select.select([self._fd], [], [], wait)
            if ready:
                data = os.read(self._fd, 4096)
                if not data:
                    return "eof"
                self._buffer += self._decoder.decode(data)
            elif time.monotonic() >= deadline:
                return self._pop_key()

    def _read_windows(self, timeout: float) -> str | None:
        """Poll the Windows console without importing Unix-only modules."""
        import msvcrt

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\x00", "\xe0"):
                    return _WINDOWS_KEYS.get(msvcrt.getwch())
                if char == "\x03":
                    return "ctrl+c"
                return "escape" if char == "\x1b" else _CHAR_KEYS.get(char, char)
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
