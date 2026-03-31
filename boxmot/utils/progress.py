from __future__ import annotations

import queue
import sys
from contextlib import contextmanager
from typing import Any, Callable


def _format_seq_progress(seq_progress: dict[str, tuple[int, int]], n_display: int = 5) -> str:
    """Format the top-N in-progress sequences as mini progress bars."""
    active = {k: v for k, v in seq_progress.items() if v[0] < v[1]}
    if not active:
        return ""

    sorted_seqs = sorted(active.items(), key=lambda item: item[1][0] / max(item[1][1], 1), reverse=True)
    display = sorted_seqs[:n_display]
    name_width = max(len(name) for name, _ in display)
    lines = []
    bar_width = 20

    for name, (current, total) in display:
        pct = current / max(total, 1)
        filled = int(bar_width * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        lines.append(f"  {name:<{name_width}s} {bar} {pct:>5.0%}  ({current}/{total})")

    return "\n".join(lines)


def _drain_progress_queue(progress_queue: Any, seq_progress: dict[str, tuple[int, int]]) -> None:
    """Read all available progress messages into the tracking progress map."""
    while True:
        try:
            name, current, total = progress_queue.get_nowait()
            seq_progress[name] = (current, total)
        except (queue.Empty, OSError):
            break


@contextmanager
def suppress_worker_thread_logs(
    configure_logging: Callable[..., Any],
    *,
    enabled: bool,
):
    """Hide worker-thread logs so the main progress display stays readable."""
    if not enabled:
        yield
        return

    configure_logging(main_thread_only=True)
    try:
        yield
    finally:
        configure_logging()


class TrackingProgressPrinter:
    """Render and update the live tracking progress block in the terminal."""

    def __init__(self, logger: Any, n_display: int = 5) -> None:
        self.logger = logger
        self.n_display = n_display
        self._prev_display_lines = 0
        self._prev_progress_msg = ""

    def render(
        self,
        *,
        progress_queue: Any,
        seq_progress: dict[str, tuple[int, int]],
        done_count: int,
        total_count: int,
    ) -> None:
        """Refresh the terminal progress block if the visible state changed."""
        _drain_progress_queue(progress_queue, seq_progress)
        header = f"Tracking: {done_count}/{total_count} sequences done"
        seq_display = _format_seq_progress(seq_progress, n_display=self.n_display)
        lines = [header] + ([seq_display] if seq_display else [])
        msg = "\n".join(lines)

        if msg == self._prev_progress_msg:
            return

        if self._prev_display_lines > 0:
            sys.stderr.write(f"\033[{self._prev_display_lines}A\033[J")
            sys.stderr.flush()

        self.logger.opt(colors=True).info(f"<cyan>{msg}</cyan>")
        self._prev_display_lines = msg.count("\n") + 1
        self._prev_progress_msg = msg

    def finish(self, *, done_count: int, total_count: int) -> None:
        """Clear the live block and optionally print a final one-line summary."""
        if self._prev_display_lines > 0:
            sys.stderr.write(f"\033[{self._prev_display_lines}A\033[J")
            sys.stderr.flush()

        final_msg = f"Tracking: {done_count}/{total_count} sequences done"
        if final_msg != self._prev_progress_msg:
            self.logger.opt(colors=True).info(f"<cyan>{final_msg}</cyan>")
