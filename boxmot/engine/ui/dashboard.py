"""Interactive, driver-local views over the existing workflow reporters.

The command keeps running on its original thread. A bounded input loop only
reads keys and repaints Rich; it never calls a tracker, model, or workflow.
"""

from __future__ import annotations

import _thread
import math
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.segment import Segment, Segments
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from boxmot.engine.ui.core.ui import WorkflowProgress
    from boxmot.engine.ui.keyboard import TerminalInput

_enabled: ContextVar[bool] = ContextVar("boxmot_dashboard", default=False)
_active: LiveDashboard | None = None
_active_lock = threading.RLock()
_TABS = ("Overview", "Sequences", "Results", "Config", "Logs")


@contextmanager
def dashboard_mode(enabled: bool = True) -> Iterator[None]:
    """Enable interactive workflow displays within one command invocation."""
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


def capture_dashboard_log(message: str, level: str = "INFO") -> bool:
    """Capture a console log and report whether terminal output is owned here."""
    with _active_lock:
        dashboard = _active
    if dashboard is None or dashboard._stop.is_set():
        return False
    dashboard.add_log(message, level)
    return True


@dataclass(frozen=True)
class SequenceRow:
    """A snapshot of one reported sequence, independent of Rich progress locks."""

    name: str
    completed: float
    total: float | None
    status: str
    detail: str | None = None
    elapsed: float | None = None
    group: str = "Tracking"


@dataclass(frozen=True)
class TrialRow:
    """Measured metrics for one completed tuning trial."""

    index: int
    metrics: Mapping[str, float]


class TuningProgressView(Group):
    """Normal Rich progress with optional inspection data for a dashboard."""

    def __init__(
        self,
        *renderables: RenderableType,
        trials: Sequence[TrialRow] = (),
        maximize: Sequence[str] = (),
        minimize: Sequence[str] = (),
    ) -> None:
        super().__init__(*renderables)
        self.dashboard_trials = tuple(trials)
        self.dashboard_objectives = tuple(maximize), tuple(minimize)


def _parts(renderable: RenderableType | None) -> Iterator[RenderableType]:
    """Visit only explicit Rich groups, without executing arbitrary renderers."""
    if renderable is not None:
        yield renderable
        if isinstance(renderable, Group):
            for child in renderable.renderables:
                yield from _parts(child)


def _sparkline(values: Sequence[float], width: int = 40) -> str:
    """Plot the most recent finite observations without fabricating history."""
    finite = [value for value in values if math.isfinite(value)][-width:]
    if not finite:
        return ""
    low, high = min(finite), max(finite)
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[0 if high == low else round(7 * (value - low) / (high - low))] for value in finite)


def _duration(seconds: float | None) -> str:
    """Format elapsed time without displaying unknown values as measured zero."""
    if seconds is None:
        return "—"
    minutes, secs = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def start_dashboard(workflow: WorkflowProgress) -> LiveDashboard | None:
    """Start an opted-in display only when this process owns a real terminal."""
    if not _enabled.get() or workflow._alt_screen_disabled():
        return None
    from boxmot.engine.ui.core.ui import get_console
    from boxmot.engine.ui.keyboard import is_interactive

    console = get_console(stderr=workflow.stderr)
    if not is_interactive(console):
        return None
    with _active_lock:
        if _active is not None:
            return None
    dashboard = LiveDashboard(workflow, console)
    dashboard.start()
    return dashboard


class LiveDashboard:
    """One alternate-screen display with bounded logs and nonblocking input."""

    def __init__(self, workflow: WorkflowProgress, console: Console) -> None:
        self.workflow = workflow
        self.console = console
        self.tab = 0
        self.cursor = 0
        self.offset = 0
        self.expanded = False
        self.help_visible = False
        self.query = ""
        self.searching = False
        self.warnings_only = False
        self.frozen_logs: tuple[tuple[str, str], ...] | None = None
        self.logs: deque[tuple[str, str]] = deque(maxlen=1000)
        self.sequences: tuple[SequenceRow, ...] = ()
        self.trials: dict[int, TrialRow] = {}
        self.maximize: tuple[str, ...] = ()
        self.minimize: tuple[str, ...] = ()
        self.sort_metric = 0
        self._best_history: list[float] = []
        self._seen_trials: set[int] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._returned = threading.Event()
        self.completed = False
        self._thread: threading.Thread | None = None
        self._input: TerminalInput | None = None
        self._live: Live | None = None
        self._error: Exception | None = None
        self._started_at = time.monotonic()
        self._finished_at: float | None = None
        self._last_refresh = 0.0
        self._content_lines = 0
        self._body_height = 1
        self.inspect()

    def inspect(self) -> None:
        """Retain the latest actual reporter data when a pipeline advances."""
        sequences: list[SequenceRow] = []
        for part in _parts(self.workflow.detail_renderable):
            sequence_reader = getattr(part, "dashboard_sequences", None)
            if callable(sequence_reader):
                sequences.extend(sequence_reader())
            for trial in getattr(part, "dashboard_trials", ()):
                self.trials[trial.index] = trial
            objectives = getattr(part, "dashboard_objectives", None)
            if objectives is not None:
                self.maximize, self.minimize = objectives
        if sequences:
            self.sequences = tuple(sequences)
        new_trials = self.trials.keys() - self._seen_trials
        metric = self.primary_metric
        for index in sorted(new_trials):
            value = self.trials[index].metrics.get(metric)
            if value is not None and math.isfinite(value):
                previous = self._best_history[-1] if self._best_history else value
                self._best_history.append(min(previous, value) if metric in self.minimize else max(previous, value))
        self._seen_trials.update(new_trials)

    @property
    def primary_metric(self) -> str:
        """Use the authored objective as the leaderboard's initial metric."""
        return next(iter((*self.maximize, *self.minimize)), "HOTA")

    @property
    def stopped(self) -> bool:
        """Whether input ownership ended, including EOF or a polling error."""
        return self._stop.is_set()

    def add_log(self, message: str, level: str) -> None:
        """Bound captured records and keep arbitrary log text literal."""
        with self._lock:
            self.logs.append((level.upper(), str(message)))

    def start(self) -> None:
        """Acquire input and screen ownership, rolling back any partial start."""
        global _active
        from boxmot.engine.ui.keyboard import TerminalInput

        self._input = TerminalInput()
        try:
            self._input.__enter__()
            self._live = Live(self.render(), console=self.console, screen=True, auto_refresh=False)
            self._live.start(refresh=True)
            with _active_lock:
                _active = self
            self._thread = threading.Thread(target=self._poll, name="boxmot-dashboard", daemon=True)
            self._thread.start()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        """Restore the terminal even after an input or rendering failure."""
        global _active
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with _active_lock:
            if _active is self:
                _active = None
        try:
            if self._live is not None:
                self._live.stop()
                self._live = None
        finally:
            if self._input is not None:
                self._input.__exit__(None, None, None)
                self._input = None
        if self._error is not None:
            self.console.print(Text(f"Dashboard stopped: {self._error}", style="yellow"))

    def finish(self) -> None:
        """Keep successful results inspectable until the user returns to the CLI."""
        with self._lock:
            self.completed = True
            self._finished_at = time.monotonic()
            self.tab = self.cursor = self.offset = 0
            self.expanded = self.help_visible = False
            self.searching = False
            self.inspect()
        while not self._returned.wait(timeout=0.1):
            if self._stop.is_set():
                return

    def _poll(self) -> None:
        """Poll input without blocking inference or reading its worker outputs."""
        try:
            while not self._stop.is_set():
                assert self._input is not None
                key = self._input.read_key(timeout=0.1)
                if key == "ctrl+c":
                    _thread.interrupt_main()
                    return
                if key == "eof":
                    self._stop.set()
                    return
                with self._lock:
                    if key is not None:
                        self.handle_key(key)
                    self.refresh()
        except Exception as exc:
            self._error = exc
            self._stop.set()

    def refresh(self) -> None:
        """Capture new state and refresh without competing Rich live regions."""
        with self._lock:
            self.inspect()
            now = time.monotonic()
            if self._live is not None and now - self._last_refresh >= 0.1:
                self._live.update(self.render(), refresh=True)
                self._last_refresh = now

    def handle_key(self, key: str) -> None:
        """Apply navigation only; computation stays under command ownership."""
        if self.searching:
            if key in {"enter", "escape"}:
                self.searching = False
                if key == "escape":
                    self.query = ""
            elif key == "backspace":
                self.query = self.query[:-1]
            elif len(key) == 1 and key.isprintable():
                self.query += key
            self.offset = 0
            return
        if self.completed and (
            key == "q" or (key == "escape" and not (self.expanded or self.help_visible or self.query))
        ):
            self._returned.set()
            return
        if key == "?":
            self.help_visible = not self.help_visible
        elif key in {"tab", "right", "shift+tab", "left"} or key in "12345":
            delta = -1 if key in {"shift+tab", "left"} else 1
            self.tab = int(key) - 1 if key in "12345" else (self.tab + delta) % len(_TABS)
            self.cursor = self.offset = 0
            self.expanded = self.help_visible = False
        elif key in {"escape", "q"}:
            self.expanded = self.help_visible = False
            self.query = ""
            self.offset = 0
        elif key in {"up", "down"}:
            delta = -1 if key == "up" else 1
            if self.tab == 4 and self.frozen_logs is None and key == "up":
                self.frozen_logs = tuple(self.logs)
                self.offset = max(0, self._content_lines - self._body_height - 1)
            elif self.tab in {1, 2} and not self.expanded:
                self.cursor = max(0, self.cursor + delta)
            else:
                self.offset = max(0, self.offset + delta)
        elif key in {"pageup", "pagedown"}:
            if self.tab == 4 and self.frozen_logs is None:
                self.frozen_logs = tuple(self.logs)
                self.offset = max(0, self._content_lines - self._body_height)
            self.offset = max(0, self.offset + (-10 if key == "pageup" else 10))
        elif key == "home":
            if self.tab == 4 and self.frozen_logs is None:
                self.frozen_logs = tuple(self.logs)
            self.cursor = self.offset = 0
        elif key == "end":
            self.offset = 0 if self.tab == 4 and self.frozen_logs is None else 1_000_000
        elif key == "enter" and self.tab in {1, 2}:
            self.expanded = not self.expanded
            self.offset = 0
        elif key == "s" and self.tab == 2:
            self.sort_metric += 1
            self.cursor = 0
        elif self.tab == 4:
            if key == "/":
                self.searching = True
            elif key == "w":
                self.warnings_only = not self.warnings_only
                self.offset = 0
            elif key == "f":
                self.frozen_logs = tuple(self.logs) if self.frozen_logs is None else None
                if self.frozen_logs is None:
                    self.offset = 0

    def _sequence_view(self, height: int) -> RenderableType:
        """Render selectable sequence rows and diagnostics from reported events."""
        if not self.sequences:
            return Text("Sequence progress will appear when replay starts.", style="dim")
        self.cursor = min(self.cursor, len(self.sequences) - 1)
        selected = self.sequences[self.cursor]
        if self.expanded:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="cyan")
            table.add_column()
            for label, value in (
                ("Sequence", selected.name), ("Stage", selected.group), ("Status", selected.status),
                ("Frames", f"{selected.completed:g} / {selected.total:g}" if selected.total is not None
                 else f"{selected.completed:g} / unknown"),
                ("Elapsed", _duration(selected.elapsed)), ("Detail", selected.detail or "—"),
            ):
                table.add_row(label, Text(str(value)))
            return table
        table = Table(expand=True, box=None, padding=(0, 1), header_style="bold cyan")
        for name in ("", "Sequence", "Progress", "Status"):
            table.add_column(name, no_wrap=name != "Sequence")
        count = max(1, height - 3)
        start = max(0, min(self.cursor - count + 1, len(self.sequences) - count))
        groups = {row.group for row in self.sequences}
        for index, row in enumerate(self.sequences[start:start + count], start):
            label = f"{row.group} / {row.name}" if len(groups) > 1 else row.name
            table.add_row(
                "›" if index == self.cursor else "", Text(label),
                ProgressBar(total=row.total, completed=row.completed, width=16),
                Text(row.status, style="red" if row.status == "failed" else "green"
                     if row.status == "completed" else "yellow"),
                style="reverse" if index == self.cursor else "",
            )
        return Group(table, Text(f"{self.cursor + 1}/{len(self.sequences)} · Enter inspect", style="dim"))

    def _trial_view(self, height: int) -> RenderableType:
        """Explore measured trial metrics with explicit baseline comparisons."""
        metrics = list(dict.fromkeys((self.primary_metric, *(
            metric for row in self.trials.values() for metric in row.metrics
        ))))
        metric = metrics[self.sort_metric % len(metrics)]
        reverse = metric not in self.minimize
        measured = [row for row in self.trials.values() if math.isfinite(row.metrics.get(metric, math.nan))]
        rows = sorted(measured, key=lambda row: row.metrics[metric], reverse=reverse)
        rows.extend(row for row in self.trials.values() if row not in measured)
        self.cursor = min(self.cursor, len(rows) - 1)
        baseline = self.trials.get(1)
        if self.expanded:
            selected = rows[self.cursor]
            table = Table("Metric", f"Trial #{selected.index}", "Δ vs trial #1", box=None, header_style="bold cyan")
            for name, value in selected.metrics.items():
                reference = baseline.metrics.get(name) if baseline else None
                delta = "—" if reference is None else f"{value - reference:+.3f}"
                table.add_row(Text(name), f"{value:.3f}", delta)
            return table
        table = Table("", "Trial", metric, "Δ vs trial #1", box=None, expand=True, header_style="bold cyan")
        count = max(1, height - 5)
        start = max(0, min(self.cursor - count + 1, len(rows) - count))
        for index, row in enumerate(rows[start:start + count], start):
            value = row.metrics.get(metric)
            reference = baseline.metrics.get(metric) if baseline else None
            delta = "—" if value is None or reference is None else f"{value - reference:+.3f}"
            table.add_row(
                "›" if index == self.cursor else "", f"#{row.index}",
                "—" if value is None else f"{value:.3f}", delta,
                style="reverse" if index == self.cursor else "",
            )
        return Group(
            Text(f"Observed best {self.primary_metric}  {_sparkline(self._best_history)}", style="cyan"),
            table, Text("s change sort metric · Enter inspect · measured trials only", style="dim"),
        )

    def _results_view(self, height: int) -> RenderableType:
        """Browse completed steps, preserving full results for vertical scrolling."""
        if self.trials:
            return self._trial_view(height)
        records = list(self.workflow._dashboard_steps.items())
        if not records:
            return self.workflow.detail_renderable or Text(
                self.workflow.detail_text or "Results are not available yet."
            )
        self.cursor = min(self.cursor, len(records) - 1)
        label, (text, renderable) = records[self.cursor]
        if self.expanded:
            return Group(Text(label, style="bold cyan"), renderable or Text(text or "No saved detail."))
        table = Table("", "Completed step", box=None, header_style="bold cyan")
        for index, (label, _) in enumerate(records):
            table.add_row("›" if index == self.cursor else "", Text(label),
                          style="reverse" if index == self.cursor else "")
        return Group(table, Text("Enter inspect a completed step", style="dim"))

    def _logs_view(self) -> RenderableType:
        """Search a bounded log buffer; freezing affects only this display."""
        logs = self.frozen_logs if self.frozen_logs is not None else tuple(self.logs)
        filtered = [(level, message) for level, message in logs
                    if (not self.warnings_only or level in {"WARNING", "ERROR", "CRITICAL"})
                    and self.query.casefold() in message.casefold()]
        state = "Frozen" if self.frozen_logs is not None else "Following"
        heading = Text(f"{state} · {'Warnings/errors' if self.warnings_only else 'All levels'} · /{self.query}",
                       style="cyan")
        lines = [Text.assemble((f"{level:<8} ", "red" if level in {"ERROR", "CRITICAL"} else "yellow"
                               if level == "WARNING" else "dim"), message) for level, message in filtered]
        return Group(heading, *(lines or [Text("No matching log messages.", style="dim")]))

    def _content(self, height: int) -> RenderableType:
        """Build the selected view from workflow state and captured observations."""
        from boxmot.engine.ui.core.ui import _split_workflow_fields

        if self.help_visible:
            return Text(
                "Tab / Shift-Tab / ←→   Switch view\n1–5                     Jump to view\n"
                "↑↓                      Select row or scroll\nEnter                   Inspect selected row\n"
                "PgUp / PgDn / Home / End Scroll details\ns (Results)             Change trial sort metric\n"
                "/ (Logs)                Search messages\nw (Logs)                Warnings/errors only\n"
                "f (Logs)                Freeze/follow log scrolling\nEsc                     Close detail/search\n"
                "Ctrl-C                  Interrupt command\n\nNavigation does not pause computation."
            )
        if self.tab == 0:
            return self.workflow.detail_renderable or Text(self.workflow.detail_text or "Preparing workflow…")
        if self.tab == 1:
            return self._sequence_view(height)
        if self.tab == 2:
            return self._results_view(height)
        if self.tab == 3:
            fields, panels = _split_workflow_fields(self.workflow.fields)
            sections: list[RenderableType] = []
            for title, items in ([("Configuration", fields)] if fields else []) + panels:
                table = Table.grid(expand=True, padding=(0, 2))
                table.add_column(style="cyan", max_width=25)
                table.add_column(ratio=1, overflow="fold")
                for label, value in items:
                    table.add_row(Text(label), Text(str(value)))
                sections.extend((Text(title, style="bold cyan"), table, Text("")))
            return Group(*sections) if sections else Text("No configuration fields yet.")
        return self._logs_view()

    def render(self) -> Layout:
        """Fit the entire screen, preserving Rich styles when details scroll."""
        height = max(10, self.console.size.height)
        width = max(20, self.console.size.width - 4)
        body_height = max(1, height - 9)
        steps = Text()
        for index, (label, status) in enumerate(tuple(self.workflow.steps)):
            if index:
                steps.append("  ›  ", style="dim")
            mark = {"done": "✓", "active": "●", "failed": "✕", "todo": "○"}[status]
            steps.append(
                f"{mark} {label}",
                style={"done": "green", "active": "yellow", "failed": "red", "todo": "dim"}[status],
            )
        tabs = Text()
        for index, tab in enumerate(_TABS):
            tabs.append(f" {index + 1} {tab} ", style="bold reverse cyan" if index == self.tab else "dim")
        content = self._content(body_height)
        lines = self.console.render_lines(content, self.console.options.update(width=width), pad=False)
        self._content_lines = len(lines)
        self._body_height = body_height
        maximum = max(0, len(lines) - body_height)
        if self.tab == 4 and not self.searching and self.frozen_logs is None and self.offset == 0:
            start = maximum
        else:
            self.offset = min(self.offset, maximum)
            start = self.offset
        visible = Segments(segment for line in lines[start:start + body_height] for segment in (*line, Segment.line()))
        footer = Text("Tab views  ↑↓ navigate  Enter inspect  ? help  Ctrl-C stop", style="dim")
        if self.completed:
            footer = Text("Completed · q return  Tab views  ↑↓ navigate  Enter inspect  ? help", style="green")
        if self.searching:
            footer = Text(f"Search logs: /{self.query}▏   Enter apply · Esc clear", style="cyan")
        elif maximum:
            footer.append(f"\nLines {start + 1}–{min(start + body_height, len(lines))}/{len(lines)} · PgUp/PgDn scroll")
        layout = Layout()
        elapsed = _duration((self._finished_at or time.monotonic()) - self._started_at)
        layout.split_column(
            Layout(
                Panel(
                    Group(
                        Text(f"{self.workflow.title} · elapsed {elapsed}"), steps,
                    ),
                    title=Text("BoxMOT", style="bold cyan"), border_style="cyan",
                ), size=4,
            ),
            Layout(tabs, size=1),
            Layout(Panel(visible, title=Text("Help" if self.help_visible else _TABS[self.tab]), border_style="dim")),
            Layout(footer, size=2),
        )
        return layout
