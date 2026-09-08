"""Shared Rich task bars for workflow detail panels."""

from __future__ import annotations

from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.text import Text

import boxmot.engine.ui.core.ui as ui


class _TaskCountColumn(ProgressColumn):
    """Render a task count in the unit selected by its workflow."""

    def __init__(self, unit: str) -> None:
        super().__init__()
        self._unit = unit

    def render(self, task: Task) -> Text:
        """Show the total when known, otherwise only the completed count."""
        completed = int(task.completed)
        if task.total is None:
            return Text(f"{completed:,} {self._unit}", style=ui.STYLE_MUTED)
        return Text(f"{completed:,}/{int(task.total):,} {self._unit}", style=ui.STYLE_MUTED)


class _TaskStatusColumn(ProgressColumn):
    """Render the shared state marker and optional workflow-specific detail."""

    _STYLES = {
        "queued": ("○", "pending", ui.STYLE_STATUS_TODO),
        "running": ("▶", "running", ui.STYLE_STATUS_ACTIVE),
        "completed": ("✓", "done", ui.STYLE_STATUS_DONE),
        "failed": ("✕", "failed", ui.STYLE_STATUS_FAILED),
    }

    def render(self, task: Task) -> Text:
        """Use task fields ``status`` and ``detail`` supplied by the presenter."""
        status = str(task.fields["status"])
        marker, label, style = self._STYLES[status]
        rendered = Text()
        rendered.append(marker, style=style)
        rendered.append(f" {label}", style=style)
        detail = task.fields.get("detail")
        if detail:
            rendered.append(f" · {detail}", style=ui.STYLE_MUTED)
        return rendered


def create_task_progress(*, unit: str) -> Progress:
    """Create label, bar, count, and status columns with driver-owned refresh.

    Tasks carry a ``status`` field containing queued, running, completed, or
    failed, and an optional ``detail`` field. Presenters own task lifecycle and
    refresh scheduling so the surrounding workflow remains the only live view.
    """
    return Progress(
        TextColumn("{task.description}", style=ui.STYLE_TEXT_STRONG, markup=False),
        BarColumn(),
        _TaskCountColumn(unit),
        _TaskStatusColumn(),
        expand=True,
        auto_refresh=False,
    )
