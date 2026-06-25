"""Interactive pipeline step viewer.

After a pipeline finishes, :func:`interactive_step_viewer` replaces the
Pipeline checklist inside the workflow panel with a navigable horizontal
step bar.  Left/right arrow keys move the cursor, Enter toggles the
detail shown in the Results panel, ``q`` exits.

Works in any terminal that supports raw/cbreak mode (virtually all Unix
terminals including VS Code's integrated terminal).
"""

from __future__ import annotations

import os
import sys
import termios
import tty
from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console as _RichConsole
from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import boxmot.utils.rich.core.ui as ui

if TYPE_CHECKING:
    from boxmot.utils.rich.workflow.pipeline import StepRecord


# ---------------------------------------------------------------------------
# Vertical scrolling helper
# ---------------------------------------------------------------------------


def _scroll_renderable(
    renderable: RenderableType,
    width: int,
    max_height: int,
    scroll_offset: int,
) -> tuple[RenderableType, int, int]:
    """Render *renderable* to text, then return a vertical slice.

    Returns ``(sliced_renderable, total_lines, clamped_offset)``.
    When the content fits within *max_height* the original renderable is
    returned unchanged.
    """
    buf = StringIO()
    cons = _RichConsole(
        file=buf,
        force_terminal=True,
        width=max(20, width),
        color_system="truecolor",
        theme=ui._boxmot_theme(),
    )
    cons.print(renderable, end="")
    lines = buf.getvalue().split("\n")
    # Strip trailing blank lines produced by Rich
    while lines and lines[-1] == "":
        lines.pop()
    total = len(lines)

    if total <= max_height:
        return renderable, total, 0

    # Reserve space for scroll indicators
    content_height = max_height
    has_above = scroll_offset > 0
    has_below = (scroll_offset + content_height) < total
    if has_above:
        content_height -= 1
    if has_below:
        content_height -= 1
    content_height = max(1, content_height)

    max_off = max(0, total - content_height)
    off = max(0, min(scroll_offset, max_off))

    visible = lines[off : off + content_height]
    has_above = off > 0
    has_below = (off + content_height) < total

    parts: list[str] = []
    if has_above:
        parts.append(f"\x1b[2m  ▲ {off} more lines above\x1b[0m")
    parts.extend(visible)
    if has_below:
        remaining = total - off - content_height
        parts.append(f"\x1b[2m  ▼ {remaining} more lines below\x1b[0m")

    return Text.from_ansi("\n".join(parts)), total, off


def _read_key() -> str:
    """Read a single keypress from stdin (raw mode).

    Returns human-readable names for special keys:
    ``'left'``, ``'right'``, ``'enter'``, ``'q'``, or the literal character.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            if seq == "[C":
                return "right"
            if seq == "[D":
                return "left"
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "enter"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _build_interactive_pipeline_section(
    labels: list[str],
    cursor: int,
    active_step: int | None,
) -> Group:
    """Build a horizontal Pipeline bar with a movable cursor.

    Matches the static ``[✓] Setup / Generate / ... DONE`` layout but
    highlights the cursor position so the user can select a step.
    """
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(ratio=1)
    table.add_column(justify="right", no_wrap=True)

    row = Text()
    row.append(f"{ui.STEP_DONE_MARKER} ", style=ui.STYLE_STATUS_DONE)

    for idx, label in enumerate(labels):
        compact = ui._compact_step_label(label)
        is_selected = idx == cursor

        if is_selected:
            row.append(compact, style="bold reverse cyan")
        else:
            row.append(compact, style=ui.STYLE_TEXT_STRONG)

        if idx < len(labels) - 1:
            row.append(" / ", style=ui.STYLE_MUTED)

    # Hint line
    hint = Text()
    hint.append("←→", style="bold cyan")
    hint.append(" navigate  ", style=ui.STYLE_MUTED)
    hint.append("↑↓", style="bold cyan")
    hint.append(" scroll  ", style=ui.STYLE_MUTED)
    hint.append("Enter", style="bold cyan")
    hint.append(" select  ", style=ui.STYLE_MUTED)
    hint.append("q", style="bold cyan")
    hint.append(" quit", style=ui.STYLE_MUTED)

    table.add_row(row, Text("DONE", style=ui.STYLE_STATUS_DONE))
    table.add_row(hint)

    return Group(
        Rule(Text("Pipeline", style=ui.STYLE_TITLE), style=ui.STYLE_RULE),
        table,
    )


def _get_step_detail(
    labels: list[str],
    records: dict[str, StepRecord],
    active_step: int | None,
) -> tuple[str | None, str | None, RenderableType | None]:
    """Return (title, text, renderable) for the active step's detail."""
    if active_step is None or active_step < 0 or active_step >= len(labels):
        return None, None, None
    label = labels[active_step]
    record = records.get(label)
    if record is None:
        return None, None, None
    compact = ui._compact_step_label(label)
    if record.detail_renderable is not None:
        return compact, None, record.detail_renderable
    if record.detail_text:
        return compact, record.detail_text, None
    return compact, "(no detail)", None


def _build_full_interactive_panel(
    labels: list[str],
    records: dict[str, StepRecord],
    workflow: ui.WorkflowProgress,
    cursor: int,
    active_step: int | None,
    scroll_offset: int = 0,
    max_detail_height: int = 0,
) -> tuple[Panel, int, int]:
    """Render the full workflow panel with interactive Pipeline and detail.

    Returns ``(panel, total_detail_lines, clamped_scroll_offset)``.
    """
    renderables: list[RenderableType] = []

    # Interactive Pipeline section (horizontal bar)
    renderables.append(_build_interactive_pipeline_section(labels, cursor, active_step))

    total_lines = 0
    clamped_offset = 0

    # Detail section — changes based on which step is selected.
    # Setup step (index 0): show the workflow fields + any step detail (YAML configs)
    # Other steps: show their stored detail
    # No selection: show the workflow's original Results
    if active_step is not None and active_step == 0:
        # Build a combined Setup view: fields panel + step detail (configs)
        detail_parts: list[RenderableType] = []
        primary_fields, extra_panels = ui._split_workflow_fields(workflow.fields)
        setup_panel = ui._build_setup_panel(primary_fields, extra_panels, compact=True)
        if setup_panel is not None:
            detail_parts.append(setup_panel)
        # Append any extra detail stored on the step (e.g. YAML configs)
        record = records.get(labels[0])
        if record is not None:
            if record.detail_renderable is not None:
                detail_parts.append(record.detail_renderable)
            elif record.detail_text:
                detail_parts.append(Text(record.detail_text, style=ui.STYLE_TEXT))
        if detail_parts:
            inner: RenderableType = Group(*detail_parts)
            if max_detail_height > 0:
                inner, total_lines, clamped_offset = _scroll_renderable(
                    inner,
                    _detail_width(),
                    max_detail_height,
                    scroll_offset,
                )
            renderables.append(inner)
    elif active_step is not None:
        title, text, renderable = _get_step_detail(labels, records, active_step)
        if renderable is not None or text:
            content: RenderableType = renderable if renderable is not None else ui._decode_ansi(text or "")
            if max_detail_height > 0:
                content, total_lines, clamped_offset = _scroll_renderable(
                    content,
                    _detail_width(),
                    max_detail_height,
                    scroll_offset,
                )
            renderables.append(
                Group(
                    Rule(Text(title or "Detail", style=ui.STYLE_TITLE), style=ui.STYLE_RULE),
                    content,
                )
            )
    else:
        detail_panel = ui._build_detail_panel(
            workflow.detail_title,
            workflow.detail_text,
            workflow.detail_renderable,
        )
        if detail_panel is not None:
            renderables.append(detail_panel)

    panel = Panel(
        Group(*renderables),
        title=Text(workflow.title, style=ui.STYLE_TITLE_MAIN),
        border_style=ui.STYLE_BORDER_OUTER,
        padding=(0, 1),
    )
    return panel, total_lines, clamped_offset


def _detail_width() -> int:
    """Estimate the usable width for detail content inside the outer panel."""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    # outer panel border (2) + padding (2)
    return max(20, cols - 4)


def interactive_step_viewer(
    labels: list[str],
    records: dict[str, StepRecord],
    workflow: ui.WorkflowProgress,
) -> None:
    """Launch a live interactive viewer embedded in the workflow panel.

    Blocks until the user presses ``q`` or ``Esc``.
    """
    if not labels:
        return

    # Only run in a real terminal
    if not sys.stdin.isatty():
        return

    cursor = 0
    active_step: int | None = None
    console = ui.get_console(stderr=workflow.stderr)

    scroll_offset = 0
    total_lines = 0

    def _max_detail_height() -> int:
        """Available lines for detail content inside the outer panel."""
        try:
            rows = os.get_terminal_size().lines
        except OSError:
            rows = 24
        # outer panel (2) + pipeline rule (1) + step bar (1) + hint (1) + detail rule (1) = 6
        return max(4, rows - 6)

    panel, total_lines, scroll_offset = _build_full_interactive_panel(
        labels,
        records,
        workflow,
        cursor,
        active_step,
        scroll_offset=scroll_offset,
        max_detail_height=_max_detail_height(),
    )
    with Live(panel, console=console, auto_refresh=False, transient=True) as live:
        live.refresh()
        while True:
            key = _read_key()
            if key in ("q", "esc", "\x03"):  # q, Esc, Ctrl-C
                break
            elif key == "left":
                cursor = max(0, cursor - 1)
            elif key == "right":
                cursor = min(len(labels) - 1, cursor + 1)
            elif key == "up":
                if active_step is not None and total_lines > 0:
                    scroll_offset = max(0, scroll_offset - 3)
                else:
                    cursor = max(0, cursor - 1)
            elif key == "down":
                if active_step is not None and total_lines > 0:
                    scroll_offset = scroll_offset + 3
                else:
                    cursor = min(len(labels) - 1, cursor + 1)
            elif key == "enter":
                # Toggle: select this step or deselect if already active
                if active_step == cursor:
                    active_step = None
                    scroll_offset = 0
                else:
                    active_step = cursor
                    scroll_offset = 0

            panel, total_lines, scroll_offset = _build_full_interactive_panel(
                labels,
                records,
                workflow,
                cursor,
                active_step,
                scroll_offset=scroll_offset,
                max_detail_height=_max_detail_height(),
            )
            live.update(panel)
            live.refresh()
