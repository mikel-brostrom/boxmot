from __future__ import annotations

import os
from dataclasses import dataclass, field
from io import StringIO
from typing import Literal, Sequence

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from boxmot.utils.rich.workflow.fields import WORKFLOW_PANEL_PREFIX

StepState = Literal["done", "active", "todo", "failed"]

STYLE_TEXT = "boxmot.text"
STYLE_TEXT_STRONG = "boxmot.text.strong"
STYLE_MUTED = "boxmot.text.muted"
STYLE_SUBTLE = "boxmot.text.subtle"
STYLE_TITLE = "boxmot.title"
STYLE_TITLE_MAIN = "boxmot.title.main"
STYLE_LABEL = "boxmot.label"
STYLE_ACCENT = "boxmot.accent"
STYLE_STATUS_DONE = "boxmot.status.done"
STYLE_STATUS_ACTIVE = "boxmot.status.active"
STYLE_STATUS_TODO = "boxmot.status.todo"
STYLE_STATUS_FAILED = "boxmot.status.failed"
STYLE_BORDER = "boxmot.border"
STYLE_BORDER_OUTER = "boxmot.border.outer"
STYLE_BORDER_DETAIL = "boxmot.border.detail"
STYLE_BORDER_FAILED = "boxmot.border.failed"
STYLE_TABLE_HEADER = "boxmot.table.header"
STYLE_COMBINED_ROW = "boxmot.row.combined"
STYLE_RULE = "boxmot.rule"

STEP_DONE_MARKER = "[✓]"

_STEP_LABELS = {
    "Set up": "Setup",
    "Generate detections and embeddings": "Generate",
    "Tune Kalman filter": "Tune KF",
    "Run tracker": "Track",
    "Postprocess tracks": "Postprocess",
    "Evaluate results": "Evaluate",
    "Export to formats": "Export",
    "Optimize trials": "Optimize",
    "Prepare workspace": "Prepare",
    "Baseline evaluation": "Baseline",
    "GEPA optimization": "Optimize",
    "Best candidate evaluation": "Best",
}

_SETUP_WORD_ABBREVIATIONS = {
    "PARAMETERS": "PARAMS",
    "PARAMETER": "PARAM",
    "THRESHOLD": "THR",
    "THRESH": "THR",
    "MATCHING": "MATCH",
    "CORRECTION": "CORR",
    "LONGTERM": "LT",
    "ASSOCIATION": "ASSOC",
    "WEIGHT": "WT",
    "LENGTH": "LEN",
    "CUSTOM": "CUST",
    "PRECISION": "PREC",
    "CONFIDENCE": "CONF",
    "POSTPROCESSING": "POST",
    "EMBEDDING": "EMB",
    "BACKEND": "BACKEND",
}

_SECTION_TITLE_ABBREVIATIONS = {
    "CONFIGURATION": "CONFIG",
    "TRACKER PARAMETERS": "TRACKER",
    "PIPELINE PARAMETERS": "PIPELINE",
}

BOXMOT_THEME_STYLES = {
    STYLE_TEXT: "#d8dee9",
    STYLE_TEXT_STRONG: "bold #eceff4",
    STYLE_MUTED: "#7d8694",
    STYLE_SUBTLE: "#5f6977",
    STYLE_TITLE: "bold cyan",
    STYLE_TITLE_MAIN: "bold #81a1c1",
    STYLE_LABEL: "bold cyan",
    STYLE_ACCENT: "bold cyan",
    STYLE_STATUS_DONE: "bold #3fb950",
    STYLE_STATUS_ACTIVE: "bold #e3b341",
    STYLE_STATUS_TODO: "bold #8b949e",
    STYLE_STATUS_FAILED: "bold #f85149",
    STYLE_BORDER: "#4c566a",
    STYLE_BORDER_OUTER: "#5e81ac",
    STYLE_BORDER_DETAIL: "#88c0d0",
    STYLE_BORDER_FAILED: "bold #f85149",
    STYLE_TABLE_HEADER: "bold cyan",
    STYLE_COMBINED_ROW: "bold #eceff4 on #3b4252",
    STYLE_RULE: "#4c566a",
}
BOXMOT_THEME = Theme(BOXMOT_THEME_STYLES)


def _boxmot_theme() -> Theme:
    return Theme(dict(BOXMOT_THEME_STYLES))


def _create_console(
    *,
    stderr: bool = False,
    file=None,
    width: int | None = None,
    force_terminal: bool | None = None,
    color_system: str | None = None,
) -> Console:
    kwargs = {
        "file": file,
        "stderr": stderr,
        "width": width,
        "highlight": False,
        "soft_wrap": False,
        "theme": _boxmot_theme(),
    }
    if force_terminal is not None:
        kwargs["force_terminal"] = force_terminal
    if force_terminal:
        kwargs["no_color"] = False
    if color_system is not None:
        kwargs["color_system"] = color_system
    return Console(**kwargs)


_stdout_console = _create_console()
_stderr_console = _create_console(stderr=True)


def get_console(*, stderr: bool = False) -> Console:
    return _stderr_console if stderr else _stdout_console


def _decode_ansi(text: str) -> Text:
    if not text:
        return Text()

    rendered = Text.from_ansi(text)
    rendered.no_wrap = True
    rendered.overflow = "ignore"
    return rendered


def print_text(text: str, *, stderr: bool = False) -> None:
    if not text:
        return
    get_console(stderr=stderr).print(_decode_ansi(text), soft_wrap=True)


def print_renderable(renderable: RenderableType, *, stderr: bool = False) -> None:
    get_console(stderr=stderr).print(renderable, crop=False)


def capture_renderable(renderable: RenderableType, *, stderr: bool = False, width: int = 120) -> str:
    return _capture_renderable(renderable, stderr=stderr, width=width)


def _capture_renderable(
    renderable: RenderableType,
    *,
    stderr: bool = False,
    width: int = 120,
    force_terminal: bool = False,
    color_system: str | None = None,
) -> str:
    buffer = StringIO()
    console = _create_console(
        file=buffer,
        stderr=stderr,
        width=width,
        force_terminal=force_terminal,
        color_system=color_system,
    )
    console.print(renderable)
    return buffer.getvalue().rstrip("\n")


def build_checklist(steps: Sequence[tuple[str, StepState]]) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(ratio=1)
    table.add_column(justify="right", no_wrap=True)

    markers = {
        "done": (STEP_DONE_MARKER, STYLE_STATUS_DONE, "DONE", STYLE_STATUS_DONE),
        "active": ("[>]", STYLE_STATUS_ACTIVE, "ACTIVE", STYLE_STATUS_ACTIVE),
        "todo": ("[ ]", STYLE_STATUS_TODO, "QUEUED", STYLE_STATUS_TODO),
        "failed": ("[!]", STYLE_STATUS_FAILED, "FAILED", STYLE_STATUS_FAILED),
    }

    for label, state in steps:
        marker, marker_style, badge, badge_style = markers.get(
            state,
            ("[ ]", STYLE_STATUS_TODO, "QUEUED", STYLE_STATUS_TODO),
        )
        row = Text()
        row.append(f"{marker} ", style=marker_style)
        row.append(
            label,
            style=STYLE_TEXT_STRONG if state in ("active", "failed") else STYLE_TEXT,
        )
        status = Text(badge, style=badge_style)
        table.add_row(row, status)

    return table


def _compact_step_label(label: str) -> str:
    return _STEP_LABELS.get(label, label)


def _build_active_steps_summary(steps: Sequence[tuple[str, StepState]]) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(ratio=1)

    markers = {
        "done": (STEP_DONE_MARKER, STYLE_STATUS_DONE),
        "active": ("[>]", STYLE_STATUS_ACTIVE),
        "todo": ("[ ]", STYLE_STATUS_TODO),
    }

    row = Text()
    for index, (label, state) in enumerate(steps):
        marker, marker_style = markers.get(state, ("[ ]", STYLE_STATUS_TODO))
        row.append(f"{marker} ", style=marker_style)
        row.append(
            _compact_step_label(label),
            style=STYLE_TEXT_STRONG if state == "active" else STYLE_TEXT,
        )
        if index != len(steps) - 1:
            row.append(" / ", style=STYLE_MUTED)

    table.add_row(row)
    return table


def _build_completed_steps_summary(steps: Sequence[tuple[str, StepState]]) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(ratio=1)
    table.add_column(justify="right", no_wrap=True)

    summary = " / ".join(_compact_step_label(label) for label, _ in steps)

    row = Text()
    row.append(f"{STEP_DONE_MARKER} ", style=STYLE_STATUS_DONE)
    row.append(summary, style=STYLE_TEXT_STRONG)
    table.add_row(row, Text("DONE", style=STYLE_STATUS_DONE))
    return table


def _normalize_panel_items(value: object) -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [(str(label), item) for label, item in value.items()]
    if isinstance(value, (list, tuple)):
        normalized: list[tuple[str, object]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append((str(item[0]), item[1]))
        return normalized
    return []


def _split_workflow_fields(
    fields: Sequence[tuple[str, object]],
) -> tuple[list[tuple[str, object]], list[tuple[str, list[tuple[str, object]]]]]:
    primary_fields: list[tuple[str, object]] = []
    extra_panels: list[tuple[str, list[tuple[str, object]]]] = []
    panel_labels: set[str] = set()

    for label, value in fields:
        label_text = str(label)
        if label_text.startswith(WORKFLOW_PANEL_PREFIX):
            title = label_text[len(WORKFLOW_PANEL_PREFIX) :].strip() or "Details"
            items = _normalize_panel_items(value)
            if items:
                extra_panels.append((title, items))
                panel_labels.update(item_label for item_label, _ in items)
            continue
        primary_fields.append((label_text, value))

    if extra_panels:
        primary_fields = [
            (label, value) for label, value in primary_fields if label not in panel_labels and label != "Image size"
        ]

    return primary_fields, extra_panels


def _format_setup_value(label: str, value: object) -> str:
    text = str(value)
    if str(label).strip().lower() in {"detector", "reid"}:
        normalized = text.replace("\\", "/")
        return os.path.basename(normalized)
    return text


def _compact_setup_label(label: str) -> str:
    words = str(label).upper().replace("_", " ").split()
    compact = " ".join(_SETUP_WORD_ABBREVIATIONS.get(word, word) for word in words)
    return compact


def _compact_section_title(title: str) -> str:
    return _SECTION_TITLE_ABBREVIATIONS.get(title.upper(), title.upper())


def _choose_compact_pair_count(
    fields: Sequence[tuple[str, object]],
    *,
    max_pairs: int = 6,
    accent_width: int = 10,
    available_width: int = 130,
) -> int:
    """Pick the largest pairs-per-row count whose actual layout fits the panel.

    This mirrors how Rich computes column widths inside a ``Table.grid``: for
    each candidate ``pair_count`` we compute, per slot ``k``, the widest label
    and widest value across the rows that land in that slot, and sum the
    resulting row width. The largest candidate that fits ``available_width``
    wins. This avoids being bullied into a low pair count by a single long
    value (e.g. file paths, the Pareto objective string) that only occupies
    one slot.
    """
    if not fields:
        return 1
    labels = [_compact_setup_label(str(label)) for label, _ in fields]
    values = [_format_setup_value(str(label), value) for label, value in fields]
    n = len(fields)
    for pair_count in range(min(max_pairs, n), 0, -1):
        slot_label = [0] * pair_count
        slot_value = [0] * pair_count
        for i in range(n):
            k = i % pair_count
            if len(labels[i]) > slot_label[k]:
                slot_label[k] = len(labels[i])
            if len(values[i]) > slot_value[k]:
                slot_value[k] = len(values[i])
        # Each pair needs label + value + ~3 cells of padding/separators.
        total = accent_width + sum(slot_label[k] + slot_value[k] + 3 for k in range(pair_count))
        if total <= available_width:
            return pair_count
    return 1


def _build_setup_section_table(
    section_title: str,
    fields: Sequence[tuple[str, object]],
    *,
    compact: bool = False,
    pair_count: int | None = None,
    min_label_width: int | None = None,
    min_value_width: int | None = None,
) -> Table:
    if compact:
        if pair_count is None:
            pair_count = _choose_compact_pair_count(fields)
    else:
        pair_count = 2

    table = Table.grid(expand=True, padding=(0, 1))
    # Auto-size all columns to their content (no fixed widths) so labels
    # never get truncated to "MIN BO…" / "USE DL…". Optional ``min_width``
    # lets the caller harmonise widths across sibling sections so their
    # columns line up vertically. The last value column gets ``ratio=1`` so
    # any leftover width flows there instead of stretching every column
    # unevenly.
    table.add_column(style=STYLE_ACCENT, no_wrap=True, width=9, overflow="ellipsis")
    for offset in range(pair_count):
        label_kwargs = {"style": STYLE_MUTED, "no_wrap": True}
        if min_label_width is not None:
            label_kwargs["min_width"] = min_label_width
        table.add_column(**label_kwargs)
        if offset == pair_count - 1:
            table.add_column(style=STYLE_TEXT, no_wrap=True, ratio=1, overflow="ellipsis")
        else:
            value_kwargs = {"style": STYLE_TEXT, "no_wrap": True}
            if min_value_width is not None:
                value_kwargs["min_width"] = min_value_width
            table.add_column(**value_kwargs)

    title_text = _compact_section_title(section_title) if compact else section_title.upper()

    for index in range(0, len(fields), pair_count):
        row: list[RenderableType] = [
            Text(title_text, style=STYLE_ACCENT, no_wrap=True, overflow="ellipsis") if index == 0 else Text("")
        ]
        for offset in range(pair_count):
            item_index = index + offset
            if item_index < len(fields):
                label, value = fields[item_index]
                label_text = _compact_setup_label(str(label)) if compact and pair_count >= 3 else str(label).upper()
                row.extend(
                    [
                        Text(label_text, style=STYLE_LABEL, no_wrap=True, overflow="ellipsis"),
                        Text(
                            _format_setup_value(str(label), value),
                            style=STYLE_TEXT,
                            no_wrap=True,
                            overflow="ellipsis",
                        ),
                    ]
                )
            else:
                row.extend([Text(""), Text("")])
        table.add_row(*row)

    return table


def _build_setup_panel(
    primary_fields: Sequence[tuple[str, object]],
    extra_panels: Sequence[tuple[str, list[tuple[str, object]]]],
    *,
    compact: bool = False,
) -> Group | None:
    sections = [("Configuration", list(primary_fields)), *[(title, list(fields)) for title, fields in extra_panels]]
    sections = [(title, fields) for title, fields in sections if fields]
    if not sections:
        return None

    # When a "Configuration" section exists (with long file paths), it keeps
    # its own layout while the remaining subsystem sections share widths.
    # When all sections are subsystem cards, each section computes its own
    # pair count for the most compact per-section layout.
    has_config_section = sections and sections[0][0] == "Configuration"

    shared_pair_count: int | None = None
    shared_min_label: int | None = None
    shared_min_value: int | None = None

    if compact and has_config_section and len(sections) > 1:
        combined: list[tuple[str, object]] = []
        for _, fields in sections[1:]:
            combined.extend(fields)
        if combined:
            shared_pair_count = _choose_compact_pair_count(combined)
            shared_min_label = max(len(_compact_setup_label(str(label))) for label, _ in combined)
            shared_min_value = max(len(_format_setup_value(str(label), value)) for label, value in combined)

    section_tables = []
    for index, (title, fields) in enumerate(sections):
        if compact and has_config_section and index > 0:
            section_tables.append(
                _build_setup_section_table(
                    title,
                    fields,
                    compact=compact,
                    pair_count=shared_pair_count,
                    min_label_width=shared_min_label,
                    min_value_width=shared_min_value,
                )
            )
        else:
            section_tables.append(_build_setup_section_table(title, fields, compact=compact))

    return Group(
        Rule(Text("Setup", style=STYLE_TITLE), style=STYLE_RULE),
        *section_tables,
    )


def _build_steps_panel(steps: Sequence[tuple[str, StepState]], *, compact: bool = False) -> Group | None:
    if not steps:
        return None
    body: RenderableType
    has_failure = any(step_state == "failed" for _, step_state in steps)
    if not has_failure and all(step_state == "done" for _, step_state in steps):
        body = _build_completed_steps_summary(steps)
    elif compact and not has_failure:
        body = _build_active_steps_summary(steps)
    else:
        body = build_checklist(steps)
    title_style = STYLE_STATUS_FAILED if has_failure else STYLE_TITLE
    return Group(
        Rule(Text("Pipeline", style=title_style), style=STYLE_RULE),
        body,
    )


def _build_detail_panel(
    detail_title: str | None,
    detail_text: str | None,
    detail_renderable: RenderableType | None,
    *,
    failed: bool = False,
    max_lines: int | None = None,
) -> Group | None:
    if detail_renderable is None and not detail_text:
        return None

    if detail_renderable is not None:
        content: RenderableType = detail_renderable
    else:
        text = detail_text or ""
        lines = text.split("\n")
        if max_lines and len(lines) > max_lines:
            hidden = len(lines) - max_lines
            lines = [f"  \u25b2 {hidden} more lines above"] + lines[-max_lines:]
            text = "\n".join(lines)
        if failed:
            # Errors are ALWAYS shown in full — fold long lines so
            # tracebacks are never silently clipped.
            rendered = Text.from_ansi(text)
            rendered.no_wrap = False
            rendered.overflow = "fold"
            content = rendered
        else:
            content = _decode_ansi(text)

    title_style = STYLE_STATUS_FAILED if failed else STYLE_TITLE
    title = Text(detail_title or "Live Detail", style=title_style)
    rule_style = STYLE_BORDER_FAILED if failed else STYLE_RULE
    return Group(
        Rule(title, style=rule_style),
        content,
    )


def build_workflow_intro(
    title: str,
    fields: Sequence[tuple[str, object]],
    *,
    steps: Sequence[tuple[str, StepState]] = (),
    detail_title: str | None = None,
    detail_text: str | None = None,
    detail_renderable: RenderableType | None = None,
    include_setup: bool = True,
    include_steps: bool = True,
    compact: bool = False,
    max_detail_lines: int | None = None,
) -> Panel:
    renderables: list[RenderableType] = []
    if include_setup:
        primary_fields, extra_panels = _split_workflow_fields(fields)
        setup_panel = _build_setup_panel(primary_fields, extra_panels, compact=compact)
        if setup_panel is not None:
            renderables.append(setup_panel)

    if include_steps:
        steps_panel = _build_steps_panel(steps, compact=compact)
        if steps_panel is not None:
            renderables.append(steps_panel)

    has_failure = any(step_state == "failed" for _, step_state in steps)
    detail_panel = _build_detail_panel(
        detail_title,
        detail_text,
        detail_renderable,
        failed=has_failure,
        max_lines=max_detail_lines,
    )
    if detail_panel is not None:
        renderables.append(detail_panel)

    return Panel(
        Group(*renderables),
        title=Text(title, style=STYLE_TITLE_MAIN),
        border_style=STYLE_BORDER_OUTER,
        padding=(0, 1),
    )


@dataclass
class WorkflowProgress:
    title: str
    fields: Sequence[tuple[str, object]]
    steps: list[tuple[str, StepState]] = field(default_factory=list)
    detail_title: str | None = None
    detail_text: str | None = None
    detail_renderable: RenderableType | None = None
    stderr: bool = False
    transient: bool = False
    prefer_alt_screen: bool = False
    prefer_compact_layout: bool = False
    _started: bool = field(default=False, init=False, repr=False)
    _live: Live | None = field(default=None, init=False, repr=False)
    include_setup: bool = True
    _oversized: bool = field(default=False, init=False, repr=False)
    _uses_alt_screen: bool = field(default=False, init=False, repr=False)
    _compact_layout: bool = field(default=False, init=False, repr=False)
    _last_rendered_state: (
        tuple[
            tuple[tuple[str, StepState], ...],
            str | None,
            str | None,
            RenderableType | None,
        ]
        | None
    ) = field(default=None, init=False, repr=False)

    def renderable(self, *, compact: bool = False, include_setup: bool = True) -> Panel:
        return build_workflow_intro(
            self.title,
            self.fields,
            steps=self.steps,
            detail_title=self.detail_title,
            detail_text=self.detail_text,
            detail_renderable=self.detail_renderable,
            include_setup=include_setup,
            compact=compact,
        )

    def _max_detail_lines(self) -> int | None:
        """Compute max lines for detail panel to fit within terminal height."""
        console = get_console(stderr=self.stderr)
        height = getattr(console.size, "height", 0) or 0
        if height <= 0:
            return None
        # Overhead: outer panel border (2) + setup (~3 compact) + steps (1) +
        # detail rule (1) + margin (2) = ~9 lines minimum for chrome
        overhead = 9
        available = height - overhead
        return max(3, available)

    def _live_renderable(self) -> Panel:
        setup = self.include_setup
        max_lines = self._max_detail_lines()
        full = self.renderable(compact=False, include_setup=setup)
        if self.prefer_compact_layout:
            self._compact_layout = True
            return self._renderable_with_limit(compact=True, include_setup=setup, max_lines=max_lines)
        if self._compact_layout or self._renderable_exceeds_console(full):
            self._compact_layout = True
            return self._renderable_with_limit(compact=True, include_setup=setup, max_lines=max_lines)
        return full

    def _renderable_with_limit(self, *, compact: bool, include_setup: bool, max_lines: int | None) -> Panel:
        return build_workflow_intro(
            self.title,
            self.fields,
            steps=self.steps,
            detail_title=self.detail_title,
            detail_text=self.detail_text,
            detail_renderable=self.detail_renderable,
            include_setup=include_setup,
            compact=compact,
            max_detail_lines=max_lines,
        )

    def _state_snapshot(
        self,
    ) -> tuple[tuple[tuple[str, StepState], ...], str | None, str | None, RenderableType | None]:
        return tuple(self.steps), self.detail_title, self.detail_text, self.detail_renderable

    def _render_snapshot(self) -> None:
        renderable = self._live_renderable()
        if self._live is not None:
            self._live.update(renderable, refresh=True)
            # WORKAROUND: Rich LiveRender.position_cursor() only erases lines
            # it moves UP through.  When the renderable shrinks (e.g. KF output
            # → progress bar), stale lines remain below.  Rich provides no
            # public API for "erase in display from cursor to end" (ED 0), so
            # we emit the ANSI escape directly.  This bypasses Rich's pipeline
            # intentionally; revisit if Rich adds a clear-below helper.
            if not self._uses_alt_screen:
                console = get_console(stderr=self.stderr)
                console.file.write("\x1b[J")
                console.file.flush()
        else:
            print_renderable(renderable, stderr=self.stderr)
        self._last_rendered_state = self._state_snapshot()

    def _renderable_exceeds_console(self, renderable: RenderableType) -> bool:
        console = get_console(stderr=self.stderr)
        height = getattr(console.size, "height", 0) or 0
        if height <= 0:
            return False
        try:
            lines = console.render_lines(renderable, pad=False)
        except Exception:
            return False
        # Leave a small margin so a refresh that fits exactly is still allowed
        # to repaint in place.
        return len(lines) > max(1, height - 1)

    def _alt_screen_disabled(self) -> bool:
        """Honor an env-var opt-out so users can preserve terminal scrollback.

        Setting ``BOXMOT_NO_ALT_SCREEN=1`` (or ``true``/``yes``/``on``) keeps
        Rich on the normal screen even when the live renderable is taller
        than the console, at the cost of writing successive frames to
        scrollback instead of updating in place.
        """
        value = os.environ.get("BOXMOT_NO_ALT_SCREEN", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _write_cursor_visibility(self, visible: bool) -> None:
        console = get_console(stderr=self.stderr)
        if not getattr(console, "is_terminal", False):
            return
        sequence = "\x1b[?25h" if visible else "\x1b[?25l"
        try:
            console.file.write(sequence)
            console.file.flush()
        except Exception:
            return

    def _start_live(self, *, screen: bool) -> None:
        if screen and self._alt_screen_disabled():
            screen = False
        self._live = Live(
            self._live_renderable(),
            console=get_console(stderr=self.stderr),
            transient=self.transient,
            auto_refresh=False,
            vertical_overflow="visible",
            screen=screen,
        )
        self._live.start(refresh=True)
        self._write_cursor_visibility(False)
        self._uses_alt_screen = screen

    def start(self) -> WorkflowProgress:
        if self._started:
            return self

        self._started = True
        self._compact_layout = False
        self._oversized = self._renderable_exceeds_console(self._live_renderable())
        # Use the alternate screen only when the live renderable itself still
        # doesn't fit. If the compact live layout fits, stay on the normal
        # screen so scrollback remains available while the command runs.
        self._start_live(screen=self.prefer_alt_screen or self._oversized)
        self._last_rendered_state = self._state_snapshot()
        return self

    def stop(self) -> None:
        if not self._started:
            return

        used_alt_screen = self._uses_alt_screen
        final_renderable = self._live_renderable()
        if self._live is not None:
            if self._last_rendered_state != self._state_snapshot():
                self._live.update(final_renderable, refresh=False)
                self._last_rendered_state = self._state_snapshot()
            self._live.stop()
            self._live = None
            self._write_cursor_visibility(True)
        self._started = False
        self._uses_alt_screen = False

        # The alternate screen buffer is restored on ``Live.stop``; print the
        # final renderable into the regular scrollback so the user keeps a
        # static copy of the completed workflow.
        if used_alt_screen and not self.transient:
            print_renderable(final_renderable, stderr=self.stderr)

    def _update_live(self, *, render: bool = True, force: bool = False) -> None:
        if not (render and self._started):
            return
        if not force and self._last_rendered_state == self._state_snapshot():
            return
        renderable = self._live_renderable()
        oversized = self._renderable_exceeds_console(renderable)

        # If the compact live renderable still doesn't fit, switch to the
        # alternate screen so in-place updates don't pollute scrollback.
        if oversized and not self._uses_alt_screen and self._live is not None and not self._alt_screen_disabled():
            # Erase the in-place render before swapping to the alternate
            # screen so the user doesn't end up with both the previous frame
            # (persisted to scrollback) and the alt-screen frame (re-printed
            # on stop) stacked on top of each other.
            self._live.transient = True
            self._live.stop()
            self._live = None
            self._start_live(screen=True)
            self._last_rendered_state = self._state_snapshot()
            return

        # If the renderable now fits but we are on the alternate screen,
        # switch back to the normal screen so terminal scrollback is
        # accessible again (e.g. after KF tuning output shrinks to the
        # compact Optimize progress bar).
        if not oversized and self._uses_alt_screen and self._live is not None:
            self._live.transient = True
            self._live.stop()
            self._live = None
            self._start_live(screen=False)
            self._last_rendered_state = self._state_snapshot()
            return

        self._oversized = oversized
        if self._live is not None:
            self._live.update(renderable, refresh=True)
        else:
            print_renderable(renderable, stderr=self.stderr)
        self._last_rendered_state = self._state_snapshot()

    def activate(self, label: str, *, render: bool = True) -> None:
        updated: list[tuple[str, StepState]] = []
        for step_label, step_state in self.steps:
            if step_label == label:
                updated.append((step_label, "active"))
            elif step_state == "active":
                updated.append((step_label, "todo"))
            else:
                updated.append((step_label, step_state))
        if updated == self.steps:
            return
        self.steps = updated
        self._update_live(render=render)

    def complete(self, label: str, *, render: bool = True) -> None:
        updated = [(step_label, "done" if step_label == label else step_state) for step_label, step_state in self.steps]
        if updated == self.steps:
            return
        self.steps = updated
        self._update_live(render=render)

    def fail(
        self,
        label: str | None = None,
        error: str | BaseException | None = None,
        *,
        render: bool = True,
    ) -> None:
        target = label
        if target is None:
            for step_label, step_state in self.steps:
                if step_state == "active":
                    target = step_label
                    break
        if target is None:
            for step_label, step_state in reversed(self.steps):
                if step_state != "done":
                    target = step_label
                    break
        if target is None and self.steps:
            target = self.steps[-1][0]

        if target is not None:
            self.steps = [
                (step_label, "failed" if step_label == target else step_state) for step_label, step_state in self.steps
            ]

        if error is not None:
            if isinstance(error, BaseException):
                import traceback

                error_text = "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip()
            else:
                error_text = str(error)
            self.detail_title = f"{target} failed" if target else "Pipeline failed"
            self.detail_text = error_text
            self.detail_renderable = None

        self._update_live(render=render)

    def set_detail(self, title: str | None, text: str | None, *, render: bool = True) -> None:
        new_title = None if title is None else str(title)
        new_text = None if text is None else str(text)
        if new_title == self.detail_title and new_text == self.detail_text and self.detail_renderable is None:
            return
        self.detail_title = new_title
        self.detail_text = new_text
        self.detail_renderable = None
        self._update_live(render=render)

    def set_detail_renderable(
        self,
        title: str | None,
        renderable: RenderableType | None,
        *,
        render: bool = True,
    ) -> None:
        new_title = None if title is None else str(title)
        if new_title == self.detail_title and renderable is self.detail_renderable and self.detail_text is None:
            return
        self.detail_title = new_title
        self.detail_text = None
        self.detail_renderable = renderable
        self._update_live(render=render)

    def clear_detail(self, *, render: bool = True) -> None:
        self.detail_title = None
        self.detail_text = None
        self.detail_renderable = None
        self._update_live(render=render)

    def set_fields(
        self,
        fields: Sequence[tuple[str, object]],
        *,
        render: bool = True,
    ) -> None:
        updated = list(fields)
        if list(self.fields) == updated:
            return
        self.fields = updated
        self._update_live(render=render)

    def transition(
        self,
        done: str,
        next_step: str,
        detail: str | None = None,
    ) -> None:
        """Complete *done*, activate *next_step*, optionally set detail text.

        Batches the state change into a single repaint.
        """
        self.complete(done, render=False)
        self.activate(next_step, render=False)
        if detail:
            self.set_detail(next_step, detail)
        else:
            self._update_live(render=True)

    # -- context manager protocol ------------------------------------------

    def __enter__(self) -> WorkflowProgress:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_val is not None:
            self.fail(error=exc_val)
        self.stop()


def create_workflow_progress(
    title: str,
    fields: Sequence[tuple[str, object]],
    *,
    steps: Sequence[tuple[str, StepState]] = (),
    stderr: bool = False,
    transient: bool = False,
) -> WorkflowProgress:
    return WorkflowProgress(
        title=title,
        fields=fields,
        steps=list(steps),
        stderr=stderr,
        transient=transient,
    )


__all__ = (
    "BOXMOT_THEME",
    "BOXMOT_THEME_STYLES",
    "STEP_DONE_MARKER",
    "STYLE_ACCENT",
    "STYLE_BORDER",
    "STYLE_BORDER_DETAIL",
    "STYLE_BORDER_FAILED",
    "STYLE_BORDER_OUTER",
    "STYLE_COMBINED_ROW",
    "STYLE_LABEL",
    "STYLE_MUTED",
    "STYLE_RULE",
    "STYLE_STATUS_ACTIVE",
    "STYLE_STATUS_DONE",
    "STYLE_STATUS_FAILED",
    "STYLE_STATUS_TODO",
    "STYLE_SUBTLE",
    "STYLE_TABLE_HEADER",
    "STYLE_TEXT",
    "STYLE_TEXT_STRONG",
    "STYLE_TITLE",
    "STYLE_TITLE_MAIN",
    "StepState",
    "WorkflowProgress",
    "build_checklist",
    "build_workflow_intro",
    "capture_renderable",
    "create_workflow_progress",
    "get_console",
    "print_renderable",
    "print_text",
)
