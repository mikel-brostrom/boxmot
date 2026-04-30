from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
import os
from typing import Literal, Sequence

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

StepState = Literal["done", "active", "todo", "failed"]
_WORKFLOW_PANEL_PREFIX = "__panel__:"

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
    "Setup evaluation environment": "Setup",
    "Generate detections and embeddings": "Generate",
    "Run tracker": "Track",
    "Evaluate results": "Evaluate",
    "Optimize trials": "Optimize",
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
        if label_text.startswith(_WORKFLOW_PANEL_PREFIX):
            title = label_text[len(_WORKFLOW_PANEL_PREFIX):].strip() or "Details"
            items = _normalize_panel_items(value)
            if items:
                extra_panels.append((title, items))
                panel_labels.update(item_label for item_label, _ in items)
            continue
        primary_fields.append((label_text, value))

    if extra_panels:
        primary_fields = [
            (label, value)
            for label, value in primary_fields
            if label not in panel_labels and label != "Image size"
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


def _build_setup_section_table(
    section_title: str,
    fields: Sequence[tuple[str, object]],
    *,
    compact: bool = False,
) -> Table:
    pair_count = 3 if compact else 2
    table = Table.grid(expand=True, padding=(0, 1))

    if compact and pair_count == 3:
        table.add_column(style=STYLE_ACCENT, no_wrap=True, width=8, overflow="ellipsis")
        for offset in range(pair_count):
            table.add_column(style=STYLE_MUTED, no_wrap=True, width=10, overflow="ellipsis")
            table.add_column(
                style=STYLE_TEXT,
                no_wrap=True,
                ratio=2 if offset == pair_count - 1 else 1,
                overflow="ellipsis",
            )
    else:
        table.add_column(style=STYLE_ACCENT, no_wrap=True)
        for offset in range(pair_count):
            if offset == pair_count - 1:
                table.add_column(style=STYLE_MUTED, no_wrap=True)
                table.add_column(style=STYLE_TEXT, ratio=1)
            else:
                table.add_column(style=STYLE_MUTED, no_wrap=True)
                table.add_column(style=STYLE_TEXT, ratio=1)

    title_text = _compact_section_title(section_title) if compact else section_title.upper()

    for index in range(0, len(fields), pair_count):
        row: list[RenderableType] = [
            Text(title_text, style=STYLE_ACCENT, no_wrap=True, overflow="ellipsis")
            if index == 0
            else Text("")
        ]
        for offset in range(pair_count):
            item_index = index + offset
            if item_index < len(fields):
                label, value = fields[item_index]
                label_text = (
                    _compact_setup_label(str(label)) if compact and pair_count == 3 else str(label).upper()
                )
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
) -> Panel | None:
    sections = [("Configuration", list(primary_fields)), *[(title, list(fields)) for title, fields in extra_panels]]
    sections = [(title, fields) for title, fields in sections if fields]
    if not sections:
        return None

    return Panel(
        Group(*[_build_setup_section_table(title, fields, compact=compact) for title, fields in sections]),
        title=Text("Setup", style=STYLE_TITLE),
        border_style=STYLE_BORDER_OUTER,
        padding=(0, 1),
    )


def _build_steps_panel(steps: Sequence[tuple[str, StepState]], *, compact: bool = False) -> Panel | None:
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
    return Panel(
        body,
        title=Text("Pipeline", style=STYLE_TITLE),
        border_style=STYLE_BORDER_FAILED if has_failure else STYLE_BORDER,
        padding=(0, 1),
    )


def _build_detail_panel(
    detail_title: str | None,
    detail_text: str | None,
    detail_renderable: RenderableType | None,
    *,
    failed: bool = False,
) -> Panel | None:
    if detail_renderable is None and not detail_text:
        return None

    content: RenderableType = detail_renderable if detail_renderable is not None else _decode_ansi(detail_text or "")
    title_style = STYLE_STATUS_FAILED if failed else STYLE_TITLE
    title = Text(detail_title or "Live Detail", style=title_style)
    return Panel(
        content,
        title=title,
        border_style=STYLE_BORDER_FAILED if failed else STYLE_BORDER_DETAIL,
        padding=(0, 1),
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
        detail_title, detail_text, detail_renderable, failed=has_failure
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
    _oversized: bool = field(default=False, init=False, repr=False)
    _uses_alt_screen: bool = field(default=False, init=False, repr=False)
    _compact_layout: bool = field(default=False, init=False, repr=False)
    _last_rendered_state: tuple[
        tuple[tuple[str, StepState], ...],
        str | None,
        str | None,
        RenderableType | None,
    ] | None = field(default=None, init=False, repr=False)

    def renderable(self, *, compact: bool = False) -> Panel:
        return build_workflow_intro(
            self.title,
            self.fields,
            steps=self.steps,
            detail_title=self.detail_title,
            detail_text=self.detail_text,
            detail_renderable=self.detail_renderable,
            compact=compact,
        )

    def _live_renderable(self) -> Panel:
        full = self.renderable(compact=False)
        if self.prefer_compact_layout:
            self._compact_layout = True
            return self.renderable(compact=True)
        if self._compact_layout or self._renderable_exceeds_console(full):
            self._compact_layout = True
            return self.renderable(compact=True)
        return full

    def _state_snapshot(
        self,
    ) -> tuple[tuple[tuple[str, StepState], ...], str | None, str | None, RenderableType | None]:
        return tuple(self.steps), self.detail_title, self.detail_text, self.detail_renderable

    def _render_snapshot(self) -> None:
        renderable = self._live_renderable()
        if self._live is not None:
            self._live.update(renderable, refresh=True)
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

    def _start_live(self, *, screen: bool) -> None:
        self._live = Live(
            self._live_renderable(),
            console=get_console(stderr=self.stderr),
            transient=self.transient,
            auto_refresh=False,
            vertical_overflow="visible",
            screen=screen,
        )
        self._live.start(refresh=True)
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
        self._started = False
        self._uses_alt_screen = False

        # The alternate screen buffer is restored on ``Live.stop``; print the
        # final renderable into the regular scrollback so the user keeps a
        # static copy of the completed workflow.
        if used_alt_screen and not self.transient:
            print_renderable(final_renderable, stderr=self.stderr)

    def _update_live(self, *, render: bool = True) -> None:
        if not (render and self._started):
            return
        if self._last_rendered_state == self._state_snapshot():
            return
        renderable = self._live_renderable()
        oversized = self._renderable_exceeds_console(renderable)

        # If the compact live renderable still doesn't fit, switch to the
        # alternate screen so in-place updates don't pollute scrollback.
        if oversized and not self._uses_alt_screen and self._live is not None:
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
        updated = [
            (step_label, "done" if step_label == label else step_state)
            for step_label, step_state in self.steps
        ]
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
                (step_label, "failed" if step_label == target else step_state)
                for step_label, step_state in self.steps
            ]

        if error is not None:
            if isinstance(error, BaseException):
                import traceback

                error_text = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ).rstrip()
            else:
                error_text = str(error)
            self.detail_title = f"{target} failed" if target else "Pipeline failed"
            self.detail_text = error_text
            self.detail_renderable = None

        self._update_live(render=render)

    def set_detail(self, title: str | None, text: str | None, *, render: bool = True) -> None:
        new_title = None if title is None else str(title)
        new_text = None if text is None else str(text)
        if (
            new_title == self.detail_title
            and new_text == self.detail_text
            and self.detail_renderable is None
        ):
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
        if (
            new_title == self.detail_title
            and renderable is self.detail_renderable
            and self.detail_text is None
        ):
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
