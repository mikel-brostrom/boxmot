"""Rich terminal application for selecting, reviewing, and launching commands."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import click
from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from boxmot.engine.ui.core.ui import get_console
from boxmot.engine.ui.keyboard import TerminalInput, is_interactive
from boxmot.engine.ui.launcher.model import (
    COMMAND_LABELS,
    COMMANDS,
    Choice,
    FormField,
    LauncherModel,
    display_value,
    field_choices,
)

# Terminal-cell renditions of the bundled assets/logo_white.png alpha mask.
# Keep them as text so opening the launcher never loads an image or ML runtime.
_LOGO = """ ⢀⣤⠶⠛⠶⣤⡀
⣾⢯⣀   ⣠⠉⣷ ⢰⢀⡀  ⡀     ⣀ ⡀  ⡀ ⣠⡀
⣿ ⣍⠛⣠⠞⠁ ⣿ ⢸⡋⢹⣶⡏⢙⡧⢻⣞⠁⣿⢹⡏⢹⣧⡟⠙⣷⢹⡋
⢿⣀⠈⠛⣿  ⣠⡿ ⠘⠛⠋ ⠙⠛⠑⠋⠘⠃⠋⠈⠃⠘⠃⠙⠛⠁⠘⠛
 ⠈⠛⠶⣿⠶⠛⠁"""
_LOGO_ICON = "⣠⠖⠋⠓⢤⡀\n⡏⢽⣢⠞ ⡇\n⠑⠦⣽⡤⠚⠁"


@dataclass
class Editor:
    """Searchable choice list or text editor, independent of terminal input."""

    name: str
    label: str
    help: str
    choices: list[Choice]
    allow_custom: bool = True
    multiple: bool = False
    query: str = ""
    cursor: int = 0
    selected: list[str] = dataclass_field(default_factory=list)
    selection_changed: bool = False

    def rows(self) -> list[tuple[str | None, str]]:
        """Return reset, matching catalog choices, and the custom text action."""
        rows: list[tuple[str | None, str]] = []
        if self.name != "command":
            rows.append((None, "Use CLI default"))
        query = self.query.casefold()
        rows.extend(
            (choice.value, f"{choice.value}  ·  {choice.detail}" if choice.detail else choice.value)
            for choice in self.choices
            if query in f"{choice.value} {choice.detail}".casefold()
        )
        if self.allow_custom and self.query and not any(value == self.query for value, _ in rows):
            rows.append((self.query, f"Use value: {self.query}"))
        return rows

    def update_query(self, query: str) -> None:
        """Reset selection to the first match, or the custom value."""
        self.query = query
        self.cursor = min(1 if self.name != "command" and query else 0, max(0, len(self.rows()) - 1))

    def handle_key(self, key: str) -> tuple[bool, str | None]:
        """Edit or select; return (accepted, value), with None meaning default."""
        rows = self.rows()
        if key in {"up", "down", "pageup", "pagedown"}:
            delta = {"up": -1, "down": 1, "pageup": -8, "pagedown": 8}[key]
            self.cursor = max(0, min(len(rows) - 1, self.cursor + delta))
        elif key == "home":
            self.cursor = 0
        elif key == "end":
            self.cursor = max(0, len(rows) - 1)
        elif key == "backspace":
            self.update_query(self.query[:-1])
        elif key in {"ctrl+u", "\x15"}:
            self.update_query("")
        elif key == " " and self.multiple and self.choices and rows:
            self.selection_changed = True
            value = rows[self.cursor][0]
            if value is None:
                self.selected.clear()
            elif value in self.selected:
                self.selected.remove(value)
            else:
                self.selected.append(value)
        elif key == "enter" and rows:
            value = rows[min(self.cursor, len(rows) - 1)][0]
            if self.multiple and self.choices and self.selection_changed and value is not None:
                value = shlex.join(self.selected) or None
            return True, value
        elif len(key) == 1 and key.isprintable():
            self.update_query(self.query + key)
        return False, None


def _panel(content: RenderableType, title: str, *, accent: bool = False, hint: str | None = None) -> Panel:
    """Use a consistent, compact visual style throughout the launcher."""
    return Panel(
        content,
        title=Text(title, style="bold cyan" if accent else "bold"),
        subtitle=Text(hint, style="dim") if hint else None,
        border_style="cyan" if accent else "bright_black",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _header(workflow: str, width: int, height: int) -> tuple[Panel, int]:
    """Show the logo at a size that leaves room for configuration and controls."""
    if width < 60 or height < 26:
        content = Text.assemble(("BoxMOT", "bold cyan"), "  /  ", (workflow, "bold"))
        content.no_wrap = True
        content.overflow = "ellipsis"
        return _panel(content, "Command launcher"), 3
    full_logo = width >= 80 and height >= 32
    logo = Text(_LOGO if full_logo else _LOGO_ICON, style="cyan", no_wrap=True)
    table = Table.grid(padding=(0, 3), expand=True)
    table.add_column(width=30 if full_logo else 6)
    table.add_column(ratio=1)
    table.add_row(
        logo,
        Align(
            Group(Text("BoxMOT", style="bold cyan"), Text(workflow, style="bold", no_wrap=True, overflow="ellipsis")),
            vertical="middle",
            height=5 if full_logo else 3,
        ),
    )
    return _panel(table, "Command launcher"), 7 if full_logo else 5


def _summary(model: LauncherModel) -> RenderableType:
    """Render current settings or resolved experiment components for review."""
    try:
        experiment = model.resolved_experiment()
    except (OSError, ValueError, TypeError) as error:
        return Text(str(error), style="yellow", overflow="fold")
    if experiment is None:
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="dim", no_wrap=True)
        table.add_column(ratio=1, overflow="fold")
        for field in model.fields:
            value = Text(model.field_display(field.name) or ("Required" if field.option.required else "Not set"))
            if field.name not in model.values:
                value.append("  · default", style="dim")
            table.add_row(Text(field.label), value)
        if model.name == "track":
            table.add_row(
                "Geometry", Text("AABB (prompts)" if model.value("tracker") == "edgetam" else "Automatic from detector")
            )
        return Group(Text("Settings", style="bold cyan"), table)
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(ratio=1, overflow="fold")
    dataset = experiment["dataset"]
    detector = experiment.get("detector") or {}
    reid = experiment.get("reid") or {}
    table.add_row("Dataset", str(dataset["id"]))
    table.add_row("Split", str(dataset["split"]))
    table.add_row("Detector", str(detector.get("id") or "Saved inputs"))
    if detector.get("checkpoint"):
        table.add_row("Checkpoint", str(detector["checkpoint"]))
    table.add_row("ReID", str(reid.get("id") or "None"))
    if reid.get("model"):
        table.add_row("Weights", str(reid["model"]))
    if experiment.get("segmentor"):
        table.add_row("Segmentor", str(experiment["segmentor"]))
    for mapping in experiment.get("evaluation", {}).get("classes", []):
        if mapping.get("detector_name"):
            table.add_row("Class map", f"{mapping['name']} → {mapping['detector_name']}")
    table.add_row("Source", str(experiment["source_path"]))
    return Group(table, Text("\nModels are supplied by this experiment.", style="dim"))


class LauncherApp:
    """Keyboard state and responsive rendering for a launcher session."""

    def __init__(self, model: LauncherModel) -> None:
        self.model = model
        self.cursor = 0
        self.editor: Editor | None = None
        self.error = ""
        self.review = False
        self.finished = False
        self.result: list[str] | None = None
        self._validated_argv: list[str] | None = None
        self.review_offset = 0

    @property
    def stage(self) -> str:
        """Derive the active stage from the focused field, including navigation rows."""
        fields = self.model.fields
        if self.cursor <= 0:
            return self.model.stages[0]
        if self.cursor > len(fields):
            return self.model.stages[-1]
        return fields[self.cursor - 1].group

    def _select_stage(self, index: int) -> None:
        """Focus the first row of a stage without changing any configured values."""
        stages = self.model.stages
        if not 0 <= index < len(stages):
            return
        self.cursor = (
            0
            if index == 0
            else next(i + 1 for i, field in enumerate(self.model.fields) if field.group == stages[index])
        )

    def _set_value(self, name: str, value: str | bool | None) -> None:
        """Keep focus on the edited field when its value changes visible stages."""
        self.model.set_value(name, value)
        fields = self.model.fields
        self.cursor = next(
            (i + 1 for i, field in enumerate(fields) if field.name == name), min(self.cursor, len(fields) + 1)
        )

    def _stage_bar(self, width: int) -> Text:
        """Show workflow order, keeping the active stage legible in narrow terminals."""
        stages = self.model.stages
        active = self.stage
        bar = Text(no_wrap=True, overflow="ellipsis")
        for index, name in enumerate(stages):
            if index:
                bar.append(" → ", style="dim")
            bar.append(
                f" {index + 1} {name} ", style="bold black on cyan" if name == active and not self.review else "dim"
            )
        if self.review:
            bar.append(" → ", style="dim")
            bar.append(" Review ", style="bold black on cyan")
        if bar.cell_len > width:
            if self.review:
                bar = Text.assemble((" Review ", "bold cyan"), (f" · Esc back to {active}", "dim"))
            else:
                bar = Text.assemble(
                    (f" {stages.index(active) + 1}/{len(stages)} ", "dim"),
                    (active, "bold cyan"),
                    ("  · Tab next · Shift+Tab back", "dim"),
                )
            bar.no_wrap = True
            bar.overflow = "ellipsis"
        return bar

    def _begin_edit(self, field: FormField | None) -> None:
        """Open a searchable selector, or immediately toggle a boolean field."""
        self.error = ""
        if field is None:
            self.editor = Editor(
                "command",
                "Choose a command",
                "Type to search the common workflows.",
                [Choice(name, COMMAND_LABELS[name]) for name in COMMANDS],
                allow_custom=False,
                cursor=COMMANDS.index(self.model.name),
            )
            return
        option = field.option
        if option.is_bool_flag:
            current = bool(self.model.value(field.name))
            value = not current
            if not value and not option.secondary_opts:
                self._set_value(field.name, None)
            else:
                self._set_value(field.name, value)
            return
        choices = field_choices(self.model.name, option)
        self.editor = Editor(
            field.name,
            field.label,
            (option.help or "")
            + ("  Multiple values: separate with spaces; quote paths containing spaces." if option.multiple else ""),
            choices,
            allow_custom=not isinstance(option.type, click.Choice),
            multiple=option.multiple,
        )
        current = display_value(self.model.value(field.name))
        if choices:
            if option.multiple:
                self.editor.selected = shlex.split(current)
            self.editor.cursor = next(
                (index for index, (value, _) in enumerate(self.editor.rows()) if value == current), 0
            )
        else:
            self.editor.update_query(current)

    def _prepare_run(self) -> None:
        """Validate the configuration before displaying the explicit Run action."""
        try:
            self._validated_argv = self.model.validate()
        except (click.ClickException, OSError, ValueError, TypeError) as error:
            self.error = error.format_message() if isinstance(error, click.ClickException) else str(error)
            if isinstance(error, click.BadParameter) and error.param is not None:
                self.cursor = next(
                    (index + 1 for index, field in enumerate(self.model.fields) if field.name == error.param.name),
                    self.cursor,
                )
            return
        self.error = ""
        self.review = True
        self.review_offset = 0

    def handle_key(self, key: str) -> None:
        """Apply one key without performing model loading or workflow execution."""
        if key in {"ctrl+c", "\x03"}:
            raise KeyboardInterrupt
        if key == "eof":
            self.finished = True
            return
        if self.editor is not None:
            if key == "escape" or (key == "left" and self.editor.choices):
                self.editor = None
                return
            accepted, value = self.editor.handle_key(key)
            if accepted:
                if self.editor.name == "command":
                    assert value is not None
                    self.model.select_command(value)
                    self.cursor = 0
                else:
                    if value and self.model.options[self.editor.name].multiple:
                        try:
                            shlex.split(value)
                        except ValueError as error:
                            self.error = str(error) + ". Finish the quoted value, or press Esc to cancel."
                            return
                    self._set_value(self.editor.name, value)
                self.error = ""
                self.editor = None
            return
        if self.review:
            if key == "escape":
                self.review = False
            elif key == "enter":
                self.result = self._validated_argv
                self.finished = True
            elif key in {"up", "down", "pageup", "pagedown"}:
                delta = {"up": -1, "down": 1, "pageup": -8, "pagedown": 8}[key]
                self.review_offset = max(0, self.review_offset + delta)
            elif key == "home":
                self.review_offset = 0
            elif key == "end":
                self.review_offset = 1_000_000
            return
        fields = self.model.fields
        self.cursor = max(0, min(self.cursor, len(fields) + 1))
        if key in {"escape", "q"}:
            self.finished = True
        elif key in {"tab", "shift+tab"}:
            self._select_stage(self.model.stages.index(self.stage) + (1 if key == "tab" else -1))
        elif len(key) == 1 and key in "123456789":
            self._select_stage(int(key) - 1)
        elif key in {"up", "down", "pageup", "pagedown"}:
            delta = {"up": -1, "down": 1, "pageup": -8, "pagedown": 8}[key]
            self.cursor = max(0, min(len(fields) + 1, self.cursor + delta))
        elif key == "home":
            self.cursor = 0
        elif key == "end":
            self.cursor = len(fields) + 1
        elif key == "r" or (key == "enter" and self.cursor == len(fields) + 1):
            self._prepare_run()
        elif key == "enter":
            self._begin_edit(fields[self.cursor - 1] if self.cursor else None)
        elif key == "right" and self.cursor <= len(fields):
            field = fields[self.cursor - 1] if self.cursor else None
            if field is None or (not field.option.is_bool_flag and field_choices(self.model.name, field.option)):
                self._begin_edit(field)
        elif key in {"d", "delete", "backspace"} and 0 < self.cursor <= len(fields):
            self._set_value(fields[self.cursor - 1].name, None)
            self.error = ""

    def _form(self, height: int, *, inline_help: bool = True) -> RenderableType:
        """Show only the active stage, keeping its selected field in view."""
        fields = self.model.fields
        self.cursor = min(self.cursor, len(fields) + 1)
        stage = self.stage
        rows: list[tuple[int, str, str]] = []
        if stage == self.model.stages[0]:
            rows.append((0, "Command", self.model.name))
        for index, field in enumerate(fields, start=1):
            if field.group != stage:
                continue
            value = self.model.field_display(field.name)
            value = value or ("Required" if field.option.required else "Not set")
            suffix = "" if field.name in self.model.values else "  · default"
            rows.append((index, field.label, value + suffix))
        if stage == self.model.stages[-1]:
            rows.append((len(fields) + 1, "Review and run", "Enter to review"))
        position = next(i for i, (index, _, _) in enumerate(rows) if index == self.cursor)
        visible = max(1, height - (5 if inline_help else 3))
        start = max(0, min(position - visible // 2, len(rows) - visible))
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=1)
        table.add_column(ratio=2, overflow="ellipsis", no_wrap=True)
        table.add_column(ratio=3, overflow="ellipsis", no_wrap=True)
        for index, label, value in rows[start : start + visible]:
            style = ("bold cyan" if self.editor else "bold black on cyan") if index == self.cursor else ""
            table.add_row(
                Text("›" if index == self.cursor else " ", style=style),
                Text(label, style=style),
                Text(value, style=style),
            )
        help_text = "Choose which workflow to configure."
        if 0 < self.cursor <= len(fields):
            help_text = fields[self.cursor - 1].option.help or ""
        elif self.cursor == len(fields) + 1:
            help_text = "Review the exact command and start it with Enter."
        content: list[RenderableType] = [
            Text(f"{stage}  ·  {position + 1}/{len(rows)}", style="cyan"),
            table,
        ]
        if inline_help:
            content.extend((Text(""), Text(help_text, style="dim", overflow="ellipsis", no_wrap=True)))
        return _panel(Group(*content), "Configure", accent=self.editor is None)

    def _details(self) -> RenderableType:
        """Explain the selected field using the command's own option metadata."""
        fields = self.model.fields
        if self.cursor == 0:
            return _panel(
                Group(
                    Text("Command", style="bold cyan"),
                    Text("Switch workflows; each keeps your settings."),
                    Text(""),
                    Text("The preview includes your edits. Other settings use CLI defaults.", style="dim"),
                ),
                "Details",
                hint="→ / Enter choose command",
            )
        if self.cursor > len(fields):
            return _panel(
                Group(
                    Text("Review and run", style="bold cyan"),
                    Text("Review the full command and settings before starting."),
                    Text(""),
                    Text("Enter open review · Enter again to run", style="cyan"),
                    Text("Esc returns from review to this form.", style="dim"),
                ),
                "Details",
            )
        field = fields[self.cursor - 1]
        option = field.option
        heading = Text.assemble((field.label, "bold cyan"), (f"  {option.opts[0]}", "dim"))
        if field.name == "experiment" and self.model.values.get("experiment"):
            return _panel(Group(heading, Text(""), _summary(self.model)), "Details", hint="→ / Enter choose · d reset")

        empty = "Required" if option.required else "Not set"
        current = Text(self.model.field_display(field.name) or empty, overflow="fold")
        edited = field.name in self.model.values
        current.append("  · edited" if edited else "  · CLI default", style="dim")
        values = Table.grid(padding=(0, 1), expand=True)
        values.add_column(style="dim", no_wrap=True)
        values.add_column(ratio=1, overflow="fold")
        values.add_row("Current", current)
        if edited:
            values.add_row("Default", Text(display_value(self.model.default(field.name)) or empty))
        if self.model.name == "track" and field.name in {"detector", "tracker", "tracker_backend"}:
            geometry = "AABB (prompts)" if self.model.value("tracker") == "edgetam" else "Automatic from detector"
            values.add_row("Geometry", Text(geometry))
        if option.is_bool_flag:
            action = "Enter toggle · d reset"
        elif isinstance(option.type, click.Choice):
            action = "→ / Enter choose · d reset"
        elif field_choices(self.model.name, option):
            action = "→ / Enter browse · d reset"
        else:
            action = "Enter edit · d reset"
        return _panel(
            Group(heading, Text(""), values, Text(""), Text(option.help or "")),
            "Details",
            hint=action,
        )

    def _editor_view(self, height: int) -> RenderableType:
        """Render a filtered picker with explicit default and custom actions."""
        assert self.editor is not None
        editor = self.editor
        rows = editor.rows()
        editor.cursor = min(editor.cursor, max(0, len(rows) - 1))
        visible = max(1, height - 7)
        start = max(0, min(editor.cursor - visible // 2, len(rows) - visible))
        table = Table.grid(expand=True)
        table.add_column(width=3, min_width=3, no_wrap=True)
        table.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        for index in range(start, min(len(rows), start + visible)):
            value, label = rows[index]
            marker = "✓" if value in editor.selected else " "
            if index == editor.cursor:
                marker = "›" + marker
            style = "bold black on cyan" if index == editor.cursor else ""
            table.add_row(Text(marker, style=style), Text(label, style=style))
        if not rows:
            table.add_row("", "No matches")
        hint = "Ctrl+U clear"
        if editor.multiple and editor.choices:
            hint = "Space toggle · Ctrl+U clear"
        query = Text.assemble(("Search / value  ", "cyan"), editor.query, ("▏", "cyan"))
        query.no_wrap = True
        query.overflow = "ellipsis"
        return _panel(
            Group(
                Text(editor.help, style="dim", overflow="ellipsis", no_wrap=True),
                query,
                Text(""),
                table,
                Text(""),
                Text(hint, style="dim"),
            ),
            editor.label,
            accent=True,
        )

    def render(self, console: Console) -> Layout:
        """Build a stable full-screen view that adapts to terminal dimensions."""
        width, height = console.size
        layout = Layout()
        preview_height = min(7, max(4, height // 6))
        header, header_height = _header(COMMAND_LABELS[self.model.name], width, height)
        stage_height = 1 if height < 24 else 2
        layout.split_column(
            Layout(name="header", size=header_height),
            Layout(name="stages", size=stage_height),
            Layout(name="body", ratio=1, minimum_size=7),
            Layout(name="preview", size=preview_height),
            Layout(name="footer", size=2),
        )
        layout["header"].update(header)
        layout["stages"].update(self._stage_bar(width))
        body_height = max(7, height - preview_height - header_height - stage_height - 2)
        if self.review:
            layout["body"].update(self._review_view(console, body_height))
        elif width >= 110 and (self.editor is None or self.editor.choices):
            layout["body"].split_row(
                Layout(name="form", ratio=3), Layout(Text(" "), size=1), Layout(name="details", ratio=2)
            )
            layout["form"].update(self._form(body_height, inline_help=False))
            layout["details"].update(self._editor_view(body_height) if self.editor else self._details())
        elif self.editor is not None:
            layout["body"].update(self._editor_view(body_height))
        else:
            layout["body"].update(self._form(body_height))
        try:
            preview = self.model.preview()
        except ValueError as error:
            preview = str(error)
        content = Text(self.error or preview, style="yellow" if self.error else "green", overflow="fold")
        layout["preview"].update(_panel(content, "Check configuration" if self.error else "Command preview"))
        footer = "↑↓ fields   Tab / Shift+Tab stages   1–9 jump   Enter edit   → choices   d reset   r review   q quit"
        if width < 60:
            footer = "↑↓ fields  Tab stages  Enter edit  → choices  d reset  r review  q quit"
        if self.review:
            footer = "Enter RUN command   ↑↓ / PgUp/PgDn scroll review   Esc back"
        elif self.editor:
            back = "← / Esc" if self.editor.choices else "Esc"
            footer = f"Type to search or enter a value   ↑↓ choose   Enter apply   {back} back"
        layout["footer"].update(Text(footer, style="dim", overflow="fold"))
        return layout

    def _review_view(self, console: Console, height: int) -> RenderableType:
        """Keep the complete command and resolved setup accessible by scrolling."""
        content = Group(
            Text("Ready to run", style="bold green"),
            Text("Press Enter to start, or Esc to keep editing.\n"),
            Text(self.model.preview(), style="green", overflow="fold"),
            Text(""),
            _summary(self.model),
        )
        lines = console.render_lines(content, console.options.update(width=max(10, console.width - 4)), pad=False)
        visible = max(1, height - 4)
        self.review_offset = min(self.review_offset, max(0, len(lines) - visible))
        window = Text()
        for line in lines[self.review_offset : self.review_offset + visible]:
            for segment in line:
                if not segment.control:
                    window.append(segment.text, segment.style)
            window.append("\n")
        if len(lines) > visible:
            window.append(
                f"Lines {self.review_offset + 1}–{min(len(lines), self.review_offset + visible)} / {len(lines)}"
                "  ·  ↑↓ scroll",
                style="dim",
            )
        return _panel(window, "Review", accent=True)


def run_launcher(
    root_command: click.Group,
    *,
    initial_command: str | None = None,
    experiment: str | None = None,
) -> list[str] | None:
    """Return reviewed command arguments, or None when the user cancels."""
    console = get_console()
    if not is_interactive(console):
        raise click.UsageError("The launcher needs an interactive terminal. Use boxmot COMMAND --help for CLI options.")
    app = LauncherApp(LauncherModel(root_command, initial_command=initial_command, experiment=experiment))
    with (
        TerminalInput() as keyboard,
        Live(app.render(console), console=console, screen=True, auto_refresh=False, transient=True) as live,
    ):
        size = console.size
        while not app.finished:
            key = keyboard.read_key(timeout=0.1)
            if key is not None:
                app.handle_key(key)
            if not app.finished and (key is not None or console.size != size):
                size = console.size
                live.update(app.render(console), refresh=True)
    return app.result
