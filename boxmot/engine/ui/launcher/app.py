"""Rich terminal application for selecting, reviewing, and launching commands."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import click
from rich import box
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


def _panel(content: RenderableType, title: str, *, accent: bool = False) -> Panel:
    """Use a consistent, compact visual style throughout the launcher."""
    return Panel(
        content,
        title=Text(title, style="bold cyan" if accent else "bold"),
        border_style="cyan" if accent else "bright_black",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _summary(model: LauncherModel) -> RenderableType:
    """Render the experiment's resolved components, without starting a run."""
    try:
        experiment = model.resolved_experiment()
    except (OSError, ValueError, TypeError) as error:
        return Text(str(error), style="yellow", overflow="fold")
    if experiment is None:
        return Group(
            Text(COMMAND_LABELS[model.name], style="bold cyan"),
            Text("\nSelect a field and press Enter to edit. Search model catalogs or type a custom path."),
            Text("\nDefaults come from the command. Only your edits appear in the command preview.", style="dim"),
        )
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
                self.model.set_value(field.name, None)
            else:
                self.model.set_value(field.name, value)
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
            if key == "escape":
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
                    self.model.set_value(self.editor.name, value)
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
        if key in {"escape", "q"}:
            self.finished = True
        elif key in {"up", "shift+tab", "down", "tab", "pageup", "pagedown"}:
            delta = {"up": -1, "shift+tab": -1, "down": 1, "tab": 1, "pageup": -8, "pagedown": 8}[key]
            self.cursor = max(0, min(len(fields) + 1, self.cursor + delta))
        elif key == "home":
            self.cursor = 0
        elif key == "end":
            self.cursor = len(fields) + 1
        elif key == "r" or (key == "enter" and self.cursor == len(fields) + 1):
            self._prepare_run()
        elif key == "enter":
            self._begin_edit(fields[self.cursor - 1] if self.cursor else None)
        elif key in {"d", "delete", "backspace"} and 0 < self.cursor <= len(fields):
            self.model.set_value(fields[self.cursor - 1].name, None)
            self.error = ""

    def _form(self, height: int) -> RenderableType:
        """Keep the selected option visible even in a short IDE terminal."""
        fields = self.model.fields
        self.cursor = min(self.cursor, len(fields) + 1)
        rows: list[tuple[str, str, str]] = [("Workflow", "Command", self.model.name)]
        for field in fields:
            value = self.model.field_display(field.name)
            value = value or ("Required" if field.option.required else "Not set")
            suffix = "" if field.name in self.model.values else "  · default"
            rows.append((field.group, field.label, value + suffix))
        rows.append(("Launch", "Review and run", "Enter to review"))
        visible = max(3, height - 7)
        start = max(0, min(self.cursor - visible // 2, len(rows) - visible))
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=1)
        table.add_column(ratio=2, overflow="ellipsis", no_wrap=True)
        table.add_column(ratio=3, overflow="ellipsis", no_wrap=True)
        for index in range(start, min(len(rows), start + visible)):
            _, label, value = rows[index]
            style = "bold black on cyan" if index == self.cursor else ""
            table.add_row(
                Text("›" if index == self.cursor else " ", style=style),
                Text(label, style=style),
                Text(value, style=style),
            )
        group = rows[self.cursor][0]
        help_text = "Choose which workflow to configure."
        if 0 < self.cursor <= len(fields):
            help_text = fields[self.cursor - 1].option.help or ""
        elif self.cursor == len(fields) + 1:
            help_text = "Review the exact command and start it with Enter."
        return _panel(
            Group(
                Text(f"{group}  ·  {self.cursor + 1}/{len(rows)}", style="cyan"),
                table,
                Text(""),
                Text(help_text, style="dim", overflow="ellipsis", no_wrap=True),
            ),
            "Configure",
            accent=True,
        )

    def _editor_view(self, height: int) -> RenderableType:
        """Render a filtered picker with explicit default and custom actions."""
        assert self.editor is not None
        editor = self.editor
        rows = editor.rows()
        editor.cursor = min(editor.cursor, max(0, len(rows) - 1))
        visible = max(2, height - 9)
        start = max(0, min(editor.cursor - visible // 2, len(rows) - visible))
        table = Table.grid(expand=True)
        table.add_column(width=3)
        table.add_column(overflow="ellipsis", no_wrap=True)
        for index in range(start, min(len(rows), start + visible)):
            value, label = rows[index]
            marker = "✓" if value in editor.selected else " "
            if index == editor.cursor:
                marker = "›" + marker
            style = "bold black on cyan" if index == editor.cursor else ""
            table.add_row(Text(marker, style=style), Text(label, style=style))
        if not rows:
            table.add_row("", "No matches")
        hint = "↑↓ select · Enter apply · Esc cancel · Ctrl+U clear"
        if editor.multiple and editor.choices:
            hint = "Space toggle · Enter apply · Esc cancel"
        return _panel(
            Group(
                Text(editor.help, style="dim", overflow="ellipsis", no_wrap=True),
                Text.assemble(("Search / value  ", "cyan"), editor.query, ("▏", "cyan")),
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
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1, minimum_size=7),
            Layout(name="preview", size=preview_height),
            Layout(name="footer", size=2),
        )
        layout["header"].update(
            _panel(
                Text.assemble(("BoxMOT", "bold cyan"), "  /  ", (COMMAND_LABELS[self.model.name], "bold")),
                "Command launcher",
            )
        )
        body_height = max(7, height - preview_height - 5)
        if self.review:
            layout["body"].update(self._review_view(console, body_height))
        elif self.editor is not None:
            layout["body"].update(self._editor_view(body_height))
        elif width >= 110:
            layout["body"].split_row(Layout(name="form", ratio=3), Layout(name="summary", ratio=2))
            layout["form"].update(self._form(body_height))
            layout["summary"].update(_panel(_summary(self.model), "Experiment"))
        else:
            layout["body"].update(self._form(body_height))
        try:
            preview = self.model.preview()
        except ValueError as error:
            preview = str(error)
        content = Text(self.error or preview, style="yellow" if self.error else "green", overflow="fold")
        layout["preview"].update(_panel(content, "Check configuration" if self.error else "Command preview"))
        footer = "↑↓ / Tab navigate   Enter edit   d reset   r review   q quit"
        if self.review:
            footer = "Enter RUN command   ↑↓ / PgUp/PgDn scroll review   Esc back"
        elif self.editor:
            footer = "Type to search or enter a value   ↑↓ choose   Enter apply   Esc back"
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
    with TerminalInput() as keyboard, Live(
        app.render(console), console=console, screen=True, auto_refresh=False, transient=True
    ) as live:
        size = console.size
        while not app.finished:
            key = keyboard.read_key(timeout=0.1)
            if key is not None:
                app.handle_key(key)
            if not app.finished and (key is not None or console.size != size):
                size = console.size
                live.update(app.render(console), refresh=True)
    return app.result
