"""Interactive launcher adapter for the existing BoxMOT commands."""

from __future__ import annotations

import shlex

import click


@click.command(help="Choose a command, models, and options in an interactive terminal launcher.")
@click.option(
    "--command",
    "initial_command",
    type=click.Choice(("track", "materialize", "eval", "tune", "export", "train-reid", "eval-reid")),
    help="Start with this command selected.",
)
@click.option("--experiment", type=str, default=None, help="Prefill an experiment ID or YAML path.")
@click.pass_context
def ui(ctx: click.Context, initial_command: str | None, experiment: str | None) -> None:
    """Collect an invocation interactively and dispatch through Click unchanged."""
    if experiment and initial_command not in {None, "materialize", "eval", "tune"}:
        raise click.BadParameter(
            "Experiments apply to materialize, eval, and tune. Select one of those commands or omit --experiment.",
            param_hint="--experiment",
        )
    from boxmot.engine.ui.core.ui import get_console
    from boxmot.engine.ui.dashboard import dashboard_mode
    from boxmot.engine.ui.keyboard import is_interactive
    from boxmot.engine.ui.launcher.app import run_launcher

    if not is_interactive(get_console()):
        raise click.UsageError(
            "The launcher requires an interactive terminal. Use 'boxmot MODE --help' for CLI options."
        )
    root = ctx.find_root()
    command_group = root.command
    assert isinstance(command_group, click.Group)
    argv = run_launcher(command_group, initial_command=initial_command, experiment=experiment)
    if argv is None:
        return
    selected = command_group.get_command(root, argv[0])
    if selected is None:
        raise click.ClickException(f"Unknown command: {argv[0]}")
    click.echo(f"Running: {shlex.join(['boxmot', '--dashboard', *argv])}")
    # Parse the chosen argv again so parameter-source tracking, callbacks, and
    # workflow-specific defaults behave exactly as a direct CLI invocation.
    with dashboard_mode(), selected.make_context(argv[0], argv[1:], parent=root) as command_context:
        selected.invoke(command_context)
