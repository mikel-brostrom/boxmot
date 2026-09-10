"""Click adapters for EagerMOT evaluation and tuning on KITTI sensor inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

import click

from boxmot.engine.commands._options import sequence_option

_Command = TypeVar("_Command", bound=Callable[..., Any])


def _sensor_options(command: _Command) -> _Command:
    """Share KITTI sensor and ground-truth selection between evaluation and tuning."""
    options = (
        click.option(
            "--dataset",
            type=click.Path(exists=True, path_type=Path),
            required=True,
            help="KITTI fusion dataset folder or dataset.yaml; replay.yaml selects saved predictions.",
        ),
        click.option(
            "--split",
            type=click.Choice(["val", "train", "fulltrain"]),
            default=None,
            help="Dataset split; defaults to the manifest's default_split.",
        ),
        sequence_option,
    )
    for option in reversed(options):
        command = option(command)
    return command


@click.command(name="eval-eagermot", help="Evaluate KITTI EagerMOT segmentation tracking from saved sensor detections.")
@_sensor_options
@click.option(
    "--class-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML containing complete car and pedestrian tracker profiles, such as tuning's best.yaml.",
)
@click.option("--show", is_flag=True, default=False, help="Preview tracked masks, IDs, and classes on the images.")
@click.option("--save", is_flag=True, default=False, help="Save one annotated MP4 per sequence under results/videos.")
@click.option(
    "--show-3d",
    is_flag=True,
    default=False,
    help="Overlay estimated tracked 3D cuboids on the preview or saved video; requires --show or --save.",
)
@click.option(
    "--project",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/eagermot"),
    show_default=True,
    help="Results root; creates a new split directory without overwriting previous runs.",
)
def eval_eagermot(**kwargs: Any) -> None:
    """Run the KITTI sensor reader, class-specific tracker presets, and mask metrics."""
    if kwargs["show_3d"] and not (kwargs["show"] or kwargs["save"]):
        raise click.UsageError("--show-3d requires --show or --save.")

    from boxmot.engine.config.runtime import build_mode_namespace

    args = build_mode_namespace("eval", {**kwargs, "tracker": "eagermot", "tracker_backend": "python"})
    try:
        # Report missing optional evaluation dependencies as concise CLI errors.
        from boxmot.engine.eval.eagermot_kitti import run_eagermot_kitti

        output = run_eagermot_kitti(args)
    except (ValueError, OSError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Results: {output}")


@click.command(
    name="tune-eagermot",
    help="Tune separate car and pedestrian EagerMOT profiles together for class-average KITTI mask HOTA.",
)
@_sensor_options
@click.option(
    "--project",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/eagermot-tune"),
    show_default=True,
    help="Results root; creates a new split directory without overwriting previous runs.",
)
@click.option(
    "--n-trials",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Number of serial CPU trials, including the first trial with the default KITTI profiles.",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0, max=2**32 - 1),
    default=0,
    show_default=True,
    help="Random seed for parameter sampling.",
)
def tune_eagermot(**kwargs: Any) -> None:
    """Optimize class-specific profiles by jointly replaying both KITTI classes."""
    from boxmot.engine.config.runtime import build_mode_namespace

    args = build_mode_namespace("tune", {**kwargs, "tracker": "eagermot", "tracker_backend": "python"})
    try:
        # Report missing optional tuning dependencies as concise CLI errors.
        from boxmot.engine.tuning.eagermot_kitti import run_eagermot_kitti_tuning

        output = run_eagermot_kitti_tuning(args)
    except (ValueError, OSError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Results: {output}")
    click.echo(f"Best profiles: {output / 'best.yaml'}")


__all__ = ("eval_eagermot", "tune_eagermot")
