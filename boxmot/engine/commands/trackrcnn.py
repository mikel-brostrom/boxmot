"""Click adapter for evaluating image trackers on saved TrackR-CNN predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from boxmot.engine.commands._options import tracker_config_option


@click.command(name="eval-trackrcnn", help="Evaluate KITTI MOTS tracking from saved TrackR-CNN detections and masks.")
@click.option(
    "--tracker",
    type=str,
    default="maf_hda",
    show_default=True,
    help="Python image tracker requiring no additional model weights.",
)
@tracker_config_option
@click.option(
    "--detections",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="TrackR-CNN directory containing one detection text file per sequence.",
)
@click.option(
    "--images",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="KITTI training/image_02 directory containing sequence PNG folders.",
)
@click.option(
    "--instances",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="KITTI MOTS ground-truth instance PNG directory.",
)
@click.option("--split", type=click.Choice(["val", "train", "fulltrain"]), default="val", show_default=True)
@click.option(
    "--sequence",
    "sequence_names",
    multiple=True,
    metavar="NAME",
    help="Restrict the selected split to a sequence. Repeat for multiple sequences.",
)
@click.option(
    "--project",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/trackrcnn"),
    show_default=True,
    help="Results root; creates a new split directory without overwriting previous runs.",
)
def eval_trackrcnn(**kwargs: Any) -> None:
    """Replay supplied detections and real image frames with class-separated tracking."""
    from boxmot.engine.config import build_mode_namespace

    try:
        from boxmot.engine.eval.trackrcnn import run_trackrcnn

        args = build_mode_namespace("eval", {**kwargs, "tracker_backend": "python", "per_class": True})
        output = run_trackrcnn(args)
    except (TypeError, ValueError, FileNotFoundError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Results: {output}")


__all__ = ("eval_trackrcnn",)
