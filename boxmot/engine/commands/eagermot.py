"""Click adapter for evaluating EagerMOT on downloaded KITTI sensor inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click


@click.command(name="eval-eagermot", help="Evaluate KITTI EagerMOT segmentation tracking from saved sensor detections.")
@click.option(
    "--data-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("eagermot-data"),
    show_default=True,
    help="Folder containing calib, ego_motion, pointgnn, and trackrcnn_detections.",
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
    help="Restrict the selected split to a sequence. Repeat for multiple sequences.",
)
@click.option(
    "--pointgnn-car",
    type=click.Choice(["t2-train", "t3-trainval"]),
    default="t2-train",
    show_default=True,
    help="Car detection variant. T2 covers validation; T3 covers all training sequences.",
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
    from boxmot.engine.config import build_mode_namespace
    from boxmot.engine.eval.eagermot_kitti import run_eagermot_kitti

    args = build_mode_namespace("eval", {**kwargs, "tracker": "eagermot", "tracker_backend": "python"})
    try:
        output = run_eagermot_kitti(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Results: {output}")


__all__ = ("eval_eagermot",)
