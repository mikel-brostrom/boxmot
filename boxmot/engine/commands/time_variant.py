"""Click adapter for timestamp-preserving frame-loss dataset variants."""

from __future__ import annotations

from pathlib import Path

import click

from boxmot.engine.commands._options import build_selection_options, data_root_option, dataset_option
from boxmot.engine.commands._support import _dispatch_cli_workflow
from boxmot.engine.config import get_mode_default


@click.command("time-variant", help="Derive a timestamped frame-loss dataset from an existing perception build")
@dataset_option(default=get_mode_default("time-variant", "dataset"))
@click.option(
    "--split",
    default=get_mode_default("time-variant", "split"),
    show_default=True,
    help="Source split containing the sequence and its ground truth.",
)
@click.option("--sequence", required=True, help="One source sequence, e.g. MOT17-10-FRCNN.")
@build_selection_options
@data_root_option
@click.option(
    "--name",
    default=None,
    help=(
        "New dataset identifier; defaults to the lowercase sequence name plus '-variable-time'. "
        "Existing outputs are refused."
    ),
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=get_mode_default("time-variant", "seed"),
    show_default=True,
    help="Seed for reproducible frame selection and simulated outages.",
)
@click.pass_context
def time_variant(
    ctx: click.Context,
    dataset: str,
    split: str,
    sequence: str,
    build_ref: str,
    build_root: Path | None,
    data_root: Path | None,
    name: str | None,
    seed: int,
) -> None:
    """Reuse real images, annotations, and cached perception at retained times."""
    _dispatch_cli_workflow(
        ctx,
        "time-variant",
        "boxmot.engine.dataset_variants.workflow",
        {
            "dataset": dataset,
            "split": split,
            "sequence": sequence,
            "build": build_ref,
            "build_root": build_root,
            "data_root": data_root,
            "name": name,
            "seed": seed,
        },
    )


__all__ = ("time_variant",)
