"""Click adapter for immutable perception-dataset materialization."""

from __future__ import annotations

from pathlib import Path

import click

from boxmot.engine.commands._options import experiment_option
from boxmot.engine.commands._support import _dispatch_cli_workflow
from boxmot.engine.config import BOXMOT_DEFAULTS


@click.command(help="Build an immutable keyed perception dataset")
@click.option(
    "--device",
    default=BOXMOT_DEFAULTS.materialize.device,
    help="Single execution device for all perception stages, e.g. cpu, mps, cuda:0, or 0.",
)
@experiment_option(required=True)
@click.option("--publish-image-refs/--no-publish-image-refs", default=True, show_default=True)
@click.option("--publish-masks/--no-publish-masks", default=False, show_default=True)
@click.option("--publish-embeddings/--no-publish-embeddings", default=True, show_default=True)
@click.option(
    "--data-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Tracking-dataset root; defaults to ./datasets/mot.",
)
@click.option(
    "--build-root",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Materialized-dataset and shared detector-cache root; overrides BOXMOT_BUILDS_DIR and "
        "defaults to ./runs/materializations."
    ),
)
@click.option(
    "--plan",
    "plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML file containing local executor stage settings.",
)
@click.option(
    "--set",
    "plan_overrides",
    multiple=True,
    metavar="STAGE.FIELD=VALUE",
    help="Override one executor plan value; repeat as needed.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Resume this build's checkpoints; compatible shared detections are reused independently.",
)
@click.pass_context
def materialize(
    ctx: click.Context,
    experiment: str,
    device: str,
    publish_image_refs: bool,
    publish_masks: bool,
    publish_embeddings: bool,
    data_root: Path | None,
    build_root: Path | None,
    plan_path: Path | None,
    plan_overrides: tuple[str, ...],
    resume: bool,
) -> None:
    """Build an immutable, reusable perception dataset."""

    _dispatch_cli_workflow(
        ctx,
        "materialize",
        "boxmot.engine.materialization.workflow",
        {
            "experiment": experiment,
            "device": device,
            "publish_image_refs": publish_image_refs,
            "publish_masks": publish_masks,
            "publish_embeddings": publish_embeddings,
            "data_root": data_root,
            "build_root": build_root,
            "plan_path": plan_path,
            "plan_overrides": tuple(plan_overrides),
            "resume": resume,
        },
    )


__all__ = ("materialize",)
