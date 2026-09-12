"""Click adapter for perception builds and timestamp-preserving dataset variants."""

from __future__ import annotations

from pathlib import Path

import click

from boxmot.engine.commands._options import _parse_device, dataset_fps_option, experiment_option
from boxmot.engine.commands._support import (
    _dispatch_cli_workflow,
    _explicit_cli_keys,
    _require_experiment_input,
)
from boxmot.engine.config.runtime import BOXMOT_DEFAULTS, get_mode_default


@click.command(help="Build an immutable perception dataset or derive a timestamped frame-loss variant")
@click.option(
    "--time-variant",
    is_flag=True,
    help="Derive a frame-loss dataset from --dataset, --sequence, and --build, preserving source timestamps.",
)
@experiment_option
@click.option(
    "--dataset",
    default=get_mode_default("materialize", "dataset"),
    show_default=True,
    help="Source dataset ID or YAML file for --time-variant.",
)
@click.option(
    "--split",
    default=get_mode_default("materialize", "split"),
    show_default=True,
    help="Source split containing the sequence and ground truth for --time-variant.",
)
@click.option("--sequence", help="One source sequence for --time-variant, e.g. MOT17-10-FRCNN.")
@click.option(
    "--build",
    "build_ref",
    help="Parent materialized build ID or directory for --time-variant.",
)
@click.option(
    "--name",
    help=(
        "New dataset ID for --time-variant; defaults to the lowercase sequence name plus '-variable-time'. "
        "Existing outputs are refused."
    ),
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=get_mode_default("materialize", "seed"),
    show_default=True,
    help="Seed for reproducible frame selection and simulated outages with --time-variant.",
)
@click.option(
    "--device",
    default=BOXMOT_DEFAULTS.materialize.device,
    callback=_parse_device,
    help="One device for all perception stages: cpu, mps, cuda:N, or N (e.g. 0). GPU lists are not supported.",
)
@dataset_fps_option
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
    time_variant: bool,
    experiment: str | None,
    dataset: str,
    split: str,
    sequence: str | None,
    build_ref: str | None,
    name: str | None,
    seed: int,
    device: str,
    fps: float | None,
    publish_image_refs: bool,
    publish_masks: bool,
    publish_embeddings: bool,
    data_root: Path | None,
    build_root: Path | None,
    plan_path: Path | None,
    plan_overrides: tuple[str, ...],
    resume: bool,
) -> None:
    """Build perception or derive a dataset while reusing cached perception."""

    explicit = _explicit_cli_keys(ctx)
    if time_variant:
        perception_options = {
            "experiment": "--experiment",
            "device": "--device",
            "fps": "--fps",
            "publish_image_refs": "--publish-image-refs/--no-publish-image-refs",
            "publish_masks": "--publish-masks/--no-publish-masks",
            "publish_embeddings": "--publish-embeddings/--no-publish-embeddings",
            "plan_path": "--plan",
            "plan_overrides": "--set",
            "resume": "--resume/--no-resume",
        }
        invalid = [option for key, option in perception_options.items() if key in explicit]
        if invalid:
            raise click.UsageError(f"{', '.join(invalid)} cannot be combined with --time-variant.")
        if not sequence:
            raise click.UsageError("materialize --time-variant requires --sequence <sequence-name>.")
        if not build_ref:
            raise click.UsageError("materialize --time-variant requires --build <build-id-or-path>.")
        _dispatch_cli_workflow(
            ctx,
            "materialize",
            "boxmot.engine.dataset_variants.workflow",
            {
                "time_variant": True,
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
        return

    variant_options = {
        "dataset": "--dataset",
        "split": "--split",
        "sequence": "--sequence",
        "build_ref": "--build",
        "name": "--name",
        "seed": "--seed",
    }
    invalid = [option for key, option in variant_options.items() if key in explicit]
    if invalid:
        raise click.UsageError(f"Dataset derivation options {', '.join(invalid)} require --time-variant.")
    experiment = _require_experiment_input(experiment, "materialize")

    _dispatch_cli_workflow(
        ctx,
        "materialize",
        "boxmot.engine.materialization.workflow",
        {
            "experiment": experiment,
            "device": device,
            "fps": fps,
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
