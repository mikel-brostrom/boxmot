"""Click adapter for cached tracker evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from boxmot.engine.commands._options import (
    association_function_option,
    build_selection_options,
    data_root_option,
    dataset_option,
    experiment_option,
    replay_options,
    split_option,
    tracker_backend_option,
)
from boxmot.engine.commands._support import (
    _build_cli_namespace,
    _dispatch_cli_workflow,
    _is_option_explicit,
    _run_engine_workflow,
)
from boxmot.engine.config import BOXMOT_DEFAULTS
from boxmot.engine.experiment_config import ConfigurationError, resolve_matching_experiment_path
from boxmot.trackers.registry import get_tracker_definition


def _require_eval_input(
    experiment: str | None,
    dataset: str | None,
) -> tuple[str | None, str | None]:
    """Require exactly one experiment or dataset selector."""

    if experiment and dataset:
        raise click.UsageError(
            "eval accepts either --dataset <dataset-id-or-yaml> or --experiment <experiment-yaml>, not both."
        )
    if not experiment and not dataset:
        raise click.UsageError("eval requires either --dataset <dataset-id-or-yaml> or --experiment <experiment-yaml>.")
    return experiment, dataset


def _validate_component_selection(
    *,
    experiment: str | None,
    dataset: str | None,
    detector: str | None,
    reid: str | None,
    build_ref: str | None,
) -> None:
    """Validate direct component selectors before materialization or replay."""

    components = tuple(name for name, value in (("--detector", detector), ("--reid", reid)) if value)
    if experiment and components:
        names = " and ".join(components)
        raise click.UsageError(
            f"{names} cannot be combined with --experiment because experiment YAML fixes perception components."
        )
    if reid and not detector:
        raise click.UsageError("--reid requires --detector when selecting evaluation components directly.")
    if detector and not dataset:
        raise click.UsageError("--detector requires --dataset when selecting evaluation components directly.")
    if build_ref is None and dataset and not detector:
        raise click.UsageError(
            "eval with --dataset requires --detector for automatic materialization, or --build to replay an "
            "existing materialized build."
        )


@click.command(name="eval", help="Evaluate tracking performance")
@experiment_option
@dataset_option(default=BOXMOT_DEFAULTS.eval.dataset)
@click.option(
    "--detector",
    type=str,
    default=None,
    help=(
        "Detector profile ID or YAML config used to resolve an authored experiment with --dataset; "
        "append /CHECKPOINT to disambiguate."
    ),
)
@click.option(
    "--reid",
    type=str,
    default=None,
    help="ReID profile used to resolve an authored experiment; omit only for experiments without ReID.",
)
@build_selection_options(required=False)
@click.option(
    "--device",
    default=BOXMOT_DEFAULTS.materialize.device,
    help="Perception device used for automatic materialization, e.g. cpu, mps, cuda:0, or 0.",
)
@data_root_option
@split_option
@tracker_backend_option(default=BOXMOT_DEFAULTS.eval.tracker_backend)
@association_function_option
@replay_options(mode="eval", parallel=True)
@click.option(
    "--sequence",
    "sequence_names",
    type=str,
    multiple=True,
    metavar="NAME",
    help="Limit evaluation to one sequence. Repeat to select multiple sequences.",
)
@click.option(
    "--allow-noncanonical-build",
    is_flag=True,
    default=False,
    help=(
        "Allow an unbound legacy build after validating it against the selected dataset. "
        "WARNING: use only when you have independently verified the build provenance."
    ),
)
@click.option(
    "--compare-trackeval/--no-compare-trackeval",
    default=False,
    help="Compare BoxMOT metrics against TrackEval for an AABB MOTChallenge benchmark.",
)
@click.pass_context
def eval(
    ctx: click.Context,
    experiment: str | None,
    dataset: str | None,
    detector: str | None,
    reid: str | None,
    build_ref: str | None,
    build_root: Path | None,
    device: str,
    data_root: Path | None,
    split: str | None,
    sequence_names: tuple[str, ...],
    allow_noncanonical_build: bool,
    compare_trackeval: bool,
    **kwargs: Any,
) -> None:
    """Evaluate a tracker, materializing the selected configuration when needed."""

    experiment, dataset = _require_eval_input(experiment, dataset)
    _validate_component_selection(
        experiment=experiment,
        dataset=dataset,
        detector=detector,
        reid=reid,
        build_ref=build_ref,
    )
    if detector is not None:
        try:
            experiment = str(
                resolve_matching_experiment_path(
                    dataset=str(dataset),
                    detector=detector,
                    reid=reid,
                    split=split,
                    mode="eval",
                )
            )
        except (ConfigurationError, FileNotFoundError) as exc:
            raise click.UsageError(str(exc)) from exc
        dataset = None
    if build_ref is not None and _is_option_explicit(ctx, "device"):
        raise click.UsageError("--device applies only when --build is omitted for automatic materialization.")
    if build_ref is None:
        if allow_noncanonical_build:
            raise click.UsageError("--allow-noncanonical-build requires an explicit --build.")
        tracker_capabilities = get_tracker_definition(str(kwargs["tracker"])).capabilities
        materialize_args = _build_cli_namespace(
            ctx,
            "materialize",
            {
                "experiment": experiment,
                "data_root": data_root,
                "build_root": build_root,
                "device": device,
                "publish_image_refs": True,
                "publish_masks": tracker_capabilities.requires_masks,
                "publish_embeddings": tracker_capabilities.accepts_embeddings,
                "plan_path": None,
                "plan_overrides": (),
                "resume": True,
            },
        )
        materialize_args.materialize_split = split
        materialize_args.materialize_mode = "eval"
        build_ref = _run_engine_workflow(
            "boxmot.engine.materialization.workflow",
            materialize_args,
        )

    _dispatch_cli_workflow(
        ctx,
        "eval",
        "boxmot.engine.eval.evaluator",
        {
            **kwargs,
            "experiment": experiment,
            "dataset": dataset,
            "build": build_ref,
            "build_root": build_root,
            "device": device,
            "data_root": data_root,
            "source": None,
            "benchmark": "",
            "split": split or "",
            "sequence_names": sequence_names,
            "allow_noncanonical_build": allow_noncanonical_build,
            "compare_trackeval": compare_trackeval,
        },
    )


__all__ = ("eval",)
