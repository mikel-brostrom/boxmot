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
from boxmot.engine.commands._support import _dispatch_cli_workflow
from boxmot.engine.config import BOXMOT_DEFAULTS


def _require_eval_input(
    experiment: str | None,
    dataset: str | None,
) -> tuple[str | None, str | None]:
    """Require exactly one experiment configuration or model-free dataset."""

    if experiment and dataset:
        raise click.UsageError(
            "eval accepts either --dataset <dataset-id-or-yaml> or "
            "--experiment <experiment-id-or-yaml>, not both."
        )
    if not experiment and not dataset:
        raise click.UsageError(
            "eval requires either --dataset <dataset-id-or-yaml> or --experiment <experiment-id-or-yaml>."
        )
    return experiment, dataset


@click.command(name="eval", help="Evaluate tracking performance")
@experiment_option
@dataset_option(default=BOXMOT_DEFAULTS.eval.dataset)
@build_selection_options
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
    build_ref: str,
    build_root: Path | None,
    data_root: Path | None,
    split: str | None,
    sequence_names: tuple[str, ...],
    allow_noncanonical_build: bool,
    compare_trackeval: bool,
    **kwargs: Any,
) -> None:
    """Evaluate a tracker against an immutable materialized build."""

    experiment, dataset = _require_eval_input(experiment, dataset)
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
