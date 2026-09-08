"""Click adapter for cached tracker evaluation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click

from boxmot.engine.commands._options import (
    association_function_option,
    data_root_option,
    dataset_fps_option,
    kalman_calibration_option,
    replay_build_options,
    replay_options,
    split_option,
    tracker_backend_option,
    tracker_config_option,
)
from boxmot.engine.commands._support import (
    _dispatch_cli_workflow,
    _prepare_replay_build,
    _require_replay_input,
)
from boxmot.engine.config import BOXMOT_DEFAULTS


@click.command(name="eval", help="Evaluate tracking performance")
@replay_build_options(dataset_default=BOXMOT_DEFAULTS.eval.dataset)
@data_root_option
@split_option
@dataset_fps_option
@tracker_backend_option(default=BOXMOT_DEFAULTS.eval.tracker_backend)
@tracker_config_option
@association_function_option
@replay_options(mode="eval", parallel=True)
@click.option(
    "--show",
    is_flag=True,
    default=False,
    help="Preview tracking at source timing, after calibration when --calibrate-kf is enabled.",
)
@click.option(
    "--save",
    is_flag=True,
    default=False,
    help="Save annotated tracking videos from cached replay, after calibration when --calibrate-kf is enabled.",
)
@kalman_calibration_option(mode="eval")
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
    calibrate_kf: bool,
    **kwargs: Any,
) -> None:
    """Evaluate a tracker, materializing the selected configuration when needed."""

    _require_replay_input(experiment, dataset, "eval")
    if calibrate_kf or kwargs.get("tracker_config") is not None or kwargs.get("variable_dt") is not None:
        from boxmot.engine.tracker_config import resolve_tracker_options
        from boxmot.engine.tuning.kalman import validate_kf_calibration
        from boxmot.trackers.specs import parse_tracker_spec

        try:
            tracker_spec = parse_tracker_spec(
                kwargs["tracker"],
                default_backend=kwargs["tracker_backend"],
            )
            if calibrate_kf:
                validate_kf_calibration(tracker_spec.name, tracker_spec.backend)
            resolve_tracker_options(
                SimpleNamespace(**{**kwargs, "tracker": tracker_spec.name, "tracker_backend": tracker_spec.backend})
            )
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise click.UsageError(str(exc)) from exc
    experiment, dataset, build_ref = _prepare_replay_build(
        ctx,
        mode="eval",
        experiment=experiment,
        dataset=dataset,
        detector=detector,
        reid=reid,
        build_ref=build_ref,
        build_root=build_root,
        device=device,
        data_root=data_root,
        split=split,
        tracker=str(kwargs["tracker"]),
        fps=kwargs.get("fps"),
        allow_noncanonical_build=allow_noncanonical_build,
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
            "calibrate_kf": calibrate_kf,
        },
    )


__all__ = ("eval",)
