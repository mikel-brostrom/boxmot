"""Click adapter for tracker tuning from perception builds or sensor datasets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click

from boxmot.engine.commands._options import (
    data_root_option,
    dataset_fps_option,
    eval_masks_option,
    kalman_calibration_option,
    replay_build_options,
    replay_options,
    sequence_option,
    split_option,
    tracker_backend_option,
    tracker_config_option,
)
from boxmot.engine.commands._support import _dispatch_cli_workflow, _prepare_replay_build, _require_replay_input
from boxmot.engine.config.runtime import BOXMOT_DEFAULTS

_TUNE_METRIC_OPTIONS = {"--objectives", "--maximize", "--minimize"}


def _normalize_tune_metric_cli_args(args: list[str]) -> list[str]:
    """Fold space-separated tune metric values into Click option values."""

    normalized: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        option = None
        inline_value = None

        if token in _TUNE_METRIC_OPTIONS:
            option = token
        else:
            for candidate in _TUNE_METRIC_OPTIONS:
                prefix = f"{candidate}="
                if token.startswith(prefix):
                    option = candidate
                    inline_value = token[len(prefix) :]
                    break

        if option is None:
            normalized.append(token)
            index += 1
            continue

        values: list[str] = []
        if inline_value not in {None, ""}:
            values.append(inline_value)
        index += 1
        while index < len(args) and not args[index].startswith("-"):
            values.append(args[index])
            index += 1

        normalized.append(option)
        if values:
            normalized.append(",".join(values))

    return normalized


class TuneCommand(click.Command):
    """Command parser that accepts repeated, comma-, or space-separated metrics."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        return super().parse_args(ctx, _normalize_tune_metric_cli_args(list(args)))


def _tune_options(func):
    options = (
        click.option(
            "--n-trials",
            type=click.IntRange(min=1),
            default=BOXMOT_DEFAULTS.tune.n_trials,
            help="number of hyperparameter optimization trials",
        ),
        click.option(
            "--seed",
            type=click.IntRange(min=0, max=2**32 - 1),
            default=None,
            help="Random seed for parameter sampling. Sensor datasets default to 0.",
        ),
        click.option(
            "--max-concurrent-trials",
            type=int,
            default=0,
            help=(
                "max concurrent trials (0 = auto, defaults to min(4, cpu_count)); "
                "controls parallelism and improves Bayesian search effectiveness"
            ),
        ),
        click.option(
            "--time-budget-s",
            type=float,
            default=None,
            help=(
                "optional time budget in seconds for the entire tuning run; "
                "Tune stops launching new trials after this time"
            ),
        ),
        click.option(
            "--resume-tune",
            type=str,
            default=None,
            help=(
                "resume a Ray Tune experiment; pass a folder name (e.g. deepocsort_tune_3) "
                "or full path under runs/ray/. Retries errored trials and continues remaining ones."
            ),
        ),
        click.option(
            "--objectives",
            type=str,
            multiple=True,
            default=BOXMOT_DEFAULTS.tune.objectives,
            help=(
                "metrics to track and return from each trial; accepts repeated, "
                "comma-separated, or space-separated values"
            ),
        ),
        click.option(
            "--maximize",
            type=str,
            multiple=True,
            default=BOXMOT_DEFAULTS.tune.maximize,
            help=(
                "metrics to maximize; accepts repeated, comma-separated, or space-separated values; "
                "defaults to first --objectives value (e.g. HOTA)"
            ),
        ),
        click.option(
            "--minimize",
            type=str,
            multiple=True,
            default=BOXMOT_DEFAULTS.tune.minimize,
            help=(
                "metrics to minimize for Pareto search; accepts repeated, comma-separated, or "
                "space-separated values (e.g. IDSW_rate); triggers multi-objective mode when set"
            ),
        ),
        click.option(
            "--search-alg",
            "search_alg",
            type=click.Choice(["optuna", "hyperopt", "random"]),
            default="optuna",
            help=(
                "search algorithm backend for hyperparameter optimization; optuna (default) uses TPE with "
                "conditional search spaces, hyperopt uses Tree-structured Parzen Estimators via HyperOpt, "
                "random uses uniform random sampling"
            ),
        ),
    )
    for option in reversed(options):
        func = option(func)
    return func


@click.command(
    cls=TuneCommand,
    help=(
        "Tune tracker parameters from a perception build or a saved-sensor dataset. "
        "EagerMOT jointly tunes car and pedestrian profiles for class-average KITTI mask HOTA."
    ),
)
@replay_build_options()
@data_root_option
@split_option
@sequence_option
@dataset_fps_option
@eval_masks_option
@tracker_backend_option(default=BOXMOT_DEFAULTS.tune.tracker_backend)
@tracker_config_option
@replay_options(mode="tune", parallel=True)
@kalman_calibration_option(mode="tune")
@_tune_options
@click.pass_context
def tune(
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
    calibrate_kf: bool,
    eval_masks: bool,
    **kwargs: Any,
) -> None:
    """Resolve dataset inputs, then tune tracker parameters."""

    _require_replay_input(experiment, dataset, "tune")
    from boxmot.engine.commands.sensor_tune import prepare_sensor_tuning

    sensor_payload = prepare_sensor_tuning(
        ctx,
        {
            **kwargs,
            "experiment": experiment,
            "dataset": dataset,
            "detector": detector,
            "reid": reid,
            "build_ref": build_ref,
            "build_root": build_root,
            "device": device,
            "data_root": data_root,
            "split": split,
            "calibrate_kf": calibrate_kf,
            "eval_masks": eval_masks,
        },
    )
    if sensor_payload is not None:
        _dispatch_cli_workflow(ctx, "tune", "boxmot.engine.tuning.tuner", sensor_payload)
        return
    if calibrate_kf and kwargs.get("resume_tune"):
        raise click.UsageError(
            "--calibrate-kf cannot be combined with --resume-tune; resume reuses the saved calibration."
        )
    if calibrate_kf or kwargs.get("tracker_config") is not None or kwargs.get("variable_dt") is not None:
        from boxmot.engine.calibration.kalman import validate_kf_calibration
        from boxmot.engine.config.trackers import resolve_tracker_options
        from boxmot.trackers.common.specs import parse_tracker_spec

        try:
            tracker_spec = parse_tracker_spec(kwargs["tracker"], default_backend=kwargs["tracker_backend"])
            if calibrate_kf:
                validate_kf_calibration(tracker_spec.name, tracker_spec.backend)
            resolve_tracker_options(
                SimpleNamespace(**{**kwargs, "tracker": tracker_spec.name, "tracker_backend": tracker_spec.backend})
            )
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise click.UsageError(str(exc)) from exc
    experiment, dataset, build_ref = _prepare_replay_build(
        ctx,
        mode="tune",
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
        eval_masks=eval_masks,
    )
    _dispatch_cli_workflow(
        ctx,
        "tune",
        "boxmot.engine.tuning.tuner",
        {
            **kwargs,
            "calibrate_kf": calibrate_kf,
            "eval_masks": eval_masks,
            "experiment": experiment,
            "dataset": dataset,
            "build": build_ref,
            "build_root": build_root,
            "device": device,
            "data_root": data_root,
            "source": None,
            "benchmark": "",
            "split": split or "",
        },
    )


__all__ = ("tune",)
