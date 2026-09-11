"""Click adapter for tracker tuning from perception builds or sensor datasets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

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
from boxmot.engine.commands._support import (
    _dispatch_cli_workflow,
    _explicit_cli_keys,
    _prepare_replay_build,
    _require_replay_input,
)
from boxmot.engine.config.runtime import BOXMOT_DEFAULTS, get_mode_default, resolve_sequence_workers

_SENSOR_OPTIONS = frozenset(
    {
        "dataset",
        "tracker",
        "tracker_backend",
        "split",
        "sequence_names",
        "n_trials",
        "seed",
        "project",
        "search_alg",
        "objectives",
        "maximize",
        "max_concurrent_trials",
        "sequence_workers",
        "device",
        "eval_masks",
        "per_class",
        "verbose",
    }
)

_TUNE_METRIC_OPTIONS = {"--objectives", "--maximize", "--minimize"}


def _validate_sensor_options(ctx: click.Context, payload: Mapping[str, Any]) -> None:
    """Reject controls the saved-sensor optimizer cannot honor before loading data."""
    explicit = _explicit_cli_keys(ctx)
    unsupported = explicit - _SENSOR_OPTIONS
    if unsupported:
        names = ", ".join(
            option.opts[0]
            for option in ctx.command.params
            if isinstance(option, click.Option) and option.name in unsupported
        )
        raise click.UsageError(
            f"Sensor dataset tuning does not support {names}; inputs come from the dataset manifest."
        )
    required_values = {
        "search_alg": ("optuna",),
        "max_concurrent_trials": (0, 1),
        "device": ("cpu",),
    }
    for name, allowed in required_values.items():
        if name in explicit and payload[name] not in allowed:
            option = "--" + name.replace("_", "-")
            choices = ", ".join(map(str, allowed))
            raise click.UsageError(
                f"Sensor dataset tuning runs serial Optuna trials on CPU; {option} must be one of: {choices}."
            )
    for name in ("objectives", "maximize"):
        if name in explicit:
            metrics = [metric for value in payload[name] for metric in value.replace(",", " ").split()]
            if metrics != ["HOTA"]:
                raise click.UsageError(
                    f"Sensor dataset tuning optimizes class-average mask HOTA; --{name} must be HOTA."
                )


def _prepare_sensor_tuning(ctx: click.Context, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize a declared sensor dataset before dispatch through the shared tuner."""
    reference = payload.get("dataset")
    if not reference:
        return None

    from boxmot.datasets.inputs import resolve_sensor_dataset_config_path
    from boxmot.engine.config.datasets import load_sensor_evaluation_inputs, validate_sensor_workflow_inputs
    from boxmot.trackers.common.specs import parse_tracker_spec

    try:
        path = resolve_sensor_dataset_config_path(reference, split=payload.get("split"))
        if path is None:
            return None
        spec = parse_tracker_spec(payload["tracker"], default_backend=payload["tracker_backend"])
        validate_sensor_workflow_inputs(path, spec, mode="tune", split=payload.get("split"))
        _validate_sensor_options(ctx, payload)
        dataset = load_sensor_evaluation_inputs(
            path,
            split=payload.get("split"),
            sequence_names=payload.get("sequence_names", ()),
        )
        explicit = _explicit_cli_keys(ctx)
        sequence_workers = resolve_sequence_workers(
            len(dataset.sequence_names),
            payload.get("sequence_workers")
            if "sequence_workers" in explicit
            else get_mode_default("tune", "sequence_workers"),
        )
    except (ValueError, OSError) as exc:
        raise click.UsageError(str(exc)) from exc

    return {
        **payload,
        "tracker": spec.name,
        "tracker_backend": spec.backend,
        "dataset": dataset.config_path,
        "split": dataset.split,
        "sequence_names": dataset.sequence_names,
        "seed": 0 if payload.get("seed") is None else payload["seed"],
        "project": payload["project"] if "project" in explicit else Path("runs/eagermot-tune"),
        "device": "cpu",
        "max_concurrent_trials": 1,
        "sequence_workers": sequence_workers,
        "objectives": ("HOTA",),
        "maximize": ("HOTA",),
        "per_class": True,
        "eval_masks": True,
    }


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
    sensor_payload = _prepare_sensor_tuning(
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
