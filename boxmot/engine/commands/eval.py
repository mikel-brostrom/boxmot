"""Click adapter for tracker evaluation from perception builds or sensor datasets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import click

from boxmot.engine.commands._options import (
    association_function_option,
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
        "project",
        "class_config",
        "show",
        "save",
        "show_3d",
        "device",
        "sequence_workers",
        "eval_masks",
        "per_class",
        "verbose",
        "show_timing",
    }
)


def _prepare_sensor_evaluation(ctx: click.Context, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize a declared sensor dataset before dispatch through the shared evaluator."""
    reference = payload.get("dataset")
    if not reference:
        return None

    from boxmot.datasets.inputs import resolve_sensor_dataset_config_path
    from boxmot.engine.config.datasets import load_sensor_evaluation_inputs
    from boxmot.trackers.common.specs import parse_tracker_spec

    explicit = _explicit_cli_keys(ctx)
    try:
        path = resolve_sensor_dataset_config_path(reference, split=payload.get("split"))
        if path is None:
            return None
        spec = parse_tracker_spec(payload["tracker"], default_backend=payload["tracker_backend"])
        if spec.name != "eagermot" or spec.backend != "python":
            raise ValueError("Sensor dataset evaluation requires --tracker eagermot --tracker-backend python.")
        unsupported = explicit - _SENSOR_OPTIONS
        if unsupported:
            names = ", ".join(
                option.opts[0]
                for option in ctx.command.params
                if isinstance(option, click.Option) and option.name in unsupported
            )
            raise click.UsageError(
                f"Sensor dataset evaluation does not support {names}; inputs come from the dataset manifest."
            )
        if "device" in explicit and payload["device"] != "cpu":
            raise click.UsageError("Sensor dataset evaluation runs on CPU; --device must be cpu.")
        dataset = load_sensor_evaluation_inputs(
            path,
            split=payload.get("split"),
            sequence_names=payload.get("sequence_names", ()),
        )
        workers = resolve_sequence_workers(
            len(dataset.sequence_names),
            payload.get("sequence_workers")
            if "sequence_workers" in explicit
            else get_mode_default("eval", "sequence_workers"),
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
        "project": payload["project"] if "project" in explicit else Path("runs/eagermot"),
        "device": "cpu",
        "sequence_workers": 1 if payload.get("show") else workers,
        "per_class": True,
        "eval_masks": True,
    }


@click.command(name="eval", help="Evaluate tracking performance from a perception build or saved-sensor dataset.")
@replay_build_options(dataset_default=BOXMOT_DEFAULTS.eval.dataset)
@data_root_option
@split_option
@dataset_fps_option
@eval_masks_option
@tracker_backend_option(default=BOXMOT_DEFAULTS.eval.tracker_backend)
@tracker_config_option
@click.option(
    "--class-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="EagerMOT sensor evaluation: YAML containing car and pedestrian profiles, such as tuning's best.yaml.",
)
@association_function_option
@replay_options(mode="eval", parallel=True)
@click.option(
    "--show",
    is_flag=True,
    default=False,
    help=(
        "Preview tracking at source timing, after calibration when --calibrate-kf is enabled. "
        "Sensor replay displays tracked masks, IDs, and classes."
    ),
)
@click.option(
    "--save",
    is_flag=True,
    default=False,
    help=(
        "Save annotated tracking videos from replay, after calibration when --calibrate-kf is enabled. "
        "EagerMOT writes one MP4 per sequence under results/videos."
    ),
)
@click.option(
    "--show-3d",
    is_flag=True,
    default=False,
    help="EagerMOT sensor evaluation: overlay estimated tracked 3D cuboids; requires --show or --save.",
)
@kalman_calibration_option(mode="eval")
@sequence_option
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
    eval_masks: bool,
    calibrate_kf: bool,
    **kwargs: Any,
) -> None:
    """Evaluate a tracker, materializing the selected configuration when needed."""

    _require_replay_input(experiment, dataset, "eval")
    if kwargs["show_3d"] and not (kwargs["show"] or kwargs["save"]):
        raise click.UsageError("--show-3d requires --show or --save.")

    sensor_payload = _prepare_sensor_evaluation(
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
            "sequence_names": sequence_names,
            "allow_noncanonical_build": allow_noncanonical_build,
            "compare_trackeval": compare_trackeval,
            "eval_masks": eval_masks,
            "calibrate_kf": calibrate_kf,
        },
    )
    if sensor_payload is not None:
        _dispatch_cli_workflow(ctx, "eval", "boxmot.engine.eval.evaluator", sensor_payload)
        return
    if kwargs["class_config"] is not None or kwargs["show_3d"]:
        raise click.UsageError("--class-config and --show-3d require a Sensor dataset dataset with --tracker eagermot.")

    if calibrate_kf or kwargs.get("tracker_config") is not None or kwargs.get("variable_dt") is not None:
        from boxmot.engine.calibration.kalman import validate_kf_calibration
        from boxmot.engine.config.trackers import resolve_tracker_options
        from boxmot.trackers.common.specs import parse_tracker_spec

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
        eval_masks=eval_masks,
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
            "eval_masks": eval_masks,
            "calibrate_kf": calibrate_kf,
        },
    )


__all__ = ("eval",)
