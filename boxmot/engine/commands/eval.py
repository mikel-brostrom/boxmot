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
        "calibrate_kf",
        "cache_inputs",
        "show",
        "save",
        "show_3d",
        "device",
        "sequence_workers",
        "eval_masks",
        "eval_3d",
        "eval_ap",
        "per_class",
        "verbose",
        "show_timing",
    }
)


def _prepare_saved_2d_evaluation(ctx: click.Context, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate a saved box dataset before importing replay or appearance models."""
    reference = payload.get("dataset")
    experiment = payload.get("experiment")
    if not reference and not experiment:
        return None

    from boxmot.datasets.config import load_dataset_config
    from boxmot.engine.config.datasets import is_saved_2d_dataset, load_saved_2d_evaluation_inputs
    from boxmot.engine.config.trackers import resolve_tracker_options
    from boxmot.trackers.common.config import load_tracker_config
    from boxmot.trackers.common.registry import get_tracker_definition
    from boxmot.trackers.common.specs import parse_tracker_spec

    explicit = _explicit_cli_keys(ctx)
    try:
        if experiment:
            from boxmot.engine.config.experiments import resolve_experiment_config

            components = explicit & {"detector", "reid"}
            if components:
                names = " and ".join("--" + name for name in sorted(components))
                raise ValueError(
                    f"{names} cannot be combined with --experiment because experiment YAML fixes perception components."
                )
            resolved = resolve_experiment_config(experiment, split=payload.get("split"), mode="eval")
            selected = resolved["dataset"]
            if not is_saved_2d_dataset(selected, selected["split"]):
                return None
            reference = selected.get("config_path", selected["id"])
            payload = {
                **payload,
                "experiment": str(resolved["source_path"]),
                "experiment_id": resolved["id"],
                "split": selected["split"],
                "reid": None if resolved["reid"] is None else resolved["reid"]["config_path"],
            }
        config = load_dataset_config(reference)
        if not is_saved_2d_dataset(config, payload.get("split")):
            return None
        spec = parse_tracker_spec(payload["tracker"], default_backend=payload["tracker_backend"])
        definition = get_tracker_definition(spec.name)
        if spec.backend == "cpp" and definition.native_class_path is None:
            raise ValueError(f"Tracker '{spec.name}' has no C++ backend.")
        if definition.capabilities.requires_masks:
            raise ValueError(
                f"Tracker '{spec.name}' requires instance masks; this dataset explicitly selects only boxes."
            )
        options = resolve_tracker_options(SimpleNamespace(**dict(payload)), include_defaults=True, factory_options=True)
        allowed = {
            "experiment",
            "dataset",
            "data_root",
            "tracker",
            "tracker_backend",
            "tracker_config",
            "reid",
            "device",
            "split",
            "sequence_names",
            "project",
            "cache_inputs",
            "asso_func",
            "variable_dt",
            "per_class",
            "sequence_workers",
            "show",
            "save",
            "verbose",
            "show_timing",
        }
        unsupported = explicit - allowed
        if unsupported:
            names = ", ".join(
                option.opts[0]
                for option in ctx.command.params
                if isinstance(option, click.Option) and option.name in unsupported
            )
            raise ValueError(f"Saved 2D evaluation does not support {names}; predictions come from the dataset YAML.")
        if "sequence_workers" in explicit and payload.get("sequence_workers") != 1:
            raise ValueError("Saved 2D evaluation currently requires --sequence-workers 1.")
        if spec.backend == "cpp" and payload.get("per_class"):
            raise ValueError("Native trackers do not support --per-class.")
        effective_options = load_tracker_config(definition.config_name or spec.name, None, options)
        needs_embeddings = definition.capabilities.requires_embeddings or effective_options.get("use_embeddings", False)
        if needs_embeddings and not payload.get("reid"):
            raise ValueError(
                f"Tracker '{spec.name}' requires appearance. Add --reid PROFILE to encode the saved boxes."
            )
        if not needs_embeddings and payload.get("reid"):
            raise ValueError(f"Tracker '{spec.name}' does not use embeddings in this configuration; omit --reid.")
        dataset = load_saved_2d_evaluation_inputs(
            reference,
            split=payload.get("split"),
            sequence_names=payload.get("sequence_names", ()),
            data_root=payload.get("data_root"),
        )
    except (TypeError, ValueError, OSError) as exc:
        raise click.UsageError(str(exc)) from exc
    return {
        **payload,
        "dataset": dataset.config_path,
        "split": dataset.split,
        "sequence_names": dataset.sequence_names,
        "tracker": spec.name,
        "tracker_backend": spec.backend,
        "sequence_workers": 1,
        "saved_detections": True,
    }


def _prepare_sensor_evaluation(ctx: click.Context, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize a declared sensor dataset before dispatch through the shared evaluator."""
    reference = payload.get("dataset")
    if not reference:
        return None

    from boxmot.datasets.inputs import resolve_sensor_dataset_config_path
    from boxmot.engine.config.datasets import load_sensor_evaluation_inputs, validate_sensor_workflow_inputs
    from boxmot.trackers.common.specs import parse_tracker_spec

    explicit = _explicit_cli_keys(ctx)
    try:
        path = resolve_sensor_dataset_config_path(reference, split=payload.get("split"))
        if path is None:
            return None
        spec = parse_tracker_spec(payload["tracker"], default_backend=payload["tracker_backend"])
        validate_sensor_workflow_inputs(
            path,
            spec,
            mode="eval",
            split=payload.get("split"),
            calibrate_kf=bool(payload.get("calibrate_kf")),
            eval_3d=bool(payload.get("eval_3d")),
            eval_ap=bool(payload.get("eval_ap")),
        )
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
            eval_3d=bool(payload.get("eval_3d")),
            eval_ap=bool(payload.get("eval_ap")),
            calibrate_kf=bool(payload.get("calibrate_kf")),
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
        "eval_masks": not payload.get("eval_3d", False),
    }


@click.command(name="eval", help="Evaluate tracking performance from a perception build or saved dataset predictions.")
@replay_build_options(dataset_default=BOXMOT_DEFAULTS.eval.dataset)
@data_root_option
@split_option
@dataset_fps_option
@eval_masks_option
@click.option(
    "--eval-3d",
    is_flag=True,
    default=False,
    help="EagerMOT: 3D tracking metrics from KITTI tracking ground truth.",
)
@click.option(
    "--eval-ap",
    is_flag=True,
    default=False,
    help="Add official KITTI 2D/3D AP40; requires --eval-3d and aligned per-image object ground truth.",
)
@tracker_backend_option(default=BOXMOT_DEFAULTS.eval.tracker_backend)
@tracker_config_option
@click.option(
    "--class-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="EagerMOT: car and pedestrian profiles from tuning's best.yaml or KF calibration's calibrated.yaml.",
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
    if kwargs["eval_ap"] and not kwargs["eval_3d"]:
        raise click.UsageError("--eval-ap requires --eval-3d.")
    if kwargs["eval_3d"] and eval_masks:
        raise click.UsageError("Choose either --eval-3d or --eval-masks.")
    if kwargs["show_3d"] and not (kwargs["show"] or kwargs["save"]):
        raise click.UsageError("--show-3d requires --show or --save.")

    dataset_payload = {
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
    }
    saved_2d_payload = _prepare_saved_2d_evaluation(ctx, dataset_payload)
    if saved_2d_payload is not None:
        _dispatch_cli_workflow(ctx, "eval", "boxmot.engine.eval.saved_detections", saved_2d_payload)
        return
    sensor_payload = _prepare_sensor_evaluation(ctx, dataset_payload)
    if sensor_payload is not None:
        _dispatch_cli_workflow(ctx, "eval", "boxmot.engine.eval.evaluator", sensor_payload)
        return
    for name in ("class_config", "show_3d", "eval_ap", "eval_3d"):
        if kwargs[name]:
            option = "--" + name.replace("_", "-")
            raise click.UsageError(f"{option} requires a sensor dataset with --tracker eagermot.")

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
        tracker_config=kwargs.get("tracker_config"),
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
