"""Shared plumbing for lightweight engine CLI command adapters."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import click
from click.core import ParameterSource

_WORKFLOW_SETUP_TITLES = {
    "boxmot.engine.materialization.workflow": "Dataset Materialization",
    "boxmot.engine.eval.evaluator": "Evaluation",
}


@contextmanager
def _workflow_setup(title: str | None, detail: str) -> Iterator[None]:
    """Paint terminal setup before runtime imports, then release the workflow's Live.

    The transient panel covers work before a workflow can create its own
    reporter. Captured commands remain quiet, and errors retain Click's normal
    handling. No reporter is attached to namespaces passed to worker processes.
    """

    from boxmot.engine.ui.core.ui import create_workflow_progress, get_console
    from boxmot.engine.ui.workflow.steps import SETUP, compose

    if title is None or not get_console(stderr=True).is_terminal:
        yield
        return

    progress = create_workflow_progress(title, (), steps=compose(SETUP), stderr=True, transient=True)
    progress.set_detail("Setup", detail)
    try:
        progress.start()
        yield
    finally:
        progress.stop(refresh_final=False)


def _is_option_explicit(ctx: click.Context, option_name: str) -> bool:
    """Return whether Click received an option from somewhere other than its default."""

    return ctx.get_parameter_source(option_name) != ParameterSource.DEFAULT


def _explicit_cli_keys(ctx: click.Context) -> set[str]:
    """Return option names explicitly supplied on the command line."""

    return {
        param.name
        for param in ctx.command.params
        if isinstance(param, click.Option) and _is_option_explicit(ctx, param.name)
    }


def _build_cli_namespace(
    ctx: click.Context,
    mode: str,
    payload: Mapping[str, Any],
):
    """Build a normalized engine namespace while retaining explicit CLI provenance."""

    from boxmot.engine.config import build_mode_namespace

    return build_mode_namespace(mode, payload, explicit_keys=_explicit_cli_keys(ctx))


def _run_engine_workflow(module_name: str, args: Any) -> Any:
    """Run an engine module through its canonical ``main(args)`` entry point.

    Engine workflows render their own Rich panels. When a panel has already
    rendered an error, it marks the exception so Click can exit cleanly without
    printing a second traceback.
    """

    if getattr(args, "tracker", None) is not None:
        from boxmot.engine.tracker_config import validate_image_tracker

        try:
            validate_image_tracker(str(args.tracker))
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    with _workflow_setup(_WORKFLOW_SETUP_TITLES.get(module_name), "Preparing workflow…"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise click.ClickException(
                f"Failed to import engine module '{module_name}': {exc}\n"
                "Install the required feature extra while repeating one PyTorch profile; "
                "for example: uv sync --extra cpu --extra yolo"
            ) from exc

    main_fn = getattr(module, "main", None)
    if main_fn is None:
        raise AttributeError(f"{module_name} does not expose main")

    try:
        return main_fn(args)
    except (KeyboardInterrupt, SystemExit, click.exceptions.Exit, click.ClickException):
        raise
    except BaseException as exc:
        if getattr(exc, "_workflow_rendered_error", False):
            raise click.exceptions.Exit(code=1) from exc
        raise


def _dispatch_cli_workflow(
    ctx: click.Context,
    mode: str,
    module_name: str,
    payload: Mapping[str, Any],
) -> Any:
    """Normalize CLI values and dispatch them to an engine workflow."""

    args = _build_cli_namespace(ctx, mode, payload)
    return _run_engine_workflow(module_name, args)


def _require_experiment_input(experiment: str | None, command_name: str) -> str:
    """Require an experiment config for experiment-only cached workflows."""

    if not experiment:
        raise click.UsageError(
            f"{command_name} requires --experiment <experiment-yaml>. "
            f"Materialize and consume builds with the same authored experiment."
        )
    return experiment


def _require_replay_input(experiment: str | None, dataset: str | None, command_name: str) -> None:
    """Require exactly one authored experiment or dataset for cached replay."""

    if experiment and dataset:
        raise click.UsageError(
            f"{command_name} accepts either --dataset <dataset-id-or-yaml> or --experiment <experiment-yaml>, not both."
        )
    if not experiment and not dataset:
        raise click.UsageError(
            f"{command_name} requires either --dataset <dataset-id-or-yaml> or --experiment <experiment-yaml>."
        )


def _prepare_replay_build(
    ctx: click.Context,
    *,
    mode: str,
    experiment: str | None,
    dataset: str | None,
    detector: str | None,
    reid: str | None,
    build_ref: str | None,
    build_root: Path | None,
    device: str,
    data_root: Path | None,
    split: str | None,
    tracker: str,
    fps: float | None,
    eval_masks: bool = False,
    allow_noncanonical_build: bool = False,
) -> tuple[str | None, str | None, str | Path]:
    """Resolve replay inputs and reuse or create one canonical build before dispatch."""

    from boxmot.engine.experiment_config import (
        ConfigurationError,
        resolve_experiment_config,
        resolve_matching_experiment_path,
    )
    from boxmot.engine.tracker_config import validate_image_tracker
    from boxmot.trackers.registry import get_tracker_definition

    try:
        validate_image_tracker(tracker)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    components = tuple(name for name, value in (("--detector", detector), ("--reid", reid)) if value)
    if experiment and components:
        names = " and ".join(components)
        raise click.UsageError(
            f"{names} cannot be combined with --experiment because experiment YAML fixes perception components."
        )
    if reid and not detector:
        raise click.UsageError("--reid requires --detector when selecting perception components directly.")
    if detector and not dataset:
        raise click.UsageError("--detector requires --dataset when selecting perception components directly.")
    if build_ref is None and dataset and not detector:
        raise click.UsageError(
            f"{mode} with --dataset requires --detector for automatic materialization, or --build to replay an "
            "existing materialized build."
        )
    if build_ref is not None and _is_option_explicit(ctx, "device"):
        raise click.UsageError("--device applies only when --build is omitted for automatic materialization.")
    if build_ref is None and allow_noncanonical_build:
        raise click.UsageError("--allow-noncanonical-build requires an explicit --build.")

    if detector is not None:
        title = "Tuning" if mode == "tune" else "Evaluation"
        with _workflow_setup(title, "Resolving experiment…"):
            try:
                experiment = str(
                    resolve_matching_experiment_path(
                        dataset=str(dataset), detector=detector, reid=reid, split=split, mode=mode
                    )
                )
            except (ConfigurationError, FileNotFoundError) as exc:
                raise click.UsageError(str(exc)) from exc
        dataset = None

    if build_ref is not None:
        return experiment, dataset, build_ref

    if eval_masks:
        try:
            resolved = resolve_experiment_config(str(experiment), split=split, mode=mode)
        except (ConfigurationError, FileNotFoundError) as exc:
            raise click.UsageError(str(exc)) from exc
        if resolved["dataset"]["layout"] != "kitti-mots":
            raise click.UsageError("--eval-masks requires a KITTI-MOTS dataset.")

    capabilities = get_tracker_definition(tracker).capabilities
    materialize_args = _build_cli_namespace(
        ctx,
        "materialize",
        {
            "experiment": experiment,
            "data_root": data_root,
            "build_root": build_root,
            "device": device,
            "fps": fps,
            "publish_image_refs": True,
            "publish_masks": capabilities.requires_masks or eval_masks,
            "publish_embeddings": capabilities.accepts_embeddings,
            "plan_path": None,
            "plan_overrides": (),
            "resume": True,
        },
    )
    materialize_args.materialize_split = split
    materialize_args.materialize_mode = mode
    build_ref = _run_engine_workflow("boxmot.engine.materialization.workflow", materialize_args)
    return experiment, dataset, build_ref


__all__ = (
    "_build_cli_namespace",
    "_dispatch_cli_workflow",
    "_explicit_cli_keys",
    "_is_option_explicit",
    "_prepare_replay_build",
    "_require_experiment_input",
    "_require_replay_input",
    "_run_engine_workflow",
    "_workflow_setup",
)
