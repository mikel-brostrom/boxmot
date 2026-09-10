"""Prepare saved-sensor inputs for the shared tuning workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import click

from boxmot.engine.commands._support import _explicit_cli_keys

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
        raise click.UsageError(f"KITTI fusion tuning does not support {names}; inputs come from the dataset manifest.")
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
                f"KITTI fusion tuning runs serial Optuna trials on CPU; {option} must be one of: {choices}."
            )
    for name in ("objectives", "maximize"):
        if name in explicit:
            metrics = [metric for value in payload[name] for metric in value.replace(",", " ").split()]
            if metrics != ["HOTA"]:
                raise click.UsageError(f"KITTI fusion tuning optimizes class-average mask HOTA; --{name} must be HOTA.")


def prepare_sensor_tuning(ctx: click.Context, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize a declared sensor dataset before dispatch through the shared tuner."""
    reference = payload.get("dataset")
    if not reference:
        return None

    from boxmot.datasets.kitti_fusion_config import load_kitti_fusion_dataset, resolve_kitti_fusion_config_path
    from boxmot.engine.config.runtime import get_mode_default, resolve_sequence_workers
    from boxmot.trackers.common.specs import parse_tracker_spec

    try:
        path = resolve_kitti_fusion_config_path(reference)
        if path is None:
            return None
        spec = parse_tracker_spec(payload["tracker"], default_backend=payload["tracker_backend"])
        if spec.name != "eagermot" or spec.backend != "python":
            raise ValueError("KITTI fusion tuning requires --tracker eagermot --tracker-backend python.")
        _validate_sensor_options(ctx, payload)
        dataset = load_kitti_fusion_dataset(
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
