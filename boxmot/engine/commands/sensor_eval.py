"""Prepare saved-sensor inputs for the shared evaluation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import click

from boxmot.engine.commands._support import _explicit_cli_keys
from boxmot.engine.config.runtime import get_mode_default, resolve_sequence_workers

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


def prepare_sensor_evaluation(ctx: click.Context, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize a declared sensor dataset before dispatch through the shared evaluator."""
    reference = payload.get("dataset")
    if not reference:
        return None

    from boxmot.datasets.kitti_fusion_config import load_kitti_fusion_dataset, resolve_kitti_fusion_config_path
    from boxmot.trackers.common.specs import parse_tracker_spec

    explicit = _explicit_cli_keys(ctx)
    try:
        path = resolve_kitti_fusion_config_path(reference)
        if path is None:
            return None
        spec = parse_tracker_spec(payload["tracker"], default_backend=payload["tracker_backend"])
        if spec.name != "eagermot" or spec.backend != "python":
            raise ValueError("KITTI fusion evaluation requires --tracker eagermot --tracker-backend python.")
        unsupported = explicit - _SENSOR_OPTIONS
        if unsupported:
            names = ", ".join(
                option.opts[0]
                for option in ctx.command.params
                if isinstance(option, click.Option) and option.name in unsupported
            )
            raise click.UsageError(
                f"KITTI fusion evaluation does not support {names}; inputs come from the dataset manifest."
            )
        if "device" in explicit and payload["device"] != "cpu":
            raise click.UsageError("KITTI fusion evaluation runs on CPU; --device must be cpu.")
        dataset = load_kitti_fusion_dataset(
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
