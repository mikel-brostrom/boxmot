"""Condition tracker searches on optional temporal EdgeTAM guidance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from boxmot.engine.config.trackers import edgetam_checkpoint
from boxmot.engine.tuning.search_space import expand_yaml_groups
from boxmot.segmentors.propagation.factory import mask_propagation_device
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.mask_guidance import (
    MASK_GUIDANCE_OPTIONS,
    mask_guidance_config_from_options,
    validate_mask_guidance_spec,
)
from boxmot.trackers.common.specs import TrackerSpec
from boxmot.utils.devices import normalize_device


def prepare_mask_guidance_tuning(
    args: Any, runtime_options: Mapping[str, Any], *, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Select IoU for guided searches while respecting explicit incompatible choices."""
    options = dict(runtime_options)
    checkpoint = edgetam_checkpoint(args)
    if checkpoint is None:
        return {key: value for key, value in options.items() if key not in MASK_GUIDANCE_OPTIONS}
    device = normalize_device(mask_propagation_device(checkpoint, str(getattr(args, "device", "cpu"))))
    if device.startswith("cuda:") and device != "cuda:0":
        raise ValueError(
            "Mask guidance tuning uses Ray-allocated GPUs and requires logical cuda:0. "
            "Select physical GPUs with CUDA_VISIBLE_DEVICES before starting tuning, then use --device cuda:0."
        )

    authored = load_tracker_config(
        args.tracker, getattr(args, "tracker_config", None), overrides, include_defaults=False
    )
    association = getattr(args, "asso_func", None) or authored.get("asso_func")
    if association is not None and association != "iou":
        raise ValueError("Mask guidance tuning requires asso_func='iou'; remove the incompatible override.")
    options["asso_func"] = "iou"
    spec = TrackerSpec(
        name=args.tracker,
        backend=getattr(args, "tracker_backend", "python"),
        geometry=getattr(args, "geometry", "aabb"),
        per_class=bool(getattr(args, "per_class", False)),
        options=tuple(sorted(options.items())),
    )
    validate_mask_guidance_spec(spec, resolved_options=options)
    if getattr(args, "eval_masks", False):
        raise ValueError("Mask guidance tuning evaluates bounding-box tracks; omit --eval-masks.")
    mask_guidance_config_from_options(checkpoint, device, options)
    args.asso_func = "iou"
    if getattr(args, "sequence_workers", None) is None:
        args.sequence_workers = 1
    return options


def condition_mask_guidance_schema(
    schema: Mapping[str, Any], args: Any, runtime_options: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep inactive knobs out of every backend and fix the required association mode."""
    enabled = edgetam_checkpoint(args) is not None
    conditioned = expand_yaml_groups(dict(schema))
    if getattr(args, "asso_func", None) is not None:
        conditioned["asso_func"] = {"default": args.asso_func}
    if not enabled or getattr(args, "tracker_backend", "python") != "python":
        conditioned = {key: entry for key, entry in conditioned.items() if key not in MASK_GUIDANCE_OPTIONS}
    if enabled:
        conditioned["asso_func"] = {"default": "iou"}
    if enabled and getattr(args, "mask_guidance_max_objects", None) is not None:
        conditioned["edgetam.max_objects"] = {"default": runtime_options["edgetam.max_objects"]}
    return conditioned


def mask_guidance_trial_resources(args: Any) -> dict[str, int]:
    """Reserve a CUDA accelerator for temporal inference even for detector-free replay."""
    checkpoint = edgetam_checkpoint(args)
    guided_cuda = checkpoint is not None and normalize_device(
        mask_propagation_device(checkpoint, str(getattr(args, "device", "cpu")))
    ).startswith("cuda:")
    return {"cpu": int(args.sequence_workers), "gpu": int(guided_cuda)}


def record_mask_guidance_tuning(
    directory: Path, args: Any, options: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    """Reject resuming results with different effective temporal guidance settings."""
    checkpoint = edgetam_checkpoint(args)
    profile: dict[str, Any] = {"enabled": checkpoint is not None}
    if checkpoint is not None:
        from boxmot.engine.eval.provenance import _mask_guidance_identity
        from boxmot.segmentors.propagation.weights import resolve_edgetam_artifact

        args.mask_guidance_weights = resolve_edgetam_artifact(checkpoint)
        profile.update(
            identity=_mask_guidance_identity(
                args.mask_guidance_weights,
                str(getattr(args, "device", "cpu")),
                tracker_spec=TrackerSpec(args.tracker, options=tuple(sorted(options.items()))),
            ),
            search={key: schema[key] for key in MASK_GUIDANCE_OPTIONS if key in schema},
        )
    # Normalize tuples to their persisted JSON representation before comparison.
    profile = json.loads(json.dumps(profile, sort_keys=True))
    path = directory / "mask-guidance.json"
    if getattr(args, "resume_tune", None):
        if not path.is_file():
            if checkpoint is not None:
                raise ValueError("Saved tuning run lacks mask guidance metadata; start a new tuning run.")
            return
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Saved mask guidance tuning metadata is invalid; start a new tuning run.") from exc
        if saved != profile:
            raise ValueError(
                "Resuming tuning requires the same mask guidance source, model, build, settings and search space. "
                "Start a new tuning run to change the guidance profile."
            )
        return
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
