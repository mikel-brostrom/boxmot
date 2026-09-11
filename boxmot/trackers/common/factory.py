"""Construction and validation for canonical tracker specifications."""

from __future__ import annotations

import importlib
from typing import Any

from boxmot.structures import GeometryKind
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.noise import normalize_kalman_options
from boxmot.trackers.common.protocols import Tracker, TrackerRequirements
from boxmot.trackers.common.registry import (
    TrackerDefinition,
    _load_tracker_class,
    get_tracker_definition,
    supported_native_trackers,
)
from boxmot.trackers.common.specs import TrackerCapabilities, TrackerSpec

_REID_MODEL_OPTIONS = frozenset(
    {
        "device",
        "half",
        "model",
        "reid_device",
        "reid_model",
        "reid_model_path",
        "reid_preprocess",
        "reid_weights",
        "weights",
    }
)
_FORBIDDEN_NATIVE_MASK_OPTIONS = frozenset({"masks", "supports_masks", "use_masks"})
_UNSUPPORTED_NATIVE_OPTIONS = {
    "botsort": frozenset({"removed_stracks_buffer"}),
    "occluboost": frozenset({"adaptive_kf"}),
}


def _load_native_tracker_class(definition: TrackerDefinition) -> type[Any]:
    if definition.native_class_path is None:
        available = ", ".join(supported_native_trackers())
        raise ValueError(f"Native backend is unavailable for {definition.name!r}; choose from: {available}.")
    module_path, class_name = definition.native_class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _validate_geometry(spec: TrackerSpec, definition: TrackerDefinition) -> GeometryKind:
    geometry_kind = GeometryKind(spec.geometry)
    supported = definition.capabilities.geometry_kinds
    if geometry_kind not in supported:
        supported_names = [kind.value for kind in sorted(supported, key=lambda kind: kind.value)]
        raise ValueError(
            f"Tracker {spec.name!r} does not support geometry kind {geometry_kind.value!r}. "
            f"Supported geometry kinds: {supported_names}."
        )
    return geometry_kind


def _bind_and_validate_capabilities(tracker: Tracker, capabilities: TrackerCapabilities) -> Tracker:
    requirements = getattr(tracker, "requirements", None)
    if not isinstance(requirements, TrackerRequirements):
        raise TypeError("A tracker factory result must expose immutable TrackerRequirements.")

    for input_name in ("embeddings", "masks", "frame", "detections_3d", "camera"):
        required = getattr(requirements, input_name)
        accepts = getattr(capabilities, f"accepts_{input_name}")
        always_required = getattr(capabilities, f"requires_{input_name}")
        if required and not accepts:
            raise ValueError(
                f"Resolved tracker requires {input_name}, but its static capabilities do not accept {input_name}."
            )
        if always_required and not required:
            raise ValueError(f"Resolved tracker omitted {input_name}, which its static capabilities require.")

    declared = getattr(tracker, "capabilities", None)
    if declared is not None and declared != capabilities:
        raise ValueError("Tracker implementation capabilities disagree with its registry definition.")
    if declared is None:
        try:
            setattr(tracker, "capabilities", capabilities)
        except (AttributeError, TypeError) as exc:
            raise TypeError("A tracker factory result must expose its registered capabilities.") from exc
    return tracker


def _create_native_tracker(
    spec: TrackerSpec,
    definition: TrackerDefinition,
    geometry_kind: GeometryKind,
) -> Tracker:
    """Validate and construct a C++ tracker through its domain-owned adapter."""

    if definition.native_class_path is None:
        available = ", ".join(supported_native_trackers())
        raise ValueError(f"Native backend is unavailable for {definition.name!r}; choose from: {available}.")
    if spec.per_class:
        raise ValueError("Native trackers do not support per_class mode.")
    if geometry_kind not in definition.native_geometry_kinds:
        raise ValueError(f"Native {spec.name} does not support {geometry_kind.value.upper()} geometry.")
    if definition.capabilities.accepts_masks:
        raise ValueError(f"Native {spec.name} does not support masks.")
    variable_dt = spec.option_dict.get("variable_dt", False)
    if not isinstance(variable_dt, bool):
        raise TypeError("variable_dt must be bool.")
    if variable_dt:
        raise ValueError("The native tracker backend does not support variable_dt=True; use the Python backend.")

    option_names = set(spec.option_dict)
    mask_options = sorted(option_names & _FORBIDDEN_NATIVE_MASK_OPTIONS)
    if mask_options:
        raise ValueError("Native trackers do not support masks: " + ", ".join(mask_options))
    unsupported_options = sorted(option_names & _UNSUPPORTED_NATIVE_OPTIONS.get(spec.name, frozenset()))
    if unsupported_options:
        raise ValueError(
            f"Native {spec.name} does not implement these tracker options: " + ", ".join(unsupported_options)
        )

    tracker_class = _load_native_tracker_class(definition)
    return tracker_class(spec.option_dict, geometry=geometry_kind.value)


def create_tracker(spec: TrackerSpec) -> Tracker:
    """Create one tracker from an immutable canonical specification.

    The factory does not construct models eagerly. Trackers consume required
    masks, embeddings, or frames through their structured update boundary;
    every ReID-enabled high-level tracker can additionally build its own ReID
    backend lazily when called without precomputed embeddings.
    """

    if not isinstance(spec, TrackerSpec):
        raise TypeError(f"spec must be TrackerSpec, got {type(spec).__name__}.")

    definition = get_tracker_definition(spec.name)
    tracker_args = load_tracker_config(definition.config_name or definition.name, None, spec.option_dict)
    normalize_kalman_options(
        tracker_args,
        variable_dt=tracker_args.get("variable_dt", False),
        tracker_name=spec.name,
        backend=spec.backend,
    )
    geometry_kind = _validate_geometry(spec, definition)
    model_options = sorted(set(spec.option_dict) & _REID_MODEL_OPTIONS)
    if model_options:
        if not definition.capabilities.accepts_embeddings:
            raise ValueError(f"Tracker {spec.name!r} does not accept ReID model options: " + ", ".join(model_options))
        raise ValueError(
            "TrackerSpec accepts tracker-algorithm options only; configure ReID on the created "
            "tracker instead: " + ", ".join(model_options)
        )
    if spec.backend == "cpp":
        tracker = _create_native_tracker(spec, definition, geometry_kind)
        return _bind_and_validate_capabilities(tracker, definition.capabilities)

    tracker_args["is_obb"] = geometry_kind is GeometryKind.OBB
    if definition.accepts_per_class:
        tracker_args["per_class"] = spec.per_class
    if spec.class_ids is not None:
        tracker_args["class_ids"] = spec.class_ids
    if spec.class_names:
        tracker_args["class_names"] = dict(spec.class_names)

    tracker_class = _load_tracker_class(definition)
    tracker = tracker_class(**tracker_args)
    return _bind_and_validate_capabilities(tracker, definition.capabilities)


__all__ = ("create_tracker",)
