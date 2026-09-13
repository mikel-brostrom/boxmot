"""Construction and validation for canonical tracker specifications."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from boxmot.components.resolution import component_options
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
_SPEC_FIELDS = frozenset({"backend", "geometry", "per_class", "class_ids", "class_names"})


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

    for input_name in ("embeddings", "masks", "frame", "detections_3d", "camera", "ego_motion"):
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


def _resolve_spec(spec: TrackerSpec | str, overrides: Mapping[str, Any]) -> TrackerSpec:
    """Merge ergonomic factory arguments into a new immutable tracker spec."""
    if isinstance(spec, str):
        spec = TrackerSpec(spec)
    elif not isinstance(spec, TrackerSpec):
        raise TypeError(f"spec must be TrackerSpec or a tracker name, got {type(spec).__name__}.")
    if not overrides:
        return spec

    supplied = dict(overrides)
    metadata = {name: supplied.pop(name) for name in _SPEC_FIELDS if name in supplied}
    option_mapping = supplied.pop("options", {})
    if not isinstance(option_mapping, Mapping):
        raise TypeError("options must be a mapping of tracker-algorithm parameters.")
    misplaced = set(option_mapping).intersection(_SPEC_FIELDS)
    if misplaced:
        raise ValueError("Pass tracker selection fields outside options: " + ", ".join(sorted(misplaced)))
    if "is_obb" in supplied or "is_obb" in option_mapping:
        raise ValueError("Select tracker geometry with geometry='aabb' or geometry='obb', not is_obb.")
    options = {**spec.option_dict, **option_mapping, **supplied}
    _validate_model_options(spec.name, options)

    if "class_ids" in metadata and metadata["class_ids"] is not None:
        values = metadata["class_ids"]
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values
        ):
            raise TypeError("class_ids must be a list or tuple of non-negative integers, or None.")
        metadata["class_ids"] = tuple(sorted(set(values)))
    if "class_names" in metadata:
        values = metadata["class_names"]
        entries = tuple(values.items()) if isinstance(values, Mapping) else values
        if not isinstance(entries, (list, tuple)) or any(
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or not isinstance(entry[0], int)
            or isinstance(entry[0], bool)
            or entry[0] < 0
            or not isinstance(entry[1], str)
            or not entry[1]
            for entry in entries
        ):
            raise TypeError("class_names must map non-negative integer IDs to non-empty names.")
        metadata["class_names"] = tuple(sorted(tuple(entry) for entry in entries))
    return replace(spec, **metadata, options=component_options(options))


def _validate_model_options(name: str, options: Mapping[str, Any]) -> None:
    """Keep encoder/model ownership outside tracking algorithm configuration."""
    model_options = sorted(set(options) & _REID_MODEL_OPTIONS)
    if not model_options:
        return
    if not get_tracker_definition(name).capabilities.accepts_embeddings:
        raise ValueError(f"Tracker {name!r} does not accept ReID model options: " + ", ".join(model_options))
    raise ValueError(
        "TrackerSpec accepts tracker-algorithm options only; configure ReID on the created "
        "tracker instead: " + ", ".join(model_options)
    )


def create_tracker(spec: TrackerSpec | str, **overrides: Any) -> Tracker:
    """Create a tracker from its registered name or an immutable specification.

    Keyword arguments override specification fields or tracker-algorithm options.
    An optional ``options`` mapping overlays spec options; direct keywords win.
    Omitted values retain the spec selection and the tracker's configured defaults.

    The factory does not construct models eagerly. Trackers consume required
    masks, embeddings, or frames through their structured update boundary;
    every ReID-enabled high-level tracker can additionally build its own ReID
    backend lazily when called without precomputed embeddings.
    """

    spec = _resolve_spec(spec, overrides)
    definition = get_tracker_definition(spec.name)
    tracker_args = load_tracker_config(definition.config_name or definition.name, None, spec.option_dict)
    normalize_kalman_options(
        tracker_args,
        variable_dt=tracker_args.get("variable_dt", False),
        tracker_name=spec.name,
        backend=spec.backend,
    )
    geometry_kind = _validate_geometry(spec, definition)
    _validate_model_options(spec.name, spec.option_dict)
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
