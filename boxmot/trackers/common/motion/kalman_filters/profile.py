"""Portable compatibility metadata for calibrated Kalman tracker profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from boxmot.trackers.common.motion.kalman_filters.noise import normalize_kalman_options

_IMAGE_FILTERS = {
    "boosttrack": ("xyhr", "xyhr"),
    "botsort": ("xywh", "xywh"),
    "bytetrack": ("xyah", "xywh"),
    "deepocsort": ("xysr", "xysr"),
    "hybridsort": ("xyscr", "xysr"),
    "occluboost": ("xyhr", "xyhr"),
    "ocsort": ("xysr", "xysr"),
    "strongsort": ("xyah", "xywh"),
}
_PROVENANCE_FIELDS = frozenset({"dataset", "split", "source", "class_id", "class_name"})


def calibration_profile_signature(
    tracker_name: str, geometry: str, options: Mapping[str, Any], *, backend: str = "python"
) -> dict[str, str | int | float | bool]:
    """Describe the tracker input geometry and the numerical filter it selects."""
    geometry = str(geometry)
    if backend != "python":
        raise ValueError("Calibrated Kalman profiles require the Python tracker backend.")
    if geometry not in {"aabb", "obb"}:
        raise ValueError(f"Unsupported calibrated tracker geometry {geometry!r}.")
    variable_dt = options.get("variable_dt", False)
    noise = normalize_kalman_options(options, variable_dt=variable_dt, tracker_name=tracker_name, backend=backend)
    signature: dict[str, str | int | float | bool] = {
        "tracker": tracker_name,
        "backend": backend,
        "geometry": geometry,
        "variable_dt": variable_dt,
        "time_unit": noise.time_unit,
        "reference_dt_s": noise.reference_dt_s,
    }
    if tracker_name == "eagermot":
        if geometry != "aabb":
            raise ValueError("Calibrated EagerMOT profiles require AABB image detections.")
        angular = options.get("is_angular", False)
        if not isinstance(angular, bool):
            raise ValueError("Calibrated EagerMOT is_angular must be a bool.")
        return {
            **signature,
            "filter": "box3d_angular" if angular else "box3d",
            "state_dimensions": 11 if angular else 10,
            "measurement_dimensions": 7,
            "is_angular": angular,
        }
    if tracker_name not in _IMAGE_FILTERS:
        raise ValueError(f"Tracker {tracker_name!r} does not support calibrated Kalman profiles.")
    oriented = geometry == "obb"
    kind = _IMAGE_FILTERS[tracker_name][oriented]
    dim_z = 5 if oriented or kind == "xyscr" else 4
    dim_x = 2 * dim_z - 1 if kind in {"xysr", "xyscr"} else 2 * dim_z
    return {**signature, "filter": kind, "state_dimensions": dim_x, "measurement_dimensions": dim_z}


def validate_calibration_profile(
    options: Mapping[str, Any], *, tracker_name: str, geometry: str, backend: str = "python"
) -> None:
    """Reject incompatible portable profiles; fresh configurations need no metadata.

    Dataset, split, and source identify the fitted evidence but do not restrict
    reuse on held-out inputs. Filter structure and resolved time units remain
    bound to the calibration even when covariance scales are refined by tuning.
    """
    from boxmot.trackers.common.config import flatten_tracker_options

    flattened = flatten_tracker_options(options)
    metadata = {
        name.removeprefix("calibration."): value for name, value in flattened.items() if name.startswith("calibration.")
    }
    if not metadata:
        return
    expected = calibration_profile_signature(tracker_name, geometry, flattened, backend=backend)
    missing = set(expected) - set(metadata)
    if missing:
        raise ValueError(f"Calibrated tracker profile is missing metadata: {', '.join(sorted(missing))}.")
    unknown = set(metadata) - set(expected) - _PROVENANCE_FIELDS
    if unknown:
        raise ValueError(f"Unknown calibrated tracker profile metadata: {', '.join(sorted(unknown))}.")
    for name, value in expected.items():
        recorded = metadata[name]
        if type(recorded) is not type(value) or recorded != value:
            raise ValueError(
                f"Calibrated tracker profile {name}={recorded!r} is incompatible with {name}={value!r}. "
                "Use a matching profile or calibrate this tracker configuration."
            )
    for name in _PROVENANCE_FIELDS.intersection(metadata):
        value = metadata[name]
        if name == "class_id":
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else:
            valid = isinstance(value, str) and bool(value)
        if not valid:
            raise ValueError(f"Calibrated tracker profile has invalid {name} provenance.")


__all__ = ("calibration_profile_signature", "validate_calibration_profile")
