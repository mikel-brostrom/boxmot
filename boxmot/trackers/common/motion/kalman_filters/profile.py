"""Portable compatibility metadata for calibrated Kalman tracker profiles."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from typing import Any

from boxmot.trackers.common.motion.cmc.registry import sof_options
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from boxmot.trackers.common.motion.models import resolve_motion_model_kind

_CMC_FIELDS = frozenset(
    {"cmc_policy", "cmc_rule", "cmc_min_conf", "cmc_sof_target_width", "cmc_sof_ransac_reproj_threshold"}
)
_PROVENANCE_FIELDS = frozenset({"dataset", "split", "source", "class_id", "class_name", "profile"}) | _CMC_FIELDS
KALMAN_CALIBRATION_PROFILES = ("default", "motion", "measurement", "calibrated")


def calibration_cmc_signature(options: Mapping[str, Any]) -> dict[str, int | float | None]:
    """Bind fitted noise to the resolved automatic camera estimator settings."""
    return {f"calibration.cmc_sof_{name}": value for name, value in sof_options(options).items()}


def apply_calibration_profile(
    baseline: Mapping[str, Any], calibrated: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    """Compose a covariance recipe without copying fitted metadata or policies.

    Motion includes process and initial covariance. Measurement includes R and
    its confidence curve. Classes inherit complete retained settings from their
    baseline class or pooled baseline, preventing a partial class recipe from
    accidentally resetting the other covariance group to constructor defaults.
    """
    from boxmot.trackers.common.config import flatten_tracker_options

    if profile not in KALMAN_CALIBRATION_PROFILES:
        raise ValueError(f"Unknown Kalman calibration profile: {profile!r}.")
    original = {
        name: value for name, value in flatten_tracker_options(baseline).items() if not name.startswith("calibration.")
    }
    if profile == "default":
        return deepcopy(original)
    fitted = flatten_tracker_options(calibrated)
    for name in ("kalman.type", "kalman.variable_dt", "kalman.noise.time_unit", "kalman.noise.reference_dt_s"):
        if name in original and name in fitted and original[name] != fitted[name]:
            raise ValueError(f"Calibrated recipe must preserve its baseline {name}.")
    groups = {
        "motion": ("process_", "initial_"),
        "measurement": ("measurement_",),
        "calibrated": ("process_", "initial_", "measurement_"),
    }[profile]

    def compose(base: Mapping[str, Any], fit: Mapping[str, Any]) -> dict[str, Any]:
        """Replace selected covariance fields, including omitted optional vectors."""
        return {
            **{name: value for name, value in base.items() if not name.startswith(groups)},
            **{name: value for name, value in fit.items() if name.startswith(groups)},
        }

    def noise(values: Mapping[str, Any]) -> dict[str, Any]:
        """Extract pooled noise fields without class overrides."""
        return {
            name.removeprefix("kalman.noise."): value
            for name, value in values.items()
            if name.startswith("kalman.noise.") and not name.startswith("kalman.noise.by_class.")
        }

    original_noise, fitted_noise = noise(original), noise(fitted)
    result = {name: value for name, value in original.items() if not name.startswith("kalman.noise.")}
    result.update({f"kalman.noise.{name}": value for name, value in compose(original_noise, fitted_noise).items()})
    class_ids = {
        name.split(".")[3]
        for values in (original, fitted)
        for name in values
        if name.startswith("kalman.noise.by_class.")
    }
    for class_id in sorted(class_ids, key=int):
        prefix = f"kalman.noise.by_class.{class_id}."
        original_class = {
            **original_noise,
            **{name.removeprefix(prefix): value for name, value in original.items() if name.startswith(prefix)},
        }
        fitted_class = {
            **fitted_noise,
            **{name.removeprefix(prefix): value for name, value in fitted.items() if name.startswith(prefix)},
        }
        result.update({prefix + name: value for name, value in compose(original_class, fitted_class).items()})
    return deepcopy(result)


def calibration_profile_signature(
    tracker_name: str, geometry: str, options: Mapping[str, Any], *, backend: str = "python"
) -> dict[str, str | int | float | bool]:
    """Describe the tracker input geometry and the numerical filter it selects."""
    geometry = str(geometry)
    if backend != "python":
        raise ValueError("Calibrated Kalman profiles require the Python tracker backend.")
    if geometry not in {"aabb", "obb"}:
        raise ValueError(f"Unsupported calibrated tracker geometry {geometry!r}.")
    fusion = tracker_name == "eagermot"
    kalman = normalize_kalman_config(options, tracker_name="eagermot" if fusion else tracker_name, backend=backend)
    noise = kalman.noise
    signature: dict[str, str | int | float | bool] = {
        "tracker": tracker_name,
        "backend": backend,
        "geometry": geometry,
        "variable_dt": kalman.variable_dt,
        "time_unit": noise.time_unit,
        "reference_dt_s": noise.reference_dt_s,
    }
    if fusion:
        if geometry != "aabb":
            raise ValueError("Calibrated EagerMOT profiles require AABB image detections.")
        angular = kalman.is_angular
        if not isinstance(angular, bool):
            raise ValueError("Calibrated EagerMOT is_angular must be a bool.")
        return {
            **signature,
            "filter": "box3d_angular" if angular else "box3d",
            "state_dimensions": 11 if angular else 10,
            "measurement_dimensions": 7,
            "is_angular": angular,
        }
    if tracker_name not in KALMAN_TRACKER_NAMES:
        raise ValueError(f"Tracker {tracker_name!r} does not support calibrated Kalman profiles.")
    oriented = geometry == "obb"
    kind = resolve_motion_model_kind(tracker_name, kalman.type, is_obb=oriented).value
    dim_z = (5 if kind == "xyscr" else 4) + int(oriented)
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
    from boxmot.trackers.common.config import flatten_tracker_options, load_tracker_config

    flattened = flatten_tracker_options(options)
    metadata = {
        name.removeprefix("calibration."): value for name, value in flattened.items() if name.startswith("calibration.")
    }
    if not metadata:
        return
    expected = calibration_profile_signature(tracker_name, geometry, flattened, backend=backend)
    missing = set(expected) - set(metadata)
    if missing:
        raise ValueError(
            f"Calibrated tracker profile is missing metadata: {', '.join(sorted(missing))}. "
            "Recalibrate against the current filter defaults."
        )
    unknown = set(metadata) - set(expected) - _PROVENANCE_FIELDS
    if unknown:
        raise ValueError(f"Unknown calibrated tracker profile metadata: {', '.join(sorted(unknown))}.")
    if _CMC_FIELDS.intersection(metadata) and not _CMC_FIELDS.issubset(metadata):
        raise ValueError(
            "Joint CMC calibration metadata must include policy, decision rule, mask threshold and SOF settings."
        )
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
        elif name == "profile":
            valid = value in KALMAN_CALIBRATION_PROFILES
        elif name == "cmc_min_conf":
            valid = type(value) in {int, float} and 0 <= value <= 1
        elif name == "cmc_sof_target_width":
            valid = value is None or (type(value) is int and value > 0)
        elif name == "cmc_sof_ransac_reproj_threshold":
            valid = type(value) in {int, float} and isfinite(value) and value > 0
        elif name == "cmc_policy":
            valid = isinstance(value, str) and value in {"auto", "off"}
        else:
            valid = isinstance(value, str) and bool(value)
        if not valid:
            raise ValueError(f"Calibrated tracker profile has invalid {name} provenance.")
    if _CMC_FIELDS.issubset(metadata):
        resolved = load_tracker_config(tracker_name, None, flattened)
        for name, value in calibration_cmc_signature(resolved).items():
            if flattened[name] != value:
                raise ValueError(
                    "The SOF estimator settings differ from this calibrated KF profile; run joint selection again."
                )


__all__ = (
    "KALMAN_CALIBRATION_PROFILES",
    "apply_calibration_profile",
    "calibration_cmc_signature",
    "calibration_profile_signature",
    "validate_calibration_profile",
)
