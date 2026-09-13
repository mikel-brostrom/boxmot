"""Resolve engine runtime overrides without changing calibrated time units."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from boxmot.trackers.common.config import flatten_tracker_options, load_tracker_config


def validate_image_tracker(tracker_name: str) -> None:
    """Reject tracker inputs that current image-only engine workflows cannot supply."""
    from boxmot.trackers.common.registry import get_tracker_definition

    capabilities = get_tracker_definition(tracker_name).capabilities
    if capabilities.requires_detections_3d or capabilities.requires_camera:
        raise ValueError(
            f"Tracker {tracker_name!r} requires 3D detections and a CameraModel. "
            "Image tracking and cached replay workflows cannot supply these inputs; "
            "use the tracker Python update() API."
        )


def resolve_tracker_options(
    args: Any,
    overrides: Mapping[str, Any] | None = None,
    *,
    include_defaults: bool = False,
    stamp_timing: bool = False,
    factory_options: bool = False,
) -> dict[str, Any]:
    """Apply configuration and explicit flags, then validate the final time basis.

    Saved calibration files contain concrete units, so a conflicting timing
    override fails here, before replay or model initialization. Uncalibrated
    defaults may leave the units unspecified until their timing mode is chosen.
    ``factory_options=True`` keeps native construction options sparse: their
    factory owns defaults, and unsupported Python defaults must not become
    explicit native options. Authored overrides are retained for backend
    validation, even when equal to defaults. Other callers retain full defaults
    for configuration inspection and tuning metadata when requested.
    """
    tracker_name = getattr(args, "tracker", None)
    if tracker_name is not None:
        validate_image_tracker(str(tracker_name))
    backend = str(getattr(args, "tracker_backend", "python"))
    sparse_native = factory_options and backend == "cpp"
    reference = getattr(args, "tracker_config", None)
    options = (
        load_tracker_config(str(tracker_name), reference, overrides, include_defaults=not sparse_native)
        if include_defaults or reference is not None
        else flatten_tracker_options(overrides or {})
    )
    if getattr(args, "asso_func", None):
        options["asso_func"] = str(args.asso_func)
    if getattr(args, "variable_dt", None) is not None:
        options["kalman.variable_dt"] = args.variable_dt
    if tracker_name is None:
        return options

    from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
    from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES

    effective = (
        options
        if not sparse_native and (include_defaults or reference is not None)
        else load_tracker_config(tracker_name, None, options)
    )
    variable_dt = effective.get("kalman.variable_dt", False)
    if getattr(args, "geometry", None) is not None:
        from boxmot.trackers.common.motion.kalman_filters.profile import validate_calibration_profile

        validate_calibration_profile(effective, tracker_name=tracker_name, geometry=args.geometry, backend=backend)
    kalman = normalize_kalman_config(
        effective,
        tracker_name=tracker_name,
        backend=backend,
    )
    if backend == "python" and tracker_name in KALMAN_TRACKER_NAMES and (include_defaults or stamp_timing):
        options.update(flatten_tracker_options({"kalman": kalman}))
        options["kalman.variable_dt"] = variable_dt
    elif stamp_timing and tracker_name in KALMAN_TRACKER_NAMES and not sparse_native:
        options.update(
            {
                "kalman.variable_dt": variable_dt,
                "kalman.noise.time_unit": kalman.noise.time_unit,
                "kalman.noise.reference_dt_s": kalman.noise.reference_dt_s,
            }
        )
    return options
