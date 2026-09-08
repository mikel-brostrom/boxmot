"""Resolve engine runtime overrides without changing calibrated time units."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from boxmot.trackers.config import load_tracker_config


def resolve_tracker_options(
    args: Any,
    overrides: Mapping[str, Any] | None = None,
    *,
    include_defaults: bool = False,
    stamp_timing: bool = False,
) -> dict[str, Any]:
    """Apply configuration and explicit flags, then validate the final time basis.

    Saved calibration files contain concrete units, so a conflicting timing
    override fails here, before replay or model initialization. Uncalibrated
    defaults may leave the units unspecified until their timing mode is chosen.
    """
    tracker_name = getattr(args, "tracker", None)
    reference = getattr(args, "tracker_config", None)
    options = (
        load_tracker_config(str(tracker_name), reference, overrides)
        if include_defaults or reference is not None
        else dict(overrides or {})
    )
    if getattr(args, "asso_func", None):
        options["asso_func"] = str(args.asso_func)
    if getattr(args, "variable_dt", None) is not None:
        options["variable_dt"] = args.variable_dt
    if tracker_name is None:
        return options

    from boxmot.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES, normalize_kalman_options

    effective = (
        options if include_defaults or reference is not None else load_tracker_config(tracker_name, None, options)
    )
    variable_dt = effective.get("variable_dt", False)
    noise = normalize_kalman_options(
        effective,
        variable_dt=variable_dt,
        tracker_name=tracker_name,
        backend=str(getattr(args, "tracker_backend", "python")),
    )
    if stamp_timing and tracker_name in KALMAN_TRACKER_NAMES:
        options.update(
            variable_dt=variable_dt,
            kf_time_unit=noise.time_unit,
            kf_reference_dt_s=noise.reference_dt_s,
        )
    return options
