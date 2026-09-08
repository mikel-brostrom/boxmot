"""Validate and restore the fixed KF calibration of a resumed tuning run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import Any

import yaml

from boxmot.engine.tracker_config import resolve_tracker_options
from boxmot.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS, KALMAN_TIMING_OPTIONS

CALIBRATED_KF_OPTIONS = (
    *KALMAN_NOISE_OPTIONS,
    *KALMAN_TIMING_OPTIONS,
    "variable_dt",
    "adaptive_kf",
)
_REQUIRED_OPTIONS = (*KALMAN_NOISE_OPTIONS, *KALMAN_TIMING_OPTIONS, "variable_dt")


def _metadata_selection(value: Any, *, key: str) -> tuple:
    """Normalize unordered selections without accepting malformed provenance."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Saved tuning calibration has invalid {key} metadata.")
    valid = (
        all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        if key == "class_ids"
        else all(isinstance(item, str) and bool(item) for item in value)
    )
    if not valid or len(set(value)) != len(value):
        raise ValueError(f"Saved tuning calibration has invalid {key} metadata.")
    return tuple(sorted(value))


def _validate_metadata(args: Any, report: Mapping[str, Any]) -> None:
    """Require the same calibration inputs as the validated resumed evaluation."""
    expected = {
        "tracker": args.tracker,
        "geometry": args.geometry,
        "dataset": args.dataset_id,
        "split": args.split,
        "per_class": bool(getattr(args, "per_class", False)),
    }
    for name, value in expected.items():
        recorded = report.get(name)
        if recorded != value or (name == "per_class" and not isinstance(recorded, bool)):
            raise ValueError(f"Saved tuning calibration {name} differs from this run; start a new tuning run.")
    recorded_build = report.get("build")
    if (
        not isinstance(recorded_build, str)
        or not recorded_build
        or Path(recorded_build).resolve() != Path(args.build_path).resolve()
    ):
        raise ValueError("Saved tuning calibration build differs from this run; start a new tuning run.")
    selections = {
        "class_ids": list(getattr(args, "tracker_class_ids", ()) or ()),
        "sequences": list(getattr(args, "sequence_names", None) or args.seq_info),
    }
    for name, selection in selections.items():
        if _metadata_selection(report.get(name), key=name) != _metadata_selection(selection, key=name):
            raise ValueError(f"Saved tuning calibration {name} differs from this run; start a new tuning run.")


def load_tuning_calibration(
    args: Any,
    tune_dir: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Restore a completed tuning calibration without refitting or starting Ray.

    The saved profile supplies runtime defaults when no explicit tracker config
    is given. Explicit flags, configs, and programmatic overrides may change
    other tracker settings, but must preserve every fixed calibrated KF prior.
    ``args`` is not modified, and its dataset/build fields must already have
    been resolved and validated by evaluation setup.
    """
    directory = Path(tune_dir) / "kf-tuning"
    config_path, report_path = directory / "calibrated.yaml", directory / "calibration.json"
    if not config_path.exists() and not report_path.exists():
        return None
    if getattr(args, "tracker_backend", "python") != "python":
        raise ValueError("Resuming a calibrated tuning run requires the Python tracker backend.")
    if not config_path.is_file() or not report_path.is_file():
        raise ValueError("Saved tuning calibration is incomplete; calibrated.yaml and calibration.json are required.")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Saved tuning calibration report is malformed: {report_path}") from exc
    if not isinstance(report, Mapping) or report.get("status") != "complete":
        raise ValueError("Saved tuning calibration is not complete; start a new tuning run.")
    if report.get("method") != "supervised_covariance_moments":
        raise ValueError("Saved tuning calibration does not describe direct KF calibration.")
    tuning = report.get("tuning")
    if not isinstance(tuning, Mapping) or not isinstance(tuning.get("fixed_options"), Mapping):
        raise ValueError("Saved calibration is missing tuning.fixed_options; it is not a calibrated tuning run.")
    fixed = dict(tuning["fixed_options"])
    _validate_metadata(args, report)
    try:
        saved_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ValueError(f"Saved tuning calibration profile is malformed: {config_path}") from exc
    if not isinstance(saved_yaml, Mapping) or any(name not in saved_yaml for name in _REQUIRED_OPTIONS):
        raise ValueError(
            "Saved tuning calibration profile is incomplete; all KF scales and timing settings are required."
        )
    saved_args = copy(args)
    saved_args.tracker_config = config_path
    saved_args.variable_dt = None
    saved_config = resolve_tracker_options(saved_args, include_defaults=True, stamp_timing=True)
    expected_keys = {name for name in CALIBRATED_KF_OPTIONS if name in saved_config}
    if set(fixed) != expected_keys or any(name not in saved_yaml for name in expected_keys):
        raise ValueError(
            "Saved tuning calibration fixed_options must contain every applicable KF prior and timing setting."
        )
    for name, value in fixed.items():
        if saved_config[name] != value or (isinstance(saved_config[name], bool) != isinstance(value, bool)):
            raise ValueError(f"Saved tuning calibration {name} disagrees with its calibrated.yaml profile.")
    effective_args = copy(args)
    if getattr(args, "tracker_config", None) is None:
        effective_args.tracker_config = config_path
    config = resolve_tracker_options(effective_args, overrides, include_defaults=True, stamp_timing=True)
    changed = sorted(
        name
        for name, value in fixed.items()
        if config.get(name) != value or (isinstance(config.get(name), bool) != isinstance(value, bool))
    )
    if changed:
        raise ValueError(
            f"Cannot change fixed KF calibration when resuming: {', '.join(changed)}. Start a new tuning run."
        )
    return config, fixed


__all__ = ("CALIBRATED_KF_OPTIONS", "load_tuning_calibration")
