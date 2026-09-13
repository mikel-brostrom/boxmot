"""Refine explicitly selected covariance scales around fixed baseline priors."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from boxmot.engine.tuning.search_space import expand_yaml_groups
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS, KALMAN_NOISE_TRACKER_NAMES

KALMAN_REFINEMENT_FIELDS = tuple(name.rsplit(".", 1)[-1] for name in KALMAN_NOISE_OPTIONS)


def validate_kalman_refinement(tracker_name: str, backend: str = "python") -> None:
    """Reject scale search when the selected runtime cannot use its settings."""
    if backend != "python" or tracker_name not in KALMAN_NOISE_TRACKER_NAMES:
        raise ValueError(
            f"--tune-kf is unavailable for {tracker_name!r} with backend {backend!r}; "
            "choose a Python tracker with a supported Kalman filter."
        )


def is_kalman_option(name: str) -> bool:
    """Include grouped class priors alongside the global noise and timing."""
    return name.startswith("kalman_noise.") or name == "variable_dt"


def selected_kalman_options(fields: Sequence[str] | None) -> tuple[str, ...]:
    """Validate CLI or programmatic field selections and return dotted keys."""
    if fields is None:
        return ()
    if isinstance(fields, str) or not isinstance(fields, (tuple, list)):
        raise ValueError("tune_kf must be a sequence of Kalman covariance scale names.")
    unknown = [field for field in fields if field not in KALMAN_REFINEMENT_FIELDS]
    if unknown:
        raise ValueError(f"Unknown --tune-kf field(s): {', '.join(map(str, unknown))}.")
    return tuple(name for name in KALMAN_NOISE_OPTIONS if name.rsplit(".", 1)[-1] in fields)


def refinement_keys(
    baseline: Mapping[str, Any], fields: Sequence[str] | None, *, class_ids: Sequence[int] | None = None
) -> tuple[str, ...]:
    """Refine class scales and a global prior only when it can supply fallback."""
    selected = selected_kalman_options(fields)
    keys = []
    allowed_classes = {str(class_id) for class_id in class_ids} if class_ids else None
    for key in selected:
        field = key.rsplit(".", 1)[-1]
        children = sorted(
            name
            for name in baseline
            if name.startswith("kalman_noise.by_class.")
            and name.endswith(f".{field}")
            and (allowed_classes is None or name.split(".")[2] in allowed_classes)
        )
        keys.extend(children)
        if not class_ids or any(f"kalman_noise.by_class.{class_id}.{field}" not in children for class_id in class_ids):
            keys.append(key)
    return tuple(keys)


def refine_kalman_schema(
    schema: dict,
    baseline: Mapping[str, Any],
    fields: Sequence[str] | None,
    *,
    class_ids: Sequence[int] | None = None,
) -> dict:
    """Freeze KF priors except selected scales searched from 0.25x to 4x."""
    selected = refinement_keys(baseline, fields, class_ids=class_ids)

    def freeze(entries: dict) -> dict:
        result = {}
        for key, entry in entries.items():
            if is_kalman_option(key) and key in baseline:
                result[key] = {"default": baseline[key]}
            elif isinstance(entry, dict) and isinstance(entry.get("activates"), dict):
                result[key] = {**entry, "activates": freeze(entry["activates"])}
            else:
                result[key] = entry
        return result

    result = freeze(expand_yaml_groups(schema))
    for key in selected:
        value = baseline.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"--tune-kf requires a finite positive baseline for {key}.")
        lower, upper = float(value) / 4.0, float(value) * 4.0
        if lower <= 0 or not math.isfinite(upper):
            raise ValueError(f"--tune-kf baseline for {key} cannot define a finite positive search interval.")
        result[key] = {"type": "loguniform", "default": value, "range": [lower, upper]}
    return result


def prepare_kalman_refinement(args: Any, baseline: Mapping[str, Any], tune_dir: Path) -> tuple[str, ...]:
    """Keep the selected dimensions and original prior basis stable on resume."""
    fields = getattr(args, "tune_kf", ())
    selected = refinement_keys(baseline, fields, class_ids=getattr(args, "tracker_class_ids", None))
    path = Path(tune_dir) / "kf-refinement.json"
    metadata = {
        "fields": list(selected),
        "base_options": {
            key: value for key, value in baseline.items() if is_kalman_option(key) or key.startswith("calibration.")
        },
    }
    if getattr(args, "resume_tune", None):
        if not path.exists():
            if selected:
                raise ValueError("Cannot add --tune-kf when resuming a run without KF refinement; start a new run.")
            return selected
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Saved KF refinement metadata is malformed; start a new tuning run.") from exc
        if not isinstance(saved, dict) or saved.get("fields") != metadata["fields"]:
            raise ValueError("Resuming tuning requires the same --tune-kf selection as the saved run.")
        if saved.get("base_options") != metadata["base_options"] or any(
            isinstance(saved["base_options"].get(key), bool) != isinstance(value, bool)
            for key, value in metadata["base_options"].items()
        ):
            raise ValueError("Resuming KF refinement requires the same baseline scales and timing; start a new run.")
    elif selected:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    return selected
