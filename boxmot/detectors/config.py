"""Detector profile loading and runtime adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from boxmot.configs import CONFIG_ROOT
from boxmot.utils.config import (
    ConfigurationError,
    iter_config_paths,
    load_yaml_mapping,
    resolve_config_path,
    validate_config_id,
)

DETECTOR_CONFIGS_DIR = CONFIG_ROOT / "detectors"


def _required_mapping(payload: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f'{context} must define a "{key}" mapping.')
    return dict(value)


def _required_text(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if value in (None, ""):
        raise ConfigurationError(f'{context} must define "{key}".')
    return str(value)


def _normalize_named_classes(raw_classes: Any, context: str) -> dict[str, int]:
    if not isinstance(raw_classes, Mapping) or not raw_classes:
        raise ConfigurationError(f"{context} must define at least one class.")

    classes: dict[str, int] = {}
    for key, value in raw_classes.items():
        if isinstance(value, bool):
            raise ConfigurationError(f"{context} class ids must be integers.")
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            name, class_id = str(value), int(key)
        else:
            name, class_id = str(key), value
            if isinstance(value, Mapping):
                class_id = value.get("id")
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ConfigurationError(f'{context} class "{name}" must define an integer id.')
        if name in classes:
            raise ConfigurationError(f'{context} defines duplicate class name "{name}".')
        if int(class_id) in classes.values():
            raise ConfigurationError(f"{context} defines duplicate class id {class_id}.")
        classes[name] = int(class_id)
    return classes


def iter_detector_config_paths() -> list[Path]:
    """Return all built-in detector profile paths."""
    return iter_config_paths(DETECTOR_CONFIGS_DIR)


def resolve_detector_config_path(reference: str | Path) -> Path:
    """Resolve a detector profile by id, filename, or explicit path."""
    return resolve_config_path(DETECTOR_CONFIGS_DIR, reference, "detector")


def load_detector_config(reference: str | Path) -> dict[str, Any]:
    """Load and validate one detector profile with named checkpoints."""
    path = resolve_detector_config_path(reference)
    raw = load_yaml_mapping(path)
    context = f'Detector config "{path}"'
    detector_id = validate_config_id(_required_text(raw, "id", context), path=path, label="detector")
    box_type = _required_text(raw, "box_type", context).lower()
    if box_type not in {"aabb", "obb"}:
        raise ConfigurationError(f'{context} box_type must be "aabb" or "obb".')
    classes_by_name = _normalize_named_classes(raw.get("classes"), context)
    inference = _required_mapping(raw, "inference", context)

    image_size = inference.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ConfigurationError(f"{context} image_size must contain exactly two positive integers.")
    confidence = inference.get("confidence_threshold")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ConfigurationError(f"{context} confidence_threshold must be within [0, 1].")

    checkpoints = _required_mapping(raw, "checkpoints", context)
    if not checkpoints:
        raise ConfigurationError(f"{context} must define at least one checkpoint.")
    normalized_checkpoints: dict[str, dict[str, str]] = {}
    for checkpoint_name, checkpoint_value in checkpoints.items():
        if not isinstance(checkpoint_value, Mapping):
            raise ConfigurationError(f'{context} checkpoint "{checkpoint_name}" must be a mapping.')
        checkpoint_context = f'{context} checkpoint "{checkpoint_name}"'
        normalized_checkpoints[str(checkpoint_name)] = {
            "path": _required_text(checkpoint_value, "path", checkpoint_context),
            "uri": str(checkpoint_value.get("uri") or ""),
        }

    return {
        "id": detector_id,
        "box_type": box_type,
        "classes_by_name": classes_by_name,
        "classes": {class_id: name for name, class_id in classes_by_name.items()},
        "image_size": list(image_size),
        "confidence_threshold": float(confidence),
        "checkpoints": normalized_checkpoints,
        "config_path": path,
    }


def resolve_detector_download_url(uri: str | None) -> str:
    """Translate a detector checkpoint URI into a URL accepted by download helpers."""
    value = str(uri or "")
    if value.startswith("gdrive://"):
        file_id = value.removeprefix("gdrive://").strip("/")
        return f"https://drive.google.com/uc?id={file_id}"
    return value


def detector_config_to_runtime(config: Mapping[str, Any], checkpoint_name: str) -> dict[str, Any]:
    """Adapt a validated detector profile/checkpoint to the runtime mapping contract."""
    checkpoints = config["checkpoints"]
    if checkpoint_name not in checkpoints:
        available = ", ".join(sorted(checkpoints))
        raise ConfigurationError(
            f'Detector "{config["id"]}" has no checkpoint "{checkpoint_name}". Available checkpoints: {available}.'
        )
    checkpoint = checkpoints[checkpoint_name]
    download_url = resolve_detector_download_url(checkpoint["uri"])
    return {
        "id": config["id"],
        "checkpoint": checkpoint_name,
        "model": checkpoint["path"],
        "default_model": checkpoint["path"],
        "uri": checkpoint["uri"],
        "url": download_url,
        "model_url": download_url,
        "box_type": config["box_type"],
        "imgsz": list(config["image_size"]),
        "conf": config["confidence_threshold"],
        "classes": dict(config["classes"]),
    }


def load_detector_profile(reference: str | Path) -> dict[str, Any]:
    """Load a detector profile, requiring a checkpoint when the profile has several."""
    checkpoint_name: str | None = None
    try:
        config = load_detector_config(reference)
    except FileNotFoundError as original_error:
        reference_text = str(reference)
        if "/" not in reference_text:
            raise original_error
        detector_ref, checkpoint = reference_text.rsplit("/", 1)
        if not detector_ref or not checkpoint:
            raise original_error
        try:
            config = load_detector_config(detector_ref)
        except FileNotFoundError:
            raise original_error from None
        checkpoint_name = checkpoint

    checkpoints = config["checkpoints"]
    if checkpoint_name is None:
        if len(checkpoints) != 1:
            available = ", ".join(sorted(checkpoints))
            raise ConfigurationError(
                f'Detector "{config["id"]}" has multiple checkpoints. '
                f"Select one as detector/checkpoint. Available checkpoints: {available}."
            )
        checkpoint_name = next(iter(checkpoints))
    return detector_config_to_runtime(config, checkpoint_name)


__all__ = (
    "ConfigurationError",
    "DETECTOR_CONFIGS_DIR",
    "detector_config_to_runtime",
    "iter_detector_config_paths",
    "load_detector_config",
    "load_detector_profile",
    "resolve_detector_download_url",
    "resolve_detector_config_path",
)
