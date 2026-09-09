"""Dataset configuration resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from boxmot.configs import CONFIG_ROOT
from boxmot.utils.config import (
    ConfigurationError,
    iter_config_paths,
    load_yaml_mapping,
    resolve_config_path,
    validate_config_id,
)

DATASET_CONFIGS_DIR = CONFIG_ROOT / "datasets"


def _required_mapping(payload: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f'{context} must define a "{key}" mapping.')
    return dict(value)


def _required_text(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if value in (None, ""):
        raise ConfigurationError(f'{context} must define "{key}".')
    return str(value)


def _safe_relative_path(payload: Mapping[str, Any], key: str, context: str) -> str:
    """Return one validated POSIX path relative to the dataset root."""

    value = _required_text(payload, key, context)
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in path.parts
    ):
        raise ConfigurationError(f'{context} "{key}" must remain beneath storage.root.')
    return path.as_posix()


def iter_dataset_config_paths() -> list[Path]:
    """Return all built-in dataset profile paths."""
    return iter_config_paths(DATASET_CONFIGS_DIR)


def resolve_dataset_config_path(reference: str | Path) -> Path:
    """Resolve a dataset profile by id, filename, or explicit path."""
    return resolve_config_path(DATASET_CONFIGS_DIR, reference, "dataset")


def validate_sequence_names(value: Any) -> tuple[str, ...]:
    """Validate an explicit, non-empty selection of sequence directory names."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigurationError("Split sequences must be a non-empty list of directory names.")
    for name in value:
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or ":" in name
        ):
            raise ConfigurationError("Split sequences must contain canonical directory names without path components.")
    if len(set(value)) != len(value):
        raise ConfigurationError("Split sequences must not contain duplicate directory names.")
    return tuple(value)


def load_dataset_config(reference: str | Path) -> dict[str, Any]:
    """Load and validate one model-free dataset profile."""
    path = resolve_dataset_config_path(reference)
    raw = load_yaml_mapping(path)
    context = f'Dataset config "{path}"'
    dataset_id = validate_config_id(_required_text(raw, "id", context), path=path, label="dataset")

    dataset_resources = raw.get("resources") or {}
    if not isinstance(dataset_resources, dict):
        raise ConfigurationError(f'{context} "resources" must be a mapping.')
    unsupported_resources = sorted(set(dataset_resources) - {"dataset"})
    if unsupported_resources:
        raise ConfigurationError(
            f'{context} may only define its own "dataset" resource; unsupported: {", ".join(unsupported_resources)}.'
        )
    dataset_resource = dataset_resources.get("dataset") or {}
    if not isinstance(dataset_resource, dict):
        raise ConfigurationError(f'{context} "resources.dataset" must be a mapping.')

    format_config = _required_mapping(raw, "format", context)
    storage_config = _required_mapping(raw, "storage", context)
    split_configs = _required_mapping(raw, "splits", context)
    class_groups = _required_mapping(raw, "classes", context)
    if not split_configs:
        raise ConfigurationError(f"{context} must define at least one split.")

    layout = _required_text(format_config, "layout", context)
    box_type = _required_text(format_config, "box_type", context).lower()
    if box_type not in {"aabb", "obb"}:
        raise ConfigurationError(f'{context} box_type must be "aabb" or "obb", got "{box_type}".')
    if layout == "kitti-mots" and box_type != "aabb":
        raise ConfigurationError(f'{context} KITTI MOTS masks require box_type "aabb".')
    root = _safe_relative_path(storage_config, "root", f"{context} storage")

    splits: dict[str, dict[str, Any]] = {}
    for split_name, split_value in split_configs.items():
        if not isinstance(split_value, dict):
            raise ConfigurationError(f'{context} split "{split_name}" must define path and has_ground_truth.')
        split_context = f'{context} split "{split_name}"'
        split_path = _safe_relative_path(split_value, "path", split_context)
        has_ground_truth = split_value.get("has_ground_truth")
        if not isinstance(has_ground_truth, bool):
            raise ConfigurationError(f"{split_context} must define boolean has_ground_truth.")
        normalized_split = {
            **deepcopy(split_value),
            "path": split_path,
            "has_ground_truth": has_ground_truth,
        }
        if split_value.get("annotations") is not None:
            normalized_split["annotations"] = _safe_relative_path(
                split_value,
                "annotations",
                split_context,
            )
        if "sequences" in split_value:
            normalized_split["sequences"] = list(validate_sequence_names(split_value["sequences"]))
        if layout == "kitti-mots" and has_ground_truth != (split_value.get("annotations") is not None):
            raise ConfigurationError(f"{split_context} must declare annotations exactly when has_ground_truth is true.")
        splits[str(split_name)] = normalized_split

    default_split = str(raw.get("default_split") or next(iter(splits)))
    if default_split not in splits:
        raise ConfigurationError(f'{context} default_split "{default_split}" is not present in splits.')

    unknown_groups = sorted(set(class_groups) - {"target", "ignore"})
    if unknown_groups:
        raise ConfigurationError(
            f'{context} classes only supports "target" and "ignore" groups; unknown: {", ".join(unknown_groups)}.'
        )
    target_classes = _required_mapping(class_groups, "target", f"{context} classes")
    if not target_classes:
        raise ConfigurationError(f"{context} classes.target must define at least one class.")
    ignore_classes = class_groups.get("ignore") or {}
    if not isinstance(ignore_classes, dict):
        raise ConfigurationError(f"{context} classes.ignore must be a mapping.")

    classes: dict[str, dict[str, Any]] = {}
    class_ids: set[int] = set()
    for evaluation, grouped_classes in (("target", target_classes), ("ignore", ignore_classes)):
        for class_name, class_id in grouped_classes.items():
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ConfigurationError(f'{context} class "{class_name}" must have an integer id.')
            if str(class_name) in classes:
                raise ConfigurationError(f'{context} defines duplicate class name "{class_name}".')
            if class_id in class_ids:
                raise ConfigurationError(f"{context} defines duplicate class id {class_id}.")
            class_ids.add(class_id)
            classes[str(class_name)] = {"id": class_id, "evaluation": evaluation}

    resources = {"dataset": deepcopy(dataset_resource)} if dataset_resource else {}
    return {
        "id": dataset_id,
        "layout": layout,
        "box_type": box_type,
        "root": root,
        "default_split": default_split,
        "splits": splits,
        "classes": classes,
        "resources": resources,
        "config_path": path,
    }


__all__ = (
    "ConfigurationError",
    "DATASET_CONFIGS_DIR",
    "iter_dataset_config_paths",
    "load_dataset_config",
    "resolve_dataset_config_path",
    "validate_sequence_names",
)
