"""Dataset configuration resolution."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from string import Formatter
from typing import Any, Mapping

from boxmot.configs import CONFIG_ROOT
from boxmot.utils.config import (
    ConfigurationError,
    index_config_ids,
    iter_config_paths,
    load_yaml_mapping,
    resolve_config_path,
    validate_config_id,
)

DATASET_CONFIGS_DIR = CONFIG_ROOT / "datasets"
_MODALITY_FORMATS = {
    "images": "image-directory",
    "ground_truth": "instance-png",
    "detections_2d": "trackrcnn",
    "detections_3d": "kitti-detections",
    "calibration": "kitti-p2",
    "poses": "camera-to-world-npy",
}
MODALITY_ROLES = frozenset(_MODALITY_FORMATS)


def _required_mapping(payload: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f'{context} must define a "{key}" mapping.')
    return dict(value)


def _required_text(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f'{context} must define "{key}" as a non-empty string.')
    return value


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
    """Resolve catalog IDs before same-named folders; explicit paths stay local."""

    text = str(reference)
    candidate = Path(reference).expanduser()
    bare_id = isinstance(reference, str) and "/" not in text and "\\" not in text and not candidate.suffix
    if bare_id:
        for extension in (".yaml", ".yml"):
            exact = DATASET_CONFIGS_DIR / f"{text}{extension}"
            if exact.is_file():
                return exact.resolve()
        catalog = index_config_ids(DATASET_CONFIGS_DIR, "dataset")
        if text in catalog:
            return catalog[text]
    if candidate.is_symlink() and not candidate.exists():
        raise ConfigurationError(f'Dataset reference is a broken link: "{candidate}". Restore its target.')
    if candidate.is_dir():
        manifest = candidate / "dataset.yaml"
        if not manifest.is_file():
            raise ConfigurationError(f'Dataset folder requires "{manifest}". Add its dataset configuration.')
        return manifest.resolve()
    explicit = isinstance(reference, Path) or "/" in text or "\\" in text or candidate.is_absolute()
    if explicit:
        if candidate.suffix.lower() in {".yaml", ".yml"} and candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f'Dataset config path does not exist: "{candidate}"')
    return resolve_config_path(DATASET_CONFIGS_DIR, candidate, "dataset")


def resolve_dataset_storage_root(config: Mapping[str, Any], data_root: str | Path | None = None) -> Path:
    """Resolve storage beneath an override, a local YAML, or the catalog data root."""

    relative = _safe_relative_path(config, "root", "Dataset storage")
    if data_root is not None:
        base = Path(data_root).expanduser().resolve()
    else:
        config_path = config.get("config_path")
        path = Path(config_path).expanduser().resolve() if config_path is not None else None
        builtin = path is None or path.is_relative_to(DATASET_CONFIGS_DIR.resolve())
        base = (Path("datasets") / "mot").resolve() if builtin else path.parent
    resolved = base.joinpath(*PurePosixPath(relative).parts).resolve()
    if not resolved.is_relative_to(base):
        raise ConfigurationError("Dataset storage.root must remain beneath the configured data root.")
    return resolved


def _modality_path(value: Any, context: str) -> str:
    """Validate portable input paths with plain split and sequence placeholders."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be a non-empty relative path.")
    try:
        fields = tuple(Formatter().parse(value))
    except ValueError as exc:
        raise ConfigurationError(f"{context} has an invalid path template: {exc}") from exc
    for _literal, name, spec, conversion in fields:
        if name is not None and (name not in {"partition", "sequence", "split"} or spec or conversion):
            raise ConfigurationError(
                f"{context} only allows plain {{partition}}, {{sequence}}, and {{split}} placeholders."
            )
    expanded = value.format(partition="partition", sequence="sequence", split="split")
    _safe_relative_path({"path": expanded}, "path", context)
    return value


def _modalities(value: Any, context: str, *, overrides: bool = False) -> dict[str, Any]:
    """Normalize each independently encoded input without selecting a reader."""

    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} must be a mapping.")
    unknown = set(value).difference(MODALITY_ROLES)
    if unknown:
        raise ConfigurationError(f"{context} has unknown modalities: {', '.join(sorted(map(str, unknown)))}.")
    normalized: dict[str, Any] = {}
    for role, specification in value.items():
        item_context = f"{context}.{role}"
        if specification is None and overrides:
            normalized[role] = None
            continue
        if not isinstance(specification, dict):
            raise ConfigurationError(f"{item_context} must be a mapping.")
        unknown_keys = set(specification).difference({"format", "path", "paths", "options"})
        if unknown_keys:
            raise ConfigurationError(f"{item_context} has unknown keys: {', '.join(sorted(map(str, unknown_keys)))}.")
        encoding = specification.get("format")
        if not isinstance(encoding, str) or not encoding.strip():
            raise ConfigurationError(f"{item_context} must define a non-empty format.")
        if encoding != _MODALITY_FORMATS[role]:
            raise ConfigurationError(f"{item_context} format must be {_MODALITY_FORMATS[role]}.")
        if ("path" in specification) == ("paths" in specification):
            raise ConfigurationError(f"{item_context} must define exactly one of path or paths.")
        paths = [specification["path"]] if "path" in specification else specification["paths"]
        if not isinstance(paths, list) or not paths:
            raise ConfigurationError(f"{item_context}.paths must be a non-empty list.")
        if role != "detections_3d" and len(paths) != 1:
            raise ConfigurationError(f"{item_context} requires exactly one input path.")
        options = specification.get("options", {})
        if not isinstance(options, dict):
            raise ConfigurationError(f"{item_context}.options must be a mapping.")
        _validate_modality_options(role, options, item_context)
        normalized[role] = {
            "format": encoding,
            "paths": [_modality_path(path, f"{item_context}.paths") for path in paths],
            "options": deepcopy(options),
        }
    return normalized


def _validate_modality_options(role: str, options: Mapping[str, Any], context: str) -> None:
    """Reject unsupported reader settings before locating or decoding inputs."""

    allowed = {
        "ground_truth": {"class_divisor", "background_id", "ignore_ids"},
        "detections_3d": {
            "score_transform",
            "class_map",
            "ignore_classes",
            "coordinate_frame",
            "box_origin",
            "dimensions",
            "yaw_axis",
        },
    }.get(role, set())
    unknown = set(options).difference(allowed)
    if unknown:
        raise ConfigurationError(f"{context} has unsupported options: {', '.join(sorted(map(str, unknown)))}.")
    if role == "ground_truth":
        for name, minimum in (("class_divisor", 1), ("background_id", 0)):
            value = options.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ConfigurationError(f"{context}.options.{name} must be an integer >= {minimum}.")
        ignored = options.get("ignore_ids", [])
        if not isinstance(ignored, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ignored
        ):
            raise ConfigurationError(f"{context}.options.ignore_ids must be a list of non-negative integer labels.")
    elif role == "detections_3d":
        for name, allowed_values in {
            "score_transform": ("identity", "odds"),
            "coordinate_frame": ("camera",),
            "box_origin": ("bottom-center",),
            "dimensions": ("hwl",),
            "yaw_axis": ("y",),
        }.items():
            if name in options and options[name] not in allowed_values:
                raise ConfigurationError(f"{context}.options.{name} must be one of {allowed_values}.")
        ignored = options.get("ignore_classes", [])
        if not isinstance(ignored, list) or any(not isinstance(value, str) or not value for value in ignored):
            raise ConfigurationError(f"{context}.options.ignore_classes must be a list of class labels.")
        class_map = options.get("class_map", {})
        if not isinstance(class_map, dict) or any(
            not isinstance(source, str)
            or not source
            or isinstance(target, bool)
            or not isinstance(target, (str, int))
            or target == ""
            for source, target in class_map.items()
        ):
            raise ConfigurationError(
                f"{context}.options.class_map must map source labels to class names or integer IDs."
            )


def dataset_modalities(config: Mapping[str, Any], split: str) -> dict[str, dict[str, Any]]:
    """Return selected modality declarations after complete per-role replacement."""

    splits = config.get("splits", {})
    if split not in splits:
        raise ConfigurationError(f"Dataset has no split {split!r}; available splits: {', '.join(sorted(splits))}.")
    merged = {**config.get("modalities", {}), **splits[split].get("modalities", {})}
    return {role: deepcopy(specification) for role, specification in merged.items() if specification is not None}


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
    unknown = set(raw).difference(
        {"id", "format", "storage", "classes", "splits", "default_split", "resources", "fps", "modalities"}
    )
    if unknown:
        raise ConfigurationError(f"{context} has unknown keys: {', '.join(sorted(map(str, unknown)))}.")
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
    for name, mapping, allowed in (
        ("format", format_config, {"layout", "box_type"}),
        ("storage", storage_config, {"root"}),
    ):
        unknown_keys = set(mapping).difference(allowed)
        if unknown_keys:
            raise ConfigurationError(f"{context} {name} has unknown keys: {', '.join(sorted(map(str, unknown_keys)))}.")

    layout = _required_text(format_config, "layout", context)
    if layout not in {"mot", "visdrone", "sequence"}:
        raise ConfigurationError(f'{context} layout must be "mot", "visdrone", or "sequence", got "{layout}".')
    box_type = _required_text(format_config, "box_type", context).lower()
    if box_type not in {"aabb", "obb"}:
        raise ConfigurationError(f'{context} box_type must be "aabb" or "obb", got "{box_type}".')
    root = _safe_relative_path(storage_config, "root", f"{context} storage")
    fps = raw.get("fps")
    if "fps" in raw and (
        isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
    ):
        raise ConfigurationError(f"{context} fps must be a finite positive number.")
    if layout == "sequence" and fps is None:
        raise ConfigurationError(f"{context} sequence layout must define fps.")
    modalities = _modalities(raw.get("modalities", {}), f"{context} modalities")
    if layout == "sequence" and not modalities:
        raise ConfigurationError(f"{context} sequence layout must define modalities.")
    try:
        validate_sequence_names(tuple(split_configs))
    except ConfigurationError as exc:
        raise ConfigurationError(f"{context} split names: {exc}") from exc

    splits: dict[str, dict[str, Any]] = {}
    for split_name, split_value in split_configs.items():
        if not isinstance(split_value, dict):
            raise ConfigurationError(f'{context} split "{split_name}" must define path and has_ground_truth.')
        split_context = f'{context} split "{split_name}"'
        normalized_split = deepcopy(split_value)
        overrides = _modalities(split_value.get("modalities", {}), f"{split_context} modalities", overrides=True)
        if "modalities" in split_value:
            normalized_split["modalities"] = overrides
        if layout == "sequence":
            unknown_keys = set(split_value).difference({"partition", "sequences", "has_ground_truth", "modalities"})
            if unknown_keys:
                raise ConfigurationError(
                    f"{split_context} has unknown keys: {', '.join(sorted(map(str, unknown_keys)))}."
                )
            partition = split_value.get("partition")
            try:
                validate_sequence_names((partition,))
            except ConfigurationError as exc:
                raise ConfigurationError(f"{split_context} partition: {exc}") from exc
            effective = {**modalities, **overrides}
            if box_type != "aabb" and any(
                effective.get(role) is not None for role in ("ground_truth", "detections_2d")
            ):
                raise ConfigurationError(f'{split_context} instance-png and trackrcnn inputs require box_type "aabb".')
            has_ground_truth = effective.get("ground_truth") is not None
            if "has_ground_truth" in split_value and (
                not isinstance(split_value["has_ground_truth"], bool)
                or split_value["has_ground_truth"] != has_ground_truth
            ):
                raise ConfigurationError(f"{split_context} has_ground_truth must agree with its ground_truth modality.")
        else:
            normalized_split["path"] = _safe_relative_path(split_value, "path", split_context)
            has_ground_truth = split_value.get("has_ground_truth")
            if not isinstance(has_ground_truth, bool):
                raise ConfigurationError(f"{split_context} must define boolean has_ground_truth.")
        normalized_split["has_ground_truth"] = has_ground_truth
        if split_value.get("annotations") is not None:
            normalized_split["annotations"] = _safe_relative_path(
                split_value,
                "annotations",
                split_context,
            )
        if "sequences" in split_value:
            normalized_split["sequences"] = list(validate_sequence_names(split_value["sequences"]))
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
        "fps": float(fps) if fps is not None else None,
        "modalities": modalities,
        "default_split": default_split,
        "splits": splits,
        "classes": classes,
        "resources": resources,
        "config_path": path,
    }


__all__ = (
    "ConfigurationError",
    "DATASET_CONFIGS_DIR",
    "MODALITY_ROLES",
    "dataset_modalities",
    "iter_dataset_config_paths",
    "load_dataset_config",
    "resolve_dataset_config_path",
    "resolve_dataset_storage_root",
    "validate_sequence_names",
)
