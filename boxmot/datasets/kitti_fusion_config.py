"""Resolve dataset facts and saved predictions for local KITTI sensor replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from string import Formatter
from typing import Any

from boxmot.datasets.config import DATASET_CONFIGS_DIR, load_dataset_config, validate_sequence_names
from boxmot.utils.config import ConfigurationError, load_yaml_mapping, validate_config_id

_DATASET_KEYS = frozenset(
    {"format", "version", "id", "classes", "default_split", "replay", "sequence_layout", "splits"}
)
_LAYOUT_KEYS = frozenset({"images", "ground_truth", "calibration", "poses"})
_PREDICTION_KEYS = frozenset({"format", "version", "id", "classes", "path", "sequences"})
_ROLES = frozenset({"image", "car", "pedestrian"})


@dataclass(frozen=True, slots=True)
class KittiFusionSequenceInputs:
    """Explicit file and directory locations for one aligned KITTI sequence."""

    sequence_id: str
    images: Path
    ground_truth: Path
    calibration: Path
    poses: Path
    detections_2d: Path
    car_detections_3d: Path
    pedestrian_detections_3d: Path


@dataclass(frozen=True, slots=True)
class KittiFusionDataset:
    """Selected dataset sequences with the prediction manifests used by replay."""

    config_path: Path
    id: str
    split: str
    sequence_names: tuple[str, ...]
    replay_path: Path
    sequences: tuple[KittiFusionSequenceInputs, ...]
    predictions: dict[str, Path]


def _mapping(
    value: Any,
    keys: frozenset[str],
    context: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Require the documented fields without silently accepting misspellings."""

    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} must be a mapping.")
    unknown = set(value).difference(keys | optional)
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ConfigurationError(f"{context} has unknown keys: {names}.")
    missing = keys.difference(value)
    if missing:
        raise ConfigurationError(f"{context} is missing required keys: {', '.join(sorted(missing))}.")
    return value


def _version(payload: dict[str, Any], context: str) -> None:
    """Reject booleans, strings, and unsupported manifest versions."""

    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ConfigurationError(f"{context} version must be integer 1.")


def _identity(value: Any, path: Path) -> None:
    """Validate declared identities without coercing other scalar types."""

    if not isinstance(value, str):
        raise ConfigurationError(f'Manifest "{path}" id must be a lowercase kebab-case string.')
    validate_config_id(value, path=path, label="KITTI manifest")


def _classes(value: Any, required: dict[int, str], context: str, *, allow_cyclist: bool = False) -> None:
    """Keep native class IDs explicit for each selected prediction role."""

    allowed = {**required, **({3: "cyclist"} if allow_cyclist else {})}
    if (
        not isinstance(value, dict)
        or any(type(key) is not int or key not in allowed or allowed[key] != name for key, name in value.items())
        or not set(required).issubset(value)
    ):
        raise ConfigurationError(
            f"{context} classes must map native IDs to {required}"
            + (", optionally including 3: cyclist." if allow_cyclist else ".")
        )


def _relative_path(value: Any, context: str, *, template: bool = False) -> str:
    """Validate portable relative paths and a restricted sequence template syntax."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be a non-empty relative path.")
    try:
        fields = tuple(Formatter().parse(value))
    except ValueError as exc:
        raise ConfigurationError(f"{context} has an invalid path template: {exc}") from exc
    names = set()
    for _literal, name, spec, conversion in fields:
        if name is not None:
            if not template or name not in {"partition", "sequence"} or spec or conversion:
                raise ConfigurationError(f"{context} only allows plain {{partition}} and {{sequence}} placeholders.")
            names.add(name)
    if template and "sequence" not in names:
        raise ConfigurationError(f"{context} must include the {{sequence}} placeholder.")
    expanded = value.format(partition="training", sequence="0000")
    relative = PurePosixPath(expanded)
    windows = PureWindowsPath(expanded)
    if "\\" in expanded or relative.is_absolute() or windows.drive or windows.root or ".." in relative.parts:
        raise ConfigurationError(f"{context} must be a relative path without '..' or a drive prefix.")
    return value


def _sequence_names(value: Any, context: str) -> tuple[str, ...]:
    """Require quoted four-digit names rather than coercing YAML numbers."""

    try:
        names = validate_sequence_names(value)
    except ConfigurationError as exc:
        raise ConfigurationError(f"{context}: {exc}") from exc
    if any(len(name) != 4 or not name.isascii() or not name.isdecimal() for name in names):
        raise ConfigurationError(f"{context} must contain quoted four-digit KITTI sequence names.")
    return names


def _official_splits() -> dict[str, frozenset[str]]:
    """Reuse the KITTI MOTS partition without loading image or tensor readers."""

    splits = load_dataset_config("kitti-mots")["splits"]
    train = frozenset(splits["train"]["sequences"])
    val = frozenset(splits["val"]["sequences"])
    return {"train": train, "val": val, "fulltrain": train | val}


def _validate_dataset(payload: dict[str, Any], path: Path) -> None:
    """Validate dataset facts independently of any prediction selection."""

    context = f'KITTI fusion dataset "{path}"'
    _mapping(payload, _DATASET_KEYS, context)
    _version(payload, context)
    _identity(payload["id"], path)
    _classes(payload["classes"], {1: "car", 2: "pedestrian"}, context)
    _relative_path(payload["replay"], f"{context} replay")
    layout = _mapping(payload["sequence_layout"], _LAYOUT_KEYS, f"{context} sequence_layout")
    for name, value in layout.items():
        _relative_path(value, f"{context} sequence_layout.{name}", template=True)
    splits = payload["splits"]
    official = _official_splits()
    if not isinstance(splits, dict) or not splits:
        raise ConfigurationError(f"{context} splits must be a non-empty mapping.")
    if set(splits).difference(official):
        raise ConfigurationError(f"{context} only supports train, val, and fulltrain splits.")
    if not isinstance(payload["default_split"], str) or payload["default_split"] not in splits:
        raise ConfigurationError(f"{context} default_split must name a declared split.")
    for name, metadata in splits.items():
        split_context = f"{context} splits.{name}"
        _mapping(metadata, frozenset({"partition", "sequences"}), split_context)
        if metadata["partition"] != "training":
            raise ConfigurationError(f"{split_context} partition must be training for annotated KITTI MOTS splits.")
        sequences = _sequence_names(metadata["sequences"], f"{split_context} sequences")
        outside = set(sequences).difference(official[name])
        if outside:
            raise ConfigurationError(
                f"{split_context} sequences are outside the official KITTI MOTS {name} split: "
                f"{', '.join(sorted(outside))}."
            )


def _manifest_for_reference(reference: str | Path) -> tuple[Path, dict[str, Any]] | None:
    """Identify dataset manifests while leaving ordinary image profiles alone."""

    if not str(reference).strip():
        return None
    if (
        isinstance(reference, str)
        and "/" not in reference
        and "\\" not in reference
        and not Path(reference).suffix
        and (DATASET_CONFIGS_DIR / f"{reference}.yaml").is_file()
    ):
        return None
    candidate = Path(reference).expanduser()
    if candidate.is_symlink() and not candidate.exists():
        raise ConfigurationError(f'Dataset reference is a broken link: "{candidate}". Restore its target.')
    directory_reference = candidate.is_dir()
    if directory_reference:
        candidate /= "dataset.yaml"
        if not candidate.is_file():
            raise ConfigurationError(f'Dataset folder requires "{candidate}". Add its KITTI fusion dataset manifest.')
    elif candidate.suffix.lower() not in {".yaml", ".yml"} or not candidate.is_file():
        return None
    payload = load_yaml_mapping(candidate)
    if payload.get("format") != "kitti-fusion":
        if directory_reference:
            raise ConfigurationError(f'Dataset folder manifest "{candidate}" must declare "format: kitti-fusion".')
        return None
    return candidate.resolve(), payload


def resolve_kitti_fusion_config_path(reference: str | Path) -> Path | None:
    """Recognize and validate a dataset YAML or a folder containing dataset.yaml."""

    manifest = _manifest_for_reference(reference)
    if manifest is None:
        return None
    path, payload = manifest
    _validate_dataset(payload, path)
    return path


def _input_path(parent: Path, relative: str, context: str, *, directory: bool = False) -> Path:
    """Resolve an existing input while permitting external directory symlinks."""

    candidate = parent / relative
    exists = candidate.is_dir() if directory else candidate.is_file()
    if exists:
        return candidate.resolve()
    for component in (candidate, *candidate.parents):
        if component.is_symlink() and not component.exists():
            raise ConfigurationError(
                f'{context} contains a broken link: "{component}". Restore its target or update the manifest.'
            )
        if component == parent:
            break
    kind = "directory" if directory else "file"
    raise ConfigurationError(f'{context} requires a {kind} at "{candidate}". Restore the input or update the manifest.')


def _replay_predictions(path: Path, dataset_splits: dict[str, Any], split: str) -> dict[str, Path]:
    """Resolve the selected prediction manifests from a separate replay profile."""

    payload = load_yaml_mapping(path)
    context = f'Replay manifest "{path}"'
    _mapping(payload, frozenset({"version", "splits"}), context)
    _version(payload, context)
    splits = _mapping(payload["splits"], frozenset(dataset_splits), f"{context} splits")
    for name, roles in splits.items():
        _mapping(roles, _ROLES, f"{context} splits.{name}")
        for role, value in roles.items():
            _relative_path(value, f"{context} splits.{name}.{role}")
    return {
        role: _input_path(path.parent, reference, f"{context} {role} prediction manifest")
        for role, reference in splits[split].items()
    }


def _prediction(path: Path, role: str, partition: str, selected: tuple[str, ...]) -> dict[str, Any]:
    """Validate format, class semantics, and sequence coverage before replay."""

    payload = load_yaml_mapping(path)
    context = f'{role.capitalize()} prediction manifest "{path}"'
    _mapping(payload, _PREDICTION_KEYS, context, optional=frozenset({"provenance"}))
    _version(payload, context)
    _identity(payload["id"], path)
    expected_format = "trackrcnn" if role == "image" else "pointgnn"
    if payload["format"] != expected_format:
        raise ConfigurationError(f"{context} format must be {expected_format}.")
    required_classes = {"image": {1: "car", 2: "pedestrian"}, "car": {1: "car"}, "pedestrian": {2: "pedestrian"}}
    _classes(payload["classes"], required_classes[role], context, allow_cyclist=role != "image")
    _relative_path(payload["path"], f"{context} path", template=True)
    coverage = payload["sequences"]
    if not isinstance(coverage, dict) or not coverage or set(coverage).difference({"training", "testing"}):
        raise ConfigurationError(f"{context} sequences must map training or testing partitions to sequence names.")
    for name, sequences in coverage.items():
        _sequence_names(sequences, f"{context} sequences.{name}")
    missing = set(selected).difference(coverage.get(partition, ()))
    if missing:
        raise ConfigurationError(
            f"{context} does not cover {partition} sequence(s): {', '.join(sorted(missing))}. "
            "Select a prediction manifest covering the dataset split."
        )
    if "provenance" in payload and not isinstance(payload["provenance"], dict):
        raise ConfigurationError(f"{context} provenance must be a mapping.")
    return payload


def load_kitti_fusion_dataset(
    reference: str | Path,
    *,
    split: str | None = None,
    sequence_names: tuple[str, ...] = (),
) -> KittiFusionDataset:
    """Load sequence inputs and selected predictions without decoding sensor data."""

    manifest = _manifest_for_reference(reference)
    if manifest is None:
        raise ConfigurationError(
            f'Expected a KITTI fusion dataset folder or YAML with "format: kitti-fusion": "{reference}".'
        )
    path, payload = manifest
    _validate_dataset(payload, path)
    split_name = payload["default_split"] if split is None else split
    if not isinstance(split_name, str) or split_name not in payload["splits"]:
        available = ", ".join(sorted(payload["splits"]))
        raise ConfigurationError(f"KITTI fusion dataset has no split {split_name!r}; available splits: {available}.")
    split_config = payload["splits"][split_name]
    available_sequences = tuple(split_config["sequences"])
    if not isinstance(sequence_names, (list, tuple)):
        raise ConfigurationError("Selected sequences must be a list or tuple of KITTI sequence names.")
    if sequence_names:
        requested = _sequence_names(sequence_names, "Selected sequences")
        missing = set(requested).difference(available_sequences)
        if missing:
            raise ConfigurationError(
                f"Selected sequences are absent from dataset split {split_name!r}: {', '.join(sorted(missing))}. "
                f"Available sequences: {', '.join(available_sequences)}."
            )
        sequence_names = tuple(name for name in available_sequences if name in requested)
    else:
        sequence_names = available_sequences

    replay_path = _input_path(path.parent, payload["replay"], "KITTI fusion replay manifest")
    predictions = _replay_predictions(replay_path, payload["splits"], split_name)
    partition = split_config["partition"]
    prediction_configs = {
        role: _prediction(source, role, partition, sequence_names) for role, source in predictions.items()
    }
    sequences = []
    for name in sequence_names:
        substitutions = {"partition": partition, "sequence": name}
        locations = {
            key: _input_path(
                path.parent,
                template.format(**substitutions),
                f"Sequence {name} {key}",
                directory=key in {"images", "ground_truth"},
            )
            for key, template in payload["sequence_layout"].items()
        }
        detections = {
            role: _input_path(
                predictions[role].parent,
                prediction_configs[role]["path"].format(**substitutions),
                f"Sequence {name} {role} predictions",
                directory=role != "image",
            )
            for role in predictions
        }
        sequences.append(
            KittiFusionSequenceInputs(
                sequence_id=name,
                **locations,
                detections_2d=detections["image"],
                car_detections_3d=detections["car"],
                pedestrian_detections_3d=detections["pedestrian"],
            )
        )
    return KittiFusionDataset(
        path, payload["id"], split_name, tuple(sequence_names), replay_path, tuple(sequences), predictions
    )


__all__ = (
    "KittiFusionDataset",
    "KittiFusionSequenceInputs",
    "load_kitti_fusion_dataset",
    "resolve_kitti_fusion_config_path",
)
