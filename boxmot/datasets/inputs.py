"""Resolve configured sequence inputs without decoding data or importing engines."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from glob import escape
from pathlib import Path
from typing import Any

from boxmot.datasets.config import (
    ConfigurationError,
    dataset_modalities,
    load_dataset_config,
    resolve_dataset_storage_root,
    validate_sequence_names,
)

_SENSOR_ROLES = frozenset({"detections_3d", "calibration", "poses"})
_DIRECTORY_FORMATS = frozenset({"image-directory", "instance-png", "kitti-detections"})
_FILE_FORMATS = frozenset({"trackrcnn", "kitti-p2", "camera-to-world-npy"})


@dataclass(frozen=True, slots=True)
class ModalityInput:
    """An encoding, its ordered existing paths, and reader-specific options."""

    format: str
    paths: tuple[Path, ...]
    options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SequenceInputs:
    """Independent named inputs associated with one authored sequence."""

    sequence_id: str
    modalities: dict[str, ModalityInput]


@dataclass(frozen=True, slots=True)
class DatasetInputs:
    """A selected dataset split with resolved paths and native class metadata."""

    config_path: Path | None
    id: str
    root: Path
    split: str
    sequence_names: tuple[str, ...]
    sequences: tuple[SequenceInputs, ...]
    classes: dict[str, dict[str, Any]]
    fps: float


def resolve_sensor_dataset_config_path(reference: str | Path, *, split: str | None = None) -> Path | None:
    """Recognize declared sensor inputs using only the shared dataset schema."""

    if not str(reference).strip():
        return None
    config = load_dataset_config(reference)
    split_name = config["default_split"] if split is None else split
    if _SENSOR_ROLES.intersection(dataset_modalities(config, split_name)):
        return config["config_path"]
    return None


def _input_path(root: Path, relative: str, context: str, encoding: str) -> Path:
    """Validate filesystem kinds while allowing portable external input symlinks."""

    candidate = root / relative
    if encoding in _DIRECTORY_FORMATS:
        exists, kind = candidate.is_dir(), "directory"
    elif encoding in _FILE_FORMATS:
        exists, kind = candidate.is_file(), "file"
    else:
        exists, kind = candidate.exists(), "file or directory"
    if exists:
        return candidate.resolve()
    for component in (candidate, *candidate.parents):
        if component.is_symlink() and not component.exists():
            raise ConfigurationError(
                f'{context} contains a broken link: "{component}". Restore its target or update the configuration.'
            )
        if component == root:
            break
    raise ConfigurationError(
        f'{context} requires a {kind} at "{candidate}". Restore the input or update the configuration.'
    )


def _discover_sequences(root: Path, modalities: Mapping[str, Any], substitutions: Mapping[str, str]) -> tuple[str, ...]:
    """Find image directories matching an authored template when membership is open."""

    images = modalities.get("images")
    if images is None or len(images["paths"]) != 1:
        raise ConfigurationError("Automatic sequence discovery requires one images path; otherwise declare sequences.")
    marker = "__BOXMOT_SEQUENCE_NAME__"
    template = images["paths"][0].format(**substitutions, sequence=marker)
    parts = template.split(marker)
    if len(parts) < 2:
        raise ConfigurationError("Automatic sequence discovery requires {sequence} in the images path.")
    pattern = "*".join(escape(part) for part in parts)
    expression = re.escape(parts[0]) + "(?P<sequence>[^/]+)" + re.escape(parts[1])
    for part in parts[2:]:
        expression += "(?P=sequence)" + re.escape(part)
    names: set[str] = set()
    for candidate in root.glob(pattern):
        match = re.fullmatch(expression, candidate.relative_to(root).as_posix())
        if candidate.is_dir() and match is not None and not match.group("sequence").startswith("."):
            names.add(match.group("sequence"))
    if not names:
        raise ConfigurationError(f'No sequence image directories match "{root / template.replace(marker, "*")}".')
    return validate_sequence_names(tuple(sorted(names)))


def resolve_dataset_inputs(
    config: Mapping[str, Any],
    *,
    split: str | None = None,
    sequence_names: Sequence[str] = (),
    data_root: str | Path | None = None,
    roles: Sequence[str] | None = None,
) -> DatasetInputs:
    """Resolve a normalized sequence profile, optionally selecting consumed roles."""

    if config.get("layout") != "sequence":
        raise ConfigurationError("Configured input resolution requires format.layout: sequence.")
    split_name = config["default_split"] if split is None else split
    modalities = dataset_modalities(config, split_name)
    split_config = config["splits"][split_name]
    root = resolve_dataset_storage_root(config, data_root)
    substitutions = {"partition": split_config["partition"], "split": split_name}
    available = (
        validate_sequence_names(split_config["sequences"])
        if "sequences" in split_config
        else _discover_sequences(root, modalities, substitutions)
    )
    if not isinstance(sequence_names, (list, tuple)):
        raise ConfigurationError("Selected sequences must be a list or tuple of sequence directory names.")
    if sequence_names:
        requested = validate_sequence_names(sequence_names)
        missing = set(requested).difference(available)
        if missing:
            raise ConfigurationError(
                f"Selected sequences are absent from dataset split {split_name!r}: {', '.join(sorted(missing))}. "
                f"Available sequences: {', '.join(available)}."
            )
        selected = tuple(name for name in available if name in requested)
    else:
        selected = available
    if roles is not None:
        modalities = {role: specification for role, specification in modalities.items() if role in roles}
    for role, specification in modalities.items():
        if role != "detections_3d" and len(specification["paths"]) != 1:
            raise ConfigurationError(f"Dataset modality {role} requires exactly one input path.")
    sequences = tuple(
        SequenceInputs(
            sequence_id=name,
            modalities={
                role: ModalityInput(
                    format=specification["format"],
                    paths=tuple(
                        _input_path(
                            root,
                            template.format(**substitutions, sequence=name),
                            f"Sequence {name} {role}",
                            specification["format"],
                        )
                        for template in specification["paths"]
                    ),
                    options=deepcopy(specification["options"]),
                )
                for role, specification in modalities.items()
            },
        )
        for name in selected
    )
    config_path = config.get("config_path")
    return DatasetInputs(
        config_path=Path(config_path).resolve() if config_path is not None else None,
        id=str(config["id"]),
        root=root,
        split=split_name,
        sequence_names=selected,
        sequences=sequences,
        classes=deepcopy(config["classes"]),
        fps=float(config["fps"]),
    )


def load_dataset_inputs(
    reference: str | Path,
    *,
    split: str | None = None,
    sequence_names: Sequence[str] = (),
    data_root: str | Path | None = None,
    roles: Sequence[str] | None = None,
) -> DatasetInputs:
    """Load one shared dataset profile and resolve its selected modality inputs."""

    return resolve_dataset_inputs(
        load_dataset_config(reference), split=split, sequence_names=sequence_names, data_root=data_root, roles=roles
    )
