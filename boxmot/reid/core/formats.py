"""Canonical ReID runtime and export format metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReIDFormat:
    """Describe one supported ReID artifact format."""

    id: str
    name: str
    argument: str
    suffix: str
    cpu: bool
    gpu: bool
    alternate_suffixes: tuple[str, ...] = ()

    @property
    def suffixes(self) -> tuple[str, ...]:
        """Return every accepted artifact suffix for this format."""
        return (self.suffix, *self.alternate_suffixes)

    def matches(self, path: str | Path) -> bool:
        """Return whether an artifact path belongs to this format."""
        artifact = Path(path)
        name = artifact.name.lower()
        file_suffix = artifact.suffix.lower()
        return any(
            file_suffix == suffix if suffix.startswith(".") else name.endswith(suffix)
            for suffix in self.suffixes
        )


REID_FORMATS = (
    ReIDFormat("pytorch", "PyTorch", "-", ".pt", True, True),
    ReIDFormat("torchscript", "TorchScript", "torchscript", ".torchscript", True, True),
    ReIDFormat("onnx", "ONNX", "onnx", ".onnx", True, True),
    ReIDFormat(
        "openvino",
        "OpenVINO",
        "openvino",
        "_openvino_model",
        True,
        False,
        alternate_suffixes=(".xml", ".bin"),
    ),
    ReIDFormat("tensorrt", "TensorRT", "engine", ".engine", False, True),
    ReIDFormat("coreml", "CoreML MLProgram", "coreml", "_coreml_model", True, True),
    ReIDFormat("tflite", "TensorFlow Lite", "tflite", ".tflite", True, False),
)
REID_FORMATS_BY_ID = {format_.id: format_ for format_ in REID_FORMATS}
REID_FORMATS_BY_ARGUMENT = {
    format_.argument: format_ for format_ in REID_FORMATS if format_.argument != "-"
}

REID_EXPORT_FORMAT_COLUMNS = ("Format", "Argument", "Suffix", "CPU", "GPU")
REID_EXPORT_FORMAT_ROWS = tuple(
    (format_.name, format_.argument, format_.suffix, format_.cpu, format_.gpu)
    for format_ in REID_FORMATS
)
REID_EXPORT_ARGUMENTS = tuple(format_.argument for format_ in REID_FORMATS)
REID_EXPORT_SUFFIXES = tuple(format_.suffix for format_ in REID_FORMATS)


def resolve_reid_format(path: str | Path) -> ReIDFormat:
    """Resolve the unique supported format for an artifact path."""
    matches = tuple(format_ for format_ in REID_FORMATS if format_.matches(path))
    if len(matches) != 1:
        accepted = tuple(suffix for format_ in REID_FORMATS for suffix in format_.suffixes)
        raise ValueError(f"Unsupported ReID artifact format for {path!s}; expected one of {accepted}")
    return matches[0]


def resolve_export_formats(arguments: object) -> tuple[ReIDFormat, ...]:
    """Resolve unique CLI export arguments into canonical formats."""
    requested = tuple(str(argument).lower() for argument in (arguments or ()))
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate ReID export formats are not allowed: {requested}")
    try:
        return tuple(REID_FORMATS_BY_ARGUMENT[argument] for argument in requested)
    except KeyError as exc:
        available = tuple(REID_FORMATS_BY_ARGUMENT)
        raise ValueError(f"Invalid ReID export format {exc.args[0]!r}; expected one of {available}") from exc


__all__ = (
    "REID_EXPORT_ARGUMENTS",
    "REID_EXPORT_FORMAT_COLUMNS",
    "REID_EXPORT_FORMAT_ROWS",
    "REID_EXPORT_SUFFIXES",
    "REID_FORMATS",
    "REID_FORMATS_BY_ARGUMENT",
    "REID_FORMATS_BY_ID",
    "ReIDFormat",
    "resolve_export_formats",
    "resolve_reid_format",
)
