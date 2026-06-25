from __future__ import annotations

from typing import Any, Iterable

WORKFLOW_PANEL_PREFIX = "__panel__:"

_MODEL_SUFFIXES = (".pt", ".onnx", ".engine", ".torchscript")


def first_value(value: Any) -> Any:
    """Return the first list/tuple item, otherwise return *value* unchanged."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def compact_path_name(value: Any) -> str:
    """Return the basename of a path-like value without importing pathlib."""
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def compact_model_name(value: Any) -> str:
    """Return a compact model filename, stripping common exported-model suffixes."""
    name = compact_path_name(first_value(value))
    for suffix in _MODEL_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def image_size_text(value: Any) -> str:
    """Format an image-size tuple as a compact width-by-height label."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{value[0]}×{value[1]}"
    return str(value)


def bool_glyph(value: Any) -> str:
    """Return the Rich UI glyph used for boolean values."""
    return "✓" if bool(value) else "✗"


def panel_field(title: str, items: Iterable[tuple[str, object]]) -> tuple[str, list[tuple[str, object]]]:
    """Build a workflow field that renders as a nested summary panel."""
    return f"{WORKFLOW_PANEL_PREFIX}{title}", list(items)


__all__ = [
    "WORKFLOW_PANEL_PREFIX",
    "bool_glyph",
    "compact_model_name",
    "compact_path_name",
    "first_value",
    "image_size_text",
    "panel_field",
]
