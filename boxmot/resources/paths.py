"""Filesystem resolution for local model resources."""

from __future__ import annotations

from pathlib import Path


def _default_weights_directory() -> Path:
    """Return the repository or installed-package model directory."""

    root = Path(__file__).resolve().parents[2]
    local_root = Path.cwd()
    if (local_root / "pyproject.toml").is_file() and (local_root / "boxmot").is_dir():
        root = local_root
    return root / "models"


def resolve_model_path(
    model_path: str | Path,
    default_dir: str | Path | None = None,
) -> Path:
    """Resolve a model path without downloading or mutating the filesystem.

    Explicit relative and absolute paths are preserved. Bare filenames also
    search the configured weights directory, including a case-insensitive
    sibling lookup for artifacts authored on case-insensitive filesystems.
    """

    path = Path(model_path)
    fallback_directory = _default_weights_directory() if default_dir is None else Path(default_dir)
    candidates = [path]
    if not path.is_absolute() and path.parent == Path("."):
        candidates.append(fallback_directory / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in candidates:
        parent = candidate.parent
        if not parent.exists():
            continue
        lowered_name = candidate.name.lower()
        for sibling in parent.iterdir():
            if sibling.name.lower() == lowered_name:
                return sibling

    return candidates[-1]


__all__ = ("resolve_model_path",)
