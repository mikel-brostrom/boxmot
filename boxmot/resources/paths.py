"""Filesystem resolution for local model resources."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_path


def repository_root() -> Path | None:
    """Find a source checkout, preferring one explicitly selected by the cwd."""

    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (root / "pyproject.toml").is_file() and (root / "boxmot").is_dir():
            return root
    return None


def default_weights_directory() -> Path:
    """Use checkout-local models or a writable per-user installed-model cache.

    Resolution does not create directories. Downloaders create the destination
    only when an artifact needs to be downloaded.
    """

    root = repository_root()
    if root is not None:
        return root / "models"
    return user_cache_path("boxmot") / "models"


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
    fallback_directory = default_weights_directory() if default_dir is None else Path(default_dir)
    candidates = [path]
    if not path.is_absolute() and path.parent == Path("."):
        candidates.append(fallback_directory / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in candidates:
        parent = candidate.parent
        if not parent.is_dir():
            continue
        lowered_name = candidate.name.lower()
        for sibling in parent.iterdir():
            if sibling.name.lower() == lowered_name:
                return sibling

    return candidates[-1]


__all__ = ("default_weights_directory", "repository_root", "resolve_model_path")
