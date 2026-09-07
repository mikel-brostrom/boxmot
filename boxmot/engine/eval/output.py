"""Evaluation output-path allocation."""

from __future__ import annotations

from pathlib import Path


def increment_path(
    path: str | Path,
    *,
    exist_ok: bool = False,
    sep: str = "",
    mkdir: bool = False,
) -> Path:
    """Return the first unused numbered path, optionally creating it."""

    candidate = Path(path)
    if candidate.exists() and not exist_ok:
        base, suffix = (
            (candidate.with_suffix(""), candidate.suffix)
            if candidate.is_file()
            else (candidate, "")
        )
        for number in range(2, 9999):
            numbered = Path(f"{base}{sep}{number}{suffix}")
            if not numbered.exists():
                candidate = numbered
                break

    if mkdir:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


__all__ = ("increment_path",)
