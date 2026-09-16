"""Bind resumed optimizer state to its effective search dimensions and fixed values."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from boxmot.datasets.manifest import canonical_json_bytes


def record_search_profile(
    tune_dir: Path,
    args: Any,
    schema: Mapping[str, Any],
    fixed_options: Mapping[str, Any],
) -> None:
    """Record a new search, or require an identical profile before restoring it.

    The schema includes effective conditional branches after workflow filtering.
    Fixed values are recorded separately so an old optimizer cannot silently
    restore dimensions or values removed from the newly requested search.
    Existing metadata is checked without rewriting it, including on resume.
    """
    profile = canonical_json_bytes(
        {
            "version": 1,
            "tracker": args.tracker,
            "tracker_backend": getattr(args, "tracker_backend", None) or "python",
            "geometry": getattr(args, "geometry", None) or "aabb",
            "search_alg": getattr(args, "search_alg", None) or "optuna",
            "schema": schema,
            "fixed_options": fixed_options,
        }
    )
    path = tune_dir / "search-space.json"
    resume = bool(getattr(args, "resume_tune", None))
    if resume and not path.is_file():
        raise ValueError(
            "Saved tuning run predates search-space metadata or is missing search-space.json; "
            "start a new tuning run instead of restoring an unverified search."
        )
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved_profile = canonical_json_bytes(saved)
        except (OSError, ValueError) as exc:
            raise ValueError("Saved search-space metadata is invalid; start a new tuning run.") from exc
        if saved_profile != profile:
            raise ValueError(
                "Saved tuning search space does not match the requested tracker, backend, geometry, "
                "search algorithm, parameter schema or fixed values; start a new tuning run."
            )
        return
    tune_dir.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(profile + b"\n")
