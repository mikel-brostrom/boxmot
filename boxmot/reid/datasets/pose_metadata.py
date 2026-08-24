"""Shared loading for cached pose and segmentation metadata."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=8)
def _load_metadata_images(manifest_path: str) -> Any:
    """Load one manifest once per process.

    Callers treat the returned image mapping as immutable. Sharing the object
    avoids parsing and retaining duplicate copies when PAV and anatomical
    supervision consume the same metadata.
    """
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return payload.get("images")


def load_metadata_images(manifest_path: str | Path) -> Any:
    """Return the cached ``images`` payload for a resolved manifest path."""
    return _load_metadata_images(str(Path(manifest_path).expanduser().resolve()))
