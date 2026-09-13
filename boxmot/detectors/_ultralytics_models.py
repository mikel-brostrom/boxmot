"""Metadata for Ultralytics checkpoints that produce object boxes.

Keep imports lightweight. The optional Ultralytics package is loaded only when
its installed asset inventory is requested, never while importing public types.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from importlib.metadata import version

_BOX_MODEL_FAMILY = re.compile(r"^(?:yolo(?:v?\d|e-|_nas_)|rtdetr-|FastSAM-)")


def ultralytics_detector_names(asset_names: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return sorted official box-producing checkpoint stems, preserving case.

    ``asset_names`` permits dependency-free filtering of an explicit inventory.
    By default, use the installed Ultralytics release's bundled asset metadata;
    this performs no network requests or model construction.
    """
    if asset_names is None:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

        asset_names = GITHUB_ASSETS_NAMES
    names = set()
    for name in asset_names:
        if not name.endswith(".pt") or "/" in name or "\\" in name:
            continue
        stem = name[:-3]
        if _BOX_MODEL_FAMILY.match(stem) is None or {"cls", "sem"}.intersection(stem.split("-")):
            continue
        names.add(stem)
    return tuple(sorted(names))


def ultralytics_inventory_version() -> str:
    """Identify the installed release used to generate autocomplete declarations."""
    return version("ultralytics")
