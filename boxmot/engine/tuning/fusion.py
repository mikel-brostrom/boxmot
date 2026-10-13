"""Keep association fusion searches within the selected runtime's capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from boxmot.engine.tuning.conditions import map_entries
from boxmot.engine.tuning.search_space import expand_yaml_groups
from boxmot.trackers.common.native import NATIVE_FUSION_DEFAULTS, validate_native_fusion_options


def validate_fusion_tuning(tracker_name: str, runtime_options: Mapping[str, Any], *, backend: str) -> None:
    """Fail before evaluation or Ray startup for unsupported authored fusion."""
    if backend == "cpp":
        validate_native_fusion_options(tracker_name, runtime_options)


def condition_fusion_schema(schema: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    """Freeze unavailable native policies; ordinary dependencies freeze children.

    Existing native score-fusion switches remain searchable. This only excludes
    newly introduced policies whose implementation currently lives in Python.
    """
    if backend != "cpp":
        return dict(schema)
    return map_entries(
        expand_yaml_groups(dict(schema)),
        lambda name, entry: {"default": NATIVE_FUSION_DEFAULTS[name]} if name in NATIVE_FUSION_DEFAULTS else entry,
    )
