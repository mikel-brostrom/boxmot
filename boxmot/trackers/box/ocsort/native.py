"""Domain-facing C++ OcSort box-tracker adapter."""

from __future__ import annotations

from typing import Any

from boxmot.native.trackers.ocsort import get_ocsort_library
from boxmot.trackers.common.native import (
    NativeTrackerAdapter,
    NativeTrackerLibrary,
    association_requires_frame,
    load_native_tracker_config,
    resolve_association_function,
)


def _resolve_tracker_config(options: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_native_tracker_config(
        "ocsort",
        options,
        native_only_keys=("iou_threshold", "max_obs"),
    )
    resolve_association_function(cfg)
    cfg.setdefault("iou_threshold", 0.3)
    cfg.setdefault("max_obs", int(cfg["max_age"]) + 5)
    return cfg


class NativeOcSortTracker(NativeTrackerAdapter):
    """Canonical tracker interface backed by the native OcSort ABI."""

    _native_display_name = "OcSort"

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        geometry: str = "aabb",
        library: NativeTrackerLibrary | None = None,
    ) -> None:
        cfg = _resolve_tracker_config(options)
        self._init_native_handle(
            library=get_ocsort_library() if library is None else library,
            cfg=cfg,
            geometry=geometry,
            use_embeddings=False,
            requires_frame=association_requires_frame(cfg),
        )


__all__ = ("NativeOcSortTracker",)
