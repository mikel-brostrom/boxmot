"""Domain-facing C++ SFSORT box-tracker adapter."""

from __future__ import annotations

from typing import Any

from boxmot.native.trackers.sfsort import get_sfsort_library
from boxmot.trackers.common.native import (
    NativeTrackerAdapter,
    NativeTrackerLibrary,
    load_native_tracker_config,
    resolve_association_function,
)


def _resolve_tracker_config(options: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_native_tracker_config(
        "sfsort",
        options,
        native_only_keys=("frame_height", "frame_rate", "frame_width", "max_obs"),
    )
    resolve_association_function(cfg)
    cfg.setdefault("frame_width", 0)
    cfg.setdefault("frame_height", 0)
    cfg.setdefault("frame_rate", 30)
    cfg.setdefault("max_obs", 50)
    return cfg


class NativeSFSORTTracker(NativeTrackerAdapter):
    """Canonical tracker interface backed by the native SFSORT ABI."""

    _native_display_name = "SFSORT"

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        geometry: str = "aabb",
        library: NativeTrackerLibrary | None = None,
    ) -> None:
        cfg = _resolve_tracker_config(options)
        self._init_native_handle(
            library=get_sfsort_library() if library is None else library,
            cfg=cfg,
            geometry=geometry,
            use_embeddings=False,
            requires_frame=True,
        )


__all__ = ("NativeSFSORTTracker",)
