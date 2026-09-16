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
from boxmot.trackers.sfsort.config import SFSORTConfig


def _resolve_tracker_config(options: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_native_tracker_config("sfsort", options)
    resolve_association_function(cfg)
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
        self.config = SFSORTConfig.from_mapping({name: cfg[name] for name in SFSORTConfig.fields()})
        has_frame_dimensions = self.config.frame_width is not None
        # The ABI uses zero for dimensions inferred from a frame and absent margins.
        for name in ("frame_width", "frame_height", "horizontal_margin", "vertical_margin"):
            if cfg[name] is None:
                cfg[name] = 0
        # Reserved ABI field; SFSORT timeouts are measured directly in frames.
        cfg["frame_rate"] = 30
        self._init_native_handle(
            library=get_sfsort_library() if library is None else library,
            cfg=cfg,
            geometry=geometry,
            use_embeddings=False,
            requires_frame=not has_frame_dimensions,
            frame_dimensions_only=not has_frame_dimensions,
        )


__all__ = ("NativeSFSORTTracker",)
