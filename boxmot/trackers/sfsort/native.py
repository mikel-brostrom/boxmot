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
        configured_dimensions = options is not None and ("frame_width" in options or "frame_height" in options)
        cfg = _resolve_tracker_config(options)
        frame_width = cfg["frame_width"]
        frame_height = cfg["frame_height"]
        for name, value in (("frame_width", frame_width), ("frame_height", frame_height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if (frame_width == 0) != (frame_height == 0):
            raise ValueError("frame_width and frame_height must be configured together.")
        if frame_width < 0 or frame_height < 0 or (configured_dimensions and frame_width == 0):
            raise ValueError("frame_width and frame_height must be positive when configured.")
        has_frame_dimensions = frame_width > 0
        self._init_native_handle(
            library=get_sfsort_library() if library is None else library,
            cfg=cfg,
            geometry=geometry,
            use_embeddings=False,
            requires_frame=not has_frame_dimensions,
            frame_dimensions_only=not has_frame_dimensions,
        )


__all__ = ("NativeSFSORTTracker",)
