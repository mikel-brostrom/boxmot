"""Domain-facing C++ ByteTrack box-tracker adapter."""

from __future__ import annotations

from typing import Any

from boxmot.native.trackers.bytetrack import get_bytetrack_library
from boxmot.trackers.common.native import (
    NativeTrackerAdapter,
    NativeTrackerLibrary,
    association_requires_frame,
    load_native_tracker_config,
    resolve_association_function,
)


def _resolve_tracker_config(options: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_native_tracker_config("bytetrack", options, native_only_keys=("max_obs",))
    resolve_association_function(cfg)
    cfg.setdefault("frame_rate", 30)
    return cfg


class NativeByteTrackTracker(NativeTrackerAdapter):
    """Canonical tracker interface backed by the native ByteTrack ABI."""

    _native_display_name = "ByteTrack"

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        geometry: str = "aabb",
        library: NativeTrackerLibrary | None = None,
    ) -> None:
        cfg = _resolve_tracker_config(options)
        self._init_native_handle(
            library=get_bytetrack_library() if library is None else library,
            cfg=cfg,
            geometry=geometry,
            use_embeddings=False,
            requires_frame=association_requires_frame(cfg),
            frame_dimensions_only=association_requires_frame(cfg),
        )


__all__ = ("NativeByteTrackTracker",)
