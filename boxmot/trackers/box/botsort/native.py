"""Domain-facing C++ BotSort box-tracker adapter."""

from __future__ import annotations

from typing import Any

from boxmot.native.trackers.botsort import get_botsort_library
from boxmot.trackers.common.native import (
    NativeTrackerAdapter,
    NativeTrackerLibrary,
    association_requires_frame,
    load_native_tracker_config,
    resolve_association_function,
)


def _resolve_tracker_config(options: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_native_tracker_config("botsort", options, native_only_keys=("max_obs",))
    resolve_association_function(cfg)
    cfg.setdefault("frame_rate", 30)
    cfg.setdefault("fuse_first_associate", False)
    cfg.setdefault("use_cmc", True)
    cfg.setdefault("use_embeddings", True)
    return cfg


class NativeBotSortTracker(NativeTrackerAdapter):
    """Canonical tracker interface backed by the native BotSort ABI."""

    _native_display_name = "BotSort"

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        geometry: str = "aabb",
        library: NativeTrackerLibrary | None = None,
    ) -> None:
        cfg = _resolve_tracker_config(options)
        self._init_native_handle(
            library=get_botsort_library() if library is None else library,
            cfg=cfg,
            geometry=geometry,
            use_embeddings=bool(cfg["use_embeddings"]),
            requires_frame=bool(cfg.get("use_cmc", True)) or association_requires_frame(cfg),
        )


__all__ = ("NativeBotSortTracker",)
