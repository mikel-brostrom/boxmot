"""Domain-facing C++ OccluBoost box-tracker adapter."""

from __future__ import annotations

from typing import Any

from boxmot.native.trackers.occluboost import get_occluboost_library
from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.common.native import (
    NativeTrackerAdapter,
    NativeTrackerLibrary,
    association_requires_frame,
    load_native_tracker_config,
    resolve_association_function,
)
from boxmot.trackers.occluboost.config import OccluBoostConfig


def _resolve_tracker_config(options: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_native_tracker_config("occluboost", options)
    resolve_association_function(cfg)
    if cfg["use_cmc"] and cfg["cmc_method"] not in {"ecc", "sof"}:
        raise ValueError("Native OccluBoost supports cmc_method 'ecc' or 'sof'; disable CMC with use_cmc=False.")
    return cfg


class NativeOccluBoostTracker(NativeTrackerAdapter):
    """Canonical tracker interface backed by the native OccluBoost ABI."""

    _native_display_name = "OccluBoost"
    accepts_embeddings = True

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        geometry: str = "aabb",
        library: NativeTrackerLibrary | None = None,
        reid: ReIDConfig | AppearanceEncoder | None = None,
    ) -> None:
        cfg = _resolve_tracker_config(options)
        self.config = OccluBoostConfig.from_mapping({name: cfg[name] for name in OccluBoostConfig.fields()})
        self._init_native_handle(
            library=get_occluboost_library() if library is None else library,
            cfg=cfg,
            geometry=geometry,
            use_embeddings=bool(cfg["use_embeddings"]),
            requires_frame=bool(cfg["use_cmc"]) or association_requires_frame(cfg),
            frame_dimensions_only=not cfg["use_cmc"] and association_requires_frame(cfg),
            reid=reid,
        )


__all__ = ("NativeOccluBoostTracker",)
