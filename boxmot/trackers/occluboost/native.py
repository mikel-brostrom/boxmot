"""Domain-facing C++ OccluBoost box-tracker adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from boxmot.native.trackers.occluboost import get_occluboost_library
from boxmot.trackers.common.native import (
    NativeTrackerAdapter,
    NativeTrackerLibrary,
    association_requires_frame,
    load_native_tracker_config,
    resolve_association_function,
)


def _resolve_tracker_config(options: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_native_tracker_config("occluboost", options, native_only_keys=("max_obs",))
    resolve_association_function(cfg)
    cfg.setdefault("use_embeddings", True)
    cfg.setdefault("use_cmc", True)
    cfg.setdefault("cmc_method", "sof")
    cfg.setdefault("max_obs", 50)
    for option in ("use_cmc", "use_embeddings"):
        if not isinstance(cfg[option], bool):
            raise TypeError(f"{option} must be bool.")
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
        reid_model: Any | None = None,
        reid_weights: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        device: Any = "cpu",
        half: bool = False,
        reid_preprocess: str | None = None,
    ) -> None:
        cfg = _resolve_tracker_config(options)
        self._init_native_handle(
            library=get_occluboost_library() if library is None else library,
            cfg=cfg,
            geometry=geometry,
            use_embeddings=bool(cfg["use_embeddings"]),
            requires_frame=bool(cfg["use_cmc"]) or association_requires_frame(cfg),
            frame_dimensions_only=not cfg["use_cmc"] and association_requires_frame(cfg),
            reid_model=reid_model,
            reid_weights=reid_weights,
            device=device,
            half=half,
            reid_preprocess=reid_preprocess,
        )


__all__ = ("NativeOccluBoostTracker",)
