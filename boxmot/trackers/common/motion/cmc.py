from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from boxmot.motion.cmc import create_cmc as create_motion_cmc
from boxmot.motion.cmc.base_cmc import BaseCMC
from boxmot.trackers.common.detections.layout import DetectionLayout


def create_cmc(method: str | None, *, enabled: bool = True, **kwargs) -> BaseCMC | None:
    """Create a CMC estimator or return ``None`` for disabled CMC."""
    if not enabled or method is None:
        return None
    return create_motion_cmc(method, **kwargs)


def cmc_detection_boxes(dets: np.ndarray, layout: DetectionLayout) -> np.ndarray:
    """Return native AABB or OBB geometry for CMC masking/estimation."""
    return layout.boxes(dets)


def apply_cmc_to_tracks(
    cmc: BaseCMC | None,
    img: np.ndarray | None,
    dets: np.ndarray,
    layout: DetectionLayout,
    tracks: Sequence,
) -> np.ndarray | None:
    """Apply CMC to tracks and return the estimated warp matrix."""
    if cmc is None:
        return None
    if img is None:
        raise ValueError("img is required when camera-motion compensation is enabled")

    warp = cmc.apply(img, cmc_detection_boxes(dets, layout))
    for track in tracks:
        track.camera_update(warp)
    return warp


def reset_cmc(cmc: BaseCMC | None) -> None:
    """Reset a CMC estimator when camera-motion compensation is enabled."""
    if cmc is None:
        return
    cmc.reset()
