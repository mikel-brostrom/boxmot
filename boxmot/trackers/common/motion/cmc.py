from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from boxmot.motion.cmc import create_cmc as create_motion_cmc
from boxmot.trackers.common.detections.layout import DetectionLayout

CMC_RESET_ATTRS = (
    "prev_img",
    "prev_img_aligned",
    "prev_frame",
    "prev_keypoints",
    "prev_descriptors",
)


def create_cmc(method: str | None, *, enabled: bool = True, **kwargs):
    """Create a CMC estimator or return ``None`` for disabled CMC."""
    if not enabled or method is None:
        return None
    return create_motion_cmc(method, **kwargs)


def cmc_detection_boxes(dets: np.ndarray, layout: DetectionLayout) -> np.ndarray:
    """Return native AABB or OBB geometry for CMC masking/estimation."""
    return layout.boxes(dets)


def apply_cmc_to_tracks(
    cmc,
    img: np.ndarray | None,
    dets: np.ndarray,
    layout: DetectionLayout,
    tracks: Sequence,
    *,
    update_method: str = "camera_update",
) -> np.ndarray | None:
    """Apply CMC to tracks and return the estimated warp matrix.

    Disabled CMC is a no-op and returns ``None``. Track-specific update methods
    are passed by name because tracker implementations expose different
    correction method names.
    """
    if cmc is None:
        return None
    if img is None:
        raise ValueError("img is required when camera-motion compensation is enabled")

    warp = cmc.apply(img, cmc_detection_boxes(dets, layout))
    for track in tracks:
        getattr(track, update_method)(warp)
    return warp


def reset_cmc(cmc) -> None:
    """Reset a CMC estimator, including estimators without a ``reset`` method."""
    if cmc is None:
        return

    reset = getattr(cmc, "reset", None)
    if callable(reset):
        reset()

    for attr_name in CMC_RESET_ATTRS:
        if hasattr(cmc, attr_name):
            setattr(cmc, attr_name, None)
    if hasattr(cmc, "initialized"):
        cmc.initialized = False
