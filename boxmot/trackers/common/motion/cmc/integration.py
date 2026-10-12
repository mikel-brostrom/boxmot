from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from boxmot.trackers.common.detections.layout import DetectionLayout
from boxmot.trackers.common.motion.cmc.base import BaseCMC


def resolve_cmc_min_conf(options: Mapping[str, Any]) -> float:
    """Freeze the camera mask threshold independently of later tracker tuning."""
    value = next(
        (options[name] for name in ("cmc_min_conf", "min_conf", "det_thresh") if options.get(name) is not None), 0.1
    )
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("CMC masking confidence must be finite and in [0, 1].")
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("CMC masking confidence must be finite and in [0, 1].")
    return threshold


def cmc_detection_boxes(dets: np.ndarray, layout: DetectionLayout, *, min_conf: float | None = None) -> np.ndarray:
    """Return native AABB or OBB geometry for CMC masking/estimation."""
    if min_conf is not None:
        dets = dets[layout.confidences(dets) >= min_conf]
    return layout.boxes(dets)


def apply_cmc_to_tracks(
    cmc: BaseCMC | None,
    img: np.ndarray | None,
    dets: np.ndarray,
    layout: DetectionLayout,
    tracks: Sequence,
    *,
    mask_boxes: np.ndarray | None = None,
) -> np.ndarray | None:
    """Apply CMC to tracks and return the estimated warp matrix."""
    if cmc is None:
        return None
    if img is None:
        raise ValueError("img is required when camera-motion compensation is enabled")

    warp = cmc.apply(img, cmc_detection_boxes(dets, layout) if mask_boxes is None else mask_boxes)
    groups = {}
    for track in tracks:
        groups.setdefault(type(track), []).append(track)
    for track_type, group in groups.items():
        multi_update = getattr(track_type, "multi_camera_update", None)
        if multi_update is not None:
            multi_update(group, warp)
        else:
            for track in group:
                track.camera_update(warp)
    return warp


def reset_cmc(cmc: BaseCMC | None) -> None:
    """Reset a CMC estimator when camera-motion compensation is enabled."""
    if cmc is None:
        return
    cmc.reset()
