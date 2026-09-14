"""Regulated mask guidance for ambiguous and isolated box associations.

The McByte++ paper gates a mask-fill cost adjustment on visibility, coverage,
and fill. Here isolation conservatively requires both endpoints to have no
admissible geometric partner, preserving every unambiguous box match.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MIN_MASK_COVERAGE = 0.90
MIN_MASK_FILL = 0.05


def apply_mask_guidance(
    cost_matrix: np.ndarray,
    detection_boxes: np.ndarray,
    track_masks: Sequence[np.ndarray | None],
    *,
    threshold: float,
    min_coverage: float = MIN_MASK_COVERAGE,
    min_fill: float = MIN_MASK_FILL,
) -> np.ndarray:
    """Copy track-by-detection costs and apply McByte++ mask guidance.

    ``detection_boxes`` contains AABB ``xyxy`` coordinates. ``track_masks``
    contains current-frame, full-resolution propagated masks in track-row
    order; ``None`` denotes a track without a mask. Positive mask values are
    foreground, accepting binary masks and EdgeTAM logits thresholded at zero.
    Visibility means at least one foreground pixel, as in McByte++.

    A pair is ambiguous when its original cost is at most ``threshold`` and
    another entry in its row or column also meets that threshold. An isolated
    pair has no admissible entry in either its row or its column. Both tests
    use original costs; clear pairs and their competing entries stay unchanged.
    Subtract mask fill only when coverage and fill meet their configured minima
    (0.90 and 0.05 by default). Negative adjusted costs remain valid solver inputs.
    """
    costs = np.asarray(cost_matrix, dtype=float)
    boxes = np.asarray(detection_boxes, dtype=float)
    if costs.ndim != 2:
        raise ValueError("Mask guidance requires a track-by-detection cost matrix")
    if boxes.shape != (costs.shape[1], 4):
        raise ValueError("Mask guidance requires one AABB xyxy box per detection column")
    if not np.isfinite(boxes).all():
        raise ValueError("Mask guidance detection boxes must contain finite coordinates")
    if len(track_masks) != costs.shape[0]:
        raise ValueError("Mask guidance requires one propagated mask per track row")

    adjusted = costs.copy()
    if costs.size == 0:
        return adjusted

    admissible = costs <= threshold
    row_counts = admissible.sum(axis=1)
    column_counts = admissible.sum(axis=0)
    ambiguous = admissible & ((row_counts[:, None] > 1) | (column_counts[None, :] > 1))
    isolated = (row_counts[:, None] == 0) & (column_counts[None, :] == 0)
    eligible = ambiguous | isolated
    for row in np.flatnonzero(eligible.any(axis=1)):
        mask = track_masks[row]
        if mask is None:
            continue
        foreground = np.asarray(mask)
        if foreground.ndim != 2:
            raise ValueError("Propagated masks must be two-dimensional full-resolution arrays")
        if foreground.dtype != np.bool_:
            foreground = foreground > 0
        mask_area = np.count_nonzero(foreground)
        if mask_area == 0:
            continue

        image_height, image_width = foreground.shape
        for column in np.flatnonzero(eligible[row]):
            x1, y1, x2, y2 = boxes[column]
            if x2 <= x1 or y2 <= y1:
                continue
            # Rasterize the actual xyxy extent before clipping it to the image.
            x = max(0, min(image_width, int(np.floor(x1))))
            y = max(0, min(image_height, int(np.floor(y1))))
            right = max(0, min(image_width, int(np.ceil(x2))))
            bottom = max(0, min(image_height, int(np.ceil(y2))))
            if right <= x or bottom <= y:
                continue
            intersection = np.count_nonzero(foreground[y:bottom, x:right])
            fill = intersection / ((bottom - y) * (right - x))
            coverage = intersection / mask_area
            if fill >= min_fill and coverage >= min_coverage:
                adjusted[row, column] -= fill

    return adjusted
