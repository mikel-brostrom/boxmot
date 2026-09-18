"""McByte++ mask conditioning for admissible, ambiguous box associations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

MIN_MASK_COVERAGE = 0.90
MIN_MASK_FILL = 0.05
_MAX_CROP_BATCH_PIXELS = 4 * 1024 * 1024
_MAX_CROP_BATCH_SIZE = 64
_TensorCrop = tuple[torch.Tensor, int, int, int, int]
_CropGroups = dict[tuple[torch.dtype, int, int], list[_TensorCrop]]


def _reference_box_crop(box: np.ndarray, height: int, width: int) -> tuple[int, int, int, int] | None:
    """Rasterize as reference integer tlwh, clamping the origin before the extent."""
    x1, y1, x2, y2 = box
    x, y = max(0, int(x1)), max(0, int(y1))
    right = min(width, x + int(x2 - x1))
    bottom = min(height, y + int(y2 - y1))
    if right <= x or bottom <= y:
        return None
    return x, y, right, bottom


def apply_mask_guidance(
    cost_matrix: np.ndarray,
    detection_boxes: np.ndarray,
    track_masks: Sequence[np.ndarray | torch.Tensor | None],
    *,
    threshold: float,
    min_coverage: float = MIN_MASK_COVERAGE,
    min_fill: float = MIN_MASK_FILL,
) -> np.ndarray:
    """Copy track-by-detection costs and apply McByte++ mask guidance.

    ``detection_boxes`` contains AABB ``xyxy`` coordinates. ``track_masks``
    contains current-frame, full-resolution propagated masks in track-row
    order, as NumPy arrays or device tensors; ``None`` denotes a track without
    a mask. Positive mask values are foreground, accepting binary masks and
    EdgeTAM logits thresholded at zero.
    Visibility means at least one foreground pixel, as in McByte++.

    A pair is ambiguous when its original cost is at most ``threshold`` and
    another entry in its row or column also meets that threshold. Pairs above
    the original threshold never receive a mask bonus. Clear one-to-one pairs
    retain their costs and add 10 to their row and column competitors, even
    without usable masks, matching the reference conditioned assignment.
    Box crops truncate tlwh coordinates and clamp the origin before the extent.
    Subtract mask fill only when coverage and fill meet their configured minima
    (0.90 and 0.05 by default). Negative adjusted costs remain valid solver inputs.
    Tensor masks stay on their device; only exact pixel counts cross to the CPU,
    once per device and association stage. Ratios and gates retain NumPy precision.
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
    clear = admissible & ~ambiguous
    clear_rows, clear_columns = np.nonzero(clear)
    adjusted[clear_rows, :] += 10.0
    adjusted[:, clear_columns] += 10.0
    adjusted[clear] = costs[clear]
    tensor_work: dict[torch.device, tuple[list[torch.Tensor], _CropGroups]] = {}
    for row in np.flatnonzero(ambiguous.any(axis=1)):
        mask = track_masks[row]
        if mask is None:
            continue
        if isinstance(mask, torch.Tensor):
            if mask.ndim != 2:
                raise ValueError("Propagated masks must be two-dimensional full-resolution arrays")
            foreground = mask if mask.dtype == torch.bool else mask > 0
            image_height, image_width = foreground.shape
            areas, crop_groups = tensor_work.setdefault(mask.device, ([], {}))
            area_index = None
            # Explicit int32 avoids MPS's implicit int64 reduction and keeps
            # counts exact. Larger-than-int32 images require int64 counts.
            count_dtype = torch.int32 if foreground.numel() <= np.iinfo(np.int32).max else torch.int64
            for column in np.flatnonzero(ambiguous[row]):
                crop = _reference_box_crop(boxes[column], image_height, image_width)
                if crop is None:
                    continue
                x, y, right, bottom = crop
                if area_index is None:
                    area_index = len(areas)
                    areas.append(foreground.sum(dtype=count_dtype))
                group = crop_groups.setdefault((count_dtype, bottom - y, right - x), [])
                group.append((foreground[y:bottom, x:right], row, column, area_index, (bottom - y) * (right - x)))
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
        for column in np.flatnonzero(ambiguous[row]):
            crop = _reference_box_crop(boxes[column], image_height, image_width)
            if crop is None:
                continue
            x, y, right, bottom = crop
            intersection = np.count_nonzero(foreground[y:bottom, x:right])
            fill = intersection / ((bottom - y) * (right - x))
            coverage = intersection / mask_area
            if fill >= min_fill and coverage >= min_coverage:
                adjusted[row, column] -= fill

    for areas, crop_groups in tensor_work.values():
        if not areas:
            continue
        counts = [torch.stack(areas)]
        adjustments = []
        offset = len(areas)
        for (count_dtype, height, width), crops in crop_groups.items():
            batch_size = max(1, min(_MAX_CROP_BATCH_SIZE, _MAX_CROP_BATCH_PIXELS // (height * width)))
            for start in range(0, len(crops), batch_size):
                chunk = crops[start : start + batch_size]
                if len(chunk) == 1:
                    # Avoid copying a full-resolution crop larger than the budget.
                    intersections = chunk[0][0].sum(dtype=count_dtype).reshape(1)
                else:
                    intersections = torch.stack([crop[0] for crop in chunk]).sum(dim=(1, 2), dtype=count_dtype)
                counts.append(intersections)
                adjustments.extend(
                    (row, column, area_index, offset + index, box_area)
                    for index, (_, row, column, area_index, box_area) in enumerate(chunk)
                )
                offset += len(chunk)
        values = torch.cat(counts).cpu().numpy()
        for row, column, area_index, intersection_index, box_area in adjustments:
            mask_area = int(values[area_index])
            if mask_area == 0:
                continue
            intersection = int(values[intersection_index])
            fill = intersection / box_area
            coverage = intersection / mask_area
            if fill >= min_fill and coverage >= min_coverage:
                adjusted[row, column] -= fill

    return adjusted
