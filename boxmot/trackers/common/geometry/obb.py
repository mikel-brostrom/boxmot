from __future__ import annotations

import numpy as np


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Normalize radians to the half-open range [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_pi_periodic(delta: float) -> float:
    """Wrap angle deltas for equivalent OBB forms with pi-period symmetry."""
    return float((delta + (np.pi / 2.0)) % np.pi - (np.pi / 2.0))


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Return corners in top-left, top-right, bottom-right, bottom-left order."""
    arr = np.asarray(corners, dtype=np.float32)
    single = arr.ndim == 2
    if single:
        arr = arr.reshape(1, 4, 2)

    ordered = np.empty_like(arr)
    rows = np.arange(arr.shape[0])
    sums = arr.sum(axis=2)
    diffs = np.diff(arr, axis=2).reshape(arr.shape[0], 4)

    ordered[:, 0] = arr[rows, np.argmin(sums, axis=1)]
    ordered[:, 2] = arr[rows, np.argmax(sums, axis=1)]
    ordered[:, 1] = arr[rows, np.argmin(diffs, axis=1)]
    ordered[:, 3] = arr[rows, np.argmax(diffs, axis=1)]
    return ordered[0] if single else ordered


def xywha_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, w, h, angle]`` OBB boxes to ordered corner rows."""
    arr = np.asarray(boxes, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 5)

    corners = np.empty((arr.shape[0], 4, 2), dtype=np.float32)
    for i, (cx, cy, w, h, angle) in enumerate(arr):
        w = max(float(w), 1e-4)
        h = max(float(h), 1e-4)
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        rect = np.array(
            [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
            dtype=np.float32,
        )
        corners[i] = rect @ rot.T + np.array([cx, cy], dtype=np.float32)

    corners = order_corners(corners)
    flattened = corners.reshape(arr.shape[0], 8)
    return flattened[0] if single else flattened


def smooth_display_angle(
    prev_angle: float | None,
    current_box: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return a continuous display angle and canonicalized display OBB.

    Equivalent OBBs may swap width/height and shift the angle by pi/2. This
    keeps the displayed angle visually continuous while preserving the box.
    """
    box = np.asarray(current_box, dtype=np.float32).copy().reshape(-1)
    if box[3] > box[2]:
        box[2], box[3] = box[3], box[2]
        box[4] = box[4] + (np.pi / 2.0)

    target = float(normalize_angle(box[4]))
    angle = target if prev_angle is None else prev_angle + wrap_pi_periodic(target - prev_angle)
    box[4] = angle
    return float(angle), box


def smooth_obb_corners(
    box: np.ndarray,
    prev_angle: float | None,
) -> tuple[np.ndarray, float]:
    """Return display-smoothed OBB corners and the updated display angle."""
    angle, display_box = smooth_display_angle(prev_angle, box)
    return xywha_to_corners(display_box).astype(np.float32), angle


def align_obb_measurement(measurement: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Align equivalent ``(cx, cy, w, h, theta)`` forms to a reference box."""
    aligned = np.asarray(measurement, dtype=np.float32).copy().reshape(-1)
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)

    ref_w = max(float(ref[2]), 1e-6)
    ref_h = max(float(ref[3]), 1e-6)
    ref_theta = float(ref[4])
    w = max(float(aligned[2]), 1e-6)
    h = max(float(aligned[3]), 1e-6)
    theta = float(aligned[4])

    candidates = (
        (w, h, theta),
        (w, h, theta + np.pi),
        (h, w, theta + (np.pi / 2.0)),
        (h, w, theta - (np.pi / 2.0)),
    )
    best_cost = float("inf")
    best = candidates[0]
    for cand_w, cand_h, cand_theta in candidates:
        theta_aligned = ref_theta + float(normalize_angle(cand_theta - ref_theta))
        angle_cost = abs(theta_aligned - ref_theta)
        size_cost = abs(np.log(max(cand_w, 1e-6) / ref_w)) + abs(np.log(max(cand_h, 1e-6) / ref_h))
        cost = angle_cost + (0.05 * size_cost)
        if cost < best_cost:
            best_cost = cost
            best = (cand_w, cand_h, theta_aligned)

    aligned[2] = float(best[0])
    aligned[3] = float(best[1])
    aligned[4] = float(normalize_angle(best[2]))
    return aligned


def xywha_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Return enclosing AABBs for ``(cx, cy, w, h, theta)`` OBB boxes."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = boxes.reshape(-1, boxes.shape[-1])
    cx, cy, w, h, theta = (boxes[:, i].astype(float) for i in range(5))
    cos_t = np.abs(np.cos(theta))
    sin_t = np.abs(np.sin(theta))
    half_w = 0.5 * (w * cos_t + h * sin_t)
    half_h = 0.5 * (w * sin_t + h * cos_t)
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1).astype(np.float32)
