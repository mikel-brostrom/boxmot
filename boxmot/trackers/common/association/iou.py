import math

import numpy as np

_EQUIVALENT_OBB_LINEAR_ATOL = 1e-9
_EQUIVALENT_OBB_RELATIVE_ATOL = 8.0 * np.finfo(np.float32).eps


def _obb_geometry(boxes: np.ndarray) -> np.ndarray:
    """Return validated ``xywha`` geometry from OBB rows with optional metadata."""
    values = np.asarray(boxes, dtype=float)
    if values.ndim == 1:
        if values.size == 0:
            raise ValueError("Empty OBB association rows must preserve at least five columns.")
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 5:
        raise ValueError(f"OBB association expects rows with at least 5 columns, got {values.shape}.")
    if values.size == 0:
        return np.empty((0, 5), dtype=float)
    geometry = values[:, :5].copy()
    if not np.isfinite(geometry).all():
        raise ValueError("OBB association geometry must contain only finite values.")
    if np.any(geometry[:, 2:4] <= 0.0):
        raise ValueError("OBB association widths and heights must be positive.")
    geometry[:, 4] = np.fromiter(
        (math.remainder(float(angle), math.pi) for angle in geometry[:, 4]),
        dtype=float,
        count=len(geometry),
    )
    return geometry


def _pi_periodic_distance(delta: np.ndarray) -> np.ndarray:
    """Return absolute angular distance for pi-periodic OBB orientations."""
    return np.abs((delta + (np.pi / 2.0)) % np.pi - (np.pi / 2.0))


def _equivalent_obb_mask(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Return pairwise equivalence under direct and width/height-swapped OBB forms."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=bool)

    center_equal = np.all(boxes1[:, None, :2] == boxes2[None, :, :2], axis=2)
    angle_delta = boxes1[:, None, 4] - boxes2[None, :, 4]
    direct = (
        (boxes1[:, None, 2] == boxes2[None, :, 2])
        & (boxes1[:, None, 3] == boxes2[None, :, 3])
        & (_pi_periodic_distance(angle_delta) == 0.0)
    )
    swapped_angle_delta = _pi_periodic_distance(angle_delta - (np.pi / 2.0))
    corner_radius = np.hypot(boxes1[:, None, 2] / 2.0, boxes1[:, None, 3] / 2.0)
    swapped_corner_displacement = corner_radius * (2.0 * np.sin(swapped_angle_delta / 2.0))
    swapped_equivalent = (swapped_corner_displacement <= _EQUIVALENT_OBB_LINEAR_ATOL) & (
        swapped_corner_displacement <= corner_radius * _EQUIVALENT_OBB_RELATIVE_ATOL
    )
    swapped = (
        (boxes1[:, None, 2] == boxes2[None, :, 3]) & (boxes1[:, None, 3] == boxes2[None, :, 2]) & swapped_equivalent
    )
    return center_equal & (direct | swapped)


def _cross_2d(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return the scalar 2D cross product."""
    return float(lhs[0] * rhs[1] - lhs[1] * rhs[0])


def _polygon_signed_area(points: np.ndarray) -> float:
    """Return the signed area of a polygon."""
    if len(points) < 3:
        return 0.0
    return 0.5 * float(
        np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )


def _polygon_area(points: np.ndarray) -> float:
    """Return the absolute area of a polygon."""
    return abs(_polygon_signed_area(points))


def _rectangle_corners(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle: float,
) -> np.ndarray:
    """Return counter-clockwise double-precision rotated-rectangle corners."""
    half_width = width / 2.0
    half_height = height / 2.0
    offsets = np.array(
        (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ),
        dtype=float,
    )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    return offsets @ rotation.T + (center_x, center_y)


def _normalized_pair_frame(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return a symmetric normalized half-delta, relative angles, and scale."""
    half_delta_x = rhs[0] / 2.0 - lhs[0] / 2.0
    half_delta_y = rhs[1] / 2.0 - lhs[1] / 2.0
    scale = float(
        max(
            abs(half_delta_x),
            abs(half_delta_y),
            lhs[2],
            lhs[3],
            rhs[2],
            rhs[3],
        )
    )
    normalized_half_delta_x = half_delta_x / scale
    normalized_half_delta_y = half_delta_y / scale
    half_distance = float(np.hypot(normalized_half_delta_x, normalized_half_delta_y))
    if half_distance > 0.0:
        unit_x = normalized_half_delta_x / half_distance
        unit_y = normalized_half_delta_y / half_distance
        frame_angle = 0.5 * float(np.arctan2(2.0 * unit_x * unit_y, unit_x**2 - unit_y**2))
    else:
        frame_angle = float(lhs[4] / 2.0 + rhs[4] / 2.0)
    cosine = np.cos(frame_angle)
    sine = np.sin(frame_angle)
    local_half_delta_x = cosine * normalized_half_delta_x + sine * normalized_half_delta_y
    local_half_delta_y = -sine * normalized_half_delta_x + cosine * normalized_half_delta_y
    lhs_angle = math.remainder(float(lhs[4] - frame_angle), math.pi)
    rhs_angle = math.remainder(float(rhs[4] - frame_angle), math.pi)
    return (
        local_half_delta_x,
        local_half_delta_y,
        lhs_angle,
        rhs_angle,
        scale,
    )


def _normalized_local_obb_corners(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return normalized pair-local corners, scale, and squared center distance."""
    half_delta_x, half_delta_y, lhs_angle, rhs_angle, scale = _normalized_pair_frame(lhs, rhs)
    lhs_corners = _rectangle_corners(
        -half_delta_x,
        -half_delta_y,
        lhs[2] / scale,
        lhs[3] / scale,
        lhs_angle,
    )
    rhs_corners = _rectangle_corners(
        half_delta_x,
        half_delta_y,
        rhs[2] / scale,
        rhs[3] / scale,
        rhs_angle,
    )
    center_distance_squared = 4.0 * (half_delta_x**2 + half_delta_y**2)
    return lhs_corners, rhs_corners, scale, center_distance_squared


def _line_intersection(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    clip_start: np.ndarray,
    clip_end: np.ndarray,
) -> np.ndarray:
    """Intersect a segment's line with a clipping edge's line."""
    segment = segment_end - segment_start
    clip_edge = clip_end - clip_start
    denominator = _cross_2d(segment, clip_edge)
    if denominator == 0.0:
        return segment_end.copy()
    factor = _cross_2d(clip_start - segment_start, clip_edge) / denominator
    return segment_start + factor * segment


def _convex_polygon_intersection(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Clip one convex polygon by another using double precision."""
    if _polygon_signed_area(clipper) < 0.0:
        clipper = clipper[::-1]

    output = [point.copy() for point in subject]
    for clip_start, clip_end in zip(clipper, np.roll(clipper, -1, axis=0)):
        if not output:
            break
        input_points = output
        output = []
        segment_start = input_points[-1]
        for segment_end in input_points:
            end_inside = _cross_2d(clip_end - clip_start, segment_end - clip_start) >= 0.0
            start_inside = _cross_2d(clip_end - clip_start, segment_start - clip_start) >= 0.0
            if end_inside:
                if not start_inside:
                    output.append(_line_intersection(segment_start, segment_end, clip_start, clip_end))
                output.append(segment_end)
            elif start_inside:
                output.append(_line_intersection(segment_start, segment_end, clip_start, clip_end))
            segment_start = segment_end

    return np.asarray(output, dtype=float).reshape(-1, 2)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the convex hull of a small 2D point set in counter-clockwise order."""
    unique_points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(unique_points) <= 1:
        return np.asarray(unique_points, dtype=float).reshape(-1, 2)

    def build_half(sequence: list[tuple[float, float]]) -> list[tuple[float, float]]:
        half: list[tuple[float, float]] = []
        for point in sequence:
            while len(half) >= 2:
                first = np.asarray(half[-2])
                second = np.asarray(half[-1])
                if _cross_2d(second - first, np.asarray(point) - second) > 0.0:
                    break
                half.pop()
            half.append(point)
        return half

    lower = build_half(unique_points)
    upper = build_half(list(reversed(unique_points)))
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _minimum_area_enclosing_rectangle_diagonal(points: np.ndarray) -> float:
    """Return the squared diagonal of the minimum-area enclosing rectangle."""
    hull = _convex_hull(points)
    best_area = np.inf
    best_diagonal = np.inf
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end - start
        edge_length = np.hypot(edge[0], edge[1])
        if edge_length == 0.0:
            continue
        axis = edge / edge_length
        perpendicular = np.array((-axis[1], axis[0]))
        along = hull @ axis
        across = hull @ perpendicular
        width = float(np.max(along) - np.min(along))
        height = float(np.max(across) - np.min(across))
        area = width * height
        diagonal = width**2 + height**2
        area_tolerance = 1e-12 * max(abs(area), abs(best_area)) if np.isfinite(best_area) else 0.0
        if (
            not np.isfinite(best_area)
            or area < best_area - area_tolerance
            or (abs(area - best_area) <= area_tolerance and diagonal < best_diagonal)
        ):
            best_area = area
            best_diagonal = diagonal
    return best_diagonal


def _obb_distance_penalty_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Return pairwise center-distance penalties in normalized local coordinates."""
    penalties = np.empty((len(boxes1), len(boxes2)), dtype=float)
    for row, lhs in enumerate(boxes1):
        for col, rhs in enumerate(boxes2):
            lhs_corners, rhs_corners, _, center_distance_squared = _normalized_local_obb_corners(lhs, rhs)
            diagonal = _minimum_area_enclosing_rectangle_diagonal(np.vstack((lhs_corners, rhs_corners)))
            penalties[row, col] = center_distance_squared / diagonal if diagonal > 0.0 else 0.0
    return penalties


def _normalized_obb_area_terms(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    """Return normalized area sum and convex-enclosure area for an OBB pair."""
    lhs_corners, rhs_corners, scale, _ = _normalized_local_obb_corners(lhs, rhs)
    area_sum = (lhs[2] / scale) * (lhs[3] / scale) + (rhs[2] / scale) * (rhs[3] / scale)
    enclosure = _polygon_area(_convex_hull(np.vstack((lhs_corners, rhs_corners))))
    return area_sum, enclosure


def _obb_enclosing_half_extents(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return global-axis half extents without adding them to box centers."""
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    cos_a = np.abs(np.cos(boxes[:, 4]))
    sin_a = np.abs(np.sin(boxes[:, 4]))
    extent_x = half_w * cos_a + half_h * sin_a
    extent_y = half_w * sin_a + half_h * cos_a
    return extent_x, extent_y


def _double_precision_obb_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Compute pairwise OBB IoU with double-precision convex clipping."""
    lhs_corners, rhs_corners, scale, _ = _normalized_local_obb_corners(lhs, rhs)
    if tuple(rhs_corners.ravel()) < tuple(lhs_corners.ravel()):
        lhs_corners, rhs_corners = rhs_corners, lhs_corners
    intersection = _convex_polygon_intersection(lhs_corners, rhs_corners)
    if len(intersection) < 3:
        return 0.0
    lhs_area = (lhs[2] / scale) * (lhs[3] / scale)
    rhs_area = (rhs[2] / scale) * (rhs[3] / scale)
    intersection_area = min(_polygon_area(intersection), lhs_area, rhs_area)
    union_area = lhs_area + rhs_area - intersection_area
    return float(np.clip(intersection_area / union_area, 0.0, 1.0))


def _iou_obb_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Compute an NxM rotated IoU matrix using normalized pair-local geometry.

    Steps:
    1. Compute enclosing axis-aligned bounding boxes for all OBBs (vectorized).
    2. Compute AABB overlap mask to identify candidate pairs (vectorized).
    3. Use normalized double-precision polygon clipping for every candidate.

    This skips the majority of pairs (typically >80%) that have zero IoU.
    """
    N, M = len(bboxes1), len(bboxes2)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float64)

    # -- Vectorized enclosing AABB computation --
    # For a rotated rect (cx, cy, w, h, angle), the enclosing AABB half-extents are:
    #   ex = |w/2 * cos(a)| + |h/2 * sin(a)|
    #   ey = |w/2 * sin(a)| + |h/2 * cos(a)|
    cx1, cy1 = bboxes1[:, 0], bboxes1[:, 1]
    cx2, cy2 = bboxes2[:, 0], bboxes2[:, 1]
    ex1, ey1 = _obb_enclosing_half_extents(bboxes1)
    ex2, ey2 = _obb_enclosing_half_extents(bboxes2)

    # AABB bounds: (cx - ex, cy - ey, cx + ex, cy + ey)
    # Compare half-distances and half-sums so subtracting opposite large
    # centers or adding near-maximum extents cannot overflow.
    half_dx = np.abs(cx1[:, None] / 2.0 - cx2[None, :] / 2.0)
    half_dy = np.abs(cy1[:, None] / 2.0 - cy2[None, :] / 2.0)
    half_sum_ex = ex1[:, None] / 2.0 + ex2[None, :] / 2.0
    half_sum_ey = ey1[:, None] / 2.0 + ey2[None, :] / 2.0

    equivalent = _equivalent_obb_mask(bboxes1, bboxes2)

    # Candidate mask: AABBs overlap. Equivalent encodings have exact identity
    # semantics and bypass the geometric intersection implementation.
    candidates = (half_dx < half_sum_ex) & (half_dy < half_sum_ey) & ~equivalent
    iou_matrix = np.zeros((N, M), dtype=np.float64)
    iou_matrix[equivalent] = 1.0

    # -- Compute rotated IoU only for candidate pairs --
    cand_i, cand_j = np.nonzero(candidates)
    if len(cand_i) == 0:
        return iou_matrix

    for idx in range(len(cand_i)):
        i, j = int(cand_i[idx]), int(cand_j[idx])
        iou_matrix[i, j] = _double_precision_obb_iou(bboxes1[i], bboxes2[j])

    return iou_matrix


class AssociationFunction:
    def __init__(self, w, h, asso_mode="iou"):
        """
        Initializes the AssociationFunction class with the necessary parameters for bounding box operations.
        The association function is selected based on the `asso_mode` string provided during class creation.

        Parameters:
        w (int): The width of the frame, used for normalizing centroid distance.
        h (int): The height of the frame, used for normalizing centroid distance.
        asso_mode (str): The association function to use (e.g., "iou", "giou", "centroid", etc.).
        """
        self.w = w
        self.h = h
        self.asso_func = self._get_asso_func(asso_mode)

    @staticmethod
    def iou_batch(bboxes1, bboxes2) -> np.ndarray:
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / (
            (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
            + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            - wh
        )
        return o

    @staticmethod
    def iou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        return _iou_obb_matrix(_obb_geometry(bboxes1), _obb_geometry(bboxes2))

    @staticmethod
    def diou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        """Compute distance IoU for oriented boxes."""
        boxes1 = _obb_geometry(bboxes1)
        boxes2 = _obb_geometry(bboxes2)
        iou = _iou_obb_matrix(boxes1, boxes2)
        if boxes1.size == 0 or boxes2.size == 0:
            return iou

        distance_penalty = _obb_distance_penalty_matrix(boxes1, boxes2)
        diou = iou - distance_penalty
        similarity = np.clip((diou + 1.0) / 2.0, 0.0, 1.0)
        return np.where(_equivalent_obb_mask(boxes1, boxes2), 1.0, similarity)

    @staticmethod
    def giou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        """Compute normalized generalized-IoU similarity for oriented boxes.

        The enclosure is the convex hull of both rotated rectangles, which is
        the smallest convex region containing their geometry.
        """
        boxes1 = _obb_geometry(bboxes1)
        boxes2 = _obb_geometry(bboxes2)
        iou = _iou_obb_matrix(boxes1, boxes2)
        if boxes1.size == 0 or boxes2.size == 0:
            return iou

        equivalent = _equivalent_obb_mask(boxes1, boxes2)
        similarity = np.empty_like(iou)
        for row in range(len(boxes1)):
            for col in range(len(boxes2)):
                if equivalent[row, col]:
                    similarity[row, col] = 1.0
                    continue
                area_sum, enclosing_area = _normalized_obb_area_terms(boxes1[row], boxes2[col])
                union = area_sum / (1.0 + iou[row, col])
                enclosing_area = max(enclosing_area, union)
                empty_fraction = (enclosing_area - union) / enclosing_area
                giou = iou[row, col] - empty_fraction
                similarity[row, col] = np.clip((giou + 1.0) / 2.0, 0.0, 1.0)
        return similarity

    @staticmethod
    def ciou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        """Compute an experimental CIoU-style similarity for oriented boxes.

        The aspect-ratio term uses ordered long and short sides so equivalent
        ``(w, h, theta)`` and ``(h, w, theta + pi/2)`` rows score identically.
        """
        boxes1 = _obb_geometry(bboxes1)
        boxes2 = _obb_geometry(bboxes2)
        iou = _iou_obb_matrix(boxes1, boxes2)
        if boxes1.size == 0 or boxes2.size == 0:
            return iou

        distance_penalty = _obb_distance_penalty_matrix(boxes1, boxes2)

        long1 = np.maximum(boxes1[:, 2], boxes1[:, 3])
        short1 = np.minimum(boxes1[:, 2], boxes1[:, 3])
        long2 = np.maximum(boxes2[:, 2], boxes2[:, 3])
        short2 = np.minimum(boxes2[:, 2], boxes2[:, 3])
        aspect1 = np.arctan(long1 / short1)
        aspect2 = np.arctan(long2 / short2)
        aspect_penalty = (4.0 / (np.pi**2)) * (aspect1[:, None] - aspect2[None, :]) ** 2
        alpha_denominator = 1.0 - iou + aspect_penalty
        alpha = np.divide(
            aspect_penalty,
            alpha_denominator,
            out=np.zeros_like(aspect_penalty),
            where=alpha_denominator > 0.0,
        )
        ciou = iou - distance_penalty - alpha * aspect_penalty
        similarity = np.clip((ciou + 1.0) / 2.0, 0.0, 1.0)
        return np.where(_equivalent_obb_mask(boxes1, boxes2), 1.0, similarity)

    @staticmethod
    def hmiou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        """Compute experimental global-y height-modulated IoU for oriented boxes."""
        boxes1 = _obb_geometry(bboxes1)
        boxes2 = _obb_geometry(bboxes2)
        iou = _iou_obb_matrix(boxes1, boxes2)
        if boxes1.size == 0 or boxes2.size == 0:
            return iou

        half_delta_y = boxes2[None, :, 1] / 2.0 - boxes1[:, None, 1] / 2.0
        scale = np.maximum(np.abs(half_delta_y), boxes1[:, None, 2])
        scale = np.maximum(scale, boxes1[:, None, 3])
        scale = np.maximum(scale, boxes2[None, :, 2])
        scale = np.maximum(scale, boxes2[None, :, 3])
        center_distance = 2.0 * np.abs(half_delta_y / scale)
        half_height1 = 0.5 * (
            (boxes1[:, None, 2] / scale) * np.abs(np.sin(boxes1[:, None, 4]))
            + (boxes1[:, None, 3] / scale) * np.abs(np.cos(boxes1[:, None, 4]))
        )
        half_height2 = 0.5 * (
            (boxes2[None, :, 2] / scale) * np.abs(np.sin(boxes2[None, :, 4]))
            + (boxes2[None, :, 3] / scale) * np.abs(np.cos(boxes2[None, :, 4]))
        )
        overlap_height = np.maximum(
            0.0,
            np.minimum(
                np.minimum(2.0 * half_height1, 2.0 * half_height2),
                half_height1 + half_height2 - center_distance,
            ),
        )
        enclosing_height = np.maximum(
            np.maximum(2.0 * half_height1, 2.0 * half_height2),
            half_height1 + half_height2 + center_distance,
        )
        similarity = np.divide(
            iou * overlap_height,
            enclosing_height,
            out=np.zeros_like(iou),
            where=enclosing_height > 0.0,
        )
        return np.where(_equivalent_obb_mask(boxes1, boxes2), 1.0, similarity)

    @staticmethod
    def hmiou_batch(bboxes1, bboxes2):
        """
        Compute a modified Intersection over Union (hIoU) between two batches of bounding boxes,
        incorporating a vertical overlap ratio.

        Parameters:
        - bboxes1: (N, 4) array of bounding boxes [x1, y1, x2, y2]
        - bboxes2: (M, 4) array of bounding boxes [x1, y1, x2, y2]

        Returns:
        - hmiou: (N, M) array where hmiou[i, j] is the modified IoU between bboxes1[i] and bboxes2[j]
        """
        # Expand dimensions for broadcasting
        bboxes1 = np.expand_dims(bboxes1, axis=1)  # Shape: (N, 1, 4)
        bboxes2 = np.expand_dims(bboxes2, axis=0)  # Shape: (1, M, 4)

        # Compute vertical overlap ratio 'o'
        intersect_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        intersect_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        intersection_height = np.maximum(0.0, intersect_y2 - intersect_y1)

        union_y1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        union_y2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        union_height = np.maximum(1e-10, union_y2 - union_y1)

        o = intersection_height / union_height

        # Compute standard IoU
        inter_x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        inter_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        inter_x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        inter_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])  # Shape: (N, 1)
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])  # Shape: (1, M)

        union_area = area1 + area2 - inter_area

        iou = inter_area / (union_area + 1e-10)

        # Modify IoU with vertical overlap ratio
        hmiou = iou * o

        return hmiou

    @staticmethod
    def giou_batch(bboxes1, bboxes2) -> np.ndarray:
        """
        :param bboxes1: predict of bbox(N,4)(x1,y1,x2,y2)
        :param bboxes2: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        """
        # Ensure predict's bbox form
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h  # Intersection area

        # Compute areas of individual boxes
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        # Union area
        union_area = area1 + area2 - wh

        iou = wh / union_area

        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        wc = xxc2 - xxc1
        hc = yyc2 - yyc1
        assert (wc > 0).all() and (hc > 0).all()
        area_enclose = wc * hc  # Area of the smallest enclosing box

        # Corrected GIoU computation
        giou = iou - (area_enclose - union_area) / area_enclose
        giou = (giou + 1.0) / 2.0  # Resize from (-1,1) to (0,1)
        return giou

    def centroid_batch(self, bboxes1, bboxes2) -> np.ndarray:
        centroids1 = np.stack(
            ((bboxes1[..., 0] + bboxes1[..., 2]) / 2, (bboxes1[..., 1] + bboxes1[..., 3]) / 2), axis=-1
        )
        centroids2 = np.stack(
            ((bboxes2[..., 0] + bboxes2[..., 2]) / 2, (bboxes2[..., 1] + bboxes2[..., 3]) / 2), axis=-1
        )

        centroids1 = np.expand_dims(centroids1, 1)
        centroids2 = np.expand_dims(centroids2, 0)

        distances = np.sqrt(np.sum((centroids1 - centroids2) ** 2, axis=-1))
        norm_factor = np.sqrt(self.w**2 + self.h**2)
        normalized_distances = distances / norm_factor

        return 1 - normalized_distances

    def centroid_batch_obb(self, bboxes1, bboxes2) -> np.ndarray:
        centroids1 = np.stack((bboxes1[..., 0], bboxes1[..., 1]), axis=-1)
        centroids2 = np.stack((bboxes2[..., 0], bboxes2[..., 1]), axis=-1)

        centroids1 = np.expand_dims(centroids1, 1)
        centroids2 = np.expand_dims(centroids2, 0)

        distances = np.sqrt(np.sum((centroids1 - centroids2) ** 2, axis=-1))
        norm_factor = np.sqrt(self.w**2 + self.h**2)
        normalized_distances = distances / norm_factor

        return 1 - normalized_distances

    @staticmethod
    def ciou_batch(bboxes1, bboxes2) -> np.ndarray:
        """
        Calculate Complete Intersection over Union (CIoU) for batches of bounding boxes.

        :param bboxes1: Predicted bounding boxes of shape (N, 4) as (x1, y1, x2, y2)
        :param bboxes2: Ground truth bounding boxes of shape (N, 4) as (x1, y1, x2, y2)
        :return: CIoU scores scaled between 0 and 1
        """
        epsilon = 1e-7  # Small value to prevent division by zero

        # Expand dimensions for broadcasting
        bboxes2 = np.expand_dims(bboxes2, 0)  # Shape: (1, M, 4)
        bboxes1 = np.expand_dims(bboxes1, 1)  # Shape: (N, 1, 4)

        # Calculate the intersection box
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h

        # Calculate IoU
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        iou = wh / (area1 + area2 - wh + epsilon)

        # Calculate center points
        centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
        centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
        centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
        centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

        # Calculate squared center distance
        inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

        # Calculate smallest enclosing box diagonal
        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2 + epsilon

        # Calculate aspect ratio consistency
        w1 = bboxes1[..., 2] - bboxes1[..., 0]
        h1 = bboxes1[..., 3] - bboxes1[..., 1]
        w2 = bboxes2[..., 2] - bboxes2[..., 0]
        h2 = bboxes2[..., 3] - bboxes2[..., 1]

        # Prevent division by zero
        h2 = h2 + epsilon
        h1 = h1 + epsilon
        arctan_diff = np.arctan(w2 / h2) - np.arctan(w1 / h1)
        v = (4 / (np.pi**2)) * (arctan_diff**2)

        # Calculate alpha
        S = 1 - iou
        alpha = v / (S + v + epsilon)

        # Compute CIoU
        ciou = iou - (inner_diag / outer_diag) - (alpha * v)

        # Scale CIoU to [0, 1]
        return np.clip((ciou + 1) / 2.0, 0.0, 1.0)

    @staticmethod
    def diou_batch(bboxes1, bboxes2) -> np.ndarray:
        """
        :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
        :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        """
        # for details should go to https://arxiv.org/pdf/1902.09630.pdf
        # ensure predict's bbox form
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        # calculate the intersection box
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        iou = wh / (
            (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
            + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            - wh
        )

        centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
        centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
        centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
        centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

        inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

        outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
        diou = iou - inner_diag / outer_diag

        return (diou + 1) / 2.0

    def _get_asso_func(self, asso_mode):
        """
        Returns the corresponding association function based on the provided mode string.

        Parameters:
        asso_mode (str): The association function to use (e.g., "iou", "giou", "centroid", etc.).

        Returns:
        function: The appropriate function for the association calculation.
        """
        ASSO_FUNCS = {
            "iou": AssociationFunction.iou_batch,
            "iou_obb": AssociationFunction.iou_batch_obb,
            "giou_obb": AssociationFunction.giou_batch_obb,
            "diou_obb": AssociationFunction.diou_batch_obb,
            "ciou_obb": AssociationFunction.ciou_batch_obb,
            "hmiou_obb": AssociationFunction.hmiou_batch_obb,
            "hmiou": AssociationFunction.hmiou_batch,
            "giou": AssociationFunction.giou_batch,
            "ciou": AssociationFunction.ciou_batch,
            "diou": AssociationFunction.diou_batch,
            "centroid": self.centroid_batch,  # only not being staticmethod
            "centroid_obb": self.centroid_batch_obb,
        }

        if asso_mode not in ASSO_FUNCS:
            raise ValueError(f"Invalid association mode: {asso_mode}. Choose from {list(ASSO_FUNCS.keys())}")

        return ASSO_FUNCS[asso_mode]
