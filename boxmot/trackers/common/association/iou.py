import cv2 as cv
import numpy as np

_EQUIVALENT_OBB_LINEAR_ATOL = 1e-9
_EQUIVALENT_OBB_ANGLE_ATOL = 1e-6
_MINIMUM_OBB_SIDE = 1e-4


def _obb_geometry(boxes: np.ndarray) -> np.ndarray:
    """Return normalized ``xywha`` geometry from OBB rows with optional metadata."""
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
    geometry[:, 2:4] = np.maximum(geometry[:, 2:4], _MINIMUM_OBB_SIDE)
    return geometry


def _pi_periodic_distance(delta: np.ndarray) -> np.ndarray:
    """Return absolute angular distance for pi-periodic OBB orientations."""
    return np.abs((delta + (np.pi / 2.0)) % np.pi - (np.pi / 2.0))


def _equivalent_obb_mask(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Return pairwise equivalence under direct and width/height-swapped OBB forms."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=bool)

    center_equal = np.all(
        np.abs(boxes1[:, None, :2] - boxes2[None, :, :2]) <= _EQUIVALENT_OBB_LINEAR_ATOL,
        axis=2,
    )
    angle_delta = boxes1[:, None, 4] - boxes2[None, :, 4]
    direct = (
        (np.abs(boxes1[:, None, 2] - boxes2[None, :, 2]) <= _EQUIVALENT_OBB_LINEAR_ATOL)
        & (np.abs(boxes1[:, None, 3] - boxes2[None, :, 3]) <= _EQUIVALENT_OBB_LINEAR_ATOL)
        & (_pi_periodic_distance(angle_delta) <= _EQUIVALENT_OBB_ANGLE_ATOL)
    )
    swapped = (
        (np.abs(boxes1[:, None, 2] - boxes2[None, :, 3]) <= _EQUIVALENT_OBB_LINEAR_ATOL)
        & (np.abs(boxes1[:, None, 3] - boxes2[None, :, 2]) <= _EQUIVALENT_OBB_LINEAR_ATOL)
        & (_pi_periodic_distance(angle_delta - (np.pi / 2.0)) <= _EQUIVALENT_OBB_ANGLE_ATOL)
    )
    return center_equal & (direct | swapped)


def _obb_enclosing_bounds(boxes: np.ndarray) -> np.ndarray:
    """Return each OBB's enclosing ``(x1, y1, x2, y2)`` bounds."""
    if boxes.size == 0:
        return np.empty((0, 4), dtype=float)

    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    cos_a = np.abs(np.cos(boxes[:, 4]))
    sin_a = np.abs(np.sin(boxes[:, 4]))
    extent_x = half_w * cos_a + half_h * sin_a
    extent_y = half_w * sin_a + half_h * cos_a
    return np.column_stack(
        (
            boxes[:, 0] - extent_x,
            boxes[:, 1] - extent_y,
            boxes[:, 0] + extent_x,
            boxes[:, 1] + extent_y,
        )
    )


def _local_obb_rectangles(lhs: np.ndarray, rhs: np.ndarray) -> tuple[tuple, tuple]:
    """Return an OBB pair translated to the midpoint of their centers."""
    origin_x = lhs[0] + (rhs[0] - lhs[0]) / 2.0
    origin_y = lhs[1] + (rhs[1] - lhs[1]) / 2.0

    def local_rectangle(box: np.ndarray) -> tuple:
        return (
            (float(box[0] - origin_x), float(box[1] - origin_y)),
            (float(box[2]), float(box[3])),
            float(np.degrees(box[4])),
        )

    return local_rectangle(lhs), local_rectangle(rhs)


def _iou_obb_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Compute NxM rotated IoU matrix using vectorized AABB pre-filtering.

    Steps:
    1. Compute enclosing axis-aligned bounding boxes for all OBBs (vectorized).
    2. Compute AABB overlap mask to identify candidate pairs (vectorized).
    3. Only call cv.rotatedRectangleIntersection for overlapping candidates.

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
    bounds1 = _obb_enclosing_bounds(bboxes1)
    bounds2 = _obb_enclosing_bounds(bboxes2)
    ex1 = (bounds1[:, 2] - bounds1[:, 0]) / 2.0
    ey1 = (bounds1[:, 3] - bounds1[:, 1]) / 2.0
    ex2 = (bounds2[:, 2] - bounds2[:, 0]) / 2.0
    ey2 = (bounds2[:, 3] - bounds2[:, 1]) / 2.0

    # AABB bounds: (cx - ex, cy - ey, cx + ex, cy + ey)
    # Vectorized overlap check for all NxM pairs using broadcasting
    # Separating axis: no overlap if gap_x > 0 or gap_y > 0
    # gap_x = |cx1[i] - cx2[j]| - (ex1[i] + ex2[j])
    dx = np.abs(cx1[:, None] - cx2[None, :])  # (N, M)
    dy = np.abs(cy1[:, None] - cy2[None, :])  # (N, M)
    sum_ex = ex1[:, None] + ex2[None, :]  # (N, M)
    sum_ey = ey1[:, None] + ey2[None, :]  # (N, M)

    equivalent = _equivalent_obb_mask(bboxes1, bboxes2)

    # Candidate mask: AABBs overlap. Equivalent encodings have exact identity
    # semantics and bypass OpenCV's float32 intersection implementation.
    candidates = (dx < sum_ex) & (dy < sum_ey) & ~equivalent
    iou_matrix = np.zeros((N, M), dtype=np.float64)
    iou_matrix[equivalent] = 1.0

    # -- Compute rotated IoU only for candidate pairs --
    cand_i, cand_j = np.nonzero(candidates)
    if len(cand_i) == 0:
        return iou_matrix

    # Pre-compute areas. Rectangle centers are translated to a pair-local origin
    # below so OpenCV's float32 geometry remains accurate for very small boxes
    # at large image coordinates.
    areas1 = bboxes1[:, 2] * bboxes1[:, 3]
    areas2 = bboxes2[:, 2] * bboxes2[:, 3]

    for idx in range(len(cand_i)):
        i, j = int(cand_i[idx]), int(cand_j[idx])
        rect1, rect2 = _local_obb_rectangles(bboxes1[i], bboxes2[j])
        ret, intersect = cv.rotatedRectangleIntersection(rect1, rect2)
        if ret == 0 or intersect is None:
            continue
        inter_area = cv.contourArea(intersect)
        union = areas1[i] + areas2[j] - inter_area
        if union > 0:
            iou_matrix[i, j] = np.clip(inter_area / union, 0.0, 1.0)

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

        bounds1 = _obb_enclosing_bounds(boxes1)
        bounds2 = _obb_enclosing_bounds(boxes2)
        enclosing_width = np.maximum(bounds1[:, None, 2], bounds2[None, :, 2]) - np.minimum(
            bounds1[:, None, 0], bounds2[None, :, 0]
        )
        enclosing_height = np.maximum(bounds1[:, None, 3], bounds2[None, :, 3]) - np.minimum(
            bounds1[:, None, 1], bounds2[None, :, 1]
        )
        enclosing_diagonal = enclosing_width**2 + enclosing_height**2
        center_distance = (boxes1[:, None, 0] - boxes2[None, :, 0]) ** 2 + (
            boxes1[:, None, 1] - boxes2[None, :, 1]
        ) ** 2
        diou = iou - center_distance / np.maximum(enclosing_diagonal, 1e-12)
        similarity = (diou + 1.0) / 2.0
        return np.where(_equivalent_obb_mask(boxes1, boxes2), 1.0, similarity)

    @staticmethod
    def giou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        """Compute generalized IoU for oriented boxes.

        The enclosure is the convex hull of both rotated rectangles, which is
        the smallest convex region containing their geometry.
        """
        boxes1 = _obb_geometry(bboxes1)
        boxes2 = _obb_geometry(bboxes2)
        iou = _iou_obb_matrix(boxes1, boxes2)
        if boxes1.size == 0 or boxes2.size == 0:
            return iou

        areas1 = boxes1[:, 2] * boxes1[:, 3]
        areas2 = boxes2[:, 2] * boxes2[:, 3]
        area_sums = areas1[:, None] + areas2[None, :]
        union = area_sums / (1.0 + iou)
        equivalent = _equivalent_obb_mask(boxes1, boxes2)
        enclosing_area = union.copy()
        for row in range(len(boxes1)):
            for col in range(len(boxes2)):
                if equivalent[row, col]:
                    continue
                rect1, rect2 = _local_obb_rectangles(boxes1[row], boxes2[col])
                corners = np.vstack((cv.boxPoints(rect1), cv.boxPoints(rect2)))
                hull = cv.convexHull(corners)
                enclosing_area[row, col] = abs(cv.contourArea(hull))

        enclosing_area = np.maximum(enclosing_area, union)
        giou = iou - (enclosing_area - union) / np.maximum(enclosing_area, 1e-12)
        similarity = (giou + 1.0) / 2.0
        return np.where(equivalent, 1.0, similarity)

    @staticmethod
    def ciou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        """Compute complete IoU for oriented boxes.

        The aspect-ratio term uses ordered long and short sides so equivalent
        ``(w, h, theta)`` and ``(h, w, theta + pi/2)`` rows score identically.
        """
        boxes1 = _obb_geometry(bboxes1)
        boxes2 = _obb_geometry(bboxes2)
        iou = _iou_obb_matrix(boxes1, boxes2)
        if boxes1.size == 0 or boxes2.size == 0:
            return iou

        bounds1 = _obb_enclosing_bounds(boxes1)
        bounds2 = _obb_enclosing_bounds(boxes2)
        enclosing_width = np.maximum(bounds1[:, None, 2], bounds2[None, :, 2]) - np.minimum(
            bounds1[:, None, 0], bounds2[None, :, 0]
        )
        enclosing_height = np.maximum(bounds1[:, None, 3], bounds2[None, :, 3]) - np.minimum(
            bounds1[:, None, 1], bounds2[None, :, 1]
        )
        enclosing_diagonal = enclosing_width**2 + enclosing_height**2
        center_distance = (boxes1[:, None, 0] - boxes2[None, :, 0]) ** 2 + (
            boxes1[:, None, 1] - boxes2[None, :, 1]
        ) ** 2

        long1 = np.maximum(boxes1[:, 2], boxes1[:, 3])
        short1 = np.maximum(np.minimum(boxes1[:, 2], boxes1[:, 3]), 1e-7)
        long2 = np.maximum(boxes2[:, 2], boxes2[:, 3])
        short2 = np.maximum(np.minimum(boxes2[:, 2], boxes2[:, 3]), 1e-7)
        aspect1 = np.arctan(long1 / short1)
        aspect2 = np.arctan(long2 / short2)
        aspect_penalty = (4.0 / (np.pi**2)) * (aspect1[:, None] - aspect2[None, :]) ** 2
        alpha = aspect_penalty / np.maximum(1.0 - iou + aspect_penalty, 1e-7)

        ciou = iou - center_distance / np.maximum(enclosing_diagonal, 1e-7) - alpha * aspect_penalty
        similarity = (ciou + 1.0) / 2.0
        return np.where(_equivalent_obb_mask(boxes1, boxes2), 1.0, similarity)

    @staticmethod
    def hmiou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        """Compute height-modulated IoU from oriented geometry."""
        boxes1 = _obb_geometry(bboxes1)
        boxes2 = _obb_geometry(bboxes2)
        iou = _iou_obb_matrix(boxes1, boxes2)
        if boxes1.size == 0 or boxes2.size == 0:
            return iou

        bounds1 = _obb_enclosing_bounds(boxes1)
        bounds2 = _obb_enclosing_bounds(boxes2)
        overlap_height = np.maximum(
            0.0,
            np.minimum(bounds1[:, None, 3], bounds2[None, :, 3]) - np.maximum(bounds1[:, None, 1], bounds2[None, :, 1]),
        )
        enclosing_height = np.maximum(bounds1[:, None, 3], bounds2[None, :, 3]) - np.minimum(
            bounds1[:, None, 1], bounds2[None, :, 1]
        )
        similarity = iou * overlap_height / np.maximum(enclosing_height, 1e-10)
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
        return (ciou + 1) / 2.0

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
