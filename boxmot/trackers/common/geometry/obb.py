from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
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
    if arr.ndim != 3 or arr.shape[1:] != (4, 2):
        raise ValueError(f"Expected corners with shape (4, 2) or (N, 4, 2), got {arr.shape}")

    center = arr.mean(axis=1, keepdims=True)
    offsets = arr - center
    angles = np.arctan2(offsets[:, :, 1], offsets[:, :, 0])
    cyclic = np.take_along_axis(arr, np.argsort(angles, axis=1, kind="stable")[:, :, None], axis=1)

    # Image coordinates grow downward, so TL->TR->BR->BL has positive
    # shoelace area. Cyclic sorting preserves four distinct vertices even
    # when a square diamond has tied coordinate extrema.
    following = np.roll(cyclic, -1, axis=1)
    twice_area = (cyclic[:, None, :, 0] @ following[:, :, 1, None])[:, 0, 0] - (
        cyclic[:, None, :, 1] @ following[:, :, 0, None]
    )[:, 0, 0]
    cyclic = np.where((twice_area < 0)[:, None, None], cyclic[:, ::-1], cyclic)
    start = np.lexsort((cyclic[:, :, 0], cyclic[:, :, 1]), axis=1)[:, 0]
    indices = (start[:, None] + np.arange(4)) % 4
    ordered = np.take_along_axis(cyclic, indices[:, :, None], axis=1)
    return ordered[0] if single else ordered


def xywha_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, w, h, angle]`` OBB boxes to ordered corner rows."""
    arr = np.asarray(boxes, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 5)

    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"Expected boxes with shape (5,) or (N, 5), got {arr.shape}")
    sizes = np.maximum(arr[:, 2:4].astype(np.float64), 1e-4)
    rect = (sizes[:, None, :] * np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])).astype(np.float32)
    cosine = np.cos(arr[:, 4])
    sine = np.sin(arr[:, 4])
    rotations = np.stack([cosine, -sine, sine, cosine], axis=1).reshape(-1, 2, 2)
    corners = rect @ rotations.swapaxes(1, 2) + arr[:, None, :2]

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
    dtype = np.result_type(np.asarray(measurement).dtype, np.float32)
    aligned = np.asarray(measurement, dtype=dtype).copy().reshape(-1)
    ref = np.asarray(reference, dtype=dtype).reshape(-1)

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
    # Keep the representation nearest to the reference without wrapping it
    # back into a principal interval.  Canonical OBB angles are deliberately
    # unwrapped so a track can cross +/-pi without a discontinuity.
    aligned[4] = float(best[2])
    return aligned


def align_obb_measurements(measurements: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Align rows of equivalent OBB forms without changing their angle continuity."""
    values = np.asarray(measurements)
    dtype = np.result_type(values.dtype, np.float32)
    aligned = np.asarray(values, dtype=dtype).copy()
    refs = np.asarray(references, dtype=dtype)
    if aligned.ndim != 2 or aligned.shape[1] < 5 or refs.shape != aligned.shape:
        raise ValueError("OBB measurements and references must have matching (N, M >= 5) shapes")
    sizes = np.maximum(aligned[:, 2:4].astype(np.float64), 1e-6)
    reference_sizes = np.maximum(refs[:, 2:4].astype(np.float64), 1e-6)
    angles = aligned[:, 4].astype(np.float64)
    reference_angles = refs[:, 4].astype(np.float64)
    candidate_sizes = sizes[:, np.array([[0, 1], [0, 1], [1, 0], [1, 0]])]
    candidate_angles = angles[:, None] + np.array([0.0, np.pi, np.pi / 2.0, -np.pi / 2.0])
    candidate_angles = reference_angles[:, None] + normalize_angle(candidate_angles - reference_angles[:, None])
    angle_cost = np.abs(candidate_angles - reference_angles[:, None])
    size_cost = np.abs(np.log(candidate_sizes / reference_sizes[:, None, :])).sum(axis=2)
    best = np.argmin(angle_cost + 0.05 * size_cost, axis=1)
    rows = np.arange(len(aligned))
    aligned[:, 2:4] = candidate_sizes[rows, best]
    aligned[:, 4] = candidate_angles[rows, best]
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


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform image points with a 2x3 affine matrix or 3x3 homography."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 2x3 affine or 3x3 homography, got {matrix.shape}")

    homogeneous = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    warped = homogeneous @ matrix.T
    denominator = warped[:, 2]
    if np.any(np.abs(denominator) <= 1e-12):
        raise ValueError("CMC transform maps an OBB point to infinity")
    return warped[:, :2] / denominator[:, None]


def transform_aabb(box: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Warp all four AABB corners and return their enclosing ``xyxy`` box.

    Any trailing row metadata is copied unchanged. Transforming all corners is
    required for rotations, shear, and projective camera motion; transforming
    only the two diagonal points can invert or collapse the result.
    """
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ValueError(f"AABB geometry requires at least four values, got {values.size}.")
    x1, y1, x2, y2 = values[:4]
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    warped = transform_points(corners, transform)
    result = values.copy()
    result[:4] = (
        warped[:, 0].min(),
        warped[:, 1].min(),
        warped[:, 0].max(),
        warped[:, 1].max(),
    )
    return result


def transform_aabbs(boxes: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Warp all corners of AABB rows together, preserving trailing metadata."""
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"Expected AABB rows with shape (N, M >= 4), got {values.shape}")
    corners = values[:, np.array([[0, 1], [2, 1], [2, 3], [0, 3]])]
    warped = transform_points(corners, transform).reshape(-1, 4, 2)
    result = values.copy()
    result[:, :2] = warped.min(axis=1)
    result[:, 2:4] = warped.max(axis=1)
    return result


def transform_aabb_kalman_state(
    mean: np.ndarray,
    covariance: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_to_box: Callable[[np.ndarray], np.ndarray],
    box_to_measurement: Callable[[np.ndarray], np.ndarray],
    velocity_measurement_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an AABB Kalman state and covariance through camera motion.

    The first four state values are the motion model's measurement state. The
    remaining values are velocities corresponding to
    ``velocity_measurement_indices``. A numerical Jacobian carries nonlinear
    width, height, area, and aspect-ratio representations consistently through
    rotations, anisotropic scale, shear, and homographies.
    """
    original_mean = np.asarray(mean, dtype=np.float64)
    state = original_mean.reshape(-1).copy()
    covariance_arr = np.asarray(covariance, dtype=np.float64)
    if state.size < 4:
        raise ValueError("AABB Kalman state must contain at least four measurement values.")
    if covariance_arr.shape != (state.size, state.size):
        raise ValueError(f"Expected covariance shape {(state.size, state.size)}, got {covariance_arr.shape}.")

    transform_arr = np.asarray(transform, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    if transform_arr.shape == (2, 3):
        identity = identity[:2]
    if transform_arr.shape == identity.shape and np.array_equal(transform_arr, identity):
        return state.reshape(original_mean.shape), covariance_arr.copy()

    measurement = state[:4].copy()

    def map_measurement(values: np.ndarray) -> np.ndarray:
        source_box = np.asarray(measurement_to_box(values), dtype=np.float64).reshape(-1)[:4]
        warped_box = transform_aabb(source_box, transform_arr)[:4]
        return np.asarray(box_to_measurement(warped_box), dtype=np.float64).reshape(-1)[:4]

    mapped = map_measurement(measurement)
    jacobian = np.empty((4, 4), dtype=np.float64)
    for index in range(4):
        step = 1e-4 * max(abs(float(measurement[index])), 1.0)
        plus = measurement.copy()
        minus = measurement.copy()
        plus[index] += step
        minus[index] -= step
        if index in (2, 3):
            minus[index] = max(minus[index], 1e-6)
        actual_step = plus[index] - minus[index]
        jacobian[:, index] = (map_measurement(plus) - map_measurement(minus)) / actual_step

    velocity_indices = tuple(int(index) for index in velocity_measurement_indices)
    if state.size != 4 + len(velocity_indices):
        raise ValueError(
            f"State has {state.size} entries but {len(velocity_indices)} AABB velocity entries were declared."
        )
    if any(index < 0 or index >= 4 for index in velocity_indices):
        raise ValueError(f"AABB velocity measurement indices must be in [0, 3], got {velocity_indices}.")

    state_transform = np.zeros((state.size, state.size), dtype=np.float64)
    state_transform[:4, :4] = jacobian
    velocity_jacobian = jacobian[np.ix_(velocity_indices, velocity_indices)]
    state_transform[4:, 4:] = velocity_jacobian

    transformed = state.copy()
    transformed[:4] = mapped
    transformed[4:] = velocity_jacobian @ state[4:]
    transformed_covariance = state_transform @ covariance_arr @ state_transform.T
    transformed_covariance = 0.5 * (transformed_covariance + transformed_covariance.T)
    return transformed.reshape(original_mean.shape), transformed_covariance


def _local_transform_jacobian(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Return the local 2D Jacobian of an affine transform or homography."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (2, 3):
        return matrix[:, :2]
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 2x3 affine or 3x3 homography, got {matrix.shape}")

    x, y = np.asarray(point, dtype=np.float64).reshape(2)
    den = (matrix[2, 0] * x) + (matrix[2, 1] * y) + matrix[2, 2]
    if abs(den) <= 1e-12:
        raise ValueError("CMC homography maps the OBB centre to infinity")
    num_x = (matrix[0, 0] * x) + (matrix[0, 1] * y) + matrix[0, 2]
    num_y = (matrix[1, 0] * x) + (matrix[1, 1] * y) + matrix[1, 2]
    den_sq = den * den
    return np.array(
        [
            [
                ((matrix[0, 0] * den) - (num_x * matrix[2, 0])) / den_sq,
                ((matrix[0, 1] * den) - (num_x * matrix[2, 1])) / den_sq,
            ],
            [
                ((matrix[1, 0] * den) - (num_y * matrix[2, 0])) / den_sq,
                ((matrix[1, 1] * den) - (num_y * matrix[2, 1])) / den_sq,
            ],
        ],
        dtype=np.float64,
    )


def _rotation_from_linear(linear: np.ndarray) -> float:
    """Extract the proper-rotation component of a local 2D linear transform."""
    u, _, vh = np.linalg.svd(np.asarray(linear, dtype=np.float64).reshape(2, 2))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def transform_obb(
    box: np.ndarray,
    transform: np.ndarray,
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Warp an OBB through camera motion and refit its oriented rectangle.

    Transforming all four corners preserves camera rotation, anisotropic scale,
    shear, and projective motion. The fitted rectangle is aligned to the
    expected orientation so equivalent width/height representations do not
    introduce 90-degree state jumps.
    """
    source = np.asarray(box, dtype=np.float64).reshape(-1)[:5]
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (2, 3):
        affine = matrix
    elif matrix.shape == (3, 3) and abs(matrix[2, 2]) > 1e-12:
        normalized = matrix / matrix[2, 2]
        affine = normalized[:2] if np.allclose(normalized[2], [0.0, 0.0, 1.0], atol=1e-12) else None
    else:
        affine = None

    if affine is not None:
        linear = affine[:, :2]
        scale_sq = float(np.trace(linear.T @ linear) / 2.0)
        is_similarity = (
            scale_sq > 0.0
            and np.linalg.det(linear) > 0.0
            and np.allclose(
                linear.T @ linear,
                scale_sq * np.eye(2),
                rtol=1e-7,
                atol=1e-10,
            )
        )
        if is_similarity:
            warped = source.copy()
            warped[:2] = linear @ source[:2] + affine[:, 2]
            warped[2:4] *= np.sqrt(scale_sq)
            warped[4] = normalize_angle(source[4] + np.arctan2(linear[1, 0], linear[0, 0]))
            if reference is not None:
                warped = align_obb_measurement(warped, reference)
            return warped.astype(np.float64)

    corners = xywha_to_corners(source).reshape(4, 2)
    warped_corners = transform_points(corners, transform).astype(np.float32)
    rect = cv2.minAreaRect(warped_corners)
    (cx, cy), (width, height), angle_deg = rect
    warped = np.array(
        [cx, cy, max(width, 1e-4), max(height, 1e-4), np.deg2rad(angle_deg)],
        dtype=np.float64,
    )

    if reference is None:
        local = _local_transform_jacobian(transform, source[:2])
        expected = source.copy()
        expected[:2] = transform_points(source[:2].reshape(1, 2), transform)[0]
        expected[4] = normalize_angle(source[4] + _rotation_from_linear(local))
        reference = expected
    return align_obb_measurement(warped, reference).astype(np.float64)


def transform_obbs(
    boxes: np.ndarray,
    transform: np.ndarray,
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Warp OBB rows together and retain the scalar rectangle-fitting contract.

    Similarities use batched centre, scale, and angle updates. Other transforms
    batch their corner warps, local Jacobians, and representation alignment;
    OpenCV's single-rectangle fitting API remains one call per box.
    """
    source = np.asarray(boxes, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 5:
        raise ValueError(f"Expected OBB rows with shape (N, 5), got {source.shape}")
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape not in ((2, 3), (3, 3)):
        raise ValueError(f"Expected a 2x3 affine or 3x3 homography, got {matrix.shape}")
    refs = None if reference is None else np.asarray(reference, dtype=np.float64)
    if refs is not None and refs.shape != source.shape:
        raise ValueError(f"Expected OBB reference shape {source.shape}, got {refs.shape}")
    if len(source) == 0:
        return source.copy()

    affine = None
    if matrix.shape == (2, 3):
        affine = matrix
    elif abs(matrix[2, 2]) > 1e-12:
        normalized = matrix / matrix[2, 2]
        if np.allclose(normalized[2], [0.0, 0.0, 1.0], atol=1e-12):
            affine = normalized[:2]
    if affine is not None:
        linear = affine[:, :2]
        scale_sq = float(np.trace(linear.T @ linear) / 2.0)
        if (
            scale_sq > 0.0
            and np.linalg.det(linear) > 0.0
            and np.allclose(linear.T @ linear, scale_sq * np.eye(2), rtol=1e-7, atol=1e-10)
        ):
            warped = source.copy()
            warped[:, :2] = (linear @ source[:, :2, None])[:, :, 0] + affine[:, 2]
            warped[:, 2:4] *= np.sqrt(scale_sq)
            warped[:, 4] = normalize_angle(source[:, 4] + np.arctan2(linear[1, 0], linear[0, 0]))
            return warped if refs is None else align_obb_measurements(warped, refs)

    corners = xywha_to_corners(source).reshape(-1, 4, 2)
    warped_corners = transform_points(corners, matrix).astype(np.float32).reshape(-1, 4, 2)
    # minAreaRect exposes no batch API. Keep its exact fitting behavior for
    # non-similarity warps; all surrounding numerical work is vectorized.
    rectangles = [cv2.minAreaRect(points) for points in warped_corners]
    centers = np.asarray([rectangle[0] for rectangle in rectangles], dtype=np.float64)
    sizes = np.maximum(np.asarray([rectangle[1] for rectangle in rectangles], dtype=np.float64), 1e-4)
    angles = np.deg2rad(np.asarray([rectangle[2] for rectangle in rectangles], dtype=np.float64))
    warped = np.column_stack([centers, sizes, angles])

    if refs is None:
        if matrix.shape == (2, 3):
            rotation = _rotation_from_linear(matrix[:, :2])
        else:
            x, y = source[:, 0], source[:, 1]
            den = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
            if np.any(np.abs(den) <= 1e-12):
                raise ValueError("CMC homography maps the OBB centre to infinity")
            numerators = np.column_stack(
                [matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2], matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]]
            )
            jacobians = (matrix[None, :2, :2] * den[:, None, None] - numerators[:, :, None] * matrix[None, 2, :2]) / (
                den * den
            )[:, None, None]
            u, _, vh = np.linalg.svd(jacobians)
            rotations = u @ vh
            reflected = np.linalg.det(rotations) < 0
            u[reflected, :, -1] *= -1.0
            rotations[reflected] = u[reflected] @ vh[reflected]
            rotation = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
        refs = source.copy()
        refs[:, :2] = transform_points(source[:, :2], matrix)
        refs[:, 4] = normalize_angle(source[:, 4] + rotation)
    return align_obb_measurements(warped, refs)


def transform_obb_kalman_state(
    mean: np.ndarray,
    covariance: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_to_box: Callable[[np.ndarray], np.ndarray],
    box_to_measurement: Callable[[np.ndarray], np.ndarray],
    velocity_measurement_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an OBB Kalman mean and covariance into the current camera frame.

    The first five state entries are the tracker's OBB measurement state. Any
    following velocity entries correspond to ``velocity_measurement_indices``.
    A numerical Jacobian of the corner-warp/refit operation carries position,
    size, ratio, angle, velocities, cross-covariances, and uncertainty through
    affine or projective CMC without assuming isotropic scale.
    """
    original_mean = np.asarray(mean, dtype=np.float64)
    column_state = original_mean.ndim == 2
    state = original_mean.reshape(-1).copy()
    covariance_arr = np.asarray(covariance, dtype=np.float64)
    if state.size < 5:
        raise ValueError("OBB Kalman state must contain at least five measurement values")
    if covariance_arr.shape != (state.size, state.size):
        raise ValueError(f"Expected covariance shape {(state.size, state.size)}, got {covariance_arr.shape}")

    transform_arr = np.asarray(transform, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    if transform_arr.shape == (2, 3):
        identity = identity[:2]
    if transform_arr.shape == identity.shape and np.array_equal(transform_arr, identity):
        transformed = state.reshape((-1, 1)) if column_state else state
        return transformed, covariance_arr.copy()

    measurement = state[:5].copy()

    def map_measurement(values: np.ndarray, reference_box: np.ndarray | None = None) -> np.ndarray:
        source_box = np.asarray(measurement_to_box(values), dtype=np.float64).reshape(-1)[:5]
        warped_box = transform_obb(source_box, transform, reference=reference_box)
        return np.asarray(box_to_measurement(warped_box), dtype=np.float64).reshape(-1)[:5]

    base_box = transform_obb(
        np.asarray(measurement_to_box(measurement), dtype=np.float64).reshape(-1)[:5],
        transform,
    )
    mapped = np.asarray(box_to_measurement(base_box), dtype=np.float64).reshape(-1)[:5]
    jacobian = np.empty((5, 5), dtype=np.float64)
    for index in range(5):
        step = 1e-3 if index == 4 else 1e-4 * max(abs(float(measurement[index])), 1.0)
        plus = measurement.copy()
        minus = measurement.copy()
        plus[index] += step
        minus[index] -= step
        if index in (2, 3):
            minus[index] = max(minus[index], 1e-6)
        actual_step = plus[index] - minus[index]
        mapped_plus = map_measurement(plus, reference_box=base_box)
        mapped_minus = map_measurement(minus, reference_box=base_box)
        delta = mapped_plus - mapped_minus
        delta[4] = normalize_angle(delta[4])
        jacobian[:, index] = delta / actual_step

    velocity_indices = tuple(int(index) for index in velocity_measurement_indices)
    velocity_count = len(velocity_indices)
    if state.size != 5 + velocity_count:
        raise ValueError(f"State has {state.size} entries but {velocity_count} OBB velocity entries were declared")

    state_transform = np.zeros((state.size, state.size), dtype=np.float64)
    state_transform[:5, :5] = jacobian
    velocity_jacobian = jacobian[np.ix_(velocity_indices, velocity_indices)]
    state_transform[5:, 5:] = velocity_jacobian

    transformed = state.copy()
    transformed[:5] = mapped
    transformed[5:] = velocity_jacobian @ state[5:]
    transformed_covariance = state_transform @ covariance_arr @ state_transform.T
    transformed_covariance = 0.5 * (transformed_covariance + transformed_covariance.T)
    if column_state:
        transformed = transformed.reshape((-1, 1))
    return transformed, transformed_covariance
