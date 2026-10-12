"""Batched camera-motion propagation for box Kalman states."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from boxmot.trackers.common.geometry.obb import normalize_angle, transform_aabbs, transform_obbs


def is_orientation_preserving_similarity(transform: np.ndarray) -> bool:
    """Recognize affine rotation/translation/uniform scale without reflection.

    Similarity coefficients may differ by floating-point composition error,
    but shear, unequal scales, and projective terms must retain the generic
    camera transform. The tolerance scales only with the linear coefficients.
    """
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape not in ((2, 3), (3, 3)) or not np.isfinite(matrix).all():
        return False
    if matrix.shape == (3, 3) and not np.array_equal(matrix[2], [0.0, 0.0, 1.0]):
        return False
    linear = matrix[:2, :2]
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or determinant <= 0.0:
        return False
    tolerance = 32 * np.finfo(np.float64).eps * np.max(np.abs(linear))
    return bool(abs(linear[0, 0] - linear[1, 1]) <= tolerance and abs(linear[0, 1] + linear[1, 0]) <= tolerance)


def transform_similarity_kalman_states(
    means: np.ndarray,
    covariances: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_size: int,
    velocity_measurement_indices: Sequence[int],
    size_scale_exponent: int,
    return_jacobian: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a shape-preserving similarity to AABB area or height states.

    Camera motion rotates/scales centers and center velocities, and scales
    size and its velocity by the scale raised to ``size_scale_exponent``:
    one for XYHR height, two for XYSR/XYSCR area. Aspect ratio remains
    an object-shape measurement, rather than the ratio of a rotated enclosing
    AABB. Repeated camera rotations therefore do not inflate shape or couple
    shape uncertainties into each other. Camera zoom remains active, and
    optional confidence coordinates are unchanged.

    Full covariance, including cross terms, follows the same state Jacobian.
    Inputs are never mutated; means retain their ``(N, D)`` or ``(N, D, 1)``
    shape. Other affine and projective transforms require the generic path.
    """
    original_means = np.asarray(means, dtype=np.float64)
    if original_means.ndim == 3 and original_means.shape[-1] == 1:
        states = original_means[..., 0]
    elif original_means.ndim == 2:
        states = original_means
    else:
        raise ValueError(f"Expected Kalman means with shape (N, D) or (N, D, 1), got {original_means.shape}")
    if measurement_size not in (4, 5):
        raise ValueError("AABB shape states require four or five measurement coordinates")
    if size_scale_exponent not in (1, 2):
        raise ValueError("Shape scale exponent must be one for height or two for area")
    velocity_indices = tuple(int(index) for index in velocity_measurement_indices)
    if (
        len(set(velocity_indices)) != len(velocity_indices)
        or not {0, 1, 2}.issubset(velocity_indices)
        or any(index < 0 or index >= measurement_size for index in velocity_indices)
    ):
        raise ValueError("Shape states require distinct valid velocity coordinates including center and size")
    count, state_size = states.shape
    if state_size != measurement_size + len(velocity_indices):
        raise ValueError(f"State has {state_size} entries but {len(velocity_indices)} velocity entries were declared")
    covariance_arr = np.asarray(covariances, dtype=np.float64)
    if covariance_arr.shape != (count, state_size, state_size):
        raise ValueError(f"Expected covariance shape {(count, state_size, state_size)}, got {covariance_arr.shape}")
    matrix = np.asarray(transform, dtype=np.float64)
    if not is_orientation_preserving_similarity(matrix):
        raise ValueError("Shape-preserving camera correction requires an orientation-preserving affine similarity")
    identity = np.eye(3, dtype=np.float64)[: matrix.shape[0]]
    if count == 0 or np.array_equal(matrix, identity):
        result = (original_means.copy(), covariance_arr.copy())
        if return_jacobian:
            return (*result, np.broadcast_to(np.eye(state_size), (count, state_size, state_size)).copy())
        return result

    linear, translation = matrix[:2, :2], matrix[:2, 2]
    determinant = float(np.linalg.det(linear))
    size_scale = np.sqrt(determinant) if size_scale_exponent == 1 else determinant
    center_velocity = [measurement_size + velocity_indices.index(index) for index in (0, 1)]
    size_velocity = measurement_size + velocity_indices.index(2)
    jacobian = np.eye(state_size, dtype=np.float64)
    jacobian[:2, :2] = linear
    jacobian[np.ix_(center_velocity, center_velocity)] = linear
    jacobian[2, 2] = jacobian[size_velocity, size_velocity] = size_scale
    transformed = states.copy()
    transformed[:, :2] = states[:, :2] @ linear.T + translation
    transformed[:, center_velocity] = states[:, center_velocity] @ linear.T
    transformed[:, (2, size_velocity)] *= size_scale
    transformed_covariances = jacobian @ covariance_arr @ jacobian.T
    result = (transformed.reshape(original_means.shape), transformed_covariances)
    if return_jacobian:
        return (*result, np.broadcast_to(jacobian, (count, state_size, state_size)).copy())
    return result


def _transform_kalman_states(
    means: np.ndarray,
    covariances: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_to_box: Callable[[np.ndarray], np.ndarray],
    box_to_measurement: Callable[[np.ndarray], np.ndarray],
    velocity_measurement_indices: Sequence[int],
    is_obb: bool,
    measurement_size: int | None = None,
    positive_indices: Sequence[int] = (2, 3),
    preserved_measurement_indices: Sequence[int] = (),
    angle_measurement_index: int = 4,
    return_jacobian: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate rows with the same finite-difference scheme as scalar CMC.

    Covariance propagation holds the measurement Jacobian fixed when mapping
    velocities. Nonlinear warp curvature coupling position uncertainty into
    velocity uncertainty is outside this local linear approximation.
    """
    original_means = np.asarray(means, dtype=np.float64)
    if original_means.ndim == 3 and original_means.shape[-1] == 1:
        states = original_means[:, :, 0]
    elif original_means.ndim == 2:
        states = original_means
    else:
        raise ValueError(f"Expected Kalman means with shape (N, D) or (N, D, 1), got {original_means.shape}")
    count, state_size = states.shape
    measurement_size = (5 if is_obb else 4) if measurement_size is None else measurement_size
    if state_size < measurement_size:
        raise ValueError(f"Kalman states must contain at least {measurement_size} measurement values")
    covariance_arr = np.asarray(covariances, dtype=np.float64)
    if covariance_arr.shape != (count, state_size, state_size):
        raise ValueError(f"Expected covariance shape {(count, state_size, state_size)}, got {covariance_arr.shape}")
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape not in ((2, 3), (3, 3)):
        raise ValueError(f"Expected a 2x3 affine or 3x3 homography, got {matrix.shape}")
    identity = np.eye(3, dtype=np.float64)[: matrix.shape[0]]
    if count == 0 or np.array_equal(matrix, identity):
        if return_jacobian:
            jacobians = np.broadcast_to(np.eye(state_size), (count, state_size, state_size)).copy()
            return original_means.copy(), covariance_arr.copy(), jacobians
        return original_means.copy(), covariance_arr.copy()

    velocity_indices = np.asarray(tuple(int(index) for index in velocity_measurement_indices), dtype=np.intp)
    if state_size != measurement_size + len(velocity_indices):
        raise ValueError(f"State has {state_size} entries but {len(velocity_indices)} velocity entries were declared")
    if np.any(velocity_indices < 0) or np.any(velocity_indices >= measurement_size):
        raise ValueError(f"Velocity measurement indices must be in [0, {measurement_size - 1}]")

    measurements = states[:, :measurement_size]
    boxes = np.asarray(measurement_to_box(measurements), dtype=np.float64)[:, :measurement_size]
    base_boxes = transform_obbs(boxes, matrix) if is_obb else transform_aabbs(boxes, matrix)
    mapped = np.asarray(box_to_measurement(base_boxes), dtype=np.float64)[:, :measurement_size]
    if preserved_measurement_indices:
        mapped[:, preserved_measurement_indices] = measurements[:, preserved_measurement_indices]

    # Each track contributes both sides of every central difference. Flatten
    # these rows once so conversion, corner warps, and fitting share a batch.
    steps = 1e-4 * np.maximum(np.abs(measurements), 1.0)
    if is_obb:
        steps[:, angle_measurement_index] = 1e-3
    perturbed = np.broadcast_to(measurements[:, None, None, :], (count, 2, measurement_size, measurement_size)).copy()
    indices = np.arange(measurement_size)
    perturbed[:, 0, indices, indices] += steps
    perturbed[:, 1, indices, indices] -= steps
    size_indices = np.asarray(positive_indices)
    perturbed[:, 1, size_indices, size_indices] = np.maximum(perturbed[:, 1, size_indices, size_indices], 1e-6)
    actual_steps = perturbed[:, 0, indices, indices] - perturbed[:, 1, indices, indices]
    perturbed_boxes = np.asarray(measurement_to_box(perturbed.reshape(-1, measurement_size)), dtype=np.float64)[
        :, :measurement_size
    ]
    if is_obb:
        warped = transform_obbs(perturbed_boxes, matrix, reference=np.repeat(base_boxes, 2 * measurement_size, axis=0))
    else:
        warped = transform_aabbs(perturbed_boxes, matrix)
    mapped_perturbations = np.asarray(box_to_measurement(warped), dtype=np.float64)[:, :measurement_size].reshape(
        count, 2, measurement_size, measurement_size
    )
    if preserved_measurement_indices:
        mapped_perturbations[..., preserved_measurement_indices] = perturbed[..., preserved_measurement_indices]
    deltas = mapped_perturbations[:, 0] - mapped_perturbations[:, 1]
    if is_obb:
        deltas[:, :, angle_measurement_index] = normalize_angle(deltas[:, :, angle_measurement_index])
    jacobians = deltas.swapaxes(1, 2) / actual_steps[:, None, :]

    state_transforms = np.zeros((count, state_size, state_size), dtype=np.float64)
    state_transforms[:, :measurement_size, :measurement_size] = jacobians
    velocity_jacobians = jacobians[:, velocity_indices[:, None], velocity_indices]
    state_transforms[:, measurement_size:, measurement_size:] = velocity_jacobians
    transformed = states.copy()
    transformed[:, :measurement_size] = mapped
    transformed[:, measurement_size:] = (velocity_jacobians @ states[:, measurement_size:, None])[:, :, 0]
    transformed_covariances = state_transforms @ covariance_arr @ state_transforms.swapaxes(1, 2)
    transformed_covariances = 0.5 * (transformed_covariances + transformed_covariances.swapaxes(1, 2))
    if return_jacobian:
        return transformed.reshape(original_means.shape), transformed_covariances, state_transforms
    return transformed.reshape(original_means.shape), transformed_covariances


def transform_aabb_kalman_states(
    means: np.ndarray,
    covariances: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_to_box: Callable[[np.ndarray], np.ndarray],
    box_to_measurement: Callable[[np.ndarray], np.ndarray],
    velocity_measurement_indices: Sequence[int],
    measurement_size: int = 4,
    positive_indices: Sequence[int] = (2, 3),
    preserved_measurement_indices: Sequence[int] = (),
    return_jacobian: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform AABB states, velocities, and covariances across all tracks.

    Conversion callbacks accept and return measurement or box rows. Means
    retain their input ``(N, D)`` or ``(N, D, 1)`` shape; inputs are not mutated.
    """
    return _transform_kalman_states(
        means,
        covariances,
        transform,
        measurement_to_box=measurement_to_box,
        box_to_measurement=box_to_measurement,
        velocity_measurement_indices=velocity_measurement_indices,
        is_obb=False,
        measurement_size=measurement_size,
        positive_indices=positive_indices,
        preserved_measurement_indices=preserved_measurement_indices,
        return_jacobian=return_jacobian,
    )


def transform_obb_kalman_states(
    means: np.ndarray,
    covariances: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_to_box: Callable[[np.ndarray], np.ndarray],
    box_to_measurement: Callable[[np.ndarray], np.ndarray],
    velocity_measurement_indices: Sequence[int],
    measurement_size: int = 5,
    positive_indices: Sequence[int] = (2, 3),
    preserved_measurement_indices: Sequence[int] = (),
    angle_measurement_index: int = 4,
    return_jacobian: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform OBB states together, preserving angle and covariance semantics.

    Box rows use angle column four; measurement rows use the declared angle index.
    Projective and non-similarity affine fits retain OpenCV's rectangle fitter.
    """
    return _transform_kalman_states(
        means,
        covariances,
        transform,
        measurement_to_box=measurement_to_box,
        box_to_measurement=box_to_measurement,
        velocity_measurement_indices=velocity_measurement_indices,
        is_obb=True,
        measurement_size=measurement_size,
        positive_indices=positive_indices,
        preserved_measurement_indices=preserved_measurement_indices,
        angle_measurement_index=angle_measurement_index,
        return_jacobian=return_jacobian,
    )
