"""Batched camera-motion propagation for box Kalman states."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from boxmot.trackers.common.geometry.obb import normalize_angle, transform_aabbs, transform_obbs


def _transform_kalman_states(
    means: np.ndarray,
    covariances: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_to_box: Callable[[np.ndarray], np.ndarray],
    box_to_measurement: Callable[[np.ndarray], np.ndarray],
    velocity_measurement_indices: Sequence[int],
    is_obb: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate rows with the same finite-difference scheme as scalar CMC."""
    original_means = np.asarray(means, dtype=np.float64)
    if original_means.ndim == 3 and original_means.shape[-1] == 1:
        states = original_means[:, :, 0]
    elif original_means.ndim == 2:
        states = original_means
    else:
        raise ValueError(f"Expected Kalman means with shape (N, D) or (N, D, 1), got {original_means.shape}")
    count, state_size = states.shape
    measurement_size = 5 if is_obb else 4
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

    # Each track contributes both sides of every central difference. Flatten
    # these rows once so conversion, corner warps, and fitting share a batch.
    steps = 1e-4 * np.maximum(np.abs(measurements), 1.0)
    if is_obb:
        steps[:, 4] = 1e-3
    perturbed = np.broadcast_to(measurements[:, None, None, :], (count, 2, measurement_size, measurement_size)).copy()
    indices = np.arange(measurement_size)
    perturbed[:, 0, indices, indices] += steps
    perturbed[:, 1, indices, indices] -= steps
    size_indices = np.array([2, 3])
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
    deltas = mapped_perturbations[:, 0] - mapped_perturbations[:, 1]
    if is_obb:
        deltas[:, :, 4] = normalize_angle(deltas[:, :, 4])
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
    return transformed.reshape(original_means.shape), transformed_covariances


def transform_aabb_kalman_states(
    means: np.ndarray,
    covariances: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_to_box: Callable[[np.ndarray], np.ndarray],
    box_to_measurement: Callable[[np.ndarray], np.ndarray],
    velocity_measurement_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
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
    )


def transform_obb_kalman_states(
    means: np.ndarray,
    covariances: np.ndarray,
    transform: np.ndarray,
    *,
    measurement_to_box: Callable[[np.ndarray], np.ndarray],
    box_to_measurement: Callable[[np.ndarray], np.ndarray],
    velocity_measurement_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform OBB states together, preserving angle and covariance semantics.

    Conversion callbacks accept and return rows with the angle in column four.
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
    )
