"""Gather and scatter track-owned camera-motion state around NumPy kernels."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

import numpy as np

from boxmot.trackers.common.geometry.obb import transform_aabbs, transform_obbs, transform_points
from boxmot.trackers.common.motion.cmc.state import transform_aabb_kalman_states, transform_obb_kalman_states
from boxmot.trackers.common.motion.models import MotionModelAdapter


def transform_filter_histories(
    filters: Sequence,
    transform_states: Callable,
    transform_measurements: Callable,
) -> None:
    """Warp live, frozen and timed states together, retaining history shapes."""
    state_records = []
    measurement_records = []
    for kalman in filters:
        state_records.append((kalman, None, kalman.x, kalman.P))
        if kalman._prediction_origin is not None:
            state_records.append((kalman, "_prediction_origin", *kalman._prediction_origin))
        owners = [kalman]
        if not kalman.observed and kalman.attr_saved is not None:
            saved = kalman.attr_saved
            state_records.append((saved, None, saved["x"], saved["P"]))
            owners.append(saved)
        measurement_records.append((kalman, "_last_observed_measurement", kalman._last_observed_measurement))
        for owner in owners:
            for name in ("history_obs", "last_measurement"):
                value = owner[name] if isinstance(owner, dict) else getattr(owner, name)
                measurement_records.append((owner, name, value))

    if state_records:
        means, covariances = transform_states(
            np.asarray([np.asarray(record[2]).reshape(-1) for record in state_records]),
            np.asarray([record[3] for record in state_records]),
        )
        for (owner, name, original, _), mean, covariance in zip(state_records, means, covariances):
            mean = mean.reshape(np.asarray(original).shape)
            if name is not None:
                setattr(owner, name, (mean, covariance))
            elif isinstance(owner, dict):
                owner["x"], owner["P"] = mean, covariance
            else:
                owner.x, owner.P = mean, covariance

    measurements = []
    for _, _, value in measurement_records:
        items = value if isinstance(value, deque) else (value,)
        measurements.extend(np.asarray(item).reshape(-1) for item in items if item is not None)
    if not measurements:
        return
    warped = iter(transform_measurements(np.asarray(measurements)))
    for owner, name, value in measurement_records:
        if isinstance(value, deque):
            result = deque(
                (None if item is None else next(warped).reshape(np.asarray(item).shape) for item in value),
                maxlen=value.maxlen,
            )
        else:
            result = None if value is None else next(warped).reshape(np.asarray(value).shape)
        if isinstance(owner, dict):
            owner[name] = result
        else:
            setattr(owner, name, result)


def transform_observation_histories(tracks: Sequence, transform: np.ndarray, *, is_obb: bool) -> None:
    """Warp shared observation arrays once, preserving aliases and metadata."""
    observations = {}
    visits = {}
    for track in tracks:
        local = {}
        if track.last_observation[-1] >= 0:
            local[id(track.last_observation)] = track.last_observation
        for observation in track.observations.values():
            local[id(observation)] = observation
        observations.update(local)
        for identity in local:
            visits[identity] = visits.get(identity, 0) + 1
    if not observations:
        return
    arrays = list(observations.values())
    width = 5 if is_obb else 4
    rows = np.asarray([np.asarray(item).reshape(-1)[:width] for item in arrays])
    # A track may alias its own latest observation in the history dictionary.
    # External callers can also share arrays between tracks; retain the prior
    # per-track mutation count in that case, including casts to owner dtypes.
    counts = np.asarray(list(visits.values()))
    warp_boxes = transform_obbs if is_obb else transform_aabbs
    for visit in range(int(counts.max())):
        indices = np.flatnonzero(counts > visit)
        warped = warp_boxes(rows[indices], transform)
        for index, row in zip(indices, warped):
            arrays[index][:width] = row[:width]
            rows[index] = arrays[index][:width]


def transform_directions(
    tracks: Sequence,
    centers: np.ndarray,
    transform: np.ndarray,
    *,
    attributes: tuple[str, ...] = ("velocity",),
    step: float = 1.0,
    invalid_to_zero: bool = False,
    zero_input_to_zero: bool = False,
) -> None:
    """Warp cached ``[dy, dx]`` directions with each track's original policy."""
    owners, vectors, origins = [], [], []
    for track, center in zip(tracks, centers):
        for name in attributes:
            direction = getattr(track, name)
            if direction is None:
                continue
            owners.append((track, name))
            vectors.append(np.asarray(direction, dtype=float).reshape(2)[::-1])
            origins.append(center)
    if not owners:
        return
    vectors, origins = np.asarray(vectors), np.asarray(origins)
    finite_inputs = np.isfinite(vectors).all(axis=1)
    input_norms = np.linalg.norm(vectors, axis=1)
    active = finite_inputs & (input_norms > 1e-12) if zero_input_to_zero else np.ones(len(vectors), dtype=bool)
    transformed = vectors.copy()
    if active.any():
        points = np.stack((origins[active], origins[active] + step * vectors[active]), axis=1)
        mapped = transform_points(points, transform).reshape(-1, 2, 2)
        transformed[active] = (mapped[:, 1] - mapped[:, 0]) / step
    norms = np.linalg.norm(transformed, axis=1)
    valid = active & np.isfinite(transformed).all(axis=1) & np.isfinite(norms) & (norms > 1e-12)
    normalized = np.zeros_like(transformed)
    normalized[valid] = transformed[valid] / norms[valid, None]
    for index, (owner, name) in enumerate(owners):
        if (
            valid[index]
            or invalid_to_zero
            or (zero_input_to_zero and finite_inputs[index] and input_norms[index] <= 1e-12)
        ):
            setattr(owner, name, normalized[index, ::-1].copy())


def transform_ocsort_tracks(
    tracks: Sequence,
    transform: np.ndarray,
    *,
    model: MotionModelAdapter,
) -> None:
    """Batch observation-centric AABB/OBB state, history and direction CMC."""
    if not tracks:
        return
    is_obb = model.is_obb
    means = np.asarray([np.asarray(track.kf.x).reshape(-1) for track in tracks])
    centers = model.to_boxes(means)[:, :2]
    transform_observation_histories(tracks, transform, is_obb=is_obb)
    transform_directions(tracks, centers, transform, step=1e-3 if is_obb else 1.0, zero_input_to_zero=is_obb)
    state_transform = transform_obb_kalman_states if is_obb else transform_aabb_kalman_states
    box_transform = transform_obbs if is_obb else transform_aabbs
    transform_filter_histories(
        [track.kf for track in tracks],
        lambda means, covariances: state_transform(
            means,
            covariances,
            transform,
            measurement_to_box=model.to_boxes,
            box_to_measurement=model.to_measurements,
            velocity_measurement_indices=(0, 1, 2, 4) if is_obb else (0, 1, 2),
        ),
        lambda measurements: model.to_measurements(box_transform(model.to_boxes(measurements), transform)),
    )
