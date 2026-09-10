"""EagerMOT's constant-velocity 3D Kalman model without FilterPy.

Adapted from EagerMOT (MIT License), Copyright (c) 2021 Aleksandr Kim.
See LICENSE in this directory.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from boxmot.trackers.eagermot.geometry import yaw_difference


class Kalman3D:
    """Filter ``[x, y, z, yaw, l, w, h]`` bottom-center, y-down boxes.

    State appends ``[vx, vy, vz]``, and optionally yaw velocity when
    ``is_angular=True``. Prediction advances exactly one frame. Covariance,
    process noise, and measurement noise retain the released source values.
    """

    def __init__(self, box: np.ndarray, is_angular: bool = False) -> None:
        measurement = self._measurement(box)
        if not isinstance(is_angular, bool):
            raise TypeError("is_angular must be bool.")
        dimensions = 11 if is_angular else 10
        self.state = np.zeros(dimensions, dtype=np.float64)
        self.state[:7] = measurement
        self.covariance = np.diag([10.0] * 7 + [10000.0] * (dimensions - 7))
        self._transition = np.eye(dimensions)
        self._transition[:3, 7:10] = np.eye(3)
        if is_angular:
            self._transition[3, 10] = 1.0
        self._process_noise = np.diag([1.0] * 7 + [0.01] * (dimensions - 7))
        self._measurement_noise = np.eye(7) * 0.01

    @staticmethod
    def _measurement(box: np.ndarray) -> np.ndarray:
        """Validate and copy an observation so angle correction never changes input."""
        value = np.array(box, dtype=np.float64, copy=True)
        if value.shape != (7,) or not np.isfinite(value).all() or np.any(value[4:] <= 0):
            raise ValueError("Kalman3D requires a finite box7 with strictly positive l, w, h.")
        return value

    @property
    def box(self) -> np.ndarray:
        """Return the current seven box coordinates as an independent array."""
        return self.state[:7].copy()

    def predict(self) -> np.ndarray:
        """Advance one frame, including frames with no incoming 3D measurements."""
        self.state = self._transition @ self.state
        self.covariance = self._transition @ self.covariance @ self._transition.T + self._process_noise
        return self.box

    def update(self, box: np.ndarray) -> np.ndarray:
        """Correct a 3D observation after aligning its pi-equivalent yaw."""
        measurement = self._measurement(box)
        measurement[3] = self.state[3] + float(yaw_difference(self.state[3], measurement[3]))
        innovation_covariance = self.covariance[:7, :7] + self._measurement_noise
        gain = np.linalg.solve(innovation_covariance, self.covariance[:7, :]).T
        self.state += gain @ (measurement - self.state[:7])
        # FilterPy uses the Joseph form; retain it for stable symmetric covariance.
        residual = np.eye(len(self.state))
        residual[:, :7] -= gain
        self.covariance = residual @ self.covariance @ residual.T + gain @ self._measurement_noise @ gain.T
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        return self.box

    @staticmethod
    def _groups(filters: Sequence[Kalman3D]) -> list[list[int]]:
        """Group compatible state dimensions without sharing filter noise settings."""
        groups: dict[int, list[int]] = {}
        for index, model in enumerate(filters):
            groups.setdefault(len(model.state), []).append(index)
        return list(groups.values())

    @staticmethod
    def multi_predict(filters: Sequence[Kalman3D]) -> np.ndarray:
        """Advance independent filters using batched matrix operations.

        Angular and linear states may be mixed. Each filter retains its own
        transition, covariance, and process noise; the returned boxes follow
        input order and do not alias the updated states.
        """
        boxes = np.empty((len(filters), 7), dtype=np.float64)
        for indices in Kalman3D._groups(filters):
            if len(indices) == 1:
                index = indices[0]
                boxes[index] = filters[index].predict()
                continue
            models = [filters[index] for index in indices]
            states = np.stack([model.state for model in models])
            covariances = np.stack([model.covariance for model in models])
            transitions = np.stack([model._transition for model in models])
            process_noise = np.stack([model._process_noise for model in models])
            states = (transitions @ states[..., None])[..., 0]
            covariances = transitions @ covariances @ transitions.swapaxes(-1, -2) + process_noise
            boxes[indices] = states[:, :7]
            for model, state, covariance in zip(models, states, covariances):
                model.state = state
                model.covariance = covariance
        return boxes

    @staticmethod
    def multi_update(filters: Sequence[Kalman3D], boxes: np.ndarray) -> np.ndarray:
        """Correct matched filters in one batch per state size, retaining yaw alignment.

        Measurements must have shape ``(len(filters), 7)``. Validation precedes
        every mutation, and neither observations nor returned boxes alias the
        persistent filter states. Each track keeps its own measurement noise.
        """
        measurements = np.array(boxes, dtype=np.float64, copy=True)
        if (
            measurements.shape != (len(filters), 7)
            or not np.isfinite(measurements).all()
            or np.any(measurements[:, 4:] <= 0)
        ):
            raise ValueError("Kalman3D requires finite (N, 7) boxes with strictly positive l, w, h.")
        result = np.empty_like(measurements)
        for indices in Kalman3D._groups(filters):
            if len(indices) == 1:
                index = indices[0]
                result[index] = filters[index].update(measurements[index])
                continue
            models = [filters[index] for index in indices]
            states = np.stack([model.state for model in models])
            covariances = np.stack([model.covariance for model in models])
            noise = np.stack([model._measurement_noise for model in models])
            observations = measurements[indices]
            observations[:, 3] = states[:, 3] + yaw_difference(states[:, 3], observations[:, 3])
            projected = covariances[:, :7, :7] + noise
            gain = np.linalg.solve(projected, covariances[:, :7, :]).swapaxes(-1, -2)
            states += (gain @ (observations - states[:, :7])[..., None])[..., 0]
            residual = np.broadcast_to(np.eye(states.shape[1]), covariances.shape).copy()
            residual[:, :, :7] -= gain
            covariances = residual @ covariances @ residual.swapaxes(-1, -2) + gain @ noise @ gain.swapaxes(-1, -2)
            covariances = (covariances + covariances.swapaxes(-1, -2)) * 0.5
            result[indices] = states[:, :7]
            for model, state, covariance in zip(models, states, covariances):
                model.state = state
                model.covariance = covariance
        return result
