"""EagerMOT's constant-velocity 3D Kalman model without FilterPy.

Adapted from EagerMOT (MIT License), Copyright (c) 2021 Aleksandr Kim.
See LICENSE in this directory.
"""

from __future__ import annotations

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
