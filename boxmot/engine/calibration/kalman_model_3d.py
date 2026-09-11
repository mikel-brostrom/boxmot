"""Unscaled EagerMOT covariance bases for supervised 3D calibration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from boxmot.engine.calibration.kalman_model import validate_calibration_options
from boxmot.trackers.common.motion.kalman_filters.noise import normalize_kalman_options
from boxmot.trackers.eagermot.geometry import yaw_difference
from boxmot.trackers.eagermot.motion import Kalman3D


class CalibrationModel3D:
    """Expose the runtime box7 state and its original, unscaled P0, R, F, Q.

    Geometry is bottom-center ``x, y, z, yaw, length, width, height`` in the
    runtime world frame. Yaw follows the filter's pi-equivalent convention.
    Prediction advances one delivered frame, including for angular velocity.
    """

    def __init__(self, options: Mapping[str, Any]) -> None:
        validate_calibration_options("eagermot", options)
        normalize_kalman_options(options, variable_dt=options.get("variable_dt", False), tracker_name="eagermot")
        self._filter = Kalman3D(
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), is_angular=options.get("is_angular", False)
        )
        self.dim_z = 7
        self.dim_x = len(self._filter.state)
        self.measurement_indices = tuple(range(self.dim_z))
        self.velocity_indices = tuple(range(self.dim_z, self.dim_x))
        self.velocity_measurement_indices = (0, 1, 2, 3) if self.dim_x == 11 else (0, 1, 2)

    def to_measurement(self, box: np.ndarray, score: float = 1.0, reference: np.ndarray | None = None) -> np.ndarray:
        """Validate box7 geometry and align yaw exactly as a runtime update."""
        measurement = Kalman3D._measurement(box)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detection confidence must be finite and between zero and one.")
        if reference is not None:
            reference = Kalman3D._measurement(reference)
            measurement[3] = reference[3] + float(yaw_difference(reference[3], measurement[3]))
        return measurement

    def initial_state(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return observed geometry, zero velocity, and the runtime birth prior."""
        mean = np.zeros(self.dim_x)
        mean[: self.dim_z] = self.to_measurement(measurement)
        return mean, self._filter.covariance.copy()

    def measurement_covariance(self, measurement: np.ndarray, score: float = 1.0) -> np.ndarray:
        """Return the source constant R, independent of detector confidence."""
        self.to_measurement(measurement, score=score)
        return self._filter._measurement_noise.copy()

    def transition(self, dt: float | None = None) -> np.ndarray:
        """Return the fixed-frame runtime transition matrix."""
        if dt is not None:
            raise ValueError("EagerMOT 3D KF calibration uses fixed frame steps.")
        return self._filter._transition.copy()

    def process_covariance_bases(self, mean: np.ndarray, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Separate the full measured-state and velocity blocks of runtime Q."""
        if dt is not None:
            raise ValueError("EagerMOT 3D KF calibration uses fixed frame steps.")
        mean = np.asarray(mean)
        if mean.shape != (self.dim_x,) or not np.all(np.isfinite(mean)):
            raise ValueError(f"Expected {self.dim_x} finite Kalman state values.")
        position, velocity = (np.zeros((self.dim_x, self.dim_x)) for _ in range(2))
        position[: self.dim_z, : self.dim_z] = self._filter._process_noise[: self.dim_z, : self.dim_z]
        velocity[self.dim_z :, self.dim_z :] = self._filter._process_noise[self.dim_z :, self.dim_z :]
        return position, velocity
