from typing import Tuple

import numpy as np

from boxmot.motion.kalman_filters.base import BaseKalmanFilter


class KalmanFilterXYAH(BaseKalmanFilter):
    """
    Kalman filter for XYAH state with optional OBB angle extension.

    - `ndim=4`: [x, y, a, h]
    - `ndim=5`: [x, y, a, h, theta]
    """

    def __init__(self, ndim: int = 4):
        if ndim not in (4, 5):
            raise ValueError("ndim must be 4 (AABB) or 5 (OBB)")
        super().__init__(ndim=ndim)
        self._is_obb = ndim == 5

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        # low uncertainty for aspect ratio and its velocity to keep ratio stable.
        std = [
            2 * self._std_weight_position * measurement[3],     # x
            2 * self._std_weight_position * measurement[3],     # y
            1e-2,                                               # a
            2 * self._std_weight_position * measurement[3],     # h
            10 * self._std_weight_velocity * measurement[3],    # vx
            10 * self._std_weight_velocity * measurement[3],    # vy
            1e-5,                                               # va
            10 * self._std_weight_velocity * measurement[3],    # vh
        ]
        if self._is_obb:
            std.insert(4, 1e-2)  # theta
            std.append(1e-5)     # v_theta
        return std

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        if self._is_obb:
            std_pos.append(1e-2)
            std_vel.append(1e-5)
        return std_pos, std_vel

    def _get_measurement_noise_std(
        self, mean: np.ndarray, confidence: float
    ) -> np.ndarray:
        std_noise = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        if self._is_obb:
            std_noise.append(1e-1)
        return std_noise

    def _get_multi_process_noise_std(
        self, mean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        if self._is_obb:
            std_pos.append(1e-2 * np.ones_like(mean[:, 3]))
            std_vel.append(1e-5 * np.ones_like(mean[:, 3]))
        return std_pos, std_vel

    @classmethod
    def _enforce_xyah_constraints(cls, mean: np.ndarray, is_obb: bool) -> np.ndarray:
        return cls._enforce_state_geometry(
            mean,
            positive_indices=(2, 3),
            angle_index=4 if is_obb else None,
            min_size=1e-4,
        )

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=float).copy()
        if self._is_obb:
            measurement[4] = self._wrap_angle(measurement[4])
        mean, covariance = super().initiate(measurement)
        mean = self._enforce_xyah_constraints(mean, self._is_obb)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, covariance = super().predict(mean, covariance)
        mean = self._enforce_xyah_constraints(mean, self._is_obb)
        return mean, covariance

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, covariance = super().multi_predict(mean, covariance)
        mean[:, 2] = np.maximum(mean[:, 2], 1e-4)
        mean[:, 3] = np.maximum(mean[:, 3], 1e-4)
        if self._is_obb:
            mean[:, 4] = self._wrap_angle(mean[:, 4])
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._is_obb:
            mean_arr = np.asarray(mean, dtype=float)
            measurement_arr = np.asarray(measurement, dtype=float).copy()
            if mean_arr.ndim == 2:
                measurement_arr = measurement_arr.reshape((self.ndim, 1))
                reference_theta = float(mean_arr[4, 0])
                measurement_arr[4, 0] = self._align_angle_to_reference(
                    measurement_arr[4, 0], reference_theta
                )
            else:
                measurement_arr = measurement_arr.reshape((self.ndim,))
                reference_theta = float(mean_arr[4])
                measurement_arr[4] = self._align_angle_to_reference(
                    measurement_arr[4], reference_theta
                )
            measurement = measurement_arr
        new_mean, new_covariance = super().update(mean, covariance, measurement, confidence)
        new_mean = self._enforce_xyah_constraints(new_mean, self._is_obb)
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        if not self._is_obb or only_position:
            return super().gating_distance(mean, covariance, measurements, only_position, metric)

        projected_mean, projected_cov, measurements = self._prepare_gating_inputs(
            mean, covariance, measurements, self.project
        )
        measurements[:, 4] = np.array(
            [
                self._align_angle_to_reference(angle, projected_mean[4])
                for angle in measurements[:, 4]
            ],
            dtype=float,
        )

        residuals = measurements - projected_mean
        return self._gating_from_residuals(residuals, projected_cov, metric)
