from typing import Tuple

import numpy as np
import scipy.linalg

from boxmot.motion.kalman_filters.base import BaseKalmanFilter


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class KalmanFilterXYWH(BaseKalmanFilter):
    """
    Kalman filter for XYWH state with optional OBB angle extension.

    - `ndim=4`: [x, y, w, h]
    - `ndim=5`: [x, y, w, h, theta]
    """

    def __init__(self, ndim: int = 4):
        if ndim not in (4, 5):
            raise ValueError("ndim must be 4 (AABB) or 5 (OBB)")
        super().__init__(ndim=ndim)
        self._is_obb = ndim == 5

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        if self._is_obb:
            std.insert(4, 1e-2)  # theta
            std.append(1e-5)     # v_theta
        return std

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        if self._is_obb:
            std_pos.append(1e-2)
            std_vel.append(1e-5)
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        std_noise = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        if self._is_obb:
            std_noise.append(1e-1)
        return std_noise

    def _get_multi_process_noise_std(
        self, mean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        if self._is_obb:
            std_pos.append(1e-2 * np.ones_like(mean[:, 2]))
            std_vel.append(1e-5 * np.ones_like(mean[:, 2]))
        return std_pos, std_vel

    @staticmethod
    def _enforce_xywh_constraints(mean: np.ndarray, is_obb: bool) -> np.ndarray:
        if mean.ndim == 1:
            mean[2] = max(mean[2], 1e-4)
            mean[3] = max(mean[3], 1e-4)
            if is_obb:
                mean[4] = _wrap_angle(mean[4])
            return mean

        mean[2, :] = np.maximum(mean[2, :], 1e-4)
        mean[3, :] = np.maximum(mean[3, :], 1e-4)
        if is_obb:
            mean[4, :] = _wrap_angle(mean[4, :])
        return mean

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=float).copy()
        if self._is_obb:
            measurement[4] = _wrap_angle(measurement[4])
        mean, covariance = super().initiate(measurement)
        mean = self._enforce_xywh_constraints(mean, self._is_obb)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, covariance = super().predict(mean, covariance)
        mean = self._enforce_xywh_constraints(mean, self._is_obb)
        return mean, covariance

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, covariance = super().multi_predict(mean, covariance)
        if self._is_obb:
            mean[:, 2] = np.maximum(mean[:, 2], 1e-4)
            mean[:, 3] = np.maximum(mean[:, 3], 1e-4)
            mean[:, 4] = _wrap_angle(mean[:, 4])
        else:
            mean[:, 2] = np.maximum(mean[:, 2], 1e-4)
            mean[:, 3] = np.maximum(mean[:, 3], 1e-4)
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
                measurement_arr[4, 0] = reference_theta + _wrap_angle(
                    measurement_arr[4, 0] - reference_theta
                )
            else:
                measurement_arr = measurement_arr.reshape((self.ndim,))
                reference_theta = float(mean_arr[4])
                measurement_arr[4] = reference_theta + _wrap_angle(
                    measurement_arr[4] - reference_theta
                )
            measurement = measurement_arr
        new_mean, new_covariance = super().update(mean, covariance, measurement, confidence)
        new_mean = self._enforce_xywh_constraints(new_mean, self._is_obb)
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

        projected_mean, projected_cov = self.project(mean, covariance)
        projected_mean = np.asarray(projected_mean, dtype=float).reshape(-1)
        measurements = np.asarray(measurements, dtype=float).copy()
        if measurements.ndim == 1:
            measurements = measurements.reshape(1, -1)
        measurements[:, 4] = projected_mean[4] + _wrap_angle(
            measurements[:, 4] - projected_mean[4]
        )

        d = measurements - projected_mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        if metric == "maha":
            cholesky_factor = np.linalg.cholesky(projected_cov)
            z = scipy.linalg.solve_triangular(
                cholesky_factor,
                d.T,
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )
            return np.sum(z * z, axis=0)
        raise ValueError("invalid distance metric")
