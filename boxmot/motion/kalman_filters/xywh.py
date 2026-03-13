from typing import Tuple

import numpy as np

from boxmot.motion.kalman_filters.base import BaseKalmanFilter


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

    @classmethod
    def _align_obb_measurement(
        cls, measurement: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """
        Resolve OBB representation ambiguity before update.

        A rectangle can be represented by equivalent parameterizations:
        - (w, h, theta)
        - (w, h, theta + pi)
        - (h, w, theta + pi/2)
        - (h, w, theta - pi/2)
        Pick the one closest to current state to avoid sudden angle flips.
        """
        aligned = np.asarray(measurement, dtype=float).copy().reshape((-1,))
        ref = np.asarray(reference, dtype=float).reshape((-1,))

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
        best_w, best_h, best_theta = cls._select_obb_candidate(
            reference_sizes=(ref_w, ref_h),
            reference_angle=ref_theta,
            candidates=candidates,
        )
        aligned[2] = best_w
        aligned[3] = best_h
        aligned[4] = best_theta
        return aligned

    @classmethod
    def _enforce_xywh_constraints(cls, mean: np.ndarray, is_obb: bool) -> np.ndarray:
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
            mean[:, 4] = self._wrap_angle(mean[:, 4])
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
                aligned = self._align_obb_measurement(
                    measurement_arr[:, 0], mean_arr[:, 0]
                )
                measurement_arr[:, 0] = aligned
            else:
                measurement_arr = measurement_arr.reshape((self.ndim,))
                measurement_arr = self._align_obb_measurement(measurement_arr, mean_arr)
            measurement = measurement_arr
        new_mean, new_covariance = super().update(mean, covariance, measurement, confidence)
        if self._is_obb:
            new_mean = self._damp_theta_velocity(new_mean, damping=0.8)
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

        projected_mean, projected_cov, measurements = self._prepare_gating_inputs(
            mean, covariance, measurements, self.project
        )
        for i in range(measurements.shape[0]):
            measurements[i, :] = self._align_obb_measurement(measurements[i, :], projected_mean)

        residuals = measurements - projected_mean
        return self._gating_from_residuals(residuals, projected_cov, metric)
