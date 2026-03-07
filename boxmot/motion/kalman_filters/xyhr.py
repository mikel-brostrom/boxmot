from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from boxmot.motion.kalman_filters.base import BaseKalmanFilter


class ConstantNoiseXYHR:
    """Constant process/measurement noise policy used by BoostTrack."""

    def __init__(self, dim_x: int, dim_z: int = 4):
        self.dim_x = dim_x
        self.dim_z = dim_z

    def get_init_state_cov(self) -> np.ndarray:
        covariance = np.eye(self.dim_x, dtype=float)
        covariance[4:, 4:] *= 1000.0
        covariance *= 10.0
        return covariance

    def get_r(self) -> np.ndarray:
        return np.diag([1.0, 1.0, 10.0, 0.01]).astype(float)

    def get_q(self) -> np.ndarray:
        process_noise = np.eye(self.dim_x, dtype=float)
        process_noise[4:, 4:] *= 0.01
        return process_noise


class KalmanFilterXYHR(BaseKalmanFilter):
    """
    Constant-noise Kalman filter for [x, y, h, r, vx, vy, vh, vr].

    This preserves BoostTrack's original state and noise model while exposing
    the filter under the shared `boxmot.motion.kalman_filters` namespace.
    """

    def __init__(
        self,
        z: Optional[np.ndarray] = None,
        ndim: int = 8,
        dt: float = 1.0,
        track_id: int = -1,
    ):
        if ndim < 8:
            raise ValueError("ndim must be >= 8 for XYHR state")

        dim_z = 4
        motion_mat = np.eye(ndim, dtype=float)
        velocity_dims = min(dim_z, max(0, ndim - dim_z))
        for i in range(velocity_dims):
            motion_mat[i, dim_z + i] = float(dt)
        update_mat = np.eye(dim_z, ndim, dtype=float)

        super().__init__(
            ndim=dim_z,
            dim_x=ndim,
            dim_z=dim_z,
            motion_mat=motion_mat,
            update_mat=update_mat,
        )

        self.dt = float(dt)
        self.id = track_id
        self.cov_update_policy = ConstantNoiseXYHR(ndim, dim_z)

        # Keep BoostTrack-compatible 1D state vector semantics.
        self.x = np.zeros((ndim,), dtype=float)
        self.covariance = self.cov_update_policy.get_init_state_cov()

        if z is not None:
            self.x[: self.dim_z] = self._reshape_measurement_vector(z)

    @property
    def covariance(self) -> np.ndarray:
        return self.P

    @covariance.setter
    def covariance(self, value: np.ndarray) -> None:
        self.P = np.asarray(value, dtype=float)

    def _reshape_measurement_vector(self, z: np.ndarray) -> np.ndarray:
        measurement = np.asarray(z, dtype=float)
        if measurement.ndim == 2:
            measurement = deepcopy(measurement.reshape((-1,)))
        else:
            measurement = measurement.reshape(-1)
        if measurement.size < self.dim_z:
            raise ValueError(
                f"measurement must have at least {self.dim_z} values, got {measurement.size}"
            )
        return measurement[: self.dim_z]

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        del measurement
        return np.sqrt(np.diag(self.cov_update_policy.get_init_state_cov()))

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        del mean
        q_diag = np.sqrt(np.diag(self.cov_update_policy.get_q()))
        velocity_dims = min(self.dim_z, max(0, self.dim_x - self.dim_z))
        std_pos = q_diag[: self.dim_z]
        std_vel = q_diag[self.dim_z : self.dim_z + velocity_dims]
        return std_pos, std_vel

    def _get_measurement_noise_std(
        self, mean: np.ndarray, confidence: float
    ) -> np.ndarray:
        del mean, confidence
        return np.sqrt(np.diag(self.cov_update_policy.get_r()))

    def _get_multi_process_noise_std(
        self, mean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if mean.ndim != 2:
            raise ValueError("Expected mean to have shape (n, dim_x)")
        n = mean.shape[0]
        std_pos, std_vel = self._get_process_noise_std(mean[0])
        std_pos_multi = [np.full(n, float(v), dtype=float) for v in std_pos]
        std_vel_multi = [np.full(n, float(v), dtype=float) for v in std_vel]
        return std_pos_multi, std_vel_multi

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((self.dim_x,), dtype=float)
        mean[: self.dim_z] = self._reshape_measurement_vector(measurement)
        covariance = self.cov_update_policy.get_init_state_cov()
        return mean, covariance

    def predict(
        self,
        mean: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        update = mean is None
        if mean is None:
            mean = self.x
            covariance = self.covariance
        mean = np.asarray(mean, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)

        motion_cov = self.cov_update_policy.get_q()
        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        if update:
            self.x = mean
            self.covariance = covariance
        return mean, covariance

    def project(
        self,
        mean: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if mean is None:
            mean = self.x
            covariance = self.covariance
        mean = np.asarray(mean, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)

        innovation_cov = self.cov_update_policy.get_r()
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return projected_mean, projected_cov + innovation_cov

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        measurement = self._reshape_measurement_vector(z)
        projected_mean, projected_cov = self.project()

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(self.covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        self.x = self.x + np.dot(innovation, kalman_gain.T)
        self.covariance = self.covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        return self.x, self.covariance
