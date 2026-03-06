from collections import deque
from typing import Optional, Tuple

import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class BaseKalmanFilter:
    """
    Base class for Kalman filters tracking bounding boxes in image space.
    """

    def __init__(
        self,
        ndim: int,
        *,
        dim_x: Optional[int] = None,
        dim_z: Optional[int] = None,
        motion_mat: Optional[np.ndarray] = None,
        update_mat: Optional[np.ndarray] = None,
        max_obs: int = 50,
    ):
        self.ndim = ndim
        self.dim_z = dim_z if dim_z is not None else ndim
        self.dim_x = dim_x if dim_x is not None else 2 * self.ndim
        self.dt = 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = (
            motion_mat.astype(float).copy()
            if motion_mat is not None
            else self._default_motion_matrix(self.dim_x, self.dim_z)
        )
        self._update_mat = (
            update_mat.astype(float).copy()
            if update_mat is not None
            else np.eye(self.dim_z, self.dim_x)
        )
        self.F = self._motion_mat.copy()
        self.H = self._update_mat.copy()

        # Motion and observation uncertainty weights.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        # Stateful Kalman filter members used by matrix-based subclasses.
        self.x = np.zeros((self.dim_x, 1))
        self.P = np.eye(self.dim_x)
        self.Q = np.eye(self.dim_x)
        self.R = np.eye(self.dim_z)
        self.B = None
        self._alpha_sq = 1.0
        self.M = np.zeros((self.dim_x, self.dim_z))
        self.z = np.array([[None] * self.dim_z]).T

        self.K = np.zeros((self.dim_x, self.dim_z))
        self.y = np.zeros((self.dim_z, 1))
        self.S = np.zeros((self.dim_z, self.dim_z))
        self.SI = np.zeros((self.dim_z, self.dim_z))
        self._I = np.eye(self.dim_x)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.max_obs = max_obs
        self.history_obs = deque([], maxlen=self.max_obs)
        self.attr_saved = None
        self.observed = False
        self.last_measurement = None

    @staticmethod
    def _default_motion_matrix(dim_x: int, dim_z: int) -> np.ndarray:
        """Build a simple constant-velocity transition matrix."""
        motion_mat = np.eye(dim_x)
        velocity_dims = min(dim_z, max(0, dim_x - dim_z))
        for i in range(velocity_dims):
            motion_mat[i, dim_z + i] = 1.0
        return motion_mat

    def _resolve_matrix(self, matrix: Optional[np.ndarray], fallback: np.ndarray) -> np.ndarray:
        return matrix if matrix is not None else fallback

    @staticmethod
    def _reshape_measurement(z: np.ndarray, dim_z: int) -> np.ndarray:
        measurement = np.asarray(z, dtype=float)
        if measurement.ndim == 1:
            measurement = measurement.reshape((-1, 1))
        if measurement.shape != (dim_z, 1):
            measurement = measurement.reshape((dim_z, 1))
        return measurement

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = self._get_initial_covariance_std(measurement)
        covariance = np.diag(np.square(std))
        return mean, covariance

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        """
        Return initial standard deviations for the covariance matrix.
        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        """
        std_pos, std_vel = self._get_process_noise_std(mean)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return standard deviations for process noise.
        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_measurement_noise_std(
        self, mean: np.ndarray, confidence: float
    ) -> np.ndarray:
        """
        Return standard deviations for measurement noise.
        Should be implemented by stateless subclasses.
        """
        raise NotImplementedError

    def project(
        self, mean: np.ndarray, covariance: np.ndarray, confidence: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        """
        std = self._get_measurement_noise_std(mean, confidence)

        # NSA Kalman algorithm from GIAOTracker, which proposes a formula to
        # adaptively calculate the noise covariance Rek:
        # Rk = (1 − ck) Rk
        # where Rk is the preset constant measurement noise covariance
        # and ck is the detection confidence score at state k. Intuitively,
        # the detection has a higher score ck when it has less noise,
        # which results in a low Re.
        std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step (Vectorized version).
        """
        std_pos, std_vel = self._get_multi_process_noise_std(mean)
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def _get_multi_process_noise_std(
        self, mean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return standard deviations for process noise in vectorized form.
        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def predict_state(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stateful predict step for matrix-based filters.
        """
        B = self._resolve_matrix(B, self.B)
        F = self._resolve_matrix(F, self.F)
        Q = self._resolve_matrix(Q, self.Q)

        if np.isscalar(Q):
            Q = np.eye(self.dim_x) * float(Q)

        if B is not None and u is not None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)

        self.P = self._alpha_sq * np.dot(np.dot(F, self.P), F.T) + Q
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        return self.x, self.P

    def project_state(
        self,
        x: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project a state distribution into measurement space.
        """
        state = self.x if x is None else x
        covariance = self.P if P is None else P
        H = self._resolve_matrix(H, self.H)
        R = self._resolve_matrix(R, self.R)
        if np.isscalar(R):
            R = np.eye(self.dim_z) * float(R)

        projected_mean = np.dot(H, state)
        projected_covariance = np.dot(np.dot(H, covariance), H.T) + R
        return projected_mean, projected_covariance

    def update_state(
        self,
        z: np.ndarray,
        R: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stateful update step for matrix-based filters.
        """
        H = self._resolve_matrix(H, self.H)
        R = self._resolve_matrix(R, self.R)
        if np.isscalar(R):
            R = np.eye(self.dim_z) * float(R)

        measurement = self._reshape_measurement(z, self.dim_z)
        projected_mean, projected_cov = self.project_state(H=H, R=R)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        self.K = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(self.P, H.T).T,
            check_finite=False,
        ).T
        self.y = measurement - projected_mean
        self.S = projected_cov
        self.SI = scipy.linalg.cho_solve(
            (chol_factor, lower), np.eye(self.dim_z), check_finite=False
        )

        self.x = self.x + np.dot(self.K, self.y)
        self.P = self.P - np.linalg.multi_dot((self.K, projected_cov, self.K.T))
        self.z = measurement.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        return self.x, self.P

    def mahalanobis_distance(
        self,
        z: np.ndarray,
        H: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Mahalanobis distance for a candidate measurement.
        """
        measurement = self._reshape_measurement(z, self.dim_z)
        projected_mean, projected_cov = self.project_state(H=H, R=R)
        innovation = measurement - projected_mean
        chol_factor = np.linalg.cholesky(projected_cov)
        solved = scipy.linalg.solve_triangular(
            chol_factor,
            innovation,
            lower=True,
            check_finite=False,
        )
        return float(np.sqrt(np.dot(solved.T, solved)).item())

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.
        """
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")
