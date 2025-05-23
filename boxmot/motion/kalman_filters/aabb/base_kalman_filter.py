from typing import Tuple

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
    9: 16.919
}

class BaseKalmanFilter:
    """
    Base class for Kalman filters tracking bounding boxes in image space.
    """

    def __init__(self, ndim: int):
        self.ndim = ndim
        self.dt = 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)  # State transition matrix
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = self.dt
        self._update_mat = np.eye(ndim, 2 * ndim)  # Observation matrix

        # Motion and observation uncertainty weights.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

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

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        """
        std_pos, std_vel = self._get_process_noise_std(mean)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return standard deviations for process noise.
        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
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
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def _get_multi_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return standard deviations for process noise in vectorized form.
        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray, only_position: bool = False, metric: str = 'maha') -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.
        """
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')