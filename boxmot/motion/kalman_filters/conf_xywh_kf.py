import numpy as np
from typing import Tuple
import scipy.linalg
from boxmot.motion.kalman_filters.base_kalman_filter import BaseKalmanFilter


class ConfKalmanFilterXYWH(BaseKalmanFilter):
    """
    A Kalman filter for tracking bounding boxes in image space with state space:
        x, y, w, h, vx, vy, vw, vh
    """

    def __init__(self, conf_thresh, cov_alpha):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self.conf_thresh = conf_thresh
        self.cov_alpha = cov_alpha

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        return [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]
        ]

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]
        ]
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        std_noise = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        return std_noise
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean

        # implement CW from ConfTrack paper
        if confidence < self.conf_thresh:
          conf_cost = 1 - confidence
          measurement = measurement + (projected_mean - measurement) * conf_cost

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
    
    def _get_multi_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3]
        ]
        return std_pos, std_vel
