from typing import Tuple

import numpy as np

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter


class KalmanFilterXYAH(BaseKalmanFilter):
    """
    A Kalman filter for tracking bounding boxes in image space with state space:
        x, y, a, h, vx, vy, va, vh
    """

    def __init__(self):
        super().__init__(ndim=4)

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        # initial uncertainty in the aspect ratio is very low,
        # suggesting that it is not expected to vary significantly.
        return [
            2 * self._std_weight_position * measurement[3],     # x
            2 * self._std_weight_position * measurement[3],     # y
            1e-2,                                               # a (aspect ratio)
            2 * self._std_weight_position * measurement[3],     # H
            10 * self._std_weight_velocity * measurement[3],    # vx
            10 * self._std_weight_velocity * measurement[3],    # vy
            1e-5,                                               # va (aspect ration vel)
            10 * self._std_weight_velocity * measurement[3]     # vh
        ]

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # very small process noise standard deviation assigned to the
        # aspect ratio state and its velocity. Suggests
        # that the aspect ratio is expected to remain relatively constant
        # over time.
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        # small measurement noise standard deviation for
        # aspect ratio state, indicating low expected measurement noise in
        # the aspect ratio.
        std_noise = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        return std_noise

    def _get_multi_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        return std_pos, std_vel
