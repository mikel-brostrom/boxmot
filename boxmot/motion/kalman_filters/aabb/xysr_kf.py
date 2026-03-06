from collections import deque
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter


class KalmanFilterXYSR(BaseKalmanFilter):
    """
    Linear Kalman filter for [x, y, s, r, vx, vy, vs].

    State:
      x, y: box center
      s: box scale (area)
      r: aspect ratio
      vx, vy, vs: velocities for x, y, s
    """

    def __init__(self, dim_x: int = 7, dim_z: int = 4, dim_u: int = 0, max_obs: int = 50):
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        motion_mat = self._build_motion_matrix(dim_x=dim_x, dim_z=dim_z)
        update_mat = np.zeros((dim_z, dim_x), dtype=float)
        update_mat[:, :dim_z] = np.eye(dim_z, dtype=float)

        super().__init__(
            ndim=dim_z,
            dim_x=dim_x,
            dim_z=dim_z,
            motion_mat=motion_mat,
            update_mat=update_mat,
            max_obs=max_obs,
        )

        self.dim_u = dim_u
        self.inv = np.linalg.inv

        self.max_obs = max_obs
        self.history_obs = deque([], maxlen=self.max_obs)
        self.attr_saved = None
        self.observed = False
        self.last_measurement = None

    @staticmethod
    def _build_motion_matrix(dim_x: int, dim_z: int) -> np.ndarray:
        """Build xysr-compatible transition matrix with a fallback for custom dims."""
        motion_mat = np.eye(dim_x, dtype=float)

        if dim_x >= 7 and dim_z >= 4:
            # [x, y, s, r, vx, vy, vs]
            motion_mat[0, 4] = 1.0
            motion_mat[1, 5] = 1.0
            motion_mat[2, 6] = 1.0
            return motion_mat

        velocity_dims = min(dim_z, max(0, dim_x - dim_z))
        for i in range(velocity_dims):
            motion_mat[i, dim_z + i] = 1.0
        return motion_mat

    @staticmethod
    def _scale_from_measurement(z: np.ndarray) -> float:
        arr = np.asarray(z, dtype=float).reshape(-1)
        if arr.size < 4:
            return 1.0

        s = max(arr[2], 1e-6)
        r = max(abs(arr[3]), 1e-6)
        w = np.sqrt(s * r)
        h = np.sqrt(s / r)
        return float(max(0.5 * (w + h), 1.0))

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        scale = self._scale_from_measurement(measurement)
        return np.array(
            [
                2.0 * self._std_weight_position * scale,
                2.0 * self._std_weight_position * scale,
                2.0 * self._std_weight_position * scale,
                1e-2,
                10.0 * self._std_weight_velocity * scale,
                10.0 * self._std_weight_velocity * scale,
                10.0 * self._std_weight_velocity * scale,
            ],
            dtype=float,
        )

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = self._scale_from_measurement(mean)
        std_pos = [
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            1e-2,
        ]
        std_vel = [
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
            1e-5,
        ]
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        scale = self._scale_from_measurement(mean)
        return np.array(
            [
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                1e-1,
            ],
            dtype=float,
        )

    def _get_multi_process_noise_std(
        self, mean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if mean.ndim != 2:
            raise ValueError("Expected mean to have shape (n, dim_x)")

        scales = np.array([self._scale_from_measurement(row) for row in mean], dtype=float)
        std_pos = [
            self._std_weight_position * scales,
            self._std_weight_position * scales,
            self._std_weight_position * scales,
            1e-2 * np.ones_like(scales),
        ]
        std_vel = [
            self._std_weight_velocity * scales,
            self._std_weight_velocity * scales,
            self._std_weight_velocity * scales,
            1e-5 * np.ones_like(scales),
        ]
        return std_pos, std_vel

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize xysr state [x, y, s, r, vx, vy, vs] from a measurement."""
        mean = np.zeros((self.dim_x, 1), dtype=float)
        measurement = self._reshape_measurement(measurement, self.dim_z)
        mean[: self.dim_z] = measurement

        std = self._get_initial_covariance_std(measurement)
        covariance = np.eye(self.dim_x, dtype=float)
        covariance[: std.shape[0], : std.shape[0]] = np.diag(np.square(std))
        return mean, covariance

    def apply_affine_correction(self, m: np.ndarray, t: np.ndarray) -> None:
        """Apply affine correction to state and covariance (used by DeepOcSort CMC)."""
        self.x[:2] = m @ self.x[:2] + t
        self.x[4:6] = m @ self.x[4:6]

        self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
        self.P[4:6, 4:6] = m @ self.P[4:6, 4:6] @ m.T

        if not self.observed and self.attr_saved is not None:
            self.attr_saved["x"][:2] = m @ self.attr_saved["x"][:2] + t
            self.attr_saved["x"][4:6] = m @ self.attr_saved["x"][4:6]

            self.attr_saved["P"][:2, :2] = m @ self.attr_saved["P"][:2, :2] @ m.T
            self.attr_saved["P"][4:6, 4:6] = m @ self.attr_saved["P"][4:6, 4:6] @ m.T

            if self.attr_saved["last_measurement"] is not None:
                self.attr_saved["last_measurement"][:2] = (
                    m @ self.attr_saved["last_measurement"][:2] + t
                )

    def predict(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ) -> None:
        """Predict one state step using shared base framework."""
        self.predict_state(u=u, B=B, F=F, Q=Q)

    def freeze(self) -> None:
        """Save parameters before non-observation forward pass."""
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self) -> None:
        """
        Restore the previously frozen state and replay interpolated observations.
        """
        if self.attr_saved is None:
            return

        new_history = deepcopy(list(self.history_obs))
        self.__dict__ = self.attr_saved
        self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)

        occur = [int(obs is None) for obs in new_history]
        indices = np.where(np.array(occur) == 0)[0]
        if len(indices) < 2:
            return

        index1, index2 = indices[-2], indices[-1]
        box1, box2 = new_history[index1], new_history[index2]

        x1, y1, s1, r1 = np.asarray(box1).flatten()
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        x2, y2, s2, r2 = np.asarray(box2).flatten()
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)

        time_gap = index2 - index1
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

        for i in range(index2 - index1):
            x = x1 + (i + 1) * dx
            y = y1 + (i + 1) * dy
            w = w1 + (i + 1) * dw
            h = h1 + (i + 1) * dh
            s, r = w * h, w / float(h)
            new_box = np.array([x, y, s, r], dtype=float).reshape((4, 1))

            self.update(new_box)
            if i != (index2 - index1 - 1):
                self.predict()
                self.history_obs.pop()

        self.history_obs.pop()

    def update(
        self,
        z: Optional[np.ndarray],
        R: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ) -> None:
        """Update state with measurement or register missing observation when z is None."""
        self.history_obs.append(z)

        if z is None:
            if self.observed and len(self.history_obs) >= 2:
                self.last_measurement = self.history_obs[-2]
                self.freeze()

            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return

        if not self.observed:
            self.unfreeze()
        self.observed = True

        self.update_state(z=z, R=R, H=H)

        # Keep legacy behavior where observed measurements are appended twice.
        self.history_obs.append(self.z.copy())

    def md_for_measurement(self, z: np.ndarray) -> float:
        """Mahalanobis distance of measurement z against current predicted state."""
        return self.mahalanobis_distance(z=z, H=self.H, R=self.R)
