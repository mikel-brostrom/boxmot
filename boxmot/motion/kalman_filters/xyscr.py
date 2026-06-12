from collections import deque
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from boxmot.motion.kalman_filters.base import BaseKalmanFilter


class KalmanFilterXYSCR(BaseKalmanFilter):
    """
    HybridSORT score-aware Kalman filter.

    State layout:
    [x, y, s, c, r, vx, vy, vs, vc]

    Measurement layout:
    [x, y, s, c, r]

    Here ``s`` is area, ``c`` is detection confidence, and ``r`` is aspect
    ratio. This differs from :class:`KalmanFilterXYSR`, whose 5-D path uses
    the last measurement dimension for OBB angle.
    """

    def __init__(
        self,
        dim_x: int = 9,
        dim_z: int = 5,
        dim_u: int = 0,
        max_obs: Optional[int] = 50,
    ):
        if dim_x != 9 or dim_z != 5:
            raise ValueError("KalmanFilterXYSCR expects dim_x=9 and dim_z=5")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        super().__init__(
            ndim=dim_z,
            dim_x=dim_x,
            dim_z=dim_z,
            motion_mat=self._build_motion_matrix(),
            update_mat=np.eye(dim_z, dim_x, dtype=float),
            max_obs=max_obs,
        )
        self.dim_u = dim_u
        self.max_obs = None if max_obs is None else max(1, int(max_obs))
        self.history_obs = deque([], maxlen=self.max_obs)
        self.inv = np.linalg.inv

    @staticmethod
    def _build_motion_matrix() -> np.ndarray:
        motion_mat = np.eye(9, dtype=float)
        motion_mat[0, 5] = 1.0
        motion_mat[1, 6] = 1.0
        motion_mat[2, 7] = 1.0
        motion_mat[3, 8] = 1.0
        return motion_mat

    @staticmethod
    def _scale_from_measurement(measurement: np.ndarray) -> float:
        arr = np.asarray(measurement, dtype=float).reshape(-1)
        if arr.size < 5:
            return 1.0

        s = max(float(arr[2]), 1e-6)
        r = max(float(arr[4]), 1e-6)
        w = np.sqrt(s * r)
        h = np.sqrt(s / r)
        return float(max(0.5 * (w + h), 1.0))

    def _prepare_measurement(self, z: np.ndarray) -> np.ndarray:
        measurement = self._reshape_measurement(z, self.dim_z)
        measurement[2, 0] = max(float(measurement[2, 0]), 1e-6)
        measurement[4, 0] = max(float(measurement[4, 0]), 1e-6)
        return measurement

    def _enforce_state_constraints(self) -> None:
        self.x = self._enforce_state_geometry(
            self.x,
            positive_indices=(2, 4),
            min_size=1e-6,
        )
        self.P = 0.5 * (self.P + self.P.T)

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        scale = self._scale_from_measurement(measurement)
        return np.array(
            [
                2.0 * self._std_weight_position * scale,
                2.0 * self._std_weight_position * scale,
                2.0 * self._std_weight_position * scale,
                1e-1,
                1e-2,
                10.0 * self._std_weight_velocity * scale,
                10.0 * self._std_weight_velocity * scale,
                10.0 * self._std_weight_velocity * scale,
                1e-2,
            ],
            dtype=float,
        )

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = self._scale_from_measurement(mean)
        return (
            [
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                1e-2,
                1e-2,
            ],
            [
                self._std_weight_velocity * scale,
                self._std_weight_velocity * scale,
                self._std_weight_velocity * scale,
                1e-3,
            ],
        )

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        scale = self._scale_from_measurement(mean)
        return np.array(
            [
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                1e-1,
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
        return (
            [
                self._std_weight_position * scales,
                self._std_weight_position * scales,
                self._std_weight_position * scales,
                1e-2 * np.ones_like(scales),
                1e-2 * np.ones_like(scales),
            ],
            [
                self._std_weight_velocity * scales,
                self._std_weight_velocity * scales,
                self._std_weight_velocity * scales,
                1e-3 * np.ones_like(scales),
            ],
        )

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        measurement = self._prepare_measurement(measurement)
        mean = np.zeros((self.dim_x, 1), dtype=float)
        mean[: self.dim_z] = measurement

        std = self._get_initial_covariance_std(measurement)
        covariance = np.diag(np.square(std))
        mean[2, 0] = max(float(mean[2, 0]), 1e-6)
        mean[4, 0] = max(float(mean[4, 0]), 1e-6)
        return mean, covariance

    def predict(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ) -> None:
        self.predict_state(u=u, B=B, F=F, Q=Q)
        self._enforce_state_constraints()

    def freeze(self) -> None:
        """Save state before a missing-observation run."""
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self) -> None:
        """Replay interpolated observations after a missing-observation run."""
        if self.attr_saved is None:
            return

        new_history = list(deepcopy(self.history_obs))
        self.__dict__ = self.attr_saved
        retained_history = list(self.history_obs)[:-1]
        self.history_obs = deque(retained_history, maxlen=self.max_obs)

        occur = [int(obs is None) for obs in new_history]
        indices = np.where(np.array(occur) == 0)[0]
        if len(indices) < 2:
            return

        index1, index2 = indices[-2], indices[-1]
        time_gap = index2 - index1
        if time_gap <= 0:
            return

        box1 = np.asarray(new_history[index1], dtype=float).reshape(-1)
        box2 = np.asarray(new_history[index2], dtype=float).reshape(-1)
        if box1.size < self.dim_z or box2.size < self.dim_z:
            return

        x1, y1, s1, c1, r1 = box1[:5]
        x2, y2, s2, c2, r2 = box2[:5]
        w1 = np.sqrt(max(s1, 1e-6) * max(r1, 1e-6))
        h1 = np.sqrt(max(s1, 1e-6) / max(r1, 1e-6))
        w2 = np.sqrt(max(s2, 1e-6) * max(r2, 1e-6))
        h2 = np.sqrt(max(s2, 1e-6) / max(r2, 1e-6))

        dx = (x2 - x1) / time_gap
        dy = (y2 - y1) / time_gap
        dw = (w2 - w1) / time_gap
        dh = (h2 - h1) / time_gap
        dc = (c2 - c1) / time_gap

        for i in range(time_gap):
            x = x1 + (i + 1) * dx
            y = y1 + (i + 1) * dy
            w = max(w1 + (i + 1) * dw, 1e-6)
            h = max(h1 + (i + 1) * dh, 1e-6)
            s = w * h
            c = c1 + (i + 1) * dc
            r = w / h
            new_box = np.array([x, y, s, c, r], dtype=float).reshape((5, 1))

            self.update(new_box)
            if i != (time_gap - 1):
                self.predict()

    def update(
        self,
        z: Optional[np.ndarray],
        R: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ) -> None:
        measurement = None if z is None else self._prepare_measurement(z)
        self.history_obs.append(None if measurement is None else measurement.copy())

        if measurement is None:
            if self.observed:
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

        self.update_state(z=measurement, R=R, H=H)
        self._enforce_state_constraints()

    def md_for_measurement(self, z: np.ndarray) -> float:
        measurement = self._prepare_measurement(z)
        return self.mahalanobis_distance(z=measurement, H=self.H, R=self.R)
