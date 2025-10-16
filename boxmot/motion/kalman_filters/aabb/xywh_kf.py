from collections import deque
from typing import Tuple

import numpy as np
import scipy.linalg

# -----------------------------------------------------------------------------
#  AMS Kalman Filter (OccluTrack, Eq. 7–9) — clean revision
# -----------------------------------------------------------------------------
# 2025-08-06 rev-3  ➜  fixed indentation slip that nested `_alpha` inside `_push`
#                   and removed duplicate legacy code fragments.
# -----------------------------------------------------------------------------

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter

__all__ = ["AMSKalmanFilterXYWH"]


# -----------------------------------------------------------------------------
# 1. Vanilla XYWH Kalman filter (unchanged)
# -----------------------------------------------------------------------------

class KalmanFilterXYWH(BaseKalmanFilter):
    """8-D state = (x, y, w, h, vx, vy, vw, vh)."""

    def __init__(self):
        super().__init__(ndim=4)

    # ----------------------------------------------------------- std helpers
    def _get_initial_covariance_std(self, m: np.ndarray):  # noqa: N803
        return [
            2 * self._std_weight_position * m[2],
            2 * self._std_weight_position * m[3],
            2 * self._std_weight_position * m[2],
            2 * self._std_weight_position * m[3],
            10 * self._std_weight_velocity * m[2],
            10 * self._std_weight_velocity * m[3],
            10 * self._std_weight_velocity * m[2],
            10 * self._std_weight_velocity * m[3],
        ]

    def _get_process_noise_std(self, mean: np.ndarray):
        std_pos = (self._std_weight_position * mean[[2, 3, 2, 3]]).tolist()
        std_vel = (self._std_weight_velocity * mean[[2, 3, 2, 3]]).tolist()
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, _: float):
        return (self._std_weight_position * mean[[2, 3, 2, 3]]).tolist()

    def _get_multi_process_noise_std(self, mean: np.ndarray):
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
        return std_pos, std_vel


# -----------------------------------------------------------------------------
# 2. Abnormal-Motion-Suppressed Kalman Filter
# -----------------------------------------------------------------------------

class AMSKalmanFilterXYWH(KalmanFilterXYWH):
    r"""AMS Kalman filter exactly following OccluTrack pseudocode (Eq. 7–9)."""

    def __init__(
        self,
        N: int = 30,
        alpha0: float = 0.25,
        theta_v: float = 0.25,
        *,
        norm_by_prev: bool = True,
    ) -> None:
        super().__init__()
        self.buffer: deque[np.ndarray] = deque(maxlen=N + 1)  # B_{k-N} … B_{k-1}
        self.alpha0 = float(alpha0)
        self.theta_v = float(theta_v)
        self.norm_prev = bool(norm_by_prev)

    # ------------------------------------------------ helper functions
    def _push(self, box: np.ndarray) -> None:
        """Append (x,y,w,h) to history buffer."""
        self.buffer.append(box.astype(float).copy())

    def _alpha(self, box_k: np.ndarray) -> float:
        if len(self.buffer) < 2:
            return 1.0

        # ---- copy buffer to an array -------------------------------------------------
        buf = np.asarray(self.buffer, dtype=float)     # shape (L, 4)

        v_k = box_k - buf[-1]

        if self.norm_prev:
            v_k[:2] /= buf[-1, 2:4]
            v_k[2:] /= buf[-1, 2:4]

        # historic velocities B_i − B_{i-1}
        hist = np.diff(buf, axis=0)                    # shape (L-1, 4)
        if self.norm_prev:
            wh = buf[:-1, 2:4]                         # widths & heights of B_{i-1}
            hist[:, :2] /= wh
            hist[:, 2:] /= wh

        v_bar = hist.mean(axis=0)

        d   = np.abs(v_k - v_bar)
        alp = np.where(d <= self.theta_v, 1.0, self.alpha0)

        return float(alp.mean())

    # ------------------------------------------------ KF interface
    def initiate(self, z: np.ndarray):
        m, P = super().initiate(z)
        self.buffer.clear()
        self._push(z)
        return m, P

    def predict(self, m: np.ndarray, P: np.ndarray):
        return super().predict(m, P)

    def update(self, m: np.ndarray, P: np.ndarray, z: np.ndarray, *, confidence: float = 0.0):
        z_hat, S = self.project(m, P, confidence)
        y = z - z_hat

        L, lower = scipy.linalg.cho_factor(S, check_finite=False, lower=True)
        K = scipy.linalg.cho_solve((L, lower), (P @ self._update_mat.T).T, check_finite=False).T
        K *= self._alpha(z)

        m_new = m + y @ K.T
        P_new = P - K @ S @ K.T

        self._push(z)
        return m_new, P_new
