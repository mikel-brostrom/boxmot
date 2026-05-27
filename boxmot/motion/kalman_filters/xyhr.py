from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from boxmot.motion.kalman_filters.base import BaseKalmanFilter


class ConstantNoiseXYHR:
    """Constant process/measurement noise policy used by BoostTrack.

    Supports per-class noise via the class-level ``_per_class_noise`` registry.
    When a ``cls_id`` is provided and found in the registry, the instance uses
    class-specific Q/R matrices. Otherwise it falls back to the global defaults.
    """

    # Per-class noise registry: {cls_id: {"q_pos_diag": ndarray, "q_vel_diag": ndarray, "r_diag": ndarray}}
    _per_class_noise: dict[int, dict] = {}

    def __init__(self, dim_x: int, dim_z: int = 4, cls_id: int | None = None):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.cls_id = cls_id

    def get_init_state_cov(self) -> np.ndarray:
        covariance = np.eye(self.dim_x, dtype=float)
        covariance[self.dim_z :, self.dim_z :] *= 1000.0
        covariance *= 10.0
        return covariance

    def _class_noise(self) -> dict | None:
        """Look up per-class noise for this instance's cls_id.

        Falls back to the global tuned entry (key -1) if the specific class
        is not registered.
        """
        if self.cls_id is not None and self.cls_id in self._per_class_noise:
            return self._per_class_noise[self.cls_id]
        # Global tuned fallback (registered with key -1)
        if -1 in self._per_class_noise:
            return self._per_class_noise[-1]
        return None

    def get_r(self) -> np.ndarray:
        cn = self._class_noise()
        if cn is not None and "r_diag" in cn:
            r_diag = np.asarray(cn["r_diag"], dtype=float)
            r = np.zeros((self.dim_z, self.dim_z), dtype=float)
            n = min(len(r_diag), self.dim_z)
            for i in range(n):
                r[i, i] = r_diag[i]
            # Fill remaining dims (e.g. theta in OBB) with defaults
            if self.dim_z > n:
                r_default = self._default_r()
                for i in range(n, self.dim_z):
                    r[i, i] = r_default[i, i]
            return r
        return self._default_r()

    def _default_r(self) -> np.ndarray:
        if self.dim_z == 5:
            return np.diag([1.0, 1.0, 10.0, 0.01, 0.01]).astype(float)
        return np.diag([1.0, 1.0, 10.0, 0.01]).astype(float)

    def get_q(self) -> np.ndarray:
        cn = self._class_noise()
        if cn is not None and "q_pos_diag" in cn:
            q_pos = np.asarray(cn["q_pos_diag"], dtype=float)
            q_vel = np.asarray(cn.get("q_vel_diag", q_pos * 0.01), dtype=float)
            q = np.zeros((self.dim_x, self.dim_x), dtype=float)
            n_pos = min(len(q_pos), self.dim_z)
            n_vel = min(len(q_vel), self.dim_z)
            for i in range(n_pos):
                q[i, i] = q_pos[i]
            for i in range(n_vel):
                q[self.dim_z + i, self.dim_z + i] = q_vel[i]
            # Fill remaining dims from default
            if self.dim_x > 2 * n_pos:
                q_default = self._default_q()
                for i in range(n_pos, self.dim_z):
                    q[i, i] = q_default[i, i]
                for i in range(max(n_pos, n_vel), self.dim_x - self.dim_z):
                    q[self.dim_z + i, self.dim_z + i] = q_default[self.dim_z + i, self.dim_z + i]
            return q
        return self._default_q()

    def _default_q(self) -> np.ndarray:
        process_noise = np.eye(self.dim_x, dtype=float)
        process_noise[self.dim_z :, self.dim_z :] *= 0.01
        if self.dim_z == 5:
            process_noise[4, 4] = 0.01
        return process_noise

    def observe_innovation(
        self,
        innovation: np.ndarray,
        K: np.ndarray,
        P: np.ndarray,
        F: np.ndarray,
    ) -> None:
        """No-op for constant policy."""


class AdaptiveNoiseXYHR(ConstantNoiseXYHR):
    """Innovation-based adaptive Q estimation (Mehra 1970).

    Maintains a sliding window of innovations and derives Q from the
    empirical innovation covariance.  Automatically accounts for CMC
    since innovations are computed after camera compensation.

    Falls back to the constant Q during the warmup period.
    """

    def __init__(self, dim_x: int, dim_z: int = 4, window: int = 30, warmup: int = 15, cls_id: int | None = None):
        super().__init__(dim_x, dim_z, cls_id=cls_id)
        self._window = window
        self._warmup = warmup
        self._innovations: list[np.ndarray] = []
        self._q_adaptive: Optional[np.ndarray] = None

    def observe_innovation(
        self,
        innovation: np.ndarray,
        K: np.ndarray,
        P: np.ndarray,
        F: np.ndarray,
    ) -> None:
        """Accumulate innovation and update Q estimate."""
        self._innovations.append(np.asarray(innovation, dtype=float).ravel())
        if len(self._innovations) > self._window:
            self._innovations.pop(0)

        if len(self._innovations) >= self._warmup:
            # Empirical innovation covariance
            inn_arr = np.array(self._innovations)
            S_hat = np.cov(inn_arr, rowvar=False, bias=True)

            # Q_hat = K * S_hat * K^T + P - F * P * F^T
            # (simplified: use diagonal of K*S*K^T as velocity noise estimate)
            KSKt = K @ S_hat @ K.T
            FPFt = F @ P @ F.T
            Q_raw = KSKt + P - FPFt

            # Keep only diagonal (stable), enforce non-negative
            Q_diag = np.maximum(np.diag(Q_raw), 0.0)
            # Blend with default to avoid degenerate zeros
            Q_default_diag = np.diag(super().get_q())
            alpha = 0.7  # trust adaptive estimate
            Q_diag = alpha * Q_diag + (1.0 - alpha) * Q_default_diag
            self._q_adaptive = np.diag(Q_diag)

    def get_q(self) -> np.ndarray:
        if self._q_adaptive is not None:
            return self._q_adaptive
        return super().get_q()


class KalmanFilterXYHR(BaseKalmanFilter):
    """
    Constant-noise Kalman filter for XYHR with optional OBB extension.

    - `dim_z=4, dim_x=8`: [x, y, h, r, vx, vy, vh, vr]
    - `dim_z=5, dim_x=10`: [x, y, h, r, theta, vx, vy, vh, vr, vtheta]

    This preserves BoostTrack's original constant-noise model while exposing
    the filter under the shared `boxmot.motion.kalman_filters` namespace.
    """

    def __init__(
        self,
        z: Optional[np.ndarray] = None,
        ndim: int = 8,
        dim_z: Optional[int] = None,
        dt: float = 1.0,
        track_id: int = -1,
        adaptive_kf: bool = False,
        cls_id: int | None = None,
    ):
        inferred_dim_z = 4
        if z is not None:
            z_size = int(np.asarray(z).size)
            inferred_dim_z = 5 if z_size >= 5 else 4
        if dim_z is None:
            dim_z = inferred_dim_z
        if dim_z not in (4, 5):
            raise ValueError("dim_z must be 4 (AABB) or 5 (OBB)")

        if dim_z == 5 and ndim == 8:
            ndim = 10
        min_dim_x = 2 * dim_z
        if ndim < min_dim_x:
            raise ValueError(f"ndim must be >= {min_dim_x} when dim_z is {dim_z}")

        self._is_obb = dim_z == 5
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
        if adaptive_kf:
            self.cov_update_policy = AdaptiveNoiseXYHR(ndim, dim_z, cls_id=cls_id)
        else:
            self.cov_update_policy = ConstantNoiseXYHR(ndim, dim_z, cls_id=cls_id)

        # Keep BoostTrack-compatible 1D state vector semantics.
        self.x = np.zeros((ndim,), dtype=float)
        self.covariance = self.cov_update_policy.get_init_state_cov()

        if z is not None:
            self.x[: self.dim_z] = self._reshape_measurement_vector(z)
            self._enforce_state_constraints()

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
        measurement = measurement[: self.dim_z]
        measurement[2] = max(float(measurement[2]), 1e-4)
        measurement[3] = max(float(measurement[3]), 1e-4)
        if self._is_obb:
            measurement[4] = float(self._wrap_angle(measurement[4]))
        return measurement

    def _enforce_state_constraints(self) -> None:
        self.x = self._enforce_state_geometry(
            self.x,
            positive_indices=(2, 3),
            angle_index=4 if self._is_obb else None,
            min_size=1e-4,
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

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
        if self._is_obb:
            mean[4] = float(self._wrap_angle(mean[4]))
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
            self._enforce_state_constraints()
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

    def update(
        self, z: np.ndarray, alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard KF correction with optional Kalman-gain suppression.

        ``alpha`` in (0, 1] scales the Kalman gain applied to the mean update
        (OccluTrack's Abnormal Motion Suppression). The covariance update is
        left unchanged so uncertainty still shrinks normally — only the mean
        trusts the (potentially abnormal) observation less. ``alpha=1.0``
        reproduces the standard update.
        """
        measurement = self._reshape_measurement_vector(z)
        if self._is_obb:
            reference_theta = float(self.x[4])
            measurement[4] = self._align_angle_to_reference(
                measurement[4], reference_theta
            )
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
        self.cov_update_policy.observe_innovation(
            innovation, kalman_gain, self.covariance, self._motion_mat
        )
        self.x = self.x + alpha * np.dot(innovation, kalman_gain.T)
        self.covariance = self.covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        self._enforce_state_constraints()

        return self.x, self.covariance
