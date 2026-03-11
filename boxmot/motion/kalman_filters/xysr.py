from collections import deque
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from boxmot.motion.kalman_filters.base import BaseKalmanFilter


class KalmanFilterXYSR(BaseKalmanFilter):
    """
    Linear Kalman filter for XYSR with optional OBB angle extension.

    - `dim_z=4, dim_x=7`: [x, y, s, r, vx, vy, vs]
    - `dim_z=5, dim_x=9`: [x, y, s, r, theta, vx, vy, vs, vtheta]
    """

    def __init__(self, dim_x: int = 7, dim_z: int = 4, dim_u: int = 0, max_obs: int = 50):
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")
        if dim_z == 5 and dim_x != 9:
            raise ValueError("dim_x must be 9 when dim_z is 5 (XYSR + theta)")
        if dim_z > 5:
            raise ValueError("dim_z > 5 is not supported for XYSR")

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
        self._is_obb = dim_z >= 5
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

        if dim_x >= 9 and dim_z >= 5:
            # [x, y, s, r, theta, vx, vy, vs, vtheta]
            motion_mat[0, 5] = 1.0
            motion_mat[1, 6] = 1.0
            motion_mat[2, 7] = 1.0
            motion_mat[4, 8] = 1.0
            return motion_mat

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

    def _measurement_reference_state(self) -> Optional[np.ndarray]:
        if self._is_obb:
            return np.asarray(self.x[: self.dim_z, 0], dtype=float).copy()
        return None

    @classmethod
    def _align_obb_measurement(
        cls, measurement: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """
        Resolve equivalent OBB forms in XYSR space before update.

        In XYSR-OBB, a rectangle can be represented as:
        - (s, r, theta)
        - (s, r, theta + pi)
        - (s, 1/r, theta + pi/2)
        - (s, 1/r, theta - pi/2)
        Choose the form closest to the reference state to prevent angle flips.
        """
        aligned = np.asarray(measurement, dtype=float).copy().reshape((-1,))
        ref = np.asarray(reference, dtype=float).reshape((-1,))

        ref_r = max(float(ref[3]), 1e-6)
        ref_theta = float(ref[4])

        s = max(float(aligned[2]), 1e-6)
        r = max(float(aligned[3]), 1e-6)
        theta = float(aligned[4])

        candidates = (
            (s, r, theta),
            (s, r, theta + np.pi),
            (s, 1.0 / r, theta + (np.pi / 2.0)),
            (s, 1.0 / r, theta - (np.pi / 2.0)),
        )
        ratio_candidates = tuple(
            (1.0, cand_r, cand_theta) for _, cand_r, cand_theta in candidates
        )
        _, best_r, best_theta = cls._select_obb_candidate(
            reference_sizes=(1.0, ref_r),
            reference_angle=ref_theta,
            candidates=ratio_candidates,
        )
        aligned[2] = max(s, 1e-6)
        aligned[3] = max(best_r, 1e-6)
        aligned[4] = best_theta
        return aligned

    def _prepare_measurement(
        self, z: np.ndarray, reference_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        measurement = self._reshape_measurement(z, self.dim_z)
        measurement[2, 0] = max(float(measurement[2, 0]), 1e-6)
        measurement[3, 0] = max(float(measurement[3, 0]), 1e-6)
        if self._is_obb:
            raw = float(self._wrap_angle(measurement[4, 0]))
            measurement[4, 0] = raw
            if reference_state is not None:
                aligned = self._align_obb_measurement(
                    measurement[:, 0], np.asarray(reference_state, dtype=float).reshape(-1)
                )
                measurement[:, 0] = aligned
        return measurement

    def _enforce_state_constraints(self) -> None:
        self.x = self._enforce_state_geometry(
            self.x,
            positive_indices=(2, 3),
            angle_index=4 if self._is_obb else None,
            min_size=1e-6,
        )
        self.P = 0.5 * (self.P + self.P.T)

    @staticmethod
    def _affine_components(m: np.ndarray) -> Tuple[float, float, float]:
        u, _, vh = np.linalg.svd(m)
        rot = u @ vh
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1.0
            rot = u @ vh
        angle = float(np.arctan2(rot[1, 0], rot[0, 0]))
        scale_x = max(float(np.linalg.norm(m[:, 0])), 1e-6)
        scale_y = max(float(np.linalg.norm(m[:, 1])), 1e-6)
        return scale_x, scale_y, angle

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        scale = self._scale_from_measurement(measurement)
        if self._is_obb:
            return np.array(
                [
                    2.0 * self._std_weight_position * scale,  # x
                    2.0 * self._std_weight_position * scale,  # y
                    2.0 * self._std_weight_position * scale,  # s
                    1e-2,                                     # r
                    1e-2,                                     # theta
                    10.0 * self._std_weight_velocity * scale, # vx
                    10.0 * self._std_weight_velocity * scale, # vy
                    10.0 * self._std_weight_velocity * scale, # vs
                    1e-5,                                     # vtheta
                ],
                dtype=float,
            )
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
        if self._is_obb:
            std_pos = [
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                1e-2,
                1e-2,
            ]
            std_vel = [
                self._std_weight_velocity * scale,
                self._std_weight_velocity * scale,
                self._std_weight_velocity * scale,
                1e-5,
            ]
            return std_pos, std_vel
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
        if self._is_obb:
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
        if self._is_obb:
            std_pos = [
                self._std_weight_position * scales,
                self._std_weight_position * scales,
                self._std_weight_position * scales,
                1e-2 * np.ones_like(scales),
                1e-2 * np.ones_like(scales),
            ]
            std_vel = [
                self._std_weight_velocity * scales,
                self._std_weight_velocity * scales,
                self._std_weight_velocity * scales,
                1e-5 * np.ones_like(scales),
            ]
            return std_pos, std_vel
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
        measurement = self._prepare_measurement(measurement, reference_state=None)
        mean[: self.dim_z] = measurement

        std = self._get_initial_covariance_std(measurement)
        covariance = np.eye(self.dim_x, dtype=float)
        covariance[: std.shape[0], : std.shape[0]] = np.diag(np.square(std))
        mean[2, 0] = max(float(mean[2, 0]), 1e-6)
        mean[3, 0] = max(float(mean[3, 0]), 1e-6)
        if self._is_obb:
            mean[4, 0] = float(self._wrap_angle(mean[4, 0]))
        covariance = 0.5 * (covariance + covariance.T)
        return mean, covariance

    def apply_affine_correction(self, m: np.ndarray, t: np.ndarray) -> None:
        """Apply affine correction to state and covariance (used by DeepOcSort CMC)."""
        m = np.asarray(m, dtype=float).reshape((2, 2))
        t = np.asarray(t, dtype=float).reshape((2, 1))
        scale_x, scale_y, rot = self._affine_components(m)
        area_scale = scale_x * scale_y
        ratio_scale = scale_x / scale_y

        vel_slice = slice(5, 7) if self._is_obb else slice(4, 6)
        self.x[:2] = m @ self.x[:2] + t
        self.x[vel_slice] = m @ self.x[vel_slice]
        if self._is_obb:
            self.x[2, 0] *= area_scale
            self.x[3, 0] *= ratio_scale
            self.x[4, 0] = float(self._wrap_angle(self.x[4, 0] + rot))
            self.x[7, 0] *= area_scale

        self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
        self.P[vel_slice, vel_slice] = m @ self.P[vel_slice, vel_slice] @ m.T
        if self._is_obb:
            self.P[2, 2] *= area_scale**2
            self.P[3, 3] *= ratio_scale**2
            self.P[7, 7] *= area_scale**2

        if not self.observed and self.attr_saved is not None:
            self.attr_saved["x"][:2] = m @ self.attr_saved["x"][:2] + t
            self.attr_saved["x"][vel_slice] = m @ self.attr_saved["x"][vel_slice]
            if self._is_obb:
                self.attr_saved["x"][2, 0] *= area_scale
                self.attr_saved["x"][3, 0] *= ratio_scale
                self.attr_saved["x"][4, 0] = float(
                    self._wrap_angle(self.attr_saved["x"][4, 0] + rot)
                )
                self.attr_saved["x"][7, 0] *= area_scale

            self.attr_saved["P"][:2, :2] = m @ self.attr_saved["P"][:2, :2] @ m.T
            self.attr_saved["P"][vel_slice, vel_slice] = (
                m @ self.attr_saved["P"][vel_slice, vel_slice] @ m.T
            )
            if self._is_obb:
                self.attr_saved["P"][2, 2] *= area_scale**2
                self.attr_saved["P"][3, 3] *= ratio_scale**2
                self.attr_saved["P"][7, 7] *= area_scale**2

            if self.attr_saved["last_measurement"] is not None:
                self.attr_saved["last_measurement"][:2] = (
                    m @ self.attr_saved["last_measurement"][:2] + t
                )
                if self._is_obb:
                    self.attr_saved["last_measurement"][2, 0] *= area_scale
                    self.attr_saved["last_measurement"][3, 0] *= ratio_scale
                    self.attr_saved["last_measurement"][4, 0] = float(
                        self._wrap_angle(self.attr_saved["last_measurement"][4, 0] + rot)
                    )

        self._enforce_state_constraints()

    def predict(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ) -> None:
        """Predict one state step using shared base framework."""
        self.predict_state(u=u, B=B, F=F, Q=Q)
        self._enforce_state_constraints()

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

        box1 = np.asarray(box1, dtype=float).reshape(-1)
        box2 = np.asarray(box2, dtype=float).reshape(-1)
        if box1.size < self.dim_z or box2.size < self.dim_z:
            return

        x1, y1, s1, r1 = box1[:4]
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        x2, y2, s2, r2 = box2[:4]
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)

        time_gap = index2 - index1
        if time_gap <= 0:
            return
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap
        if self._is_obb:
            t1, t2 = box1[4], box2[4]
            dtheta = float(self._wrap_angle(t2 - t1)) / time_gap

        for i in range(index2 - index1):
            x = x1 + (i + 1) * dx
            y = y1 + (i + 1) * dy
            w = w1 + (i + 1) * dw
            h = h1 + (i + 1) * dh
            s, r = w * h, w / float(h)
            if self._is_obb:
                theta = float(self._wrap_angle(t1 + (i + 1) * dtheta))
                new_box = np.array([x, y, s, r, theta], dtype=float).reshape((5, 1))
            else:
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
        measurement = None
        if z is not None:
            measurement = self._prepare_measurement(
                z, reference_state=self._measurement_reference_state()
            )
        self.history_obs.append(None if measurement is None else measurement.copy())

        if measurement is None:
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

        self.update_state(z=measurement, R=R, H=H)
        if self._is_obb and self.dim_x >= 9:
            self.x = self._damp_theta_velocity(self.x, damping=0.8)
        self._enforce_state_constraints()

        # Keep legacy behavior where observed measurements are appended twice.
        self.history_obs.append(self.z.copy())

    def md_for_measurement(self, z: np.ndarray) -> float:
        """Mahalanobis distance of measurement z against current predicted state."""
        measurement = self._prepare_measurement(
            z, reference_state=self._measurement_reference_state()
        )
        return self.mahalanobis_distance(z=measurement, H=self.H, R=self.R)
