from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.kalman_filters.stateful import constrain_area_state, predict_many, update_many


class KalmanFilterXYSCR(BaseKalmanFilter):
    """
    HybridSORT score-aware Kalman filter.

    AABB state: [x, y, s, c, r, vx, vy, vs, vc].
    OBB state: [x, y, s, c, r, theta, vx, vy, vs, vc, omega].
    The first five/six coordinates form the corresponding measurement.

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
        *,
        noise_config: KalmanNoiseConfig | None = None,
    ):
        if (dim_x, dim_z) not in ((9, 5), (11, 6)):
            raise ValueError("KalmanFilterXYSCR expects dim_x=9, dim_z=5 (AABB) or dim_x=11, dim_z=6 (OBB)")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        super().__init__(
            ndim=dim_z,
            dim_x=dim_x,
            dim_z=dim_z,
            motion_mat=self._build_motion_matrix(dim_x=dim_x, dim_z=dim_z),
            update_mat=np.eye(dim_z, dim_x, dtype=float),
            max_obs=max_obs,
            noise_config=noise_config,
        )
        self.dim_u = dim_u
        self._is_obb = dim_z == 6
        self.max_obs = None if max_obs is None else max(1, int(max_obs))
        self.history_obs = deque([], maxlen=self.max_obs)
        self.inv = np.linalg.inv

    @staticmethod
    def _build_motion_matrix(dim_x: int = 9, dim_z: int = 5) -> np.ndarray:
        """Extend the score/box model with angle velocity, keeping ratio static."""
        motion_mat = np.eye(dim_x, dtype=float)
        motion_mat[0, dim_z] = 1.0
        motion_mat[1, dim_z + 1] = 1.0
        motion_mat[2, dim_z + 2] = 1.0
        motion_mat[3, dim_z + 3] = 1.0
        if dim_z == 6:
            motion_mat[5, 10] = 1.0
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

    def _measurement_reference_state(self) -> np.ndarray | None:
        """Return the current oriented measurement for equivalent-box alignment."""
        return self.x[: self.dim_z, 0].copy() if self._is_obb else None

    @classmethod
    def _align_obb_measurement(cls, measurement: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Resolve reciprocal aspect/quarter-turn forms without changing score."""
        aligned = np.asarray(measurement, dtype=float).reshape(-1).copy()
        reference = np.asarray(reference, dtype=float).reshape(-1)
        ratio = max(float(aligned[4]), 1e-6)
        theta = float(aligned[5])
        candidates = (
            (1.0, ratio, theta),
            (1.0, ratio, theta + np.pi),
            (1.0, 1.0 / ratio, theta + np.pi / 2.0),
            (1.0, 1.0 / ratio, theta - np.pi / 2.0),
        )
        _, aligned[4], aligned[5] = cls._select_obb_candidate(
            reference_sizes=(1.0, max(float(reference[4]), 1e-6)),
            reference_angle=float(reference[5]),
            candidates=candidates,
        )
        aligned[2] = max(float(aligned[2]), 1e-6)
        return aligned

    def _prepare_measurement(self, z: np.ndarray, reference_state: np.ndarray | None = None) -> np.ndarray:
        """Prepare positive geometry and align the optional oriented rectangle."""
        measurement = self._reshape_measurement(z, self.dim_z).copy()
        measurement[2, 0] = max(float(measurement[2, 0]), 1e-6)
        measurement[4, 0] = max(float(measurement[4, 0]), 1e-6)
        if self._is_obb:
            measurement[5, 0] = self._wrap_angle(measurement[5, 0])
            if reference_state is not None:
                measurement[:, 0] = self._align_obb_measurement(measurement[:, 0], reference_state)
        return measurement

    def _enforce_state_constraints(self) -> None:
        self.x = constrain_area_state(
            self.x[:, 0],
            dim_z=self.dim_z,
            ratio_index=4,
            angle_index=5 if self._is_obb else None,
        )[:, None]
        self.P = 0.5 * (self.P + self.P.T)

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        scale = self._scale_from_measurement(measurement)
        std = [
            2.0 * self._std_weight_position * scale,
            2.0 * self._std_weight_position * scale,
            2.0 * self._std_weight_position * scale,
            1e-1,
            1e-2,
            10.0 * self._std_weight_velocity * scale,
            10.0 * self._std_weight_velocity * scale,
            10.0 * self._std_weight_velocity * scale,
            1e-2,
        ]
        if self._is_obb:
            std.insert(5, 1e-2)
            std.append(1e-5)
        return np.array(std, dtype=float)

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = self._scale_from_measurement(mean)
        position = [
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            1e-2,
            1e-2,
        ]
        velocity = [
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
            1e-3,
        ]
        if self._is_obb:
            position.append(1e-2)
            velocity.append(1e-5)
        return position, velocity

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        scale = self._scale_from_measurement(mean)
        std = [
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            1e-1,
            1e-1,
        ]
        if self._is_obb:
            std.append(1e-1)
        return np.array(std, dtype=float)

    def _get_multi_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if mean.ndim != 2:
            raise ValueError("Expected mean to have shape (n, dim_x)")

        area = np.maximum(mean[:, 2], 1e-6)
        ratio = np.maximum(mean[:, 4], 1e-6)
        scales = np.maximum(0.5 * (np.sqrt(area * ratio) + np.sqrt(area / ratio)), 1.0)
        position = [
            self._std_weight_position * scales,
            self._std_weight_position * scales,
            self._std_weight_position * scales,
            1e-2 * np.ones_like(scales),
            1e-2 * np.ones_like(scales),
        ]
        velocity = [
            self._std_weight_velocity * scales,
            self._std_weight_velocity * scales,
            self._std_weight_velocity * scales,
            1e-3 * np.ones_like(scales),
        ]
        if self._is_obb:
            position.append(1e-2 * np.ones_like(scales))
            velocity.append(1e-5 * np.ones_like(scales))
        return position, velocity

    def _get_multi_measurement_noise_std(self, mean: np.ndarray) -> np.ndarray:
        """Return scale-based measurement noise for the stateless batch API."""
        area = np.maximum(mean[:, 2], 1e-6)
        ratio = np.maximum(mean[:, 4], 1e-6)
        scales = np.maximum(0.5 * (np.sqrt(area * ratio) + np.sqrt(area / ratio)), 1.0)
        std = np.full((len(mean), self.dim_z), 1e-1, dtype=float)
        std[:, :3] = self._std_weight_position * scales[:, None]
        return std

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        measurement = self._prepare_measurement(measurement)
        mean = np.zeros((self.dim_x, 1), dtype=float)
        mean[: self.dim_z] = measurement

        std = self._get_initial_covariance_std(measurement)
        covariance = np.diag(np.square(std))
        mean[2, 0] = max(float(mean[2, 0]), 1e-6)
        mean[4, 0] = max(float(mean[4, 0]), 1e-6)
        return mean, self.noise_config.initial_covariance(covariance, self.dim_z)

    def predict(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        *,
        dt: float | None = None,
    ) -> None:
        """Predict score and box geometry over the supplied elapsed interval."""
        self.predict_state(u=u, B=B, F=F, Q=Q, dt=dt)
        self._enforce_state_constraints()

    def freeze(self) -> None:
        """Save state before a missing-observation run."""
        self.attr_saved = deepcopy(self.__dict__)

    @classmethod
    def predict_many(
        cls,
        filters: Sequence["KalmanFilterXYSCR"],
        *,
        dt: float | None = None,
        Q=None,
        F=None,
    ) -> None:
        """Predict independent filters, retaining their own models and histories."""
        predict_many(filters, dt=dt, Q=Q, F=F, score_state=True)

    @classmethod
    def update_many(
        cls,
        filters: Sequence["KalmanFilterXYSCR"],
        measurements: Sequence[np.ndarray | None],
        *,
        R=None,
        H=None,
    ) -> None:
        """Batch matched corrections; replay each recovering track's own history."""
        update_many(filters, measurements, R=R, H=H, score_state=True)

    def unfreeze(
        self,
        R: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ) -> None:
        """Replay missing observations, leaving recovery for one real correction."""
        if self.attr_saved is None:
            return

        history = deepcopy(list(self.history_obs))
        indices = [index for index, observation in enumerate(history) if observation is not None]
        if len(indices) < 2:
            # The anchor may have fallen out of bounded history. Retain the
            # current prediction instead of restoring an earlier missing frame.
            self.attr_saved = None
            return
        previous_index, current_index = indices[-2:]
        previous = np.asarray(history[previous_index], dtype=float).reshape(-1)
        current = np.asarray(history[current_index], dtype=float).reshape(-1)
        gap = current_index - previous_index
        if gap <= 1 or previous.size != self.dim_z or current.size != self.dim_z:
            self.attr_saved = None
            return

        # The snapshot already includes prediction to the first missing frame.
        # Correct only virtual observations, then predict to the real recovery.
        self.__dict__ = self.attr_saved
        for step in range(1, gap):
            interpolated = self._interpolate_observation(previous, current, step / gap)
            self._correct_observation(interpolated, R=R, H=H)
            self.predict()
        self.history_obs = deque(history, maxlen=self.max_obs)

    def update(
        self,
        z: Optional[np.ndarray],
        R: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ) -> None:
        measurement = (
            None if z is None else self._prepare_measurement(z, reference_state=self._measurement_reference_state())
        )
        self.history_obs.append(None if measurement is None else measurement.copy())

        if measurement is None:
            if not self._time_aware and self.observed:
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return

        if not self.observed:
            if self._time_aware:
                self._replay_timed_observations(measurement, R=R, H=H)
            else:
                self.unfreeze(R=R, H=H)
        self.observed = True

        self._correct_observation(measurement, R=R, H=H)
        self._remember_observation(measurement)

    def _correct_observation(
        self,
        measurement: np.ndarray,
        R: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ) -> None:
        """Correct a real or interpolated score-aware observation."""
        if self._is_obb:
            measurement = self._prepare_measurement(measurement, reference_state=self._measurement_reference_state())
        self.update_state(z=measurement, R=R, H=H)
        if self._is_obb:
            self.x = self._damp_theta_velocity(self.x, damping=0.8)
        self._enforce_state_constraints()

    def _interpolate_observation(self, previous: np.ndarray, current: np.ndarray, fraction: float) -> np.ndarray:
        """Interpolate position, confidence and box width/height in elapsed time."""
        previous = np.asarray(previous, dtype=float).reshape(-1)
        current = np.asarray(current, dtype=float).reshape(-1)
        if self._is_obb:
            current = self._align_obb_measurement(current, previous)
        measurement = previous + fraction * (current - previous)
        previous_w = np.sqrt(previous[2] * previous[4])
        previous_h = np.sqrt(previous[2] / previous[4])
        current_w = np.sqrt(current[2] * current[4])
        current_h = np.sqrt(current[2] / current[4])
        width = max(previous_w + fraction * (current_w - previous_w), 1e-6)
        height = max(previous_h + fraction * (current_h - previous_h), 1e-6)
        measurement[2], measurement[4] = width * height, width / height
        return measurement.reshape((self.dim_z, 1))

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Gate against native observation noise with equivalent OBB alignment."""
        projected, uncertainty, measurements = self._prepare_gating_inputs(
            mean, covariance, measurements, self.project_state
        )
        reference = np.asarray(mean, dtype=float).reshape(-1)[: self.dim_z] if self._is_obb else None
        measurements = np.asarray(
            [self._prepare_measurement(measurement, reference).reshape(-1) for measurement in measurements]
        ).reshape(-1, self.dim_z)
        if only_position:
            projected, uncertainty = projected[:2], uncertainty[:2, :2]
            measurements = measurements[:, :2]
        return self._gating_from_residuals(measurements - projected, uncertainty, metric)

    def md_for_measurement(self, z: np.ndarray) -> float:
        """Measure distance after aligning the candidate rectangle to the state."""
        measurement = self._prepare_measurement(z, reference_state=self._measurement_reference_state())
        return self.mahalanobis_distance(z=measurement, H=self.H)
