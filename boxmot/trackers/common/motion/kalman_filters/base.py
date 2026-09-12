from collections import deque
from numbers import Real
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.linalg

from boxmot.trackers.common.motion.kalman_filters import batch
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig

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
        noise_config: KalmanNoiseConfig | None = None,
    ):
        if noise_config is not None and not isinstance(noise_config, KalmanNoiseConfig):
            raise TypeError("noise_config must be a KalmanNoiseConfig object.")
        self.noise_config = KalmanNoiseConfig() if noise_config is None else noise_config
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
        self._update_mat = update_mat.astype(float).copy() if update_mat is not None else np.eye(self.dim_z, self.dim_x)
        self.F = self._motion_mat.copy()
        self.H = self._update_mat.copy()

        # Motion and observation uncertainty weights.
        self._std_weight_position = getattr(type(self), "_tuned_std_weight_position", 1.0 / 20)
        self._std_weight_velocity = getattr(type(self), "_tuned_std_weight_velocity", 1.0 / 160)

        # Stateful Kalman filter members used by matrix-based subclasses.
        self.x = np.zeros((self.dim_x, 1))
        self.P = self.noise_config.initial_covariance(np.eye(self.dim_x), self.dim_z)
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

        # These records belong to stateful tracks. Stateless/batch prediction
        # never changes timing on a filter shared by several tracks.
        self._time_aware = False
        self._last_observed_measurement = None
        self._prediction_origin = None
        self._prediction_steps = []
        self._prediction_history_overflowed = False
        self._unrecorded_prediction = False

    @staticmethod
    def _default_motion_matrix(dim_x: int, dim_z: int) -> np.ndarray:
        """Build a simple constant-velocity transition matrix."""
        motion_mat = np.eye(dim_x)
        velocity_dims = min(dim_z, max(0, dim_x - dim_z))
        for i in range(velocity_dims):
            motion_mat[i, dim_z + i] = 1.0
        return motion_mat

    @staticmethod
    def validate_dt(dt: float | None) -> float | None:
        """Validate an elapsed interval without accepting booleans or arrays."""
        if dt is None:
            return None
        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real):
            raise ValueError("dt must be a finite, strictly positive real scalar")
        try:
            dt = float(dt)
        except OverflowError as error:
            raise ValueError("dt must be a finite, strictly positive real scalar") from error
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite, strictly positive real scalar")
        return dt

    def _motion_generator(self, motion_mat: np.ndarray | None = None) -> np.ndarray:
        """Return A for the configured constant-velocity model, where A² = 0."""
        motion_mat = self._motion_mat if motion_mat is None else motion_mat
        return (motion_mat - np.eye(self.dim_x)) / self.dt

    def _validate_prediction_dt(self, dt: float | None) -> float | None:
        """Require a measured interval when state velocities use seconds."""
        dt = self.validate_dt(dt)
        if dt is None and self.noise_config.time_unit == "seconds":
            raise ValueError("A Kalman filter configured in seconds requires an explicit measured dt for prediction.")
        return dt

    def _elapsed_motion(
        self,
        noise: np.ndarray,
        dt: float | None,
        *,
        motion_mat: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build F(dt) and integrate continuous process noise over the interval.

        With an explicit interval, calibrated baseline noise becomes a spectral
        density L in the configured time unit. Seconds convert frame priors
        using the fixed reference interval independently of measured dt.
        Integrating (I + t A) L (I + t A).T from zero to dt gives
        dt L + dt²/2 (A L + L A.T) + dt³/3 A L A.T. This preserves
        positive semidefiniteness and composes over successive intervals for
        constant L. Noise may also contain a leading batch dimension.
        Omitting dt retains the configured discrete, fixed-step model.
        """
        dt = self._validate_prediction_dt(dt)
        motion_mat = self._motion_mat if motion_mat is None else motion_mat
        noise = self.noise_config.process_covariance(noise, self.dim_z, continuous=dt is not None)
        if dt is None:
            return motion_mat, noise
        generator = self._motion_generator(motion_mat)
        try:
            with np.errstate(over="raise", invalid="raise"):
                left = generator @ noise
                noise = dt * noise + (dt**2 / 2.0) * (left + noise @ generator.T) + (dt**3 / 3.0) * (left @ generator.T)
                motion_mat = np.eye(self.dim_x) + dt * generator
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("dt produces an unrepresentable motion covariance") from error
        return motion_mat, noise

    def _remember_observation(self, measurement: np.ndarray) -> None:
        """Retain one real observation as the origin for timed gap replay."""
        self._last_observed_measurement = measurement.copy()
        self._prediction_origin = None
        self._prediction_steps.clear()
        self._prediction_history_overflowed = False
        self._unrecorded_prediction = False

    def _record_prediction(
        self,
        interval: float,
        transition: np.ndarray,
        noise: np.ndarray,
        control: np.ndarray | None,
    ) -> None:
        """Keep actual prediction models between observations for gap replay."""
        if self._last_observed_measurement is None or self._prediction_history_overflowed:
            return
        if self.max_obs is not None and len(self._prediction_steps) >= self.max_obs:
            # An overlong gap cannot be reconstructed with bounded history.
            # Keep the current prediction and correct the recovery directly.
            self._prediction_origin = None
            self._prediction_steps.clear()
            self._prediction_history_overflowed = True
            return
        if not self._prediction_steps:
            self._prediction_origin = (self.x.copy(), self.P.copy())
        self._prediction_steps.append((interval, transition.copy(), noise.copy(), control))

    def transform_timed_history(
        self,
        transform_state: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        transform_measurement: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Move replay's origin into the current camera coordinates.

        Reuse the track's state/covariance and measurement transforms so timed
        replay has the same camera compensation as its live Kalman state.
        """
        if self._last_observed_measurement is not None:
            self._last_observed_measurement = transform_measurement(self._last_observed_measurement)
        if self._prediction_origin is not None:
            self._prediction_origin = transform_state(*self._prediction_origin)

    def _replay_timed_observations(
        self,
        measurement: np.ndarray,
        R: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ) -> None:
        """Reconstruct missing observations at their actual elapsed intervals.

        Subclasses supply geometry interpolation and correction hooks. The
        recovery measurement is left for the caller to correct exactly once.
        Separate prediction records avoid duplicate observation-history entries
        affecting elapsed time in observation-centric filters.
        """
        if self._prediction_origin is None or len(self._prediction_steps) < 2:
            return
        self.x, self.P = (value.copy() for value in self._prediction_origin)
        total = sum(step[0] for step in self._prediction_steps)
        elapsed = 0.0
        for index, (interval, transition, noise, control) in enumerate(self._prediction_steps):
            self.x = transition @ self.x
            if control is not None:
                self.x = self.x + control
            self.P = self._alpha_sq * (transition @ self.P @ transition.T) + noise
            self.x_prior, self.P_prior = self.x.copy(), self.P.copy()
            self._enforce_state_constraints()
            elapsed += interval
            if index < len(self._prediction_steps) - 1:
                interpolated = self._interpolate_observation(
                    self._last_observed_measurement, measurement, elapsed / total
                )
                self._correct_observation(interpolated, R=R, H=H)

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

    @staticmethod
    def _wrap_angle(angle: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        wrapped = (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
        if np.isscalar(angle):
            return float(wrapped)
        return wrapped

    @classmethod
    def _align_angle_to_reference(cls, angle: float, reference_angle: float) -> float:
        return float(reference_angle + cls._wrap_angle(float(angle) - float(reference_angle)))

    @staticmethod
    def _theta_velocity_index(dim_x: int) -> int:
        return dim_x - 1

    @classmethod
    def _select_obb_candidate(
        cls,
        *,
        reference_sizes: Tuple[float, float],
        reference_angle: float,
        candidates: Tuple[Tuple[float, float, float], ...],
        size_weight: float = 0.05,
        eps: float = 1e-6,
    ) -> Tuple[float, float, float]:
        """Choose equivalent OBB parameterization closest to reference state."""
        ref_s0 = max(float(reference_sizes[0]), eps)
        ref_s1 = max(float(reference_sizes[1]), eps)
        ref_theta = float(reference_angle)

        best_cost = float("inf")
        best: Tuple[float, float, float] = candidates[0]
        for cand_s0, cand_s1, cand_theta in candidates:
            s0 = max(float(cand_s0), eps)
            s1 = max(float(cand_s1), eps)
            theta_aligned = cls._align_angle_to_reference(cand_theta, ref_theta)
            angle_cost = abs(theta_aligned - ref_theta)
            size_cost = abs(np.log(s0 / ref_s0)) + abs(np.log(s1 / ref_s1))
            cost = angle_cost + (size_weight * size_cost)
            if cost < best_cost:
                best_cost = cost
                best = (s0, s1, theta_aligned)
        return best

    @classmethod
    def _enforce_state_geometry(
        cls,
        mean: np.ndarray,
        *,
        positive_indices: Tuple[int, ...],
        angle_index: Optional[int] = None,
        min_size: float = 1e-4,
    ) -> np.ndarray:
        """Clamp geometry dimensions positive and optionally wrap angle."""
        if mean.ndim == 1:
            for idx in positive_indices:
                mean[idx] = max(float(mean[idx]), min_size)
            if angle_index is not None:
                mean[angle_index] = float(cls._wrap_angle(mean[angle_index]))
            return mean

        for idx in positive_indices:
            mean[idx, :] = np.maximum(mean[idx, :], min_size)
        if angle_index is not None:
            mean[angle_index, :] = cls._wrap_angle(mean[angle_index, :])
        return mean

    @staticmethod
    def _prepare_gating_inputs(
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        project_fn,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        projected_mean, projected_cov = project_fn(mean, covariance)
        projected_mean = np.asarray(projected_mean, dtype=float).reshape(-1)
        measurements = np.asarray(measurements, dtype=float).copy()
        if measurements.ndim == 1:
            measurements = measurements.reshape(1, -1)
        return projected_mean, projected_cov, measurements

    @staticmethod
    def _gating_from_residuals(residuals: np.ndarray, covariance: np.ndarray, metric: str) -> np.ndarray:
        if metric == "gaussian":
            return np.sum(residuals * residuals, axis=1)
        if metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            solved = scipy.linalg.solve_triangular(
                cholesky_factor,
                residuals.T,
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )
            return np.sum(solved * solved, axis=0)
        raise ValueError("invalid distance metric")

    def _zero_theta_velocity(self, mean: np.ndarray) -> np.ndarray:
        theta_vel_idx = self._theta_velocity_index(self.dim_x)
        if mean.ndim == 2:
            mean[theta_vel_idx, :] = 0.0
        else:
            mean[theta_vel_idx] = 0.0
        return mean

    def _damp_theta_velocity(self, mean: np.ndarray, damping: float = 0.8) -> np.ndarray:
        """Damp angular velocity to reduce jitter while preserving turn dynamics."""
        theta_vel_idx = self._theta_velocity_index(self.dim_x)
        damping = float(np.clip(damping, 0.0, 1.0))
        if mean.ndim == 2:
            mean[theta_vel_idx, :] *= damping
        else:
            mean[theta_vel_idx] *= damping
        return mean

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = self._get_initial_covariance_std(measurement)
        covariance = self.noise_config.initial_covariance(np.diag(np.square(std)), self.dim_z)
        return mean, covariance

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        """
        Return initial standard deviations for the covariance matrix.
        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray, *, dt: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using the configured fixed step or an explicit elapsed interval."""
        dt = self.validate_dt(dt)
        std_pos, std_vel = self._get_process_noise_std(mean)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        motion_mat, motion_cov = self._elapsed_motion(motion_cov, dt)
        mean = np.dot(mean, motion_mat.T)
        covariance = np.linalg.multi_dot((motion_mat, covariance, motion_mat.T)) + motion_cov

        return mean, covariance

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return standard deviations for process noise.
        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
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

        innovation_cov = self.noise_config.measurement_covariance(np.diag(np.square(std)))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray, *, dt: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict a batch over one shared interval without changing filter state."""
        dt = self.validate_dt(dt)
        if len(mean) == 0:
            return mean.copy(), covariance.copy()
        if len(mean) == 1:
            state, uncertainty = BaseKalmanFilter.predict(self, mean[0], covariance[0], dt=dt)
            return state[None], uncertainty[None]
        std_pos, std_vel = self._get_multi_process_noise_std(mean)
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = batch.diagonal(sqr)
        motion_mat, motion_cov = self._elapsed_motion(motion_cov, dt)
        return batch.predict(mean, covariance, motion_mat, motion_cov)

    def _multi_measurement_covariance(self, mean: np.ndarray, confidence: float | np.ndarray) -> np.ndarray:
        """Build independent confidence-scaled measurement covariances."""
        std = self._get_multi_measurement_noise_std(mean)
        confidence = np.broadcast_to(np.asarray(confidence, dtype=float), (len(mean),))
        return self.noise_config.measurement_covariance(batch.diagonal(np.square(std * (1.0 - confidence[:, None]))))

    def _get_multi_measurement_noise_std(self, mean: np.ndarray) -> np.ndarray:
        """Return measurement standard deviations as an (N, dim_z) array."""
        raise NotImplementedError

    def multi_project(
        self, mean: np.ndarray, covariance: np.ndarray, confidence: float | np.ndarray = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project a batch using shared or per-track detection confidences."""
        noise = self._multi_measurement_covariance(mean, confidence)
        return mean @ self._update_mat.T, self._update_mat @ covariance @ self._update_mat.T + noise

    def multi_update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float | np.ndarray = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct independent states using batched Cholesky factorization."""
        if len(mean) == 1:
            score = float(np.broadcast_to(np.asarray(confidence), (1,))[0])
            state, uncertainty = BaseKalmanFilter.update(self, mean[0], covariance[0], measurement[0], score)
            return state[None], uncertainty[None]
        noise = self._multi_measurement_covariance(mean, confidence)
        new_mean, new_covariance, *_ = batch.correct(mean, covariance, measurement, self._update_mat, noise)
        return new_mean, new_covariance

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

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
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

    def predict_state(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        *,
        dt: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict state over dt; explicitly supplied F/Q are already discrete.

        An explicit F or Q overrides only that matrix and is never rescaled.
        B and u retain their discrete control-input semantics.
        """
        dt = self._validate_prediction_dt(dt)
        B = self._resolve_matrix(B, self.B)
        noise = self._resolve_matrix(Q, self.Q)
        if np.isscalar(noise):
            noise = np.eye(self.dim_x) * float(noise)
        if F is not None and Q is not None:
            motion, integrated_noise = F, noise
        else:
            motion, integrated_noise = self._elapsed_motion(noise, dt, motion_mat=self.F)
        F = self._resolve_matrix(F, self.F if dt is None else motion)
        Q = noise if Q is not None else integrated_noise
        control = np.dot(B, u) if B is not None and u is not None else None
        if dt is not None:
            self._time_aware = True
            if self._unrecorded_prediction:
                # Timing started partway through a gap whose earlier models
                # were intentionally not retained. Correct recovery directly.
                self._prediction_history_overflowed = True
        if self._time_aware:
            self._record_prediction(self.dt if dt is None else dt, F, Q, control)
        else:
            self._unrecorded_prediction = self._last_observed_measurement is not None

        if control is not None:
            self.x = np.dot(F, self.x) + control
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
        R = self.noise_config.measurement_covariance(self.R) if R is None else R
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
        R = self.noise_config.measurement_covariance(self.R) if R is None else R
        if np.isscalar(R):
            R = np.eye(self.dim_z) * float(R)

        measurement = self._reshape_measurement(z, self.dim_z)
        projected_mean, projected_cov = self.project_state(H=H, R=R)

        # Symmetrize the projected covariance to compensate for floating-point
        # drift accumulated through repeated predict/update cycles before
        # factorization.
        projected_cov = 0.5 * (projected_cov + projected_cov.T)

        chol_factor, lower = self._safe_cho_factor(projected_cov)
        self.K = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(self.P, H.T).T,
            check_finite=False,
        ).T
        self.y = measurement - projected_mean
        self.S = projected_cov
        self.SI = scipy.linalg.cho_solve((chol_factor, lower), np.eye(self.dim_z), check_finite=False)

        self.x = self.x + np.dot(self.K, self.y)
        # Joseph form keeps P symmetric and positive semi-definite under
        # numerical noise; the simple ``P - K S K^T`` form does not.
        I_KH = self._I - np.dot(self.K, H)
        self.P = np.linalg.multi_dot((I_KH, self.P, I_KH.T)) + np.linalg.multi_dot((self.K, R, self.K.T))
        self.P = 0.5 * (self.P + self.P.T)
        self.z = measurement.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        return self.x, self.P

    @staticmethod
    def _safe_cho_factor(matrix: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Cholesky factorize ``matrix`` with adaptive jitter on failure.

        Repeated predict/update cycles can drive ``H P H^T + R`` slightly
        non-positive-definite due to floating-point noise. Add a tiny ridge
        to the diagonal and retry; if that still fails, fall back to clipping
        negative eigenvalues so the filter can recover instead of crashing
        the whole tracking sequence.
        """
        try:
            return scipy.linalg.cho_factor(matrix, lower=True, check_finite=False)
        except scipy.linalg.LinAlgError:
            pass

        n = matrix.shape[0]
        diag = np.diagonal(matrix)
        scale = float(np.max(np.abs(diag))) if diag.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        eye = np.eye(n)
        for exponent in range(-12, 4):
            jitter = scale * (10.0**exponent)
            try:
                return scipy.linalg.cho_factor(matrix + jitter * eye, lower=True, check_finite=False)
            except scipy.linalg.LinAlgError:
                continue

        # Last resort: project ``matrix`` onto the nearest PSD matrix by
        # clipping negative eigenvalues, then add a tiny ridge to ensure
        # strict positive definiteness for cho_factor.
        symmetric = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(symmetric)
        floor = max(scale * 1e-6, 1e-12)
        eigvals = np.clip(eigvals, floor, None)
        repaired = (eigvecs * eigvals) @ eigvecs.T
        repaired = 0.5 * (repaired + repaired.T)
        return scipy.linalg.cho_factor(repaired, lower=True, check_finite=False)

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
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")
