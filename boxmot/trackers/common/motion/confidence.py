"""Independent confidence dynamics for score-aware tracking associations."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace

import numpy as np

from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_VECTOR_OPTIONS, KalmanNoiseConfig


class _ConfidenceFilter(BaseKalmanFilter):
    """Independent XYSCR score/velocity block with its observation recovery."""

    def __init__(self, initial_score: float, *, noise_config: KalmanNoiseConfig | None, max_obs: int | None) -> None:
        super().__init__(ndim=1, max_obs=max_obs, noise_config=noise_config)
        # These are the native SORT/HybridSORT runtime priors, rather than
        # the size-dependent priors used by stateless box calibration.
        self.x[0, 0] = float(initial_score)
        self.P = self.noise_config.initial_covariance(np.diag([10.0, 10000.0]), self.dim_z)
        self.Q = np.diag([1.0, 0.0001])
        self.R = np.array([[10.0]])
        self.x_prior, self.x_post = self.x.copy(), self.x.copy()
        self.P_prior, self.P_post = self.P.copy(), self.P.copy()

    def _enforce_state_constraints(self) -> None:
        """Match XYSCR symmetry without clipping its unbounded confidence."""
        self.P = 0.5 * (self.P + self.P.T)

    def predict(self, *, dt: float | None = None) -> None:
        """Use the shared elapsed-time noise model and bounded replay records."""
        self.predict_state(dt=dt)
        self._enforce_state_constraints()

    @staticmethod
    def _interpolate_observation(previous: np.ndarray, current: np.ndarray, fraction: float) -> np.ndarray:
        """Confidence is the linearly interpolated independent XYSCR coordinate."""
        return previous + fraction * (current - previous)

    def _correct_observation(self, measurement: np.ndarray, R=None, H=None) -> None:
        """Correct only score/velocity, preserving the native Joseph covariance."""
        if R is not None or H is not None:
            self.update_state(measurement, R=R, H=H)
            self._enforce_state_constraints()
            return
        noise = self.noise_config.measurement_covariance(self.R)
        variance = self.P[0, 0] + noise[0, 0]
        if not np.isfinite(variance) or variance <= 0.0:
            # Retain the shared jitter/eigenvalue recovery for a damaged or
            # extreme covariance; ordinary scalar innovations need no LAPACK.
            self.update_state(measurement)
            self._enforce_state_constraints()
            return
        inverse_factor = 1.0 / np.sqrt(variance)
        self.K = (self.P[:, :1] * inverse_factor) * inverse_factor
        self.y = measurement - self.x[:1]
        self.S = np.array([[variance]])
        self.SI = np.array([[inverse_factor * inverse_factor]])
        self.x = self.x + self.K @ self.y
        residual = self._I - self.K @ self.H
        # Match the multiplication order chosen by the native three-matrix
        # Joseph products while avoiding generic matrix-chain dispatch.
        self.P = residual @ (self.P @ residual.T) + self.K @ (noise @ self.K.T)
        self.P = 0.5 * (self.P + self.P.T)
        self.z = measurement.copy()
        self.x_post, self.P_post = self.x.copy(), self.P.copy()
        self._enforce_state_constraints()

    def _recover_fixed_observations(self) -> None:
        """Replay fixed-step gaps from the first missing prediction, as XYSCR."""
        if self.attr_saved is None:
            return
        history = deepcopy(list(self.history_obs))
        indices = [index for index, observation in enumerate(history) if observation is not None]
        if len(indices) < 2:
            self.attr_saved = None
            return
        previous_index, current_index = indices[-2:]
        previous, current = history[previous_index], history[current_index]
        gap = current_index - previous_index
        if gap <= 1:
            self.attr_saved = None
            return
        self.__dict__ = self.attr_saved
        for step in range(1, gap):
            self._correct_observation(self._interpolate_observation(previous, current, step / gap))
            self.predict()
        self.history_obs = deque(history, maxlen=self.max_obs)

    def update(self, measurement: np.ndarray | None) -> None:
        """Observe a score or mark a gap for the matching fixed/timed replay."""
        self.history_obs.append(None if measurement is None else measurement.copy())
        if measurement is None:
            if not self._time_aware and self.observed:
                self.attr_saved = deepcopy(self.__dict__)
            self.observed = False
            self.z = np.array([[None]])
            self.x_post, self.P_post = self.x.copy(), self.P.copy()
            self.y = np.zeros((1, 1))
            return
        if not self.observed:
            if self._time_aware:
                self._replay_timed_observations(measurement)
            else:
                self._recover_fixed_observations()
        self.observed = True
        self._correct_observation(measurement)
        self._remember_observation(measurement)


class ConfidencePrediction:
    """Retain HybridSORT's confidence block independently of box coordinates.

    Confidence and confidence velocity are independent of the geometric blocks
    in XYSCR. Keeping that authored model preserves both fixed-step and timed
    observation replay instead of replacing the association cue with a score EMA.
    """

    def __init__(
        self, initial_score: float, *, noise_config: KalmanNoiseConfig | None = None, max_obs: int | None = 50
    ) -> None:
        if noise_config is not None:
            # This auxiliary filter has its own confidence state. Geometric
            # calibration coordinates belong to the selected box filter and
            # must never be reinterpreted as confidence or dummy-box noise.
            noise_config = replace(
                noise_config,
                **{option.removeprefix("kalman.noise."): () for option in KALMAN_VECTOR_OPTIONS},
                by_class={},
            )
        self._filter = _ConfidenceFilter(initial_score, noise_config=noise_config, max_obs=max_obs)

    @staticmethod
    def _measurement(score: float) -> np.ndarray:
        """Express a score as the independent scalar measurement."""
        return np.array([[float(score)]])

    @property
    def value(self) -> float:
        """Return the unbounded predicted score for the caller's own clipping."""
        return float(self._filter.x[0, 0])

    def predict(self, *, dt: float | None = None) -> None:
        """Advance confidence over the same elapsed interval as geometry."""
        self._filter.predict(dt=dt)

    def update(self, score: float | None) -> None:
        """Observe confidence or mark a missing observation for later replay."""
        self._filter.update(None if score is None else self._measurement(score))
