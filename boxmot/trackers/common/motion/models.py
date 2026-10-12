"""Tracker motion-model adapters, state conversions, and filter construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from boxmot.trackers.common.geometry.obb import normalize_angle
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.trackers.common.motion.kalman_filters.xyhr import KalmanFilterXYHR
from boxmot.trackers.common.motion.kalman_filters.xyscr import KalmanFilterXYSCR
from boxmot.trackers.common.motion.kalman_filters.xysr import KalmanFilterXYSR
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH


class MotionModelKind(str, Enum):
    XYAH = "xyah"
    XYWH = "xywh"
    XYSR = "xysr"
    XYHR = "xyhr"
    XYSCR = "xyscr"


def supported_motion_model_kinds(*, is_obb: bool = False) -> tuple[MotionModelKind, ...]:
    """Return the filter policies supported by the requested box geometry."""
    return tuple(MotionModelKind)


def resolve_motion_model_kind(
    tracker_name: str, kind: MotionModelKind | str | None = None, *, is_obb: bool = False
) -> MotionModelKind:
    """Resolve a tracker's authored default or an explicit box-state model."""
    defaults = {
        "bytetrack": MotionModelKind.XYAH,
        "strongsort": MotionModelKind.XYAH,
        "botsort": MotionModelKind.XYWH,
        "ocsort": MotionModelKind.XYSR,
        "deepocsort": MotionModelKind.XYSR,
        "hybridsort": MotionModelKind.XYSCR,
        "boosttrack": MotionModelKind.XYHR,
        "occluboost": MotionModelKind.XYHR,
    }
    if tracker_name not in defaults:
        raise ValueError(f"Tracker {tracker_name!r} does not support selectable box Kalman models")
    default = defaults[tracker_name]
    if is_obb:
        if tracker_name in {"bytetrack", "strongsort"}:
            default = MotionModelKind.XYWH
        elif tracker_name == "hybridsort":
            default = MotionModelKind.XYSR
    resolved = default if kind is None else MotionModelKind(kind)
    if resolved not in supported_motion_model_kinds(is_obb=is_obb):
        raise ValueError("Unsupported Kalman model for the configured geometry.")
    return resolved


FilterFactory = Callable[[np.ndarray | None, KalmanNoiseConfig | None], Any]
MeasurementConverter = Callable[[np.ndarray], np.ndarray]
StateConverter = Callable[[np.ndarray, float | None], np.ndarray]


@dataclass(frozen=True)
class MotionModelAdapter:
    """Canonical adapter for tracker motion-state conventions."""

    kind: MotionModelKind
    dim_x: int
    dim_z: int
    is_obb: bool
    _filter_factory: FilterFactory
    _measurement_from_box: MeasurementConverter
    _box_from_state: StateConverter
    max_obs: int = 50

    @property
    def angle_measurement_index(self) -> int | None:
        """Locate orientation independently of optional confidence coordinates."""
        if not self.is_obb:
            return None
        return 5 if self.kind is MotionModelKind.XYSCR else 4

    @property
    def velocity_measurement_indices(self) -> tuple[int, ...]:
        """Measured coordinates whose derivatives occupy the velocity block."""
        if self.kind is MotionModelKind.XYSR:
            return (0, 1, 2, 4) if self.is_obb else (0, 1, 2)
        if self.kind is MotionModelKind.XYSCR:
            return (0, 1, 2, 3, 5) if self.is_obb else (0, 1, 2, 3)
        return tuple(range(self.dim_z))

    @property
    def geometry_measurement_indices(self) -> tuple[int, ...]:
        """Measured box coordinates, excluding detector confidence."""
        if self.kind is MotionModelKind.XYSCR:
            return (0, 1, 2, 4, 5) if self.is_obb else (0, 1, 2, 4)
        return tuple(range(self.dim_z))

    @property
    def size_velocity_indices(self) -> tuple[int, ...]:
        """State indices of size/ratio derivatives, excluding angle and score."""
        coordinates = (2, 4) if self.kind is MotionModelKind.XYSCR else (2, 3)
        return tuple(
            self.dim_z + i
            for i, coordinate in enumerate(self.velocity_measurement_indices)
            if coordinate in coordinates
        )

    @property
    def height_velocity_index(self) -> int:
        """Height derivative, or area derivative for an area-state model."""
        coordinate = 3 if self.kind in (MotionModelKind.XYAH, MotionModelKind.XYWH) else 2
        return self.dim_z + self.velocity_measurement_indices.index(coordinate)

    def stop_size_velocity(
        self, mean: np.ndarray, *, height_only: bool = False, include_angle: bool = False
    ) -> np.ndarray:
        """Copy a state, resetting selected size derivatives for a lost track."""
        result = np.asarray(mean).copy()
        indices = (self.height_velocity_index,) if height_only else self.size_velocity_indices
        if include_angle and self.is_obb:
            indices += (self.dim_z + self.velocity_measurement_indices.index(self.angle_measurement_index),)
        result.reshape(-1)[list(indices)] = 0.0
        return result

    def prevent_invalid_size(self, kalman: Any, *, dt: float | None = None) -> None:
        """Stop shape derivatives that would predict a nonpositive dimension."""
        interval = 1.0 if dt is None else dt
        state = np.asarray(kalman.x).reshape(-1)
        for index in self.size_velocity_indices:
            coordinate = self.velocity_measurement_indices[index - self.dim_z]
            if state[coordinate] + interval * state[index] <= 0.0:
                state[index] = 0.0

    def configure_filter(self, kalman: Any) -> None:
        """Set the matrix-model observation/process priors used by its tracker."""
        if self.kind not in (MotionModelKind.XYSR, MotionModelKind.XYSCR):
            return
        kalman.R = np.eye(self.dim_z)
        kalman.R[2:, 2:] *= 10.0
        kalman.Q = np.eye(self.dim_x)
        kalman.Q[self.dim_z :, self.dim_z :] *= 0.01
        kalman.Q[self.dim_z + 2, self.dim_z + 2] = 0.0001
        if self.kind is MotionModelKind.XYSCR:
            kalman.Q[self.dim_z + 3, self.dim_z + 3] = 0.0001
        if self.is_obb or self.kind is MotionModelKind.XYSCR:
            kalman.Q[-1, -1] = 0.0001

    def initiate(
        self, kalman: Any, measurement: np.ndarray, *, confidence: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize native priors and optional track-owned confidence history."""
        if self.kind in (MotionModelKind.XYSR, MotionModelKind.XYSCR):
            self.configure_filter(kalman)
            mean = np.zeros(self.dim_x)
            mean[: self.dim_z] = np.asarray(measurement).reshape(-1)
            covariance = np.eye(self.dim_x) * 10.0
            covariance[self.dim_z :, self.dim_z :] *= 1000.0
            return mean, kalman.noise_config.initial_covariance(covariance, self.dim_z)
        mean, covariance = kalman.initiate(np.asarray(measurement).reshape(-1))
        return np.asarray(mean).reshape(-1), covariance

    def initialize_filter(self, kalman: Any, measurement: np.ndarray, *, confidence: float | None = None) -> None:
        """Initialize independently owned state while retaining native layouts."""
        mean, kalman.P = self.initiate(kalman, measurement, confidence=confidence)
        kalman.x = mean if self.kind is MotionModelKind.XYHR else mean[:, None]
        kalman.x_prior, kalman.x_post = kalman.x.copy(), kalman.x.copy()
        kalman.P_prior, kalman.P_post = kalman.P.copy(), kalman.P.copy()

    def predict(
        self, kalman: Any, mean: np.ndarray, covariance: np.ndarray, *, dt: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict one externally owned distribution without sharing histories."""
        from boxmot.trackers.common.motion.operations import predict

        return predict(self, kalman, mean, covariance, dt=dt)

    def multi_predict(
        self, kalman: Any, mean: np.ndarray, covariance: np.ndarray, *, dt: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict independent externally owned states in one batch."""
        from boxmot.trackers.common.motion.operations import multi_predict

        return multi_predict(self, kalman, mean, covariance, dt=dt)

    def update(
        self, kalman: Any, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct an externally owned state with the selected model."""
        from boxmot.trackers.common.motion.operations import update

        return update(self, kalman, mean, covariance, measurement, confidence)

    def multi_update(
        self,
        kalman: Any,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float | np.ndarray = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct a batch without mixing track-specific observation histories."""
        from boxmot.trackers.common.motion.operations import multi_update

        return multi_update(self, kalman, mean, covariance, measurement, confidence)

    def project(
        self, kalman: Any, mean: np.ndarray, covariance: np.ndarray, confidence: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a distribution using the selected filter's observation noise."""
        from boxmot.trackers.common.motion.operations import project

        return project(self, kalman, mean, covariance, confidence)

    def gating_distance(
        self,
        kalman: Any,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Compute squared association distances in the selected coordinates."""
        from boxmot.trackers.common.motion.operations import gating_distance

        return gating_distance(self, kalman, mean, covariance, measurements, only_position, metric)

    def predict_filter(self, kalman: Any, *, dt: float | None = None, observation_centric: bool = False) -> None:
        """Advance one independently owned filter and its replay records."""
        from boxmot.trackers.common.motion.operations import predict_filter

        predict_filter(self, kalman, dt=dt, observation_centric=observation_centric)

    def update_filter(
        self,
        kalman: Any,
        measurement: np.ndarray | None,
        *,
        observation_centric: bool = False,
        alpha: float = 1.0,
        confidence: float = 0.0,
    ) -> None:
        """Correct owned state, optionally reconstructing missing observations."""
        from boxmot.trackers.common.motion.operations import update_filter

        update_filter(
            self, kalman, measurement, observation_centric=observation_centric, alpha=alpha, confidence=confidence
        )

    def predict_filters(
        self, filters: Sequence[Any], *, dt: float | None = None, observation_centric: bool = False
    ) -> None:
        """Predict independently configured native filters."""
        from boxmot.trackers.common.motion.operations import predict_filters

        predict_filters(self, filters, dt=dt, observation_centric=observation_centric)

    def update_filters(
        self,
        filters: Sequence[Any],
        measurements: Sequence[np.ndarray | None],
        *,
        observation_centric: bool = False,
        alpha: float | Sequence[float] = 1.0,
        confidence: float | Sequence[float] = 0.0,
    ) -> None:
        """Correct independently configured filters with optional gap replay."""
        from boxmot.trackers.common.motion.operations import update_filters

        update_filters(
            self, filters, measurements, observation_centric=observation_centric, alpha=alpha, confidence=confidence
        )

    def transform_measurements(self, measurements: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Warp box measurements while retaining non-geometric confidence."""
        from boxmot.trackers.common.geometry.obb import transform_aabbs, transform_obbs
        from boxmot.trackers.common.motion.cmc.state import is_orientation_preserving_similarity

        if (
            not self.is_obb
            and self.kind in (MotionModelKind.XYSR, MotionModelKind.XYSCR, MotionModelKind.XYHR)
            and is_orientation_preserving_similarity(transform)
        ):
            transformed = np.asarray(measurements, dtype=float).copy()
            matrix = np.asarray(transform, dtype=np.float64)
            transformed[:, :2] = transformed[:, :2] @ matrix[:2, :2].T + matrix[:2, 2]
            determinant = np.linalg.det(matrix[:2, :2])
            transformed[:, 2] *= np.sqrt(determinant) if self.kind is MotionModelKind.XYHR else determinant
            return transformed

        warp = transform_obbs if self.is_obb else transform_aabbs
        transformed = self.to_measurements(warp(self.to_boxes(measurements), transform))
        if self.kind is MotionModelKind.XYSCR:
            transformed[:, 3] = measurements[:, 3]
        return transformed

    def transform_boxes(self, boxes: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Warp observation boxes using the same shape policy as live filter states."""
        from boxmot.trackers.common.geometry.obb import transform_aabbs, transform_obbs
        from boxmot.trackers.common.motion.cmc.state import is_orientation_preserving_similarity

        if (
            not self.is_obb
            and self.kind in (MotionModelKind.XYSR, MotionModelKind.XYSCR, MotionModelKind.XYHR)
            and is_orientation_preserving_similarity(transform)
        ):
            result = np.asarray(boxes, dtype=float).copy()
            matrix = np.asarray(transform, dtype=np.float64)
            centers = (result[:, :2] + result[:, 2:4]) / 2
            half_sizes = (result[:, 2:4] - result[:, :2]) / 2 * np.sqrt(np.linalg.det(matrix[:2, :2]))
            centers = centers @ matrix[:2, :2].T + matrix[:2, 2]
            result[:, :2], result[:, 2:4] = centers - half_sizes, centers + half_sizes
            return result
        return (transform_obbs if self.is_obb else transform_aabbs)(boxes, transform)

    def transform_states(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        transform: np.ndarray,
        *,
        return_jacobian: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Warp state/covariance, optionally exposing the runtime local Jacobian."""
        from boxmot.trackers.common.motion.operations import transform_states

        return transform_states(self, means, covariances, transform, return_jacobian=return_jacobian)

    def create_filter(
        self, initial_measurement: np.ndarray | None = None, *, noise_config: KalmanNoiseConfig | None = None
    ) -> Any:
        """Create an independent filter with the caller's noise configuration."""
        measurement = None
        if initial_measurement is not None:
            measurement = np.asarray(initial_measurement, dtype=float).reshape(-1)
        kalman = self._filter_factory(measurement, noise_config)
        kalman.max_obs = self.max_obs
        kalman.history_obs = type(kalman.history_obs)(maxlen=self.max_obs)
        return kalman

    def to_measurement(self, box: np.ndarray, column: bool = True) -> np.ndarray:
        measurement = self._measurement_from_box(box)
        if column:
            return measurement.reshape((self.dim_z, 1))
        return measurement

    def to_box(self, state: np.ndarray, score: float | None = None) -> np.ndarray:
        return self._box_from_state(state, score)

    def to_measurements(self, boxes: np.ndarray) -> np.ndarray:
        """Convert rows of boxes using the scalar adapter's geometry policy."""
        boxes = np.asarray(boxes, dtype=float)
        if self.is_obb:
            result = np.empty((len(boxes), self.dim_z), dtype=float)
            result[:, :2] = boxes[:, :2]
            minimum = 1e-4 if self.kind in (MotionModelKind.XYWH, MotionModelKind.XYHR) else 1e-6
            width = np.maximum(boxes[:, 2], minimum)
            height = np.maximum(boxes[:, 3], minimum)
            result[:, self.angle_measurement_index] = (boxes[:, 4] + np.pi) % (2 * np.pi) - np.pi
        else:
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            result = np.empty((len(boxes), self.dim_z), dtype=float)
            result[:, :2] = boxes[:, :2] + np.column_stack((width, height)) / 2.0
        if self.kind in (MotionModelKind.XYWH,):
            result[:, 2:4] = np.column_stack((width, height))
        elif self.kind is MotionModelKind.XYAH:
            height = np.maximum(height, 1e-6)
            result[:, 2:4] = np.column_stack((width / height, height))
        elif self.kind is MotionModelKind.XYHR:
            denominator = height if self.is_obb else height + 1e-6
            result[:, 2:4] = np.column_stack((height, width / denominator))
        elif self.kind is MotionModelKind.XYSR:
            denominator = height if self.is_obb else height + 1e-6
            result[:, 2:4] = np.column_stack((width * height, width / denominator))
        else:
            result[:, 2] = width * height
            score_index = 5 if self.is_obb else 4
            result[:, 3] = boxes[:, score_index] if boxes.shape[1] > score_index else 0.0
            result[:, 4] = width / (height if self.is_obb else height + 1e-6)
        return result

    def to_boxes(self, states: np.ndarray, *, include_score: bool = False) -> np.ndarray:
        """Convert state rows to boxes, optionally retaining XYSCR confidence."""
        values = np.asarray(states, dtype=float)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if self.kind in (MotionModelKind.XYWH,):
            width, height = values[:, 2], values[:, 3]
        elif self.kind is MotionModelKind.XYAH:
            width, height = values[:, 2] * values[:, 3], values[:, 3]
        elif self.kind is MotionModelKind.XYHR:
            height = values[:, 2]
            width = height * values[:, 3]
            if not self.is_obb:
                width = np.where(values[:, 3] <= 0.0, 0.0, width)
        else:
            ratio_index = 4 if self.kind is MotionModelKind.XYSCR else 3
            width = np.sqrt(np.maximum(values[:, 2] * values[:, ratio_index], 1e-12))
            height = values[:, 2] / np.maximum(width, 1e-6)
        if self.is_obb:
            angles = (values[:, self.angle_measurement_index] + np.pi) % (2 * np.pi) - np.pi
            boxes = np.column_stack((values[:, :2], width, height, angles))
            if include_score and self.kind is MotionModelKind.XYSCR:
                boxes = np.column_stack((boxes, values[:, 3]))
            return boxes
        half_size = np.column_stack((width, height)) / 2.0
        boxes = np.concatenate((values[:, :2] - half_size, values[:, :2] + half_size), axis=1)
        if include_score and self.kind is MotionModelKind.XYSCR:
            boxes = np.column_stack((boxes, values[:, 3]))
        return boxes


def _as_vector(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def xyxy_to_xywh_measurement(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in _as_vector(box)[:4])
    w = x2 - x1
    h = y2 - y1
    return np.array([x1 + (w / 2.0), y1 + (h / 2.0), w, h], dtype=float)


def xywha_to_xywh_measurement(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h, theta = (float(v) for v in _as_vector(box)[:5])
    return np.array([cx, cy, max(w, 1e-4), max(h, 1e-4), normalize_angle(theta)], dtype=float)


def xywh_state_to_xyxy(state: np.ndarray, score: float | None = None) -> np.ndarray:
    x, y, w, h = (float(v) for v in _as_vector(state)[:4])
    row = np.array([x - (w / 2.0), y - (h / 2.0), x + (w / 2.0), y + (h / 2.0)], dtype=float)
    if score is None:
        return row.reshape((1, 4))
    return np.r_[row, float(score)].reshape((1, 5))


def xywh_state_to_xywha(state: np.ndarray, score: float | None = None) -> np.ndarray:
    del score
    x, y, w, h, theta = (float(v) for v in _as_vector(state)[:5])
    return np.array([x, y, w, h, normalize_angle(theta)], dtype=float).reshape((1, 5))


def xyxy_to_xyah_measurement(box: np.ndarray) -> np.ndarray:
    x, y, w, h = xyxy_to_xywh_measurement(box)
    h = max(float(h), 1e-6)
    return np.array([x, y, w / h, h], dtype=float)


def xywha_to_xyah_measurement(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h, theta = (float(v) for v in _as_vector(box)[:5])
    h = max(h, 1e-6)
    return np.array([cx, cy, max(w, 1e-6) / h, h, normalize_angle(theta)], dtype=float)


def xyah_state_to_xyxy(state: np.ndarray, score: float | None = None) -> np.ndarray:
    x, y, a, h = (float(v) for v in _as_vector(state)[:4])
    w = a * h
    row = np.array([x - (w / 2.0), y - (h / 2.0), x + (w / 2.0), y + (h / 2.0)], dtype=float)
    if score is None:
        return row.reshape((1, 4))
    return np.r_[row, float(score)].reshape((1, 5))


def xyah_state_to_xywha(state: np.ndarray, score: float | None = None) -> np.ndarray:
    del score
    x, y, a, h, theta = (float(v) for v in _as_vector(state)[:5])
    return np.array([x, y, a * h, h, normalize_angle(theta)], dtype=float).reshape((1, 5))


def xyxy_to_xysr_measurement(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in _as_vector(box)[:4])
    w = x2 - x1
    h = y2 - y1
    return np.array(
        [x1 + (w / 2.0), y1 + (h / 2.0), w * h, w / (h + 1e-6)],
        dtype=float,
    )


def xywha_to_xysr_measurement(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h, theta = (float(v) for v in _as_vector(box)[:5])
    w = max(w, 1e-6)
    h = max(h, 1e-6)
    return np.array([cx, cy, w * h, w / h, normalize_angle(theta)], dtype=float)


def xysr_state_to_xyxy(state: np.ndarray, score: float | None = None) -> np.ndarray:
    x = _as_vector(state)
    w = np.sqrt(max(float(x[2] * x[3]), 1e-12))
    h = float(x[2]) / max(w, 1e-6)
    row = np.array([x[0] - (w / 2.0), x[1] - (h / 2.0), x[0] + (w / 2.0), x[1] + (h / 2.0)], dtype=float)
    if score is None:
        return row.reshape((1, 4))
    return np.r_[row, float(score)].reshape((1, 5))


def xysr_state_to_xywha(state: np.ndarray, score: float | None = None) -> np.ndarray:
    x = _as_vector(state)
    w = np.sqrt(max(float(x[2] * x[3]), 1e-12))
    h = float(x[2]) / max(w, 1e-6)
    theta = normalize_angle(float(x[4]))
    row = np.array([x[0], x[1], w, h, theta], dtype=float)
    if score is None:
        return row.reshape((1, 5))
    return np.r_[row, float(score)].reshape((1, 6))


def xyxy_to_xyhr_measurement(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in _as_vector(box)[:4])
    w = x2 - x1
    h = y2 - y1
    return np.array([x1 + (w / 2.0), y1 + (h / 2.0), h, w / (h + 1e-6)], dtype=float)


def xywha_to_xyhr_measurement(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h, theta = (float(v) for v in _as_vector(box)[:5])
    h = max(h, 1e-4)
    w = max(w, 1e-4)
    return np.array([cx, cy, h, w / h, normalize_angle(theta)], dtype=float)


def xyhr_state_to_xyxy(state: np.ndarray, score: float | None = None) -> np.ndarray:
    x = _as_vector(state)
    h = float(x[2])
    r = float(x[3])
    w = 0.0 if r <= 0.0 else r * h
    row = np.array([x[0] - (w / 2.0), x[1] - (h / 2.0), x[0] + (w / 2.0), x[1] + (h / 2.0)], dtype=float)
    if score is None:
        return row.reshape((1, 4))
    return np.r_[row, float(score)].reshape((1, 5))


def xyhr_state_to_xywha(state: np.ndarray, score: float | None = None) -> np.ndarray:
    del score
    x = _as_vector(state)
    h = float(x[2])
    r = float(x[3])
    return np.array([x[0], x[1], h * r, h, normalize_angle(float(x[4]))], dtype=float).reshape((1, 5))


def xyxy_to_xyscr_measurement(box: np.ndarray) -> np.ndarray:
    values = _as_vector(box)
    x1, y1, x2, y2 = (float(v) for v in values[:4])
    score = float(values[4]) if values.size > 4 else 0.0
    w = x2 - x1
    h = y2 - y1
    return np.array(
        [x1 + (w / 2.0), y1 + (h / 2.0), w * h, score, w / (h + 1e-6)],
        dtype=float,
    )


def xyscr_state_to_xyxy(state: np.ndarray, score: float | None = None) -> np.ndarray:
    x = _as_vector(state)
    w = np.sqrt(max(float(x[2] * x[4]), 1e-12))
    h = float(x[2]) / max(w, 1e-6)
    row = np.array([x[0] - (w / 2.0), x[1] - (h / 2.0), x[0] + (w / 2.0), x[1] + (h / 2.0)], dtype=float)
    if score is None:
        return row.reshape((1, 4))
    return np.r_[row, float(x[3])].reshape((1, 5))


def xywha_to_xyscr_measurement(box: np.ndarray) -> np.ndarray:
    """Retain area, confidence and ratio, then append oriented-box angle."""
    values = _as_vector(box)
    cx, cy, width, height, theta = values[:5]
    width, height = max(width, 1e-6), max(height, 1e-6)
    score = float(values[5]) if values.size > 5 else 0.0
    return np.array([cx, cy, width * height, score, width / height, normalize_angle(theta)], dtype=float)


def xyscr_state_to_xywha(state: np.ndarray, score: float | None = None) -> np.ndarray:
    """Decode oriented geometry without interpreting confidence as angle."""
    values = _as_vector(state)
    width = np.sqrt(max(float(values[2] * values[4]), 1e-12))
    height = float(values[2]) / max(width, 1e-6)
    box = np.array([values[0], values[1], width, height, normalize_angle(float(values[5]))])
    return box.reshape((1, 5)) if score is None else np.r_[box, values[3]].reshape((1, 6))


def create_motion_model(
    kind: MotionModelKind | str,
    *,
    is_obb: bool = False,
    max_obs: int = 50,
    adaptive_kf: bool = False,
    cls_id: int | None = None,
) -> MotionModelAdapter:
    kind = MotionModelKind(kind)
    is_obb = bool(is_obb)
    if kind not in supported_motion_model_kinds(is_obb=is_obb):
        raise ValueError("Unsupported Kalman model for the configured geometry.")
    if adaptive_kf and kind is not MotionModelKind.XYHR:
        raise ValueError("Adaptive Kalman noise requires the XYHR motion model")

    if kind is MotionModelKind.XYAH:
        dim_z = 5 if is_obb else 4
        return MotionModelAdapter(
            kind=kind,
            dim_x=2 * dim_z,
            dim_z=dim_z,
            is_obb=is_obb,
            _filter_factory=lambda _, noise: KalmanFilterXYAH(ndim=dim_z, noise_config=noise),
            _measurement_from_box=xywha_to_xyah_measurement if is_obb else xyxy_to_xyah_measurement,
            _box_from_state=xyah_state_to_xywha if is_obb else xyah_state_to_xyxy,
            max_obs=max_obs,
        )

    if kind in (MotionModelKind.XYWH,):
        dim_z = 5 if is_obb else 4
        filter_class = KalmanFilterXYWH
        return MotionModelAdapter(
            kind=kind,
            dim_x=2 * dim_z,
            dim_z=dim_z,
            is_obb=is_obb,
            _filter_factory=lambda _, noise: filter_class(ndim=dim_z, noise_config=noise),
            _measurement_from_box=xywha_to_xywh_measurement if is_obb else xyxy_to_xywh_measurement,
            _box_from_state=xywh_state_to_xywha if is_obb else xywh_state_to_xyxy,
            max_obs=max_obs,
        )

    if kind is MotionModelKind.XYSR:
        dim_z = 5 if is_obb else 4
        dim_x = 9 if is_obb else 7
        return MotionModelAdapter(
            kind=kind,
            dim_x=dim_x,
            dim_z=dim_z,
            is_obb=is_obb,
            _filter_factory=lambda _, noise: KalmanFilterXYSR(
                dim_x=dim_x, dim_z=dim_z, max_obs=max_obs, noise_config=noise
            ),
            _measurement_from_box=xywha_to_xysr_measurement if is_obb else xyxy_to_xysr_measurement,
            _box_from_state=xysr_state_to_xywha if is_obb else xysr_state_to_xyxy,
            max_obs=max_obs,
        )

    if kind is MotionModelKind.XYHR:
        dim_z = 5 if is_obb else 4
        dim_x = 10 if is_obb else 8
        return MotionModelAdapter(
            kind=kind,
            dim_x=dim_x,
            dim_z=dim_z,
            is_obb=is_obb,
            _filter_factory=lambda measurement, noise: KalmanFilterXYHR(
                measurement,
                ndim=dim_x,
                dim_z=dim_z,
                adaptive_kf=adaptive_kf,
                cls_id=cls_id,
                noise_config=noise,
            ),
            _measurement_from_box=xywha_to_xyhr_measurement if is_obb else xyxy_to_xyhr_measurement,
            _box_from_state=xyhr_state_to_xywha if is_obb else xyhr_state_to_xyxy,
            max_obs=max_obs,
        )

    dim_x, dim_z = (11, 6) if is_obb else (9, 5)
    return MotionModelAdapter(
        kind=kind,
        dim_x=dim_x,
        dim_z=dim_z,
        is_obb=is_obb,
        _filter_factory=lambda _, noise: KalmanFilterXYSCR(
            dim_x=dim_x, dim_z=dim_z, max_obs=max_obs, noise_config=noise
        ),
        _measurement_from_box=xywha_to_xyscr_measurement if is_obb else xyxy_to_xyscr_measurement,
        _box_from_state=xyscr_state_to_xywha if is_obb else xyscr_state_to_xyxy,
        max_obs=max_obs,
    )
