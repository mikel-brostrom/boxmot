"""Tracker motion-model adapters, state conversions, and filter construction."""

from __future__ import annotations

from collections.abc import Callable
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

    def create_filter(
        self, initial_measurement: np.ndarray | None = None, *, noise_config: KalmanNoiseConfig | None = None
    ) -> Any:
        """Create an independent filter with the caller's noise configuration."""
        measurement = None
        if initial_measurement is not None:
            measurement = np.asarray(initial_measurement, dtype=float).reshape(-1)
        return self._filter_factory(measurement, noise_config)

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
            result = boxes[:, :5].copy()
            minimum = 1e-4 if self.kind in (MotionModelKind.XYWH, MotionModelKind.XYHR) else 1e-6
            width = np.maximum(boxes[:, 2], minimum)
            height = np.maximum(boxes[:, 3], minimum)
            result[:, 4] = (result[:, 4] + np.pi) % (2 * np.pi) - np.pi
        else:
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            result = np.empty((len(boxes), self.dim_z), dtype=float)
            result[:, :2] = boxes[:, :2] + np.column_stack((width, height)) / 2.0
        if self.kind is MotionModelKind.XYWH:
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
            result[:, 3] = boxes[:, 4] if boxes.shape[1] > 4 else 0.0
            result[:, 4] = width / (height + 1e-6)
        return result

    def to_boxes(self, states: np.ndarray, *, include_score: bool = False) -> np.ndarray:
        """Convert state rows to boxes, optionally retaining XYSCR confidence."""
        values = np.asarray(states, dtype=float)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if self.kind is MotionModelKind.XYWH:
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
            angles = (values[:, 4] + np.pi) % (2 * np.pi) - np.pi
            return np.column_stack((values[:, :2], width, height, angles))
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
        )

    if kind is MotionModelKind.XYWH:
        dim_z = 5 if is_obb else 4
        return MotionModelAdapter(
            kind=kind,
            dim_x=2 * dim_z,
            dim_z=dim_z,
            is_obb=is_obb,
            _filter_factory=lambda _, noise: KalmanFilterXYWH(ndim=dim_z, noise_config=noise),
            _measurement_from_box=xywha_to_xywh_measurement if is_obb else xyxy_to_xywh_measurement,
            _box_from_state=xywh_state_to_xywha if is_obb else xywh_state_to_xyxy,
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
        )

    if is_obb:
        raise ValueError("XYSCR does not support OBB state")
    return MotionModelAdapter(
        kind=kind,
        dim_x=9,
        dim_z=5,
        is_obb=False,
        _filter_factory=lambda _, noise: KalmanFilterXYSCR(max_obs=max_obs, noise_config=noise),
        _measurement_from_box=xyxy_to_xyscr_measurement,
        _box_from_state=xyscr_state_to_xyxy,
    )
