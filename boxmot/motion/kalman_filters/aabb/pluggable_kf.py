# box_kalman_filter.py

from typing import List, Tuple, Union, Type
import numpy as np

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter
from boxmot.motion.kalman_filters.aabb.xyah_kf import KalmanFilterXYAH
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH

class BoxKalmanFilter:
    """
    Wraps a BaseKalmanFilter subclass so that it takes/returns
    detections and boxes in [xmin, ymin, xmax, ymax] (xyxy) format,
    while still exposing the full Kalman API (.initiate, .predict,
    .project, .update, .gating_distance, etc.).
    """

    _EPS = 1e-6

    def __init__(self, filter_type: Union[str, Type[BaseKalmanFilter]] = "xyah"):
        # Choose your filter backend
        if filter_type == "xyah" or filter_type is KalmanFilterXYAH:
            self.kf: BaseKalmanFilter = KalmanFilterXYAH()
            self._to_meas = self._xyxy_to_xyah
            self._to_box = self._xyah_to_xyxy
        elif filter_type == "xywh" or filter_type is KalmanFilterXYWH:
            self.kf = KalmanFilterXYWH()
            self._to_meas = self._xyxy_to_xywh
            self._to_box = self._xywh_to_xyxy
        else:
            raise ValueError(
                "filter_type must be 'xyah', 'xywh', KalmanFilterXYAH or KalmanFilterXYWH"
            )

    # ──────────────────────────────────────────────────────────────────────────────
    # Converters between xyxy and the filter’s internal measurement format
    # with safety clamps to avoid division by zero.
    # ──────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(val: float, eps: float = _EPS) -> float:
        return val if val > eps else eps

    @classmethod
    def _xyxy_to_xyah(cls, box: np.ndarray) -> np.ndarray:
        xmin, ymin, xmax, ymax = box
        w = cls._clamp(xmax - xmin)
        h = cls._clamp(ymax - ymin)
        cx = xmin + w / 2
        cy = ymin + h / 2
        a = w / h
        return np.array([cx, cy, a, h], dtype=float)

    @classmethod
    def _xyah_to_xyxy(cls, meas: np.ndarray) -> np.ndarray:
        cx, cy, a, h = meas
        h = cls._clamp(h)
        w = cls._clamp(a * h)
        xmin = cx - w / 2
        ymin = cy - h / 2
        return np.array([xmin, ymin, xmin + w, ymin + h], dtype=float)

    @classmethod
    def _xyxy_to_xywh(cls, box: np.ndarray) -> np.ndarray:
        xmin, ymin, xmax, ymax = box
        w = cls._clamp(xmax - xmin)
        h = cls._clamp(ymax - ymin)
        cx = xmin + w / 2
        cy = ymin + h / 2
        return np.array([cx, cy, w, h], dtype=float)

    @classmethod
    def _xywh_to_xyxy(cls, meas: np.ndarray) -> np.ndarray:
        cx, cy, w, h = meas
        w = cls._clamp(w)
        h = cls._clamp(h)
        xmin = cx - w / 2
        ymin = cy - h / 2
        return np.array([xmin, ymin, xmin + w, ymin + h], dtype=float)

    # ──────────────────────────────────────────────────────────────────────────────
    # Core API: mirror the BaseKalmanFilter interface
    # ──────────────────────────────────────────────────────────────────────────────
        

    def initiate(self, box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for .initiate_from_box(), so you can call .initiate(box_xyxy).
        """
        return self.kf.initiate(box)

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict one step ahead in state space.
        """
        return self.kf.predict(mean, covariance)

    def multi_predict(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> List[np.ndarray]:
        """
        Just mirror BaseKalmanFilter.multi_predict:
        """
        return self.kf.multi_predict(means, covariances)

    def project(
        self, mean: np.ndarray, covariance: np.ndarray, confidence: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for projecting into measurement space
        (same as .projected_measurement).
        """
        return self.project(mean, covariance, confidence)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct the track state with a new detection (xyxy).
        """
        return self.kf.update(mean, covariance, measurement, confidence)
        

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between a single track (mean,cov)
        and a batch of detections (xyxy). Returns one distance per box.
        """
        return self.kf.gating_distance(mean, covariance, measurements, only_position, metric)