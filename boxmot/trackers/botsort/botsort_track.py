from __future__ import annotations

"""STrack refactored for better readability and single‑responsibility.

Key ideas
---------
1. **Detection** – lightweight dataclass for raw detector output.
2. **ClassificationHistory** – confidence‑weighted running majority vote.
3. **FeatureBuffer** – appearance history with exponential smoothing.
4. **STrack** – orchestration layer that plugs the helpers together
   and wraps an AMSKalmanFilterXYWH instance.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Union

import numpy as np

from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh
from boxmot.motion.kalman_filters.aabb.xywh_kf import AMSKalmanFilterXYWH


# ---------------------------------------------------------------------------
#  Helper data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Detection:
    """Container for a single detection in `(x1, y1, x2, y2)` format."""

    xyxy: np.ndarray
    conf: float
    cls_id: int
    det_idx: int

    # -----------------------------------------------------------
    def to_xywh(self) -> np.ndarray:
        """Convert `(x1, y1, x2, y2)` to `(xc, yc, w, h)`."""
        return xyxy2xywh(self.xyxy)


class ClassificationHistory:
    """Running class tracker using confidence‑weighted voting."""

    def __init__(self) -> None:
        self._hist: dict[int, float] = {}

    # -----------------------------------------------------------
    def update(self, cls_id: int, conf: float) -> int:
        self._hist[cls_id] = self._hist.get(cls_id, 0.0) + conf
        return max(self._hist, key=self._hist.get, default=cls_id)


class FeatureBuffer:
    """Stores and exponentially‑smooths appearance features."""

    def __init__(self, history: int = 50, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self._buf: Deque[np.ndarray] = deque(maxlen=history)
        self.smooth: Optional[np.ndarray] = None

    # -----------------------------------------------------------
    def update(self, feat: np.ndarray) -> None:
        feat = feat / np.linalg.norm(feat)
        if self.smooth is None:
            self.smooth = feat
        else:
            self.smooth = self.alpha * self.smooth + (1.0 - self.alpha) * feat
            self.smooth /= np.linalg.norm(self.smooth)
        self._buf.append(feat)

    # -----------------------------------------------------------
    @property
    def latest(self) -> Optional[np.ndarray]:
        return self._buf[-1] if self._buf else None


# ---------------------------------------------------------------------------
#  STrack (single object track)
# ---------------------------------------------------------------------------


class STrack(BaseTrack):
    """Single‑object track with appearance smoothing and AMS Kalman filtering."""

    _shared_kf = AMSKalmanFilterXYWH()

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        detection: Union[Detection, np.ndarray],
        feat: Optional[np.ndarray] = None,
        *,
        feat_history: int = 50,
        max_obs: int = 50,
    ) -> None:
        super().__init__()

        # Raw detection ---------------------------------------------------
        if isinstance(detection, np.ndarray):
            if detection.shape[0] < 7:
                raise ValueError("Detection array must have length 7: x1, y1, x2, y2, conf, cls_id, det_idx")
            detection = Detection(detection[:4], float(detection[4]), int(detection[5]), int(detection[6]))
        self.xywh = detection.to_xywh()
        self.conf = detection.conf
        self.cls_id = detection.cls_id
        self.det_idx = detection.det_idx

        # Rolling buffers -------------------------------------------------
        self.features = FeatureBuffer(feat_history)
        self.cls_hist = ClassificationHistory()
        self._obs_buf: Deque[np.ndarray] = deque(maxlen=max_obs)

        # Kalman filter state --------------------------------------------
        self.kf: Optional[AMSKalmanFilterXYWH] = None
        self.mean: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None

        # Lifecycle flags -------------------------------------------------
        self.is_activated = False
        self.track_len = 0

        # Optional appearance / class updates ----------------------------
        if feat is not None:
            self.features.update(feat)
        self.cls_id = self.cls_hist.update(self.cls_id, self.conf)

    # ---------------------------------------------------------------- predictions
    def predict(self) -> None:
        """One‑step motion prediction."""
        if self.mean is None or self.cov is None or self.kf is None:
            return
        mean = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean[6:8] = 0.0  # freeze velocity for lost / tentative tracks
        self.mean, self.cov = self.kf.predict(mean, self.cov)

    @staticmethod
    def batch_predict(tracks: list["STrack"]) -> None:
        """Vectorised prediction for a list of tracks."""
        if not tracks:
            return
        means = np.asarray([t.mean.copy() for t in tracks])
        covs = np.asarray([t.cov for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                means[i, 6:8] = 0.0
        means, covs = STrack._shared_kf.multi_predict(means, covs)
        for t, m, c in zip(tracks, means, covs):
            t.mean, t.cov = m, c

    # ------------------------------------------------------------------ alias for backward‑compat
    multi_predict = batch_predict

    # ---------------------------------------------------------------- GMC for legacy code
    @staticmethod
    def multi_gmc(stracks: list["STrack"], H: np.ndarray = np.eye(2, 3)) -> None:
        """Apply geometric motion compensation (legacy helper)."""
        if not stracks:
            return
        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4), R)
        t = H[:2, 2]
        for st in stracks:
            if st.mean is None or st.cov is None:
                continue
            mean = R8x8.dot(st.mean)
            mean[:2] += t
            st.mean = mean
            st.cov = R8x8.dot(st.cov).dot(R8x8.T)

    # ---------------------------------------------------------------- KF updates
    def activate(self, kalman_filter: AMSKalmanFilterXYWH, frame_id: int) -> None:
        self.kf = kalman_filter
        self.id = self.next_id()
        self.mean, self.cov = self.kf.initiate(self.xywh)
        self.state = TrackState.Tracked
        self.frame_id = self.start_frame = frame_id
        self.is_activated = frame_id == 1

    def update(self, detection: "STrack", frame_id: int) -> None:
        """Update track with matched detection."""
        assert self.kf is not None
        self.frame_id = frame_id
        self.track_len += 1
        self._obs_buf.append(self.xyxy)

        self.mean, self.cov = self.kf.update(self.mean, self.cov, detection.xywh)
        if detection.features.latest is not None:
            self.features.update(detection.features.latest)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.conf = detection.conf
        self.cls_id = self.cls_hist.update(detection.cls_id, detection.conf)
        self.det_idx = detection.det_idx

    def reactivate(self, detection: "STrack", frame_id: int, new_id: bool = False) -> None:
        """Re‑activate a lost track with fresh detection."""
        assert self.kf is not None
        self.mean, self.cov = self.kf.update(self.mean, self.cov, detection.xywh)
        if detection.features.latest is not None:
            self.features.update(detection.features.latest)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = detection.conf
        self.cls_id = self.cls_hist.update(detection.cls_id, detection.conf)
        self.det_idx = detection.det_idx

    # ---------------------------------------------------------------- geometry
    @property
    def xyxy(self) -> np.ndarray:
        """Bounding box in `(x1, y1, x2, y2)` format."""  # noqa: D401,E501
        if self.mean is not None:
            return xywh2xyxy(self.mean[:4])
        return xywh2xyxy(self.xywh)
