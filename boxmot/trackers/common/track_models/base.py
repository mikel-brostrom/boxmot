from __future__ import annotations

from collections import deque
from typing import ClassVar

import numpy as np

from boxmot.trackers.common.geometry import xywh2xyxy
from boxmot.trackers.common.geometry.obb import smooth_obb_corners, xywha_to_xyxy
from boxmot.trackers.common.tracking.track import (
    TrackIdAllocator,
    TrackLifecycleMixin,
    sync_track_meta,
)
from boxmot.trackers.common.tracking.track import (
    TrackState as CommonTrackState,
)


class BoxTrack(TrackLifecycleMixin):
    """Common lifecycle and geometry substrate for bbox track objects.

    Algorithm-specific subclasses still own their motion measurement format and
    optional appearance/class-history updates.
    """

    tracked_state: ClassVar[object | None] = None
    common_tracked_state: ClassVar[CommonTrackState] = CommonTrackState.TRACKED

    def __init__(
        self,
        det,
        *,
        id_allocator: TrackIdAllocator,
        max_obs: int,
        is_obb: bool = False,
    ) -> None:
        self.id_allocator = id_allocator
        self.is_obb = bool(is_obb)
        det = np.asarray(det, dtype=np.float32)
        if self.is_obb:
            self._init_from_obb_detection(det)
        else:
            self._init_from_aabb_detection(det)

        self.max_obs = int(max_obs)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
        self.history_observations = deque([], maxlen=self.max_obs)
        self._plot_angle = None

    def _init_from_aabb_detection(self, det: np.ndarray) -> None:
        raise NotImplementedError

    def _init_from_obb_detection(self, det: np.ndarray) -> None:
        self.xywh = det[:5].copy()
        self.conf = det[5]
        self.cls = det[6]
        self.det_ind = det[7]

    def _measurement_for_update(self, track: "BoxTrack"):
        raise NotImplementedError

    def _current_aabb_xywh(self) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _reset_inactive_aabb_motion(cls, mean_state: np.ndarray) -> None:
        mean_state[6:8] = 0

    @classmethod
    def _local_tracked_state(cls):
        if cls.tracked_state is None:
            raise NotImplementedError(f"{cls.__name__} must define tracked_state")
        return cls.tracked_state

    @staticmethod
    def _copy_detection_metadata(dst, src) -> None:
        dst.conf = src.conf
        dst.cls = src.cls
        dst.det_ind = src.det_ind

    def _after_reactivate(self, new_track: "BoxTrack") -> None:
        pass

    def _after_update(self, new_track: "BoxTrack") -> None:
        pass

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != self._local_tracked_state():
            if self.is_obb:
                mean_state[7:10] = 0
            else:
                self._reset_inactive_aabb_motion(mean_state)
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state,
            self.covariance,
        )

    @classmethod
    def multi_predict(cls, stracks):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        is_obb = getattr(stracks[0], "is_obb", False)
        for i, st in enumerate(stracks):
            if st.state != cls._local_tracked_state():
                if is_obb:
                    multi_mean[i][7:10] = 0
                else:
                    cls._reset_inactive_aabb_motion(multi_mean[i])
        kalman = cls.shared_kalman_obb if is_obb else cls.shared_kalman
        multi_mean, multi_covariance = kalman.multi_predict(
            multi_mean,
            multi_covariance,
        )
        for st, mean, cov in zip(stracks, multi_mean, multi_covariance):
            st.mean, st.covariance = mean, cov

    def activate(self, kalman_filter, frame_id):
        """Activate a new track."""
        self.kalman_filter = kalman_filter
        self.id = self.id_allocator.alloc()
        self.mean, self.covariance = self.kalman_filter.initiate(self._measurement_for_update(self))
        self.tracklet_len = 0
        self.state = self._local_tracked_state()
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        sync_track_meta(self, self.common_tracked_state)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a track with a new detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self._measurement_for_update(new_track),
        )
        self.tracklet_len = 0
        self.state = self._local_tracked_state()
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.id_allocator.alloc()
        self._copy_detection_metadata(self, new_track)
        self._after_reactivate(new_track)
        sync_track_meta(self, self.common_tracked_state)

    def update(self, new_track, frame_id):
        """Update the current track with a matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        if not self.is_obb:
            self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self._measurement_for_update(new_track),
        )
        self.state = self._local_tracked_state()
        self.is_activated = True
        self._copy_detection_metadata(self, new_track)
        self._after_update(new_track)
        if self.is_obb:
            self.history_observations.append(self._state_obb_for_plot())
        sync_track_meta(self, self.common_tracked_state)

    def _state_obb_for_plot(self) -> np.ndarray:
        """Return post-update OBB state as corners with state-only angle smoothing."""
        corners, self._plot_angle = smooth_obb_corners(self.xywha, self._plot_angle)
        return corners

    @classmethod
    def obb_to_xyxy(cls, box: np.ndarray) -> np.ndarray:
        """Return the enclosing AABB for an OBB box."""
        return xywha_to_xyxy(np.asarray(box, dtype=np.float32).reshape(1, 5))[0]

    @property
    def xyxy(self) -> np.ndarray:
        """Return the current track box as ``(x1, y1, x2, y2)``."""
        if self.is_obb:
            return self.obb_to_xyxy(self.xywha)
        return xywh2xyxy(self._current_aabb_xywh())

    @property
    def xywha(self) -> np.ndarray:
        """Return the current track box as ``(cx, cy, w, h, angle)``."""
        if self.is_obb:
            ret = self.mean[:5].copy() if self.mean is not None else self.xywh.copy()
            return np.asarray(ret, dtype=np.float32)
        xywh = self._current_aabb_xywh()
        return np.array([xywh[0], xywh[1], xywh[2], xywh[3], 0.0], dtype=np.float32)


class SortBoxTrack(TrackLifecycleMixin):
    """Common substrate for SORT-style KalmanBoxTracker implementations."""

    common_initial_state: ClassVar[CommonTrackState] = CommonTrackState.TENTATIVE

    def _assign_sort_id(
        self,
        *,
        id_allocator: TrackIdAllocator | None,
        track_id: int | None = None,
    ) -> None:
        if track_id is not None:
            self.id = int(track_id)
            return
        if id_allocator is None:
            raise ValueError(f"{self.__class__.__name__} requires an instance TrackIdAllocator")
        self.id = id_allocator.alloc()

    def _init_sort_counters(self, *, max_obs: int) -> None:
        self.max_obs = int(max_obs)
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _sync_initial_sort_meta(self) -> None:
        sync_track_meta(self, self.common_initial_state)
