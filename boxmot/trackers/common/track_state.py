from __future__ import annotations

from collections import deque
from typing import ClassVar

import numpy as np

from boxmot.trackers.common.geometry import xywh2xyxy
from boxmot.trackers.common.geometry.obb import smooth_obb_corners, xywha_to_xyxy
from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter
from boxmot.trackers.common.motion.models import MotionModelAdapter, MotionModelKind
from boxmot.trackers.common.tracking.track import (
    TrackIdAllocator,
    TrackLifecycleMixin,
    sync_track_meta,
)
from boxmot.trackers.common.tracking.track import (
    TrackState as CommonTrackState,
)


class BoxTrack(TrackLifecycleMixin):
    """Common lifecycle and geometry substrate for box-track objects.

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
        motion_model: MotionModelAdapter,
    ) -> None:
        self.id_allocator = id_allocator
        self.is_obb = bool(is_obb)
        self.motion_model = motion_model
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

    def _reset_inactive_aabb_motion(self, mean_state: np.ndarray) -> None:
        mean_state[:] = self.motion_model.stop_size_velocity(mean_state)

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

    def predict(self, *, dt: float | None = None) -> None:
        """Propagate this track over an optional elapsed time interval."""
        mean_state = self.mean.copy()
        if self.state != self._local_tracked_state():
            if self.is_obb:
                mean_state = self.motion_model.stop_size_velocity(mean_state, include_angle=True)
            else:
                self._reset_inactive_aabb_motion(mean_state)
        self.mean, self.covariance = self.motion_model.predict(
            self.kalman_filter,
            mean_state,
            self.covariance,
            dt=dt,
        )

    @classmethod
    def multi_predict(cls, stracks, *, dt: float | None = None) -> None:
        """Batch shared filters and predict independently owned histories directly."""
        groups: dict[BaseKalmanFilter, list[BoxTrack]] = {}
        for track in stracks:
            groups.setdefault(track.kalman_filter, []).append(track)
        for kalman, tracks in groups.items():
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])
            for index, track in enumerate(tracks):
                if track.state != cls._local_tracked_state():
                    if track.is_obb:
                        multi_mean[index] = track.motion_model.stop_size_velocity(multi_mean[index], include_angle=True)
                    else:
                        track._reset_inactive_aabb_motion(multi_mean[index])
            multi_mean, multi_covariance = tracks[0].motion_model.multi_predict(
                kalman, multi_mean, multi_covariance, dt=dt
            )
            for track, mean, covariance in zip(tracks, multi_mean, multi_covariance):
                track.mean, track.covariance = mean, covariance

    def activate(self, kalman_filter, frame_id):
        """Activate a new track."""
        self.kalman_filter = kalman_filter
        self.id = self.id_allocator.alloc()
        self.mean, self.covariance = self.motion_model.initiate(
            self.kalman_filter, self._measurement_for_update(self), confidence=float(self.conf)
        )
        self.tracklet_len = 0
        self.state = self._local_tracked_state()
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self._append_current_history()
        sync_track_meta(self, self.common_tracked_state)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a track with a new detection."""
        self.mean, self.covariance = self.motion_model.update(
            self.kalman_filter,
            self.mean,
            self.covariance,
            self._measurement_for_update(new_track),
            confidence=0.0,
        )
        self._finish_reactivate(new_track, frame_id, new_id)

    def _finish_reactivate(self, new_track: "BoxTrack", frame_id: int, new_id: bool) -> None:
        """Apply lifecycle bookkeeping after scalar or batched correction."""
        self.tracklet_len = 0
        self.state = self._local_tracked_state()
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.id_allocator.alloc()
        self._copy_detection_metadata(self, new_track)
        self._after_reactivate(new_track)
        self._append_current_history()
        sync_track_meta(self, self.common_tracked_state)

    def update(self, new_track, frame_id):
        """Update the current track with a matched detection."""
        self.mean, self.covariance = self.motion_model.update(
            self.kalman_filter,
            self.mean,
            self.covariance,
            self._measurement_for_update(new_track),
            confidence=0.0,
        )
        self._finish_update(new_track, frame_id)

    def _finish_update(self, new_track: "BoxTrack", frame_id: int) -> None:
        """Store the corrected geometry, detection metadata, and lifecycle."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = self._local_tracked_state()
        self.is_activated = True
        self._copy_detection_metadata(self, new_track)
        self._after_update(new_track)
        self._append_current_history()
        sync_track_meta(self, self.common_tracked_state)

    @classmethod
    def multi_update(cls, pairs, frame_id: int, *, reactivate: bool = False, new_id: bool = False) -> None:
        """Correct shared batches or owned states before ordered lifecycle hooks."""
        pairs = list(pairs)
        groups: dict[BaseKalmanFilter, list[tuple[BoxTrack, BoxTrack]]] = {}
        for track, detection in pairs:
            groups.setdefault(track.kalman_filter, []).append((track, detection))
        for kalman, matches in groups.items():
            mean = np.asarray([track.mean for track, _ in matches])
            covariance = np.asarray([track.covariance for track, _ in matches])
            measurement = np.asarray([track._measurement_for_update(detection) for track, detection in matches])
            model = matches[0][0].motion_model
            mean, covariance = model.multi_update(kalman, mean, covariance, measurement, confidence=0.0)
            for (track, _), state, uncertainty in zip(matches, mean, covariance):
                track.mean, track.covariance = state, uncertainty
        # Keep original association order for ID allocation and lifecycle hooks.
        for track, detection in pairs:
            if reactivate:
                track._finish_reactivate(detection, frame_id, new_id)
            else:
                track._finish_update(detection, frame_id)

    def _append_current_history(self) -> None:
        """Store the post-transition geometry used by trajectory rendering."""
        geometry = self._state_obb_for_plot() if self.is_obb else self.xyxy.copy()
        self.history_observations.append(np.asarray(geometry, dtype=np.float32).copy())

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
            if self.mean is not None and self.motion_model.kind is not MotionModelKind.XYWH:
                ret = self.motion_model.to_box(self.mean)[0]
            else:
                ret = self.mean[:5].copy() if self.mean is not None else self.xywh.copy()
            return np.asarray(ret, dtype=np.float32)
        xywh = self._current_aabb_xywh()
        return np.array([xywh[0], xywh[1], xywh[2], xywh[3], 0.0], dtype=np.float32)


class SortBoxTrack(TrackLifecycleMixin):
    """Shared identity, counters, and metadata for SORT-style per-object tracks."""

    common_initial_state: ClassVar[CommonTrackState] = CommonTrackState.TENTATIVE
    observation_centric: ClassVar[bool] = False

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
