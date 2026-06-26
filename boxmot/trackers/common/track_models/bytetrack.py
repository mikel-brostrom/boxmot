# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import numpy as np

from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.geometry import tlwh2xyah, xywh2tlwh, xyxy2xywh
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.common.track_models.base import BoxTrack


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(BoxTrack):
    state = TrackState.New
    lost_state = TrackState.Lost
    removed_state = TrackState.Removed
    tracked_state = TrackState.Tracked

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            super().__init__(*args, **kwargs)


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()
    shared_kalman_obb = KalmanFilterXYWH(ndim=5)

    def __init__(self, det, max_obs, id_allocator: TrackIdAllocator, is_obb: bool = False):
        super().__init__(det, id_allocator=id_allocator, max_obs=max_obs, is_obb=is_obb)

    def _init_from_aabb_detection(self, det: np.ndarray) -> None:
        self.xywh = xyxy2xywh(det[0:4])
        self.tlwh = xywh2tlwh(self.xywh)
        self.xyah = tlwh2xyah(self.tlwh)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

    def _measurement_for_update(self, track: "STrack") -> np.ndarray:
        return track.xywh if track.is_obb else track.xyah

    @classmethod
    def _reset_inactive_aabb_motion(cls, mean_state: np.ndarray) -> None:
        mean_state[7] = 0

    def _current_aabb_xywh(self) -> np.ndarray:
        if self.mean is None:
            return self.xywh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        return ret
