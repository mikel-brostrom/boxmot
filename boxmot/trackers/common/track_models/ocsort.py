# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""

from collections import deque

import numpy as np

from boxmot.trackers.common.geometry.obb import smooth_obb_corners
from boxmot.trackers.common.motion import MotionModelKind, create_motion_model
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState, sync_track_meta
from boxmot.trackers.common.track_models.base import SortBoxTrack


def k_previous_obs(observations, cur_age, k, is_obb=False):
    if len(observations) == 0:
        if is_obb:
            return [-1, -1, -1, -1, -1, -1]
        else:
            return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_obb(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(SortBoxTrack):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    def __init__(
        self,
        bbox,
        cls,
        det_ind,
        delta_t=3,
        max_obs=50,
        Q_xy_scaling=0.01,
        Q_s_scaling=0.0001,
        is_obb=False,
        Q_a_scaling=0.0001,
        id_allocator: TrackIdAllocator | None = None,
    ):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        self.det_ind = det_ind

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling
        self.Q_a_scaling = Q_a_scaling
        self.is_obb = is_obb
        self.motion_model = create_motion_model(
            MotionModelKind.XYSR,
            is_obb=self.is_obb,
            max_obs=max_obs,
        )

        if self.is_obb:
            self.kf = self.motion_model.create_filter()
            self.kf.F = np.array(
                [
                    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # x += vx
                    [0, 1, 0, 0, 0, 0, 1, 0, 0],  # y += vy
                    [0, 0, 1, 0, 0, 0, 0, 1, 0],  # s += vs
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # r static
                    [0, 0, 0, 0, 1, 0, 0, 0, 1],  # theta += vtheta
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],  # y
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # s
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # r
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # theta
                ]
            )
            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[5:, 5:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.0

            self.kf.Q[5:7, 5:7] *= self.Q_xy_scaling
            self.kf.Q[7, 7] *= self.Q_s_scaling
            self.kf.Q[8, 8] *= self.Q_a_scaling
            self.kf.x[:5] = self.motion_model.to_measurement(bbox[:5])
        else:
            self.kf = self.motion_model.create_filter()
            self.kf.F = np.array(
                [
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )

            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.0

            self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling
            self.kf.Q[-1, -1] *= self.Q_s_scaling
            self.kf.x[:4] = self.motion_model.to_measurement(bbox)
        self._assign_sort_id(id_allocator=id_allocator)
        self._init_sort_counters(max_obs=max_obs)
        self.history = deque([], maxlen=self.max_obs)
        self.conf = bbox[-1]
        self.cls = cls
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1, -1]) if self.is_obb else np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity = None
        self.delta_t = delta_t
        self._plot_angle = None
        self._sync_initial_sort_meta()

    def _state_obb_for_plot(self) -> np.ndarray:
        """Return current OBB state as corners with state-only angle smoothing."""
        box = self.motion_model.to_box(self.kf.x)[0].astype(np.float32)
        corners, self._plot_angle = smooth_obb_corners(box, self._plot_angle)
        return corners

    def update(self, bbox, cls, det_ind):
        """
        Updates the state vector with observed bbox.
        """
        self.det_ind = det_ind
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                # Estimate the track speed direction with observations Δt steps away
                if self.is_obb:
                    self.velocity = speed_direction_obb(previous_box, bbox)
                else:
                    self.velocity = speed_direction(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            if self.is_obb:
                self.kf.update(self.motion_model.to_measurement(bbox[:5]))
                self.history_observations.append(self._state_obb_for_plot())
            else:
                self.kf.update(self.motion_model.to_measurement(bbox))
                self.history_observations.append(bbox)
            sync_track_meta(self, TrackState.TRACKED)
        else:
            self.kf.update(bbox)
            sync_track_meta(self)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.is_obb:
            if (self.kf.x[7] + self.kf.x[2]) <= 0:
                self.kf.x[7] *= 0.0
        else:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        if self.is_obb:
            self.history.append(self.motion_model.to_box(self.kf.x))
        else:
            self.history.append(self.motion_model.to_box(self.kf.x))
        sync_track_meta(self)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.motion_model.to_box(self.kf.x)
