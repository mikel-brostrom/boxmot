from __future__ import annotations

# Hybrid-SORT-ReID with ECC + ReID (explicit config, BaseTracker-style)
# - Assumes detection input is M x [x1, y1, x2, y2, conf, cls]
# - ECC via shared CMC factory and BaseTracker.apply_cmc(...)
# - ReID via pre-built backend passed as ``reid_model``
# - update(dets, img, embs=None) signature compatible with BoxMOT trackers
# - Emits rows: [x1,y1,x2,y2, track_id, conf, cls, det_ind]
# - Preserves detector class IDs and det_ind; guards out-of-range indices
from collections import deque
from typing import Optional

import numpy as np

from boxmot.trackers.common.appearance import (
    blend_embeddings,
    ema_update_embedding,
    normalize_embedding,
)
from boxmot.trackers.common.motion import MotionModelKind, create_motion_model
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState, sync_track_meta
from boxmot.trackers.common.track_models.base import SortBoxTrack


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(SortBoxTrack):
    """
    Single-object tracker with 9D custom KF (u,v,s,c,r, du,dv,ds,dc) by default.
    Stores `cls` and `det_ind` metadata from the most recent matched detection.
    """

    def __init__(
        self,
        bbox,
        temp_feat,
        *,
        delta_t: int = 3,
        longterm_bank_length: int = 30,
        max_obs: int = 50,
        alpha: float = 0.9,
        adapfs: bool = False,
        track_thresh: float = 0.5,
        cls: int = 0,
        det_ind: int = -1,
        id_allocator: TrackIdAllocator | None = None,
    ):
        self.motion_model = create_motion_model(MotionModelKind.XYSCR, max_obs=max_obs)
        self.kf = self.motion_model.create_filter()
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[5:, 5:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01
        self.kf.x[:5] = self.motion_model.to_measurement(bbox)

        # tracker state
        self._assign_sort_id(id_allocator=id_allocator)
        self._init_sort_counters(max_obs=max(1, int(max_obs)))
        self.history = deque([], maxlen=self.max_obs)

        # observations
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.last_observation_save = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)

        # velocity aids
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None

        # parameters
        self.delta_t = int(delta_t)
        self.confidence_pre = None
        self.conf = float(bbox[-1])

        # ReID buffers
        self.smooth_feat = None
        self.features = deque([], maxlen=int(longterm_bank_length))
        self.alpha = float(alpha)
        self.adapfs = bool(adapfs)
        self.track_thresh = float(track_thresh)

        # metadata
        self.cls = int(cls)
        self.det_ind = int(det_ind)

        # first feature update
        self.update_features(temp_feat)
        self._sync_initial_sort_meta()

    def _prune_observations(self) -> None:
        cutoff = self.age - self.max_obs + 1
        for obs_age in list(self.observations):
            if obs_age < cutoff:
                self.observations.pop(obs_age, None)

    def update_features(self, feat, score: float = -1.0):
        feat = normalize_embedding(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if self.adapfs:
                assert score > 0, "score must be > 0 when adapfs=True"
                pre_w = self.alpha * (self.conf / (self.conf + score))
                cur_w = (1.0 - self.alpha) * (score / (self.conf + score))
                s = pre_w + cur_w
                pre_w /= s
                cur_w /= s
                self.smooth_feat = blend_embeddings(self.smooth_feat, feat, pre_w, cur_w)
            else:
                self.smooth_feat = ema_update_embedding(
                    self.smooth_feat,
                    feat,
                    alpha=self.alpha,
                )
        self.features.append(feat)

    def camera_update(self, warp_matrix):
        # get box + score from KF state
        x1, y1, x2, y2, score = self.motion_model.to_box(self.kf.x, score=1.0)[0]

        M = np.asarray(warp_matrix, dtype=float)
        # normalize to 3x3 homogeneous matrix
        if M.shape == (2, 3):
            M = np.vstack([M, [0.0, 0.0, 1.0]])
        elif M.shape != (3, 3):
            M = np.eye(3, dtype=float)

        # transform corners in homogeneous coords
        p1 = (M @ np.array([x1, y1, 1.0], dtype=float)).ravel()
        p2 = (M @ np.array([x2, y2, 1.0], dtype=float)).ravel()

        # homogeneous divide
        w1 = p1[2] if abs(p1[2]) > 1e-12 else 1.0
        w2 = p2[2] if abs(p2[2]) > 1e-12 else 1.0
        x1_, y1_ = p1[0] / w1, p1[1] / w1
        x2_, y2_ = p2[0] / w2, p2[1] / w2

        # write back to KF (keep score)
        self.kf.x[:5] = self.motion_model.to_measurement([x1_, y1_, x2_, y2_, float(score)])

    def update(
        self,
        bbox,
        id_feature,
        update_feature: bool = True,
        *,
        cls: Optional[int] = None,
        det_ind: Optional[int] = None,
    ):
        vlt = vrt = vlb = vrb = None
        if bbox is not None:
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if vlt is not None:
                            vlt += speed_direction_lt(previous_box, bbox)
                            vrt += speed_direction_rt(previous_box, bbox)
                            vlb += speed_direction_lb(previous_box, bbox)
                            vrb += speed_direction_rb(previous_box, bbox)
                        else:
                            vlt = speed_direction_lt(previous_box, bbox)
                            vrt = speed_direction_rt(previous_box, bbox)
                            vlb = speed_direction_lb(previous_box, bbox)
                            vrb = speed_direction_rb(previous_box, bbox)
                if previous_box is None:
                    previous_box = self.last_observation
                    self.velocity_lt = speed_direction_lt(previous_box, bbox)
                    self.velocity_rt = speed_direction_rt(previous_box, bbox)
                    self.velocity_lb = speed_direction_lb(previous_box, bbox)
                    self.velocity_rb = speed_direction_rb(previous_box, bbox)
                else:
                    self.velocity_lt, self.velocity_rt = vlt, vrt
                    self.velocity_lb, self.velocity_rb = vlb, vrb

            self.last_observation = bbox
            self.last_observation_save = bbox
            self.observations[self.age] = bbox
            self._prune_observations()
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history.clear()
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(self.motion_model.to_measurement(bbox))

            # update metadata
            if cls is not None:
                self.cls = int(cls)
            if det_ind is not None:
                self.det_ind = int(det_ind)

            if update_feature:
                if self.adapfs:
                    self.update_features(id_feature, score=bbox[-1])
                else:
                    self.update_features(id_feature)
            self.confidence_pre = self.conf
            self.conf = float(bbox[-1])
            sync_track_meta(self, TrackState.TRACKED)
        else:
            self.kf.update(bbox)
            self.confidence_pre = None
            sync_track_meta(self)

    def predict(self):
        if (self.kf.x[7] + self.kf.x[2]) <= 0:
            self.kf.x[7] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(self.motion_model.to_box(self.kf.x))
        sync_track_meta(self)

        # --- make scalars robustly ---
        x3 = self.kf.x[3, 0] if self.kf.x.ndim == 2 else self.kf.x[3]
        kalman_score = float(np.clip(x3, self.track_thresh, 1.0))

        if not self.confidence_pre:
            simple_score = float(np.clip(self.conf, 0.1, self.track_thresh))
        else:
            simple_score = float(
                np.clip(
                    self.conf - (self.confidence_pre - self.conf),
                    0.1,
                    self.track_thresh,
                )
            )

        return self.history[-1], kalman_score, simple_score
