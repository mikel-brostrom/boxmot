# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections import deque

import numpy as np

from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH


def _normalize_feature(feature: np.ndarray | None) -> np.ndarray | None:
    if feature is None:
        return None

    normalized = np.asarray(feature, dtype=np.float32)
    norm = np.linalg.norm(normalized)
    if norm > 0:
        normalized = normalized / norm
    return normalized


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(
        self,
        detection,
        id,
        n_init,
        max_age,
        ema_alpha,
        max_obs,
    ):
        self.id = id
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha

        self.state = TrackState.Tentative
        self.features = []
        feature = _normalize_feature(detection.feat)
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.history_observations = deque(maxlen=max_obs)

        self.kf = KalmanFilterXYAH()
        self.mean, self.covariance = self.kf.initiate(self.bbox)
        self.history_observations.append(self.to_tlbr())

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def camera_update(self, warp_matrix):
        warp_matrix = np.asarray(warp_matrix, dtype=np.float32)
        if warp_matrix.shape == (2, 3):
            warp_matrix = np.vstack((warp_matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = warp_matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / max(h, 1e-6), h]

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        """
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.bbox, self.conf
        )

        feature = _normalize_feature(detection.feat)
        if feature is not None:
            if self.features:
                smooth_feat = (
                    self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
                )
                smooth_feat = _normalize_feature(smooth_feat)
                self.features = [smooth_feat if smooth_feat is not None else feature]
            else:
                self.features = [feature]

        self.hits += 1
        self.time_since_update = 0
        self.history_observations.append(self.to_tlbr())
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    @property
    def xyxy(self):
        return self.to_tlbr()

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
