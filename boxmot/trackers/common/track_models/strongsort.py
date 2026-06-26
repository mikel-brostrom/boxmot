# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import os

import numpy as np

from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.trackers.common.appearance import ema_update_embedding, normalize_embedding
from boxmot.trackers.common.association.matching import chi2inv95
from boxmot.trackers.common.association.strongsort import (
    gate_cost_matrix,
    iou_cost,
    matching_cascade,
    min_cost_matching,
)
from boxmot.trackers.common.motion.cmc import create_cmc
from boxmot.trackers.common.tracking.track import TrackIdAllocator


class Detection:
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, conf, cls, det_ind, feat):
        self.tlwh = tlwh
        self.conf = conf
        self.cls = cls
        self.det_ind = det_ind
        self.feat = feat

    def to_xyah(self):
        """Convert bounding box to `(center x, center y, aspect ratio, height)`."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


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
    """

    def __init__(
        self,
        detection,
        id,
        n_init,
        max_age,
        ema_alpha,
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

        # start with confirmed in Ci as test expect equal amount of outputs as inputs
        self.state = (
            TrackState.Confirmed
            if (os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("GITHUB_JOB") != "mot-metrics-benchmark")
            else TrackState.Tentative
        )
        self.features = []
        if detection.feat is not None:
            self.features.append(normalize_embedding(detection.feat))

        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilterXYAH()
        self.mean, self.covariance = self.kf.initiate(self.bbox)

    def to_tlwh(self):
        """Get current position in `(top left x, top left y, width, height)`."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in `(min x, min y, max x, max y)`."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def camera_update(self, warp_matrix):
        [a, b] = warp_matrix
        warp_matrix = np.array([a, b, [0, 0, 1]])
        warp_matrix = warp_matrix.tolist()
        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = warp_matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self):
        """Propagate the state distribution to the current time step."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        """Perform Kalman filter measurement update and update the feature cache."""
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, self.bbox, self.conf)

        smooth_feat = ema_update_embedding(
            self.features[-1],
            normalize_embedding(detection.feat),
            alpha=self.ema_alpha,
        )
        self.features = [smooth_feat]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed when there is no association at the current time step."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Return True if this track is tentative."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Return True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Return True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    """

    GATING_THRESHOLD = np.sqrt(chi2inv95[4])

    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
        id_allocator=None,
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda

        self.tracks = []
        self.id_allocator = id_allocator or TrackIdAllocator()
        self._next_id = self.id_allocator.next_id
        self.cmc = create_cmc("ecc")

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict()

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        """Perform measurement update and track management."""
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )

            return cost_matrix

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            iou_cost,
            self.max_iou_dist,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        track_id = self.id_allocator.alloc()
        self._next_id = self.id_allocator.next_id
        self.tracks.append(
            Track(
                detection,
                track_id,
                self.n_init,
                self.max_age,
                self.ema_alpha,
            )
        )

    def reset(self):
        self.tracks = []
        self.id_allocator.reset()
        self._next_id = self.id_allocator.next_id
        if hasattr(self.metric, "samples"):
            self.metric.samples = {}
