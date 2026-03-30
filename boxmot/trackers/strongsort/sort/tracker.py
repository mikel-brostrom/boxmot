# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import absolute_import

from collections.abc import Callable

import numpy as np

from boxmot.trackers.strongsort.sort import iou_matching, linear_assignment
from boxmot.trackers.strongsort.sort.track import Track
from boxmot.utils.matching import chi2inv95


class Tracker:
    """
    Multi-target StrongSORT tracker state and association logic.
    """

    GATING_THRESHOLD = np.sqrt(chi2inv95[4])

    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        max_obs=50,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
        tracks=None,
        next_track_id_fn: Callable[[], int] | None = None,
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.max_obs = max_obs
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        self._next_track_id_fn = next_track_id_fn

        self.tracks = [] if tracks is None else tracks
        self._next_id = 1

    def predict(self):
        """Propagate active track state distributions one step forward."""
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

        self.tracks[:] = [track for track in self.tracks if not track.is_deleted()]
        self._update_metric()

    def _update_metric(self) -> None:
        active_targets = []
        features = []
        targets = []

        for track in self.tracks:
            if not track.is_confirmed():
                continue
            active_targets.append(track.id)
            if track.features:
                features.extend(track.features)
                targets.extend([track.id] * len(track.features))

        self.metric.partial_fit(
            np.asarray(features, dtype=np.float32),
            np.asarray(targets, dtype=np.int32),
            active_targets,
        )

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            if len(track_indices) == 0 or len(detection_indices) == 0:
                return np.empty(
                    (len(track_indices), len(detection_indices)),
                    dtype=np.float32,
                )

            features = np.asarray(
                [dets[detection_idx].feat for detection_idx in detection_indices],
                dtype=np.float32,
            )
            targets = np.asarray(
                [tracks[track_idx].id for track_idx in track_indices],
                dtype=np.int32,
            )
            cost_matrix = self.metric.distance(features, targets)
            return linear_assignment.gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )

        confirmed_tracks = []
        unconfirmed_tracks = []
        for track_idx, track in enumerate(self.tracks):
            if track.is_confirmed():
                confirmed_tracks.append(track_idx)
            else:
                unconfirmed_tracks.append(track_idx)

        matches_a, unmatched_tracks_a, unmatched_detections = (
            linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks,
            )
        )

        iou_track_candidates = unconfirmed_tracks + [
            track_idx
            for track_idx in unmatched_tracks_a
            if self.tracks[track_idx].time_since_update == 1
        ]
        unmatched_tracks_a = [
            track_idx
            for track_idx in unmatched_tracks_a
            if self.tracks[track_idx].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = (
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )
        )

        matches = matches_a + matches_b
        unmatched_tracks = unmatched_tracks_a.copy()
        seen = set(unmatched_tracks)
        for track_idx in unmatched_tracks_b:
            if track_idx not in seen:
                unmatched_tracks.append(track_idx)
                seen.add(track_idx)
        return matches, unmatched_tracks, unmatched_detections

    def _next_track_id(self) -> int:
        if self._next_track_id_fn is not None:
            return self._next_track_id_fn()
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def _initiate_track(self, detection):
        self.tracks.append(
            Track(
                detection,
                self._next_track_id(),
                self.n_init,
                self.max_age,
                self.ema_alpha,
                self.max_obs,
            )
        )
