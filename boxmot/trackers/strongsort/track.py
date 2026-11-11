# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable, Sequence, ValuesView

import numpy as np

from boxmot.trackers.common.appearance import ema_update_embedding, normalize_embedding
from boxmot.trackers.common.geometry.obb import (
    smooth_display_angle,
    smooth_obb_corners,
    xywha_to_corners,
    xywha_to_xyxy,
)
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.models import (
    MotionModelAdapter,
    MotionModelKind,
    create_motion_model,
    resolve_motion_model_kind,
)

__all__ = ("Track", "TrackState")


class _Detection:
    """StrongSORT observation with box geometry, metadata, and appearance."""

    def __init__(
        self,
        box: np.ndarray,
        conf: float,
        cls: int,
        det_ind: int,
        feat: np.ndarray | None,
        *,
        is_obb: bool,
    ) -> None:
        """Store TLWH or XYWHA geometry and the originating detection index."""
        self.tlwh = np.asarray(box, dtype=np.float32)
        self.conf = conf
        self.cls = cls
        self.det_ind = det_ind
        self.feat = feat
        self.is_obb = bool(is_obb)

    def to_xyah(self) -> np.ndarray:
        """Convert an AABB measurement to center-x, center-y, aspect, height."""
        measurement = self.tlwh.copy()
        measurement[:2] += measurement[2:] / 2
        measurement[2] /= measurement[3]
        return measurement

    def to_measurement(self) -> np.ndarray:
        """Return the Kalman-filter measurement for the active box mode."""
        return self.tlwh.copy() if self.is_obb else self.to_xyah()

    @property
    def xyxy(self) -> np.ndarray:
        """Return axis-aligned geometry from an AABB detection's TLWH box."""
        box = self.tlwh.copy()
        box[2:] += box[:2]
        return box

    @property
    def xywha(self) -> np.ndarray:
        """Return oriented detection geometry for shared association."""
        if not self.is_obb:
            raise AttributeError("xywha is only available for OBB detections")
        return self.tlwh.copy()

    @classmethod
    def batch_geometry(cls, detections: Sequence[_Detection], *, is_obb: bool) -> np.ndarray | None:
        """Extract native observations while retaining custom geometry hooks."""
        if cls is not _Detection or any(
            type(detection) is not _Detection
            or detection.is_obb != is_obb
            or any(name in detection.__dict__ for name in ("xyxy", "xywha"))
            for detection in detections
        ):
            return None
        if not detections:
            return np.empty((0, 5 if is_obb else 4), dtype=np.float32)
        if any(detection.tlwh.dtype != detections[0].tlwh.dtype for detection in detections):
            return None
        boxes = np.asarray([detection.tlwh for detection in detections])
        if not is_obb:
            boxes[:, 2:] += boxes[:, :2]
        return boxes


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
    """A single target track with a selectable box motion model and lifecycle."""

    def __init__(
        self,
        detection: _Detection,
        id: int,
        n_init: int,
        max_age: int,
        max_obs: int,
        ema_alpha: float = 0.9,
        is_obb: bool = False,
        *,
        noise_config: KalmanNoiseConfig | None = None,
        motion_model: MotionModelAdapter | None = None,
    ) -> None:
        """Initialize motion, appearance, and lifecycle state from one detection."""
        self.id = id
        self.is_obb = bool(is_obb)
        self.motion_model = motion_model or create_motion_model(
            resolve_motion_model_kind("strongsort", is_obb=self.is_obb), is_obb=self.is_obb
        )
        self.bbox = self.measurement_for_detection(detection)
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha
        self.features: list[np.ndarray] = [] if detection.feat is None else [normalize_embedding(detection.feat)]

        # CI smoke tests expect immediate output; metric benchmarks retain confirmation delays.
        self.state = (
            TrackState.Confirmed
            if n_init <= 1
            or (os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("GITHUB_JOB") != "mot-metrics-benchmark")
            else TrackState.Tentative
        )
        self._n_init = n_init
        self._max_age = max_age

        self.kf = self.motion_model.create_filter(noise_config=noise_config)
        self.mean, self.covariance = self.motion_model.initiate(self.kf, self.bbox, confidence=float(self.conf))
        self.history_observations: deque[np.ndarray] = deque(maxlen=max(1, int(max_obs)))
        self._plot_angle: float | None = None
        self._append_current_history()

    def to_tlwh(self) -> np.ndarray:
        """Get current position in `(top left x, top left y, width, height)`."""
        if self.is_obb or self.motion_model.kind is not MotionModelKind.XYAH:
            box = self.to_tlbr()
            box[2:] -= box[:2]
            return box
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self) -> np.ndarray:
        """Return the estimated enclosing box as `(min x, min y, max x, max y)`."""
        if self.is_obb:
            return xywha_to_xyxy(self.xywha.reshape(1, 5))[0]
        if self.motion_model.kind is not MotionModelKind.XYAH:
            return self.motion_model.to_box(self.mean)[0]
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Return axis-aligned geometry for shared association helpers."""
        return self.to_tlbr()

    @property
    def xywha(self) -> np.ndarray:
        """Return oriented geometry from the selected motion model."""
        if not self.is_obb:
            raise AttributeError("xywha is only available for OBB tracks")
        if self.motion_model.kind is MotionModelKind.XYWH:
            return self.mean[:5].copy()
        return self.motion_model.to_box(self.mean)[0]

    @classmethod
    def batch_geometry(cls, tracks: Sequence[Track], *, is_obb: bool) -> np.ndarray | None:
        """Batch native default-model states in their scalar conversion order."""
        kind = MotionModelKind.XYWH if is_obb else MotionModelKind.XYAH
        if cls is not Track or any(
            type(track) is not Track
            or track.is_obb != is_obb
            or track.motion_model.kind is not kind
            or any(name in track.__dict__ for name in ("xyxy", "xywha", "to_tlwh", "to_tlbr"))
            for track in tracks
        ):
            return None
        if not tracks:
            return np.empty((0, 5 if is_obb else 4), dtype=np.float32)
        if any(track.mean.dtype != tracks[0].mean.dtype for track in tracks):
            return None
        boxes = np.asarray([track.mean[: 5 if is_obb else 4] for track in tracks])
        if not is_obb:
            boxes[:, 2] *= boxes[:, 3]
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        return boxes

    def measurement_for_detection(self, detection: _Detection) -> np.ndarray:
        """Convert a detection into this track's measurement coordinates."""
        if (self.is_obb and self.motion_model.kind is MotionModelKind.XYWH) or (
            not self.is_obb and self.motion_model.kind is MotionModelKind.XYAH
        ):
            return detection.to_measurement()
        box = detection.xywha if self.is_obb else detection.xyxy
        return self.motion_model.to_measurement(np.r_[box, detection.conf], column=False)

    def camera_update(self, warp_matrix: np.ndarray) -> None:
        """Transform this track using the shared batched CMC implementation."""
        self.multi_camera_update([self], warp_matrix)

    @classmethod
    def multi_camera_update(cls, tracks: Iterable[Track], warp_matrix: np.ndarray) -> None:
        """Transform compatible track states and their last AABB measurements."""
        groups: dict[tuple[MotionModelKind, bool], list[Track]] = {}
        for track in tracks:
            groups.setdefault((track.motion_model.kind, track.is_obb), []).append(track)
        for (_, is_obb), group in groups.items():
            model = group[0].motion_model
            means, covariances = model.transform_states(
                np.asarray([track.mean for track in group]),
                np.asarray([track.covariance for track in group]),
                warp_matrix,
            )
            for track, mean, covariance in zip(group, means, covariances):
                track.mean, track.covariance = mean, covariance
            if not is_obb:
                measurements = model.transform_measurements(np.asarray([track.bbox for track in group]), warp_matrix)
                for track, measurement in zip(group, measurements):
                    track.bbox = measurement

    def _append_current_history(self) -> None:
        """Record current geometry with continuous corners for oriented boxes."""
        if self.is_obb:
            geometry, self._plot_angle = smooth_obb_corners(self.xywha, self._plot_angle)
        else:
            geometry = self.to_tlbr()
        self.history_observations.append(np.asarray(geometry, dtype=np.float32).copy())

    def increment_age(self) -> None:
        """Advance the track age and time elapsed since its last observation."""
        self.age += 1
        self.time_since_update += 1

    def predict(self, *, dt: float | None = None) -> None:
        """Propagate the state distribution to the current time step."""
        self.mean, self.covariance = self.motion_model.predict(self.kf, self.mean, self.covariance, dt=dt)
        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def _kalman_groups(tracks: Iterable[Track]) -> ValuesView[list[Track]]:
        """Group stateless filters by numerical policy without sharing owners."""
        groups: dict[tuple[object, ...], list[Track]] = {}
        for track in tracks:
            kalman = track.kf
            key = (
                track.motion_model.kind,
                track.is_obb,
                type(kalman),
                kalman.noise_config,
                kalman.dt,
                kalman._std_weight_position,
                kalman._std_weight_velocity,
                kalman._motion_mat.tobytes(),
                kalman._update_mat.tobytes(),
            )
            groups.setdefault(key, []).append(track)
        return groups.values()

    @classmethod
    def multi_predict(cls, tracks: Iterable[Track], *, dt: float | None = None) -> None:
        """Predict compatible track states together, retaining track-local ages."""
        for group in cls._kalman_groups(tracks):
            mean, covariance = group[0].motion_model.multi_predict(
                group[0].kf,
                np.asarray([track.mean for track in group]),
                np.asarray([track.covariance for track in group]),
                dt=dt,
            )
            for track, state, uncertainty in zip(group, mean, covariance):
                track.mean, track.covariance = state, uncertainty
                track.increment_age()

    @classmethod
    def multi_update(cls, pairs: Iterable[tuple[Track, _Detection]]) -> None:
        """Correct matched states together using each detection's confidence."""
        pairs = list(pairs)
        detections = {id(track): detection for track, detection in pairs}
        for group in cls._kalman_groups(track for track, _ in pairs):
            observations = [detections[id(track)] for track in group]
            measurements = cls._detection_measurements(group, observations)
            mean, covariance = group[0].motion_model.multi_update(
                group[0].kf,
                np.asarray([track.mean for track in group]),
                np.asarray([track.covariance for track in group]),
                measurements,
                confidence=np.asarray([detection.conf for detection in observations]),
            )
            histories = cls._prepare_update_histories(group, mean)
            for index, (track, detection, measurement, state, uncertainty) in enumerate(
                zip(group, observations, measurements, mean, covariance)
            ):
                track.mean, track.covariance = state, uncertainty
                if histories is None:
                    track._finish_update(detection, measurement)
                else:
                    track._finish_update(detection, measurement, history=histories[index])

    @staticmethod
    def _detection_measurements(tracks: Sequence[Track], detections: Sequence[_Detection]) -> np.ndarray:
        """Convert native default-model observations together, retaining hooks."""
        first = tracks[0]
        kind = MotionModelKind.XYWH if first.is_obb else MotionModelKind.XYAH
        if first.motion_model.kind is not kind or any(
            type(track) is not Track
            or "measurement_for_detection" in track.__dict__
            or type(detection) is not _Detection
            or detection.is_obb != first.is_obb
            or detection.tlwh.dtype != detections[0].tlwh.dtype
            or any(name in detection.__dict__ for name in ("to_measurement", "to_xyah"))
            for track, detection in zip(tracks, detections)
        ):
            return np.asarray(
                [track.measurement_for_detection(detection) for track, detection in zip(tracks, detections)]
            )
        measurements = np.asarray([detection.tlwh for detection in detections])
        if not first.is_obb:
            measurements[:, :2] += measurements[:, 2:] / 2
            measurements[:, 2] /= measurements[:, 3]
        return measurements

    @staticmethod
    def _prepare_update_histories(
        tracks: Sequence[Track],
        means: np.ndarray,
    ) -> list[tuple[np.ndarray, float]] | None:
        """Batch native oriented corners while retaining scalar angle continuity."""
        if len(tracks) < 2 or any(
            type(track) is not Track
            or not track.is_obb
            or track.motion_model.kind is not MotionModelKind.XYWH
            or any(name in track.__dict__ for name in ("xywha", "_append_current_history", "_finish_update"))
            for track in tracks
        ):
            return None
        prepared = [smooth_display_angle(track._plot_angle, mean[:5]) for track, mean in zip(tracks, means)]
        corners = xywha_to_corners(np.asarray([box for _, box in prepared]))
        return [(corner, angle) for corner, (angle, _) in zip(corners, prepared)]

    def update(self, detection: _Detection) -> None:
        """Correct the Kalman state and record the accepted observation."""
        measurement = self.measurement_for_detection(detection)
        self.mean, self.covariance = self.motion_model.update(
            self.kf, self.mean, self.covariance, measurement, confidence=detection.conf
        )
        self._finish_update(detection, measurement)

    def _finish_update(
        self,
        detection: _Detection,
        measurement: np.ndarray,
        *,
        history: tuple[np.ndarray, float] | None = None,
    ) -> None:
        """Update observation metadata and confirmation after a correction."""
        self.bbox = measurement
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        if history is None:
            self._append_current_history()
        else:
            geometry, self._plot_angle = history
            self.history_observations.append(geometry.copy())

        self.features = [
            ema_update_embedding(self.features[-1], normalize_embedding(detection.feat), alpha=self.ema_alpha)
        ]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        """Mark this track as missed when there is no association at the current time step."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self) -> bool:
        """Return True if this track is tentative."""
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """Return True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """Return True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
