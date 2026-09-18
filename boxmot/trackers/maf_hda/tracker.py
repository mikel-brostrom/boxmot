"""Python GMPHD-MAF tracker with hierarchical data association.

Port of Young-min Song's MAF_HDA/GMPHD_MAF implementation (2021).
The upstream BSD 2-Clause notice is retained in this package's LICENSE.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from typing_extensions import Unpack

from boxmot.structures import GeometryKind
from boxmot.trackers.common.association.matching import linear_assignment
from boxmot.trackers.common.base import BaseTracker
from boxmot.trackers.common.constructor import TrackerMetadataOptions, validate_runtime_options
from boxmot.trackers.common.specs import TrackerCapabilities, TrackerFamily
from boxmot.trackers.maf_hda.appearance import MaskedKCF
from boxmot.trackers.maf_hda.association import (
    INITIAL_COVARIANCE,
    MAX_COST,
    fusion_cost,
    gaussian_affinity,
    mask_merge_groups,
    predict_covariance,
)
from boxmot.trackers.maf_hda.config import MafHdaConfig


@dataclass
class _Observation:
    """One filtered or merged detection with its original input row index."""

    bbox: np.ndarray
    mask: np.ndarray
    conf: float
    cls: int
    det_ind: int
    weight: float = 0.0


@dataclass
class _Track:
    """A Gaussian component, mask, appearance filter, and bounded track history."""

    id: int
    bbox: np.ndarray
    mask: np.ndarray
    conf: float
    cls: int
    det_ind: int
    weight: float
    covariance: np.ndarray
    velocity: np.ndarray
    appearance: MaskedKCF | None
    first_frame: int
    first_bbox: np.ndarray
    last_frame: int
    hits: int
    history_observations: deque

    @property
    def xyxy(self) -> np.ndarray:
        """Expose the current AABB to shared display consumers."""
        return self.bbox


class MafHda(BaseTracker):
    """Track instance masks using GMPHD motion, masked KCF appearance, and HDA.

    Inputs are AABB ``Detections`` with full-frame boolean masks. An image is
    required when either association stage uses appearance; both stages can
    run with ``motion`` alone without image pixels.
    Only observed, confirmed tracks are emitted. Missing tracks retain their
    appearance and trajectory for ``max_age`` frames of track-to-track recovery.
    """

    capabilities = TrackerCapabilities(
        family=TrackerFamily.MULTIMODAL,
        geometry_kinds=frozenset({GeometryKind.AABB}),
        requires_masks=True,
        accepts_masks=True,
        accepts_frame=True,
    )
    _requires_masks = True

    def __init__(
        self,
        config: MafHdaConfig | None = None,
        **kwargs: Unpack[TrackerMetadataOptions],
    ) -> None:
        """Configure mask association, KCF appearance, and tracklet recovery.

        Args:
            config: Immutable algorithm settings. None selects MafHdaConfig defaults.
            **kwargs: Runtime ``per_class``, ``class_ids``, and ``class_names`` settings.
        """
        config = MafHdaConfig.resolve(config)
        validate_runtime_options(kwargs, box=False)
        self.config = config
        self._requires_frame = config.s2ta_mode != "motion" or config.t2ta_mode != "motion"
        super().__init__(
            det_thresh=config.det_thresh,
            max_age=config.max_age,
            max_obs=config.max_obs,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold,
            asso_func=config.asso_func,
            **kwargs,
        )
        self.velocity_alpha = config.velocity_alpha
        self.merge_iou_thresh = config.merge_iou_thresh
        self.appearance_lower = config.appearance_lower
        self.appearance_upper = config.appearance_upper
        self.appearance_gate = config.appearance_gate
        self.s2ta_mode = config.s2ta_mode
        self.t2ta_mode = config.t2ta_mode
        self.template_size = config.template_size
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        """Discard all sequence state while preserving configuration."""
        self._reset_common_state()
        self._tracks = []

    def _observations(self, dets: np.ndarray, masks: np.ndarray) -> list[_Observation]:
        """Filter and merge source detections without changing caller-owned masks."""
        batch = self._make_detection_batch(dets, masks=masks)
        observations = [
            _Observation(box.copy(), mask.copy(), float(conf), int(cls), int(index))
            for box, mask, conf, cls, index in zip(batch.boxes, masks, batch.confs, batch.clss, batch.det_inds)
            if conf >= self.det_thresh
        ]
        groups = mask_merge_groups(
            [observation.mask for observation in observations],
            [observation.cls for observation in observations],
            self.merge_iou_thresh,
        )
        merged = []
        for group in groups:
            representative = max(group, key=lambda index: (observations[index].conf, -observations[index].det_ind))
            observation = observations[representative]
            if len(group) > 1:
                observation.mask = np.logical_or.reduce([observations[index].mask for index in group])
                boxes = np.asarray([observations[index].bbox for index in group])
                observation.bbox = np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]
            merged.append(observation)
        total = sum(observation.conf for observation in merged)
        for observation in merged:
            observation.weight = observation.conf / total if total > 0 else 1.0 / len(merged)
        return merged

    def _new_track(self, observation: _Observation, image: np.ndarray | None) -> _Track:
        """Create a birth component with the source motion and appearance priors."""
        appearance = None
        if self.s2ta_mode != "motion" or self.t2ta_mode != "motion":
            appearance = MaskedKCF(image, observation.bbox, observation.mask, template_size=self.template_size)
        return _Track(
            id=self.id_allocator.alloc(),
            bbox=observation.bbox.copy(),
            mask=observation.mask.copy(),
            conf=observation.conf,
            cls=observation.cls,
            det_ind=observation.det_ind,
            weight=observation.weight,
            covariance=INITIAL_COVARIANCE.copy(),
            velocity=np.zeros(2),
            appearance=appearance,
            first_frame=self.frame_count,
            first_bbox=observation.bbox.copy(),
            last_frame=self.frame_count,
            hits=1,
            history_observations=deque([observation.bbox.copy()], maxlen=self.max_obs),
        )

    def _associate(
        self,
        tracks: list[_Track],
        observations: list[_Observation],
        image: np.ndarray | None,
        *,
        candidates: list[_Track] | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Compute independent candidates, then solve a gated assignment once."""
        shape = (len(tracks), len(observations))
        motion = np.zeros(shape, dtype=np.float64)
        appearance = np.zeros(shape, dtype=np.float64)
        overlap = np.zeros(shape, dtype=np.float64)
        allowed = np.zeros(shape, dtype=bool)
        posteriors = []
        recovery = candidates is not None
        mode = self.t2ta_mode if recovery else self.s2ta_mode
        boxes = np.asarray([observation.bbox for observation in observations], dtype=np.float64).reshape(-1, 4)
        for row, track in enumerate(tracks):
            if recovery:
                duration = track.last_frame - track.first_frame
                displacement = (track.bbox[:2] + track.bbox[2:] - track.first_bbox[:2] - track.first_bbox[2:]) * 0.5
                velocity = displacement / duration if duration > 0 else np.zeros(2)
                posterior = track.covariance
                for col, candidate in enumerate(candidates):
                    gap = candidate.first_frame - track.last_frame
                    if not 1 <= gap <= self.max_age or candidate.cls != track.cls:
                        continue
                    predicted = track.bbox + np.tile(velocity * gap, 2)
                    likelihood, _ = gaussian_affinity(
                        candidate.first_bbox[None], predicted, track.covariance, recovery=True
                    )
                    motion[row, col] = track.weight * likelihood[0]
                    # The temporal bridge ends at the candidate's birth frame.
                    # Both geometry affinities must compare that same instant,
                    # even when confirmation takes several observations.
                    overlap[row, col] = np.clip(
                        self.asso_func(predicted[None], candidate.first_bbox[None])[0, 0], 0.0, 1.0
                    )
                    allowed[row, col] = True
            else:
                predicted = track.bbox + np.tile(track.velocity, 2)
                covariance = predict_covariance(track.covariance)
                likelihood, posterior = gaussian_affinity(boxes, predicted, covariance)
                motion[row] = track.weight * likelihood
                overlap[row] = np.clip(self.asso_func(predicted[None], boxes)[0], 0.0, 1.0)
                allowed[row] = [observation.cls == track.cls for observation in observations]
            posteriors.append(posterior)
            if mode != "motion" and track.appearance is not None:
                for col, observation in enumerate(observations):
                    if allowed[row, col] and (not self.appearance_gate or overlap[row, col] >= self.iou_threshold):
                        appearance[row, col] = track.appearance.score(image, observation.bbox, observation.mask)
        motion[~allowed] = 0.0
        costs = fusion_cost(
            motion,
            appearance,
            overlap,
            mode=mode,
            recovery=recovery,
            appearance_lower=self.appearance_lower,
            appearance_upper=self.appearance_upper,
            overlap_lower=self.iou_threshold,
        )
        costs[~allowed] = MAX_COST
        matches, _, _ = linear_assignment(costs, thresh=np.nextafter(MAX_COST, 0.0))
        totals = motion.sum(axis=0, keepdims=True)
        weights = np.divide(motion, totals, out=np.zeros_like(motion), where=totals > 0)
        return matches, posteriors, weights

    def _update_track(
        self,
        track: _Track,
        observation: _Observation,
        covariance: np.ndarray,
        weight: float,
        image: np.ndarray | None,
    ) -> None:
        """Commit a matched observation; candidate scoring never updates a filter."""
        predicted_center = (track.bbox[:2] + track.bbox[2:]) * 0.5 + track.velocity
        center = (observation.bbox[:2] + observation.bbox[2:]) * 0.5
        track.velocity = self.velocity_alpha * track.velocity + (1.0 - self.velocity_alpha) * (
            center - predicted_center
        )
        track.bbox = observation.bbox.copy()
        track.mask = observation.mask.copy()
        track.conf = observation.conf
        track.det_ind = observation.det_ind
        track.covariance = covariance
        track.weight = weight if weight > 0 else observation.weight
        track.last_frame = self.frame_count
        track.hits += 1
        track.history_observations.append(track.bbox.copy())
        if track.appearance is not None:
            # The released source reinitializes appearance after every accepted
            # S2TA match (APPEARANCE_STRICT_UPDATE_ON=0).
            track.appearance = MaskedKCF(image, track.bbox, track.mask, template_size=self.template_size)

    def _merge_tracks(self, tracks: list[_Track], image: np.ndarray | None) -> list[_Track]:
        """Merge current duplicate masks, retaining the oldest identity."""
        groups = mask_merge_groups(
            [track.mask for track in tracks], [track.cls for track in tracks], self.merge_iou_thresh
        )
        merged = []
        for group in groups:
            representative = min(group, key=lambda index: tracks[index].id)
            track = tracks[representative]
            if len(group) > 1:
                track.mask = np.logical_or.reduce([tracks[index].mask for index in group])
                boxes = np.asarray([tracks[index].bbox for index in group])
                track.bbox = np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]
                track.history_observations[-1] = track.bbox.copy()
                if track.appearance is not None:
                    track.appearance = MaskedKCF(image, track.bbox, track.mask, template_size=self.template_size)
            merged.append(track)
        return merged

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray | None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance S2TA, mask merging, and T2TA and emit current-frame tracks."""
        self.frame_count += 1
        observations = self._observations(dets, masks)
        live = [track for track in self._tracks if track.last_frame == self.frame_count - 1]
        # An in-time reappearance may still be awaiting min_hits observations.
        # Retain old tracklets until such a candidate can be confirmed; T2TA
        # separately enforces max_age on the actual disappearance/birth gap.
        retention = self.max_age + self.min_hits - 1
        lost = [track for track in self._tracks if 1 < self.frame_count - track.last_frame <= retention]
        matches, posteriors, weights = self._associate(live, observations, img)
        matched_tracks = set(matches[:, 0].tolist())
        matched_observations = set(matches[:, 1].tolist())
        current = []
        for track_index, observation_index in matches:
            track = live[track_index]
            self._update_track(
                track,
                observations[observation_index],
                posteriors[track_index][observation_index],
                weights[track_index, observation_index],
                img,
            )
            current.append(track)
        lost.extend(track for index, track in enumerate(live) if index not in matched_tracks)
        current.extend(
            self._new_track(observation, img)
            for index, observation in enumerate(observations)
            if index not in matched_observations
        )
        current = self._merge_tracks(current, img)

        # Only newly reliable tracklets enter T2TA. An established trajectory
        # cannot acquire a second identity merely by passing a lost object.
        candidates = [track for track in current if track.hits == self.min_hits]
        lost = [track for track in lost if track.hits >= self.min_hits]
        recovery_observations = [
            _Observation(track.bbox, track.mask, track.conf, track.cls, track.det_ind, track.weight)
            for track in candidates
        ]
        recovery_matches, _, _ = self._associate(lost, recovery_observations, img, candidates=candidates)
        recovered_ids = set()
        for lost_index, candidate_index in recovery_matches:
            old, candidate = lost[lost_index], candidates[candidate_index]
            candidate.id = old.id
            candidate.first_frame = old.first_frame
            candidate.first_bbox = old.first_bbox.copy()
            candidate.hits += old.hits
            candidate.history_observations = deque(
                (*old.history_observations, *candidate.history_observations), maxlen=self.max_obs
            )
            recovered_ids.add(old.id)
        lost = [track for track in lost if track.id not in recovered_ids]
        total = sum(track.weight for track in current)
        for track in current:
            track.weight = track.weight / total if total > 0 else 1.0 / len(current)
        self._tracks = current + lost
        self.active_tracks = sorted(
            (track for track in current if track.hits >= self.min_hits), key=lambda track: track.id
        )
        rows = [
            self.format_output_row(track.bbox, track.id, track.conf, track.cls, track.det_ind)
            for track in self.active_tracks
        ]
        output_masks = (
            np.stack([track.mask for track in self.active_tracks])
            if rows
            else np.empty((0, *masks.shape[1:]), dtype=bool)
        )
        return self.format_output_rows(rows), output_masks
