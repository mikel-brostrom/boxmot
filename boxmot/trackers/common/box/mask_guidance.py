"""Shared temporal-mask lifecycle for Python box trackers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from boxmot.structures import Frame
from boxmot.trackers.common.mask_guidance import (
    MaskGuidance,
    MaskGuidanceConfig,
    _validated_mask_guidance_values,
)
from boxmot.trackers.common.track_state import BoxTrack
from boxmot.trackers.common.tracking.collections import LIVE_STATE_GROUPS, tracks_from_owner
from boxmot.trackers.common.tracking.track import TrackState


class BoxMaskGuidanceMixin:
    """Own one bounded propagation stream around the validated tracking kernel."""

    @property
    def guidance_masks(self) -> Mapping[int, np.ndarray] | None:
        """Borrow current propagated masks, or return None when guidance is disabled."""
        return None if self._mask_guidance is None else self._mask_guidance.masks

    def _init_mask_guidance(
        self, guidance: MaskGuidanceConfig | MaskGuidance | None, *, edgetam: Mapping[str, object] | None
    ) -> None:
        """Validate optional guidance without loading its model."""
        if edgetam is not None and not isinstance(edgetam, Mapping):
            raise TypeError("edgetam must be a mapping of guidance parameters or None.")
        overrides = _validated_mask_guidance_values({} if edgetam is None else edgetam)
        if guidance is not None:
            if not isinstance(guidance, (MaskGuidanceConfig, MaskGuidance)):
                raise TypeError("mask_guidance must be a MaskGuidanceConfig, MaskGuidance, or None.")
            if isinstance(guidance, MaskGuidance):
                conflicts = [name for name, value in overrides.items() if getattr(guidance.config, name) != value]
                if conflicts:
                    raise ValueError(
                        "Explicit mask guidance options conflict with the prebuilt component: " + ", ".join(conflicts)
                    )
            else:
                guidance = replace(guidance, **overrides)
            if self.is_obb or self.per_class or self.asso_func_name != "iou":
                raise ValueError("Mask guidance requires AABB, asso_func='iou', and per_class=False.")
            self._requires_frame = True
            self._requires_frame_dimensions_only = False
        self._mask_guidance = MaskGuidance(guidance) if isinstance(guidance, MaskGuidanceConfig) else guidance
        self._mask_frame_index = 0
        self._mask_confirmed_ids: set[int] = set()

    def _validate_frame_context(self, frame: Frame | np.ndarray | None) -> None:
        """Reject sequence changes before any model inference."""
        super()._validate_frame_context(frame)
        if self._mask_guidance is not None:
            self._mask_guidance.validate_frame(frame)

    def _before_track_detections(self, img: np.ndarray | None) -> None:
        """Propagate once per input frame, before any association stage."""
        if self._mask_guidance is not None:
            self._mask_guidance.advance(self._mask_frame_index, img)

    def _observe_track_outputs(self, boxes: np.ndarray, track_ids: np.ndarray, detection_indices: np.ndarray) -> None:
        """Condition next-frame propagation on fresh, confirmed detection boxes."""
        if self._mask_guidance is None:
            return
        retained = {
            int(track.id): track
            for group in LIVE_STATE_GROUPS
            for track in tracks_from_owner(self, group)
            if getattr(getattr(track, "meta", None), "state", None) is not TrackState.REMOVED
        }
        self._mask_confirmed_ids.intersection_update(retained)
        observed = {
            int(track_id): boxes[int(index)]
            for track_id, index in zip(track_ids, detection_indices)
            if index >= 0 and int(track_id) in retained
        }
        newly_confirmed = sorted(observed.keys() - self._mask_confirmed_ids)
        self._mask_confirmed_ids.update(observed)
        # SORT variants temporarily suppress outputs while rebuilding a hit
        # streak after recovery. Previously confirmed, freshly matched tracks
        # still supply causal detection observations during that interval.
        for track_id in self._mask_confirmed_ids:
            track = retained[track_id]
            if isinstance(track, BoxTrack):
                fresh = track.is_activated and track.frame_id == self.frame_count
            else:
                fresh = track.time_since_update == 0
            if not fresh or track.det_ind is None:
                continue
            index = int(track.det_ind)
            if 0 <= index < len(boxes):
                observed[track_id] = boxes[index]
        self._mask_guidance.observe(
            observed,
            newly_confirmed if self._mask_frame_index else (),
            retained_track_ids=tuple(retained),
        )
        self._mask_frame_index += 1

    def _condition_association(
        self, costs: np.ndarray, tracks: Sequence, detections: Sequence, *, threshold: float
    ) -> np.ndarray:
        """Condition a track-by-detection cost matrix at the stage's own gate."""
        if self._mask_guidance is None or not self._mask_guidance._masks:
            return costs
        return self._mask_guidance.condition(
            costs, [track.id for track in tracks], self._association_boxes(detections), threshold=threshold
        )

    def _condition_similarity(
        self, similarity: np.ndarray, tracks: Sequence, detections: Sequence, *, threshold: float
    ) -> np.ndarray:
        """Apply the same cue to detection-by-track geometry similarities."""
        if self._mask_guidance is None or not self._mask_guidance._masks:
            return similarity
        costs = 1.0 - np.asarray(similarity, dtype=float).T
        adjusted = self._condition_association(costs, tracks, detections, threshold=1.0 - threshold)
        if np.array_equal(costs, adjusted):
            return similarity
        return similarity + (costs - adjusted).T

    def _reset_common_state(self) -> None:
        """Release mask state before IDs can be reused in a new sequence."""
        if self._mask_guidance is not None:
            self._mask_guidance.reset()
        self._mask_frame_index = 0
        self._mask_confirmed_ids.clear()
        super()._reset_common_state()
