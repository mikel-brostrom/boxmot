from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from boxmot.trackers.common.tracking.collections import (
    LIVE_STATE_GROUPS,
    TRACK_COLLECTION_ATTRS,
    empty_track_collection_like,
    track_collection_attrs,
    tracks_from_mapping,
    validate_track_group,
)
from boxmot.utils import logger as LOGGER


class _PrecomputedCMC:
    """Frame-local CMC adapter used by class-separated updates."""

    def __init__(self, warp: np.ndarray | None) -> None:
        self.warp = warp

    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray | None:
        return self.warp


@dataclass
class ClassTrackState:
    """Class-local tracker collections grouped by lifecycle role."""

    attrs: dict[str, object] = field(default_factory=dict)

    def tracks(self, group: str) -> list:
        return tracks_from_mapping(self.attrs, group)

    def has_live_tracks(self) -> bool:
        return any(self.tracks(group) for group in LIVE_STATE_GROUPS)


class PerClassUpdateMixin:
    """Class-separated update pipeline for trackers that share one public instance."""

    class_track_collection_attrs = TRACK_COLLECTION_ATTRS

    def _update_per_class(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, masks: np.ndarray = None):
        """Run one frame with class-local tracker collections."""
        self._ensure_class_track_states()
        per_class_tracks = []
        per_class_masks = []
        frame_count = self.frame_count
        classes_to_update = self._class_update_ids(dets)
        original_cmc = getattr(self, "cmc", None)
        precomputed_cmc = self._precompute_per_frame_cmc(original_cmc, dets, img, len(classes_to_update))

        if precomputed_cmc is not None:
            self.cmc = precomputed_cmc

        try:
            for cls_id in classes_to_update:
                class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)
                class_masks = self._get_class_masks(dets, masks, cls_id)

                LOGGER.debug(
                    f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings"
                    f" {class_embs.shape if class_embs is not None else None}"
                )

                self._load_class_track_state(cls_id)
                self.frame_count = frame_count

                result = self._update_impl(dets=class_dets, img=img, embs=class_embs, masks=class_masks)
                if isinstance(result, tuple):
                    tracks, track_masks = result
                else:
                    tracks, track_masks = result, None

                self._save_class_track_state(cls_id)

                if tracks.size > 0:
                    per_class_tracks.append(tracks)
                    if track_masks is not None:
                        per_class_masks.append(track_masks)
        finally:
            if precomputed_cmc is not None:
                self.cmc = original_cmc

        self.frame_count = frame_count + 1
        self._restore_class_track_collections()
        if per_class_tracks:
            combined_tracks = np.vstack(per_class_tracks)
            combined_masks = np.vstack(per_class_masks) if per_class_masks else None
            if combined_masks is not None:
                return combined_tracks, combined_masks
            return combined_tracks

        return self.empty_output()

    def _initialize_class_track_states(self) -> None:
        self.class_track_states = {}

    def _ensure_class_track_states(self) -> None:
        if self.class_track_states is None:
            self._initialize_class_track_states()

    def _reset_class_track_states(self) -> None:
        if self.class_track_states is None:
            return
        self.class_track_states = {}

    def _ensure_class_track_state(self, cls_id: int) -> ClassTrackState:
        self._ensure_class_track_states()
        cls_id = int(cls_id)
        self.class_catalog.validate_ids([cls_id])
        state = self.class_track_states.setdefault(cls_id, ClassTrackState())
        for attr_name in self._class_state_attr_names():
            state.attrs.setdefault(attr_name, self._empty_state_like(getattr(self, attr_name)))
        return state

    def _class_state_attr_names(self) -> tuple[str, ...]:
        return tuple(attr_name for attr_name in self.class_track_collection_attrs if hasattr(self, attr_name))

    @staticmethod
    def _empty_state_like(value):
        return empty_track_collection_like(value)

    def _class_update_ids(self, dets: np.ndarray) -> list[int]:
        detected_classes = self.class_catalog.detected_ids(dets, self.detection_layout)

        active_classes = {
            cls_id
            for cls_id, state in self.class_track_states.items()
            if state.has_live_tracks()
        }
        return sorted(detected_classes | active_classes)

    def _load_class_track_state(self, cls_id: int) -> None:
        state = self._ensure_class_track_state(cls_id)
        for attr_name in self._class_state_attr_names():
            setattr(self, attr_name, state.attrs[attr_name])

    def _save_class_track_state(self, cls_id: int) -> None:
        state = self._ensure_class_track_state(cls_id)
        for attr_name in self._class_state_attr_names():
            state.attrs[attr_name] = getattr(self, attr_name)

    def _restore_class_track_collections(self) -> None:
        self.active_tracks = self.all_class_tracks("active")
        for group in ("pool", "lost", "removed"):
            for attr_name in track_collection_attrs(group):
                if hasattr(self, attr_name):
                    setattr(self, attr_name, self._flatten_class_state_attr(attr_name))

    def _flatten_class_state_attr(self, attr_name: str) -> list:
        if self.class_track_states is None:
            return []
        values = []
        for state in self.class_track_states.values():
            attr_value = state.attrs.get(attr_name)
            if attr_value:
                values.extend(list(attr_value))
        return values

    def get_class_track_state(self, cls_id: int) -> ClassTrackState:
        return self._ensure_class_track_state(cls_id)

    def get_class_tracks(self, cls_id: int, group: str = "active") -> list:
        """Return class-local tracks by lifecycle group."""
        validate_track_group(group)
        return self.get_class_track_state(cls_id).tracks(group)

    def all_class_tracks(self, group: str = "active") -> list:
        """Return tracks from all classes for one lifecycle group."""
        if self.class_track_states is None:
            return []
        tracks = []
        for state in self.class_track_states.values():
            tracks.extend(state.tracks(group))
        return tracks

    def _precompute_per_frame_cmc(self, cmc, dets: np.ndarray, img: np.ndarray, class_count: int):
        if cmc is None or class_count <= 1 or not callable(getattr(cmc, "apply", None)):
            return None
        warp = cmc.apply(img, self.cmc_detection_boxes(dets))
        return _PrecomputedCMC(warp)

    def _get_class_masks(self, dets: np.ndarray, masks: np.ndarray, cls_id: int):
        """Slice masks by class, matching ``get_class_dets_n_embs``."""
        if masks is None or dets.size == 0:
            return None
        class_indices = np.where(dets[:, self.detection_layout.cls_idx] == cls_id)[0]
        if len(class_indices) == 0:
            return None
        return masks[class_indices]

    def get_class_dets_n_embs(self, dets, embs, cls_id):
        class_dets = self.detection_layout.empty_dets(dtype=np.float32)
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        if dets.size == 0:
            return class_dets, class_embs

        class_indices = np.where(dets[:, self.detection_layout.cls_idx] == cls_id)[0]
        class_dets = dets[class_indices]

        if embs is None:
            return class_dets, class_embs

        assert dets.shape[0] == embs.shape[0], (
            "Detections and embeddings must have the same number of elements when both are provided"
        )
        class_embs = None
        if embs.size > 0:
            class_embs = embs[class_indices]
            self.last_emb_size = class_embs.shape[1]
        return class_dets, class_embs
