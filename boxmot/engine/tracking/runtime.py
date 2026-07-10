from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np

from boxmot.engine.tracking.mot import convert_to_mmot_obb_format, convert_to_mot_format
from boxmot.trackers.registry import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.trackers.results import TrackResults
from boxmot.utils.timing import TimingStats, wrap_tracker_reid


class TrackerRuntime:
    """Wrap one tracker instance with timing and formatting helpers."""

    def __init__(self, tracker: Any, timing_stats: TimingStats | None = None) -> None:
        self.tracker = tracker
        self.timing_stats = timing_stats
        self._accepts_embs = True
        self._accepts_masks = True
        self._inspect_update_signature()

    def _inspect_update_signature(self) -> None:
        try:
            signature = inspect.signature(self.tracker.update)
        except (ValueError, TypeError):
            return

        params = signature.parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        self._accepts_embs = "embs" in params or accepts_kwargs
        self._accepts_masks = "masks" in params or accepts_kwargs

    @classmethod
    def create(
        cls,
        tracker_name: str,
        reid_weights,
        device,
        half: bool,
        per_class: bool,
        evolve_param_dict: dict | None = None,
        target_id: int | None = None,
        timing_stats: TimingStats | None = None,
        reid_preprocess: str | None = None,
        class_ids: tuple[int, ...] | None = None,
        class_names: dict[int, str] | None = None,
        precomputed_reid: bool = False,
    ) -> "TrackerRuntime":
        normalized_tracker = str(tracker_name).lower()
        if normalized_tracker not in TRACKER_MAPPING:
            available = ", ".join(sorted(TRACKER_MAPPING))
            raise ValueError(f"'{tracker_name}' is not supported. Supported ones are {available}")

        tracker = create_tracker(
            tracker_type=normalized_tracker,
            tracker_config=get_tracker_config(normalized_tracker),
            reid_weights=reid_weights,
            device=device,
            half=half,
            per_class=per_class,
            class_ids=class_ids,
            class_names=class_names,
            evolve_param_dict=evolve_param_dict,
            reid_preprocess=reid_preprocess,
            precomputed_reid=precomputed_reid,
        )
        if target_id is not None:
            tracker.target_id = target_id
        if timing_stats is not None:
            wrap_tracker_reid(tracker, timing_stats)
        return cls(tracker, timing_stats=timing_stats)

    @staticmethod
    def _ensure_2d_tracks(tracks: np.ndarray) -> np.ndarray:
        arr = np.asarray(tracks, dtype=np.float32)
        if arr.size == 0:
            if arr.ndim == 2:
                return arr
            return np.empty((0, 0), dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    @staticmethod
    def format_for_mot(tracks: np.ndarray, frame_idx: int) -> np.ndarray:
        track_results = TrackResults(TrackerRuntime._ensure_2d_tracks(tracks))
        if track_results.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        if track_results.is_obb:
            return convert_to_mmot_obb_format(track_results, frame_idx)
        return convert_to_mot_format(track_results, frame_idx)

    @property
    def names(self):
        return getattr(self.tracker, "names", None)

    @names.setter
    def names(self, value) -> None:
        setattr(self.tracker, "names", value)

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        started = False
        if self.timing_stats is not None:
            self.timing_stats.reset_frame_reid()
            self.timing_stats.start_tracking()
            started = True
        else:
            start_time = time.perf_counter()

        try:
            kwargs = {}
            if embs is not None and self._accepts_embs:
                kwargs["embs"] = embs
            if masks is not None and self._accepts_masks:
                kwargs["masks"] = masks

            tracks = self.tracker.update(dets, img, **kwargs) if kwargs else self.tracker.update(dets, img)
        finally:
            if started:
                self.timing_stats.end_tracking()
                elapsed_ms = self.timing_stats.get_last_track_time()
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

        return self._ensure_2d_tracks(tracks), elapsed_ms

    def plot_results(
        self,
        img: np.ndarray,
        show_trajectories: bool,
        *,
        thickness: int = 2,
        show_kf_preds: bool = False,
    ) -> np.ndarray:
        if hasattr(self.tracker, "plot_results"):
            return self.tracker.plot_results(
                img,
                show_trajectories,
                thickness=thickness,
                show_kf_preds=show_kf_preds,
            )
        return img

    def __getattr__(self, name: str):
        return getattr(self.tracker, name)


__all__ = ("TrackerRuntime",)
