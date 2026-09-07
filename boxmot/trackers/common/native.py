"""Tracker-domain adapters for low-level native C++ bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, overload

import numpy as np
import torch

from boxmot.components.timing import timed_component_phase
from boxmot.native.trackers._common import NativeTrackBatch
from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes, Tracks
from boxmot.trackers.common.appearance.live import _REID_OPTION_UNSET, LiveReIDMixin
from boxmot.trackers.common.geometry.obb import align_obb_measurement
from boxmot.trackers.common.input import (
    frame_image_size,
    pack_numpy_track_rows,
    parse_numpy_detection_rows,
    prepare_frame,
)
from boxmot.trackers.config import load_tracker_defaults
from boxmot.trackers.protocols import TrackerRequirements

ASSOCIATION_FUNCTIONS = frozenset({"centroid", "ciou", "diou", "giou", "hmiou", "iou"})


class NativeTrackerLibrary(Protocol):
    """Typed interface implemented by low-level native tracker bindings."""

    def create(self, cfg: Mapping[str, Any]) -> Any: ...

    def destroy(self, handle: Any) -> None: ...

    def reset(self, handle: Any) -> None: ...

    def update(
        self,
        handle: Any,
        *,
        geometry: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        detection_indices: np.ndarray,
        embeddings: np.ndarray | None,
        image: np.ndarray | None,
    ) -> NativeTrackBatch: ...


def resolve_association_function(cfg: dict[str, Any]) -> str:
    """Validate a native tracker's canonical association identifier."""

    asso_func = cfg.get("asso_func", "iou")
    if not isinstance(asso_func, str):
        raise TypeError(f"asso_func must be a string, got {type(asso_func).__name__}.")
    if asso_func != asso_func.strip().lower():
        raise ValueError(f"asso_func must use its canonical lowercase identifier: {asso_func!r}")
    if asso_func not in ASSOCIATION_FUNCTIONS:
        available = ", ".join(sorted(ASSOCIATION_FUNCTIONS))
        raise ValueError(f"Unknown association function {asso_func!r}. Choose from: {available}.")
    cfg["asso_func"] = asso_func
    return asso_func


def association_requires_frame(cfg: Mapping[str, Any]) -> bool:
    """Return whether centroid association requires frame dimensions."""

    return cfg.get("asso_func", "iou") == "centroid"


def load_native_tracker_config(
    tracker_name: str,
    options: dict[str, Any] | None,
    *,
    native_only_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Load defaults and reject options unsupported by one native backend."""

    resolved = load_tracker_defaults(tracker_name)
    if options is not None:
        accepted_keys = set(resolved) | set(native_only_keys)
        unexpected_keys = set(options) - accepted_keys
        if unexpected_keys:
            unexpected = next(key for key in options if key in unexpected_keys)
            raise TypeError(f"Native tracker '{tracker_name}' got an unexpected option {unexpected!r}.")
        resolved.update(options)
    return resolved


def _frame_to_bgr(
    frame: Frame | np.ndarray | None,
    *,
    dimensions_only: bool = False,
    placeholders: dict[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        return frame
    if dimensions_only:
        image = None if placeholders is None else placeholders.get(frame.image_size)
        if image is None:
            image = np.empty((*frame.image_size, 3), dtype=np.uint8)
            if placeholders is not None:
                placeholders[frame.image_size] = image
        return image
    rgb = frame.image.permute(1, 2, 0).numpy()
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _to_tracks(batch: NativeTrackBatch, *, sample_id: str, is_obb: bool) -> Tracks:
    geometry = torch.from_numpy(batch.geometry)
    geometry_value = OrientedBoxes(geometry) if is_obb else Boxes(geometry)
    return Tracks(
        geometry=geometry_value,
        track_ids=torch.from_numpy(batch.track_ids),
        scores=torch.from_numpy(batch.scores),
        class_ids=torch.from_numpy(batch.class_ids),
        detection_indices=torch.from_numpy(batch.detection_indices),
        sample_id=sample_id,
    )


class NativeTrackerAdapter(LiveReIDMixin):
    """Canonical tracker wrapper shared by domain-facing C++ adapters."""

    _native_display_name: str
    supports_obb = True
    supports_masks = False
    accepts_embeddings = False

    def _init_native_handle(
        self,
        *,
        library: NativeTrackerLibrary,
        cfg: dict[str, Any],
        geometry: str,
        use_embeddings: bool,
        requires_frame: bool,
        frame_dimensions_only: bool = False,
        reid_model: Any | None = _REID_OPTION_UNSET,
        reid_weights: str | Path | list[str | Path] | tuple[str | Path, ...] | None = _REID_OPTION_UNSET,
        device: Any = _REID_OPTION_UNSET,
        half: bool = _REID_OPTION_UNSET,
        reid_preprocess: str | None = _REID_OPTION_UNSET,
    ) -> None:
        if geometry not in {"aabb", "obb"}:
            raise ValueError("Native tracker geometry must be 'aabb' or 'obb'.")
        self.name = self.__class__.__name__
        self.cfg = cfg
        self.geometry = geometry
        self.is_obb = geometry == "obb"
        self.use_embeddings = use_embeddings
        self._init_live_reid(
            reid_model=reid_model,
            reid_weights=reid_weights,
            device=device,
            half=half,
            reid_preprocess=reid_preprocess,
        )
        self._requirements = TrackerRequirements(
            embeddings=use_embeddings,
            frame=requires_frame,
            frame_dimensions_only=frame_dimensions_only,
        )
        self._library = library
        self._handle = self._library.create(self.cfg)
        self._obb_output_by_track_id: dict[int, np.ndarray] = {}
        self._dimension_only_images: dict[tuple[int, int], np.ndarray] = {}

    @property
    def requirements(self) -> TrackerRequirements:
        """Return requirements frozen from the resolved tracker configuration."""

        return self._requirements

    @overload
    def update(self, detections: Detections, frame: Frame | np.ndarray | None = None) -> Tracks: ...

    @overload
    def update(self, detections: np.ndarray, frame: Frame | np.ndarray | None = None) -> np.ndarray: ...

    def update(
        self, detections: Detections | np.ndarray, frame: Frame | np.ndarray | None = None
    ) -> Tracks | np.ndarray:
        """Invoke a native update with an optional Frame or uint8 HWC BGR image.

        The detection representation determines the output representation.
        """

        frame = prepare_frame(frame)

        is_obb = self.is_obb
        numpy_input = type(detections) is np.ndarray
        canonical_detections = detections if isinstance(detections, Detections) else None
        if isinstance(detections, Detections):
            # Canonical dataclasses are frozen, but their tensor storage remains
            # mutable. Revalidate before exposing that storage to ctypes.
            detections.validate()
            if detections.is_obb != is_obb:
                raise ValueError(f"Native {self._native_display_name} is fixed to {self.geometry.upper()} geometry.")
            if isinstance(frame, Frame) and frame.sample_id != detections.sample_id:
                raise ValueError("Frame and detections must have the same sample_id.")
            if (
                detections.masks is not None
                and frame is not None
                and detections.masks.image_size != frame_image_size(frame)
            ):
                raise ValueError(
                    "Detection masks must match the frame spatial size, "
                    f"got {detections.masks.image_size} and {frame_image_size(frame)}."
                )
            sample_id = detections.sample_id
            geometry = detections.geometry.values.detach().numpy()
            scores = detections.scores.detach().numpy()
            class_ids = detections.class_ids.detach().numpy()
            embeddings = (
                detections.embeddings.detach().numpy()
                if self.use_embeddings and detections.embeddings is not None
                else None
            )
        else:
            rows = parse_numpy_detection_rows(detections, is_obb=is_obb)
            sample_id = None
            geometry = rows.geometry
            scores = rows.scores
            class_ids = rows.class_ids
            embeddings = None

        prepared_bgr = None
        if (
            embeddings is None
            and self.generates_embeddings
            and len(geometry)
            and frame is not None
            and self._reid_encoder_spec is None
        ):
            with timed_component_phase("reid", "preprocess", device=self._reid_device):
                prepared_bgr = self._frame_to_bgr(frame)

        embeddings = self._resolve_input_embeddings(
            geometry=geometry,
            embeddings=embeddings,
            frame=frame,
            detections=canonical_detections,
            scores=scores,
            class_ids=class_ids,
            prepared_bgr=prepared_bgr,
        )
        if self.requirements.embeddings and embeddings is None:
            raise ValueError(f"Native {self._native_display_name} requires detection embeddings.")
        if self.requirements.frame and frame is None:
            raise ValueError(f"Native {self._native_display_name} requires a frame.")

        self._mark_live_reid_updated()
        image = prepared_bgr
        if frame is not None and (image is None or self.requirements.frame_dimensions_only):
            image = _frame_to_bgr(
                frame,
                dimensions_only=self.requirements.frame_dimensions_only,
                placeholders=self._dimension_only_images,
            )
        batch = self._library.update(
            self._handle,
            geometry=geometry,
            scores=scores,
            class_ids=class_ids,
            detection_indices=np.arange(len(scores), dtype=np.int64),
            embeddings=embeddings,
            image=image,
        )
        output_geometry = batch.geometry
        if is_obb and len(batch.track_ids):
            output_geometry = batch.geometry.copy()
            for index, track_id in enumerate(batch.track_ids.tolist()):
                previous = self._obb_output_by_track_id.get(track_id)
                if previous is not None:
                    output_geometry[index] = align_obb_measurement(output_geometry[index], previous)
                self._obb_output_by_track_id[track_id] = output_geometry[index].copy()
        if batch.detection_indices.size and np.any(batch.detection_indices >= len(scores)):
            raise ValueError(
                f"Native {self._native_display_name} returned a detection index outside the current batch."
            )

        if numpy_input:
            return pack_numpy_track_rows(
                geometry=output_geometry,
                track_ids=batch.track_ids,
                scores=batch.scores,
                class_ids=batch.class_ids,
                detection_indices=batch.detection_indices,
                is_obb=is_obb,
                detection_count=len(scores),
                owner=f"Native {self._native_display_name}",
            )

        assert sample_id is not None
        output_batch = NativeTrackBatch(
            geometry=output_geometry,
            scores=batch.scores,
            track_ids=batch.track_ids,
            class_ids=batch.class_ids,
            detection_indices=batch.detection_indices,
        )
        return _to_tracks(output_batch, sample_id=sample_id, is_obb=is_obb)

    def reset(self) -> None:
        """Reset native state and OBB continuity state."""

        self._library.reset(self._handle)
        self._obb_output_by_track_id.clear()
        self._dimension_only_images.clear()
        self._reset_live_reid_sequence()

    def close(self) -> None:
        """Release the native tracker handle."""

        handle = getattr(self, "_handle", None)
        if handle:
            self._library.destroy(handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown is nondeterministic
        try:
            self.close()
        except Exception:
            pass


__all__ = (
    "NativeTrackerAdapter",
    "NativeTrackerLibrary",
    "association_requires_frame",
    "load_native_tracker_config",
    "resolve_association_function",
)
