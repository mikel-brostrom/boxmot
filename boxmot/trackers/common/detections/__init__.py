from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxmot.trackers.common.detections.layout import (
    AABB_DETECTIONS,
    OBB_DETECTIONS,
    AxisAlignedDetections,
    DetectionLayout,
    OrientedDetections,
    get_detection_layout,
    infer_detection_layout,
)
from boxmot.trackers.common.tracking.records import DetectionRecord


def _validate_aligned_optional(name: str, values: np.ndarray | None, size: int) -> None:
    if values is not None and len(values) != size:
        raise ValueError(f"{name} must be aligned with detections")


@dataclass(frozen=True)
class DetectionBatch:
    """Immutable view of detections parsed through a detection layout."""

    boxes: np.ndarray
    confs: np.ndarray
    clss: np.ndarray
    det_inds: np.ndarray
    embs: np.ndarray | None = None
    masks: np.ndarray | None = None

    @classmethod
    def from_layout(
        cls,
        dets: np.ndarray,
        layout: DetectionLayout,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        copy: bool = True,
    ) -> DetectionBatch:
        """Parse raw or det-indexed detections using ``layout``."""
        dets = np.asarray(dets)
        if dets.ndim != 2:
            raise ValueError(f"Detections must be a 2D array, got shape {dets.shape}")
        if dets.shape[1] not in (layout.det_cols, layout.det_cols + 1):
            raise ValueError(
                "Unsupported detection column count "
                f"{dets.shape[1]}; expected {layout.det_cols} or {layout.det_cols + 1}"
            )

        size = len(dets)
        _validate_aligned_optional("Embeddings", embs, size)
        _validate_aligned_optional("Masks", masks, size)

        boxes = layout.boxes(dets)
        confs = layout.confidences(dets)
        clss = layout.classes(dets)
        if dets.shape[1] == layout.det_cols + 1:
            det_inds = dets[:, layout.det_cols].astype(np.int32, copy=copy)
        else:
            det_inds = np.arange(size, dtype=np.int32)

        embs_arr = None if embs is None else np.asarray(embs)
        masks_arr = None if masks is None else np.asarray(masks)
        if copy:
            boxes = boxes.copy()
            confs = confs.copy()
            clss = clss.copy()
            if embs_arr is not None:
                embs_arr = embs_arr.copy()
            if masks_arr is not None:
                masks_arr = masks_arr.copy()

        return cls(
            boxes=boxes,
            confs=confs,
            clss=clss,
            det_inds=det_inds,
            embs=embs_arr,
            masks=masks_arr,
        )

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def select(self, indices: np.ndarray | list[int] | list[bool]) -> DetectionBatch:
        """Return a detection batch filtered by integer indices or a boolean mask."""
        return DetectionBatch(
            boxes=self.boxes[indices],
            confs=self.confs[indices],
            clss=self.clss[indices],
            det_inds=self.det_inds[indices],
            embs=None if self.embs is None else self.embs[indices],
            masks=None if self.masks is None else self.masks[indices],
        )

    def split_by_confidence(
        self,
        high_thresh: float,
        low_thresh: float | None = None,
    ) -> tuple[DetectionBatch, DetectionBatch]:
        """Return high-confidence and optional second-stage detections.

        The split follows the tracker convention used by ByteTrack-like update
        loops: high-confidence detections satisfy ``conf > high_thresh`` and
        second-stage detections satisfy ``low_thresh < conf < high_thresh``.
        """
        high = self.select(self.confs > high_thresh)
        if low_thresh is None:
            low = self.select(np.zeros(len(self), dtype=bool))
        else:
            low = self.select((self.confs > low_thresh) & (self.confs < high_thresh))
        return high, low

    def with_confs(self, confs: np.ndarray) -> DetectionBatch:
        """Return a copy of this batch with updated confidence scores."""
        confs = np.asarray(confs)
        _validate_aligned_optional("Confidences", confs, len(self))
        return DetectionBatch(
            boxes=self.boxes,
            confs=confs.astype(self.confs.dtype, copy=True),
            clss=self.clss,
            det_inds=self.det_inds,
            embs=self.embs,
            masks=self.masks,
        )

    def with_embs(self, embs: np.ndarray | None) -> DetectionBatch:
        """Return a copy of this batch with updated aligned embeddings."""
        if embs is None:
            embs_arr = None
        else:
            embs_arr = np.asarray(embs)
            _validate_aligned_optional("Embeddings", embs_arr, len(self))
            embs_arr = embs_arr.copy()
        return DetectionBatch(
            boxes=self.boxes,
            confs=self.confs,
            clss=self.clss,
            det_inds=self.det_inds,
            embs=embs_arr,
            masks=self.masks,
        )

    def as_records(self) -> list[DetectionRecord]:
        """Return one record per detection for code that prefers scalar records."""
        return [
            DetectionRecord(
                box=self.boxes[i].copy(),
                conf=float(self.confs[i]),
                cls=int(self.clss[i]),
                det_ind=int(self.det_inds[i]),
                emb=None if self.embs is None else self.embs[i],
                mask=None if self.masks is None else self.masks[i],
            )
            for i in range(len(self))
        ]

    def as_indexed_detections(self, dtype=np.float32) -> np.ndarray:
        """Reconstruct ``[box..., conf, cls, det_ind]`` detections."""
        if len(self) == 0:
            return np.empty((0, self.boxes.shape[1] + 3), dtype=dtype)
        return np.column_stack((self.boxes, self.confs, self.clss, self.det_inds)).astype(dtype, copy=False)

    def as_box_conf_detections(self, dtype=np.float32) -> np.ndarray:
        """Reconstruct ``[box..., conf]`` detections for AABB-only association."""
        if len(self) == 0:
            return np.empty((0, self.boxes.shape[1] + 1), dtype=dtype)
        return np.column_stack((self.boxes, self.confs)).astype(dtype, copy=False)


__all__ = (
    "AABB_DETECTIONS",
    "OBB_DETECTIONS",
    "AxisAlignedDetections",
    "DetectionBatch",
    "DetectionLayout",
    "OrientedDetections",
    "get_detection_layout",
    "infer_detection_layout",
)
