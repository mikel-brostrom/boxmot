from __future__ import annotations

import numpy as np
import torch

from boxmot.native.trackers._common import NativeTrackBatch
from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes, Tracks


def empty_native_batch(geometry_columns: int) -> NativeTrackBatch:
    """Create an empty low-level native result for adapter tests."""

    return NativeTrackBatch(
        geometry=np.empty((0, geometry_columns), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        track_ids=np.empty((0,), dtype=np.int64),
        class_ids=np.empty((0,), dtype=np.int64),
        detection_indices=np.empty((0,), dtype=np.int64),
    )


def detections_from_rows(
    rows: np.ndarray,
    *,
    sample_id: str = "sample",
    embeddings: np.ndarray | None = None,
) -> Detections:
    """Explicit test-boundary conversion from legacy rows to structures."""
    values = np.ascontiguousarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in {6, 7}:
        raise ValueError(f"Expected AABB6 or OBB7 rows, got {values.shape}.")
    geometry_cols = 5 if values.shape[1] == 7 else 4
    geometry_tensor = torch.from_numpy(values[:, :geometry_cols].copy())
    geometry = OrientedBoxes(geometry_tensor) if geometry_cols == 5 else Boxes(geometry_tensor)
    embedding_tensor = None
    if embeddings is not None:
        embedding_tensor = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
    return Detections(
        geometry=geometry,
        scores=torch.from_numpy(values[:, geometry_cols].copy()),
        class_ids=torch.from_numpy(values[:, geometry_cols + 1].astype(np.int64, copy=True)),
        sample_id=sample_id,
        embeddings=embedding_tensor,
    )


def frame_from_bgr(image: np.ndarray, *, sample_id: str = "sample") -> Frame:
    """Explicit test-boundary conversion from HWC BGR to canonical CHW RGB."""
    bgr = np.ascontiguousarray(image, dtype=np.uint8)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected an HWC BGR image, got {bgr.shape}.")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return Frame(torch.from_numpy(rgb).permute(2, 0, 1).contiguous(), sample_id=sample_id)


def update_tracks(
    tracker,
    rows: np.ndarray,
    image: np.ndarray | None = None,
    *,
    sample_id: str = "sample",
    embeddings: np.ndarray | None = None,
) -> Tracks:
    detections = detections_from_rows(rows, sample_id=sample_id, embeddings=embeddings)
    frame = None if image is None else frame_from_bgr(image, sample_id=sample_id)
    return tracker.update(detections, frame)


def update_rows(
    tracker,
    rows: np.ndarray,
    image: np.ndarray | None = None,
    *,
    sample_id: str = "sample",
    embeddings: np.ndarray | None = None,
) -> np.ndarray:
    tracks = update_tracks(
        tracker,
        rows,
        image,
        sample_id=sample_id,
        embeddings=embeddings,
    )
    serialized = tracks.to_obb_rows() if tracks.is_obb else tracks.to_aabb_rows()
    return serialized.numpy()
