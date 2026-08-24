from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from boxmot.trackers.common.geometry.obb import xywha_to_corners, xywha_to_xyxy


class TrackResults(np.ndarray):
    """Thin zero-copy view over the (N, 8) or (N, 9) tracker output array.

    Provides named property accessors and export methods while remaining fully
    compatible with numpy operations (slicing, indexing, stacking, etc.).

    AABB columns (8): x1, y1, x2, y2, id, conf, cls, det_ind
    OBB  columns (9): cx, cy, w, h, angle, id, conf, cls, det_ind
    """

    def __new__(cls, data: np.ndarray, masks: np.ndarray = None) -> TrackResults:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1 and arr.size > 0:
            arr = arr.reshape(1, -1)
        elif arr.size == 0:
            cols = arr.shape[1] if arr.ndim == 2 else 0
            arr = arr.reshape(0, cols)
        obj = arr.view(cls)
        obj._masks = masks
        return obj

    def __array_finalize__(self, obj):
        self._masks = getattr(obj, "_masks", None)

    def __getitem__(self, key):
        """Slice optional masks with the same row selection as track rows."""
        result = super().__getitem__(key)
        masks = self._masks
        if not isinstance(result, TrackResults) or masks is None:
            return result
        if self.ndim != 2:
            # A one-dimensional row may subsequently be indexed by its
            # columns (NumPy testing utilities do this with boolean masks).
            # Its already-selected object mask remains the correct metadata;
            # that column operation is not another track-row selection.
            return result

        row_key = key[0] if isinstance(key, tuple) else key
        selected = np.asarray(masks)[row_key]
        if np.asarray(selected).ndim == np.asarray(masks).ndim - 1:
            selected = np.expand_dims(selected, axis=0)
        result._masks = selected
        return result

    @property
    def masks(self) -> np.ndarray | None:
        """Segmentation masks for tracked objects, shape (M, H, W) or None."""
        return self._masks

    @property
    def is_obb(self) -> bool:
        """Whether the results contain oriented bounding boxes."""
        return self.shape[1] >= 9 if self.ndim == 2 else False

    # ------------------------------------------------------------------
    # Box geometry
    # ------------------------------------------------------------------

    @property
    def boxes(self) -> np.ndarray:
        """Return native geometry: ``xyxy`` for AABB or ``xywha`` for OBB."""
        return self.xywha if self.is_obb else self.xyxy

    @property
    def xyxy(self) -> np.ndarray:
        """Return AABBs, enclosing each oriented track when in OBB mode."""
        if self.is_obb:
            return xywha_to_xyxy(np.asarray(self[:, :5]))
        return np.asarray(self[:, :4])

    @property
    def xywh(self) -> np.ndarray:
        """Bounding boxes as (x_center, y_center, width, height)."""
        boxes = np.asarray(self[:, :4])
        if boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        if self.is_obb:
            return boxes
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    @property
    def xywha(self) -> np.ndarray:
        """Oriented boxes as (cx, cy, w, h, angle). OBB mode only."""
        if not self.is_obb:
            return np.empty((len(self), 0), dtype=np.float32)
        return np.asarray(self[:, :5])

    # ------------------------------------------------------------------
    # Track metadata
    # ------------------------------------------------------------------

    @property
    def id(self) -> np.ndarray:
        """Integer track IDs."""
        col = 5 if self.is_obb else 4
        return np.asarray(self[:, col], dtype=int)

    @property
    def conf(self) -> np.ndarray:
        """Detection confidence scores."""
        col = 6 if self.is_obb else 5
        return np.asarray(self[:, col])

    @property
    def cls(self) -> np.ndarray:
        """Integer class IDs."""
        col = 7 if self.is_obb else 6
        return np.asarray(self[:, col], dtype=int)

    @property
    def det_ind(self) -> np.ndarray:
        """Detection indices mapping tracks back to input detections (-1 if unmatched)."""
        col = 8 if self.is_obb else 7
        return np.asarray(self[:, col], dtype=int)

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    @property
    def _csv_fields(self) -> list[str]:
        """Column names for CSV export."""
        if self.is_obb:
            return ["cx", "cy", "w", "h", "angle", "id", "conf", "cls", "det_ind"]
        return ["x1", "y1", "x2", "y2", "id", "conf", "cls", "det_ind"]

    def _row(self, i: int) -> list[Any]:
        """Build a single export row from named accessors."""
        box = [float(v) for v in (self.xywha[i] if self.is_obb else self.xyxy[i])]
        return box + [int(self.id[i]), float(self.conf[i]), int(self.cls[i]), int(self.det_ind[i])]

    def summary(self) -> list[dict[str, Any]]:
        """Convert track results to a list of dictionaries.

        Returns:
            list[dict]: One dict per track with keys: id, conf, cls,
                and either 'box' with x1/y1/x2/y2 (AABB) or cx/cy/w/h/angle (OBB).
        """
        results = []
        for i in range(len(self)):
            entry: dict[str, Any] = {"id": int(self.id[i]), "conf": float(self.conf[i]), "cls": int(self.cls[i])}
            if self.is_obb:
                cx, cy, w, h, angle = self.xywha[i]
                entry["box"] = {"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h), "angle": float(angle)}
            else:
                x1, y1, x2, y2 = self.xyxy[i]
                entry["box"] = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
            results.append(entry)
        return results

    def to_json(self, indent: int | None = None) -> str:
        """Convert track results to a JSON string.

        Args:
            indent: JSON indentation level. None for compact output.

        Returns:
            str: JSON-encoded string of the track summaries.
        """
        return json.dumps(self.summary(), indent=indent)

    def to_csv(self, frame_id: int | None = None) -> str:
        """Convert track results to CSV-formatted string.

        Args:
            frame_id: Optional frame number to include as the first column.

        Returns:
            str: CSV string with one row per track.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        for i in range(len(self)):
            row = [frame_id] + self._row(i) if frame_id is not None else self._row(i)
            writer.writerow(row)
        return buf.getvalue()

    def save_csv(self, path: str | Path, frame_id: int | None = None, header: bool = True) -> None:
        """Append track results to a CSV file.

        Args:
            path: File path to write/append to.
            frame_id: Optional frame number to include as the first column.
            header: Write header row if the file doesn't exist yet.
        """
        path = Path(path)
        write_header = header and not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", newline="") as f:
            if write_header:
                fields = (["frame"] + self._csv_fields) if frame_id is not None else self._csv_fields
                csv.writer(f).writerow(fields)
            f.write(self.to_csv(frame_id=frame_id))

    def save_mot(self, path: str | Path, frame_id: int = 0) -> None:
        """Append track results in canonical MOT or corner-based MMOT format.

        Args:
            path: File path to append to.
            frame_id: Frame index to serialize.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not len(self):
            path.touch(exist_ok=True)
            return

        frame = np.full((len(self), 1), frame_id, dtype=np.float32)
        track_ids = self.id.reshape(-1, 1).astype(np.float32)
        confidence = self.conf.reshape(-1, 1).astype(np.float32)
        det_ind = self.det_ind.reshape(-1, 1).astype(np.float32)
        if self.is_obb:
            rows = np.column_stack(
                (frame, track_ids, xywha_to_corners(self.xywha), confidence, self.cls, det_ind)
            )
            fmt = "%d,%d," + ",".join(["%.6f"] * 9) + ",%d,%d"
        else:
            xyxy = self.xyxy
            ltwh = np.rint(
                np.column_stack(
                    (xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1])
                )
            ).astype(np.int32)
            rows = np.column_stack((frame, track_ids, ltwh, confidence, self.cls + 1, det_ind))
            fmt = "%d,%d,%d,%d,%d,%d,%.6f,%d,%d"
        with open(path, "a") as file:
            np.savetxt(file, rows, fmt=fmt)
