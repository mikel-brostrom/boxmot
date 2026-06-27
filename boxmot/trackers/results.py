from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


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
        self._masks = getattr(obj, '_masks', None)

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
    def xyxy(self) -> np.ndarray:
        """Bounding boxes as (x1, y1, x2, y2). AABB mode only."""
        return np.asarray(self[:, :4])

    @property
    def xywh(self) -> np.ndarray:
        """Bounding boxes as (x_center, y_center, width, height). AABB mode only."""
        boxes = np.asarray(self[:, :4])
        if boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    @property
    def xywha(self) -> np.ndarray:
        """Oriented boxes as (cx, cy, w, h, angle). OBB mode only."""
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
        """Append track results in MOT challenge format to a text file.

        Format: frame_id, track_id, left, top, width, height, conf, cls, -1
        Track IDs are converted from BoxMOT's internal 0-based IDs to MOT's
        1-based IDs during export.

        Args:
            path: File path to append to.
            frame_id: 1-based frame index.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a") as f:
            for i in range(len(self)):
                mot_id = int(self.id[i]) + 1
                if self.is_obb:
                    cx, cy, w, h, angle = self.xywha[i]
                    f.write(f"{frame_id},{mot_id},{cx:.2f},{cy:.2f},{w:.2f},{h:.2f},"
                            f"{angle:.4f},{self.conf[i]:.6f},{int(self.cls[i])},-1\n")
                else:
                    x1, y1, x2, y2 = self.xyxy[i]
                    w, h = x2 - x1, y2 - y1
                    f.write(f"{frame_id},{mot_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                            f"{self.conf[i]:.6f},{int(self.cls[i])},-1\n")
