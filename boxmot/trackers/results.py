from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from boxmot.box_schema import BoxSchema, get_box_schema_for_mode, schema_from_track_columns
from boxmot.trackers.common.geometry.obb import xywha_to_corners, xywha_to_xyxy


def _restore_track_results(data: np.ndarray, masks: np.ndarray | None, schema: BoxSchema) -> "TrackResults":
    """Reconstruct validated tracker results for pickle/copy protocols."""
    return TrackResults(data, masks=masks, schema=schema)


class TrackResults(np.ndarray):
    """Thin zero-copy view over the (N, 8) or (N, 9) tracker output array.

    Provides named property accessors and export methods. Complete row slices
    preserve this type and aligned masks; transformations that change the row
    contract deliberately return plain NumPy arrays.

    AABB columns (8): x1, y1, x2, y2, id, conf, cls, det_ind
    OBB  columns (9): cx, cy, w, h, angle, id, conf, cls, det_ind
    """

    def __new__(
        cls,
        data: np.ndarray,
        masks: np.ndarray = None,
        *,
        schema: BoxSchema | None = None,
        is_obb: bool | None = None,
    ) -> TrackResults:
        arr = np.asarray(data, dtype=np.float32)
        if schema is not None and is_obb is not None:
            requested = get_box_schema_for_mode(is_obb)
            if requested != schema:
                raise ValueError("schema and is_obb describe different tracker output modes.")
        if schema is None and is_obb is not None:
            schema = get_box_schema_for_mode(is_obb)

        if arr.ndim == 1:
            if arr.size == 0:
                if schema is None:
                    raise ValueError("Empty 1D tracker output is ambiguous; provide an 8- or 9-column array.")
                arr = schema.empty_tracks()
            else:
                arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Tracker output must be a 2D array, got shape {arr.shape}.")

        inferred = schema_from_track_columns(arr.shape[1])
        if schema is None:
            schema = inferred
        elif inferred != schema:
            raise ValueError(
                f"Tracker output has {arr.shape[1]} columns, but {schema.box_type.value} requires {schema.track_cols}."
            )

        if arr.size and not np.isfinite(arr).all():
            raise ValueError("Tracker output must contain only finite values.")
        if arr.size:
            geometry = arr[:, : schema.geometry_cols]
            if schema.is_obb:
                if np.any(geometry[:, 2:4] <= 0):
                    raise ValueError("OBB tracker output must have positive width and height.")
            elif np.any(geometry[:, 2] <= geometry[:, 0]) or np.any(geometry[:, 3] <= geometry[:, 1]):
                raise ValueError("AABB tracker output must satisfy x2 > x1 and y2 > y1.")

            for label, index in (
                ("track IDs", schema.track_id_index),
                ("class IDs", schema.track_class_index),
                ("detection indices", schema.track_detection_index),
            ):
                values = arr[:, index]
                if not np.equal(values, np.floor(values)).all():
                    raise ValueError(f"Tracker output {label} must be integers.")

        masks_arr = None if masks is None else np.asarray(masks)
        if masks_arr is not None:
            if masks_arr.ndim != 3:
                raise ValueError(f"Tracker masks must have shape (N, H, W), got {masks_arr.shape}.")
            if len(masks_arr) != len(arr):
                raise ValueError(
                    f"Tracker mask count must match output rows, got masks={len(masks_arr)} tracks={len(arr)}."
                )
        obj = arr.view(cls)
        obj._masks = masks_arr
        obj._schema = schema
        return obj

    def __array_finalize__(self, obj):
        # NumPy invokes this hook for many operations that can reorder rows
        # without exposing their indices (take, roll, delete, sort, ...).
        # Metadata is restored explicitly only by known row-preserving paths.
        self._schema = None
        self._masks = None

    @staticmethod
    def _plain_array_tree(value):
        if isinstance(value, TrackResults):
            return np.asarray(value)
        if isinstance(value, tuple):
            return tuple(TrackResults._plain_array_tree(item) for item in value)
        if isinstance(value, list):
            return [TrackResults._plain_array_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: TrackResults._plain_array_tree(item) for key, item in value.items()}
        return value

    def __array_function__(self, func, types, args, kwargs):
        """Keep arbitrary NumPy transformations outside the typed row wrapper."""
        del types
        return func(
            *self._plain_array_tree(args),
            **self._plain_array_tree(kwargs),
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Return plain arrays for computations that do not preserve track semantics."""
        array_inputs = tuple(np.asarray(value) if isinstance(value, TrackResults) else value for value in inputs)
        if "out" in kwargs and kwargs["out"] is not None:
            kwargs["out"] = tuple(
                np.asarray(value) if isinstance(value, TrackResults) else value for value in kwargs["out"]
            )
        return getattr(ufunc, method)(*array_inputs, **kwargs)

    def reshape(self, *shape, **kwargs) -> np.ndarray:
        """Reshaping changes the row contract, so return a plain ndarray."""
        return np.asarray(self).reshape(*shape, **kwargs)

    def transpose(self, *axes) -> np.ndarray:
        """Transposition changes the row contract, so return a plain ndarray."""
        return np.asarray(self).transpose(*axes)

    @property
    def T(self) -> np.ndarray:  # noqa: N802 - NumPy-compatible public attribute
        return np.asarray(self).T

    def ravel(self, order: str = "C") -> np.ndarray:
        return np.asarray(self).ravel(order)

    def flatten(self, order: str = "C") -> np.ndarray:
        return np.asarray(self).flatten(order)

    def squeeze(self, axis=None) -> np.ndarray:
        return np.asarray(self).squeeze(axis=axis)

    def swapaxes(self, axis1: int, axis2: int) -> np.ndarray:
        return np.asarray(self).swapaxes(axis1, axis2)

    def take(self, indices, axis=None, out=None, mode="raise") -> np.ndarray:
        return np.asarray(self).take(indices, axis=axis, out=out, mode=mode)

    def repeat(self, repeats, axis=None) -> np.ndarray:
        return np.asarray(self).repeat(repeats, axis=axis)

    def compress(self, condition, axis=None, out=None) -> np.ndarray:
        return np.asarray(self).compress(condition, axis=axis, out=out)

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True) -> np.ndarray:
        del subok
        return np.asarray(self).astype(dtype, order=order, casting=casting, subok=False, copy=copy)

    def byteswap(self, inplace=False) -> np.ndarray:
        return np.asarray(self).byteswap(inplace=inplace)

    def view(self, dtype=None, type=None) -> np.ndarray:
        values = np.asarray(self)
        if dtype is None and type is None:
            return values.view()
        if type is None:
            return values.view(dtype=dtype)
        if dtype is None:
            return values.view(type=type)
        return values.view(dtype=dtype, type=type)

    def getfield(self, dtype=None, offset=0) -> np.ndarray:
        return np.asarray(self).getfield(dtype=dtype, offset=offset)

    def copy(self, order="C") -> TrackResults:
        masks = None if self._masks is None else np.array(self._masks, copy=True)
        return TrackResults(np.array(self, copy=True, order=order), masks=masks, schema=self.schema)

    def __copy__(self) -> TrackResults:
        return self.copy()

    def __deepcopy__(self, memo) -> TrackResults:
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    def __reduce__(self):
        masks = None if self._masks is None else np.array(self._masks, copy=True)
        return _restore_track_results, (np.array(self, copy=True), masks, self.schema)

    def __getitem__(self, key):
        """Slice optional masks with the same row selection as track rows."""
        result = super().__getitem__(key)
        masks = self._masks
        if not isinstance(result, TrackResults):
            return result

        if self.ndim != 2 or result.ndim != 2 or result.shape[1] != self.schema.track_cols:
            return np.asarray(result)

        if self.ndim == 2 and isinstance(key, tuple):
            if len(key) != 2:
                return np.asarray(result)
            column_key = key[1]
            full_columns = column_key is Ellipsis or (
                isinstance(column_key, slice)
                and column_key.start is None
                and column_key.stop is None
                and column_key.step is None
            )
            if not full_columns:
                return np.asarray(result)
        elif self.ndim == 2:
            key_array = np.asarray(key) if isinstance(key, (list, np.ndarray)) else None
            if key is None or (key_array is not None and key_array.ndim != 1):
                return np.asarray(result)

        result._schema = self.schema
        if masks is None:
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
    def schema(self) -> BoxSchema:
        """Canonical schema carried by this result, including empty results."""
        if self._schema is None:
            raise ValueError("Tracker result schema metadata is unavailable.")
        if self.ndim != 2 or self.shape[1] != self._schema.track_cols:
            raise ValueError(
                f"Tracker result shape {self.shape} no longer matches its {self._schema.box_type.value} schema."
            )
        if self._masks is not None and len(self._masks) != self.shape[0]:
            raise ValueError(f"Tracker mask count {len(self._masks)} no longer matches {self.shape[0]} result rows.")
        return self._schema

    @property
    def is_obb(self) -> bool:
        """Whether the results contain oriented bounding boxes."""
        return self.schema.is_obb

    def _rows(self) -> np.ndarray:
        values = np.asarray(self)
        if values.ndim == 1:
            return values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(f"Tracker results must have one or two dimensions, got {values.shape}.")
        return values

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
        rows = self._rows()
        if self.is_obb:
            return xywha_to_xyxy(rows[:, : self.schema.geometry_cols])
        return rows[:, : self.schema.geometry_cols]

    @property
    def xywh(self) -> np.ndarray:
        """Bounding boxes as (x_center, y_center, width, height)."""
        boxes = self._rows()[:, :4]
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
        return self._rows()[:, : self.schema.geometry_cols]

    # ------------------------------------------------------------------
    # Track metadata
    # ------------------------------------------------------------------

    @property
    def id(self) -> np.ndarray:
        """Integer track IDs."""
        return np.asarray(self._rows()[:, self.schema.track_id_index], dtype=int)

    @property
    def conf(self) -> np.ndarray:
        """Detection confidence scores."""
        return np.asarray(self._rows()[:, self.schema.track_conf_index])

    @property
    def cls(self) -> np.ndarray:
        """Integer class IDs."""
        return np.asarray(self._rows()[:, self.schema.track_class_index], dtype=int)

    @property
    def det_ind(self) -> np.ndarray:
        """Detection indices mapping tracks back to input detections (-1 if unmatched)."""
        return np.asarray(self._rows()[:, self.schema.track_detection_index], dtype=int)

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
            rows = np.column_stack((frame, track_ids, xywha_to_corners(self.xywha), confidence, self.cls, det_ind))
            fmt = "%d,%d," + ",".join(["%.6f"] * 9) + ",%d,%d"
        else:
            xyxy = self.xyxy
            ltwh = np.rint(
                np.column_stack((xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]))
            ).astype(np.int32)
            rows = np.column_stack((frame, track_ids, ltwh, confidence, self.cls + 1, det_ind))
            fmt = "%d,%d,%d,%d,%d,%d,%.6f,%d,%d"
        with open(path, "a") as file:
            np.savetxt(file, rows, fmt=fmt)
