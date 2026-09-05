from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class _KernelSchema:
    """Private column indices used only by the NumPy tracker kernels."""

    geometry_cols: int
    detection_cols: int
    track_cols: int

    @property
    def is_obb(self) -> bool:
        return self.geometry_cols == 5

    @property
    def detection_conf_index(self) -> int:
        return self.geometry_cols

    @property
    def detection_class_index(self) -> int:
        return self.geometry_cols + 1

    @property
    def indexed_detection_cols(self) -> int:
        return self.detection_cols + 1

    @property
    def track_id_index(self) -> int:
        return self.geometry_cols

    @property
    def track_conf_index(self) -> int:
        return self.geometry_cols + 1

    @property
    def track_class_index(self) -> int:
        return self.geometry_cols + 2

    @property
    def track_detection_index(self) -> int:
        return self.geometry_cols + 3

    def empty_detections(self, dtype=np.float32) -> np.ndarray:
        return np.empty((0, self.detection_cols), dtype=dtype)

    def empty_tracks(self, dtype=np.float32) -> np.ndarray:
        return np.empty((0, self.track_cols), dtype=dtype)


_AABB_KERNEL_SCHEMA = _KernelSchema(geometry_cols=4, detection_cols=6, track_cols=8)
_OBB_KERNEL_SCHEMA = _KernelSchema(geometry_cols=5, detection_cols=7, track_cols=9)


@dataclass(frozen=True)
class DetectionLayout:
    """Shared indexing and shape rules for tracker detection tensors."""

    name: str
    schema: _KernelSchema

    @property
    def is_obb(self) -> bool:
        return self.schema.is_obb

    @property
    def det_cols(self) -> int:
        return self.schema.detection_cols

    @property
    def box_cols(self) -> int:
        return self.schema.geometry_cols

    @property
    def conf_idx(self) -> int:
        return self.schema.detection_conf_index

    @property
    def cls_idx(self) -> int:
        return self.schema.detection_class_index

    @property
    def output_cols(self) -> int:
        return self.schema.track_cols

    @property
    def box_with_conf_cols(self) -> int:
        return self.box_cols + 1

    def association_mode_name(self, base_name: str) -> str:
        if not self.is_obb:
            return base_name
        oriented_modes = {
            "iou": "iou_obb",
            "giou": "giou_obb",
            "centroid": "centroid_obb",
            "diou": "diou_obb",
            "ciou": "ciou_obb",
            "hmiou": "hmiou_obb",
        }
        try:
            return oriented_modes[base_name]
        except KeyError as exc:
            raise ValueError(
                f"Association mode '{base_name}' has no oriented-box implementation. "
                f"Choose from {sorted(oriented_modes)} for OBB tracking."
            ) from exc

    def empty_dets(self, dtype=np.float32) -> np.ndarray:
        return self.schema.empty_detections(dtype=dtype)

    def empty_output(self, dtype=float) -> np.ndarray:
        return self.schema.empty_tracks(dtype=dtype)

    def _validate_access_rows(self, dets: np.ndarray) -> None:
        if not isinstance(dets, np.ndarray):
            raise TypeError(f"Detections must be a numpy array, got {type(dets).__name__}.")
        if dets.ndim != 2:
            raise ValueError(f"Detections must be a 2D array, got shape {dets.shape}.")
        if dets.shape[1] not in (self.det_cols, self.schema.indexed_detection_cols):
            raise ValueError(
                f"Unsupported detection column count {dets.shape[1]}; expected "
                f"{self.det_cols} raw or {self.schema.indexed_detection_cols} indexed {self.name}."
            )

    def boxes(self, dets: np.ndarray) -> np.ndarray:
        self._validate_access_rows(dets)
        if dets.size == 0:
            return np.empty((0, self.box_cols), dtype=dets.dtype if hasattr(dets, "dtype") else np.float32)
        return dets[:, : self.box_cols]

    def confidences(self, dets: np.ndarray) -> np.ndarray:
        self._validate_access_rows(dets)
        if dets.size == 0:
            return np.empty((0,), dtype=dets.dtype if hasattr(dets, "dtype") else np.float32)
        return dets[:, self.conf_idx]

    def classes(self, dets: np.ndarray) -> np.ndarray:
        self._validate_access_rows(dets)
        if dets.size == 0:
            return np.empty((0,), dtype=dets.dtype if hasattr(dets, "dtype") else np.float32)
        return dets[:, self.cls_idx]

    def with_detection_indices(self, dets: np.ndarray) -> np.ndarray:
        self.validate_dets(dets)
        if dets.size == 0:
            return np.empty((0, self.det_cols + 1), dtype=dets.dtype if hasattr(dets, "dtype") else np.float32)
        det_inds = np.arange(len(dets), dtype=np.int32).reshape(-1, 1)
        return np.hstack([dets, det_inds])

    def validate_dets(self, dets: np.ndarray) -> None:
        if not isinstance(dets, np.ndarray):
            raise TypeError(f"Detections must be a numpy array, got {type(dets).__name__}.")
        if dets.ndim != 2:
            raise ValueError(f"Detections must be a 2D array, got shape {dets.shape}.")
        if dets.shape[1] != self.det_cols:
            raise ValueError(
                f"Unsupported detection column count {dets.shape[1]}; expected {self.det_cols} {self.name}."
            )


class AxisAlignedDetections(DetectionLayout):
    def __init__(self) -> None:
        super().__init__(
            name="(x1,y1,x2,y2,conf,cls)",
            schema=_AABB_KERNEL_SCHEMA,
        )


class OrientedDetections(DetectionLayout):
    def __init__(self) -> None:
        super().__init__(
            name="(cx,cy,w,h,angle,conf,cls)",
            schema=_OBB_KERNEL_SCHEMA,
        )


AABB_DETECTIONS = AxisAlignedDetections()
OBB_DETECTIONS = OrientedDetections()


def get_detection_layout(is_obb: bool) -> DetectionLayout:
    return OBB_DETECTIONS if is_obb else AABB_DETECTIONS


def infer_detection_layout(dets: np.ndarray) -> DetectionLayout | None:
    if dets is None or not isinstance(dets, np.ndarray) or dets.ndim != 2:
        return None
    if dets.shape[1] == 6:
        return AABB_DETECTIONS
    if dets.shape[1] == 7:
        return OBB_DETECTIONS
    return None
