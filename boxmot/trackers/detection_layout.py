from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionLayout:
    """Shared indexing and shape rules for tracker detection tensors."""

    name: str
    is_obb: bool
    det_cols: int
    box_cols: int
    conf_idx: int
    cls_idx: int
    output_cols: int

    @property
    def box_with_conf_cols(self) -> int:
        return self.box_cols + 1

    def association_mode_name(self, base_name: str) -> str:
        return f"{base_name}_obb" if self.is_obb else base_name

    def empty_dets(self, dtype=np.float32) -> np.ndarray:
        return np.empty((0, self.det_cols), dtype=dtype)

    def empty_output(self, dtype=float) -> np.ndarray:
        return np.empty((0, self.output_cols), dtype=dtype)

    def boxes(self, dets: np.ndarray) -> np.ndarray:
        if dets.size == 0:
            return np.empty((0, self.box_cols), dtype=dets.dtype if hasattr(dets, "dtype") else np.float32)
        return dets[:, : self.box_cols]

    def confidences(self, dets: np.ndarray) -> np.ndarray:
        if dets.size == 0:
            return np.empty((0,), dtype=dets.dtype if hasattr(dets, "dtype") else np.float32)
        return dets[:, self.conf_idx]

    def classes(self, dets: np.ndarray) -> np.ndarray:
        if dets.size == 0:
            return np.empty((0,), dtype=dets.dtype if hasattr(dets, "dtype") else np.float32)
        return dets[:, self.cls_idx]

    def with_detection_indices(self, dets: np.ndarray) -> np.ndarray:
        if dets.size == 0:
            return np.empty((0, self.det_cols + 1), dtype=dets.dtype if hasattr(dets, "dtype") else np.float32)
        det_inds = np.arange(len(dets), dtype=np.int32).reshape(-1, 1)
        return np.hstack([dets, det_inds])

    def validate_dets(self, dets: np.ndarray) -> None:
        assert dets.shape[1] == self.det_cols, (
            "Unsupported 'dets' 2nd dimension length, valid length is "
            f"{self.det_cols} {self.name}"
        )


class AxisAlignedDetections(DetectionLayout):
    def __init__(self) -> None:
        super().__init__(
            name="(x1,y1,x2,y2,conf,cls)",
            is_obb=False,
            det_cols=6,
            box_cols=4,
            conf_idx=4,
            cls_idx=5,
            output_cols=8,
        )


class OrientedDetections(DetectionLayout):
    def __init__(self) -> None:
        super().__init__(
            name="(cx,cy,w,h,angle,conf,cls)",
            is_obb=True,
            det_cols=7,
            box_cols=5,
            conf_idx=5,
            cls_idx=6,
            output_cols=9,
        )


AABB_DETECTIONS = AxisAlignedDetections()
OBB_DETECTIONS = OrientedDetections()


def get_detection_layout(is_obb: bool) -> DetectionLayout:
    return OBB_DETECTIONS if is_obb else AABB_DETECTIONS


def infer_detection_layout(dets: np.ndarray) -> DetectionLayout | None:
    if dets is None or not isinstance(dets, np.ndarray) or dets.ndim != 2:
        return None
    if dets.shape[1] == AABB_DETECTIONS.det_cols:
        return AABB_DETECTIONS
    if dets.shape[1] == OBB_DETECTIONS.det_cols:
        return OBB_DETECTIONS
    return None