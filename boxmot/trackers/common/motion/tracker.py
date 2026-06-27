from __future__ import annotations

import numpy as np

from boxmot.trackers.common.motion import cmc as cmc_utils


class TrackerMotionMixin:
    """Tracker-level motion and camera-motion helper methods."""

    def cmc_detection_boxes(self, dets: np.ndarray) -> np.ndarray:
        """Return AABB boxes used for camera-motion estimation."""
        return cmc_utils.cmc_detection_boxes(dets, self.detection_layout)

    def aabb_detections_for_association(self, dets: np.ndarray) -> np.ndarray:
        """Return AABB-layout detections for AABB-only association functions."""
        if not self.is_obb:
            return dets

        has_indices = dets.ndim == 2 and dets.shape[1] == self.detection_layout.det_cols + 1
        out_cols = 7 if has_indices else 6
        if dets.size == 0:
            dtype = dets.dtype if hasattr(dets, "dtype") else np.float32
            return np.empty((0, out_cols), dtype=dtype)

        boxes = self.cmc_detection_boxes(dets)
        confs = self.detection_layout.confidences(dets).reshape(-1, 1)
        clss = self.detection_layout.classes(dets).reshape(-1, 1)
        columns = [boxes, confs, clss]
        if has_indices:
            columns.append(dets[:, self.detection_layout.det_cols].reshape(-1, 1))
        return np.hstack(columns).astype(dets.dtype, copy=False)

    def apply_cmc(
        self,
        img: np.ndarray,
        dets: np.ndarray,
        tracks,
        update_method: str = "camera_update",
    ) -> np.ndarray | None:
        """Apply CMC to tracks using OBB-safe detection boxes for estimation."""
        return cmc_utils.apply_cmc_to_tracks(
            getattr(self, "cmc", None),
            img,
            dets,
            self.detection_layout,
            tracks,
            update_method=update_method,
        )

    def _reset_cmc_state(self) -> None:
        """Reset CMC adapters that keep frame-to-frame state."""
        cmc_utils.reset_cmc(getattr(self, "cmc", None))
