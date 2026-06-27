from __future__ import annotations

import numpy as np

from boxmot.trackers.common.tracking import outputs as output_utils
from boxmot.trackers.common.tracking.records import TrackRecord


class TrackFormattingMixin:
    """Output formatting helpers shared by tracker implementations."""

    def _track_box_for_output(self, track) -> np.ndarray:
        for attr_name in (
            "output_box",
            "box",
            "bbox",
            "xywha" if self.is_obb else "xyxy",
        ):
            if not attr_name:
                continue
            box = self._resolve_track_box_attr(track, attr_name)
            if box is not None:
                return np.asarray(box, dtype=np.float32).reshape(-1)
        if hasattr(track, "get_state"):
            return np.asarray(track.get_state()[0], dtype=np.float32).reshape(-1)
        raise AttributeError(f"{track.__class__.__name__} does not expose an output box")

    @staticmethod
    def _track_id(track) -> int:
        if hasattr(track, "id"):
            return int(getattr(track, "id"))
        return int(getattr(track, "track_id"))

    def track_record(self, track, state: str = "active") -> TrackRecord:
        """Return a canonical snapshot for a tracker-local track object."""
        return TrackRecord(
            box=self._track_box_for_output(track),
            track_id=self._track_id(track),
            conf=float(getattr(track, "conf", 1.0)),
            cls=int(getattr(track, "cls", -1)),
            det_ind=int(getattr(track, "det_ind", -1)),
            state=state,
            age=int(getattr(track, "age", 0)),
            time_since_update=int(getattr(track, "time_since_update", 0)),
        )

    def format_output_row(
        self,
        box: np.ndarray,
        track_id: int,
        conf: float,
        cls: int,
        det_ind: int,
        dtype=np.float32,
    ) -> np.ndarray:
        """Format one track row using the public tracker output contract."""
        return output_utils.format_output_row(
            self.detection_layout,
            box,
            track_id,
            conf,
            cls,
            det_ind,
            dtype=dtype,
        )

    def format_outputs(self, tracks, dtype=np.float32) -> np.ndarray:
        """Format track-like objects into the public output array."""
        rows = [
            self.format_output_row(
                self._track_box_for_output(track),
                self._track_id(track),
                float(getattr(track, "conf", 1.0)),
                int(getattr(track, "cls", -1)),
                int(getattr(track, "det_ind", -1)),
                dtype=dtype,
            )
            for track in tracks
        ]
        return output_utils.format_output_rows(self.detection_layout, rows, dtype=dtype)

    def format_output_rows(self, rows, dtype=np.float32) -> np.ndarray:
        """Return rows with the tracker-specific empty shape when no rows exist."""
        return output_utils.format_output_rows(self.detection_layout, rows, dtype=dtype)

    def filter_outputs_by_geometry(
        self,
        outputs: np.ndarray,
        min_box_area: float | None = None,
        max_aspect_ratio: float | None = None,
    ) -> np.ndarray:
        """Filter output rows by area and width/height ratio in AABB or OBB mode."""
        if outputs.size == 0:
            dtype = outputs.dtype if hasattr(outputs, "dtype") else np.float32
            return self.empty_output(dtype=dtype)

        outputs = np.asarray(outputs)
        if self.is_obb:
            widths = outputs[:, 2]
            heights = outputs[:, 3]
        else:
            widths = outputs[:, 2] - outputs[:, 0]
            heights = outputs[:, 3] - outputs[:, 1]

        keep = np.ones(len(outputs), dtype=bool)
        if max_aspect_ratio is not None:
            keep &= widths / np.maximum(heights, 1e-6) <= max_aspect_ratio
        if min_box_area is not None:
            keep &= widths * heights > min_box_area
        return outputs[keep]
