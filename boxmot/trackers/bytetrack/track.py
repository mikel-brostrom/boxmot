# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from dataclasses import dataclass

import numpy as np

from boxmot.trackers.common.geometry import tlwh2xyah, xywh2tlwh, xywh2xyxy, xyxy2xywh
from boxmot.trackers.common.motion.models import (
    MotionModelAdapter,
    MotionModelKind,
    create_motion_model,
    resolve_motion_model_kind,
)
from boxmot.trackers.common.track_state import BoxTrack
from boxmot.trackers.common.tracking.track import TrackIdAllocator


@dataclass(slots=True, eq=False)
class _Detection:
    """One frame's observation, without allocating persistent track state."""

    raw_row: np.ndarray
    xywh: np.ndarray
    xyah: np.ndarray | None
    conf: np.float32
    cls: np.float32
    det_ind: np.float32
    is_obb: bool

    @classmethod
    def from_rows(cls, rows: np.ndarray, *, is_obb: bool) -> list["_Detection"]:
        """Prepare owned observations in batches with scalar-track rounding."""
        if not len(rows):
            return []
        values = np.array(rows, dtype=np.float32, copy=True)
        if is_obb:
            boxes = values[:, :5]
            measurements = [None] * len(values)
        else:
            boxes = xyxy2xywh(values[:, :4])
            # Preserve the float32 center round trip used by STrack. Computing
            # XYAH directly from XYWH can change a subsequent match boundary.
            measurements = tlwh2xyah(xywh2tlwh(boxes))
        return [
            cls(row, box, measurement, row[-3], row[-2], row[-1], is_obb)
            for row, box, measurement in zip(values, boxes, measurements, strict=True)
        ]

    @property
    def xyxy(self) -> np.ndarray:
        """Expose current geometry to generic association consumers."""
        return BoxTrack.obb_to_xyxy(self.xywh) if self.is_obb else xywh2xyxy(self.xywh)

    @property
    def xywha(self) -> np.ndarray:
        """Expose an oriented observation without sharing its mutable storage."""
        if self.is_obb:
            return self.xywh.copy()
        return np.array([*self.xywh, 0.0], dtype=np.float32)


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(BoxTrack):
    state = TrackState.New
    lost_state = TrackState.Lost
    removed_state = TrackState.Removed
    tracked_state = TrackState.Tracked

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            super().__init__(*args, **kwargs)


class STrack(BaseTrack):
    """One ByteTrack identity with motion state and explicit activation/expiry."""

    def __init__(
        self,
        det,
        max_obs,
        id_allocator: TrackIdAllocator,
        is_obb: bool = False,
        *,
        motion_model: MotionModelAdapter | None = None,
    ) -> None:
        """Create a detection track with its selected motion representation."""
        if motion_model is None:
            motion_model = create_motion_model(resolve_motion_model_kind("bytetrack", is_obb=is_obb), is_obb=is_obb)
        super().__init__(det, id_allocator=id_allocator, max_obs=max_obs, is_obb=is_obb, motion_model=motion_model)

    def _init_from_aabb_detection(self, det: np.ndarray) -> None:
        self.xywh = xyxy2xywh(det[0:4])
        self.tlwh = xywh2tlwh(self.xywh)
        self.xyah = tlwh2xyah(self.tlwh)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

    def _measurement_for_update(self, track: "STrack | _Detection") -> np.ndarray:
        if self.is_obb and self.motion_model.kind is MotionModelKind.XYWH:
            return track.xywh
        if not self.is_obb and self.motion_model.kind is MotionModelKind.XYAH:
            return track.xyah
        box = track.xywh if self.is_obb else xywh2xyxy(track.xywh)
        return self.motion_model.to_measurement(np.r_[box, track.conf], column=False)

    def _reset_inactive_aabb_motion(self, mean_state: np.ndarray) -> None:
        mean_state[:] = self.motion_model.stop_size_velocity(mean_state, height_only=True)

    @staticmethod
    def multi_camera_update(tracks: list["STrack"], warp: np.ndarray) -> None:
        """Warp live means and covariances in their selected AABB/OBB models."""
        groups: dict[MotionModelAdapter, list[STrack]] = {}
        for track in tracks:
            if track.mean is not None and track.covariance is not None:
                groups.setdefault(track.motion_model, []).append(track)
        for model, group in groups.items():
            means, covariances = model.transform_states(
                np.asarray([track.mean for track in group]),
                np.asarray([track.covariance for track in group]),
                warp,
            )
            for track, mean, covariance in zip(group, means, covariances):
                track.mean, track.covariance = mean, covariance

    def _current_aabb_xywh(self) -> np.ndarray:
        if self.mean is None:
            return self.xywh.copy()
        if self.motion_model.kind is not MotionModelKind.XYAH:
            return xyxy2xywh(self.motion_model.to_box(self.mean)[0])
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        return ret
