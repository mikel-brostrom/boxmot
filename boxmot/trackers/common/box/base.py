"""Shared invariant boundary for box-state trackers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from boxmot.structures import Geometry, GeometryKind
from boxmot.trackers.common.base import BaseTracker
from boxmot.trackers.common.box.geometry import get_box_geometry_ops
from boxmot.trackers.common.detections.layout import DetectionLayout
from boxmot.trackers.common.specs import TrackerCapabilities, TrackerFamily


class BoxTracker(BaseTracker):
    """Base class for trackers whose primary state is an AABB or OBB.

    The class selects one immutable geometry policy at construction. That
    policy owns canonical-geometry validation, private row layout selection,
    and association-mode dispatch for the lifetime of the tracker.
    """

    supported_geometry_kinds: frozenset[GeometryKind] = frozenset(
        {GeometryKind.AABB, GeometryKind.OBB}
    )
    supports_obb = GeometryKind.OBB in supported_geometry_kinds
    accepts_embeddings = False
    accepts_masks = False
    accepts_frame = True
    requires_embeddings = False
    requires_masks = False
    requires_frame = False
    capabilities = TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=supported_geometry_kinds,
        accepts_frame=True,
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """Materialize one immutable capability declaration per implementation."""

        super().__init_subclass__(**kwargs)
        cls.supports_obb = GeometryKind.OBB in cls.supported_geometry_kinds
        cls.capabilities = TrackerCapabilities(
            family=TrackerFamily.BOX,
            geometry_kinds=cls.supported_geometry_kinds,
            requires_embeddings=bool(cls.requires_embeddings),
            accepts_embeddings=bool(cls.accepts_embeddings),
            requires_masks=bool(cls.requires_masks),
            accepts_masks=bool(cls.accepts_masks),
            requires_frame=bool(cls.requires_frame),
            accepts_frame=bool(cls.accepts_frame),
        )

    def __init__(self, *args, is_obb: bool = False, **kwargs) -> None:
        self.geometry_ops = get_box_geometry_ops(is_obb=is_obb)
        self.validate_geometry_kind(self.geometry_ops.kind)
        super().__init__(*args, is_obb=is_obb, **kwargs)

    def validate_geometry_kind(self, kind: GeometryKind) -> None:
        """Validate a fixed geometry kind before allocating tracker state."""

        if kind not in self.supported_geometry_kinds:
            supported = ", ".join(sorted(value.value for value in self.supported_geometry_kinds))
            raise ValueError(
                f"{type(self).__name__} does not support {kind.value.upper()} geometry. "
                f"Supported geometry kinds: {supported}."
            )

    def _resolve_detection_layout(self, is_obb: bool) -> DetectionLayout:
        if is_obb is not self.geometry_ops.is_obb:
            raise ValueError("Box tracker geometry cannot change after construction.")
        return self.geometry_ops.layout

    def _resolve_association_mode_name(self, base_name: str) -> str:
        return self.geometry_ops.association_mode_name(base_name)

    def _build_association_function(
        self,
        *,
        width: int | None,
        height: int | None,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self.geometry_ops.association_function(
            self._asso_func_base_name,
            width=width,
            height=height,
        )

    def _validate_geometry(self, geometry: Geometry) -> None:
        self.geometry_ops.validate(geometry)


__all__ = ("BoxTracker",)
