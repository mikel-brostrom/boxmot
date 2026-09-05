"""Geometry policy for trackers whose primary state is an AABB or OBB."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from boxmot.structures import Boxes, Geometry, GeometryKind, OrientedBoxes
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.detections.layout import AABB_DETECTIONS, OBB_DETECTIONS, DetectionLayout

AssociationCallable: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class BoxGeometryOps:
    """Bind one public box kind to its private kernel layout and association modes."""

    kind: GeometryKind
    layout: DetectionLayout

    def __post_init__(self) -> None:
        if self.layout.is_obb != (self.kind is GeometryKind.OBB):
            raise ValueError(f"Geometry kind {self.kind!r} does not match its detection layout.")

    @property
    def is_obb(self) -> bool:
        """Whether this policy operates on oriented boxes."""

        return self.kind is GeometryKind.OBB

    def validate(self, geometry: Geometry) -> None:
        """Reject geometry whose canonical structure does not match this policy."""

        expected_type = OrientedBoxes if self.is_obb else Boxes
        if not isinstance(geometry, expected_type):
            expected = "OrientedBoxes" if self.is_obb else "Boxes"
            raise ValueError(f"{self.kind.value.upper()} box tracking requires {expected} geometry.")

    def association_mode_name(self, base_name: str) -> str:
        """Resolve a public association name for this geometry kind."""

        return self.layout.association_mode_name(base_name)

    def association_function(
        self,
        base_name: str,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> AssociationCallable:
        """Build the existing association kernel for this geometry kind."""

        mode = self.association_mode_name(base_name)
        return AssociationFunction(w=width, h=height, asso_mode=mode).asso_func


AABB_GEOMETRY_OPS = BoxGeometryOps(kind=GeometryKind.AABB, layout=AABB_DETECTIONS)
OBB_GEOMETRY_OPS = BoxGeometryOps(kind=GeometryKind.OBB, layout=OBB_DETECTIONS)


def get_box_geometry_ops(*, is_obb: bool) -> BoxGeometryOps:
    """Return the immutable geometry policy selected by tracker configuration."""

    if not isinstance(is_obb, bool):
        raise TypeError(f"is_obb must be bool, got {type(is_obb).__name__}.")
    return OBB_GEOMETRY_OPS if is_obb else AABB_GEOMETRY_OPS


__all__ = (
    "AABB_GEOMETRY_OPS",
    "OBB_GEOMETRY_OPS",
    "BoxGeometryOps",
    "get_box_geometry_ops",
)
