"""Tests for the representation-level box tracker boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from boxmot.structures import Boxes, GeometryKind, OrientedBoxes
from boxmot.trackers.common.box.base import BoxTracker
from boxmot.trackers.common.box.geometry import (
    AABB_GEOMETRY_OPS,
    OBB_GEOMETRY_OPS,
    BoxGeometryOps,
    get_box_geometry_ops,
)
from boxmot.trackers.common.detections.layout import AABB_DETECTIONS, OBB_DETECTIONS


def test_geometry_ops_bind_canonical_kind_layout_and_association() -> None:
    aabb = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    obb = np.array([[5.0, 5.0, 10.0, 10.0, 0.25]], dtype=np.float32)

    assert get_box_geometry_ops(is_obb=False) is AABB_GEOMETRY_OPS
    assert get_box_geometry_ops(is_obb=True) is OBB_GEOMETRY_OPS
    assert AABB_GEOMETRY_OPS.layout is AABB_DETECTIONS
    assert OBB_GEOMETRY_OPS.layout is OBB_DETECTIONS
    assert AABB_GEOMETRY_OPS.association_function("iou")(aabb, aabb)[0, 0] == pytest.approx(1.0)
    assert OBB_GEOMETRY_OPS.association_function("iou")(obb, obb)[0, 0] == pytest.approx(1.0)


def test_geometry_ops_are_immutable_and_reject_mismatched_contracts() -> None:
    with pytest.raises(FrozenInstanceError):
        AABB_GEOMETRY_OPS.kind = GeometryKind.OBB  # type: ignore[misc]
    with pytest.raises(ValueError, match="does not match"):
        BoxGeometryOps(kind=GeometryKind.AABB, layout=OBB_DETECTIONS)
    with pytest.raises(TypeError, match="is_obb must be bool"):
        get_box_geometry_ops(is_obb=1)  # type: ignore[arg-type]


def test_geometry_ops_validate_canonical_structure_kind() -> None:
    boxes = Boxes(torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32))
    oriented = OrientedBoxes(torch.tensor([[5.0, 5.0, 10.0, 10.0, 0.0]], dtype=torch.float32))

    AABB_GEOMETRY_OPS.validate(boxes)
    OBB_GEOMETRY_OPS.validate(oriented)
    with pytest.raises(ValueError, match="requires Boxes"):
        AABB_GEOMETRY_OPS.validate(oriented)
    with pytest.raises(ValueError, match="requires OrientedBoxes"):
        OBB_GEOMETRY_OPS.validate(boxes)


def test_supported_geometry_kinds_drive_legacy_guard_and_capabilities() -> None:
    class AabbOnlyTracker(BoxTracker):
        supported_geometry_kinds = frozenset({GeometryKind.AABB})

        def _track_detections(self, dets, img, embs=None, masks=None):
            return self.detection_layout.empty_output()

    assert AabbOnlyTracker.supports_obb is False
    assert AabbOnlyTracker.capabilities.geometry_kinds == frozenset({GeometryKind.AABB})
    AabbOnlyTracker()
    with pytest.raises(ValueError, match="does not support OBB geometry"):
        AabbOnlyTracker(is_obb=True)
