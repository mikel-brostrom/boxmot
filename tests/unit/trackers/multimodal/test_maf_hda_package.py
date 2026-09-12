"""Public construction and structured input contracts for MAF-HDA."""

from __future__ import annotations

import importlib

import pytest
import torch

from boxmot import MafHda
from boxmot.structures import Boxes, Detections, Frame, MaskBatch
from boxmot.trackers import TrackerRequirements, TrackerSpec, create_tracker
from boxmot.trackers.common.base import BaseTracker


def _observations(frame_index: int, *, masks: bool = True) -> tuple[Detections, Frame]:
    """Build two separated objects with deliberately reversed class ordering."""
    sample_id = f"sequence/{frame_index:06d}"
    boxes = torch.tensor([[8, 8, 24, 28], [40, 30, 58, 58]], dtype=torch.float32)
    values = torch.zeros((2, 64, 64), dtype=torch.bool)
    values[0, 8:28, 8:24] = True
    values[1, 30:58, 40:58] = True
    detections = Detections(
        geometry=Boxes(boxes),
        scores=torch.tensor([0.95, 0.95]),
        class_ids=torch.tensor([9, 2]),
        sample_id=sample_id,
        masks=MaskBatch(values) if masks else None,
    )
    frame = Frame(
        image=torch.zeros((3, 64, 64), dtype=torch.uint8),
        sample_id=sample_id,
        sequence_id="sequence",
        frame_index=frame_index,
    )
    return detections, frame


def test_maf_hda_uses_the_shared_public_update_boundary() -> None:
    """Lazy public construction resolves to the domain implementation."""
    implementation = importlib.import_module("boxmot.trackers.maf_hda.tracker")
    tracker = create_tracker(TrackerSpec("maf_hda"))

    assert type(tracker) is MafHda is implementation.MafHda
    assert MafHda.update is BaseTracker.update
    assert tracker.requirements == TrackerRequirements(masks=True, frame=True)


def test_maf_hda_rejects_unsupported_geometry_and_backend() -> None:
    """Registry validation rejects unsupported modes before tracking begins."""
    with pytest.raises(ValueError, match="does not support geometry kind 'obb'"):
        create_tracker(TrackerSpec("maf_hda", geometry="obb"))
    with pytest.raises(ValueError, match="does not support OBB geometry"):
        MafHda(is_obb=True)
    with pytest.raises(ValueError, match="Native backend is unavailable"):
        create_tracker(TrackerSpec("maf_hda", backend="cpp"))


def test_maf_hda_requires_foreground_masks_and_a_frame() -> None:
    """Missing or empty instance evidence fails at the shared input boundary."""
    tracker = create_tracker(TrackerSpec("maf_hda"))
    unmasked, frame = _observations(0, masks=False)
    with pytest.raises(ValueError, match="requires full-frame detection masks"):
        tracker.update(unmasked, frame)
    detections, frame = _observations(0)
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(detections)
    detections.masks.values[0].zero_()
    with pytest.raises(ValueError, match="foreground in every non-empty detection mask"):
        tracker.update(detections, frame)
    assert tracker.frame_count == 0


@pytest.mark.parametrize("per_class", (False, True))
def test_maf_hda_preserves_global_detection_and_mask_alignment(per_class: bool) -> None:
    """Class partitioning keeps IDs, masks, and frame-level detection rows aligned."""
    tracker = create_tracker(TrackerSpec("maf_hda", per_class=per_class))
    first_detections, first_frame = _observations(0)
    first = tracker.update(first_detections, first_frame)
    detections, frame = _observations(1)
    second = tracker.update(detections, frame)

    assert len(first) == len(second) == 2
    assert first.track_ids.tolist() == second.track_ids.tolist()
    assert len(set(second.track_ids.tolist())) == 2
    assert sorted(second.detection_indices.tolist()) == [0, 1]
    assert second.masks is not None
    for output_index, detection_index in enumerate(second.detection_indices.tolist()):
        assert second.class_ids[output_index] == detections.class_ids[detection_index]
        torch.testing.assert_close(second.masks.values[output_index], detections.masks.values[detection_index])

    tracker.reset()
    restarted = tracker.update(first_detections, first_frame)
    assert tracker.frame_count == 1
    assert restarted.track_ids.tolist() == first.track_ids.tolist()
