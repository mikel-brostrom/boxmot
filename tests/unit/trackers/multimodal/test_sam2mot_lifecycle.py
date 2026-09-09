"""Public Sam2Mot lifecycle regression tests."""

from __future__ import annotations

import pytest
import torch

from boxmot.structures import Boxes, Detections, Frame, MaskBatch
from boxmot.trackers import TrackerSpec, create_tracker


@pytest.mark.parametrize("per_class", (False, True))
def test_moving_aabb_tracks_survive_and_recover_from_a_missing_detection(per_class: bool) -> None:
    """Tracks with learned velocity retain masks and IDs while coasting."""

    tracker = create_tracker(TrackerSpec("sam2mot", per_class=per_class))
    initial_boxes = torch.tensor([[8.0, 8.0, 20.0, 24.0], [32.0, 30.0, 44.0, 50.0]])
    velocity = torch.tensor([2.0, 1.0, 2.0, 1.0])
    observations: list[Detections] = []
    frames: list[Frame] = []
    for frame_index in range(4):
        sample_id = f"sequence/{frame_index:06d}"
        boxes = initial_boxes + frame_index * velocity
        masks = torch.zeros((2, 64, 64), dtype=torch.bool)
        for row, (x1, y1, x2, y2) in enumerate(boxes.to(torch.int64).tolist()):
            masks[row, y1:y2, x1:x2] = True
        observations.append(
            Detections(
                geometry=Boxes(boxes),
                scores=torch.tensor([0.95, 0.95]),
                class_ids=torch.tensor([1, 2]),
                sample_id=sample_id,
                masks=MaskBatch(masks),
            )
        )
        frames.append(
            Frame(
                image=torch.zeros((3, 64, 64), dtype=torch.uint8),
                sample_id=sample_id,
                sequence_id="sequence",
                frame_index=frame_index,
            )
        )

    first = tracker.update(observations[0], frames[0])
    matched = tracker.update(observations[1], frames[1])
    empty = observations[2].select(torch.empty(0, dtype=torch.int64))
    propagated = tracker.update(empty, frames[2])
    recovered = tracker.update(observations[3], frames[3])

    assert first.track_ids.tolist() == matched.track_ids.tolist() == propagated.track_ids.tolist()
    assert recovered.track_ids.tolist() == first.track_ids.tolist()
    assert propagated.class_ids.tolist() == recovered.class_ids.tolist() == [1, 2]
    assert propagated.detection_indices.tolist() == [-1, -1]
    assert recovered.detection_indices.tolist() == [0, 1]
    torch.testing.assert_close(propagated.geometry.values, observations[2].geometry.values)
    torch.testing.assert_close(recovered.geometry.values, observations[3].geometry.values)
    assert propagated.masks is not None
    assert recovered.masks is not None
    torch.testing.assert_close(propagated.masks.values, observations[1].masks.values)
    torch.testing.assert_close(recovered.masks.values, observations[3].masks.values)
