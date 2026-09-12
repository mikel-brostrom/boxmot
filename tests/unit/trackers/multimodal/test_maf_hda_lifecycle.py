"""Behavioral coverage for the Python MAF-HDA port and its two association stages."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import MafHda
from boxmot.structures import Boxes, Detections, MaskBatch
from boxmot.trackers.maf_hda.association import (
    INITIAL_COVARIANCE,
    MAX_COST,
    fusion_cost,
    gaussian_affinity,
    minmax_affinity,
    predict_covariance,
)


def _sample(
    frame: int,
    boxes: list[list[float]],
    *,
    scores: list[float] | None = None,
    classes: list[int] | None = None,
) -> tuple[Detections, np.ndarray]:
    """Draw a deterministic object texture and full-frame masks on a small frame."""
    values = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    masks = torch.zeros((len(boxes), 96, 128), dtype=torch.bool)
    image = np.full((96, 128, 3), 30, dtype=np.uint8)
    for index, box in enumerate(values.to(torch.int64).tolist()):
        x1, y1, x2, y2 = box
        masks[index, y1:y2, x1:x2] = True
        yy, xx = np.indices((y2 - y1, x2 - x1))
        image[y1:y2, x1:x2, 0] = 80 + (xx % 4) * 40
        image[y1:y2, x1:x2, 1] = 60 + (yy % 4) * 40
        image[y1:y2, x1:x2, 2] = 180
    detections = Detections(
        geometry=Boxes(values),
        scores=torch.tensor(scores or [0.95] * len(boxes), dtype=torch.float32),
        class_ids=torch.tensor(classes or [0] * len(boxes), dtype=torch.int64),
        sample_id=f"sequence/{frame}",
        masks=MaskBatch(masks),
    )
    return detections, image


@pytest.mark.parametrize("per_class", (False, True))
@pytest.mark.parametrize("missing_frames", (1, 2, 3))
def test_recovery_preserves_ids_only_within_max_age(per_class: bool, missing_frames: int) -> None:
    """T2TA uses trajectory motion across a gap without emitting stale masks."""
    tracker = MafHda(max_age=3, per_class=per_class)
    first = tracker.update(*_sample(0, [[10, 20, 30, 48]]))
    second = tracker.update(*_sample(1, [[12, 20, 32, 48]]))
    assert second.track_ids.tolist() == first.track_ids.tolist()
    for frame in range(2, 2 + missing_frames):
        missing = tracker.update(*_sample(frame, []))
        assert len(missing) == 0
        assert missing.masks.values.shape == (0, 96, 128)
    frame = 2 + missing_frames
    detection, image = _sample(frame, [[10 + 2 * frame, 20, 30 + 2 * frame, 48]])
    recovered = tracker.update(detection, image)
    assert len(recovered) == 1
    assert (recovered.track_ids.tolist() == first.track_ids.tolist()) == (missing_frames < 3)
    assert recovered.detection_indices.tolist() == [0]
    torch.testing.assert_close(recovered.masks.values, detection.masks.values)


def test_tracklet_stage_recovers_a_match_rejected_by_segment_stage() -> None:
    """The second stage permits smaller overlap than the first motion gate."""
    tracker = MafHda(s2ta_mode="motion", t2ta_mode="motion")
    first = tracker.update(*_sample(0, [[10, 20, 30, 40]]))
    # Nine overlapping columns are below S2TA's half-area requirement.
    recovered = tracker.update(*_sample(1, [[21, 20, 41, 40]]))
    assert recovered.track_ids.tolist() == first.track_ids.tolist()
    assert tracker._tracks[0].hits == 2
    assert tracker.id_allocator.next_id == 2  # A birth was linked back to ID 0.


def test_mask_merging_is_transitive_and_preserves_representative_input_index() -> None:
    """Duplicate mask components retain the most confident original detection."""
    tracker = MafHda()
    detections, image = _sample(
        0,
        [[8, 8, 28, 24], [14, 8, 34, 24], [20, 8, 40, 24]],
        scores=[0.8, 0.99, 0.9],
    )
    original_masks = detections.masks.values.clone()
    result = tracker.update(detections, image)
    assert len(result) == 1
    assert result.detection_indices.tolist() == [1]
    assert result.scores.item() == pytest.approx(0.99)
    assert result.geometry.values.tolist() == [[8, 8, 40, 24]]
    torch.testing.assert_close(result.masks.values[0], original_masks.any(dim=0))
    torch.testing.assert_close(detections.masks.values, original_masks)


@pytest.mark.parametrize("per_class", (False, True))
def test_masks_and_ids_do_not_merge_across_classes(per_class: bool) -> None:
    """Overlapping semantic classes remain independent even without partitioning."""
    tracker = MafHda(per_class=per_class)
    first = tracker.update(*_sample(0, [[8, 8, 28, 24], [8, 8, 28, 24]], classes=[3, 4]))
    assert len(first) == 2
    assert len(set(first.track_ids.tolist())) == 2
    second = tracker.update(*_sample(1, [[8, 8, 28, 24]], classes=[4]))
    assert len(second) == 1
    assert second.track_ids.item() == first.track_ids[first.class_ids == 4].item()


def test_min_hits_requires_observations_including_at_sequence_start() -> None:
    """A single detection cannot become confirmed by simply waiting."""
    tracker = MafHda(min_hits=3)
    assert len(tracker.update(*_sample(0, [[10, 20, 30, 48]]))) == 0
    assert len(tracker.update(*_sample(1, []))) == 0
    assert len(tracker.update(*_sample(2, []))) == 0
    assert len(tracker.update(*_sample(3, [[10, 20, 30, 48]]))) == 0
    assert len(tracker.update(*_sample(4, [[10, 20, 30, 48]]))) == 0
    assert len(tracker.update(*_sample(5, [[10, 20, 30, 48]]))) == 1


def test_detection_filter_retains_original_row_indices() -> None:
    """Low-confidence detections do not birth or refresh tracks."""
    tracker = MafHda()
    detection, image = _sample(0, [[8, 8, 28, 24], [40, 8, 60, 24]], scores=[0.1, 0.9])
    result = tracker.update(detection, image)
    assert result.detection_indices.tolist() == [1]
    torch.testing.assert_close(result.masks.values[0], detection.masks.values[1])
    assert len(tracker.update(*_sample(1, [[40, 8, 60, 24]], scores=[0.1]))) == 0


def test_gaussian_motion_matches_the_source_covariance_and_likelihood() -> None:
    """Check a source-prior numerical example independent of an association run."""
    predicted = predict_covariance(INITIAL_COVARIANCE)
    np.testing.assert_allclose(np.diag(predicted), [62.5, 250.0, 37.5, 150.0])
    box = np.array([10.0, 10.0, 30.0, 30.0])
    likelihood, posterior = gaussian_affinity(np.array([box, box + [100, 0, 100, 0]]), box, predicted)
    assert likelihood[0] == pytest.approx(1.0 / (350.0 * np.pi))
    assert likelihood[1] == 0.0
    np.testing.assert_allclose(np.diag(posterior[0]), [125.0 / 7.0, 500.0 / 7.0, 37.5, 150.0])
    np.testing.assert_array_equal(posterior[1], predicted)


def test_appearance_rescue_preserves_uncertainty_when_motion_gate_rejects() -> None:
    """MAF's size-change fallback does not falsely count a Gaussian measurement."""
    tracker = MafHda()
    first = tracker.update(*_sample(0, [[10, 20, 30, 40]]))
    rescued = tracker.update(*_sample(1, [[10, 20, 50, 60]]))
    assert rescued.track_ids.tolist() == first.track_ids.tolist()
    np.testing.assert_array_equal(tracker._tracks[0].covariance, predict_covariance(INITIAL_COVARIANCE))


def test_reappearing_track_can_finish_confirmation_after_recovery_deadline() -> None:
    """max_age measures the gap to reappearance, excluding confirmation latency."""
    tracker = MafHda(min_hits=3, max_age=3, s2ta_mode="motion", t2ta_mode="motion")
    for frame in range(3):
        initial = tracker.update(*_sample(frame, [[10, 20, 30, 48]]))
    tracker.update(*_sample(3, []))
    for frame in range(4, 7):
        recovered = tracker.update(*_sample(frame, [[10, 20, 30, 48]]))
    assert recovered.track_ids.tolist() == initial.track_ids.tolist()


def test_recovery_geometry_uses_birth_frame_while_candidate_keeps_moving() -> None:
    """Confirmation must not compare an old prediction with a later observation."""
    tracker = MafHda(min_hits=4, max_age=30)
    for frame in range(9):
        boxes = [] if frame == 4 else [[10 + 9 * frame, 20, 30 + 9 * frame, 48]]
        result = tracker.update(*_sample(frame, boxes))
        if frame == 3:
            initial_ids = result.track_ids.tolist()
    assert initial_ids
    assert result.track_ids.tolist() == initial_ids


def test_fusion_uses_source_minmax_product_and_overlap_substitution() -> None:
    """Check source MAF costs, including the zero-affinity rejection sentinel."""
    cost = fusion_cost(
        np.array([[0.2, 0.3], [0.6, 0.4]]),
        np.array([[0.7, 0.9], [0.2, 0.4]]),
        np.array([[0.5, 0.2], [0.0, 0.0]]),
        mode="maf",
        recovery=False,
        appearance_lower=0.1,
        appearance_upper=0.7,
        overlap_lower=0.1,
    )
    assert cost[0, 0] == pytest.approx(-100 * np.log(5.0 / 14.0))
    assert cost[0, 1] == pytest.approx(-100 * np.log(0.2))
    assert cost[1, 0] == MAX_COST
    assert cost[1, 1] == pytest.approx(-100 * np.log(1.0 / 7.0))
    np.testing.assert_array_equal(minmax_affinity(np.zeros((1, 1))), [[0]])
    np.testing.assert_array_equal(minmax_affinity(np.full((1, 1), 0.75)), [[1]])


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_age", 0),
        ("min_hits", 0),
        ("merge_iou_thresh", 0),
        ("velocity_alpha", float("nan")),
        ("s2ta_mode", "unknown"),
        ("t2ta_mode", "unknown"),
        ("template_size", 8),
    ],
)
def test_invalid_tracker_parameters_fail_before_tracking(option: str, value: object) -> None:
    """Reject configurations that cannot define a meaningful tracker state."""
    with pytest.raises(ValueError, match=option):
        MafHda(**{option: value})
