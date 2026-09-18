"""Exercise OccluBoost's temporal-mask lifecycle through public updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from boxmot import OccluBoostConfig
from boxmot.structures import Boxes, Detections, Frame
from boxmot.trackers import MaskGuidance, MaskGuidanceConfig
from boxmot.trackers.occluboost.tracker import OccluBoost

FRAME_SHAPE = (96, 128)


@dataclass
class _Propagator:
    """Record causal prompts and supply deterministic device mask tensors."""

    device: str = "cpu"
    max_objects: int = 96
    prompt_overlap: float = 0.10
    masks: dict[int, torch.Tensor] = field(default_factory=dict)
    calls: list[tuple[int, dict[int, np.ndarray], dict[int, np.ndarray]]] = field(default_factory=list)
    retained: list[set[int]] = field(default_factory=list)
    resets: int = 0

    def propagate(
        self,
        frame_index: int,
        frame: np.ndarray,
        active_boxes: dict[int, np.ndarray],
        new_boxes: dict[int, np.ndarray],
    ) -> dict[int, torch.Tensor]:
        assert frame.shape == (*FRAME_SHAPE, 3)
        self.calls.append(
            (
                frame_index,
                {identity: box.copy() for identity, box in active_boxes.items()},
                {identity: box.copy() for identity, box in new_boxes.items()},
            )
        )
        return self.masks.copy()

    def retain_tracks(self, track_ids: set[int]) -> None:
        self.retained.append(set(track_ids))
        self.masks = {identity: mask for identity, mask in self.masks.items() if identity in track_ids}

    def reset(self) -> None:
        self.resets += 1
        self.masks.clear()


def _tracker(*, guided: bool = True, **kwargs: object) -> tuple[OccluBoost, _Propagator]:
    """Use supplied embeddings and avoid loading any image inference model."""
    propagator = _Propagator()
    guidance = MaskGuidance(MaskGuidanceConfig("unused.pt", device="cpu"), propagator=propagator) if guided else None
    options = dict(
        use_embeddings=True,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        min_hits=1,
        det_thresh=0.5,
        iou_threshold=0.4,
        use_second_pass=True,
        second_pass_min_hits=0,
        second_iou_thresh=0.55,
    )
    options.update(kwargs)
    return OccluBoost(config=OccluBoostConfig(**options), mask_guidance=guidance), propagator


def _rows(score: float = 0.95) -> np.ndarray:
    """Keep an ignored leading detection to expose incorrect subset indexing."""
    return np.array(
        [[90, 10, 110, 40, 0.01, 0], [10, 10, 30, 40, score, 0], [14, 10, 34, 40, score, 0]],
        dtype=np.float32,
    )


def _update(tracker: OccluBoost, rows: np.ndarray, index: int, *, sequence: str = "test") -> np.ndarray:
    """Supply a real frame and cached appearance vectors at the public boundary."""
    sample_id = f"{sequence}/{index}"
    detections = Detections(
        Boxes(torch.from_numpy(np.ascontiguousarray(rows[:, :4]))),
        scores=torch.from_numpy(np.ascontiguousarray(rows[:, 4])),
        class_ids=torch.from_numpy(np.ascontiguousarray(rows[:, 5], dtype=np.int64)),
        embeddings=torch.tensor([[1.0, 0.0]]).repeat(len(rows), 1),
        sample_id=sample_id,
    )
    frame = Frame(
        torch.zeros((3, *FRAME_SHAPE), dtype=torch.uint8),
        sample_id=sample_id,
        sequence_id=sequence,
        frame_index=index,
    )
    result = tracker.update(detections, frame)
    result.validate()
    assert result.sample_id == sample_id
    assert result.masks is None
    return result.to_aabb_rows().numpy()


def _mask(box: np.ndarray) -> torch.Tensor:
    foreground = torch.zeros(FRAME_SHAPE, dtype=torch.bool)
    x1, y1, x2, y2 = box.astype(int)
    foreground[y1:y2, x1:x2] = True
    return foreground


@pytest.mark.parametrize("score", [0.95, 0.30], ids=["high_confidence", "low_confidence"])
def test_temporal_masks_change_assignment_and_preserve_original_detection_indices(score: float) -> None:
    """Real conditioning must receive original IDs and the proper stage subset."""
    guided, propagator = _tracker()
    baseline, _ = _tracker(guided=False)
    initial = _rows()
    first = _update(guided, initial, 0)
    _update(baseline, initial, 0)
    identities = {int(row[7]): int(row[4]) for row in first}
    assert set(identities) == {1, 2}
    for detection_index, identity in identities.items():
        propagator.masks[identity] = _mask(initial[3 - detection_index, :4])

    expected = _update(baseline, _rows(score), 1)
    actual = _update(guided, _rows(score), 1)

    assert {int(row[7]): int(row[4]) for row in expected} == identities
    assert {int(row[7]): int(row[4]) for row in actual} == {1: identities[2], 2: identities[1]}
    np.testing.assert_allclose(actual[:, 5], score)
    assert len(propagator.calls) == 2
    assert propagator.calls[0] == (0, {}, {})
    assert propagator.calls[1][2] == {}
    for detection_index, identity in identities.items():
        np.testing.assert_array_equal(propagator.calls[1][1][identity], initial[detection_index, :4])
        np.testing.assert_array_equal(guided._mask_guidance._active_boxes[identity], initial[3 - detection_index, :4])


@pytest.mark.parametrize("score", [0.95, 0.30], ids=["high_confidence", "low_confidence"])
def test_temporal_masks_do_not_admit_geometrically_isolated_pairs(score: float) -> None:
    """A perfect propagated mask cannot rescue a pair rejected by geometry."""
    guided, propagator = _tracker()
    baseline, _ = _tracker(guided=False)
    initial = _rows()[:2]
    identity = int(_update(guided, initial, 0)[0, 4])
    _update(baseline, initial, 0)
    moved = _rows(score)[:2]
    moved[1, [0, 2]] += 60
    propagator.masks[identity] = _mask(moved[1, :4])

    expected = _update(baseline, moved, 1)
    actual = _update(guided, moved, 1)

    np.testing.assert_array_equal(actual, expected)
    assert identity not in actual[:, 4]


def test_enabled_guidance_protects_clear_pairs_before_masks_are_available() -> None:
    """The real tracker hook must apply clear-pair protection with empty masks."""
    guided, _ = _tracker()
    baseline, _ = _tracker(guided=False)
    initial = _rows()
    _update(guided, initial, 0)
    _update(baseline, initial, 0)
    similarity = np.array([[0.8, 0.1], [0.1, 0.8]])
    assert guided.guidance_masks == {}

    actual = guided._condition_similarity(similarity, guided.trackers, initial[1:, :4], threshold=0.4)
    disabled = baseline._condition_similarity(similarity, baseline.trackers, initial[1:, :4], threshold=0.4)

    np.testing.assert_allclose(actual, [[0.8, -19.9], [-19.9, 0.8]])
    np.testing.assert_array_equal(disabled, similarity)
    np.testing.assert_array_equal(similarity, [[0.8, 0.1], [0.1, 0.8]])


def test_next_frame_prompts_use_copied_raw_detections_instead_of_kalman_geometry() -> None:
    """Confidence filtering and caller mutation must not corrupt causal prompts."""
    tracker, propagator = _tracker()
    initial = _rows()[:2]
    identity = int(_update(tracker, initial, 0)[0, 4])
    moved = initial.copy()
    moved[1, [0, 2]] += 3
    result = _update(tracker, moved, 1)
    saved_box = moved[1, :4].copy()
    assert not np.array_equal(result[0, :4], saved_box)
    moved[1, :4] += 20

    _update(tracker, np.empty((0, 6), dtype=np.float32), 2)

    np.testing.assert_array_equal(propagator.calls[1][1][identity], initial[1, :4])
    np.testing.assert_array_equal(propagator.calls[2][1][identity], saved_box)
    assert propagator.calls[2][2] == {}


def test_late_tracks_are_seeded_after_confirmation_and_keep_observations_during_recovery() -> None:
    """Realistic hit streaks must neither seed tentative tracks nor lose confirmed ones."""
    tracker, propagator = _tracker(min_hits=3)
    original = _rows()[:2]
    for index in range(3):
        first = _update(tracker, original, index)
    first_id = int(first[0, 4])
    late = np.vstack((original, [70, 10, 90, 40, 0.95, 0])).astype(np.float32)
    for index in range(3, 6):
        result = _update(tracker, late, index)
        assert set(result[:, 4].astype(int)) == {first_id}
        assert set(propagator.calls[-1][1]) == {first_id}
        assert propagator.calls[-1][2] == {}
    confirmed = _update(tracker, late, 6)
    late_id = next(int(identity) for identity in confirmed[:, 4] if identity != first_id)

    _update(tracker, late, 7)

    assert set(propagator.calls[-1][1]) == {first_id, late_id}
    assert set(propagator.calls[-1][2]) == {late_id}
    np.testing.assert_array_equal(propagator.calls[-1][2][late_id], late[2, :4])
    empty = np.empty((0, 6), dtype=np.float32)
    _update(tracker, empty, 8)
    _update(tracker, empty, 9)
    assert len(_update(tracker, late, 10)) == 0  # A recovered hit streak is not emitted yet.
    _update(tracker, late, 11)
    assert set(propagator.calls[-1][1]) == {first_id, late_id}
    assert propagator.calls[-1][2] == {}
    np.testing.assert_array_equal(propagator.calls[-1][1][late_id], late[2, :4])


def test_lost_track_memory_is_retained_until_retirement_and_reset_clears_reused_ids() -> None:
    tracker, propagator = _tracker(max_age=2)
    rows = _rows()[:2]
    first = _update(tracker, rows, 0)
    identity = int(first[0, 4])
    propagator.masks[identity] = _mask(rows[1, :4])
    _update(tracker, rows, 1)
    empty = np.empty((0, 6), dtype=np.float32)
    for index in (2, 3):
        assert len(_update(tracker, empty, index)) == 0
        assert propagator.retained[-1] == {identity}
        assert set(tracker.guidance_masks) == {identity}

    _update(tracker, empty, 4)

    assert propagator.retained[-1] == set()
    assert tracker.guidance_masks == {}
    assert propagator.masks == {}
    assert [call[0] for call in propagator.calls] == list(range(5))
    tracker.reset()
    assert propagator.resets == 1
    assert tracker.guidance_masks == {}
    restarted = _update(tracker, rows, 0, sequence="restarted")
    np.testing.assert_array_equal(restarted, first)
    assert propagator.calls[-1] == (0, {}, {})
