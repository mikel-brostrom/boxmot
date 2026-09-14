"""Exercise temporal guidance through all Python box trackers' public updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from boxmot.structures import Boxes, Detections, Frame, Tracks
from boxmot.trackers import MaskGuidance, MaskGuidanceConfig
from boxmot.trackers.boosttrack.tracker import BoostTrack
from boxmot.trackers.botsort.tracker import BotSort
from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.box.base import BoxTracker
from boxmot.trackers.deepocsort.tracker import DeepOcSort
from boxmot.trackers.hybridsort.tracker import HybridSort
from boxmot.trackers.occluboost.tracker import OccluBoost
from boxmot.trackers.ocsort.tracker import OcSort
from boxmot.trackers.sfsort.tracker import SFSORT
from boxmot.trackers.strongsort.tracker import StrongSort

TRACKERS = {
    "boosttrack": BoostTrack,
    "botsort": BotSort,
    "bytetrack": ByteTrack,
    "deepocsort": DeepOcSort,
    "hybridsort": HybridSort,
    "occluboost": OccluBoost,
    "ocsort": OcSort,
    "sfsort": SFSORT,
    "strongsort": StrongSort,
}
EMPTY = np.empty((0, 6), dtype=np.float32)


@dataclass
class _Call:
    index: int
    active: dict[int, np.ndarray]
    new: dict[int, np.ndarray]


@dataclass
class _Propagator:
    """Record prompts and lifetimes without loading a segmentation checkpoint."""

    max_objects: int = 32
    prompt_overlap: float = 0.10
    device: torch.device = torch.device("cpu")
    calls: list[_Call] = field(default_factory=list)
    retained: list[set[int]] = field(default_factory=list)
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    resets: int = 0
    shape: tuple[int, int] | None = None

    def propagate(self, index, frame, active_boxes, new_boxes) -> dict[int, np.ndarray]:
        if self.shape is not None and frame.shape[:2] != self.shape:
            raise ValueError("EdgeTAM requires constant frame dimensions; reset before a new resolution.")
        self.shape = frame.shape[:2]
        self.calls.append(
            _Call(
                index,
                {key: box.copy() for key, box in active_boxes.items()},
                {key: box.copy() for key, box in new_boxes.items()},
            )
        )
        return self.masks.copy()

    def retain_tracks(self, track_ids: set[int]) -> None:
        self.retained.append(set(track_ids))
        self.masks = {key: mask for key, mask in self.masks.items() if key in track_ids}

    def reset(self) -> None:
        self.resets += 1
        self.shape = None
        self.masks.clear()


def _tracker(name: str, *, guidance: MaskGuidance | None = None, min_hits: int = 3, max_age: int = 3) -> BoxTracker:
    options = {"min_hits": min_hits, "max_age": max_age, "mask_guidance": guidance, "asso_func": "iou"}
    if name in {"boosttrack", "occluboost", "botsort"}:
        options.update(use_embeddings=False, use_cmc=False)
    if name in {"botsort", "bytetrack"}:
        options["track_buffer"] = max_age
    if name == "deepocsort":
        options.update(use_embeddings=False, cmc_off=True)
    if name == "hybridsort":
        options.update(use_embeddings=False, cmc_method=None)
    if name == "strongsort":
        options["n_init"] = min_hits
    if name == "sfsort":
        options.update(frame_width=128, frame_height=96, central_timeout=max_age, marginal_timeout=max_age)
    return TRACKERS[name](**options)


def _guided(name: str, **options) -> tuple[BoxTracker, _Propagator]:
    propagator = _Propagator()
    guidance = MaskGuidance(MaskGuidanceConfig("unused-edgetam.pt", device="cpu"), propagator=propagator)
    return _tracker(name, guidance=guidance, **options), propagator


def _rows(offset: float = 0) -> np.ndarray:
    return np.array([[20 + offset, 20, 40 + offset, 60, 0.95, 0]], dtype=np.float32)


def _input(tracker: BoxTracker, rows: np.ndarray, index: int, sequence: str = "first", shape=(96, 128)):
    sample = f"{sequence}/{index}"
    embeddings = torch.zeros((len(rows), 4), dtype=torch.float32) if tracker.requirements.embeddings else None
    if embeddings is not None:
        embeddings[:, 0] = 1
    detections = Detections(
        geometry=Boxes(torch.from_numpy(np.ascontiguousarray(rows[:, :4]))),
        scores=torch.from_numpy(np.ascontiguousarray(rows[:, 4])),
        class_ids=torch.from_numpy(np.ascontiguousarray(rows[:, 5], dtype=np.int64)),
        embeddings=embeddings,
        sample_id=sample,
    )
    frame = Frame(
        torch.zeros((3, *shape), dtype=torch.uint8), sample_id=sample, sequence_id=sequence, frame_index=index
    )
    return detections, frame


def _update(tracker: BoxTracker, rows: np.ndarray, index: int, **kwargs) -> Tracks:
    return tracker.update(*_input(tracker, rows, index, **kwargs))


def _warmup(tracker: BoxTracker) -> int:
    for index in range(6):
        result = _update(tracker, _rows(), index)
    assert len(result) == 1
    return int(result.track_ids[0])


@pytest.mark.parametrize("name", TRACKERS)
def test_guidance_requires_pixels_even_with_optional_cmc_disabled_and_dimensions_configured(name: str) -> None:
    tracker, propagator = _guided(name)
    assert tracker.requirements.frame
    detections, _ = _input(tracker, _rows(), 0)

    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(detections)

    assert propagator.calls == []


@pytest.mark.parametrize("name", TRACKERS)
def test_empty_masks_preserve_canonical_outputs_and_advance_once_on_empty_frames(name: str) -> None:
    ordinary = _tracker(name)
    guided, propagator = _guided(name)
    sequence = [_rows()] * 6 + [_rows(2), EMPTY, EMPTY] + [_rows(2)] * 5

    for index, rows in enumerate(sequence):
        expected = _update(ordinary, rows, index)
        actual = _update(guided, rows, index)
        np.testing.assert_array_equal(actual.to_aabb_rows().numpy(), expected.to_aabb_rows().numpy())
        assert actual.sample_id == expected.sample_id
        assert actual.masks is None
        actual.validate()

    assert [call.index for call in propagator.calls] == list(range(len(sequence)))
    assert propagator.calls[0].active == {}
    assert propagator.calls[0].new == {}
    assert propagator.calls[8].active == {}


@pytest.mark.parametrize("name", TRACKERS)
def test_prompts_follow_previous_raw_detection_boxes_and_stop_during_gaps(name: str) -> None:
    tracker, propagator = _guided(name)
    identity = _warmup(tracker)
    moved = _rows(2)
    _update(tracker, moved, 6)
    _update(tracker, EMPTY, 7)
    _update(tracker, EMPTY, 8)

    np.testing.assert_array_equal(propagator.calls[6].active[identity], _rows()[0, :4])
    np.testing.assert_array_equal(propagator.calls[7].active[identity], moved[0, :4])
    assert propagator.calls[8].active == {}
    assert propagator.calls[7].new == {}
    assert propagator.calls[8].new == {}


@pytest.mark.parametrize("name", TRACKERS)
def test_late_arrivals_are_prompted_only_after_the_tracker_emits_them(name: str) -> None:
    tracker, propagator = _guided(name)
    first_id = _warmup(tracker)
    two = np.concatenate((_rows(), np.array([[80, 20, 100, 60, 0.95, 0]], dtype=np.float32)))
    previous = None
    saw_new_identity = False

    for index in range(6, 12):
        result = _update(tracker, two, index)
        if previous is not None:
            previously_emitted = set(previous.track_ids.tolist())
            assert set(propagator.calls[-1].active) == previously_emitted
            if previously_emitted - {first_id}:
                saw_new_identity = True
        previous = result

    assert saw_new_identity


@pytest.mark.parametrize("name", ["boosttrack", "deepocsort", "hybridsort", "ocsort"])
def test_confirmed_recovered_tracks_prompt_while_rebuilding_their_output_hit_streak(name: str) -> None:
    tracker, propagator = _guided(name)
    identity = _warmup(tracker)
    _update(tracker, EMPTY, 6)
    result = _update(tracker, _rows(2), 7)
    assert len(result) == 0

    _update(tracker, _rows(4), 8)

    np.testing.assert_array_equal(propagator.calls[8].active[identity], _rows(2)[0, :4])
    assert propagator.calls[8].new == {}


@pytest.mark.parametrize("name", TRACKERS)
def test_retirement_releases_masks_and_stops_retaining_identity(name: str) -> None:
    tracker, propagator = _guided(name, max_age=2)
    identity = _warmup(tracker)
    # Empty masks are still stored state that must be evicted on retirement.
    propagator.masks[identity] = np.zeros((96, 128), dtype=bool)

    for index in range(6, 14):
        _update(tracker, EMPTY, index)

    assert propagator.retained[6] == {identity}
    assert propagator.retained[-1] == set()
    assert propagator.masks == {}
    assert all(call.active == {} for call in propagator.calls[7:])


@pytest.mark.parametrize("name", TRACKERS)
def test_reset_clears_sequence_binding_frame_indices_masks_and_reused_ids(name: str) -> None:
    tracker, propagator = _guided(name)
    identity = _warmup(tracker)
    detections, changed_sequence = _input(tracker, _rows(), 6, sequence="second")
    with pytest.raises(ValueError, match="reset before processing sequence"):
        tracker.update(detections, changed_sequence)
    with pytest.raises(ValueError, match="increasing source frame indices"):
        _update(tracker, _rows(), 5)
    assert len(propagator.calls) == 6
    propagator.masks[identity] = np.ones((96, 128), dtype=bool)

    tracker.reset()

    assert propagator.resets == 1
    assert propagator.masks == {}
    for index in range(6):
        result = _update(tracker, _rows(), index, sequence="second")
    assert propagator.calls[6].index == 0
    assert propagator.calls[6].active == {}
    assert propagator.calls[6].new == {}
    assert result.track_ids.tolist() == [identity]


@pytest.mark.parametrize("name", TRACKERS)
def test_resolution_changes_require_reset_before_propagation(name: str) -> None:
    tracker, propagator = _guided(name)
    _update(tracker, _rows(), 0)
    with pytest.raises(ValueError, match="reset.*resolution"):
        _update(tracker, _rows(), 1, shape=(120, 160))
    assert len(propagator.calls) == 1

    tracker.reset()
    _update(tracker, _rows(), 0, sequence="second", shape=(120, 160))

    assert propagator.calls[-1].index == 0
    assert propagator.shape == (120, 160)
