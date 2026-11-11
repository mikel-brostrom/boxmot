"""Prompt-driven EdgeTAM tracking without model weights or accelerator inference."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from types import ModuleType

import numpy as np
import pytest
import torch

from boxmot import EdgeTAM, EdgeTAMConfig, EdgeTAMModelConfig
from boxmot.structures import Boxes, Detections, Frame, GeometryKind, Tracks
from boxmot.trackers.common.base import BaseTracker
from boxmot.trackers.common.specs import TrackerFamily


class _Propagator:
    """Deterministically move masks while recording actual prompt corrections."""

    def __init__(self) -> None:
        self.calls = []
        self.masks = {}
        self.removed = []
        self.resets = 0

    def process(self, frame_index, frame, prompts=()):
        self.calls.append((frame_index, frame.copy(), prompts))
        self.masks = {identity: torch.roll(mask, 1, dims=1) for identity, mask in self.masks.items()}
        for prompt in prompts:
            mask = torch.zeros(frame.shape[:2], dtype=torch.bool)
            if prompt.box is not None:
                x1, y1, x2, y2 = (int(value) for value in prompt.box)
                mask[y1:y2, x1:x2] = True
            else:
                for (x, y), label in zip(prompt.points, prompt.labels, strict=True):
                    mask[int(y), int(x)] = bool(label)
            self.masks[prompt.object_id] = mask
        return self.masks

    def remove_object(self, object_id):
        self.removed.append(object_id)
        self.masks.pop(object_id, None)

    def reset(self):
        self.resets += 1
        self.masks.clear()


def _empty(sample_id: str | None = None) -> Detections:
    return Detections(
        geometry=Boxes(torch.empty((0, 4), dtype=torch.float32)),
        scores=torch.empty(0, dtype=torch.float32),
        class_ids=torch.empty(0, dtype=torch.int64),
        sample_id=sample_id,
    )


def _frame(sample_id: str = "frame") -> Frame:
    return Frame(image=torch.zeros((3, 12, 16), dtype=torch.uint8), sample_id=sample_id)


def test_prompted_tracker_inherits_validated_boundary_and_declares_model_memory() -> None:
    tracker = EdgeTAM()
    assert EdgeTAM.update is BaseTracker.update
    assert tracker.capabilities.family is TrackerFamily.MULTIMODAL
    assert tracker.capabilities.geometry_kinds == frozenset({GeometryKind.AABB})
    assert tracker.requirements.frame_pixels
    assert not tracker.requirements.masks
    assert not tracker.requirements.embeddings
    assert not tracker.capabilities.accepts_masks


def test_model_loads_on_first_prompted_frame_and_uses_separate_model_settings(monkeypatch) -> None:
    module = ModuleType("boxmot.segmentors.propagation.prompted")
    created = []
    propagator = _Propagator()

    def construct(checkpoint, **options):
        created.append((checkpoint, options))
        return propagator

    module.EdgeTAMPromptPropagator = construct
    monkeypatch.setitem(sys.modules, module.__name__, module)
    tracker = EdgeTAM(
        EdgeTAMConfig(max_objects=4, batch_size=2),
        prompt_model=EdgeTAMModelConfig(checkpoint="chosen.pt", device="cpu"),
    )
    for _ in range(2):
        result = tracker.update(_empty(), _frame())
        assert result.masks.values.shape == (0, 12, 16)
    assert not created
    tracker.add_box((2, 3, 7, 9))
    tracker.update(_empty(), _frame())
    assert created == [("chosen.pt", {"device": "cpu", "max_objects": 4, "batch_size": 2})]
    assert propagator.calls[0][0] == 0


def test_boxes_and_points_emit_same_frame_masks_derived_boxes_and_stable_ids() -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    box_id = tracker.add_box((2, 3, 7, 9), object_id=41, class_id=6)
    point_id = tracker.add_points(((10, 2), (11, 4), (12, 5)), (1, 1, 0), object_id=3)
    result = tracker.update(_empty(), _frame("first"))
    assert isinstance(result, Tracks)
    assert result.sample_id == "first"
    assert result.track_ids.tolist() == [point_id, box_id]
    assert result.class_ids.tolist() == [0, 6]
    assert result.detection_indices.tolist() == [-1, -1]
    assert result.geometry.values.tolist() == [[10, 2, 12, 5], [2, 3, 7, 9]]
    assert result.scores.tolist() == [1, 1]
    assert result.masks.values.shape == (2, 12, 16)
    next_result = tracker.update(_empty(), _frame("second"))
    assert next_result.track_ids.tolist() == [point_id, box_id]
    assert next_result.geometry.values.tolist() == [[11, 2, 13, 5], [3, 3, 8, 9]]
    assert backend.calls[1][2] == ()
    assert tracker.frame_count == 2


def test_corrections_preserve_other_objects_and_existing_class() -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    tracker.add_box((2, 3, 7, 9), object_id=5, class_id=12)
    tracker.add_box((9, 2, 12, 5), object_id=2)
    tracker.update(_empty(), _frame())
    tracker.add_points(((1, 1), (2, 2)), object_id=5)
    tracker.add_points(((4, 4),), object_id=5)
    result = tracker.update(_empty(), _frame())
    assert len(backend.calls[-1][2]) == 1
    assert backend.calls[-1][2][0].object_id == 5
    assert result.track_ids.tolist() == [2, 5]
    assert result.class_ids.tolist() == [0, 12]
    assert result.geometry.values.tolist() == [[10, 2, 13, 5], [4, 4, 5, 5]]


def test_canonical_ids_preserve_full_int64_without_float32_rounding() -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    object_id, class_id = 2**62 + 7, 2**62 + 9
    tracker.add_box((1, 2, 3, 4), object_id=object_id, class_id=class_id)
    result = tracker.update(_empty(), _frame())
    assert result.track_ids.tolist() == [object_id]
    assert result.class_ids.tolist() == [class_id]


def test_packed_outputs_have_float64_rows_and_exact_ids() -> None:
    tracker = EdgeTAM(propagator=_Propagator())
    tracker.add_box((1, 2, 3, 4), object_id=2**40 + 3, class_id=2**40 + 5)
    result = tracker.update(np.empty((0, 6)), np.zeros((12, 16, 3), dtype=np.uint8))
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert result.tolist() == [[1, 2, 3, 4, 2**40 + 3, 1, 2**40 + 5, -1]]


@pytest.mark.parametrize("object_id,class_id", [(2**53 + 1, 0), (0, 2**53 + 1)])
def test_inexact_packed_ids_are_rejected_before_propagation(object_id: int, class_id: int) -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    tracker.add_box((1, 2, 3, 4), object_id=object_id, class_id=class_id)
    with pytest.raises(ValueError, match="exact float64"):
        tracker.update(np.empty((0, 6)), np.zeros((12, 16, 3), dtype=np.uint8))
    assert backend.calls == []
    assert tracker.frame_count == tracker._sample_index == 0


def test_output_masks_are_owned_and_empty_masks_keep_object_memory() -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    identity = tracker.add_box((1, 2, 3, 4))
    result = tracker.update(_empty(), _frame())
    backend.masks[identity].zero_()
    assert result.masks.values.any()
    invisible = tracker.update(_empty(), _frame())
    assert not len(invisible)
    assert invisible.masks.values.shape == (0, 12, 16)
    assert tracker.object_ids == (identity,)
    tracker.add_points(((2, 2),), object_id=identity)
    visible = tracker.update(_empty(), _frame())
    assert visible.track_ids.tolist() == [identity]


@pytest.mark.parametrize(
    "foreground, expected_box",
    [
        pytest.param((), None, id="empty"),
        pytest.param(((0, 0),), [0, 0, 1, 1], id="top-left-pixel"),
        pytest.param(((11, 15),), [15, 11, 16, 12], id="bottom-right-pixel"),
        pytest.param(((2, 3), (7, 10)), [3, 2, 11, 8], id="sparse"),
        pytest.param(((0, 15), (11, 0)), [0, 0, 16, 12], id="opposite-borders"),
        pytest.param(None, [0, 0, 16, 12], id="full-frame"),
    ],
)
def test_mask_bounds_preserve_exclusive_maxima_and_empty_visibility(
    monkeypatch: pytest.MonkeyPatch,
    foreground: tuple[tuple[int, int], ...] | None,
    expected_box: list[int] | None,
) -> None:
    mask = torch.ones((12, 16), dtype=torch.bool) if foreground is None else torch.zeros((12, 16), dtype=torch.bool)
    for row, column in foreground or ():
        mask[row, column] = True
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    identity = tracker.add_box((1, 1, 2, 2))
    monkeypatch.setattr(backend, "process", lambda *_args: {identity: mask})

    result = tracker.update(_empty(), _frame())

    assert tracker.object_ids == (identity,)
    if expected_box is None:
        assert not len(result)
        assert result.masks.values.shape == (0, 12, 16)
    else:
        assert result.track_ids.tolist() == [identity]
        assert result.geometry.values.tolist() == [expected_box]
        torch.testing.assert_close(result.masks.values[0], mask)


def test_capacity_removal_and_reset_preserve_loaded_backend() -> None:
    backend = _Propagator()
    tracker = EdgeTAM(EdgeTAMConfig(max_objects=1), propagator=backend)
    identity = tracker.add_box((1, 2, 3, 4))
    with pytest.raises(ValueError, match="at most 1"):
        tracker.add_box((2, 3, 4, 5))
    tracker.update(_empty(), _frame())
    tracker.remove_object(identity)
    assert tracker.object_ids == ()
    assert identity in backend.removed
    assert not len(tracker.update(_empty(), _frame()))
    assert tracker.add_box((1, 2, 3, 4)) == identity + 1
    tracker.reset()
    assert backend.resets == 1
    assert tracker.object_ids == ()
    assert tracker.frame_count == tracker._sample_index == 0
    assert tracker.add_box((1, 2, 3, 4)) == 0
    tracker.update(_empty(), _frame())
    assert backend.calls[-1][0] == 0
    assert tracker._propagator is backend


@pytest.mark.parametrize("packed", [False, True])
def test_detector_observations_are_rejected_without_consuming_prompts(packed: bool) -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    tracker.add_box((1, 2, 3, 4))
    rows = np.asarray([[1, 2, 3, 4, 0.9, 0]], dtype=np.float64)
    detections = (
        rows
        if packed
        else Detections(
            geometry=Boxes(torch.tensor(rows[:, :4], dtype=torch.float32)),
            scores=torch.tensor(rows[:, 4], dtype=torch.float32),
            class_ids=torch.tensor(rows[:, 5], dtype=torch.int64),
        )
    )
    with pytest.raises(ValueError, match="manual prompts"):
        tracker.update(detections, _frame())
    assert tracker.object_ids == (0,)
    assert tracker.frame_count == tracker._sample_index == 0
    assert not backend.calls
    assert len(tracker.update(_empty(), _frame())) == 1


def test_invalid_bounds_and_frames_do_not_advance_backend_or_consume_prompts() -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    identity = tracker.add_box((1, 2, 20, 4))
    with pytest.raises(ValueError, match="frame dimensions"):
        tracker.update(_empty(), _frame())
    assert not backend.calls
    assert tracker.frame_count == tracker._sample_index == 0
    tracker.add_box((1, 2, 3, 4), object_id=identity)
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(_empty())
    assert not backend.calls
    tracker.update(_empty(), _frame())
    with pytest.raises(ValueError, match="frame dimensions"):
        tracker.add_points(((16, 1),))
    assert tracker.object_ids == (identity,)
    with pytest.raises(ValueError, match="constant frame dimensions"):
        tracker.update(_empty(), np.zeros((20, 16, 3), dtype=np.uint8))
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "method, values, options, error",
    [
        ("add_box", (2, 2, 1, 1), {}, ValueError),
        ("add_box", (1, 2, float("nan"), 3), {}, ValueError),
        ("add_box", (1, 2, 3, 4), {"object_id": True}, TypeError),
        ("add_box", (1, 2, 3, 4), {"object_id": -1}, ValueError),
        ("add_box", (1, 2, 3, 4), {"class_id": 2**63}, ValueError),
        ("add_points", (), {}, ValueError),
        ("add_points", ((1, 2),), {"labels": (2,)}, ValueError),
        ("add_points", ((1, 2),), {"labels": (0,)}, ValueError),
        ("add_points", ((1, 2),), {"labels": (1, 0)}, ValueError),
    ],
)
def test_invalid_prompt_does_not_reserve_identity(method, values, options, error) -> None:
    tracker = EdgeTAM()
    with pytest.raises(error):
        getattr(tracker, method)(values, **options)
    assert tracker.object_ids == ()
    assert tracker.add_box((1, 2, 3, 4)) == 0


def test_negative_only_correction_is_forwarded_for_an_already_tracked_object() -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    identity = tracker.add_box((1, 2, 3, 4))
    tracker.update(_empty(), _frame())
    tracker.add_points(((1, 2),), (0,), object_id=identity)
    tracker.update(_empty(), _frame())
    assert backend.calls[-1][0] == 1
    (prompt,) = backend.calls[-1][2]
    assert prompt.object_id == identity and prompt.box is None and prompt.labels == (0,)
    assert tracker.object_ids == (identity,)
    tracker.update(_empty(), _frame())
    assert backend.calls[-1][0] == 2


def test_queued_new_object_still_needs_foreground_context() -> None:
    tracker = EdgeTAM(propagator=_Propagator())
    identity = tracker.add_points(((1, 2),))
    with pytest.raises(ValueError, match="foreground point"):
        tracker.add_points(((1, 2),), (0,), object_id=identity)
    assert tracker._pending_prompts[identity].labels == (1,)
    assert tracker.update(_empty(), _frame()).track_ids.tolist() == [identity]


@pytest.mark.parametrize("operation", ["remove_object", "reset"])
def test_released_object_requires_foreground_to_reinitialize(operation: str) -> None:
    tracker = EdgeTAM(propagator=_Propagator())
    identity = tracker.add_box((1, 2, 3, 4))
    tracker.update(_empty(), _frame())
    if operation == "remove_object":
        tracker.remove_object(identity)
    else:
        tracker.reset()
    with pytest.raises(ValueError, match="foreground point"):
        tracker.add_points(((1, 2),), (0,), object_id=identity)
    assert not tracker.object_ids


def test_pending_box_provides_context_for_negative_point_corrections() -> None:
    backend = _Propagator()
    tracker = EdgeTAM(propagator=backend)
    identity = tracker.add_box((1, 2, 3, 4))
    tracker.add_points(((1, 2),), (0,), object_id=identity)
    tracker.update(_empty(), _frame())
    (prompt,) = backend.calls[-1][2]
    assert prompt.object_id == identity
    assert prompt.box == (1, 2, 3, 4)
    assert prompt.points == ((1, 2),)
    assert prompt.labels == (0,)


@pytest.mark.parametrize("options", [{"per_class": True}, {"per_class": 1}, {"class_ids": [-1]}, {"is_obb": True}])
def test_invalid_runtime_metadata_is_rejected(options) -> None:
    with pytest.raises((TypeError, ValueError)):
        EdgeTAM(**options)


def test_config_separates_model_selection_and_is_frozen() -> None:
    config = EdgeTAMConfig()
    assert config.max_objects == 32
    assert config.batch_size == 8
    assert "checkpoint" not in config.fields()
    assert "device" not in config.fields()
    with pytest.raises(FrozenInstanceError):
        config.max_objects = 10
    with pytest.raises(ValueError, match="max_objects"):
        EdgeTAMConfig(max_objects=0)
    with pytest.raises(TypeError, match="batch_size"):
        EdgeTAMConfig(batch_size=True)
    assert EdgeTAMModelConfig(device="0").device == "cuda:0"
    with pytest.raises(ValueError, match="checkpoint"):
        EdgeTAMModelConfig(checkpoint="")
