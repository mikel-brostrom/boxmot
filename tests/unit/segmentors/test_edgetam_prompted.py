"""Manual prompts seed the current frame without changing other object memories."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from test_edgetam_propagation import RecordingPredictor

from boxmot.segmentors.propagation.prompted import EdgeTAMPromptPropagator
from boxmot.segmentors.propagation.prompts import EdgeTAMPrompt


class PromptPredictor(RecordingPredictor):
    """Extend the temporal-memory fake with official point and prompt-return contracts."""

    def _obj_id_to_idx(self, state, track_id):
        """Match upstream registration and its nested per-object output dictionaries."""
        if track_id not in state["obj_id_to_idx"]:
            assert not state["tracking_has_started"]
            index = len(state["obj_id_to_idx"])
            state["obj_id_to_idx"][track_id] = index
            state["obj_idx_to_id"][index] = track_id
            state["obj_ids"] = list(state["obj_id_to_idx"])
            state["point_inputs_per_obj"][index] = {}
            state["mask_inputs_per_obj"][index] = {}
            for name in ("output_dict_per_obj", "temp_output_dict_per_obj"):
                state[name][index] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        return state["obj_id_to_idx"][track_id]

    def add_new_points_or_box(self, state, frame_index, track_id, *, box=None, points=None, labels=None):
        """Refine the current prediction using upstream's frame and mask lookups."""
        obj_idx = self._obj_id_to_idx(state, track_id)
        current = self._output(state, frame_index, track_id)
        is_initial = frame_index not in state["frames_already_tracked"]
        storage_key = "cond_frame_outputs" if is_initial else "non_cond_frame_outputs"
        if not is_initial:
            assert state["frames_already_tracked"][frame_index] == {"reverse": False}
            assert state["tracking_has_started"]
            assert box is None
            previous = state["output_dict_per_obj"][obj_idx][storage_key][frame_index]
            assert previous["maskmem_features"] is None
            logits = previous["pred_masks"].clone()
            self.events.append(("refine", frame_index, track_id, logits.clone()))
            for (x, y), label in zip(points, labels, strict=True):
                logits[0, 0, int(y), int(x)] = 2 if label else -2
            current["pred_masks"] = logits
        self.events.append(("input", frame_index, track_id, box, points, labels))
        # Like upstream, point decoding has not encoded memory yet.
        current["maskmem_features"] = None
        state["point_inputs_per_obj"][obj_idx][frame_index] = (box, points, labels)
        state["temp_output_dict_per_obj"][obj_idx][storage_key][frame_index] = current
        return frame_index, [track_id], current["pred_masks"]

    def propagate_in_video_preflight(self, state):
        """Encode point-derived masks before they become future temporal memory."""
        assert len(state["obj_ids"]) == 1
        for storage_key, frames in state["temp_output_dict_per_obj"][0].items():
            for frame_index, output in frames.items():
                output["maskmem_features"] = torch.full((1, 4, 8), float(frame_index), dtype=torch.bfloat16)
                state["output_dict"][storage_key][frame_index] = output

    def _run_single_frame_inference(self, **kwargs):
        """Allow temporal prediction without encoding memory before point refinement."""
        encode_memory = kwargs["run_mem_encoder"]
        current, logits = super()._run_single_frame_inference(**{**kwargs, "run_mem_encoder": True})
        if not encode_memory:
            current["maskmem_features"] = None
        return current, logits


@pytest.fixture
def predictor() -> PromptPredictor:
    return PromptPredictor()


@pytest.fixture
def backend(predictor) -> EdgeTAMPromptPropagator:
    return EdgeTAMPromptPropagator("unused.pt", predictor=predictor, max_objects=3, batch_size=2)


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((12, 24, 3), dtype=np.uint8)


def _box(object_id: int) -> EdgeTAMPrompt:
    return EdgeTAMPrompt(object_id, box=(0, 0, 8, 10))


def test_box_and_foreground_background_points_emit_frame_zero(backend, predictor, frame) -> None:
    logits = np.zeros(frame.shape[:2], dtype=np.float32)
    logits[2:5, 3:8] = 2
    logits[6:9, 3:8] = -1
    predictor.masks[0, 4] = logits
    predictor.masks[0, 9] = logits
    point_prompt = EdgeTAMPrompt(9, points=((4, 3), (4, 7)), labels=(1, 0))
    result = backend.process(0, frame, (_box(4), point_prompt))
    assert set(result) == {4, 9}
    for object_id in result:
        torch.testing.assert_close(result[object_id], torch.from_numpy(logits > 0))
        assert result[object_id].device == backend.device
        assert backend._objects[object_id]["cond_frame_outputs"][0]["maskmem_features"] is not None
    events = [event for event in predictor.events if event[0] == "input"]
    np.testing.assert_array_equal(events[1][4], [[4, 3], [4, 7]])
    np.testing.assert_array_equal(events[1][5], [1, 0])
    assert [event[1] for event in predictor.events if event[0] == "encode"] == [0]
    assert not predictor.batches
    result.clear()
    assert set(backend._masks) == {4, 9}


def test_combined_box_and_points_use_both_official_inputs(backend, predictor, frame) -> None:
    backend.process(0, frame, (EdgeTAMPrompt(1, box=(0, 0, 8, 10), points=((1, 2),), labels=(0,)),))
    event = next(event for event in predictor.events if event[0] == "input")
    np.testing.assert_array_equal(event[3], [0, 0, 8, 10])
    np.testing.assert_array_equal(event[4], [[1, 2]])
    np.testing.assert_array_equal(event[5], [0])


def test_correction_and_new_objects_preserve_other_conditioning(backend, predictor, frame) -> None:
    backend.process(0, frame, (_box(1), _box(2)))
    unchanged = backend._objects[2]["cond_frame_outputs"][0]
    backend.process(1, frame, (EdgeTAMPrompt(1, points=((2, 3),), labels=(1,)), _box(3)))
    assert backend._objects[2]["cond_frame_outputs"][0] is unchanged
    assert set(backend._objects[1]["cond_frame_outputs"]) == {1}
    assert set(backend._objects[3]["cond_frame_outputs"]) == {1}
    assert predictor.batches == [(1, (2,)), (1, (1,))]
    backend.process(2, frame)
    assert set(predictor.batches[-2:]) == {(2, (1, 3)), (2, (2,))}
    assert len([event for event in predictor.events if event[0] == "encode"]) == 3


def test_negative_points_refine_current_prediction_and_preserve_other_objects(backend, predictor, frame) -> None:
    old_logits = np.full(frame.shape[:2], -3.0, dtype=np.float32)
    old_logits[2:5, 3:7] = 1.75
    current_logits = np.full_like(old_logits, -3.0)
    current_logits[3:7, 12:16] = 1.75  # Object has moved away from its original box.
    predictor.masks[0, 1] = old_logits
    predictor.masks[1, 1] = current_logits
    backend.process(0, frame, (_box(1), _box(2)))
    unchanged = backend._objects[2]["cond_frame_outputs"][0]

    result = backend.process(1, frame, (EdgeTAMPrompt(1, points=((13, 4),), labels=(0,)),))

    expected = current_logits > 0
    expected[4, 13] = False
    torch.testing.assert_close(result[1], torch.from_numpy(expected))
    refinement = next(event for event in predictor.events if event[0] == "refine")
    torch.testing.assert_close(refinement[3], torch.from_numpy(current_logits)[None, None])
    assert set(backend._objects[1]["cond_frame_outputs"]) == {1}
    assert backend._objects[1]["cond_frame_outputs"][1]["maskmem_features"] is not None
    assert backend._objects[2]["cond_frame_outputs"][0] is unchanged
    assert [event[1] for event in predictor.events if event[0] == "encode"] == [0, 1]
    backend.process(2, frame)
    assert backend.last_frame_index == 2
    assert set(predictor.batches[-2:]) == {(2, (1,)), (2, (2,))}


def test_negative_correction_of_invisible_object_retains_temporal_memory(backend, predictor, frame) -> None:
    backend.process(0, frame, (_box(1),))
    result = backend.process(1, frame, (EdgeTAMPrompt(1, points=((2, 3),), labels=(0,)),))
    assert not result[1].any()
    assert set(backend._objects[1]["cond_frame_outputs"]) == {1}
    assert set(backend.process(2, frame)) == {1}


def test_overlapping_prompts_are_not_filtered(backend, frame) -> None:
    assert set(backend.process(0, frame, (_box(1), _box(2), _box(3)))) == {1, 2, 3}


@pytest.mark.parametrize(
    "prompts, message",
    [
        ((_box(1), _box(1)), "one prompt per object"),
        ((_box(1), _box(2), _box(3), _box(4)), "at most 3"),
        ((EdgeTAMPrompt(1, box=(0, 0, 25, 10)),), "frame dimensions"),
        ((EdgeTAMPrompt(1, points=((24, 2),), labels=(1,)),), "frame dimensions"),
        ((EdgeTAMPrompt(1, points=((2, 12),), labels=(1,)),), "frame dimensions"),
        ((EdgeTAMPrompt(1, points=((2, 3),), labels=(0,)),), "New objects"),
    ],
)
def test_invalid_prompt_batch_does_not_consume_frame(backend, predictor, frame, prompts, message) -> None:
    with pytest.raises(ValueError, match=message):
        backend.process(0, frame, prompts)
    assert backend.last_frame_index == -1 and backend._state is None
    assert not predictor.events


def test_capacity_correction_and_bounds_validate_before_advancing(backend, predictor, frame) -> None:
    backend.process(0, frame, (_box(1), _box(2), _box(3)))
    before = backend._objects[1]
    events = len(predictor.events)
    with pytest.raises(ValueError, match="at most 3"):
        backend.process(1, frame, (_box(4),))
    with pytest.raises(ValueError, match="frame dimensions"):
        backend.process(1, frame, (EdgeTAMPrompt(1, box=(0, 0, 8, 13)),))
    assert backend.last_frame_index == 0 and set(backend._state["images"]) == {0}
    assert backend._objects[1] is before and len(predictor.events) == events
    assert set(backend.process(1, frame, (_box(1),))) == {1, 2, 3}


def test_removal_and_reset_release_state_but_keep_model(backend, predictor, frame) -> None:
    backend.process(0, frame, (_box(1), _box(2)))
    survivor = backend._objects[2]
    backend.remove_object(1)
    backend.remove_object(1)
    assert backend._objects[2] is survivor
    assert set(backend.process(1, frame)) == {2}
    backend.reset()
    assert backend.predictor is predictor and backend.last_frame_index == -1
    assert backend._state is None and not backend._objects and not backend._masks
    assert set(backend.process(0, np.zeros((10, 20, 3), dtype=np.uint8), (_box(3),))) == {3}


def test_long_stream_keeps_bounded_history_and_releases_empty_caches(backend, frame) -> None:
    backend.process(0, frame, (_box(1), _box(2)))
    for index in range(1, 60):
        correction = EdgeTAMPrompt(1, points=((2, 3),), labels=(0,)) if index % 6 == 0 else _box(1)
        backend.process(index, frame, (correction,) if index % 3 == 0 else ())
        assert len(backend._state["images"]) <= 2
        assert len(backend._state["cached_features"]) <= 1
        for outputs in backend._objects.values():
            assert len(outputs["cond_frame_outputs"]) == 1
            assert len(outputs["non_cond_frame_outputs"]) <= 15
    backend.remove_object(1)
    backend.remove_object(2)
    assert backend.process(60, frame) == {}
    assert backend.process(61, frame) == {}
    assert not backend._state["cached_features"]


@pytest.mark.parametrize("index", [-1, 1, 0.0, True])
def test_invalid_sequence_position_does_not_consume_frame(backend, frame, index) -> None:
    with pytest.raises((TypeError, ValueError)):
        backend.process(index, frame, (_box(1),))
    assert backend._state is None


def test_changed_frame_dimensions_do_not_consume_state(backend, frame) -> None:
    backend.process(0, frame, (_box(1),))
    with pytest.raises(ValueError, match="constant frame dimensions"):
        backend.process(1, np.zeros((10, 20, 3), dtype=np.uint8))
    assert backend.last_frame_index == 0


def test_prompt_copies_sequences_and_is_frozen() -> None:
    points, labels, box = [[2, 3]], [1], [0, 0, 8, 10]
    prompt = EdgeTAMPrompt(np.int64(2), box=box, points=points, labels=labels)
    points[0][0], labels[0], box[0] = 12, 0, 4
    assert prompt.points == ((2.0, 3.0),) and prompt.labels == (1,) and prompt.box == (0.0, 0.0, 8.0, 10.0)
    with pytest.raises(FrozenInstanceError):
        prompt.object_id = 4


@pytest.mark.parametrize(
    "values",
    [
        {"object_id": -1, "box": (0, 0, 2, 2)},
        {"object_id": True, "box": (0, 0, 2, 2)},
        {"object_id": 1},
        {"object_id": 1, "box": (0, 0, 0, 2)},
        {"object_id": 1, "box": (0, 0, float("inf"), 2)},
        {"object_id": 1, "box": (-1, 0, 2, 2)},
        {"object_id": 1, "box": (False, 0, 2, 2)},
        {"object_id": 1, "points": ((1, 2),)},
        {"object_id": 1, "points": ((1, 2, 3),), "labels": (1,)},
        {"object_id": 1, "points": ((1, 2),), "labels": (2,)},
        {"object_id": 1, "points": ((1, 2),), "labels": (True,)},
    ],
)
def test_invalid_prompt_structure_is_rejected(values) -> None:
    with pytest.raises((TypeError, ValueError)):
        EdgeTAMPrompt(**values)
