"""Causal prompting, shared features, bounded identities and compact memory."""

from __future__ import annotations

import weakref

import numpy as np
import pytest
import torch
from PIL import Image

from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator


class RecordingPredictor:
    """Record package calls and enforce the memory reads of the official model."""

    image_size = 16
    training = False
    add_tpos_enc_to_obj_ptrs = False
    memory_temporal_stride_for_eval = 1
    num_maskmem = 7
    max_obj_ptrs_in_encoder = 16
    non_overlap_masks = False
    non_overlap_masks_for_mem_enc = False
    clear_non_cond_mem_around_input = False
    add_all_frames_to_correct_as_cond = False
    use_obj_ptrs_in_encoder = True

    def __init__(self) -> None:
        self.events = []
        self.batches = []
        self.masks = {}
        self.prompt_refs = []

    def _feature(self, state, frame_index):
        if frame_index not in state["cached_features"]:
            state["cached_features"] = {frame_index: (state["images"][frame_index], torch.ones((1, 4, 4)))}
            self.events.append(("encode", frame_index))

    def _output(self, state, frame_index, track_id):
        self._feature(state, frame_index)
        shape = (state["video_height"], state["video_width"])
        logits = torch.tensor(self.masks.get((frame_index, track_id), np.zeros(shape)))[None, None].float()
        positions = state["constants"].setdefault("maskmem_pos_enc", [torch.zeros((1, 4, 8))])
        return {
            "pred_masks": logits,
            "maskmem_features": torch.full((1, 4, 8), float(frame_index), dtype=torch.bfloat16),
            "maskmem_pos_enc": positions,
            "obj_ptr": torch.tensor([[float(track_id), float(frame_index)]]),
            "object_score_logits": torch.ones((1, 1)),
        }

    def _prompt(self, kind, state, frame_index, track_id, values):
        assert state["obj_ids"] == []
        state["obj_ids"] = [track_id]
        inputs = values.clone() if isinstance(values, torch.Tensor) else torch.tensor(np.array(values, copy=True))
        state["mask_inputs_per_obj" if kind == "mask" else "point_inputs_per_obj"][0] = {frame_index: inputs}
        self.prompt_refs.append(weakref.ref(inputs))
        state["temp_output_dict_per_obj"][0] = {frame_index: self._output(state, frame_index, track_id)}
        self.events.append((kind, frame_index, track_id, inputs.clone()))

    def add_new_points_or_box(self, state, frame_index, track_id, *, box):
        self._prompt("box", state, frame_index, track_id, box)

    def add_new_mask(self, state, frame_index, track_id, *, mask):
        self._prompt("mask", state, frame_index, track_id, mask)

    def propagate_in_video_preflight(self, state):
        assert len(state["obj_ids"]) == 1
        state["output_dict"]["cond_frame_outputs"].update(state["temp_output_dict_per_obj"][0])

    def _run_single_frame_inference(self, *, inference_state, output_dict, frame_idx, **kwargs):
        batch_size = kwargs.pop("batch_size")
        assert kwargs == {
            "is_init_cond_frame": False,
            "point_inputs": None,
            "mask_inputs": None,
            "reverse": False,
            "run_mem_encoder": True,
        }
        assert frame_idx < inference_state["num_frames"]
        conditioning = next(iter(output_dict["cond_frame_outputs"].values()))
        track_ids = tuple(int(value) for value in conditioning["obj_ptr"][:, 0])
        assert len(track_ids) == batch_size
        assert "maskmem_features" in conditioning
        recent = output_dict["non_cond_frame_outputs"]
        # Match upstream's distinct spatial-memory and pointer lookup windows.
        for previous in range(max(0, frame_idx - 6), frame_idx):
            if previous in recent:
                assert {"maskmem_features", "maskmem_pos_enc", "obj_ptr"} <= recent[previous].keys()
        for previous in range(max(0, frame_idx - 15), frame_idx):
            if previous in recent:
                assert "obj_ptr" in recent[previous]
        for history in output_dict.values():
            for memory in history.values():
                assert tuple(int(value) for value in memory["obj_ptr"][:, 0]) == track_ids
                for name, value in memory.items():
                    if name == "maskmem_pos_enc":
                        assert all(tensor.shape[0] == batch_size for tensor in value)
                    else:
                        assert value.shape[0] == batch_size
        self.batches.append((frame_idx, track_ids))
        self.events.extend(("infer", frame_idx, track_id) for track_id in track_ids)
        outputs = [self._output(inference_state, frame_idx, track_id) for track_id in track_ids]
        current = {
            name: [torch.cat([output[name][i] for output in outputs]) for i in range(len(value))]
            if name == "maskmem_pos_enc"
            else torch.cat([output[name] for output in outputs])
            for name, value in outputs[0].items()
        }
        return current, current["pred_masks"]

    def _get_orig_video_res_output(self, state, predictions):
        return predictions, predictions


@pytest.fixture
def predictor():
    return RecordingPredictor()


@pytest.fixture
def propagator(predictor):
    return EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor)


@pytest.fixture
def frame():
    return np.zeros((12, 24, 3), dtype=np.uint8)


def _boxes(*ids):
    return {track_id: np.array([0, 0, 8, 10]) for track_id in ids}


def test_injected_official_perceiver_supports_batches_without_mutating_weights(predictor) -> None:
    """Raw official predictors receive the same local batch fix as loaded predictors."""
    upstream = pytest.importorskip("sam2.modeling.perceiver")
    encoding = pytest.importorskip("sam2.modeling.position_encoding")
    perceiver = upstream.PerceiverResampler(
        dim=8,
        depth=1,
        dim_head=4,
        heads=2,
        num_latents=0,
        num_latents_2d=4,
        position_encoding=encoding.PositionEmbeddingSine(num_pos_feats=8, normalize=True),
    ).eval()
    features = torch.arange(2 * 8 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8, 8) / 1000
    original_method = type(perceiver).forward_2d
    original = {name: (parameter, parameter.clone()) for name, parameter in perceiver.named_parameters()}
    with torch.inference_mode():
        expected = [perceiver.forward_2d(row) for row in features.split(1)]
    predictor.spatial_perceiver = perceiver

    propagator = EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor)
    with torch.inference_mode():
        actual, actual_positions = propagator.predictor.spatial_perceiver.forward_2d(features)

    torch.testing.assert_close(actual, torch.cat([result[0] for result in expected]), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_positions, torch.cat([result[1] for result in expected]), rtol=0, atol=0)
    assert type(perceiver).forward_2d is original_method
    for name, parameter in perceiver.named_parameters():
        assert parameter is original[name][0]
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter, original[name][1])


def test_previous_frame_prompt_and_borrowed_device_tensor_output(propagator, predictor, frame):
    logits = np.zeros(frame.shape[:2], dtype=np.float32)
    logits[1:3, 2:4] = 0.01
    logits[5, 6] = -1
    predictor.masks[1, 7] = logits
    assert propagator.propagate(0, frame, {}, {}) == {}
    result = propagator.propagate(1, frame, _boxes(7), {})
    torch.testing.assert_close(result[7], torch.from_numpy(logits > 0))
    assert result[7] is propagator._masks[7]
    assert result[7].device == propagator.device
    assert result[7].dtype == torch.bool
    result.clear()
    assert 7 in propagator._masks
    assert [(event[0], *event[1:3]) for event in predictor.events if event[0] != "encode"] == [
        ("box", 0, 7),
        ("infer", 1, 7),
    ]


def test_collective_reseed_uses_exact_previous_masks_and_reuses_features(propagator, predictor, frame):
    predictor.masks[1, 0] = np.ones(frame.shape[:2])
    propagator.propagate(0, frame, {}, {})
    propagator.propagate(1, frame, _boxes(0), _boxes(0))
    old_memory = weakref.ref(propagator._objects[0]["cond_frame_outputs"][0]["maskmem_features"])
    result = propagator.propagate(2, frame, _boxes(0, 1), _boxes(1))
    assert set(result) == {0, 1}
    assert [(event[0], *event[1:3]) for event in predictor.events if event[0] in {"box", "mask"}] == [
        ("box", 0, 0),
        ("mask", 1, 0),
        ("box", 1, 1),
    ]
    torch.testing.assert_close(
        next(event[3] for event in predictor.events if event[0] == "mask"),
        torch.ones(frame.shape[:2], dtype=torch.bool),
    )
    assert [event[1] for event in predictor.events if event[0] == "encode"] == [0, 1, 2]
    assert old_memory() is None
    assert all(reference() is None for reference in predictor.prompt_refs)
    assert all(set(outputs["cond_frame_outputs"]) == {1} for outputs in propagator._objects.values())


def test_occluded_prompt_retries_without_new_identity_event(propagator, frame):
    active = {0: np.array([0, 0, 10, 10]), 1: np.array([9, 0, 19, 11])}
    propagator.propagate(0, frame, {}, {})
    assert set(propagator.propagate(1, frame, active, active)) == {1}
    assert set(propagator.propagate(2, frame, active, {})) == {1}
    active[0] = np.array([0, 0, 8, 10])
    assert set(propagator.propagate(3, frame, active, {})) == {0, 1}


@pytest.mark.parametrize("prompt_overlap,expected", [(0.09, {1}), (0.10, {1}), (0.11, {0, 1})])
def test_configured_prompt_overlap_controls_admission(predictor, frame, prompt_overlap, expected):
    propagator = EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor, prompt_overlap=prompt_overlap)
    active = {0: np.array([0, 0, 10, 10]), 1: np.array([9, 0, 19, 11])}
    propagator.propagate(0, frame, {}, {})

    result = propagator.propagate(1, frame, active, active)

    assert set(result) == expected


@pytest.mark.parametrize("value", [0, -0.1, 1.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_prompt_overlap_rejected_before_model_loading(monkeypatch, value):
    from boxmot.segmentors.propagation import edgetam

    monkeypatch.setattr(edgetam, "build_edgetam_predictor", lambda *args: pytest.fail("Loaded invalid propagator"))
    with pytest.raises((TypeError, ValueError), match="prompt_overlap"):
        EdgeTAMMaskPropagator("unused.pt", device="cpu", prompt_overlap=value)


@pytest.mark.parametrize(
    "active",
    [
        {0: np.array([0, 0, 10, 10]), 1: np.array([1, 0, 11, 10])},
        {0: np.array([0, 0, 10, 10]), 1: np.array([0, 4, 1, 11]), 2: np.array([9, 4, 10, 11])},
    ],
)
def test_gate_uses_lower_bottom_and_maximum_not_union(propagator, frame, active):
    propagator.propagate(0, frame, {}, {})
    assert set(propagator.propagate(1, frame, active, active)) == set(active)


def test_temporal_windows_drop_masks_and_old_spatial_memory(propagator, frame):
    for index in range(50):
        propagator.propagate(index, frame, _boxes(0) if index else {}, {})
    outputs = propagator._objects[0]
    assert set(outputs["cond_frame_outputs"]) == {0}
    assert set(outputs["non_cond_frame_outputs"]) == set(range(35, 50))
    assert set(propagator._state["images"]) == {48, 49}
    for index, output in outputs["non_cond_frame_outputs"].items():
        assert set(output) == ({"obj_ptr"} if index < 44 else {"obj_ptr", "maskmem_features", "maskmem_pos_enc"})
    assert not {"output_dict", "point_inputs_per_obj", "mask_inputs_per_obj"} & propagator._state.keys()


def test_long_identity_turnover_releases_unreachable_tensors(predictor, frame):
    propagator = EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor, max_objects=3)
    old_images, old_masks, old_memory = {}, {}, {}
    for index in range(1000):
        active = _boxes(index // 40, index // 40 + 1, index // 40 + 2, index // 40 + 3)
        propagator.propagate(index, frame, active if index else {}, {})
        propagator.retain_tracks(set(active))
        assert len(propagator._objects) <= 3
        assert len(propagator._last_observed) <= 3
        assert len(propagator._state["images"]) <= 2
        assert len(propagator._state["cached_features"]) <= 1
        old_images[index] = weakref.ref(propagator._state["images"][index])
        for track_id, outputs in propagator._objects.items():
            recent = outputs["non_cond_frame_outputs"]
            assert len(recent) <= 15
            assert sum("maskmem_features" in item for item in recent.values()) <= 6
            old_masks[index, track_id] = weakref.ref(propagator._masks[track_id])
            old_memory[index, track_id] = weakref.ref(recent[index]["maskmem_features"])
        if index >= 2:
            assert old_images[index - 2]() is None
            assert all(ref() is None for (previous, _), ref in old_masks.items() if previous == index - 1)
        if index >= 6:
            assert all(ref() is None for (previous, _), ref in old_memory.items() if previous == index - 6)
        predictor.events.clear()
    # Loop locals deliberately alias the final state: release them before checking reset.
    del outputs, recent
    propagator.reset()
    assert all(ref() is None for refs in (old_images, old_masks, old_memory) for ref in refs.values())
    assert all(ref() is None for ref in predictor.prompt_refs)


def test_capacity_prioritizes_visible_existing_new_then_recent_lost(predictor, frame):
    propagator = EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor, max_objects=3)
    propagator.propagate(0, frame, {}, {})
    assert set(propagator.propagate(1, frame, _boxes(9, 8, 7, 6), {})) == {6, 7, 8}
    # Existing visible wins despite higher ID; a visible newcomer displaces lost memory.
    assert set(propagator.propagate(2, frame, _boxes(8, 5), {})) == {5, 6, 8}
    # Track 8 was observed more recently than 6, so it survives the next arrival.
    assert set(propagator.propagate(3, frame, _boxes(5, 4), {})) == {4, 5, 8}
    # Evicted IDs can receive fresh prompts when visible again.
    assert set(propagator.propagate(4, frame, _boxes(6), {})) == {4, 5, 6}


def test_occluded_newcomer_does_not_evict_lost_memory(predictor, frame):
    propagator = EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor, max_objects=1)
    propagator.propagate(0, frame, {}, {})
    propagator.propagate(1, frame, _boxes(0), {})
    active = {1: np.array([0, 0, 10, 10]), 0: np.array([9, 0, 19, 11])}
    assert set(propagator.propagate(2, frame, active, _boxes(1))) == {0}


def test_retirement_and_reset_release_state_without_copying_survivors(propagator, predictor, frame):
    propagator.propagate(0, frame, {}, {})
    propagator.propagate(1, frame, _boxes(0, 1), {})
    surviving = propagator._objects[1]
    retired = weakref.ref(propagator._objects[0]["cond_frame_outputs"][0]["maskmem_features"])
    propagator.retain_tracks({1})
    assert propagator._objects[1] is surviving and retired() is None
    assert set(propagator.propagate(2, frame, {}, {})) == {1}
    propagator.retain_tracks(set())
    assert not propagator._objects and not propagator._masks and not propagator._last_observed
    propagator.reset()
    assert propagator.predictor is predictor and propagator.last_frame_index == -1
    assert propagator.propagate(0, np.zeros((24, 48, 3), dtype=np.uint8), {}, {}) == {}


def test_empty_frames_release_cached_features_after_all_objects_retire(propagator, predictor, frame):
    propagator.propagate(0, frame, {}, {})
    propagator.propagate(1, frame, _boxes(0), {})
    image_ref = weakref.ref(propagator._state["images"][1])
    feature_ref = weakref.ref(propagator._state["cached_features"][1][1])
    propagator.retain_tracks(set())
    assert propagator.propagate(2, frame, {}, {}) == {}
    # Keep previous-frame features briefly for a possible fresh box prompt.
    assert image_ref() is not None and feature_ref() is not None
    assert propagator.propagate(3, frame, {}, {}) == {}
    assert image_ref() is None and feature_ref() is None
    assert propagator._state["cached_features"] == {}
    assert [event[1] for event in predictor.events if event[0] == "encode"] == [0, 1]


def test_arriving_bgr_preprocessing_matches_reference(propagator, frame):
    frame[:] = np.random.default_rng(3).integers(0, 256, frame.shape, dtype=np.uint8)
    rgb = Image.fromarray(frame[..., ::-1])
    expected = torch.from_numpy(np.array(rgb.resize((16, 16))) / 255.0).permute(2, 0, 1).float()
    expected -= torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    expected /= torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    propagator.propagate(0, frame, {}, {})
    frame[:] = 0
    torch.testing.assert_close(propagator._state["images"][0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("index", [-1, 1, 4])
def test_bad_order_does_not_consume_frame(propagator, frame, index):
    with pytest.raises(ValueError, match="consecutive frame indices"):
        propagator.propagate(index, frame, {}, {})
    assert propagator._state is None


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 3, 3)), np.zeros((2, 3), dtype=np.uint8), np.zeros((0, 3, 3), dtype=np.uint8)]
)
def test_rejects_invalid_frames(propagator, bad):
    with pytest.raises(ValueError, match="frames must|dimensions"):
        propagator.propagate(0, bad, {}, {})
    assert propagator._state is None


def test_resolution_change_rejected_before_advancing(propagator, frame):
    propagator.propagate(0, frame, {}, {})
    with pytest.raises(ValueError, match="constant frame dimensions"):
        propagator.propagate(1, np.zeros((24, 48, 3), dtype=np.uint8), {}, {})
    assert propagator.last_frame_index == 0 and set(propagator._state["images"]) == {0}


@pytest.mark.parametrize("track_id", [-1, True, 1.5])
def test_invalid_ids(propagator, frame, track_id):
    with pytest.raises(ValueError, match="nonnegative integers"):
        propagator.propagate(0, frame, _boxes(track_id), {})


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [2, 0, 1, 1], [0, 0, float("nan"), 1], [0, 0, 1]])
def test_bad_geometry(propagator, frame, box):
    with pytest.raises(ValueError, match="boxes must"):
        propagator.propagate(0, frame, {0: np.array(box)}, {})


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("training", True),
        ("add_tpos_enc_to_obj_ptrs", True),
        ("memory_temporal_stride_for_eval", 2),
        ("non_overlap_masks", True),
        ("non_overlap_masks_for_mem_enc", True),
        ("clear_non_cond_mem_around_input", True),
        ("add_all_frames_to_correct_as_cond", True),
    ],
)
def test_rejects_settings_without_streaming_equivalence(predictor, attribute, value):
    setattr(predictor, attribute, value)
    with pytest.raises(ValueError, match="reference EdgeTAM evaluation memory configuration"):
        EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor)


@pytest.mark.parametrize("cap", [0, -1, True, 2.5])
def test_invalid_cap_precedes_model_loading(cap):
    with pytest.raises(ValueError, match="positive integer"):
        EdgeTAMMaskPropagator("missing.pt", device="cpu", max_objects=cap)


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.5])
def test_invalid_batch_size_precedes_model_loading(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        EdgeTAMMaskPropagator("missing.pt", device="cpu", batch_size=batch_size)


def test_object_batches_include_partial_batch_without_transferring_masks(predictor, frame, monkeypatch):
    propagator = EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor, batch_size=4)
    transfers = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        if tensor.dtype == torch.bool:
            transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    propagator.propagate(0, frame, {}, {})
    masks = propagator.propagate(1, frame, _boxes(8, 2, 9, 1, 7, 4), {})

    assert predictor.batches == [(1, (1, 2, 4, 7)), (1, (8, 9))]
    assert transfers == []
    assert set(masks) == {1, 2, 4, 7, 8, 9}
    assert all(mask.dtype == torch.bool and mask.device == propagator.device for mask in masks.values())
    assert len(propagator._objects) == 6  # Batch size does not cap retained IDs.
    assert [event for event in predictor.events if event[0] == "encode"] == [("encode", 0), ("encode", 1)]


@pytest.mark.parametrize("batch_size", [2, 4, 16])
def test_batched_propagation_matches_singletons_through_history_and_turnover(frame, batch_size):
    predictors = [RecordingPredictor(), RecordingPredictor()]
    propagators = [
        EdgeTAMMaskPropagator("unused.pt", device="cpu", predictor=predictor, batch_size=size, max_objects=7)
        for predictor, size in zip(predictors, (1, batch_size))
    ]
    for frame_index in range(45):
        ids = (9, 2, 7, 4, 1, 8, 3) if frame_index < 24 else (2, 4, 10, 11, 15)
        active = _boxes(*ids) if frame_index else {}
        for predictor in predictors:
            for track_id in range(16):
                logits = np.full(frame.shape[:2], -1.0, dtype=np.float32)
                logits[(frame_index + track_id) % frame.shape[0], track_id] = 1
                predictor.masks[frame_index, track_id] = logits
        results = [propagator.propagate(frame_index, frame, active, {}) for propagator in propagators]
        assert results[0].keys() == results[1].keys()
        for track_id in results[0]:
            torch.testing.assert_close(results[0][track_id], results[1][track_id], rtol=0, atol=0)
            for kind, history in propagators[0]._objects[track_id].items():
                other = propagators[1]._objects[track_id][kind]
                assert history.keys() == other.keys()
                for index, memory in history.items():
                    assert memory.keys() == other[index].keys()
                    for name, tensor in memory.items():
                        torch.testing.assert_close(tensor, other[index][name], rtol=0, atol=0)
        if frame_index == 24:
            for propagator in propagators:
                propagator.retain_tracks(set(ids))
            assert set(propagators[1]._objects) == set(ids)
    assert max(len(ids) for _, ids in predictors[1].batches) == min(batch_size, 7)


def test_distinct_conditioning_histories_are_inferred_in_separate_batches(propagator, predictor, frame):
    propagator.propagate(0, frame, {}, {})
    propagator.propagate(1, frame, _boxes(1, 2, 3), {})
    # A separately conditioned object must not inherit another object's old memory.
    outputs = propagator._objects[2]
    outputs["cond_frame_outputs"] = {1: outputs["non_cond_frame_outputs"].pop(1)}
    predictor.batches.clear()

    propagator.propagate(2, frame, _boxes(1, 2, 3), {})

    assert predictor.batches == [(2, (1, 3)), (2, (2,))]
    assert set(propagator._objects[1]["cond_frame_outputs"]) == {0}
    assert set(propagator._objects[2]["cond_frame_outputs"]) == {1}


def test_missing_or_pointer_only_history_does_not_pad_other_objects(propagator, frame):
    for index in range(18):
        propagator.propagate(index, frame, _boxes(1, 2, 3, 4) if index else {}, {})
    del propagator._objects[2]["non_cond_frame_outputs"][3]
    memory = propagator._objects[3]["non_cond_frame_outputs"][3]
    memory["maskmem_features"] = torch.zeros((1, 4, 8))
    memory["maskmem_pos_enc"] = [torch.zeros((1, 4, 8))]

    assert list(propagator._object_batches()) == [[1, 4], [2], [3]]


def test_retained_batch_outputs_own_storage_after_other_ids_retire(propagator, predictor, frame):
    batch_refs = []
    original_infer = predictor._run_single_frame_inference

    def capture_output(**kwargs):
        current, logits = original_infer(**kwargs)
        batch_refs.extend(weakref.ref(current[name]) for name in ("maskmem_features", "obj_ptr"))
        batch_refs.extend(weakref.ref(tensor) for tensor in current["maskmem_pos_enc"])
        return current, logits

    predictor._run_single_frame_inference = capture_output
    propagator.propagate(0, frame, {}, {})
    propagator.propagate(1, frame, _boxes(1, 2, 3), {})
    assert all(reference() is None for reference in batch_refs)
    memory = propagator._objects[2]["non_cond_frame_outputs"][1]
    tensors = [memory["maskmem_features"], memory["obj_ptr"], *memory["maskmem_pos_enc"]]
    assert all(tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size() for tensor in tensors)
    mask = propagator._masks[2]
    assert mask.untyped_storage().nbytes() == mask.numel() * mask.element_size()
    retired = weakref.ref(propagator._objects[1]["non_cond_frame_outputs"][1]["maskmem_features"])

    propagator.retain_tracks({2})

    assert retired() is None
    assert propagator._objects[2]["non_cond_frame_outputs"][1] is memory


def test_wrong_batch_mask_count_is_rejected(propagator, predictor, frame):
    predictor._get_orig_video_res_output = lambda state, predictions: (predictions, predictions[:1])
    propagator.propagate(0, frame, {}, {})
    with pytest.raises(RuntimeError, match="one original-resolution mask per object ID"):
        propagator.propagate(1, frame, _boxes(1, 2), {})


def test_batched_objects_share_only_singleton_positional_constant_storage(propagator, predictor, frame):
    original_infer = predictor._run_single_frame_inference

    def broadcast_positions(**kwargs):
        current, logits = original_infer(**kwargs)
        positions = kwargs["inference_state"]["constants"]["maskmem_pos_enc"]
        current["maskmem_pos_enc"] = [tensor.expand(kwargs["batch_size"], -1, -1) for tensor in positions]
        return current, logits

    predictor._run_single_frame_inference = broadcast_positions
    for index in range(3):
        propagator.propagate(index, frame, _boxes(1, 2, 3) if index else {}, {})

    constant = propagator._state["constants"]["maskmem_pos_enc"][0]
    for outputs in propagator._objects.values():
        for memory in outputs["non_cond_frame_outputs"].values():
            position = memory["maskmem_pos_enc"][0]
            assert position.untyped_storage().data_ptr() == constant.untyped_storage().data_ptr()
            assert position.untyped_storage().nbytes() == position.numel() * position.element_size()

    batched = propagator._batch_history([1, 2, 3])
    for memory in batched["non_cond_frame_outputs"].values():
        position = memory["maskmem_pos_enc"][0]
        assert position.shape == (3, *constant.shape[1:])
        assert position.stride(0) == 0
        assert position.untyped_storage().data_ptr() == constant.untyped_storage().data_ptr()

    propagator.retain_tracks({2})
    assert propagator._objects[2]["non_cond_frame_outputs"][2]["maskmem_pos_enc"][0].shape == constant.shape


def test_broadcast_position_view_of_larger_allocation_is_copied():
    backing = torch.ones((8, 4, 8))
    broadcast = backing[:1].expand(3, -1, -1)
    current = {
        "maskmem_features": torch.ones((3, 4, 8)),
        "maskmem_pos_enc": [broadcast],
        "obj_ptr": torch.ones((3, 2)),
    }

    memory = EdgeTAMMaskPropagator._object_memory(current, 1, batch_size=3)

    position = memory["maskmem_pos_enc"][0]
    torch.testing.assert_close(position, backing[:1])
    assert position.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
    assert position.untyped_storage().nbytes() == position.numel() * position.element_size()


def test_reseed_reuses_resident_mask_tensor_without_cpu_or_numpy(propagator, predictor, frame, monkeypatch):
    predictor.masks[1, 1] = np.ones(frame.shape[:2])
    propagator.propagate(0, frame, {}, {})
    propagator.propagate(1, frame, _boxes(1), {})
    resident = propagator._masks[1]
    original_prompt = predictor.add_new_mask
    observed = []

    def capture_prompt(state, frame_index, track_id, *, mask):
        assert track_id == 1
        assert mask is resident
        observed.append(track_id)
        return original_prompt(state, frame_index, track_id, mask=mask)

    predictor.add_new_mask = capture_prompt
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *args, **kwargs: pytest.fail("Copied mask to CPU"))
    monkeypatch.setattr(torch.Tensor, "numpy", lambda *args, **kwargs: pytest.fail("Converted mask to NumPy"))

    results = propagator.propagate(2, frame, _boxes(1, 2), _boxes(2))

    assert observed == [1]
    assert set(results) == {1, 2}
    assert all(isinstance(mask, torch.Tensor) for mask in results.values())


def test_frame_byte_lookup_matches_reference_rounding_for_every_value():
    from boxmot.segmentors.propagation.edgetam import _UINT8_TO_FLOAT32

    values = np.arange(256, dtype=np.uint8)
    reference = torch.from_numpy(values / 255.0).float()

    torch.testing.assert_close(torch.from_numpy(_UINT8_TO_FLOAT32[values]), reference, rtol=0, atol=0)
