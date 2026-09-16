"""FP16 memory storage preserves EdgeTAM inference and prompt-preflight semantics."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from boxmot.segmentors.propagation.inference import _run_memory_encoder, _run_single_frame_inference


def _fixture(dtype: torch.dtype = torch.float16, batch: int = 2) -> tuple:
    """Include FP16 values that would lose information in an intermediate BF16 cast."""
    memory = torch.tensor([1.001953125, 1.00390625, 256.5], dtype=dtype).repeat(batch, 1, 1)
    positions = [torch.ones_like(memory)]
    cached_positions = [positions[0].clone()]
    features = [torch.zeros(batch, 3, 2, 2, dtype=dtype)]
    vision_positions = [torch.ones_like(features[0])]
    feat_sizes = [(2, 2)]
    current = {
        "maskmem_features": memory,
        "maskmem_pos_enc": positions,
        "pred_masks": torch.zeros(batch, 1, 8, 8),
        "obj_ptr": torch.zeros(batch, 3),
        "object_score_logits": torch.ones(batch, 1),
        "unused_intermediate": object(),
    }
    predictor = SimpleNamespace(
        _get_image_feature=Mock(return_value=(None, None, features, vision_positions, feat_sizes)),
        track_step=Mock(return_value=current),
        _encode_new_memory=Mock(return_value=(memory, positions)),
        _get_maskmem_pos_enc=Mock(return_value=cached_positions),
        fill_hole_area=0,
    )
    state = {"storage_device": torch.device("cpu"), "num_frames": 11}
    return predictor, state, current, features, vision_positions, feat_sizes, cached_positions


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("prompt_kind", ["points", "mask", "none"])
def test_single_frame_preserves_memory_values_and_forwards_inputs(dtype, batch, prompt_kind) -> None:
    predictor, state, current, features, positions, sizes, cached_positions = _fixture(dtype, batch)
    point_inputs = {"point_coords": torch.ones(batch, 2, 2)} if prompt_kind == "points" else None
    mask_inputs = torch.ones(batch, 1, 8, 8) if prompt_kind == "mask" else None
    previous_logits = torch.ones(batch, 1, 2, 2)
    outputs = {"cond_frame_outputs": {0: object()}}
    source = current["maskmem_features"].clone()

    actual, logits = _run_single_frame_inference(
        predictor, state, outputs, 6, batch, prompt_kind != "none", point_inputs, mask_inputs,
        True, True, previous_logits,
    )

    predictor._get_image_feature.assert_called_once_with(state, 6, batch)
    predictor.track_step.assert_called_once_with(
        frame_idx=6,
        is_init_cond_frame=prompt_kind != "none",
        current_vision_feats=features,
        current_vision_pos_embeds=positions,
        feat_sizes=sizes,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        output_dict=outputs,
        num_frames=11,
        track_in_reverse=True,
        run_mem_encoder=True,
        prev_sam_mask_logits=previous_logits,
    )
    predictor._get_maskmem_pos_enc.assert_called_once_with(state, current)
    expected = source.to(torch.float16)
    assert actual["maskmem_features"].dtype == torch.float16
    assert torch.equal(actual["maskmem_features"].view(torch.int16), expected.view(torch.int16))
    assert not torch.equal(expected, source.to(torch.bfloat16).to(torch.float16))
    assert torch.equal(current["maskmem_features"], source)
    assert actual["maskmem_pos_enc"] is cached_positions
    assert logits is current["pred_masks"] and actual["pred_masks"] is logits
    assert actual["pred_masks"].dtype == torch.float32
    assert actual["obj_ptr"] is current["obj_ptr"]
    assert actual["object_score_logits"] is current["object_score_logits"]
    assert set(actual) == set(current) - {"unused_intermediate"}


def test_single_frame_without_memory_keeps_optional_outputs_and_default_previous_logits() -> None:
    predictor, state, current, *_ = _fixture()
    current["maskmem_features"] = None
    current["maskmem_pos_enc"] = None
    predictor._get_maskmem_pos_enc.return_value = None

    actual, _ = _run_single_frame_inference(predictor, state, {}, 2, 2, False, None, None, False, False)

    assert actual["maskmem_features"] is None
    assert actual["maskmem_pos_enc"] is None
    assert predictor.track_step.call_args.kwargs["run_mem_encoder"] is False
    assert predictor.track_step.call_args.kwargs["prev_sam_mask_logits"] is None
    assert predictor.track_step.call_args.kwargs["track_in_reverse"] is False


def test_single_frame_rejects_simultaneous_point_and_mask_prompts() -> None:
    predictor, state, *_ = _fixture()
    with pytest.raises(AssertionError):
        _run_single_frame_inference(
            predictor, state, {}, 2, 2, True, {"point_coords": torch.zeros(2, 2, 2)},
            torch.zeros(2, 1, 8, 8), False, True,
        )
    predictor.track_step.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("is_mask_from_pts", [False, True])
def test_prompt_preflight_memory_preserves_fp16_values_and_position_cache(dtype, batch, is_mask_from_pts) -> None:
    predictor, state, current, features, _, sizes, cached_positions = _fixture(dtype, batch)
    masks = torch.ones(batch, 1, 8, 8)
    scores = torch.ones(batch, 1)
    source = current["maskmem_features"].clone()

    memory, positions = _run_memory_encoder(predictor, state, 6, batch, masks, scores, is_mask_from_pts)

    predictor._get_image_feature.assert_called_once_with(state, 6, batch)
    predictor._encode_new_memory.assert_called_once_with(
        current_vision_feats=features,
        feat_sizes=sizes,
        pred_masks_high_res=masks,
        object_score_logits=scores,
        is_mask_from_pts=is_mask_from_pts,
    )
    predictor._get_maskmem_pos_enc.assert_called_once_with(state, {"maskmem_pos_enc": current["maskmem_pos_enc"]})
    assert memory.dtype == torch.float16
    assert torch.equal(memory.view(torch.int16), source.to(torch.float16).view(torch.int16))
    assert not torch.equal(memory, source.to(torch.bfloat16).to(torch.float16))
    assert torch.equal(current["maskmem_features"], source)
    assert positions is cached_positions


def test_hole_filling_precedes_mask_storage_and_position_cache(monkeypatch) -> None:
    predictor, state, current, *_ = _fixture()
    events = []
    stored = object()

    class FilledMasks:
        def to(self, device, *, non_blocking):
            assert device is state["storage_device"] and non_blocking is True
            events.append("store")
            return stored

    filled = FilledMasks()

    def fill_holes(logits, area):
        assert logits is current["pred_masks"] and area == 8
        events.append("fill")
        return filled

    def get_positions(actual_state, output):
        assert actual_state is state and output is current
        events.append("positions")
        return current["maskmem_pos_enc"]

    monkeypatch.setitem(sys.modules, "sam2.utils.misc", SimpleNamespace(fill_holes_in_mask_scores=fill_holes))
    predictor.fill_hole_area = 8
    predictor._get_maskmem_pos_enc = get_positions

    actual, logits = _run_single_frame_inference(predictor, state, {}, 2, 2, False, None, None, False, True)

    assert events == ["fill", "store", "positions"]
    assert logits is filled
    assert actual["pred_masks"] is stored


@pytest.mark.parametrize("operation", ["frame", "memory"])
def test_compact_outputs_match_upstream_except_memory_precision(operation) -> None:
    """Call both implementations independently, using the installed upstream API."""
    upstream = pytest.importorskip("sam2.sam2_video_predictor").SAM2VideoPredictor
    predictor, state, *_ = _fixture()
    reference, reference_state, *_ = _fixture()
    if operation == "frame":
        arguments = ({}, 2, 2, False, None, None, False, True)
        actual, actual_logits = _run_single_frame_inference(predictor, state, *arguments)
        expected, expected_logits = upstream._run_single_frame_inference(reference, reference_state, *arguments)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        assert actual.keys() == expected.keys()
        for key in actual.keys() - {"maskmem_features"}:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        actual_memory, expected_memory = actual["maskmem_features"], expected["maskmem_features"]
    else:
        arguments = (2, 2, torch.ones(2, 1, 8, 8), torch.ones(2, 1), True)
        actual_memory, actual_positions = _run_memory_encoder(predictor, state, *arguments)
        expected_memory, expected_positions = upstream._run_memory_encoder(reference, reference_state, *arguments)
        torch.testing.assert_close(actual_positions, expected_positions, rtol=0, atol=0)

    assert actual_memory.dtype == torch.float16
    assert expected_memory.dtype == torch.bfloat16
    assert torch.equal(actual_memory.to(torch.bfloat16), expected_memory)
