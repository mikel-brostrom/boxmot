"""Keep EdgeTAM's compact temporal memory in FP16 on MPS."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Adapted from EdgeTAM's sam2/sam2_video_predictor.py under EDGETAM_LICENSE.
# BoxMOT modification: retain FP16 memory instead of converting it to BF16.

from __future__ import annotations

from typing import Any

import torch


def _run_single_frame_inference(
    predictor: Any,
    inference_state: dict[str, Any],
    output_dict: dict[str, Any],
    frame_idx: int,
    batch_size: int,
    is_init_cond_frame: bool,
    point_inputs: dict[str, torch.Tensor] | None,
    mask_inputs: torch.Tensor | None,
    reverse: bool,
    run_mem_encoder: bool,
    prev_sam_mask_logits: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Preserve upstream inference and output ownership while storing FP16 memory.

    This method is bound only to an FP16 MPS predictor. In particular, prompts
    and propagation preflight use the same memory precision as normal tracking.
    Model inference, mask postprocessing and positional-cache handling remain
    delegated to the predictor's original implementations.
    """
    _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = predictor._get_image_feature(
        inference_state, frame_idx, batch_size
    )
    assert point_inputs is None or mask_inputs is None
    current_out = predictor.track_step(
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        current_vision_feats=current_vision_feats,
        current_vision_pos_embeds=current_vision_pos_embeds,
        feat_sizes=feat_sizes,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        output_dict=output_dict,
        num_frames=inference_state["num_frames"],
        track_in_reverse=reverse,
        run_mem_encoder=run_mem_encoder,
        prev_sam_mask_logits=prev_sam_mask_logits,
    )

    storage_device = inference_state["storage_device"]
    maskmem_features = current_out["maskmem_features"]
    if maskmem_features is not None:
        maskmem_features = maskmem_features.to(dtype=torch.float16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
    pred_masks_gpu = current_out["pred_masks"]
    if predictor.fill_hole_area > 0:
        from sam2.utils.misc import fill_holes_in_mask_scores

        pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, predictor.fill_hole_area)
    pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
    maskmem_pos_enc = predictor._get_maskmem_pos_enc(inference_state, current_out)
    compact_current_out = {
        "maskmem_features": maskmem_features,
        "maskmem_pos_enc": maskmem_pos_enc,
        "pred_masks": pred_masks,
        "obj_ptr": current_out["obj_ptr"],
        "object_score_logits": current_out["object_score_logits"],
    }
    return compact_current_out, pred_masks_gpu


def _run_memory_encoder(
    predictor: Any,
    inference_state: dict[str, Any],
    frame_idx: int,
    batch_size: int,
    high_res_masks: torch.Tensor,
    object_score_logits: torch.Tensor,
    is_mask_from_pts: bool,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Encode corrected or prompted masks and cache their memories in FP16."""
    _, _, current_vision_feats, _, feat_sizes = predictor._get_image_feature(
        inference_state, frame_idx, batch_size
    )
    maskmem_features, maskmem_pos_enc = predictor._encode_new_memory(
        current_vision_feats=current_vision_feats,
        feat_sizes=feat_sizes,
        pred_masks_high_res=high_res_masks,
        object_score_logits=object_score_logits,
        is_mask_from_pts=is_mask_from_pts,
    )
    maskmem_features = maskmem_features.to(dtype=torch.float16)
    maskmem_features = maskmem_features.to(inference_state["storage_device"], non_blocking=True)
    maskmem_pos_enc = predictor._get_maskmem_pos_enc(
        inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
    )
    return maskmem_features, maskmem_pos_enc
