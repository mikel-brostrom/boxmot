"""Typed configuration objects for ReID training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

from boxmot.reid.backbones.anatomical_registry import (
    DEFAULT_ANATOMICAL_TARGET_TYPE,
    V8_ANATOMICAL_TARGET_TYPE,
)

TRAIN_HPARAM_SECTIONS = frozenset(
    {
        "run",
        "data",
        "model",
        "optimization",
        "losses",
        "augmentation",
        "evaluation",
        "system",
        "derived",
        "resume",
    }
)

TRAIN_HPARAM_TO_ARG = {
    "model_name": "model",
    "dataset": "dataset",
    "data_dir": "data_dir",
    "data_specs": "data_specs",
    "source_balance": "source_balance",
    "pk_steps_per_epoch": "pk_steps_per_epoch",
    "camera_aware_sampler": "camera_aware_sampler",
    "img_size": "imgsz",
    "preprocess": "preprocess",
    "loss_type": "loss",
    "pretrained": "pretrained",
    "pretrained_weights": "pretrained_weights",
    "epochs": "epochs",
    "batch_size": "batch_size",
    "lr": "lr",
    "weight_decay": "weight_decay",
    "eta_min": "eta_min",
    "warmup_epochs": "warmup_epochs",
    "label_smooth": "label_smooth",
    "classifier_loss": "classifier_loss",
    "margin": "margin",
    "triplet_soft_margin": "triplet_soft_margin",
    "arcface_scale": "arcface_scale",
    "arcface_margin": "arcface_margin",
    "cosface_scale": "cosface_scale",
    "cosface_margin": "cosface_margin",
    "center_loss_weight": "center_loss_weight",
    "id_loss_weight": "id_loss_weight",
    "metric_loss_weight": "metric_loss_weight",
    "adasp_loss_weight": "adasp_loss_weight",
    "adasp_temperature": "adasp_temperature",
    "adasp_scale": "adasp_scale",
    "coarse_branch_ce_weight": "coarse_branch_ce_weight",
    "fine_branch_ce_weight": "fine_branch_ce_weight",
    "part_relation_weight": "part_relation_weight",
    "part_to_global_weight": "part_to_global_weight",
    "part_relation_teacher_momentum": "part_relation_teacher_momentum",
    "part_relation_temperature": "part_relation_temperature",
    "compact_metric_loss_weight": "compact_metric_loss_weight",
    "compact_cosine_distill_weight": "compact_cosine_distill_weight",
    "compact_pairwise_distill_weight": "compact_pairwise_distill_weight",
    "csmm_loss_weight": "csmm_loss_weight",
    "csmm_margin": "csmm_margin",
    "csmm_temperature": "csmm_temperature",
    "csmm_topk_negatives": "csmm_topk_negatives",
    "csmm_start_epoch": "csmm_start_epoch",
    "csmm_ramp_end_epoch": "csmm_ramp_end_epoch",
    "treeboost_loss_weight": "treeboost_loss_weight",
    "treeboost_coarse_coefficient": "treeboost_coarse_coefficient",
    "treeboost_fine_coefficient": "treeboost_fine_coefficient",
    "treeboost_node_coefficient": "treeboost_node_coefficient",
    "treeboost_regression_coefficient": "treeboost_regression_coefficient",
    "treeboost_difficulty_floor": "treeboost_difficulty_floor",
    "treeboost_regression_tolerance": "treeboost_regression_tolerance",
    "treeboost_temperature": "treeboost_temperature",
    "treeboost_start_epoch": "treeboost_start_epoch",
    "treeboost_ramp_end_epoch": "treeboost_ramp_end_epoch",
    "global_ap_loss_weight": "global_ap_loss_weight",
    "global_ap_temperature": "global_ap_temperature",
    "global_ap_topk": "global_ap_topk",
    "global_ap_memory_size": "global_ap_memory_size",
    "global_ap_momentum": "global_ap_momentum",
    "global_ap_max_age": "global_ap_max_age",
    "global_ap_start_epoch": "global_ap_start_epoch",
    "global_ap_ramp_end_epoch": "global_ap_ramp_end_epoch",
    "global_ap_decay_start_epoch": "global_ap_decay_start_epoch",
    "global_ap_decay_end_epoch": "global_ap_decay_end_epoch",
    "hpgrd_cache_dir": "hpgrd_cache_dir",
    "hpgrd_global_weight": "hpgrd_global_weight",
    "hpgrd_part_weight": "hpgrd_part_weight",
    "hpgrd_background_weight": "hpgrd_background_weight",
    "hpgrd_part_drop_weight": "hpgrd_part_drop_weight",
    "hpgrd_part_drop_probability": "hpgrd_part_drop_probability",
    "hpgrd_gradient_fraction": "hpgrd_gradient_fraction",
    "hpgrd_min_confidence": "hpgrd_min_confidence",
    "early_id_loss_weight": "early_id_loss_weight",
    "early_id_loss_epochs": "early_id_loss_epochs",
    "center_loss_ramp_start_epoch": "center_loss_ramp_start_epoch",
    "center_loss_ramp_end_epoch": "center_loss_ramp_end_epoch",
    "aux_ce_weight": "aux_ce_weight",
    "aux_ce_drop_epoch": "aux_ce_drop_epoch",
    "branch_loss_agg": "branch_loss_agg",
    "scale_balanced_branches": "scale_balanced_branches",
    "multilevel_suppression": "multilevel_suppression",
    "multilevel_suppression_ratio": "multilevel_suppression_ratio",
    "multilevel_suppression_loss_weight": "multilevel_suppression_loss_weight",
    "multilevel_suppression_start_epoch": "multilevel_suppression_start_epoch",
    "multilevel_suppression_ramp_end_epoch": "multilevel_suppression_ramp_end_epoch",
    "multilevel_suppression_decay_start_epoch": "multilevel_suppression_decay_start_epoch",
    "multilevel_suppression_decay_end_epoch": "multilevel_suppression_decay_end_epoch",
    "hierarchical_branch_attention": "hierarchical_branch_attention",
    "branch_attention_token_dim": "branch_attention_token_dim",
    "branch_attention_num_heads": "branch_attention_num_heads",
    "branch_attention_num_layers": "branch_attention_num_layers",
    "branch_attention_mlp_ratio": "branch_attention_mlp_ratio",
    "branch_attention_dropout": "branch_attention_dropout",
    "branch_set_attention": "branch_set_attention",
    "branch_set_attention_token_dim": "branch_set_attention_token_dim",
    "branch_set_attention_num_heads": "branch_set_attention_num_heads",
    "branch_set_attention_num_layers": "branch_set_attention_num_layers",
    "branch_set_attention_mlp_ratio": "branch_set_attention_mlp_ratio",
    "branch_set_attention_dropout": "branch_set_attention_dropout",
    "multiscale_query_decoder": "multiscale_query_decoder",
    "query_decoder_dim": "query_decoder_dim",
    "query_decoder_num_heads": "query_decoder_num_heads",
    "query_decoder_num_layers": "query_decoder_num_layers",
    "query_decoder_mlp_ratio": "query_decoder_mlp_ratio",
    "query_decoder_dropout": "query_decoder_dropout",
    "hierarchical_late_interaction": "hierarchical_late_interaction",
    "late_interaction_dim": "late_interaction_dim",
    "late_interaction_num_heads": "late_interaction_num_heads",
    "late_interaction_num_layers": "late_interaction_num_layers",
    "late_interaction_sinkhorn_iters": "late_interaction_sinkhorn_iters",
    "late_interaction_null_tokens": "late_interaction_null_tokens",
    "late_interaction_negative_identities": "late_interaction_negative_identities",
    "late_interaction_rerank_topk": "late_interaction_rerank_topk",
    "late_interaction_base_score_init": "late_interaction_base_score_init",
    "late_interaction_loss_weight": "late_interaction_loss_weight",
    "late_interaction_distill_weight": "late_interaction_distill_weight",
    "late_interaction_temperature": "late_interaction_temperature",
    "late_interaction_start_epoch": "late_interaction_start_epoch",
    "late_interaction_ramp_end_epoch": "late_interaction_ramp_end_epoch",
    "mcpt_mode": "mcpt_mode",
    "mcpt_hidden_dim": "mcpt_hidden_dim",
    "mcpt_max_displacement": "mcpt_max_displacement",
    "mcpt_smoothness_weight": "mcpt_smoothness_weight",
    "mcpt_identity_weight": "mcpt_identity_weight",
    "mcpt_identity_decay_epoch": "mcpt_identity_decay_epoch",
    "mcpt_lr_multiplier": "mcpt_lr_multiplier",
    "mcpt_start_epoch": "mcpt_start_epoch",
    "mcpt_ramp_end_epoch": "mcpt_ramp_end_epoch",
    "mcpt_disabled_eval": "mcpt_disabled_eval",
    "jpm": "jpm",
    "jpm_num_groups": "jpm_num_groups",
    "jpm_shift": "jpm_shift",
    "jpm_token_dim": "jpm_token_dim",
    "jpm_num_heads": "jpm_num_heads",
    "jpm_mlp_ratio": "jpm_mlp_ratio",
    "jpm_dropout": "jpm_dropout",
    "jpm_id_loss_weight": "jpm_id_loss_weight",
    "jpm_metric_loss_weight": "jpm_metric_loss_weight",
    "metric_feature": "metric_feature",
    "inference_feature": "inference_feature",
    "feature_fusion": "feature_fusion",
    "pyramid_resize_mode": "pyramid_resize_mode",
    "spatial_conv_mode": "spatial_conv_mode",
    "post_fusion_mixer": "post_fusion_mixer",
    "post_fusion_mixer_reduction": "post_fusion_mixer_reduction",
    "post_fusion_mixer_kernel": "post_fusion_mixer_kernel",
    "post_fusion_mixer_gamma_init": "post_fusion_mixer_gamma_init",
    "feat_dim": "feat_dim",
    "neck_dim": "neck_dim",
    "drop_path_rate": "drop_path_rate",
    "timm_model_name": "timm_model_name",
    "timm_head_mode": "timm_head_mode",
    "mobilenetv4_last_stride": "mobilenetv4_last_stride",
    "mobilenetv4_neck_mode": "mobilenetv4_neck_mode",
    "attention_window_layout": "attention_window_layout",
    "attention_bias": "attention_bias",
    "interpolate_pretrained_attention_bias": "interpolate_pretrained_attention_bias",
    "attention_mask": "attention_mask",
    "attention_shift": "attention_shift",
    "stage3_global": "stage3_global",
    "stage3_downsample": "stage3_downsample",
    "stage2_width_merge_after": "stage2_width_merge_after",
    "stage2_mlp_ratio": "stage2_mlp_ratio",
    "stage3_mlp_ratio": "stage3_mlp_ratio",
    "stage2_depth": "stage2_depth",
    "stage3_depth": "stage3_depth",
    "width_first_hierarchy": "width_first_hierarchy",
    "identity_registers": "identity_registers",
    "identity_register_count": "identity_register_count",
    "identity_register_dim": "identity_register_dim",
    "identity_register_num_heads": "identity_register_num_heads",
    "identity_register_dropout": "identity_register_dropout",
    "identity_register_gate_init": "identity_register_gate_init",
    "identity_register_diversity_weight": (
        "identity_register_diversity_weight"
    ),
    "identity_register_diversity_margin": (
        "identity_register_diversity_margin"
    ),
    "native_branch_widths": "native_branch_widths",
    "fine_map_dim": "fine_map_dim",
    "compact_deployment_head": "compact_deployment_head",
    "reid_adapter_stages": "reid_adapter_stages",
    "reid_adapter_reduction": "reid_adapter_reduction",
    "reid_adapter_suppression_tau": "reid_adapter_suppression_tau",
    "head_pool": "head_pool",
    "head_parts": "head_parts",
    "head_type": "head_type",
    "multiscale_channel_alpha": "multiscale_channel_alpha",
    "body_slot_mode": "body_slot_mode",
    "body_slot_alpha": "body_slot_alpha",
    "body_slot_visibility_floor": "body_slot_visibility_floor",
    "part_pooling": "part_pooling",
    "num_part_tokens": "num_part_tokens",
    "evidence_num_roles": "evidence_num_roles",
    "decouple_patterns": "decouple_patterns",
    "pattern_adapter_dim": "pattern_adapter_dim",
    "stripe_visibility": "stripe_visibility",
    "drop_global_aux": "drop_global_aux",
    "drop_global_aux_ratio": "drop_global_aux_ratio",
    "branch_aware_metric": "branch_aware_metric",
    "branch_metric_part_weight": "branch_metric_part_weight",
    "evidence_alignment_loss_weight": "evidence_alignment_loss_weight",
    "evidence_alignment_margin": "evidence_alignment_margin",
    "evidence_sinkhorn_iters": "evidence_sinkhorn_iters",
    "evidence_sinkhorn_temperature": "evidence_sinkhorn_temperature",
    "evidence_rerank_topk": "evidence_rerank_topk",
    "evidence_null_loss_weight": "evidence_null_loss_weight",
    "evidence_diversity_loss_weight": "evidence_diversity_loss_weight",
    "anatomical_auxiliary": "anatomical_auxiliary",
    "anatomical_metadata_dir": "anatomical_metadata_dir",
    "anatomical_person_mask_dir": "anatomical_person_mask_dir",
    "anatomical_min_keypoint_confidence": "anatomical_min_keypoint_confidence",
    "anatomical_token_dim": "anatomical_token_dim",
    "anatomical_distill_weight": "anatomical_distill_weight",
    "anatomical_attention_weight": "anatomical_attention_weight",
    "anatomical_foreground_weight": "anatomical_foreground_weight",
    "anatomical_semantic_part_weight": (
        "anatomical_semantic_part_weight"
    ),
    "anatomical_visibility_weight": "anatomical_visibility_weight",
    "anatomical_contrastive_weight": "anatomical_contrastive_weight",
    "anatomical_descriptor_distill_weight": ("anatomical_descriptor_distill_weight"),
    "anatomical_branch_distill_weight": "anatomical_branch_distill_weight",
    "anatomical_branch_global_coefficient": (
        "anatomical_branch_global_coefficient"
    ),
    "anatomical_branch_coarse_coefficient": (
        "anatomical_branch_coarse_coefficient"
    ),
    "anatomical_branch_fine_coefficient": (
        "anatomical_branch_fine_coefficient"
    ),
    "anatomical_pose_teacher_weight": "anatomical_pose_teacher_weight",
    "anatomical_query_distill_weight": "anatomical_query_distill_weight",
    "anatomical_query_relational_distill_weight": (
        "anatomical_query_relational_distill_weight"
    ),
    "anatomical_query_diversity_weight": "anatomical_query_diversity_weight",
    "anatomical_query_diversity_margin": "anatomical_query_diversity_margin",
    "anatomical_part_triplet_weight": "anatomical_part_triplet_weight",
    "anatomical_target_type": "anatomical_target_type",
    "anatomical_teacher_momentum": "anatomical_teacher_momentum",
    "anatomical_multiscale": "anatomical_multiscale",
    "anatomical_accessory_query": "anatomical_accessory_query",
    "anatomical_deployment": "anatomical_deployment",
    "anatomical_deployment_dim": "anatomical_deployment_dim",
    "anatomical_deployment_alpha": "anatomical_deployment_alpha",
    "anatomical_deployment_id_weight": (
        "anatomical_deployment_id_weight"
    ),
    "anatomical_deployment_metric_weight": (
        "anatomical_deployment_metric_weight"
    ),
    "anatomical_local_scale_weight": "anatomical_local_scale_weight",
    "anatomical_fine_scale_weight": "anatomical_fine_scale_weight",
    "anatomical_cross_scale_weight": "anatomical_cross_scale_weight",
    "anatomical_pose_only_reliability": ("anatomical_pose_only_reliability"),
    "anatomical_min_effective_coverage": (
        "anatomical_min_effective_coverage"
    ),
    "anatomical_student_start_epoch": "anatomical_student_start_epoch",
    "anatomical_student_ramp_end_epoch": ("anatomical_student_ramp_end_epoch"),
    "anatomical_query_start_epoch": "anatomical_query_start_epoch",
    "anatomical_query_ramp_end_epoch": "anatomical_query_ramp_end_epoch",
    "anatomical_fine_start_epoch": "anatomical_fine_start_epoch",
    "anatomical_fine_ramp_end_epoch": "anatomical_fine_ramp_end_epoch",
    "anatomical_decay_start_epoch": "anatomical_decay_start_epoch",
    "anatomical_decay_end_epoch": "anatomical_decay_end_epoch",
    "anatomical_temperature": "anatomical_temperature",
    "head_warmup_epochs": "head_warmup_epochs",
    "head_warmup_lr_mult": "head_warmup_lr_mult",
    "vit_lr_profile": "vit_lr_profile",
    "layer_decay": "layer_decay",
    "backbone_lr_mult": "backbone_lr_mult",
    "backbone_freeze_epochs": "backbone_freeze_epochs",
    "gradual_unfreeze": "gradual_unfreeze",
    "gradual_unfreeze_head_epochs": "gradual_unfreeze_head_epochs",
    "gradual_unfreeze_stage_epochs": "gradual_unfreeze_stage_epochs",
    "gradual_unfreeze_backbone_lr_mult": "gradual_unfreeze_backbone_lr_mult",
    "gradual_unfreeze_backbone_lr_epochs": "gradual_unfreeze_backbone_lr_epochs",
    "p": "p_ids",
    "k": "k_instances",
    "seed": "seed",
    "deterministic": "deterministic",
    "device": "device",
    "num_workers": "num_workers",
    "ema_decay": "ema_decay",
    "gaussian_blur": "gaussian_blur",
    "random_grayscale": "random_grayscale",
    "color_jitter": "color_jitter",
    "random_erasing": "random_erasing",
    "random_patch": "random_patch",
    "random_crop_scale": "random_crop_scale",
    "color_augmentation": "color_augmentation",
    "background_mosaic": "background_mosaic",
    "background_mosaic_mask_dir": "background_mosaic_mask_dir",
    "background_mosaic_probability": "background_mosaic_probability",
    "background_mosaic_start_epoch": "background_mosaic_start_epoch",
    "background_mosaic_ramp_end_epoch": "background_mosaic_ramp_end_epoch",
    "background_mosaic_min_foreground_ratio": "background_mosaic_min_foreground_ratio",
    "background_mosaic_max_foreground_ratio": "background_mosaic_max_foreground_ratio",
    "background_mosaic_feather": "background_mosaic_feather",
    "background_mosaic_dilation": "background_mosaic_dilation",
    "background_mosaic_occluder_probability": "background_mosaic_occluder_probability",
    "background_mosaic_occluder_min_area": "background_mosaic_occluder_min_area",
    "background_mosaic_occluder_max_area": "background_mosaic_occluder_max_area",
    "same_id_part_mosaic": "same_id_part_mosaic",
    "same_id_part_mosaic_probability": "same_id_part_mosaic_probability",
    "same_id_part_mosaic_max_regions": "same_id_part_mosaic_max_regions",
    "same_id_part_mosaic_min_area": "same_id_part_mosaic_min_area",
    "same_id_part_mosaic_max_area": "same_id_part_mosaic_max_area",
    "same_id_part_mosaic_boundary_jitter": "same_id_part_mosaic_boundary_jitter",
    "same_id_part_mosaic_cross_camera_rate": "same_id_part_mosaic_cross_camera_rate",
    "same_id_part_mosaic_min_unaltered": "same_id_part_mosaic_min_unaltered",
    "pav_mosaic": "pav_mosaic",
    "pav_metadata_dir": "pav_metadata_dir",
    "pav_mosaic_probability": "pav_mosaic_probability",
    "pav_mosaic_max_parts": "pav_mosaic_max_parts",
    "pav_mosaic_max_foreground_replacement": "pav_mosaic_max_foreground_replacement",
    "pav_mosaic_cross_camera_rate": "pav_mosaic_cross_camera_rate",
    "pav_mosaic_different_pose_rate": "pav_mosaic_different_pose_rate",
    "pav_mosaic_min_keypoint_confidence": "pav_mosaic_min_keypoint_confidence",
    "pav_mosaic_min_unaltered": "pav_mosaic_min_unaltered",
    "pav_mosaic_warmup_epochs": "pav_mosaic_warmup_epochs",
    "pav_mosaic_decay_start_epoch": "pav_mosaic_decay_start_epoch",
    "pav_mosaic_final_probability_scale": "pav_mosaic_final_probability_scale",
    "pav_consistency_weight": "pav_consistency_weight",
    "clean_student_consistency_weight": "clean_student_consistency_weight",
    "flip_tta": "flip_tta",
    "eval_interval": "eval_interval",
}


def _nested_get(data: dict[str, Any], *keys: str) -> Any | None:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def flatten_train_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    """Normalize saved nested hparams into flat trainer hparam keys."""
    if not any(key in hparams for key in TRAIN_HPARAM_SECTIONS):
        return dict(hparams)

    flat = {key: value for key, value in hparams.items() if key not in TRAIN_HPARAM_SECTIONS}
    mappings = {
        "model_name": ("run", "model_name"),
        "seed": ("run", "seed"),
        "deterministic": ("run", "deterministic"),
        "pretrained": ("run", "pretrained"),
        "pretrained_weights": ("run", "pretrained_weights"),
        "dataset": ("data", "dataset"),
        "data_dir": ("data", "data_dir"),
        "data_specs": ("data", "data_specs"),
        "img_size": ("data", "img_size"),
        "preprocess": ("data", "preprocess"),
        "num_classes": ("data", "num_classes"),
        "batch_size": ("data", "batch_size"),
        "p": ("data", "sampler", "p"),
        "k": ("data", "sampler", "k"),
        "source_balance": ("data", "sampler", "source_balance"),
        "pk_steps_per_epoch": ("data", "sampler", "steps_per_epoch"),
        "camera_aware_sampler": ("data", "sampler", "camera_aware"),
        "num_workers": ("data", "num_workers"),
        "is_vit": ("model", "is_vit"),
        "feature_fusion": ("model", "feature_fusion"),
        "pyramid_resize_mode": ("model", "pyramid_resize_mode"),
        "spatial_conv_mode": ("model", "spatial_conv_mode"),
        "timm_model_name": ("model", "timm_model_name"),
        "timm_head_mode": ("model", "timm_head_mode"),
        "mobilenetv4_last_stride": ("model", "mobilenetv4_last_stride"),
        "mobilenetv4_neck_mode": ("model", "mobilenetv4_neck_mode"),
        "post_fusion_mixer": ("model", "post_fusion_mixer", "mode"),
        "post_fusion_mixer_reduction": ("model", "post_fusion_mixer", "reduction"),
        "post_fusion_mixer_kernel": ("model", "post_fusion_mixer", "kernel"),
        "post_fusion_mixer_gamma_init": ("model", "post_fusion_mixer", "gamma_init"),
        "feat_dim": ("model", "feat_dim"),
        "neck_dim": ("model", "neck_dim"),
        "attention_window_layout": ("model", "attention", "window_layout"),
        "attention_bias": ("model", "attention", "bias"),
        "interpolate_pretrained_attention_bias": ("model", "attention", "interpolate_pretrained_bias"),
        "attention_mask": ("model", "attention", "mask"),
        "attention_shift": ("model", "attention", "shift"),
        "stage3_global": ("model", "attention", "stage3_global"),
        "stage3_downsample": ("model", "speed", "stage3_downsample"),
        "stage2_width_merge_after": ("model", "speed", "stage2_width_merge_after"),
        "stage2_mlp_ratio": ("model", "speed", "stage2_mlp_ratio"),
        "stage3_mlp_ratio": ("model", "speed", "stage3_mlp_ratio"),
        "stage2_depth": ("model", "speed", "stage2_depth"),
        "stage3_depth": ("model", "speed", "stage3_depth"),
        "width_first_hierarchy": (
            "model",
            "hierarchy",
            "width_first",
        ),
        "identity_registers": (
            "model",
            "identity_registers",
            "enabled",
        ),
        "identity_register_count": (
            "model",
            "identity_registers",
            "count",
        ),
        "identity_register_dim": (
            "model",
            "identity_registers",
            "dim",
        ),
        "identity_register_num_heads": (
            "model",
            "identity_registers",
            "num_heads",
        ),
        "identity_register_dropout": (
            "model",
            "identity_registers",
            "dropout",
        ),
        "identity_register_gate_init": (
            "model",
            "identity_registers",
            "gate_init",
        ),
        "identity_register_diversity_weight": (
            "model",
            "identity_registers",
            "diversity_weight",
        ),
        "identity_register_diversity_margin": (
            "model",
            "identity_registers",
            "diversity_margin",
        ),
        "native_branch_widths": ("model", "speed", "native_branch_widths"),
        "fine_map_dim": ("model", "speed", "fine_map_dim"),
        "compact_deployment_head": ("model", "deployment", "compact_head"),
        "reid_adapter_stages": ("model", "reid_adapters", "stages"),
        "reid_adapter_reduction": ("model", "reid_adapters", "reduction"),
        "reid_adapter_suppression_tau": ("model", "reid_adapters", "suppression_tau"),
        "head_pool": ("model", "head", "pool"),
        "head_parts": ("model", "head", "parts"),
        "head_type": ("model", "head", "head_type"),
        "multiscale_channel_alpha": (
            "model",
            "head",
            "multiscale_channel_alpha",
        ),
        "body_slot_mode": ("model", "head", "body_slots", "mode"),
        "body_slot_alpha": ("model", "head", "body_slots", "alpha"),
        "body_slot_visibility_floor": (
            "model",
            "head",
            "body_slots",
            "visibility_floor",
        ),
        "part_pooling": ("model", "head", "part_pooling"),
        "num_part_tokens": ("model", "head", "num_part_tokens"),
        "evidence_num_roles": ("model", "head", "evidence_num_roles"),
        "decouple_patterns": ("model", "head", "decouple_patterns"),
        "pattern_adapter_dim": ("model", "head", "pattern_adapter_dim"),
        "stripe_visibility": ("model", "head", "stripe_visibility"),
        "drop_global_aux": ("model", "head", "drop_global_aux"),
        "drop_global_aux_ratio": ("model", "head", "drop_global_aux_ratio"),
        "multilevel_suppression": (
            "model",
            "head",
            "multilevel_suppression",
            "enabled",
        ),
        "multilevel_suppression_ratio": (
            "model",
            "head",
            "multilevel_suppression",
            "ratio",
        ),
        "head_warmup_epochs": ("model", "head", "warmup_epochs"),
        "head_warmup_lr_mult": ("model", "head", "warmup_lr_mult"),
        "metric_feature": ("model", "feature_selection", "metric_feature"),
        "inference_feature": ("model", "feature_selection", "inference_feature"),
        "branch_aware_metric": ("model", "branch", "aware_metric"),
        "branch_metric_part_weight": ("model", "branch", "metric_part_weight"),
        "branch_loss_agg": ("model", "branch", "loss_agg"),
        "scale_balanced_branches": ("model", "branch", "scale_balanced"),
        "hierarchical_branch_attention": ("model", "head", "hierarchical_attention", "enabled"),
        "branch_attention_token_dim": ("model", "head", "hierarchical_attention", "token_dim"),
        "branch_attention_num_heads": ("model", "head", "hierarchical_attention", "num_heads"),
        "branch_attention_num_layers": ("model", "head", "hierarchical_attention", "num_layers"),
        "branch_attention_mlp_ratio": ("model", "head", "hierarchical_attention", "mlp_ratio"),
        "branch_attention_dropout": ("model", "head", "hierarchical_attention", "dropout"),
        "branch_set_attention": ("model", "head", "branch_set_attention", "enabled"),
        "branch_set_attention_token_dim": ("model", "head", "branch_set_attention", "token_dim"),
        "branch_set_attention_num_heads": ("model", "head", "branch_set_attention", "num_heads"),
        "branch_set_attention_num_layers": ("model", "head", "branch_set_attention", "num_layers"),
        "branch_set_attention_mlp_ratio": ("model", "head", "branch_set_attention", "mlp_ratio"),
        "branch_set_attention_dropout": ("model", "head", "branch_set_attention", "dropout"),
        "multiscale_query_decoder": ("model", "head", "multiscale_query_decoder", "enabled"),
        "query_decoder_dim": ("model", "head", "multiscale_query_decoder", "token_dim"),
        "query_decoder_num_heads": ("model", "head", "multiscale_query_decoder", "num_heads"),
        "query_decoder_num_layers": ("model", "head", "multiscale_query_decoder", "num_layers"),
        "query_decoder_mlp_ratio": ("model", "head", "multiscale_query_decoder", "mlp_ratio"),
        "query_decoder_dropout": ("model", "head", "multiscale_query_decoder", "dropout"),
        "hierarchical_late_interaction": ("model", "head", "hierarchical_late_interaction", "enabled"),
        "late_interaction_dim": ("model", "head", "hierarchical_late_interaction", "token_dim"),
        "late_interaction_num_heads": ("model", "head", "hierarchical_late_interaction", "num_heads"),
        "late_interaction_num_layers": ("model", "head", "hierarchical_late_interaction", "num_layers"),
        "late_interaction_sinkhorn_iters": ("model", "head", "hierarchical_late_interaction", "sinkhorn_iters"),
        "late_interaction_null_tokens": ("model", "head", "hierarchical_late_interaction", "null_tokens"),
        "late_interaction_base_score_init": ("model", "head", "hierarchical_late_interaction", "base_score_init"),
        "late_interaction_rerank_topk": ("model", "head", "hierarchical_late_interaction", "rerank_topk"),
        "mcpt_mode": ("model", "head", "mcpt", "mode"),
        "mcpt_hidden_dim": ("model", "head", "mcpt", "hidden_dim"),
        "mcpt_max_displacement": ("model", "head", "mcpt", "max_displacement"),
        "mcpt_start_epoch": ("model", "head", "mcpt", "start_epoch"),
        "mcpt_ramp_end_epoch": ("model", "head", "mcpt", "ramp_end_epoch"),
        "jpm": ("model", "head", "jpm", "enabled"),
        "jpm_num_groups": ("model", "head", "jpm", "num_groups"),
        "jpm_shift": ("model", "head", "jpm", "shift"),
        "jpm_token_dim": ("model", "head", "jpm", "token_dim"),
        "jpm_num_heads": ("model", "head", "jpm", "num_heads"),
        "jpm_mlp_ratio": ("model", "head", "jpm", "mlp_ratio"),
        "jpm_dropout": ("model", "head", "jpm", "dropout"),
        "evidence_alignment_loss_weight": ("model", "evidence", "alignment_loss_weight"),
        "evidence_alignment_margin": ("model", "evidence", "alignment_margin"),
        "evidence_sinkhorn_iters": ("model", "evidence", "sinkhorn_iters"),
        "evidence_sinkhorn_temperature": ("model", "evidence", "sinkhorn_temperature"),
        "evidence_rerank_topk": ("model", "evidence", "rerank_topk"),
        "evidence_null_loss_weight": ("model", "evidence", "null_loss_weight"),
        "evidence_diversity_loss_weight": ("model", "evidence", "diversity_loss_weight"),
        "anatomical_auxiliary": (
            "augmentation",
            "anatomical_supervision",
            "enabled",
        ),
        "anatomical_metadata_dir": (
            "augmentation",
            "anatomical_supervision",
            "metadata_dir",
        ),
        "anatomical_person_mask_dir": (
            "augmentation",
            "anatomical_supervision",
            "person_mask_dir",
        ),
        "anatomical_min_keypoint_confidence": (
            "augmentation",
            "anatomical_supervision",
            "min_keypoint_confidence",
        ),
        "anatomical_token_dim": (
            "model",
            "head",
            "anatomical_auxiliary",
            "token_dim",
        ),
        "anatomical_distill_weight": (
            "augmentation",
            "anatomical_supervision",
            "distill_weight",
        ),
        "anatomical_attention_weight": (
            "augmentation",
            "anatomical_supervision",
            "attention_weight",
        ),
        "anatomical_foreground_weight": (
            "augmentation",
            "anatomical_supervision",
            "foreground_weight",
        ),
        "anatomical_semantic_part_weight": (
            "augmentation",
            "anatomical_supervision",
            "semantic_part_weight",
        ),
        "anatomical_visibility_weight": (
            "augmentation",
            "anatomical_supervision",
            "visibility_weight",
        ),
        "anatomical_contrastive_weight": (
            "augmentation",
            "anatomical_supervision",
            "contrastive_weight",
        ),
        "anatomical_descriptor_distill_weight": (
            "augmentation",
            "anatomical_supervision",
            "descriptor_distill_weight",
        ),
        "anatomical_branch_distill_weight": (
            "augmentation",
            "anatomical_supervision",
            "branch_distill_weight",
        ),
        "anatomical_branch_global_coefficient": (
            "augmentation",
            "anatomical_supervision",
            "branch_global_coefficient",
        ),
        "anatomical_branch_coarse_coefficient": (
            "augmentation",
            "anatomical_supervision",
            "branch_coarse_coefficient",
        ),
        "anatomical_branch_fine_coefficient": (
            "augmentation",
            "anatomical_supervision",
            "branch_fine_coefficient",
        ),
        "anatomical_pose_teacher_weight": (
            "augmentation",
            "anatomical_supervision",
            "pose_teacher_weight",
        ),
        "anatomical_query_distill_weight": (
            "augmentation",
            "anatomical_supervision",
            "query_distill_weight",
        ),
        "anatomical_query_relational_distill_weight": (
            "augmentation",
            "anatomical_supervision",
            "query_relational_distill_weight",
        ),
        "anatomical_query_diversity_weight": (
            "augmentation",
            "anatomical_supervision",
            "query_diversity_weight",
        ),
        "anatomical_query_diversity_margin": (
            "augmentation",
            "anatomical_supervision",
            "query_diversity_margin",
        ),
        "anatomical_part_triplet_weight": (
            "augmentation",
            "anatomical_supervision",
            "part_triplet_weight",
        ),
        "anatomical_target_type": (
            "augmentation",
            "anatomical_supervision",
            "teacher",
        ),
        "anatomical_teacher_momentum": (
            "augmentation",
            "anatomical_supervision",
            "teacher_momentum",
        ),
        "anatomical_multiscale": (
            "model",
            "head",
            "anatomical_auxiliary",
            "multiscale",
        ),
        "anatomical_accessory_query": (
            "model",
            "head",
            "anatomical_auxiliary",
            "accessory_query",
        ),
        "anatomical_deployment": (
            "model",
            "head",
            "anatomical_auxiliary",
            "deployment",
        ),
        "anatomical_deployment_dim": (
            "model",
            "head",
            "anatomical_auxiliary",
            "deployment_dim",
        ),
        "anatomical_deployment_alpha": (
            "model",
            "head",
            "anatomical_auxiliary",
            "deployment_alpha",
        ),
        "anatomical_deployment_id_weight": (
            "augmentation",
            "anatomical_supervision",
            "deployment_id_weight",
        ),
        "anatomical_deployment_metric_weight": (
            "augmentation",
            "anatomical_supervision",
            "deployment_metric_weight",
        ),
        "anatomical_local_scale_weight": (
            "augmentation",
            "anatomical_supervision",
            "local_scale_weight",
        ),
        "anatomical_fine_scale_weight": (
            "augmentation",
            "anatomical_supervision",
            "fine_scale_weight",
        ),
        "anatomical_cross_scale_weight": (
            "augmentation",
            "anatomical_supervision",
            "cross_scale_weight",
        ),
        "anatomical_pose_only_reliability": (
            "augmentation",
            "anatomical_supervision",
            "pose_only_reliability",
        ),
        "anatomical_min_effective_coverage": (
            "augmentation",
            "anatomical_supervision",
            "min_effective_coverage",
        ),
        "anatomical_student_start_epoch": (
            "augmentation",
            "anatomical_supervision",
            "student_start_epoch",
        ),
        "anatomical_student_ramp_end_epoch": (
            "augmentation",
            "anatomical_supervision",
            "student_ramp_end_epoch",
        ),
        "anatomical_query_start_epoch": (
            "augmentation",
            "anatomical_supervision",
            "query_start_epoch",
        ),
        "anatomical_query_ramp_end_epoch": (
            "augmentation",
            "anatomical_supervision",
            "query_ramp_end_epoch",
        ),
        "anatomical_fine_start_epoch": (
            "augmentation",
            "anatomical_supervision",
            "fine_start_epoch",
        ),
        "anatomical_fine_ramp_end_epoch": (
            "augmentation",
            "anatomical_supervision",
            "fine_ramp_end_epoch",
        ),
        "anatomical_decay_start_epoch": (
            "augmentation",
            "anatomical_supervision",
            "decay_start_epoch",
        ),
        "anatomical_decay_end_epoch": (
            "augmentation",
            "anatomical_supervision",
            "decay_end_epoch",
        ),
        "anatomical_temperature": (
            "augmentation",
            "anatomical_supervision",
            "temperature",
        ),
        "drop_path_rate": ("model", "regularization", "drop_path_rate"),
        "epochs": ("optimization", "epochs"),
        "optimizer": ("optimization", "optimizer"),
        "lr": ("optimization", "lr"),
        "weight_decay": ("optimization", "weight_decay"),
        "grad_clip": ("optimization", "grad_clip"),
        "layer_decay": ("optimization", "layer_decay"),
        "vit_lr_profile": ("optimization", "vit_lr_profile"),
        "backbone_lr_mult": ("optimization", "backbone_lr_mult"),
        "backbone_freeze_epochs": ("optimization", "backbone_freeze_epochs"),
        "gradual_unfreeze": ("optimization", "gradual_unfreeze", "enabled"),
        "gradual_unfreeze_head_epochs": ("optimization", "gradual_unfreeze", "head_epochs"),
        "gradual_unfreeze_stage_epochs": ("optimization", "gradual_unfreeze", "stage_epochs"),
        "gradual_unfreeze_backbone_lr_mult": ("optimization", "gradual_unfreeze", "backbone_lr_mult"),
        "gradual_unfreeze_backbone_lr_epochs": ("optimization", "gradual_unfreeze", "backbone_lr_epochs"),
        "scheduler": ("optimization", "scheduler", "name"),
        "eta_min": ("optimization", "scheduler", "eta_min"),
        "warmup_epochs": ("optimization", "scheduler", "warmup_epochs"),
        "ema_decay": ("optimization", "ema_decay"),
        "loss_type": ("losses", "loss_type"),
        "classifier_loss": ("losses", "classifier_loss"),
        "label_smooth": ("losses", "label_smooth"),
        "margin": ("losses", "triplet", "margin"),
        "triplet_soft_margin": ("losses", "triplet", "soft_margin"),
        "soft_margin_triplet": ("losses", "triplet", "soft_margin"),
        "id_loss_weight": ("losses", "weights", "id_loss_weight"),
        "metric_loss_weight": ("losses", "weights", "metric_loss_weight"),
        "adasp_loss_weight": ("losses", "adaptive_sparse_pairwise", "weight"),
        "adasp_temperature": ("losses", "adaptive_sparse_pairwise", "temperature"),
        "adasp_scale": ("losses", "adaptive_sparse_pairwise", "scale"),
        "coarse_branch_ce_weight": ("losses", "part_relation", "coarse_branch_ce_weight"),
        "fine_branch_ce_weight": ("losses", "part_relation", "fine_branch_ce_weight"),
        "part_relation_weight": ("losses", "part_relation", "weight"),
        "part_to_global_weight": ("losses", "part_relation", "global_distill_weight"),
        "part_relation_teacher_momentum": ("losses", "part_relation", "teacher_momentum"),
        "part_relation_temperature": ("losses", "part_relation", "temperature"),
        "compact_metric_loss_weight": ("losses", "distillation", "metric_weight"),
        "compact_cosine_distill_weight": ("losses", "distillation", "cosine_weight"),
        "compact_pairwise_distill_weight": ("losses", "distillation", "pairwise_weight"),
        "csmm_loss_weight": ("losses", "cross_scale_majority_margin", "weight"),
        "csmm_margin": ("losses", "cross_scale_majority_margin", "margin"),
        "csmm_temperature": ("losses", "cross_scale_majority_margin", "temperature"),
        "csmm_topk_negatives": ("losses", "cross_scale_majority_margin", "topk_negatives"),
        "csmm_start_epoch": ("losses", "cross_scale_majority_margin", "start_epoch"),
        "csmm_ramp_end_epoch": ("losses", "cross_scale_majority_margin", "ramp_end_epoch"),
        "treeboost_loss_weight": ("losses", "treeboost_ap", "weight"),
        "treeboost_coarse_coefficient": ("losses", "treeboost_ap", "coarse_coefficient"),
        "treeboost_fine_coefficient": ("losses", "treeboost_ap", "fine_coefficient"),
        "treeboost_node_coefficient": ("losses", "treeboost_ap", "node_coefficient"),
        "treeboost_regression_coefficient": ("losses", "treeboost_ap", "regression_coefficient"),
        "treeboost_difficulty_floor": ("losses", "treeboost_ap", "difficulty_floor"),
        "treeboost_regression_tolerance": ("losses", "treeboost_ap", "regression_tolerance"),
        "treeboost_temperature": ("losses", "treeboost_ap", "temperature"),
        "treeboost_start_epoch": ("losses", "treeboost_ap", "start_epoch"),
        "treeboost_ramp_end_epoch": ("losses", "treeboost_ap", "ramp_end_epoch"),
        "global_ap_loss_weight": ("losses", "global_ap", "weight"),
        "global_ap_temperature": ("losses", "global_ap", "temperature"),
        "global_ap_topk": ("losses", "global_ap", "topk"),
        "global_ap_memory_size": ("losses", "global_ap", "memory_size"),
        "global_ap_momentum": ("losses", "global_ap", "momentum"),
        "global_ap_max_age": ("losses", "global_ap", "max_age"),
        "global_ap_start_epoch": ("losses", "global_ap", "start_epoch"),
        "global_ap_ramp_end_epoch": ("losses", "global_ap", "ramp_end_epoch"),
        "global_ap_decay_start_epoch": ("losses", "global_ap", "decay_start_epoch"),
        "global_ap_decay_end_epoch": ("losses", "global_ap", "decay_end_epoch"),
        "hpgrd_cache_dir": ("losses", "hpgrd", "cache_dir"),
        "hpgrd_global_weight": ("losses", "hpgrd", "global_weight"),
        "hpgrd_part_weight": ("losses", "hpgrd", "part_weight"),
        "hpgrd_background_weight": ("losses", "hpgrd", "background_weight"),
        "hpgrd_part_drop_weight": ("losses", "hpgrd", "part_drop_weight"),
        "hpgrd_part_drop_probability": ("losses", "hpgrd", "part_drop_probability"),
        "hpgrd_gradient_fraction": ("losses", "hpgrd", "gradient_fraction"),
        "hpgrd_min_confidence": ("losses", "hpgrd", "min_confidence"),
        "late_interaction_loss_weight": ("losses", "hierarchical_late_interaction", "matcher_weight"),
        "late_interaction_distill_weight": ("losses", "hierarchical_late_interaction", "distill_weight"),
        "late_interaction_negative_identities": ("losses", "hierarchical_late_interaction", "negative_identities"),
        "late_interaction_temperature": ("losses", "hierarchical_late_interaction", "temperature"),
        "late_interaction_start_epoch": ("losses", "hierarchical_late_interaction", "start_epoch"),
        "late_interaction_ramp_end_epoch": ("losses", "hierarchical_late_interaction", "ramp_end_epoch"),
        "mcpt_smoothness_weight": ("losses", "mcpt", "smoothness_weight"),
        "jpm_id_loss_weight": ("losses", "jpm", "id_loss_weight"),
        "jpm_metric_loss_weight": ("losses", "jpm", "metric_loss_weight"),
        "multilevel_suppression_loss_weight": (
            "losses",
            "multilevel_suppression",
            "weight",
        ),
        "multilevel_suppression_start_epoch": (
            "losses",
            "multilevel_suppression",
            "start_epoch",
        ),
        "multilevel_suppression_ramp_end_epoch": (
            "losses",
            "multilevel_suppression",
            "ramp_end_epoch",
        ),
        "multilevel_suppression_decay_start_epoch": (
            "losses",
            "multilevel_suppression",
            "decay_start_epoch",
        ),
        "multilevel_suppression_decay_end_epoch": (
            "losses",
            "multilevel_suppression",
            "decay_end_epoch",
        ),
        "mcpt_identity_weight": ("losses", "mcpt", "identity_weight"),
        "mcpt_identity_decay_epoch": ("losses", "mcpt", "identity_decay_epoch"),
        "center_loss_weight": ("losses", "weights", "center_loss_weight"),
        "early_id_loss_weight": ("losses", "schedules", "early_id_loss", "weight"),
        "early_id_loss_epochs": ("losses", "schedules", "early_id_loss", "epochs"),
        "center_loss_ramp_start_epoch": ("losses", "schedules", "center_loss_ramp", "start_epoch"),
        "center_loss_ramp_end_epoch": ("losses", "schedules", "center_loss_ramp", "end_epoch"),
        "aux_ce_weight": ("losses", "weights", "aux_ce_weight"),
        "aux_ce_drop_epoch": ("losses", "aux_ce_drop_epoch"),
        "arcface_scale": ("losses", "arcface", "scale"),
        "arcface_margin": ("losses", "arcface", "margin"),
        "cosface_scale": ("losses", "cosface", "scale"),
        "cosface_margin": ("losses", "cosface", "margin"),
        "mcpt_lr_multiplier": ("optimization", "mcpt_lr_multiplier"),
        "mcpt_disabled_eval": ("evaluation", "mcpt_disabled_eval"),
        "color_jitter": ("augmentation", "color_jitter"),
        "gaussian_blur": ("augmentation", "gaussian_blur"),
        "random_grayscale": ("augmentation", "random_grayscale"),
        "random_erasing": ("augmentation", "random_erasing"),
        "random_patch": ("augmentation", "random_patch"),
        "random_crop_scale": ("augmentation", "random_crop_scale"),
        "color_augmentation": ("augmentation", "color_augmentation"),
        "background_mosaic": ("augmentation", "background_mosaic", "enabled"),
        "background_mosaic_mask_dir": ("augmentation", "background_mosaic", "mask_dir"),
        "background_mosaic_probability": (
            "augmentation",
            "background_mosaic",
            "probability",
        ),
        "background_mosaic_start_epoch": (
            "augmentation",
            "background_mosaic",
            "start_epoch",
        ),
        "background_mosaic_ramp_end_epoch": (
            "augmentation",
            "background_mosaic",
            "ramp_end_epoch",
        ),
        "background_mosaic_min_foreground_ratio": (
            "augmentation",
            "background_mosaic",
            "min_foreground_ratio",
        ),
        "background_mosaic_max_foreground_ratio": (
            "augmentation",
            "background_mosaic",
            "max_foreground_ratio",
        ),
        "background_mosaic_feather": (
            "augmentation",
            "background_mosaic",
            "feather",
        ),
        "background_mosaic_dilation": (
            "augmentation",
            "background_mosaic",
            "dilation",
        ),
        "background_mosaic_occluder_probability": (
            "augmentation",
            "background_mosaic",
            "occluder_probability",
        ),
        "background_mosaic_occluder_min_area": (
            "augmentation",
            "background_mosaic",
            "occluder_min_area",
        ),
        "background_mosaic_occluder_max_area": (
            "augmentation",
            "background_mosaic",
            "occluder_max_area",
        ),
        "same_id_part_mosaic": ("augmentation", "same_id_part_mosaic", "enabled"),
        "same_id_part_mosaic_probability": (
            "augmentation",
            "same_id_part_mosaic",
            "probability",
        ),
        "same_id_part_mosaic_max_regions": (
            "augmentation",
            "same_id_part_mosaic",
            "max_regions",
        ),
        "same_id_part_mosaic_min_area": (
            "augmentation",
            "same_id_part_mosaic",
            "min_area",
        ),
        "same_id_part_mosaic_max_area": (
            "augmentation",
            "same_id_part_mosaic",
            "max_area",
        ),
        "same_id_part_mosaic_boundary_jitter": (
            "augmentation",
            "same_id_part_mosaic",
            "boundary_jitter",
        ),
        "same_id_part_mosaic_cross_camera_rate": (
            "augmentation",
            "same_id_part_mosaic",
            "cross_camera_rate",
        ),
        "same_id_part_mosaic_min_unaltered": (
            "augmentation",
            "same_id_part_mosaic",
            "min_unaltered",
        ),
        "pav_mosaic": ("augmentation", "pav_mosaic", "enabled"),
        "pav_metadata_dir": ("augmentation", "pav_mosaic", "metadata_dir"),
        "pav_mosaic_probability": (
            "augmentation",
            "pav_mosaic",
            "probability",
        ),
        "pav_mosaic_max_parts": (
            "augmentation",
            "pav_mosaic",
            "max_parts",
        ),
        "pav_mosaic_max_foreground_replacement": (
            "augmentation",
            "pav_mosaic",
            "max_foreground_replacement",
        ),
        "pav_mosaic_cross_camera_rate": (
            "augmentation",
            "pav_mosaic",
            "cross_camera_rate",
        ),
        "pav_mosaic_different_pose_rate": (
            "augmentation",
            "pav_mosaic",
            "different_pose_rate",
        ),
        "pav_mosaic_min_keypoint_confidence": (
            "augmentation",
            "pav_mosaic",
            "min_keypoint_confidence",
        ),
        "pav_mosaic_min_unaltered": (
            "augmentation",
            "pav_mosaic",
            "min_unaltered",
        ),
        "pav_mosaic_warmup_epochs": (
            "augmentation",
            "pav_mosaic",
            "warmup_epochs",
        ),
        "pav_mosaic_decay_start_epoch": (
            "augmentation",
            "pav_mosaic",
            "decay_start_epoch",
        ),
        "pav_mosaic_final_probability_scale": (
            "augmentation",
            "pav_mosaic",
            "final_probability_scale",
        ),
        "pav_consistency_weight": (
            "augmentation",
            "pav_mosaic",
            "consistency_weight",
        ),
        "clean_student_consistency_weight": (
            "augmentation",
            "anatomical_supervision",
            "clean_student_consistency_weight",
        ),
        "eval_interval": ("evaluation", "eval_interval"),
        "eval_datasets": ("evaluation", "eval_datasets"),
        "flip_tta": ("evaluation", "flip_tta"),
        "device": ("system", "device"),
        "metric_dim": ("derived", "metric_dim"),
        "classifier_dim": ("derived", "classifier_dim"),
        "n_params": ("derived", "n_params"),
    }

    for key, path in mappings.items():
        value = _nested_get(hparams, *path)
        if value is not None:
            flat[key] = value
    saved_run_schema = (
        isinstance(hparams.get("resume"), dict)
        and isinstance(hparams.get("run"), dict)
    )
    if (
        flat.get("anatomical_auxiliary")
        and flat.get("anatomical_target_type") is None
        and flat.get("anatomical_teacher_momentum") is not None
    ):
        # A11v8 predates the explicit target-type field, but its EMA momentum
        # unambiguously identifies the learned pose-concatenation teacher.
        flat["anatomical_target_type"] = V8_ANATOMICAL_TARGET_TYPE
    if saved_run_schema and flat.get("anatomical_auxiliary"):
        # These losses were introduced after the original v8 run. Their
        # historical absence means disabled, not today's non-zero defaults.
        flat.setdefault("anatomical_foreground_weight", 0.0)
        flat.setdefault("anatomical_semantic_part_weight", 0.0)

    return flat


def train_hparams_to_args(hparams: dict[str, Any]) -> dict[str, Any]:
    """Convert canonical trainer hparams to public train-command argument keys."""
    return {
        TRAIN_HPARAM_TO_ARG.get(key, key): value
        for key, value in flatten_train_hparams(hparams).items()
    }


def load_train_hparams(resume_path: str | Path) -> dict[str, Any]:
    """Load normalized train hparams from a resume directory or checkpoint path."""
    path = Path(resume_path)
    hparams_file = path / "hparams.json" if path.is_dir() else path.parent / "hparams.json"
    if not hparams_file.exists():
        return {}
    return flatten_train_hparams(json.loads(hparams_file.read_text(encoding="utf-8")))


def _arg_or_hparam(
    args: Any,
    hparams: dict[str, Any],
    explicit_keys: set[str],
    hparam_key: str,
    arg_key: str,
    default: Any = None,
) -> Any:
    if hparam_key in hparams and arg_key not in explicit_keys:
        return hparams[hparam_key]
    return getattr(args, arg_key, default)


def _normalize_img_size(img_size: Any) -> tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size // 2)
    if isinstance(img_size, (list, tuple)) and len(img_size) == 1:
        return (img_size[0], img_size[0] // 2)
    return tuple(img_size)


def trainer_kwargs_from_args(
    args: Any,
    hparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build trainer kwargs from a CLI/API namespace and optional resume hparams."""
    explicit_keys = set(getattr(args, "train_explicit_keys", ()))
    resume = getattr(args, "resume", None)
    if hparams is None:
        hparams = load_train_hparams(resume) if resume else {}

    def value(hparam_key: str, arg_key: str | None = None, default: Any = None) -> Any:
        resolved_arg_key = arg_key or TRAIN_HPARAM_TO_ARG.get(hparam_key, hparam_key)
        return _arg_or_hparam(args, hparams, explicit_keys, hparam_key, resolved_arg_key, default)

    img_size = _normalize_img_size(value("img_size", "imgsz", (256, 128)))
    data_dir = value("data_dir", "data_dir")
    if not data_dir:
        raise ValueError("--data-dir is required (not found in hparams.json either)")

    return {
        "model_name": value("model_name", "model"),
        "dataset_name": value("dataset", "dataset"),
        "data_dir": data_dir,
        "data_specs": value("data_specs", "data_specs", ()),
        "loss_type": value("loss_type", "loss", "triplet"),
        "preprocess": value("preprocess", "preprocess", "resize"),
        "img_size": img_size,
        "batch_size": value("batch_size", "batch_size", 64),
        "lr": value("lr", "lr", 3.5e-4),
        "weight_decay": value("weight_decay", "weight_decay", 5e-4),
        "epochs": value("epochs", "epochs", 120),
        "warmup_epochs": value("warmup_epochs", "warmup_epochs", 10),
        "eval_interval": value("eval_interval", "eval_interval", 10),
        "p": value("p", "p_ids", 16),
        "k": value("k", "k_instances", 4),
        "source_balance": value("source_balance", "source_balance", ""),
        "pk_steps_per_epoch": value("pk_steps_per_epoch", "pk_steps_per_epoch", 0),
        "camera_aware_sampler": value(
            "camera_aware_sampler",
            "camera_aware_sampler",
            False,
        ),
        "margin": value("margin", "margin", 0.3),
        "label_smooth": value("label_smooth", "label_smooth", 0.1),
        "classifier_loss": value("classifier_loss", "classifier_loss", "ce"),
        "triplet_soft_margin": value("triplet_soft_margin", "triplet_soft_margin"),
        "arcface_scale": value("arcface_scale", "arcface_scale", 30.0),
        "arcface_margin": value("arcface_margin", "arcface_margin", 0.5),
        "cosface_scale": value("cosface_scale", "cosface_scale", 30.0),
        "cosface_margin": value("cosface_margin", "cosface_margin", 0.35),
        "center_loss_weight": value("center_loss_weight", "center_loss_weight", 5e-4),
        "id_loss_weight": value("id_loss_weight", "id_loss_weight", 1.0),
        "metric_loss_weight": value("metric_loss_weight", "metric_loss_weight", 1.0),
        "adasp_loss_weight": value("adasp_loss_weight", "adasp_loss_weight", 0.0),
        "adasp_temperature": value("adasp_temperature", "adasp_temperature", 0.04),
        "adasp_scale": value("adasp_scale", "adasp_scale", 0.1),
        "coarse_branch_ce_weight": value("coarse_branch_ce_weight", "coarse_branch_ce_weight", 1.0),
        "fine_branch_ce_weight": value("fine_branch_ce_weight", "fine_branch_ce_weight", 1.0),
        "part_relation_weight": value("part_relation_weight", "part_relation_weight", 0.0),
        "part_to_global_weight": value("part_to_global_weight", "part_to_global_weight", 0.0),
        "part_relation_teacher_momentum": value(
            "part_relation_teacher_momentum", "part_relation_teacher_momentum", 0.999
        ),
        "part_relation_temperature": value(
            "part_relation_temperature", "part_relation_temperature", 0.07
        ),
        "compact_metric_loss_weight": value("compact_metric_loss_weight", "compact_metric_loss_weight", 1.0),
        "compact_cosine_distill_weight": value("compact_cosine_distill_weight", "compact_cosine_distill_weight", 1.0),
        "compact_pairwise_distill_weight": value(
            "compact_pairwise_distill_weight", "compact_pairwise_distill_weight", 1.0
        ),
        "csmm_loss_weight": value("csmm_loss_weight", "csmm_loss_weight", 0.0),
        "csmm_margin": value("csmm_margin", "csmm_margin", 0.10),
        "csmm_temperature": value("csmm_temperature", "csmm_temperature", 0.05),
        "csmm_topk_negatives": value("csmm_topk_negatives", "csmm_topk_negatives", 8),
        "csmm_start_epoch": value("csmm_start_epoch", "csmm_start_epoch", 20),
        "csmm_ramp_end_epoch": value("csmm_ramp_end_epoch", "csmm_ramp_end_epoch", 40),
        "treeboost_loss_weight": value("treeboost_loss_weight", "treeboost_loss_weight", 0.0),
        "treeboost_coarse_coefficient": value("treeboost_coarse_coefficient", "treeboost_coarse_coefficient", 1.0),
        "treeboost_fine_coefficient": value("treeboost_fine_coefficient", "treeboost_fine_coefficient", 1.0),
        "treeboost_node_coefficient": value("treeboost_node_coefficient", "treeboost_node_coefficient", 0.25),
        "treeboost_regression_coefficient": value(
            "treeboost_regression_coefficient", "treeboost_regression_coefficient", 0.10
        ),
        "treeboost_difficulty_floor": value("treeboost_difficulty_floor", "treeboost_difficulty_floor", 0.25),
        "treeboost_regression_tolerance": value(
            "treeboost_regression_tolerance", "treeboost_regression_tolerance", 0.02
        ),
        "treeboost_temperature": value("treeboost_temperature", "treeboost_temperature", 0.05),
        "treeboost_start_epoch": value("treeboost_start_epoch", "treeboost_start_epoch", 30),
        "treeboost_ramp_end_epoch": value("treeboost_ramp_end_epoch", "treeboost_ramp_end_epoch", 60),
        "global_ap_loss_weight": value("global_ap_loss_weight", "global_ap_loss_weight", 0.0),
        "global_ap_temperature": value("global_ap_temperature", "global_ap_temperature", 0.05),
        "global_ap_topk": value("global_ap_topk", "global_ap_topk", 500),
        "global_ap_memory_size": value("global_ap_memory_size", "global_ap_memory_size", 16384),
        "global_ap_momentum": value("global_ap_momentum", "global_ap_momentum", 0.2),
        "global_ap_max_age": value("global_ap_max_age", "global_ap_max_age", 0),
        "global_ap_start_epoch": value("global_ap_start_epoch", "global_ap_start_epoch", 20),
        "global_ap_ramp_end_epoch": value("global_ap_ramp_end_epoch", "global_ap_ramp_end_epoch", 50),
        "global_ap_decay_start_epoch": value("global_ap_decay_start_epoch", "global_ap_decay_start_epoch", 130),
        "global_ap_decay_end_epoch": value("global_ap_decay_end_epoch", "global_ap_decay_end_epoch", 170),
        "hpgrd_cache_dir": value("hpgrd_cache_dir", "hpgrd_cache_dir"),
        "hpgrd_global_weight": value("hpgrd_global_weight", "hpgrd_global_weight", 0.0),
        "hpgrd_part_weight": value("hpgrd_part_weight", "hpgrd_part_weight", 0.0),
        "hpgrd_background_weight": value("hpgrd_background_weight", "hpgrd_background_weight", 0.0),
        "hpgrd_part_drop_weight": value("hpgrd_part_drop_weight", "hpgrd_part_drop_weight", 0.0),
        "hpgrd_part_drop_probability": value(
            "hpgrd_part_drop_probability", "hpgrd_part_drop_probability", 0.0
        ),
        "hpgrd_gradient_fraction": value("hpgrd_gradient_fraction", "hpgrd_gradient_fraction", 0.30),
        "hpgrd_min_confidence": value("hpgrd_min_confidence", "hpgrd_min_confidence", 0.05),
        "early_id_loss_weight": value("early_id_loss_weight", "early_id_loss_weight", 0.0),
        "early_id_loss_epochs": value("early_id_loss_epochs", "early_id_loss_epochs", 0),
        "center_loss_ramp_start_epoch": value(
            "center_loss_ramp_start_epoch",
            "center_loss_ramp_start_epoch",
            0,
        ),
        "center_loss_ramp_end_epoch": value(
            "center_loss_ramp_end_epoch",
            "center_loss_ramp_end_epoch",
            0,
        ),
        "aux_ce_weight": value("aux_ce_weight", "aux_ce_weight", 1.0),
        "aux_ce_drop_epoch": value("aux_ce_drop_epoch", "aux_ce_drop_epoch", 0),
        "branch_loss_agg": value("branch_loss_agg", "branch_loss_agg", "mean"),
        "scale_balanced_branches": value("scale_balanced_branches", "scale_balanced_branches", False),
        "multilevel_suppression": value("multilevel_suppression", "multilevel_suppression", False),
        "multilevel_suppression_ratio": value(
            "multilevel_suppression_ratio", "multilevel_suppression_ratio", 0.15
        ),
        "multilevel_suppression_loss_weight": value(
            "multilevel_suppression_loss_weight", "multilevel_suppression_loss_weight", 0.20
        ),
        "multilevel_suppression_start_epoch": value(
            "multilevel_suppression_start_epoch", "multilevel_suppression_start_epoch", 20
        ),
        "multilevel_suppression_ramp_end_epoch": value(
            "multilevel_suppression_ramp_end_epoch", "multilevel_suppression_ramp_end_epoch", 50
        ),
        "multilevel_suppression_decay_start_epoch": value(
            "multilevel_suppression_decay_start_epoch", "multilevel_suppression_decay_start_epoch", 140
        ),
        "multilevel_suppression_decay_end_epoch": value(
            "multilevel_suppression_decay_end_epoch", "multilevel_suppression_decay_end_epoch", 170
        ),
        "hierarchical_branch_attention": value("hierarchical_branch_attention", "hierarchical_branch_attention", False),
        "branch_attention_token_dim": value("branch_attention_token_dim", "branch_attention_token_dim", 96),
        "branch_attention_num_heads": value("branch_attention_num_heads", "branch_attention_num_heads", 4),
        "branch_attention_num_layers": value("branch_attention_num_layers", "branch_attention_num_layers", 1),
        "branch_attention_mlp_ratio": value("branch_attention_mlp_ratio", "branch_attention_mlp_ratio", 2.0),
        "branch_attention_dropout": value("branch_attention_dropout", "branch_attention_dropout", 0.0),
        "branch_set_attention": value("branch_set_attention", "branch_set_attention", False),
        "branch_set_attention_token_dim": value(
            "branch_set_attention_token_dim", "branch_set_attention_token_dim", 128
        ),
        "branch_set_attention_num_heads": value("branch_set_attention_num_heads", "branch_set_attention_num_heads", 4),
        "branch_set_attention_num_layers": value(
            "branch_set_attention_num_layers", "branch_set_attention_num_layers", 1
        ),
        "branch_set_attention_mlp_ratio": value(
            "branch_set_attention_mlp_ratio", "branch_set_attention_mlp_ratio", 2.0
        ),
        "branch_set_attention_dropout": value("branch_set_attention_dropout", "branch_set_attention_dropout", 0.0),
        "multiscale_query_decoder": value("multiscale_query_decoder", "multiscale_query_decoder", False),
        "query_decoder_dim": value("query_decoder_dim", "query_decoder_dim", 128),
        "query_decoder_num_heads": value("query_decoder_num_heads", "query_decoder_num_heads", 4),
        "query_decoder_num_layers": value("query_decoder_num_layers", "query_decoder_num_layers", 1),
        "query_decoder_mlp_ratio": value("query_decoder_mlp_ratio", "query_decoder_mlp_ratio", 2.0),
        "query_decoder_dropout": value("query_decoder_dropout", "query_decoder_dropout", 0.0),
        "hierarchical_late_interaction": value("hierarchical_late_interaction", "hierarchical_late_interaction", False),
        "late_interaction_dim": value("late_interaction_dim", "late_interaction_dim", 128),
        "late_interaction_num_heads": value("late_interaction_num_heads", "late_interaction_num_heads", 4),
        "late_interaction_num_layers": value("late_interaction_num_layers", "late_interaction_num_layers", 1),
        "late_interaction_sinkhorn_iters": value(
            "late_interaction_sinkhorn_iters", "late_interaction_sinkhorn_iters", 5
        ),
        "late_interaction_null_tokens": value("late_interaction_null_tokens", "late_interaction_null_tokens", 1),
        "late_interaction_negative_identities": value(
            "late_interaction_negative_identities", "late_interaction_negative_identities", 16
        ),
        "late_interaction_rerank_topk": value("late_interaction_rerank_topk", "late_interaction_rerank_topk", 100),
        "late_interaction_base_score_init": value(
            "late_interaction_base_score_init", "late_interaction_base_score_init", 0.9
        ),
        "late_interaction_loss_weight": value("late_interaction_loss_weight", "late_interaction_loss_weight", 0.20),
        "late_interaction_distill_weight": value(
            "late_interaction_distill_weight", "late_interaction_distill_weight", 0.05
        ),
        "late_interaction_temperature": value("late_interaction_temperature", "late_interaction_temperature", 0.07),
        "late_interaction_start_epoch": value("late_interaction_start_epoch", "late_interaction_start_epoch", 20),
        "late_interaction_ramp_end_epoch": value(
            "late_interaction_ramp_end_epoch", "late_interaction_ramp_end_epoch", 50
        ),
        "mcpt_mode": value("mcpt_mode", "mcpt_mode", "none"),
        "mcpt_hidden_dim": value("mcpt_hidden_dim", "mcpt_hidden_dim", 64),
        "mcpt_max_displacement": value(
            "mcpt_max_displacement", "mcpt_max_displacement", 0.15
        ),
        "mcpt_smoothness_weight": value(
            "mcpt_smoothness_weight", "mcpt_smoothness_weight", 0.01
        ),
        "mcpt_identity_weight": value(
            "mcpt_identity_weight", "mcpt_identity_weight", 0.02
        ),
        "mcpt_identity_decay_epoch": value(
            "mcpt_identity_decay_epoch", "mcpt_identity_decay_epoch", 60
        ),
        "mcpt_lr_multiplier": value(
            "mcpt_lr_multiplier", "mcpt_lr_multiplier", 2.0
        ),
        "mcpt_start_epoch": value(
            "mcpt_start_epoch", "mcpt_start_epoch", 10
        ),
        "mcpt_ramp_end_epoch": value(
            "mcpt_ramp_end_epoch", "mcpt_ramp_end_epoch", 40
        ),
        "mcpt_disabled_eval": value(
            "mcpt_disabled_eval", "mcpt_disabled_eval", False
        ),
        "jpm": value("jpm", "jpm", False),
        "jpm_num_groups": value("jpm_num_groups", "jpm_num_groups", 4),
        "jpm_shift": value("jpm_shift", "jpm_shift", 5),
        "jpm_token_dim": value("jpm_token_dim", "jpm_token_dim", 96),
        "jpm_num_heads": value("jpm_num_heads", "jpm_num_heads", 4),
        "jpm_mlp_ratio": value("jpm_mlp_ratio", "jpm_mlp_ratio", 4.0),
        "jpm_dropout": value("jpm_dropout", "jpm_dropout", 0.0),
        "jpm_id_loss_weight": value(
            "jpm_id_loss_weight", "jpm_id_loss_weight", 1.0
        ),
        "jpm_metric_loss_weight": value(
            "jpm_metric_loss_weight", "jpm_metric_loss_weight", 1.0
        ),
        "metric_feature": value("metric_feature", "metric_feature", "auto"),
        "inference_feature": value("inference_feature", "inference_feature", "concat_bn"),
        "feature_fusion": value("feature_fusion", "feature_fusion", "last3"),
        "pyramid_resize_mode": value("pyramid_resize_mode", "pyramid_resize_mode", "bilinear"),
        "spatial_conv_mode": value("spatial_conv_mode", "spatial_conv_mode", "standard"),
        "post_fusion_mixer": value("post_fusion_mixer", "post_fusion_mixer", "none"),
        "post_fusion_mixer_reduction": value(
            "post_fusion_mixer_reduction",
            "post_fusion_mixer_reduction",
            4,
        ),
        "post_fusion_mixer_kernel": value(
            "post_fusion_mixer_kernel",
            "post_fusion_mixer_kernel",
            (5, 3),
        ),
        "post_fusion_mixer_gamma_init": value(
            "post_fusion_mixer_gamma_init",
            "post_fusion_mixer_gamma_init",
            0.0,
        ),
        "feat_dim": value("feat_dim", "feat_dim", 512),
        "neck_dim": value("neck_dim", "neck_dim", 512),
        "drop_path_rate": value("drop_path_rate", "drop_path_rate", 0.1),
        "timm_model_name": value("timm_model_name", "timm_model_name", ""),
        "timm_head_mode": value("timm_head_mode", "timm_head_mode", "pooled"),
        "mobilenetv4_last_stride": value(
            "mobilenetv4_last_stride",
            "mobilenetv4_last_stride",
            2,
        ),
        "mobilenetv4_neck_mode": value(
            "mobilenetv4_neck_mode",
            "mobilenetv4_neck_mode",
            "cnn",
        ),
        "attention_window_layout": value("attention_window_layout", "attention_window_layout", "legacy"),
        "attention_bias": value("attention_bias", "attention_bias", "absolute"),
        "interpolate_pretrained_attention_bias": value(
            "interpolate_pretrained_attention_bias",
            "interpolate_pretrained_attention_bias",
            False,
        ),
        "attention_mask": value("attention_mask", "attention_mask", False),
        "attention_shift": value("attention_shift", "attention_shift", False),
        "stage3_global": value("stage3_global", "stage3_global", False),
        "stage3_downsample": value("stage3_downsample", "stage3_downsample", False),
        "stage2_width_merge_after": value("stage2_width_merge_after", "stage2_width_merge_after", 0),
        "stage2_mlp_ratio": value("stage2_mlp_ratio", "stage2_mlp_ratio", 4.0),
        "stage3_mlp_ratio": value("stage3_mlp_ratio", "stage3_mlp_ratio", 4.0),
        "stage2_depth": value("stage2_depth", "stage2_depth", 6),
        "stage3_depth": value("stage3_depth", "stage3_depth", 2),
        "width_first_hierarchy": value(
            "width_first_hierarchy",
            "width_first_hierarchy",
            False,
        ),
        "identity_registers": value(
            "identity_registers",
            "identity_registers",
            False,
        ),
        "identity_register_count": value(
            "identity_register_count",
            "identity_register_count",
            4,
        ),
        "identity_register_dim": value(
            "identity_register_dim",
            "identity_register_dim",
            128,
        ),
        "identity_register_num_heads": value(
            "identity_register_num_heads",
            "identity_register_num_heads",
            4,
        ),
        "identity_register_dropout": value(
            "identity_register_dropout",
            "identity_register_dropout",
            0.10,
        ),
        "identity_register_gate_init": value(
            "identity_register_gate_init",
            "identity_register_gate_init",
            0.0,
        ),
        "identity_register_diversity_weight": value(
            "identity_register_diversity_weight",
            "identity_register_diversity_weight",
            0.0,
        ),
        "identity_register_diversity_margin": value(
            "identity_register_diversity_margin",
            "identity_register_diversity_margin",
            0.10,
        ),
        "native_branch_widths": value("native_branch_widths", "native_branch_widths", False),
        "fine_map_dim": value("fine_map_dim", "fine_map_dim", 0),
        "compact_deployment_head": value("compact_deployment_head", "compact_deployment_head", False),
        "reid_adapter_stages": value("reid_adapter_stages", "reid_adapter_stages", ()),
        "reid_adapter_reduction": value("reid_adapter_reduction", "reid_adapter_reduction", 4),
        "reid_adapter_suppression_tau": value(
            "reid_adapter_suppression_tau", "reid_adapter_suppression_tau", 0.0
        ),
        "head_pool": value("head_pool", "head_pool", "avg"),
        "head_parts": value("head_parts", "head_parts", (1, 2)),
        "head_type": value("head_type", "head_type", "standard"),
        "multiscale_channel_alpha": value(
            "multiscale_channel_alpha",
            "multiscale_channel_alpha",
            0.5,
        ),
        "body_slot_mode": value(
            "body_slot_mode",
            "body_slot_mode",
            "recurrent_read",
        ),
        "body_slot_alpha": value(
            "body_slot_alpha",
            "body_slot_alpha",
            0.45,
        ),
        "body_slot_visibility_floor": value(
            "body_slot_visibility_floor",
            "body_slot_visibility_floor",
            0.05,
        ),
        "part_pooling": value("part_pooling", "part_pooling", "stripes"),
        "num_part_tokens": value("num_part_tokens", "num_part_tokens", 4),
        "evidence_num_roles": value("evidence_num_roles", "evidence_num_roles", 8),
        "decouple_patterns": value("decouple_patterns", "decouple_patterns", False),
        "pattern_adapter_dim": value("pattern_adapter_dim", "pattern_adapter_dim", 128),
        "stripe_visibility": value("stripe_visibility", "stripe_visibility", False),
        "drop_global_aux": value("drop_global_aux", "drop_global_aux", False),
        "drop_global_aux_ratio": value("drop_global_aux_ratio", "drop_global_aux_ratio", 0.25),
        "branch_aware_metric": value("branch_aware_metric", "branch_aware_metric", False),
        "branch_metric_part_weight": value("branch_metric_part_weight", "branch_metric_part_weight", 0.5),
        "evidence_alignment_loss_weight": value(
            "evidence_alignment_loss_weight",
            "evidence_alignment_loss_weight",
            0.0,
        ),
        "evidence_alignment_margin": value("evidence_alignment_margin", "evidence_alignment_margin", 0.2),
        "evidence_sinkhorn_iters": value("evidence_sinkhorn_iters", "evidence_sinkhorn_iters", 20),
        "evidence_sinkhorn_temperature": value(
            "evidence_sinkhorn_temperature",
            "evidence_sinkhorn_temperature",
            0.1,
        ),
        "evidence_rerank_topk": value("evidence_rerank_topk", "evidence_rerank_topk", 100),
        "evidence_null_loss_weight": value("evidence_null_loss_weight", "evidence_null_loss_weight", 0.0),
        "evidence_diversity_loss_weight": value(
            "evidence_diversity_loss_weight",
            "evidence_diversity_loss_weight",
            0.0,
        ),
        "anatomical_auxiliary": value(
            "anatomical_auxiliary",
            "anatomical_auxiliary",
            False,
        ),
        "anatomical_metadata_dir": value(
            "anatomical_metadata_dir",
            "anatomical_metadata_dir",
            None,
        ),
        "anatomical_person_mask_dir": value(
            "anatomical_person_mask_dir",
            "anatomical_person_mask_dir",
            None,
        ),
        "anatomical_min_keypoint_confidence": value(
            "anatomical_min_keypoint_confidence",
            "anatomical_min_keypoint_confidence",
            0.5,
        ),
        "anatomical_token_dim": value(
            "anatomical_token_dim",
            "anatomical_token_dim",
            128,
        ),
        "anatomical_distill_weight": value(
            "anatomical_distill_weight",
            "anatomical_distill_weight",
            0.20,
        ),
        "anatomical_attention_weight": value(
            "anatomical_attention_weight",
            "anatomical_attention_weight",
            0.10,
        ),
        "anatomical_foreground_weight": value(
            "anatomical_foreground_weight",
            "anatomical_foreground_weight",
            0.15,
        ),
        "anatomical_semantic_part_weight": value(
            "anatomical_semantic_part_weight",
            "anatomical_semantic_part_weight",
            0.0,
        ),
        "anatomical_visibility_weight": value(
            "anatomical_visibility_weight",
            "anatomical_visibility_weight",
            0.05,
        ),
        "anatomical_contrastive_weight": value(
            "anatomical_contrastive_weight",
            "anatomical_contrastive_weight",
            0.10,
        ),
        "anatomical_descriptor_distill_weight": value(
            "anatomical_descriptor_distill_weight",
            "anatomical_descriptor_distill_weight",
            0.0,
        ),
        "anatomical_branch_distill_weight": value(
            "anatomical_branch_distill_weight",
            "anatomical_branch_distill_weight",
            0.0,
        ),
        "anatomical_branch_global_coefficient": value(
            "anatomical_branch_global_coefficient",
            "anatomical_branch_global_coefficient",
            0.20,
        ),
        "anatomical_branch_coarse_coefficient": value(
            "anatomical_branch_coarse_coefficient",
            "anatomical_branch_coarse_coefficient",
            0.30,
        ),
        "anatomical_branch_fine_coefficient": value(
            "anatomical_branch_fine_coefficient",
            "anatomical_branch_fine_coefficient",
            0.50,
        ),
        "anatomical_pose_teacher_weight": value(
            "anatomical_pose_teacher_weight",
            "anatomical_pose_teacher_weight",
            0.0,
        ),
        "anatomical_query_distill_weight": value(
            "anatomical_query_distill_weight",
            "anatomical_query_distill_weight",
            0.0,
        ),
        "anatomical_query_relational_distill_weight": value(
            "anatomical_query_relational_distill_weight",
            "anatomical_query_relational_distill_weight",
            0.0,
        ),
        "anatomical_query_diversity_weight": value(
            "anatomical_query_diversity_weight",
            "anatomical_query_diversity_weight",
            0.0,
        ),
        "anatomical_query_diversity_margin": value(
            "anatomical_query_diversity_margin",
            "anatomical_query_diversity_margin",
            0.10,
        ),
        "anatomical_part_triplet_weight": value(
            "anatomical_part_triplet_weight",
            "anatomical_part_triplet_weight",
            0.0,
        ),
        "anatomical_target_type": value(
            "anatomical_target_type",
            "anatomical_target_type",
            DEFAULT_ANATOMICAL_TARGET_TYPE,
        ),
        "anatomical_teacher_momentum": value(
            "anatomical_teacher_momentum",
            "anatomical_teacher_momentum",
            0.99,
        ),
        "anatomical_multiscale": value(
            "anatomical_multiscale",
            "anatomical_multiscale",
            False,
        ),
        "anatomical_accessory_query": value(
            "anatomical_accessory_query",
            "anatomical_accessory_query",
            False,
        ),
        "anatomical_deployment": value(
            "anatomical_deployment",
            "anatomical_deployment",
            False,
        ),
        "anatomical_deployment_dim": value(
            "anatomical_deployment_dim",
            "anatomical_deployment_dim",
            64,
        ),
        "anatomical_deployment_alpha": value(
            "anatomical_deployment_alpha",
            "anatomical_deployment_alpha",
            0.25,
        ),
        "anatomical_deployment_id_weight": value(
            "anatomical_deployment_id_weight",
            "anatomical_deployment_id_weight",
            0.25,
        ),
        "anatomical_deployment_metric_weight": value(
            "anatomical_deployment_metric_weight",
            "anatomical_deployment_metric_weight",
            0.10,
        ),
        "anatomical_local_scale_weight": value(
            "anatomical_local_scale_weight",
            "anatomical_local_scale_weight",
            0.60,
        ),
        "anatomical_fine_scale_weight": value(
            "anatomical_fine_scale_weight",
            "anatomical_fine_scale_weight",
            0.40,
        ),
        "anatomical_cross_scale_weight": value(
            "anatomical_cross_scale_weight",
            "anatomical_cross_scale_weight",
            0.05,
        ),
        "anatomical_pose_only_reliability": value(
            "anatomical_pose_only_reliability",
            "anatomical_pose_only_reliability",
            0.35,
        ),
        "anatomical_min_effective_coverage": value(
            "anatomical_min_effective_coverage",
            "anatomical_min_effective_coverage",
            0.0,
        ),
        "anatomical_student_start_epoch": value(
            "anatomical_student_start_epoch",
            "anatomical_student_start_epoch",
            0,
        ),
        "anatomical_student_ramp_end_epoch": value(
            "anatomical_student_ramp_end_epoch",
            "anatomical_student_ramp_end_epoch",
            0,
        ),
        "anatomical_query_start_epoch": value(
            "anatomical_query_start_epoch",
            "anatomical_query_start_epoch",
            20,
        ),
        "anatomical_query_ramp_end_epoch": value(
            "anatomical_query_ramp_end_epoch",
            "anatomical_query_ramp_end_epoch",
            50,
        ),
        "anatomical_fine_start_epoch": value(
            "anatomical_fine_start_epoch",
            "anatomical_fine_start_epoch",
            0,
        ),
        "anatomical_fine_ramp_end_epoch": value(
            "anatomical_fine_ramp_end_epoch",
            "anatomical_fine_ramp_end_epoch",
            0,
        ),
        "anatomical_decay_start_epoch": value(
            "anatomical_decay_start_epoch",
            "anatomical_decay_start_epoch",
            0,
        ),
        "anatomical_decay_end_epoch": value(
            "anatomical_decay_end_epoch",
            "anatomical_decay_end_epoch",
            0,
        ),
        "anatomical_temperature": value(
            "anatomical_temperature",
            "anatomical_temperature",
            0.07,
        ),
        "head_warmup_epochs": value("head_warmup_epochs", "head_warmup_epochs", 0),
        "head_warmup_lr_mult": value("head_warmup_lr_mult", "head_warmup_lr_mult", 2.0),
        "vit_lr_profile": value("vit_lr_profile", "vit_lr_profile", "layer_decay"),
        "layer_decay": value("layer_decay", "layer_decay", 0.95),
        "backbone_lr_mult": value("backbone_lr_mult", "backbone_lr_mult", 1.0),
        "backbone_freeze_epochs": value("backbone_freeze_epochs", "backbone_freeze_epochs", 0),
        "gradual_unfreeze": value("gradual_unfreeze", "gradual_unfreeze", False),
        "gradual_unfreeze_head_epochs": value(
            "gradual_unfreeze_head_epochs",
            "gradual_unfreeze_head_epochs",
            5,
        ),
        "gradual_unfreeze_stage_epochs": value(
            "gradual_unfreeze_stage_epochs",
            "gradual_unfreeze_stage_epochs",
            10,
        ),
        "gradual_unfreeze_backbone_lr_mult": value(
            "gradual_unfreeze_backbone_lr_mult",
            "gradual_unfreeze_backbone_lr_mult",
            0.1,
        ),
        "gradual_unfreeze_backbone_lr_epochs": value(
            "gradual_unfreeze_backbone_lr_epochs",
            "gradual_unfreeze_backbone_lr_epochs",
            5,
        ),
        "eta_min": value("eta_min", "eta_min", 1e-7),
        "pretrained": value("pretrained", "pretrained", True),
        "pretrained_weights": value("pretrained_weights", "pretrained_weights"),
        "device": value("device", "device", "cpu"),
        "project": str(value("project", "project", "runs/reid_train")),
        "name": value("name", "name", "exp"),
        "num_workers": value("num_workers", "num_workers", 4),
        "seed": value("seed", "seed", 0),
        "deterministic": value("deterministic", "deterministic", True),
        "eval_datasets": value("eval_datasets", "eval_datasets"),
        "ema_decay": value("ema_decay", "ema_decay"),
        "gaussian_blur": value("gaussian_blur", "gaussian_blur", False),
        "random_grayscale": value("random_grayscale", "random_grayscale", 0.0),
        "color_jitter": value("color_jitter", "color_jitter", False),
        "random_erasing": value("random_erasing", "random_erasing", 0.5),
        "random_patch": value("random_patch", "random_patch", True),
        "random_crop_scale": value("random_crop_scale", "random_crop_scale", 1.05),
        "color_augmentation": value("color_augmentation", "color_augmentation", True),
        "background_mosaic": value("background_mosaic", "background_mosaic", False),
        "background_mosaic_mask_dir": value(
            "background_mosaic_mask_dir",
            "background_mosaic_mask_dir",
        ),
        "background_mosaic_probability": value(
            "background_mosaic_probability",
            "background_mosaic_probability",
            0.3,
        ),
        "background_mosaic_start_epoch": value(
            "background_mosaic_start_epoch",
            "background_mosaic_start_epoch",
            10,
        ),
        "background_mosaic_ramp_end_epoch": value(
            "background_mosaic_ramp_end_epoch",
            "background_mosaic_ramp_end_epoch",
            30,
        ),
        "background_mosaic_min_foreground_ratio": value(
            "background_mosaic_min_foreground_ratio",
            "background_mosaic_min_foreground_ratio",
            0.2,
        ),
        "background_mosaic_max_foreground_ratio": value(
            "background_mosaic_max_foreground_ratio",
            "background_mosaic_max_foreground_ratio",
            0.9,
        ),
        "background_mosaic_feather": value(
            "background_mosaic_feather",
            "background_mosaic_feather",
            1.5,
        ),
        "background_mosaic_dilation": value(
            "background_mosaic_dilation",
            "background_mosaic_dilation",
            2,
        ),
        "background_mosaic_occluder_probability": value(
            "background_mosaic_occluder_probability",
            "background_mosaic_occluder_probability",
            0.0,
        ),
        "background_mosaic_occluder_min_area": value(
            "background_mosaic_occluder_min_area",
            "background_mosaic_occluder_min_area",
            0.05,
        ),
        "background_mosaic_occluder_max_area": value(
            "background_mosaic_occluder_max_area",
            "background_mosaic_occluder_max_area",
            0.20,
        ),
        "same_id_part_mosaic": value(
            "same_id_part_mosaic",
            "same_id_part_mosaic",
            False,
        ),
        "same_id_part_mosaic_probability": value(
            "same_id_part_mosaic_probability",
            "same_id_part_mosaic_probability",
            0.35,
        ),
        "same_id_part_mosaic_max_regions": value(
            "same_id_part_mosaic_max_regions",
            "same_id_part_mosaic_max_regions",
            2,
        ),
        "same_id_part_mosaic_min_area": value(
            "same_id_part_mosaic_min_area",
            "same_id_part_mosaic_min_area",
            0.15,
        ),
        "same_id_part_mosaic_max_area": value(
            "same_id_part_mosaic_max_area",
            "same_id_part_mosaic_max_area",
            0.40,
        ),
        "same_id_part_mosaic_boundary_jitter": value(
            "same_id_part_mosaic_boundary_jitter",
            "same_id_part_mosaic_boundary_jitter",
            0.05,
        ),
        "same_id_part_mosaic_cross_camera_rate": value(
            "same_id_part_mosaic_cross_camera_rate",
            "same_id_part_mosaic_cross_camera_rate",
            1.0,
        ),
        "same_id_part_mosaic_min_unaltered": value(
            "same_id_part_mosaic_min_unaltered",
            "same_id_part_mosaic_min_unaltered",
            0.5,
        ),
        "pav_mosaic": value("pav_mosaic", "pav_mosaic", False),
        "pav_metadata_dir": value("pav_metadata_dir", "pav_metadata_dir"),
        "pav_mosaic_probability": value(
            "pav_mosaic_probability",
            "pav_mosaic_probability",
            0.25,
        ),
        "pav_mosaic_max_parts": value(
            "pav_mosaic_max_parts",
            "pav_mosaic_max_parts",
            3,
        ),
        "pav_mosaic_max_foreground_replacement": value(
            "pav_mosaic_max_foreground_replacement",
            "pav_mosaic_max_foreground_replacement",
            0.45,
        ),
        "pav_mosaic_cross_camera_rate": value(
            "pav_mosaic_cross_camera_rate",
            "pav_mosaic_cross_camera_rate",
            0.8,
        ),
        "pav_mosaic_different_pose_rate": value(
            "pav_mosaic_different_pose_rate",
            "pav_mosaic_different_pose_rate",
            0.5,
        ),
        "pav_mosaic_min_keypoint_confidence": value(
            "pav_mosaic_min_keypoint_confidence",
            "pav_mosaic_min_keypoint_confidence",
            0.5,
        ),
        "pav_mosaic_min_unaltered": value(
            "pav_mosaic_min_unaltered",
            "pav_mosaic_min_unaltered",
            0.5,
        ),
        "pav_mosaic_warmup_epochs": value(
            "pav_mosaic_warmup_epochs",
            "pav_mosaic_warmup_epochs",
            40,
        ),
        "pav_mosaic_decay_start_epoch": value(
            "pav_mosaic_decay_start_epoch",
            "pav_mosaic_decay_start_epoch",
            170,
        ),
        "pav_mosaic_final_probability_scale": value(
            "pav_mosaic_final_probability_scale",
            "pav_mosaic_final_probability_scale",
            0.5,
        ),
        "pav_consistency_weight": value(
            "pav_consistency_weight",
            "pav_consistency_weight",
            0.0,
        ),
        "clean_student_consistency_weight": value(
            "clean_student_consistency_weight",
            "clean_student_consistency_weight",
            0.0,
        ),
        "flip_tta": value("flip_tta", "flip_tta"),
        "resume": resume,
        "explicit_hparams": explicit_keys,
    }


@dataclass(frozen=True)
class RunConfig:
    """Execution, persistence, and reproducibility settings."""

    device: str = "cpu"
    project: str = "runs/reid_train"
    name: str = "exp"
    seed: int = 0
    deterministic: bool = True
    resume: Optional[str] = None
    explicit_hparams: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class DataConfig:
    """Dataset, preprocessing, sampling, and loader settings."""

    dataset_name: str = "market1501"
    data_dir: str = "."
    data_specs: Tuple[dict[str, Any], ...] = ()
    preprocess: str = "resize"
    img_size: Tuple[int, int] = (256, 128)
    batch_size: int = 64
    p: int = 16
    k: int = 4
    source_balance: str = ""
    pk_steps_per_epoch: int = 0
    camera_aware_sampler: bool = False
    num_workers: int = 4


@dataclass(frozen=True)
class ModelConfig:
    """Backbone and embedding-head settings."""

    model_name: str = "osnet_x0_25"
    pretrained: bool = True
    pretrained_weights: Optional[str] = None
    metric_feature: str = "auto"
    inference_feature: str = "concat_bn"
    feature_fusion: str = "last3"
    pyramid_resize_mode: str = "bilinear"
    spatial_conv_mode: str = "standard"
    post_fusion_mixer: str = "none"
    post_fusion_mixer_reduction: int = 4
    post_fusion_mixer_kernel: tuple[int, int] | list[int] = (5, 3)
    post_fusion_mixer_gamma_init: float = 0.0
    feat_dim: int = 512
    neck_dim: int = 512
    drop_path_rate: float = 0.1
    timm_model_name: str = ""
    timm_head_mode: str = "pooled"
    mobilenetv4_last_stride: int = 2
    mobilenetv4_neck_mode: str = "cnn"
    attention_window_layout: str = "legacy"
    attention_bias: str = "absolute"
    interpolate_pretrained_attention_bias: bool = False
    attention_mask: bool = False
    attention_shift: bool = False
    stage3_global: bool = False
    stage3_downsample: bool = False
    stage2_width_merge_after: int = 0
    stage2_mlp_ratio: float = 4.0
    stage3_mlp_ratio: float = 4.0
    stage2_depth: int = 6
    stage3_depth: int = 2
    width_first_hierarchy: bool = False
    identity_registers: bool = False
    identity_register_count: int = 4
    identity_register_dim: int = 128
    identity_register_num_heads: int = 4
    identity_register_dropout: float = 0.10
    identity_register_gate_init: float = 0.0
    identity_register_diversity_weight: float = 0.0
    identity_register_diversity_margin: float = 0.10
    native_branch_widths: bool = False
    fine_map_dim: int = 0
    compact_deployment_head: bool = False
    reid_adapter_stages: tuple[int, ...] | list[int] = ()
    reid_adapter_reduction: int = 4
    reid_adapter_suppression_tau: float = 0.0
    multilevel_suppression: bool = False
    multilevel_suppression_ratio: float = 0.15
    branch_aware_metric: bool = False
    branch_metric_part_weight: float = 0.5
    scale_balanced_branches: bool = False
    hierarchical_branch_attention: bool = False
    branch_attention_token_dim: int = 96
    branch_attention_num_heads: int = 4
    branch_attention_num_layers: int = 1
    branch_attention_mlp_ratio: float = 2.0
    branch_attention_dropout: float = 0.0
    branch_set_attention: bool = False
    branch_set_attention_token_dim: int = 128
    branch_set_attention_num_heads: int = 4
    branch_set_attention_num_layers: int = 1
    branch_set_attention_mlp_ratio: float = 2.0
    branch_set_attention_dropout: float = 0.0
    multiscale_query_decoder: bool = False
    query_decoder_dim: int = 128
    query_decoder_num_heads: int = 4
    query_decoder_num_layers: int = 1
    query_decoder_mlp_ratio: float = 2.0
    query_decoder_dropout: float = 0.0
    hierarchical_late_interaction: bool = False
    late_interaction_dim: int = 128
    late_interaction_num_heads: int = 4
    late_interaction_num_layers: int = 1
    late_interaction_sinkhorn_iters: int = 5
    late_interaction_null_tokens: int = 1
    late_interaction_negative_identities: int = 16
    late_interaction_rerank_topk: int = 100
    late_interaction_base_score_init: float = 0.9
    mcpt_mode: str = "none"
    mcpt_hidden_dim: int = 64
    mcpt_max_displacement: float = 0.15
    mcpt_start_epoch: int = 10
    mcpt_ramp_end_epoch: int = 40
    jpm: bool = False
    jpm_num_groups: int = 4
    jpm_shift: int = 5
    jpm_token_dim: int = 96
    jpm_num_heads: int = 4
    jpm_mlp_ratio: float = 4.0
    jpm_dropout: float = 0.0
    evidence_num_roles: int = 8
    anatomical_token_dim: int = 128
    anatomical_multiscale: bool = False
    anatomical_accessory_query: bool = False
    anatomical_target_type: str = DEFAULT_ANATOMICAL_TARGET_TYPE
    anatomical_deployment: bool = False
    anatomical_deployment_dim: int = 64
    anatomical_deployment_alpha: float = 0.25
    head_pool: str = "avg"
    head_parts: tuple[int, ...] | list[int] = (1, 2)
    head_type: str = "standard"
    multiscale_channel_alpha: float = 0.5
    body_slot_mode: str = "recurrent_read"
    body_slot_alpha: float = 0.45
    body_slot_visibility_floor: float = 0.05
    part_pooling: str = "stripes"
    num_part_tokens: int = 4
    decouple_patterns: bool = False
    pattern_adapter_dim: int = 128
    stripe_visibility: bool = False
    drop_global_aux: bool = False
    drop_global_aux_ratio: float = 0.25
    head_warmup_epochs: int = 0
    head_warmup_lr_mult: float = 2.0


@dataclass(frozen=True)
class LossConfig:
    """Classification, metric, center, and branch-loss settings."""

    loss_type: str = "triplet"
    margin: float = 0.3
    label_smooth: float = 0.1
    classifier_loss: str = "ce"
    triplet_soft_margin: Optional[bool] = None
    arcface_scale: float = 30.0
    arcface_margin: float = 0.5
    cosface_scale: float = 30.0
    cosface_margin: float = 0.35
    center_loss_weight: float = 5e-4
    id_loss_weight: float = 1.0
    metric_loss_weight: float = 1.0
    adasp_loss_weight: float = 0.0
    adasp_temperature: float = 0.04
    adasp_scale: float = 0.1
    coarse_branch_ce_weight: float = 1.0
    fine_branch_ce_weight: float = 1.0
    part_relation_weight: float = 0.0
    part_to_global_weight: float = 0.0
    part_relation_teacher_momentum: float = 0.999
    part_relation_temperature: float = 0.07
    compact_metric_loss_weight: float = 1.0
    compact_cosine_distill_weight: float = 1.0
    compact_pairwise_distill_weight: float = 1.0
    csmm_loss_weight: float = 0.0
    csmm_margin: float = 0.10
    csmm_temperature: float = 0.05
    csmm_topk_negatives: int = 8
    csmm_start_epoch: int = 20
    csmm_ramp_end_epoch: int = 40
    treeboost_loss_weight: float = 0.0
    treeboost_coarse_coefficient: float = 1.0
    treeboost_fine_coefficient: float = 1.0
    treeboost_node_coefficient: float = 0.25
    treeboost_regression_coefficient: float = 0.10
    treeboost_difficulty_floor: float = 0.25
    treeboost_regression_tolerance: float = 0.02
    treeboost_temperature: float = 0.05
    treeboost_start_epoch: int = 30
    treeboost_ramp_end_epoch: int = 60
    global_ap_loss_weight: float = 0.0
    global_ap_temperature: float = 0.05
    global_ap_topk: int = 500
    global_ap_memory_size: int = 16384
    global_ap_momentum: float = 0.2
    global_ap_max_age: int = 0
    global_ap_start_epoch: int = 20
    global_ap_ramp_end_epoch: int = 50
    global_ap_decay_start_epoch: int = 130
    global_ap_decay_end_epoch: int = 170
    hpgrd_cache_dir: Optional[str] = None
    hpgrd_global_weight: float = 0.0
    hpgrd_part_weight: float = 0.0
    hpgrd_background_weight: float = 0.0
    hpgrd_part_drop_weight: float = 0.0
    hpgrd_part_drop_probability: float = 0.0
    hpgrd_gradient_fraction: float = 0.30
    hpgrd_min_confidence: float = 0.05
    late_interaction_loss_weight: float = 0.20
    late_interaction_distill_weight: float = 0.05
    late_interaction_temperature: float = 0.07
    late_interaction_start_epoch: int = 20
    late_interaction_ramp_end_epoch: int = 50
    mcpt_smoothness_weight: float = 0.01
    mcpt_identity_weight: float = 0.02
    mcpt_identity_decay_epoch: int = 60
    jpm_id_loss_weight: float = 1.0
    jpm_metric_loss_weight: float = 1.0
    early_id_loss_weight: float = 0.0
    early_id_loss_epochs: int = 0
    center_loss_ramp_start_epoch: int = 0
    center_loss_ramp_end_epoch: int = 0
    aux_ce_weight: float = 1.0
    aux_ce_drop_epoch: int = 0
    branch_loss_agg: str = "mean"
    multilevel_suppression_loss_weight: float = 0.20
    multilevel_suppression_start_epoch: int = 20
    multilevel_suppression_ramp_end_epoch: int = 50
    multilevel_suppression_decay_start_epoch: int = 140
    multilevel_suppression_decay_end_epoch: int = 170
    evidence_alignment_loss_weight: float = 0.0
    evidence_alignment_margin: float = 0.2
    evidence_sinkhorn_iters: int = 20
    evidence_sinkhorn_temperature: float = 0.1
    evidence_rerank_topk: int = 100
    evidence_null_loss_weight: float = 0.0
    evidence_diversity_loss_weight: float = 0.0


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer and scheduler settings."""

    lr: float = 3.5e-4
    weight_decay: float = 5e-4
    epochs: int = 120
    warmup_epochs: int = 10
    eta_min: float = 1e-7
    ema_decay: Optional[float] = None
    vit_lr_profile: str = "layer_decay"
    layer_decay: float = 0.95
    backbone_lr_mult: float = 1.0
    backbone_freeze_epochs: int = 0
    gradual_unfreeze: bool = False
    gradual_unfreeze_head_epochs: int = 5
    gradual_unfreeze_stage_epochs: int = 10
    gradual_unfreeze_backbone_lr_mult: float = 0.1
    gradual_unfreeze_backbone_lr_epochs: int = 5
    mcpt_lr_multiplier: float = 2.0


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-time image augmentation settings."""

    gaussian_blur: bool = False
    random_grayscale: float = 0.0
    color_jitter: bool = False
    random_erasing: float = 0.5
    random_patch: bool = True
    random_crop_scale: float = 1.05
    color_augmentation: bool = True
    background_mosaic: bool = False
    background_mosaic_mask_dir: Optional[str] = None
    background_mosaic_probability: float = 0.3
    background_mosaic_start_epoch: int = 10
    background_mosaic_ramp_end_epoch: int = 30
    background_mosaic_min_foreground_ratio: float = 0.2
    background_mosaic_max_foreground_ratio: float = 0.9
    background_mosaic_feather: float = 1.5
    background_mosaic_dilation: int = 2
    background_mosaic_occluder_probability: float = 0.0
    background_mosaic_occluder_min_area: float = 0.05
    background_mosaic_occluder_max_area: float = 0.20
    same_id_part_mosaic: bool = False
    same_id_part_mosaic_probability: float = 0.35
    same_id_part_mosaic_max_regions: int = 2
    same_id_part_mosaic_min_area: float = 0.15
    same_id_part_mosaic_max_area: float = 0.40
    same_id_part_mosaic_boundary_jitter: float = 0.05
    same_id_part_mosaic_cross_camera_rate: float = 1.0
    same_id_part_mosaic_min_unaltered: float = 0.5
    pav_mosaic: bool = False
    pav_metadata_dir: Optional[str] = None
    pav_mosaic_probability: float = 0.25
    pav_mosaic_max_parts: int = 3
    pav_mosaic_max_foreground_replacement: float = 0.45
    pav_mosaic_cross_camera_rate: float = 0.8
    pav_mosaic_different_pose_rate: float = 0.5
    pav_mosaic_min_keypoint_confidence: float = 0.5
    pav_mosaic_min_unaltered: float = 0.5
    pav_mosaic_warmup_epochs: int = 40
    pav_mosaic_decay_start_epoch: int = 170
    pav_mosaic_final_probability_scale: float = 0.5
    pav_consistency_weight: float = 0.0
    clean_student_consistency_weight: float = 0.0
    anatomical_auxiliary: bool = False
    anatomical_metadata_dir: Optional[str] = None
    anatomical_person_mask_dir: Optional[str] = None
    anatomical_min_keypoint_confidence: float = 0.5
    anatomical_distill_weight: float = 0.20
    anatomical_attention_weight: float = 0.10
    anatomical_foreground_weight: float = 0.15
    anatomical_semantic_part_weight: float = 0.0
    anatomical_visibility_weight: float = 0.05
    anatomical_contrastive_weight: float = 0.10
    anatomical_descriptor_distill_weight: float = 0.0
    anatomical_branch_distill_weight: float = 0.0
    anatomical_branch_global_coefficient: float = 0.20
    anatomical_branch_coarse_coefficient: float = 0.30
    anatomical_branch_fine_coefficient: float = 0.50
    anatomical_pose_teacher_weight: float = 0.0
    anatomical_query_distill_weight: float = 0.0
    anatomical_query_relational_distill_weight: float = 0.0
    anatomical_query_diversity_weight: float = 0.0
    anatomical_query_diversity_margin: float = 0.10
    anatomical_part_triplet_weight: float = 0.0
    anatomical_teacher_momentum: float = 0.99
    anatomical_deployment_id_weight: float = 0.25
    anatomical_deployment_metric_weight: float = 0.10
    anatomical_local_scale_weight: float = 0.60
    anatomical_fine_scale_weight: float = 0.40
    anatomical_cross_scale_weight: float = 0.05
    anatomical_pose_only_reliability: float = 0.35
    anatomical_min_effective_coverage: float = 0.0
    anatomical_student_start_epoch: int = 0
    anatomical_student_ramp_end_epoch: int = 0
    anatomical_query_start_epoch: int = 20
    anatomical_query_ramp_end_epoch: int = 50
    anatomical_fine_start_epoch: int = 0
    anatomical_fine_ramp_end_epoch: int = 0
    anatomical_decay_start_epoch: int = 0
    anatomical_decay_end_epoch: int = 0
    anatomical_temperature: float = 0.07


@dataclass(frozen=True)
class EvalConfig:
    """Validation frequency and inference augmentation settings."""

    eval_interval: int = 10
    eval_datasets: tuple[str, ...] = ()
    flip_tta: Optional[bool] = None
    mcpt_disabled_eval: bool = False


@dataclass(frozen=True)
class ReIDTrainConfig:
    """Complete typed ReID training configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_flat_kwargs(cls, **values) -> "ReIDTrainConfig":
        """Build nested configuration from the legacy trainer keyword surface."""
        explicit = values.get("explicit_hparams") or ()
        image_size = values.get("img_size", (256, 128))
        if isinstance(image_size, int):
            image_size = (image_size, image_size // 2)
        return cls(
            run=RunConfig(
                device=values.get("device", "cpu"),
                project=str(values.get("project", "runs/reid_train")),
                name=values.get("name", "exp"),
                seed=values.get("seed", 0),
                deterministic=values.get("deterministic", True),
                resume=values.get("resume"),
                explicit_hparams=set(explicit),
            ),
            data=DataConfig(
                dataset_name=values["dataset_name"],
                data_dir=str(values["data_dir"]),
                data_specs=tuple(dict(spec) for spec in values.get("data_specs") or ()),
                preprocess=values.get("preprocess", "resize"),
                img_size=tuple(image_size),
                batch_size=values.get("batch_size", 64),
                p=values.get("p", 16),
                k=values.get("k", 4),
                source_balance=values.get("source_balance", ""),
                pk_steps_per_epoch=values.get("pk_steps_per_epoch", 0),
                camera_aware_sampler=values.get("camera_aware_sampler", False),
                num_workers=values.get("num_workers", 4),
            ),
            model=ModelConfig(
                model_name=values["model_name"],
                pretrained=values.get("pretrained", True),
                pretrained_weights=values.get("pretrained_weights"),
                metric_feature=values.get("metric_feature", "auto"),
                inference_feature=values.get("inference_feature", "concat_bn"),
                feature_fusion=values.get("feature_fusion", "last3"),
                pyramid_resize_mode=values.get("pyramid_resize_mode", "bilinear"),
                spatial_conv_mode=values.get("spatial_conv_mode", "standard"),
                post_fusion_mixer=values.get("post_fusion_mixer", "none"),
                post_fusion_mixer_reduction=values.get("post_fusion_mixer_reduction", 4),
                post_fusion_mixer_kernel=values.get("post_fusion_mixer_kernel", (5, 3)),
                post_fusion_mixer_gamma_init=values.get("post_fusion_mixer_gamma_init", 0.0),
                feat_dim=values.get("feat_dim", 512),
                neck_dim=values.get("neck_dim", 512),
                drop_path_rate=values.get("drop_path_rate", 0.1),
                timm_model_name=values.get("timm_model_name", ""),
                timm_head_mode=values.get("timm_head_mode", "pooled"),
                mobilenetv4_last_stride=values.get("mobilenetv4_last_stride", 2),
                mobilenetv4_neck_mode=values.get("mobilenetv4_neck_mode", "cnn"),
                attention_window_layout=values.get("attention_window_layout", "legacy"),
                attention_bias=values.get("attention_bias", "absolute"),
                interpolate_pretrained_attention_bias=values.get(
                    "interpolate_pretrained_attention_bias",
                    False,
                ),
                attention_mask=values.get("attention_mask", False),
                attention_shift=values.get("attention_shift", False),
                stage3_global=values.get("stage3_global", False),
                stage3_downsample=values.get("stage3_downsample", False),
                stage2_width_merge_after=values.get("stage2_width_merge_after", 0),
                stage2_mlp_ratio=values.get("stage2_mlp_ratio", 4.0),
                stage3_mlp_ratio=values.get("stage3_mlp_ratio", 4.0),
                stage2_depth=values.get("stage2_depth", 6),
                stage3_depth=values.get("stage3_depth", 2),
                width_first_hierarchy=values.get(
                    "width_first_hierarchy",
                    False,
                ),
                identity_registers=values.get(
                    "identity_registers",
                    False,
                ),
                identity_register_count=values.get(
                    "identity_register_count",
                    4,
                ),
                identity_register_dim=values.get(
                    "identity_register_dim",
                    128,
                ),
                identity_register_num_heads=values.get(
                    "identity_register_num_heads",
                    4,
                ),
                identity_register_dropout=values.get(
                    "identity_register_dropout",
                    0.10,
                ),
                identity_register_gate_init=values.get(
                    "identity_register_gate_init",
                    0.0,
                ),
                identity_register_diversity_weight=values.get(
                    "identity_register_diversity_weight",
                    0.0,
                ),
                identity_register_diversity_margin=values.get(
                    "identity_register_diversity_margin",
                    0.10,
                ),
                native_branch_widths=values.get("native_branch_widths", False),
                fine_map_dim=values.get("fine_map_dim", 0),
                compact_deployment_head=values.get("compact_deployment_head", False),
                reid_adapter_stages=values.get("reid_adapter_stages", ()),
                reid_adapter_reduction=values.get("reid_adapter_reduction", 4),
                reid_adapter_suppression_tau=values.get("reid_adapter_suppression_tau", 0.0),
                multilevel_suppression=values.get("multilevel_suppression", False),
                multilevel_suppression_ratio=values.get("multilevel_suppression_ratio", 0.15),
                branch_aware_metric=values.get("branch_aware_metric", False),
                branch_metric_part_weight=values.get("branch_metric_part_weight", 0.5),
                scale_balanced_branches=values.get("scale_balanced_branches", False),
                hierarchical_branch_attention=values.get("hierarchical_branch_attention", False),
                branch_attention_token_dim=values.get("branch_attention_token_dim", 96),
                branch_attention_num_heads=values.get("branch_attention_num_heads", 4),
                branch_attention_num_layers=values.get("branch_attention_num_layers", 1),
                branch_attention_mlp_ratio=values.get("branch_attention_mlp_ratio", 2.0),
                branch_attention_dropout=values.get("branch_attention_dropout", 0.0),
                branch_set_attention=values.get("branch_set_attention", False),
                branch_set_attention_token_dim=values.get("branch_set_attention_token_dim", 128),
                branch_set_attention_num_heads=values.get("branch_set_attention_num_heads", 4),
                branch_set_attention_num_layers=values.get("branch_set_attention_num_layers", 1),
                branch_set_attention_mlp_ratio=values.get("branch_set_attention_mlp_ratio", 2.0),
                branch_set_attention_dropout=values.get("branch_set_attention_dropout", 0.0),
                multiscale_query_decoder=values.get("multiscale_query_decoder", False),
                query_decoder_dim=values.get("query_decoder_dim", 128),
                query_decoder_num_heads=values.get("query_decoder_num_heads", 4),
                query_decoder_num_layers=values.get("query_decoder_num_layers", 1),
                query_decoder_mlp_ratio=values.get("query_decoder_mlp_ratio", 2.0),
                query_decoder_dropout=values.get("query_decoder_dropout", 0.0),
                hierarchical_late_interaction=values.get("hierarchical_late_interaction", False),
                late_interaction_dim=values.get("late_interaction_dim", 128),
                late_interaction_num_heads=values.get("late_interaction_num_heads", 4),
                late_interaction_num_layers=values.get("late_interaction_num_layers", 1),
                late_interaction_sinkhorn_iters=values.get("late_interaction_sinkhorn_iters", 5),
                late_interaction_null_tokens=values.get("late_interaction_null_tokens", 1),
                late_interaction_negative_identities=values.get(
                    "late_interaction_negative_identities",
                    16,
                ),
                late_interaction_rerank_topk=values.get("late_interaction_rerank_topk", 100),
                late_interaction_base_score_init=values.get("late_interaction_base_score_init", 0.9),
                mcpt_mode=values.get("mcpt_mode", "none"),
                mcpt_hidden_dim=values.get("mcpt_hidden_dim", 64),
                mcpt_max_displacement=values.get("mcpt_max_displacement", 0.15),
                mcpt_start_epoch=values.get("mcpt_start_epoch", 10),
                mcpt_ramp_end_epoch=values.get("mcpt_ramp_end_epoch", 40),
                jpm=values.get("jpm", False),
                jpm_num_groups=values.get("jpm_num_groups", 4),
                jpm_shift=values.get("jpm_shift", 5),
                jpm_token_dim=values.get("jpm_token_dim", 96),
                jpm_num_heads=values.get("jpm_num_heads", 4),
                jpm_mlp_ratio=values.get("jpm_mlp_ratio", 4.0),
                jpm_dropout=values.get("jpm_dropout", 0.0),
                evidence_num_roles=values.get("evidence_num_roles", 8),
                anatomical_token_dim=values.get("anatomical_token_dim", 128),
                anatomical_multiscale=values.get(
                    "anatomical_multiscale",
                    False,
                ),
                anatomical_accessory_query=values.get(
                    "anatomical_accessory_query",
                    False,
                ),
                anatomical_target_type=values.get(
                    "anatomical_target_type",
                    DEFAULT_ANATOMICAL_TARGET_TYPE,
                ),
                anatomical_deployment=values.get(
                    "anatomical_deployment",
                    False,
                ),
                anatomical_deployment_dim=values.get(
                    "anatomical_deployment_dim",
                    64,
                ),
                anatomical_deployment_alpha=values.get(
                    "anatomical_deployment_alpha",
                    0.25,
                ),
                head_pool=values.get("head_pool", "avg"),
                head_parts=values.get("head_parts", (1, 2)),
                head_type=values.get("head_type", "standard"),
                multiscale_channel_alpha=values.get(
                    "multiscale_channel_alpha",
                    0.5,
                ),
                body_slot_mode=values.get(
                    "body_slot_mode",
                    "recurrent_read",
                ),
                body_slot_alpha=values.get("body_slot_alpha", 0.45),
                body_slot_visibility_floor=values.get(
                    "body_slot_visibility_floor",
                    0.05,
                ),
                part_pooling=values.get("part_pooling", "stripes"),
                num_part_tokens=values.get("num_part_tokens", 4),
                decouple_patterns=values.get("decouple_patterns", False),
                pattern_adapter_dim=values.get("pattern_adapter_dim", 128),
                stripe_visibility=values.get("stripe_visibility", False),
                drop_global_aux=values.get("drop_global_aux", False),
                drop_global_aux_ratio=values.get("drop_global_aux_ratio", 0.25),
                head_warmup_epochs=values.get("head_warmup_epochs", 0),
                head_warmup_lr_mult=values.get("head_warmup_lr_mult", 2.0),
            ),
            loss=LossConfig(
                loss_type=values.get("loss_type", "triplet"),
                margin=values.get("margin", 0.3),
                label_smooth=values.get("label_smooth", 0.1),
                classifier_loss=values.get("classifier_loss", "ce"),
                triplet_soft_margin=values.get("triplet_soft_margin"),
                arcface_scale=values.get("arcface_scale", 30.0),
                arcface_margin=values.get("arcface_margin", 0.5),
                cosface_scale=values.get("cosface_scale", 30.0),
                cosface_margin=values.get("cosface_margin", 0.35),
                center_loss_weight=values.get("center_loss_weight", 5e-4),
                id_loss_weight=values.get("id_loss_weight", 1.0),
                metric_loss_weight=values.get("metric_loss_weight", 1.0),
                adasp_loss_weight=values.get("adasp_loss_weight", 0.0),
                adasp_temperature=values.get("adasp_temperature", 0.04),
                adasp_scale=values.get("adasp_scale", 0.1),
                coarse_branch_ce_weight=values.get("coarse_branch_ce_weight", 1.0),
                fine_branch_ce_weight=values.get("fine_branch_ce_weight", 1.0),
                part_relation_weight=values.get("part_relation_weight", 0.0),
                part_to_global_weight=values.get("part_to_global_weight", 0.0),
                part_relation_teacher_momentum=values.get(
                    "part_relation_teacher_momentum", 0.999
                ),
                part_relation_temperature=values.get(
                    "part_relation_temperature", 0.07
                ),
                compact_metric_loss_weight=values.get("compact_metric_loss_weight", 1.0),
                compact_cosine_distill_weight=values.get("compact_cosine_distill_weight", 1.0),
                compact_pairwise_distill_weight=values.get("compact_pairwise_distill_weight", 1.0),
                csmm_loss_weight=values.get("csmm_loss_weight", 0.0),
                csmm_margin=values.get("csmm_margin", 0.10),
                csmm_temperature=values.get("csmm_temperature", 0.05),
                csmm_topk_negatives=values.get("csmm_topk_negatives", 8),
                csmm_start_epoch=values.get("csmm_start_epoch", 20),
                csmm_ramp_end_epoch=values.get("csmm_ramp_end_epoch", 40),
                treeboost_loss_weight=values.get("treeboost_loss_weight", 0.0),
                treeboost_coarse_coefficient=values.get("treeboost_coarse_coefficient", 1.0),
                treeboost_fine_coefficient=values.get("treeboost_fine_coefficient", 1.0),
                treeboost_node_coefficient=values.get("treeboost_node_coefficient", 0.25),
                treeboost_regression_coefficient=values.get("treeboost_regression_coefficient", 0.10),
                treeboost_difficulty_floor=values.get("treeboost_difficulty_floor", 0.25),
                treeboost_regression_tolerance=values.get("treeboost_regression_tolerance", 0.02),
                treeboost_temperature=values.get("treeboost_temperature", 0.05),
                treeboost_start_epoch=values.get("treeboost_start_epoch", 30),
                treeboost_ramp_end_epoch=values.get("treeboost_ramp_end_epoch", 60),
                global_ap_loss_weight=values.get("global_ap_loss_weight", 0.0),
                global_ap_temperature=values.get("global_ap_temperature", 0.05),
                global_ap_topk=values.get("global_ap_topk", 500),
                global_ap_memory_size=values.get("global_ap_memory_size", 16384),
                global_ap_momentum=values.get("global_ap_momentum", 0.2),
                global_ap_max_age=values.get("global_ap_max_age", 0),
                global_ap_start_epoch=values.get("global_ap_start_epoch", 20),
                global_ap_ramp_end_epoch=values.get("global_ap_ramp_end_epoch", 50),
                global_ap_decay_start_epoch=values.get("global_ap_decay_start_epoch", 130),
                global_ap_decay_end_epoch=values.get("global_ap_decay_end_epoch", 170),
                hpgrd_cache_dir=values.get("hpgrd_cache_dir"),
                hpgrd_global_weight=values.get("hpgrd_global_weight", 0.0),
                hpgrd_part_weight=values.get("hpgrd_part_weight", 0.0),
                hpgrd_background_weight=values.get("hpgrd_background_weight", 0.0),
                hpgrd_part_drop_weight=values.get("hpgrd_part_drop_weight", 0.0),
                hpgrd_part_drop_probability=values.get("hpgrd_part_drop_probability", 0.0),
                hpgrd_gradient_fraction=values.get("hpgrd_gradient_fraction", 0.30),
                hpgrd_min_confidence=values.get("hpgrd_min_confidence", 0.05),
                late_interaction_loss_weight=values.get("late_interaction_loss_weight", 0.20),
                late_interaction_distill_weight=values.get("late_interaction_distill_weight", 0.05),
                late_interaction_temperature=values.get("late_interaction_temperature", 0.07),
                late_interaction_start_epoch=values.get("late_interaction_start_epoch", 20),
                late_interaction_ramp_end_epoch=values.get("late_interaction_ramp_end_epoch", 50),
                mcpt_smoothness_weight=values.get("mcpt_smoothness_weight", 0.01),
                mcpt_identity_weight=values.get("mcpt_identity_weight", 0.02),
                mcpt_identity_decay_epoch=values.get("mcpt_identity_decay_epoch", 60),
                jpm_id_loss_weight=values.get("jpm_id_loss_weight", 1.0),
                jpm_metric_loss_weight=values.get(
                    "jpm_metric_loss_weight", 1.0
                ),
                early_id_loss_weight=values.get("early_id_loss_weight", 0.0),
                early_id_loss_epochs=values.get("early_id_loss_epochs", 0),
                center_loss_ramp_start_epoch=values.get("center_loss_ramp_start_epoch", 0),
                center_loss_ramp_end_epoch=values.get("center_loss_ramp_end_epoch", 0),
                aux_ce_weight=values.get("aux_ce_weight", 1.0),
                aux_ce_drop_epoch=values.get("aux_ce_drop_epoch", 0),
                branch_loss_agg=values.get("branch_loss_agg", "mean"),
                multilevel_suppression_loss_weight=values.get(
                    "multilevel_suppression_loss_weight", 0.20
                ),
                multilevel_suppression_start_epoch=values.get(
                    "multilevel_suppression_start_epoch", 20
                ),
                multilevel_suppression_ramp_end_epoch=values.get(
                    "multilevel_suppression_ramp_end_epoch", 50
                ),
                multilevel_suppression_decay_start_epoch=values.get(
                    "multilevel_suppression_decay_start_epoch", 140
                ),
                multilevel_suppression_decay_end_epoch=values.get(
                    "multilevel_suppression_decay_end_epoch", 170
                ),
                evidence_alignment_loss_weight=values.get("evidence_alignment_loss_weight", 0.0),
                evidence_alignment_margin=values.get("evidence_alignment_margin", 0.2),
                evidence_sinkhorn_iters=values.get("evidence_sinkhorn_iters", 20),
                evidence_sinkhorn_temperature=values.get("evidence_sinkhorn_temperature", 0.1),
                evidence_rerank_topk=values.get("evidence_rerank_topk", 100),
                evidence_null_loss_weight=values.get("evidence_null_loss_weight", 0.0),
                evidence_diversity_loss_weight=values.get("evidence_diversity_loss_weight", 0.0),
            ),
            optimization=OptimizationConfig(
                lr=values.get("lr", 3.5e-4),
                weight_decay=values.get("weight_decay", 5e-4),
                epochs=values.get("epochs", 120),
                warmup_epochs=values.get("warmup_epochs", 10),
                eta_min=values.get("eta_min", 1e-7),
                ema_decay=values.get("ema_decay"),
                vit_lr_profile=values.get("vit_lr_profile", "layer_decay"),
                layer_decay=values.get("layer_decay", 0.95),
                backbone_lr_mult=values.get("backbone_lr_mult", 1.0),
                backbone_freeze_epochs=values.get("backbone_freeze_epochs", 0),
                gradual_unfreeze=values.get("gradual_unfreeze", False),
                gradual_unfreeze_head_epochs=values.get("gradual_unfreeze_head_epochs", 5),
                gradual_unfreeze_stage_epochs=values.get("gradual_unfreeze_stage_epochs", 10),
                gradual_unfreeze_backbone_lr_mult=values.get("gradual_unfreeze_backbone_lr_mult", 0.1),
                gradual_unfreeze_backbone_lr_epochs=values.get("gradual_unfreeze_backbone_lr_epochs", 5),
                mcpt_lr_multiplier=values.get("mcpt_lr_multiplier", 2.0),
            ),
            augmentation=AugmentationConfig(
                gaussian_blur=values.get("gaussian_blur", False),
                random_grayscale=values.get("random_grayscale", 0.0),
                color_jitter=values.get("color_jitter", False),
                random_erasing=values.get("random_erasing", 0.5),
                random_patch=values.get("random_patch", True),
                random_crop_scale=values.get("random_crop_scale", 1.05),
                color_augmentation=values.get("color_augmentation", True),
                background_mosaic=values.get("background_mosaic", False),
                background_mosaic_mask_dir=values.get("background_mosaic_mask_dir"),
                background_mosaic_probability=values.get(
                    "background_mosaic_probability",
                    0.3,
                ),
                background_mosaic_start_epoch=values.get(
                    "background_mosaic_start_epoch",
                    10,
                ),
                background_mosaic_ramp_end_epoch=values.get(
                    "background_mosaic_ramp_end_epoch",
                    30,
                ),
                background_mosaic_min_foreground_ratio=values.get(
                    "background_mosaic_min_foreground_ratio",
                    0.2,
                ),
                background_mosaic_max_foreground_ratio=values.get(
                    "background_mosaic_max_foreground_ratio",
                    0.9,
                ),
                background_mosaic_feather=values.get(
                    "background_mosaic_feather",
                    1.5,
                ),
                background_mosaic_dilation=values.get(
                    "background_mosaic_dilation",
                    2,
                ),
                background_mosaic_occluder_probability=values.get(
                    "background_mosaic_occluder_probability",
                    0.0,
                ),
                background_mosaic_occluder_min_area=values.get(
                    "background_mosaic_occluder_min_area",
                    0.05,
                ),
                background_mosaic_occluder_max_area=values.get(
                    "background_mosaic_occluder_max_area",
                    0.20,
                ),
                same_id_part_mosaic=values.get("same_id_part_mosaic", False),
                same_id_part_mosaic_probability=values.get(
                    "same_id_part_mosaic_probability",
                    0.35,
                ),
                same_id_part_mosaic_max_regions=values.get(
                    "same_id_part_mosaic_max_regions",
                    2,
                ),
                same_id_part_mosaic_min_area=values.get(
                    "same_id_part_mosaic_min_area",
                    0.15,
                ),
                same_id_part_mosaic_max_area=values.get(
                    "same_id_part_mosaic_max_area",
                    0.40,
                ),
                same_id_part_mosaic_boundary_jitter=values.get(
                    "same_id_part_mosaic_boundary_jitter",
                    0.05,
                ),
                same_id_part_mosaic_cross_camera_rate=values.get(
                    "same_id_part_mosaic_cross_camera_rate",
                    1.0,
                ),
                same_id_part_mosaic_min_unaltered=values.get(
                    "same_id_part_mosaic_min_unaltered",
                    0.5,
                ),
                pav_mosaic=values.get("pav_mosaic", False),
                pav_metadata_dir=values.get("pav_metadata_dir"),
                pav_mosaic_probability=values.get(
                    "pav_mosaic_probability",
                    0.25,
                ),
                pav_mosaic_max_parts=values.get("pav_mosaic_max_parts", 3),
                pav_mosaic_max_foreground_replacement=values.get(
                    "pav_mosaic_max_foreground_replacement",
                    0.45,
                ),
                pav_mosaic_cross_camera_rate=values.get(
                    "pav_mosaic_cross_camera_rate",
                    0.8,
                ),
                pav_mosaic_different_pose_rate=values.get(
                    "pav_mosaic_different_pose_rate",
                    0.5,
                ),
                pav_mosaic_min_keypoint_confidence=values.get(
                    "pav_mosaic_min_keypoint_confidence",
                    0.5,
                ),
                pav_mosaic_min_unaltered=values.get(
                    "pav_mosaic_min_unaltered",
                    0.5,
                ),
                pav_mosaic_warmup_epochs=values.get(
                    "pav_mosaic_warmup_epochs",
                    40,
                ),
                pav_mosaic_decay_start_epoch=values.get(
                    "pav_mosaic_decay_start_epoch",
                    170,
                ),
                pav_mosaic_final_probability_scale=values.get(
                    "pav_mosaic_final_probability_scale",
                    0.5,
                ),
                pav_consistency_weight=values.get(
                    "pav_consistency_weight",
                    0.0,
                ),
                clean_student_consistency_weight=values.get(
                    "clean_student_consistency_weight",
                    0.0,
                ),
                anatomical_auxiliary=values.get(
                    "anatomical_auxiliary",
                    False,
                ),
                anatomical_metadata_dir=values.get("anatomical_metadata_dir"),
                anatomical_person_mask_dir=values.get(
                    "anatomical_person_mask_dir"
                ),
                anatomical_min_keypoint_confidence=values.get(
                    "anatomical_min_keypoint_confidence",
                    0.5,
                ),
                anatomical_distill_weight=values.get(
                    "anatomical_distill_weight",
                    0.20,
                ),
                anatomical_attention_weight=values.get(
                    "anatomical_attention_weight",
                    0.10,
                ),
                anatomical_foreground_weight=values.get(
                    "anatomical_foreground_weight",
                    0.15,
                ),
                anatomical_semantic_part_weight=values.get(
                    "anatomical_semantic_part_weight",
                    0.0,
                ),
                anatomical_visibility_weight=values.get(
                    "anatomical_visibility_weight",
                    0.05,
                ),
                anatomical_contrastive_weight=values.get(
                    "anatomical_contrastive_weight",
                    0.10,
                ),
                anatomical_descriptor_distill_weight=values.get(
                    "anatomical_descriptor_distill_weight",
                    0.0,
                ),
                anatomical_branch_distill_weight=values.get(
                    "anatomical_branch_distill_weight",
                    0.0,
                ),
                anatomical_branch_global_coefficient=values.get(
                    "anatomical_branch_global_coefficient",
                    0.20,
                ),
                anatomical_branch_coarse_coefficient=values.get(
                    "anatomical_branch_coarse_coefficient",
                    0.30,
                ),
                anatomical_branch_fine_coefficient=values.get(
                    "anatomical_branch_fine_coefficient",
                    0.50,
                ),
                anatomical_pose_teacher_weight=values.get(
                    "anatomical_pose_teacher_weight",
                    0.0,
                ),
                anatomical_query_distill_weight=values.get(
                    "anatomical_query_distill_weight",
                    0.0,
                ),
                anatomical_query_relational_distill_weight=values.get(
                    "anatomical_query_relational_distill_weight",
                    0.0,
                ),
                anatomical_query_diversity_weight=values.get(
                    "anatomical_query_diversity_weight",
                    0.0,
                ),
                anatomical_query_diversity_margin=values.get(
                    "anatomical_query_diversity_margin",
                    0.10,
                ),
                anatomical_part_triplet_weight=values.get(
                    "anatomical_part_triplet_weight",
                    0.0,
                ),
                anatomical_teacher_momentum=values.get(
                    "anatomical_teacher_momentum",
                    0.99,
                ),
                anatomical_deployment_id_weight=values.get(
                    "anatomical_deployment_id_weight",
                    0.25,
                ),
                anatomical_deployment_metric_weight=values.get(
                    "anatomical_deployment_metric_weight",
                    0.10,
                ),
                anatomical_local_scale_weight=values.get(
                    "anatomical_local_scale_weight",
                    0.60,
                ),
                anatomical_fine_scale_weight=values.get(
                    "anatomical_fine_scale_weight",
                    0.40,
                ),
                anatomical_cross_scale_weight=values.get(
                    "anatomical_cross_scale_weight",
                    0.05,
                ),
                anatomical_pose_only_reliability=values.get(
                    "anatomical_pose_only_reliability",
                    0.35,
                ),
                anatomical_min_effective_coverage=values.get(
                    "anatomical_min_effective_coverage",
                    0.0,
                ),
                anatomical_student_start_epoch=values.get(
                    "anatomical_student_start_epoch",
                    0,
                ),
                anatomical_student_ramp_end_epoch=values.get(
                    "anatomical_student_ramp_end_epoch",
                    0,
                ),
                anatomical_query_start_epoch=values.get(
                    "anatomical_query_start_epoch",
                    20,
                ),
                anatomical_query_ramp_end_epoch=values.get(
                    "anatomical_query_ramp_end_epoch",
                    50,
                ),
                anatomical_fine_start_epoch=values.get(
                    "anatomical_fine_start_epoch",
                    0,
                ),
                anatomical_fine_ramp_end_epoch=values.get(
                    "anatomical_fine_ramp_end_epoch",
                    0,
                ),
                anatomical_decay_start_epoch=values.get(
                    "anatomical_decay_start_epoch",
                    0,
                ),
                anatomical_decay_end_epoch=values.get(
                    "anatomical_decay_end_epoch",
                    0,
                ),
                anatomical_temperature=values.get(
                    "anatomical_temperature",
                    0.07,
                ),
            ),
            evaluation=EvalConfig(
                eval_interval=values.get("eval_interval", 10),
                eval_datasets=tuple(values.get("eval_datasets") or ()),
                flip_tta=values.get("flip_tta"),
                mcpt_disabled_eval=values.get("mcpt_disabled_eval", False),
            ),
        )

    def to_trainer_kwargs(self) -> dict:
        """Flatten nested configuration for the compatibility constructor."""
        return {
            "model_name": self.model.model_name,
            "dataset_name": self.data.dataset_name,
            "data_dir": self.data.data_dir,
            "data_specs": [dict(spec) for spec in self.data.data_specs],
            **self.loss.__dict__,
            "preprocess": self.data.preprocess,
            "img_size": self.data.img_size,
            "batch_size": self.data.batch_size,
            "p": self.data.p,
            "k": self.data.k,
            "source_balance": self.data.source_balance,
            "pk_steps_per_epoch": self.data.pk_steps_per_epoch,
            "camera_aware_sampler": self.data.camera_aware_sampler,
            "num_workers": self.data.num_workers,
            **self.optimization.__dict__,
            **self.augmentation.__dict__,
            "eval_interval": self.evaluation.eval_interval,
            "eval_datasets": list(self.evaluation.eval_datasets),
            "flip_tta": self.evaluation.flip_tta,
            "mcpt_disabled_eval": self.evaluation.mcpt_disabled_eval,
            **self.model.__dict__,
            **self.run.__dict__,
        }
