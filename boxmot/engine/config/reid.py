"""Translate ReID command arguments and resume overrides into trainer settings."""

from __future__ import annotations

from typing import Any

from boxmot.reid.backbones.anatomical_registry import (
    DEFAULT_ANATOMICAL_TARGET_TYPE,
)
from boxmot.reid.training.hparams import flatten_train_hparams, load_train_hparams

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


def train_hparams_to_args(hparams: dict[str, Any]) -> dict[str, Any]:
    """Convert canonical trainer hparams to public train-command argument keys."""
    return {
        TRAIN_HPARAM_TO_ARG.get(key, key): value
        for key, value in flatten_train_hparams(hparams).items()
    }


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
