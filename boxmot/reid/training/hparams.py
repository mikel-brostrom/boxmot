"""Serialization and loading of reusable ReID training hyperparameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from boxmot.reid.backbones.anatomical_registry import V8_ANATOMICAL_TARGET_TYPE

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


def load_train_hparams(resume_path: str | Path) -> dict[str, Any]:
    """Load normalized train hparams from a resume directory or checkpoint path."""
    path = Path(resume_path)
    hparams_file = path / "hparams.json" if path.is_dir() else path.parent / "hparams.json"
    if not hparams_file.exists():
        return {}
    return flatten_train_hparams(json.loads(hparams_file.read_text(encoding="utf-8")))
