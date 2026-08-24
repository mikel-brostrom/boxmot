"""Canonical compatibility contracts for resumable ReID training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

RESUME_CONTRACT_VERSION = 1
# Version 2 changes optimizer parameter-group membership (shape/module-aware
# no-WD), configurable stage decay, and the warmup LR used by each epoch.
# Optimizer states from older implementations cannot be mapped safely even
# though their model state_dict remains valid for inference/pretraining.
OPTIMIZER_CONTRACT_VERSION = 2
_MISSING = object()

# Every field that changes learned weights, optimizer evolution, sampling, or
# validation model selection belongs here. Runtime placement and output paths
# are deliberately absent so a run can move machines and directories safely.
RESUME_CONTRACT_FIELDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "identity": (
        ("model", ("model_name", "model")),
        ("dataset", ("dataset_name", "dataset")),
        ("pretrained", ("pretrained",)),
    ),
    "data": (
        ("preprocess", ("preprocess",)),
        ("img_size", ("img_size", "imgsz")),
        ("p", ("p", "p_ids")),
        ("k", ("k", "k_instances")),
        ("source_balance", ("source_balance",)),
        ("pk_steps_per_epoch", ("pk_steps_per_epoch",)),
        ("camera_aware_sampler", ("camera_aware_sampler",)),
        ("num_workers", ("num_workers",)),
        ("data_specs", ("data_specs",)),
    ),
    "model": (
        ("metric_feature", ("metric_feature",)),
        ("inference_feature", ("inference_feature",)),
        ("feature_fusion", ("feature_fusion",)),
        ("pyramid_resize_mode", ("pyramid_resize_mode",)),
        ("spatial_conv_mode", ("spatial_conv_mode",)),
        ("post_fusion_mixer", ("post_fusion_mixer",)),
        ("post_fusion_mixer_reduction", ("post_fusion_mixer_reduction",)),
        ("post_fusion_mixer_kernel", ("post_fusion_mixer_kernel",)),
        ("post_fusion_mixer_gamma_init", ("post_fusion_mixer_gamma_init",)),
        ("feat_dim", ("feat_dim",)),
        ("neck_dim", ("neck_dim",)),
        ("drop_path_rate", ("drop_path_rate",)),
        ("timm_model_name", ("timm_model_name",)),
        ("timm_head_mode", ("timm_head_mode",)),
        ("mobilenetv4_last_stride", ("mobilenetv4_last_stride",)),
        ("mobilenetv4_neck_mode", ("mobilenetv4_neck_mode",)),
        ("attention_window_layout", ("attention_window_layout",)),
        ("attention_bias", ("attention_bias",)),
        ("interpolate_pretrained_attention_bias", ("interpolate_pretrained_attention_bias",)),
        ("attention_mask", ("attention_mask",)),
        ("attention_shift", ("attention_shift",)),
        ("stage3_global", ("stage3_global",)),
        ("stage3_downsample", ("stage3_downsample",)),
        ("stage2_width_merge_after", ("stage2_width_merge_after",)),
        ("stage2_mlp_ratio", ("stage2_mlp_ratio",)),
        ("stage3_mlp_ratio", ("stage3_mlp_ratio",)),
        ("stage2_depth", ("stage2_depth",)),
        ("stage3_depth", ("stage3_depth",)),
        ("width_first_hierarchy", ("width_first_hierarchy",)),
        ("identity_registers", ("identity_registers",)),
        ("identity_register_count", ("identity_register_count",)),
        ("identity_register_dim", ("identity_register_dim",)),
        (
            "identity_register_num_heads",
            ("identity_register_num_heads",),
        ),
        ("identity_register_dropout", ("identity_register_dropout",)),
        ("identity_register_gate_init", ("identity_register_gate_init",)),
        (
            "identity_register_diversity_weight",
            ("identity_register_diversity_weight",),
        ),
        (
            "identity_register_diversity_margin",
            ("identity_register_diversity_margin",),
        ),
        ("native_branch_widths", ("native_branch_widths",)),
        ("fine_map_dim", ("fine_map_dim",)),
        ("compact_deployment_head", ("compact_deployment_head",)),
        ("reid_adapter_stages", ("reid_adapter_stages",)),
        ("reid_adapter_reduction", ("reid_adapter_reduction",)),
        ("reid_adapter_suppression_tau", ("reid_adapter_suppression_tau",)),
        ("head_pool", ("head_pool",)),
        ("head_parts", ("head_parts",)),
        ("head_type", ("head_type",)),
        (
            "multiscale_channel_alpha",
            ("multiscale_channel_alpha",),
        ),
        ("body_slot_mode", ("body_slot_mode",)),
        ("body_slot_alpha", ("body_slot_alpha",)),
        (
            "body_slot_visibility_floor",
            ("body_slot_visibility_floor",),
        ),
        ("part_pooling", ("part_pooling",)),
        ("num_part_tokens", ("num_part_tokens",)),
        ("evidence_num_roles", ("evidence_num_roles",)),
        ("anatomical_token_dim", ("anatomical_token_dim",)),
        ("anatomical_multiscale", ("anatomical_multiscale",)),
        ("anatomical_accessory_query", ("anatomical_accessory_query",)),
        ("anatomical_target_type", ("anatomical_target_type",)),
        ("anatomical_deployment", ("anatomical_deployment",)),
        (
            "anatomical_deployment_dim",
            ("anatomical_deployment_dim",),
        ),
        (
            "anatomical_deployment_alpha",
            ("anatomical_deployment_alpha",),
        ),
        ("decouple_patterns", ("decouple_patterns",)),
        ("pattern_adapter_dim", ("pattern_adapter_dim",)),
        ("stripe_visibility", ("stripe_visibility",)),
        ("drop_global_aux", ("drop_global_aux",)),
        ("drop_global_aux_ratio", ("drop_global_aux_ratio",)),
        ("branch_aware_metric", ("branch_aware_metric",)),
        ("branch_metric_part_weight", ("branch_metric_part_weight",)),
        ("branch_loss_agg", ("branch_loss_agg",)),
        ("scale_balanced_branches", ("scale_balanced_branches",)),
        ("multilevel_suppression", ("multilevel_suppression",)),
        (
            "multilevel_suppression_ratio",
            ("multilevel_suppression_ratio",),
        ),
        ("hierarchical_branch_attention", ("hierarchical_branch_attention",)),
        ("branch_attention_token_dim", ("branch_attention_token_dim",)),
        ("branch_attention_num_heads", ("branch_attention_num_heads",)),
        ("branch_attention_num_layers", ("branch_attention_num_layers",)),
        ("branch_attention_mlp_ratio", ("branch_attention_mlp_ratio",)),
        ("branch_attention_dropout", ("branch_attention_dropout",)),
        ("branch_set_attention", ("branch_set_attention",)),
        ("branch_set_attention_token_dim", ("branch_set_attention_token_dim",)),
        ("branch_set_attention_num_heads", ("branch_set_attention_num_heads",)),
        ("branch_set_attention_num_layers", ("branch_set_attention_num_layers",)),
        ("branch_set_attention_mlp_ratio", ("branch_set_attention_mlp_ratio",)),
        ("branch_set_attention_dropout", ("branch_set_attention_dropout",)),
        ("multiscale_query_decoder", ("multiscale_query_decoder",)),
        ("query_decoder_dim", ("query_decoder_dim",)),
        ("query_decoder_num_heads", ("query_decoder_num_heads",)),
        ("query_decoder_num_layers", ("query_decoder_num_layers",)),
        ("query_decoder_mlp_ratio", ("query_decoder_mlp_ratio",)),
        ("query_decoder_dropout", ("query_decoder_dropout",)),
        ("hierarchical_late_interaction", ("hierarchical_late_interaction",)),
        ("late_interaction_dim", ("late_interaction_dim",)),
        ("late_interaction_num_heads", ("late_interaction_num_heads",)),
        ("late_interaction_num_layers", ("late_interaction_num_layers",)),
        ("late_interaction_sinkhorn_iters", ("late_interaction_sinkhorn_iters",)),
        ("late_interaction_null_tokens", ("late_interaction_null_tokens",)),
        ("late_interaction_base_score_init", ("late_interaction_base_score_init",)),
        ("mcpt_mode", ("mcpt_mode",)),
        ("mcpt_hidden_dim", ("mcpt_hidden_dim",)),
        ("mcpt_max_displacement", ("mcpt_max_displacement",)),
        ("mcpt_start_epoch", ("mcpt_start_epoch",)),
        ("mcpt_ramp_end_epoch", ("mcpt_ramp_end_epoch",)),
        ("jpm", ("jpm",)),
        ("jpm_num_groups", ("jpm_num_groups",)),
        ("jpm_shift", ("jpm_shift",)),
        ("jpm_token_dim", ("jpm_token_dim",)),
        ("jpm_num_heads", ("jpm_num_heads",)),
        ("jpm_mlp_ratio", ("jpm_mlp_ratio",)),
        ("jpm_dropout", ("jpm_dropout",)),
        ("head_warmup_epochs", ("head_warmup_epochs",)),
        ("head_warmup_lr_mult", ("head_warmup_lr_mult",)),
    ),
    "loss": (
        ("loss_type", ("loss_type", "loss")),
        ("classifier_loss", ("classifier_loss",)),
        ("margin", ("margin",)),
        ("triplet_soft_margin", ("triplet_soft_margin", "soft_margin_triplet")),
        ("label_smooth", ("label_smooth",)),
        ("arcface_scale", ("arcface_scale",)),
        ("arcface_margin", ("arcface_margin",)),
        ("cosface_scale", ("cosface_scale",)),
        ("cosface_margin", ("cosface_margin",)),
        ("center_loss_weight", ("center_loss_weight",)),
        ("id_loss_weight", ("id_loss_weight",)),
        ("metric_loss_weight", ("metric_loss_weight",)),
        ("adasp_loss_weight", ("adasp_loss_weight",)),
        ("adasp_temperature", ("adasp_temperature",)),
        ("adasp_scale", ("adasp_scale",)),
        ("coarse_branch_ce_weight", ("coarse_branch_ce_weight",)),
        ("fine_branch_ce_weight", ("fine_branch_ce_weight",)),
        ("part_relation_weight", ("part_relation_weight",)),
        ("part_to_global_weight", ("part_to_global_weight",)),
        ("part_relation_teacher_momentum", ("part_relation_teacher_momentum",)),
        ("part_relation_temperature", ("part_relation_temperature",)),
        ("compact_metric_loss_weight", ("compact_metric_loss_weight",)),
        ("compact_cosine_distill_weight", ("compact_cosine_distill_weight",)),
        ("compact_pairwise_distill_weight", ("compact_pairwise_distill_weight",)),
        ("csmm_loss_weight", ("csmm_loss_weight",)),
        ("csmm_margin", ("csmm_margin",)),
        ("csmm_temperature", ("csmm_temperature",)),
        ("csmm_topk_negatives", ("csmm_topk_negatives",)),
        ("csmm_start_epoch", ("csmm_start_epoch",)),
        ("csmm_ramp_end_epoch", ("csmm_ramp_end_epoch",)),
        ("treeboost_loss_weight", ("treeboost_loss_weight",)),
        ("treeboost_coarse_coefficient", ("treeboost_coarse_coefficient",)),
        ("treeboost_fine_coefficient", ("treeboost_fine_coefficient",)),
        ("treeboost_node_coefficient", ("treeboost_node_coefficient",)),
        ("treeboost_regression_coefficient", ("treeboost_regression_coefficient",)),
        ("treeboost_difficulty_floor", ("treeboost_difficulty_floor",)),
        ("treeboost_regression_tolerance", ("treeboost_regression_tolerance",)),
        ("treeboost_temperature", ("treeboost_temperature",)),
        ("treeboost_start_epoch", ("treeboost_start_epoch",)),
        ("treeboost_ramp_end_epoch", ("treeboost_ramp_end_epoch",)),
        ("global_ap_loss_weight", ("global_ap_loss_weight",)),
        ("global_ap_temperature", ("global_ap_temperature",)),
        ("global_ap_topk", ("global_ap_topk",)),
        ("global_ap_memory_size", ("global_ap_memory_size",)),
        ("global_ap_momentum", ("global_ap_momentum",)),
        ("global_ap_max_age", ("global_ap_max_age",)),
        ("global_ap_start_epoch", ("global_ap_start_epoch",)),
        ("global_ap_ramp_end_epoch", ("global_ap_ramp_end_epoch",)),
        ("global_ap_decay_start_epoch", ("global_ap_decay_start_epoch",)),
        ("global_ap_decay_end_epoch", ("global_ap_decay_end_epoch",)),
        ("retrieval_dataset_sha256", ("retrieval_dataset_sha256",)),
        ("hpgrd_manifest_sha256", ("hpgrd_manifest_sha256",)),
        ("hpgrd_global_weight", ("hpgrd_global_weight",)),
        ("hpgrd_part_weight", ("hpgrd_part_weight",)),
        ("hpgrd_background_weight", ("hpgrd_background_weight",)),
        ("hpgrd_part_drop_weight", ("hpgrd_part_drop_weight",)),
        ("hpgrd_part_drop_probability", ("hpgrd_part_drop_probability",)),
        ("hpgrd_gradient_fraction", ("hpgrd_gradient_fraction",)),
        ("hpgrd_min_confidence", ("hpgrd_min_confidence",)),
        ("late_interaction_negative_identities", ("late_interaction_negative_identities",)),
        ("late_interaction_loss_weight", ("late_interaction_loss_weight",)),
        ("late_interaction_distill_weight", ("late_interaction_distill_weight",)),
        ("late_interaction_temperature", ("late_interaction_temperature",)),
        ("late_interaction_start_epoch", ("late_interaction_start_epoch",)),
        ("late_interaction_ramp_end_epoch", ("late_interaction_ramp_end_epoch",)),
        ("mcpt_smoothness_weight", ("mcpt_smoothness_weight",)),
        ("mcpt_identity_weight", ("mcpt_identity_weight",)),
        ("mcpt_identity_decay_epoch", ("mcpt_identity_decay_epoch",)),
        ("jpm_id_loss_weight", ("jpm_id_loss_weight",)),
        ("jpm_metric_loss_weight", ("jpm_metric_loss_weight",)),
        (
            "multilevel_suppression_loss_weight",
            ("multilevel_suppression_loss_weight",),
        ),
        (
            "multilevel_suppression_start_epoch",
            ("multilevel_suppression_start_epoch",),
        ),
        (
            "multilevel_suppression_ramp_end_epoch",
            ("multilevel_suppression_ramp_end_epoch",),
        ),
        (
            "multilevel_suppression_decay_start_epoch",
            ("multilevel_suppression_decay_start_epoch",),
        ),
        (
            "multilevel_suppression_decay_end_epoch",
            ("multilevel_suppression_decay_end_epoch",),
        ),
        ("early_id_loss_weight", ("early_id_loss_weight",)),
        ("early_id_loss_epochs", ("early_id_loss_epochs",)),
        ("center_loss_ramp_start_epoch", ("center_loss_ramp_start_epoch",)),
        ("center_loss_ramp_end_epoch", ("center_loss_ramp_end_epoch",)),
        ("aux_ce_weight", ("aux_ce_weight",)),
        ("aux_ce_drop_epoch", ("aux_ce_drop_epoch",)),
        ("evidence_alignment_loss_weight", ("evidence_alignment_loss_weight",)),
        ("evidence_alignment_margin", ("evidence_alignment_margin",)),
        ("evidence_sinkhorn_iters", ("evidence_sinkhorn_iters",)),
        ("evidence_sinkhorn_temperature", ("evidence_sinkhorn_temperature",)),
        ("evidence_rerank_topk", ("evidence_rerank_topk",)),
        ("evidence_null_loss_weight", ("evidence_null_loss_weight",)),
        ("evidence_diversity_loss_weight", ("evidence_diversity_loss_weight",)),
    ),
    "optimization": (
        ("optimizer_contract_version", ("optimizer_contract_version",)),
        ("training_recipe", ("training_recipe",)),
        ("optimizer", ("optimizer", "optimizer_name")),
        ("lr", ("lr",)),
        ("weight_decay", ("weight_decay",)),
        ("layer_decay", ("layer_decay",)),
        ("grad_clip", ("grad_clip",)),
        ("warmup_epochs", ("warmup_epochs",)),
        ("eta_min", ("eta_min",)),
        ("ema_decay", ("ema_decay",)),
        ("vit_lr_profile", ("vit_lr_profile",)),
        ("mcpt_lr_multiplier", ("mcpt_lr_multiplier",)),
        ("backbone_lr_mult", ("backbone_lr_mult",)),
        ("backbone_freeze_epochs", ("backbone_freeze_epochs",)),
        ("gradual_unfreeze", ("gradual_unfreeze",)),
        ("gradual_unfreeze_head_epochs", ("gradual_unfreeze_head_epochs",)),
        ("gradual_unfreeze_stage_epochs", ("gradual_unfreeze_stage_epochs",)),
        ("gradual_unfreeze_backbone_lr_mult", ("gradual_unfreeze_backbone_lr_mult",)),
        ("gradual_unfreeze_backbone_lr_epochs", ("gradual_unfreeze_backbone_lr_epochs",)),
    ),
    "augmentation": (
        ("gaussian_blur", ("gaussian_blur",)),
        ("random_grayscale", ("random_grayscale",)),
        ("color_jitter", ("color_jitter",)),
        ("random_erasing", ("random_erasing",)),
        ("random_patch", ("random_patch",)),
        ("random_crop_scale", ("random_crop_scale",)),
        ("color_augmentation", ("color_augmentation",)),
        ("background_mosaic", ("background_mosaic",)),
        ("background_mosaic_probability", ("background_mosaic_probability",)),
        ("background_mosaic_start_epoch", ("background_mosaic_start_epoch",)),
        ("background_mosaic_ramp_end_epoch", ("background_mosaic_ramp_end_epoch",)),
        (
            "background_mosaic_min_foreground_ratio",
            ("background_mosaic_min_foreground_ratio",),
        ),
        (
            "background_mosaic_max_foreground_ratio",
            ("background_mosaic_max_foreground_ratio",),
        ),
        ("background_mosaic_feather", ("background_mosaic_feather",)),
        ("background_mosaic_dilation", ("background_mosaic_dilation",)),
        (
            "background_mosaic_occluder_probability",
            ("background_mosaic_occluder_probability",),
        ),
        (
            "background_mosaic_occluder_min_area",
            ("background_mosaic_occluder_min_area",),
        ),
        (
            "background_mosaic_occluder_max_area",
            ("background_mosaic_occluder_max_area",),
        ),
        ("same_id_part_mosaic", ("same_id_part_mosaic",)),
        ("same_id_part_mosaic_probability", ("same_id_part_mosaic_probability",)),
        ("same_id_part_mosaic_max_regions", ("same_id_part_mosaic_max_regions",)),
        ("same_id_part_mosaic_min_area", ("same_id_part_mosaic_min_area",)),
        ("same_id_part_mosaic_max_area", ("same_id_part_mosaic_max_area",)),
        (
            "same_id_part_mosaic_boundary_jitter",
            ("same_id_part_mosaic_boundary_jitter",),
        ),
        (
            "same_id_part_mosaic_cross_camera_rate",
            ("same_id_part_mosaic_cross_camera_rate",),
        ),
        (
            "same_id_part_mosaic_min_unaltered",
            ("same_id_part_mosaic_min_unaltered",),
        ),
        ("pav_mosaic", ("pav_mosaic",)),
        ("pav_mosaic_probability", ("pav_mosaic_probability",)),
        ("pav_mosaic_max_parts", ("pav_mosaic_max_parts",)),
        (
            "pav_mosaic_max_foreground_replacement",
            ("pav_mosaic_max_foreground_replacement",),
        ),
        ("pav_mosaic_cross_camera_rate", ("pav_mosaic_cross_camera_rate",)),
        ("pav_mosaic_different_pose_rate", ("pav_mosaic_different_pose_rate",)),
        (
            "pav_mosaic_min_keypoint_confidence",
            ("pav_mosaic_min_keypoint_confidence",),
        ),
        ("pav_mosaic_min_unaltered", ("pav_mosaic_min_unaltered",)),
        ("pav_mosaic_warmup_epochs", ("pav_mosaic_warmup_epochs",)),
        ("pav_mosaic_decay_start_epoch", ("pav_mosaic_decay_start_epoch",)),
        (
            "pav_mosaic_final_probability_scale",
            ("pav_mosaic_final_probability_scale",),
        ),
        ("pav_consistency_weight", ("pav_consistency_weight",)),
        (
            "clean_student_consistency_weight",
            ("clean_student_consistency_weight",),
        ),
        ("anatomical_auxiliary", ("anatomical_auxiliary",)),
        (
            "anatomical_min_keypoint_confidence",
            ("anatomical_min_keypoint_confidence",),
        ),
        ("anatomical_distill_weight", ("anatomical_distill_weight",)),
        ("anatomical_attention_weight", ("anatomical_attention_weight",)),
        (
            "anatomical_foreground_weight",
            ("anatomical_foreground_weight",),
        ),
        (
            "anatomical_semantic_part_weight",
            ("anatomical_semantic_part_weight",),
        ),
        ("anatomical_visibility_weight", ("anatomical_visibility_weight",)),
        (
            "anatomical_contrastive_weight",
            ("anatomical_contrastive_weight",),
        ),
        (
            "anatomical_descriptor_distill_weight",
            ("anatomical_descriptor_distill_weight",),
        ),
        (
            "anatomical_branch_distill_weight",
            ("anatomical_branch_distill_weight",),
        ),
        (
            "anatomical_branch_global_coefficient",
            ("anatomical_branch_global_coefficient",),
        ),
        (
            "anatomical_branch_coarse_coefficient",
            ("anatomical_branch_coarse_coefficient",),
        ),
        (
            "anatomical_branch_fine_coefficient",
            ("anatomical_branch_fine_coefficient",),
        ),
        (
            "anatomical_pose_teacher_weight",
            ("anatomical_pose_teacher_weight",),
        ),
        (
            "anatomical_query_distill_weight",
            ("anatomical_query_distill_weight",),
        ),
        (
            "anatomical_query_relational_distill_weight",
            ("anatomical_query_relational_distill_weight",),
        ),
        (
            "anatomical_query_diversity_weight",
            ("anatomical_query_diversity_weight",),
        ),
        (
            "anatomical_query_diversity_margin",
            ("anatomical_query_diversity_margin",),
        ),
        (
            "anatomical_part_triplet_weight",
            ("anatomical_part_triplet_weight",),
        ),
        (
            "anatomical_teacher_momentum",
            ("anatomical_teacher_momentum",),
        ),
        (
            "anatomical_deployment_id_weight",
            ("anatomical_deployment_id_weight",),
        ),
        (
            "anatomical_deployment_metric_weight",
            ("anatomical_deployment_metric_weight",),
        ),
        (
            "anatomical_local_scale_weight",
            ("anatomical_local_scale_weight",),
        ),
        (
            "anatomical_fine_scale_weight",
            ("anatomical_fine_scale_weight",),
        ),
        (
            "anatomical_cross_scale_weight",
            ("anatomical_cross_scale_weight",),
        ),
        (
            "anatomical_pose_only_reliability",
            ("anatomical_pose_only_reliability",),
        ),
        (
            "anatomical_min_effective_coverage",
            ("anatomical_min_effective_coverage",),
        ),
        (
            "anatomical_student_start_epoch",
            ("anatomical_student_start_epoch",),
        ),
        (
            "anatomical_student_ramp_end_epoch",
            ("anatomical_student_ramp_end_epoch",),
        ),
        (
            "anatomical_query_start_epoch",
            ("anatomical_query_start_epoch",),
        ),
        (
            "anatomical_query_ramp_end_epoch",
            ("anatomical_query_ramp_end_epoch",),
        ),
        (
            "anatomical_fine_start_epoch",
            ("anatomical_fine_start_epoch",),
        ),
        (
            "anatomical_fine_ramp_end_epoch",
            ("anatomical_fine_ramp_end_epoch",),
        ),
        (
            "anatomical_decay_start_epoch",
            ("anatomical_decay_start_epoch",),
        ),
        (
            "anatomical_decay_end_epoch",
            ("anatomical_decay_end_epoch",),
        ),
        ("anatomical_temperature", ("anatomical_temperature",)),
    ),
    "evaluation": (
        ("eval_interval", ("eval_interval",)),
        ("eval_datasets", ("eval_datasets",)),
        ("flip_tta", ("flip_tta",)),
        ("late_interaction_rerank_topk", ("late_interaction_rerank_topk",)),
        ("mcpt_disabled_eval", ("mcpt_disabled_eval",)),
    ),
    "reproducibility": (
        ("seed", ("seed",)),
        ("deterministic", ("deterministic",)),
    ),
}


def _mapping_value(values: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in values:
            return values[alias]
    return _MISSING


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _normalize_data_specs(value: Any) -> list[dict[str, Any]]:
    """Keep dataset identity/sampling fields while allowing roots to move."""
    normalized = []
    for raw in value or ():
        if not isinstance(raw, Mapping):
            continue
        normalized.append(
            {
                str(key): _json_value(item)
                for key, item in sorted(raw.items())
                if key not in {"root", "data_dir", "path"}
            }
        )
    return normalized


def build_resume_contract(values: Mapping[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """Build a stable compatibility contract from trainer args or flat hparams."""
    contract: dict[str, Any] = {"version": RESUME_CONTRACT_VERSION}
    for group_name, fields in RESUME_CONTRACT_FIELDS.items():
        group: dict[str, Any] = {}
        for field_name, aliases in fields:
            value = _mapping_value(values, aliases)
            if value is _MISSING:
                if partial:
                    continue
                raise KeyError(f"Missing resume-contract value: {aliases[0]}")
            if field_name == "data_specs":
                value = _normalize_data_specs(value)
            group[field_name] = _json_value(value)
        if group or not partial:
            contract[group_name] = group
    return contract


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for a resume contract."""
    payload = json.dumps(_json_value(contract), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_fingerprint(contract: Mapping[str, Any], target_epochs: int) -> str:
    """Fingerprint a complete ablation specification, including its target."""
    return contract_fingerprint({"contract": contract, "target_epochs": int(target_epochs)})


def contract_differences(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    compare_common_only: bool = False,
) -> list[str]:
    """Describe value differences between saved and requested contracts."""

    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {prefix: value}
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten(item, path))
        return flattened

    left = flatten(_json_value(expected))
    right = flatten(_json_value(actual))

    # A11v8 predates these explicit anatomy fields. Its persisted EMA teacher
    # settings identify learned_pose_concat_ema, and the two then-unknown
    # losses were necessarily disabled. Normalize the semantic contract before
    # comparing it with the current complete schema.
    for contract in (left, right):
        # Version-1 contracts predate the MobileNetV4 head-path switch and
        # necessarily used the original pool-before-head behavior.
        contract.setdefault("model.timm_model_name", "")
        contract.setdefault("model.timm_head_mode", "pooled")
        contract.setdefault("model.mobilenetv4_last_stride", 2)
        contract.setdefault("model.mobilenetv4_neck_mode", "cnn")
        contract.setdefault("optimization.backbone_lr_mult", 1.0)
        if (
            contract.get("augmentation.anatomical_auxiliary") is True
            and "model.anatomical_target_type" not in contract
            and "augmentation.anatomical_teacher_momentum" in contract
        ):
            contract["model.anatomical_target_type"] = (
                "learned_pose_concat_ema"
            )
        if contract.get("augmentation.anatomical_auxiliary") is True:
            contract.setdefault(
                "augmentation.anatomical_foreground_weight",
                0.0,
            )
            contract.setdefault(
                "augmentation.anatomical_semantic_part_weight",
                0.0,
            )

    keys = left.keys() & right.keys() if compare_common_only else left.keys() | right.keys()
    keys = set(keys)

    left_hpgrd_enabled = any(
        float(left.get(f"loss.hpgrd_{name}_weight", 0.0)) > 0
        for name in ("global", "part", "background", "part_drop")
    )
    right_hpgrd_enabled = any(
        float(right.get(f"loss.hpgrd_{name}_weight", 0.0)) > 0
        for name in ("global", "part", "background", "part_drop")
    )

    # Optional auxiliary-loss fields did not exist in older version-1
    # contracts. A missing family is semantically identical to a disabled
    # family, but enabled families must still match every setting exactly.
    for prefix, weight_key in (
        ("loss.csmm_", "loss.csmm_loss_weight"),
        ("loss.treeboost_", "loss.treeboost_loss_weight"),
    ):
        left_weight = left.get(weight_key, 0.0)
        right_weight = right.get(weight_key, 0.0)
        both_disabled = (
            isinstance(left_weight, Real)
            and not isinstance(left_weight, bool)
            and isinstance(right_weight, Real)
            and not isinstance(right_weight, bool)
            and math.isclose(float(left_weight), 0.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(right_weight), 0.0, rel_tol=0.0, abs_tol=1e-12)
        )
        if both_disabled:
            keys = {key for key in keys if not key.startswith(prefix)}

    left_global_ap_weight = float(left.get("loss.global_ap_loss_weight", 0.0))
    right_global_ap_weight = float(right.get("loss.global_ap_loss_weight", 0.0))
    if math.isclose(left_global_ap_weight, 0.0, abs_tol=1e-12) and math.isclose(
        right_global_ap_weight,
        0.0,
        abs_tol=1e-12,
    ):
        if not left_hpgrd_enabled and not right_hpgrd_enabled:
            keys = {key for key in keys if not key.startswith("loss.global_ap_")}
        else:
            global_ap_only_keys = {
                "loss.global_ap_loss_weight",
                "loss.global_ap_temperature",
                "loss.global_ap_topk",
                "loss.global_ap_memory_size",
                "loss.global_ap_momentum",
                "loss.global_ap_max_age",
            }
            keys.difference_update(global_ap_only_keys)

    if not left_hpgrd_enabled and not right_hpgrd_enabled:
        keys = {key for key in keys if not key.startswith("loss.hpgrd_")}

    # The part-objective ablations were added after some version-1 contracts
    # had already been written. Missing branch CE weights mean the historical
    # full-strength defaults, while missing AdaSP/part-relation families mean
    # those optional objectives were disabled.
    for contract in (left, right):
        contract.setdefault("loss.coarse_branch_ce_weight", 1.0)
        contract.setdefault("loss.fine_branch_ce_weight", 1.0)
    left_adasp_weight = left.get("loss.adasp_loss_weight", 0.0)
    right_adasp_weight = right.get("loss.adasp_loss_weight", 0.0)
    if left_adasp_weight == 0 and right_adasp_weight == 0:
        keys = {key for key in keys if not key.startswith("loss.adasp_")}
    left_part_relation = (
        left.get("loss.part_relation_weight", 0.0),
        left.get("loss.part_to_global_weight", 0.0),
    )
    right_part_relation = (
        right.get("loss.part_relation_weight", 0.0),
        right.get("loss.part_to_global_weight", 0.0),
    )
    if left_part_relation == (0.0, 0.0) and right_part_relation == (0.0, 0.0):
        keys = {
            key
            for key in keys
            if not key.startswith("loss.part_relation_")
            and key != "loss.part_to_global_weight"
        }

    # Hierarchical branch attention is another optional family introduced
    # after version-1 contracts were already in use. Missing settings are
    # equivalent to today's disabled defaults, while enabled runs must retain
    # their complete architectural specification.
    left_branch_attention = left.get("model.hierarchical_branch_attention", False)
    right_branch_attention = right.get("model.hierarchical_branch_attention", False)
    if left_branch_attention is False and right_branch_attention is False:
        keys = {
            key
            for key in keys
            if key != "model.hierarchical_branch_attention" and not key.startswith("model.branch_attention_")
        }

    # The unmasked pre-reduction branch set is also optional. Older contracts
    # without these fields remain equivalent to the disabled default.
    left_branch_set = left.get("model.branch_set_attention", False)
    right_branch_set = right.get("model.branch_set_attention", False)
    if left_branch_set is False and right_branch_set is False:
        keys = {
            key
            for key in keys
            if key != "model.branch_set_attention" and not key.startswith("model.branch_set_attention_")
        }

    left_query_decoder = left.get("model.multiscale_query_decoder", False)
    right_query_decoder = right.get("model.multiscale_query_decoder", False)
    if left_query_decoder is False and right_query_decoder is False:
        keys = {
            key
            for key in keys
            if key != "model.multiscale_query_decoder" and not key.startswith("model.query_decoder_")
        }

    left_head_type = left.get("model.head_type", "standard")
    right_head_type = right.get("model.head_type", "standard")
    if (
        left_head_type != "multiscale_channel2"
        and right_head_type != "multiscale_channel2"
    ):
        keys.discard("model.multiscale_channel_alpha")

    left_late_interaction = left.get("model.hierarchical_late_interaction", False)
    right_late_interaction = right.get("model.hierarchical_late_interaction", False)
    if left_late_interaction is False and right_late_interaction is False:
        keys = {
            key
            for key in keys
            if key != "model.hierarchical_late_interaction"
            and not key.startswith("model.late_interaction_")
            and not key.startswith("loss.late_interaction_")
            and not key.startswith("evaluation.late_interaction_")
        }

    left_mcpt = left.get("model.mcpt_mode", "none")
    right_mcpt = right.get("model.mcpt_mode", "none")
    if left_mcpt == "none" and right_mcpt == "none":
        keys = {
            key
            for key in keys
            if not key.startswith("model.mcpt_")
            and not key.startswith("loss.mcpt_")
            and not key.startswith("optimization.mcpt_")
            and not key.startswith("evaluation.mcpt_")
        }

    left_jpm = left.get("model.jpm", False)
    right_jpm = right.get("model.jpm", False)
    if left_jpm is False and right_jpm is False:
        keys = {
            key
            for key in keys
            if key != "model.jpm"
            and not key.startswith("model.jpm_")
            and not key.startswith("loss.jpm_")
        }

    left_width_first = left.get("model.width_first_hierarchy", False)
    right_width_first = right.get("model.width_first_hierarchy", False)
    if left_width_first is False and right_width_first is False:
        keys.discard("model.width_first_hierarchy")

    left_identity_registers = left.get("model.identity_registers", False)
    right_identity_registers = right.get("model.identity_registers", False)
    if left_identity_registers is False and right_identity_registers is False:
        keys = {
            key
            for key in keys
            if key != "model.identity_registers"
            and not key.startswith("model.identity_register_")
        }

    left_background_mosaic = left.get("augmentation.background_mosaic", False)
    right_background_mosaic = right.get("augmentation.background_mosaic", False)
    if left_background_mosaic is False and right_background_mosaic is False:
        keys = {
            key
            for key in keys
            if key != "augmentation.background_mosaic" and not key.startswith("augmentation.background_mosaic_")
        }
    else:
        left_occluder = left.get(
            "augmentation.background_mosaic_occluder_probability",
            0.0,
        )
        right_occluder = right.get(
            "augmentation.background_mosaic_occluder_probability",
            0.0,
        )
        if left_occluder == 0 and right_occluder == 0:
            keys = {key for key in keys if not key.startswith("augmentation.background_mosaic_occluder_")}

    left_same_id_mosaic = left.get("augmentation.same_id_part_mosaic", False)
    right_same_id_mosaic = right.get("augmentation.same_id_part_mosaic", False)
    if left_same_id_mosaic is False and right_same_id_mosaic is False:
        keys = {
            key
            for key in keys
            if key != "augmentation.same_id_part_mosaic" and not key.startswith("augmentation.same_id_part_mosaic_")
        }

    left_pav_mosaic = left.get("augmentation.pav_mosaic", False)
    right_pav_mosaic = right.get("augmentation.pav_mosaic", False)
    if left_pav_mosaic is False and right_pav_mosaic is False:
        keys = {
            key
            for key in keys
            if key != "augmentation.pav_mosaic"
            and not key.startswith("augmentation.pav_mosaic_")
            and key != "augmentation.pav_consistency_weight"
        }

    left_anatomical = left.get("augmentation.anatomical_auxiliary", False)
    right_anatomical = right.get("augmentation.anatomical_auxiliary", False)
    if left_anatomical is False and right_anatomical is False:
        keys = {
            key
            for key in keys
            if key != "augmentation.anatomical_auxiliary"
            and not key.startswith("augmentation.anatomical_")
            and key != "augmentation.clean_student_consistency_weight"
            and key != "model.anatomical_token_dim"
            and key != "model.anatomical_multiscale"
            and key != "model.anatomical_target_type"
            and not key.startswith("model.anatomical_deployment")
        }
    else:
        left_decoupled_queries = (
            left.get("model.anatomical_target_type")
            == "decoupled_pose_parsing_teacher"
        )
        right_decoupled_queries = (
            right.get("model.anatomical_target_type")
            == "decoupled_pose_parsing_teacher"
        )
        if not left_decoupled_queries and not right_decoupled_queries:
            keys = {
                key
                for key in keys
                if key != "model.anatomical_accessory_query"
                and key
                not in {
                    "augmentation.anatomical_query_distill_weight",
                    "augmentation.anatomical_query_relational_distill_weight",
                    "augmentation.clean_student_consistency_weight",
                    "augmentation.anatomical_query_diversity_weight",
                    "augmentation.anatomical_query_diversity_margin",
                    "augmentation.anatomical_part_triplet_weight",
                    "augmentation.anatomical_query_start_epoch",
                    "augmentation.anatomical_query_ramp_end_epoch",
                }
            }
        left_branch_distill = left.get(
            "augmentation.anatomical_branch_distill_weight",
            0.0,
        )
        right_branch_distill = right.get(
            "augmentation.anatomical_branch_distill_weight",
            0.0,
        )
        if left_branch_distill == 0 and right_branch_distill == 0:
            keys = {
                key
                for key in keys
                if key
                not in {
                    "augmentation.anatomical_branch_distill_weight",
                    "augmentation.anatomical_branch_global_coefficient",
                    "augmentation.anatomical_branch_coarse_coefficient",
                    "augmentation.anatomical_branch_fine_coefficient",
                }
            }
        left_deployment = left.get(
            "model.anatomical_deployment",
            False,
        )
        right_deployment = right.get(
            "model.anatomical_deployment",
            False,
        )
        if left_deployment is False and right_deployment is False:
            keys = {
                key
                for key in keys
                if key
                not in {
                    "model.anatomical_deployment",
                    "model.anatomical_deployment_dim",
                    "model.anatomical_deployment_alpha",
                    "augmentation.anatomical_deployment_id_weight",
                    "augmentation.anatomical_deployment_metric_weight",
                }
            }
        left_multiscale_anatomical = left.get(
            "model.anatomical_multiscale",
            False,
        )
        right_multiscale_anatomical = right.get(
            "model.anatomical_multiscale",
            False,
        )
        if left_multiscale_anatomical is False and right_multiscale_anatomical is False:
            keys = {
                key
                for key in keys
                if key != "model.anatomical_multiscale"
                and key
                not in {
                    "augmentation.anatomical_local_scale_weight",
                    "augmentation.anatomical_fine_scale_weight",
                    "augmentation.anatomical_cross_scale_weight",
                    "augmentation.anatomical_fine_start_epoch",
                    "augmentation.anatomical_fine_ramp_end_epoch",
                }
            }
        else:
            left_fine_schedule = (
                left.get("augmentation.anatomical_fine_start_epoch", 0),
                left.get("augmentation.anatomical_fine_ramp_end_epoch", 0),
            )
            right_fine_schedule = (
                right.get("augmentation.anatomical_fine_start_epoch", 0),
                right.get("augmentation.anatomical_fine_ramp_end_epoch", 0),
            )
            if left_fine_schedule == (0, 0) and right_fine_schedule == (0, 0):
                keys.discard("augmentation.anatomical_fine_start_epoch")
                keys.discard("augmentation.anatomical_fine_ramp_end_epoch")

    left_pk_steps = left.get("data.pk_steps_per_epoch", 0)
    right_pk_steps = right.get("data.pk_steps_per_epoch", 0)
    left_camera_aware = left.get("data.camera_aware_sampler", False)
    right_camera_aware = right.get("data.camera_aware_sampler", False)
    if left_pk_steps == 0 and right_pk_steps == 0 and left_camera_aware is False and right_camera_aware is False:
        keys.discard("data.pk_steps_per_epoch")
        keys.discard("data.camera_aware_sampler")

    differences = []
    for key in sorted(keys):
        saved_value = left.get(key, _MISSING)
        requested_value = right.get(key, _MISSING)
        numerically_equal = (
            isinstance(saved_value, Real)
            and not isinstance(saved_value, bool)
            and isinstance(requested_value, Real)
            and not isinstance(requested_value, bool)
            and math.isclose(float(saved_value), float(requested_value), rel_tol=1e-7, abs_tol=1e-12)
        )
        if saved_value != requested_value and not numerically_equal:
            differences.append(
                f"{key}: saved={left.get(key, '<missing>')!r}, requested={right.get(key, '<missing>')!r}"
            )
    return differences
