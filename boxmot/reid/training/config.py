"""Typed configuration objects for ReID training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from boxmot.reid.backbones.anatomical_registry import (
    DEFAULT_ANATOMICAL_TARGET_TYPE,
)


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
