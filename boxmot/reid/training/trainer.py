"""ReID model trainer with training loop, validation, and checkpointing."""

from __future__ import annotations

import gc as gc
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import torch

from boxmot.reid.backbones.anatomical_registry import (
    DEFAULT_ANATOMICAL_TARGET_TYPE,
)
from boxmot.reid.backbones.option_registry import normalize_selector
from boxmot.reid.datasets.sampler import parse_source_balance
from boxmot.reid.training.base import BaseTrainer
from boxmot.reid.training.checkpoint import CheckpointManager
from boxmot.reid.training.config import ReIDTrainConfig
from boxmot.reid.training.trainer_components.anatomical_attention import (
    _PrivilegedAttentionMixin,
)
from boxmot.reid.training.trainer_components.anatomical_body_slots import (
    _BodySlotAnatomicalMixin,
)
from boxmot.reid.training.trainer_components.anatomical_common import (
    _AnatomicalCommonMixin,
)
from boxmot.reid.training.trainer_components.anatomical_ema import (
    _EmaAnatomicalMixin,
)
from boxmot.reid.training.trainer_components.anatomical_integration import (
    _AnatomicalIntegrationMixin,
)
from boxmot.reid.training.trainer_components.anatomical_queries import (
    _QueryAnatomicalMixin,
)
from boxmot.reid.training.trainer_components.checkpointing import (
    _CheckpointingMixin,
)
from boxmot.reid.training.trainer_components.configuration import (
    _ConfigurationMixin,
)
from boxmot.reid.training.trainer_components.data import _DataMixin
from boxmot.reid.training.trainer_components.helpers import (
    _cross_scale_role_relation_loss,
    _scale_aware_anatomical_targets,
    _seed_data_worker,
)
from boxmot.reid.training.trainer_components.hpgrd_integration import (
    _HumanPrivilegedRetrievalMixin,
)
from boxmot.reid.training.trainer_components.loop import _TrainingLoopMixin
from boxmot.reid.training.trainer_components.multilevel_suppression import (
    _MultilevelSuppressionMixin,
)
from boxmot.reid.training.trainer_components.objectives import _ObjectiveMixin
from boxmot.reid.training.trainer_components.optimization import (
    _OptimizationMixin,
)
from boxmot.reid.training.trainer_components.reporting import _ReportingMixin
from boxmot.reid.training.trainer_components.resume import _ResumeMixin
from boxmot.reid.training.trainer_components.run_state import _RunStateMixin
from boxmot.reid.training.trainer_components.runtime import _RuntimeMixin
from boxmot.reid.training.trainer_components.setup import _SetupMixin
from boxmot.reid.training.trainer_components.types import (
    DatasetBundle,
    LoaderBundle,
    LossBundle,
    ModelBundle,
    OptimizationBundle,
    ResumeState,
    TrainMetrics,
    TrainResult,
    ValMetrics,
    _TrainingTimeEstimator,
)
from boxmot.reid.training.trainer_components.validation import _ValidationMixin

__all__ = (
    "DatasetBundle",
    "LoaderBundle",
    "LossBundle",
    "ModelBundle",
    "OptimizationBundle",
    "ReIDTrainer",
    "ResumeState",
    "TrainMetrics",
    "TrainResult",
    "ValMetrics",
    "_TrainingTimeEstimator",
    "_cross_scale_role_relation_loss",
    "_scale_aware_anatomical_targets",
    "_seed_data_worker",
)


class ReIDTrainer(
    _ConfigurationMixin,
    _RuntimeMixin,
    _DataMixin,
    _SetupMixin,
    _ResumeMixin,
    _RunStateMixin,
    _CheckpointingMixin,
    _ReportingMixin,
    _TrainingLoopMixin,
    _ValidationMixin,
    _OptimizationMixin,
    _ObjectiveMixin,
    _MultilevelSuppressionMixin,
    _HumanPrivilegedRetrievalMixin,
    _AnatomicalCommonMixin,
    _PrivilegedAttentionMixin,
    _EmaAnatomicalMixin,
    _QueryAnatomicalMixin,
    _BodySlotAnatomicalMixin,
    _AnatomicalIntegrationMixin,
    BaseTrainer,
):
    """Orchestrates ReID model training.

    Supports softmax (cross-entropy with label smoothing) and triplet loss
    with optional center loss, matching the existing backbone forward()
    contracts.
    """

    MEMORY_CLEAR_THRESHOLD = 0.90

    @classmethod
    def from_config(cls, config: ReIDTrainConfig) -> "ReIDTrainer":
        """Construct a trainer from the typed nested configuration surface."""
        return cls(**config.to_trainer_kwargs())

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        data_dir: str,
        *,
        loss_type: str = "triplet",
        preprocess: str = "resize",
        img_size: Tuple[int, int] = (256, 128),
        batch_size: int = 64,
        lr: float = 3.5e-4,
        weight_decay: float = 5e-4,
        epochs: int = 120,
        warmup_epochs: int = 10,
        eval_interval: int = 10,
        p: int = 16,
        k: int = 4,
        source_balance: str = "",
        pk_steps_per_epoch: int = 0,
        camera_aware_sampler: bool = False,
        margin: float = 0.3,
        label_smooth: float = 0.1,
        classifier_loss: str = "ce",
        triplet_soft_margin: Optional[bool] = None,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.5,
        cosface_scale: float = 30.0,
        cosface_margin: float = 0.35,
        center_loss_weight: float = 5e-4,
        id_loss_weight: float = 1.0,
        metric_loss_weight: float = 1.0,
        adasp_loss_weight: float = 0.0,
        adasp_temperature: float = 0.04,
        adasp_scale: float = 0.1,
        coarse_branch_ce_weight: float = 1.0,
        fine_branch_ce_weight: float = 1.0,
        part_relation_weight: float = 0.0,
        part_to_global_weight: float = 0.0,
        part_relation_teacher_momentum: float = 0.999,
        part_relation_temperature: float = 0.07,
        compact_metric_loss_weight: float = 1.0,
        compact_cosine_distill_weight: float = 1.0,
        compact_pairwise_distill_weight: float = 1.0,
        csmm_loss_weight: float = 0.0,
        csmm_margin: float = 0.10,
        csmm_temperature: float = 0.05,
        csmm_topk_negatives: int = 8,
        csmm_start_epoch: int = 20,
        csmm_ramp_end_epoch: int = 40,
        treeboost_loss_weight: float = 0.0,
        treeboost_coarse_coefficient: float = 1.0,
        treeboost_fine_coefficient: float = 1.0,
        treeboost_node_coefficient: float = 0.25,
        treeboost_regression_coefficient: float = 0.10,
        treeboost_difficulty_floor: float = 0.25,
        treeboost_regression_tolerance: float = 0.02,
        treeboost_temperature: float = 0.05,
        treeboost_start_epoch: int = 30,
        treeboost_ramp_end_epoch: int = 60,
        global_ap_loss_weight: float = 0.0,
        global_ap_temperature: float = 0.05,
        global_ap_topk: int = 500,
        global_ap_memory_size: int = 16384,
        global_ap_momentum: float = 0.2,
        global_ap_max_age: int = 0,
        global_ap_start_epoch: int = 20,
        global_ap_ramp_end_epoch: int = 50,
        global_ap_decay_start_epoch: int = 130,
        global_ap_decay_end_epoch: int = 170,
        hpgrd_cache_dir: Optional[str] = None,
        hpgrd_global_weight: float = 0.0,
        hpgrd_part_weight: float = 0.0,
        hpgrd_background_weight: float = 0.0,
        hpgrd_part_drop_weight: float = 0.0,
        hpgrd_part_drop_probability: float = 0.0,
        hpgrd_gradient_fraction: float = 0.30,
        hpgrd_min_confidence: float = 0.05,
        early_id_loss_weight: float = 0.0,
        early_id_loss_epochs: int = 0,
        center_loss_ramp_start_epoch: int = 0,
        center_loss_ramp_end_epoch: int = 0,
        aux_ce_weight: float = 1.0,
        aux_ce_drop_epoch: int = 0,
        branch_loss_agg: str = "mean",
        eta_min: float = 1e-7,
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        device: str = "cpu",
        project: str = "runs/reid_train",
        name: str = "exp",
        num_workers: int = 4,
        seed: int = 0,
        deterministic: bool = True,
        eval_datasets: Optional[List[str]] = None,
        data_specs: Optional[List[dict[str, Any]]] = None,
        ema_decay: Optional[float] = None,
        gaussian_blur: bool = False,
        random_grayscale: float = 0.0,
        color_jitter: bool = False,
        random_erasing: float = 0.5,
        random_patch: bool = True,
        random_crop_scale: float = 1.05,
        color_augmentation: bool = True,
        background_mosaic: bool = False,
        background_mosaic_mask_dir: Optional[str] = None,
        background_mosaic_probability: float = 0.3,
        background_mosaic_start_epoch: int = 10,
        background_mosaic_ramp_end_epoch: int = 30,
        background_mosaic_min_foreground_ratio: float = 0.2,
        background_mosaic_max_foreground_ratio: float = 0.9,
        background_mosaic_feather: float = 1.5,
        background_mosaic_dilation: int = 2,
        background_mosaic_occluder_probability: float = 0.0,
        background_mosaic_occluder_min_area: float = 0.05,
        background_mosaic_occluder_max_area: float = 0.20,
        same_id_part_mosaic: bool = False,
        same_id_part_mosaic_probability: float = 0.35,
        same_id_part_mosaic_max_regions: int = 2,
        same_id_part_mosaic_min_area: float = 0.15,
        same_id_part_mosaic_max_area: float = 0.40,
        same_id_part_mosaic_boundary_jitter: float = 0.05,
        same_id_part_mosaic_cross_camera_rate: float = 1.0,
        same_id_part_mosaic_min_unaltered: float = 0.5,
        pav_mosaic: bool = False,
        pav_metadata_dir: Optional[str] = None,
        pav_mosaic_probability: float = 0.25,
        pav_mosaic_max_parts: int = 3,
        pav_mosaic_max_foreground_replacement: float = 0.45,
        pav_mosaic_cross_camera_rate: float = 0.8,
        pav_mosaic_different_pose_rate: float = 0.5,
        pav_mosaic_min_keypoint_confidence: float = 0.5,
        pav_mosaic_min_unaltered: float = 0.5,
        pav_mosaic_warmup_epochs: int = 40,
        pav_mosaic_decay_start_epoch: int = 170,
        pav_mosaic_final_probability_scale: float = 0.5,
        pav_consistency_weight: float = 0.0,
        clean_student_consistency_weight: float = 0.0,
        anatomical_auxiliary: bool = False,
        anatomical_metadata_dir: Optional[str] = None,
        anatomical_person_mask_dir: Optional[str] = None,
        anatomical_min_keypoint_confidence: float = 0.5,
        anatomical_token_dim: int = 128,
        anatomical_distill_weight: float = 0.20,
        anatomical_attention_weight: float = 0.10,
        anatomical_foreground_weight: float = 0.15,
        anatomical_semantic_part_weight: float = 0.0,
        anatomical_visibility_weight: float = 0.05,
        anatomical_contrastive_weight: float = 0.10,
        anatomical_descriptor_distill_weight: float = 0.0,
        anatomical_branch_distill_weight: float = 0.0,
        anatomical_branch_global_coefficient: float = 0.20,
        anatomical_branch_coarse_coefficient: float = 0.30,
        anatomical_branch_fine_coefficient: float = 0.50,
        anatomical_pose_teacher_weight: float = 0.0,
        anatomical_query_distill_weight: float = 0.0,
        anatomical_query_relational_distill_weight: float = 0.0,
        anatomical_query_diversity_weight: float = 0.0,
        anatomical_query_diversity_margin: float = 0.10,
        anatomical_part_triplet_weight: float = 0.0,
        anatomical_target_type: str = DEFAULT_ANATOMICAL_TARGET_TYPE,
        anatomical_teacher_momentum: float = 0.99,
        anatomical_multiscale: bool = False,
        anatomical_accessory_query: bool = False,
        anatomical_deployment: bool = False,
        anatomical_deployment_dim: int = 64,
        anatomical_deployment_alpha: float = 0.25,
        anatomical_deployment_id_weight: float = 0.25,
        anatomical_deployment_metric_weight: float = 0.10,
        anatomical_local_scale_weight: float = 0.60,
        anatomical_fine_scale_weight: float = 0.40,
        anatomical_cross_scale_weight: float = 0.05,
        anatomical_pose_only_reliability: float = 0.35,
        anatomical_min_effective_coverage: float = 0.0,
        anatomical_student_start_epoch: int = 0,
        anatomical_student_ramp_end_epoch: int = 0,
        anatomical_query_start_epoch: int = 20,
        anatomical_query_ramp_end_epoch: int = 50,
        anatomical_fine_start_epoch: int = 0,
        anatomical_fine_ramp_end_epoch: int = 0,
        anatomical_decay_start_epoch: int = 0,
        anatomical_decay_end_epoch: int = 0,
        anatomical_temperature: float = 0.07,
        flip_tta: Optional[bool] = None,
        resume: Optional[str] = None,
        metric_feature: str = "auto",
        inference_feature: str = "concat_bn",
        feature_fusion: str = "last3",
        pyramid_resize_mode: str = "bilinear",
        spatial_conv_mode: str = "standard",
        post_fusion_mixer: str = "none",
        post_fusion_mixer_reduction: int = 4,
        post_fusion_mixer_kernel: tuple[int, int] = (5, 3),
        post_fusion_mixer_gamma_init: float = 0.0,
        feat_dim: int = 512,
        neck_dim: int = 512,
        drop_path_rate: float = 0.1,
        timm_model_name: str = "",
        timm_head_mode: str = "pooled",
        mobilenetv4_last_stride: int = 2,
        mobilenetv4_neck_mode: str = "cnn",
        attention_window_layout: str = "legacy",
        attention_bias: str = "absolute",
        interpolate_pretrained_attention_bias: bool = False,
        attention_mask: bool = False,
        attention_shift: bool = False,
        stage3_global: bool = False,
        stage3_downsample: bool = False,
        stage2_width_merge_after: int = 0,
        stage2_mlp_ratio: float = 4.0,
        stage3_mlp_ratio: float = 4.0,
        stage2_depth: int = 6,
        stage3_depth: int = 2,
        width_first_hierarchy: bool = False,
        identity_registers: bool = False,
        identity_register_count: int = 4,
        identity_register_dim: int = 128,
        identity_register_num_heads: int = 4,
        identity_register_dropout: float = 0.10,
        identity_register_gate_init: float = 0.0,
        identity_register_diversity_weight: float = 0.0,
        identity_register_diversity_margin: float = 0.10,
        native_branch_widths: bool = False,
        fine_map_dim: int = 0,
        compact_deployment_head: bool = False,
        vit_lr_profile: str = "layer_decay",
        layer_decay: float = 0.95,
        backbone_lr_mult: float = 1.0,
        backbone_freeze_epochs: int = 0,
        gradual_unfreeze: bool = False,
        gradual_unfreeze_head_epochs: int = 5,
        gradual_unfreeze_stage_epochs: int = 10,
        gradual_unfreeze_backbone_lr_mult: float = 0.1,
        gradual_unfreeze_backbone_lr_epochs: int = 5,
        branch_aware_metric: bool = False,
        branch_metric_part_weight: float = 0.5,
        scale_balanced_branches: bool = False,
        multilevel_suppression: bool = False,
        multilevel_suppression_ratio: float = 0.15,
        multilevel_suppression_loss_weight: float = 0.20,
        multilevel_suppression_start_epoch: int = 20,
        multilevel_suppression_ramp_end_epoch: int = 50,
        multilevel_suppression_decay_start_epoch: int = 140,
        multilevel_suppression_decay_end_epoch: int = 170,
        hierarchical_branch_attention: bool = False,
        branch_attention_token_dim: int = 96,
        branch_attention_num_heads: int = 4,
        branch_attention_num_layers: int = 1,
        branch_attention_mlp_ratio: float = 2.0,
        branch_attention_dropout: float = 0.0,
        branch_set_attention: bool = False,
        branch_set_attention_token_dim: int = 128,
        branch_set_attention_num_heads: int = 4,
        branch_set_attention_num_layers: int = 1,
        branch_set_attention_mlp_ratio: float = 2.0,
        branch_set_attention_dropout: float = 0.0,
        multiscale_query_decoder: bool = False,
        query_decoder_dim: int = 128,
        query_decoder_num_heads: int = 4,
        query_decoder_num_layers: int = 1,
        query_decoder_mlp_ratio: float = 2.0,
        query_decoder_dropout: float = 0.0,
        hierarchical_late_interaction: bool = False,
        late_interaction_dim: int = 128,
        late_interaction_num_heads: int = 4,
        late_interaction_num_layers: int = 1,
        late_interaction_sinkhorn_iters: int = 5,
        late_interaction_null_tokens: int = 1,
        late_interaction_negative_identities: int = 16,
        late_interaction_rerank_topk: int = 100,
        late_interaction_base_score_init: float = 0.9,
        late_interaction_loss_weight: float = 0.20,
        late_interaction_distill_weight: float = 0.05,
        late_interaction_temperature: float = 0.07,
        late_interaction_start_epoch: int = 20,
        late_interaction_ramp_end_epoch: int = 50,
        mcpt_mode: str = "none",
        mcpt_hidden_dim: int = 64,
        mcpt_max_displacement: float = 0.15,
        mcpt_smoothness_weight: float = 0.01,
        mcpt_identity_weight: float = 0.02,
        mcpt_identity_decay_epoch: int = 60,
        mcpt_lr_multiplier: float = 2.0,
        mcpt_start_epoch: int = 10,
        mcpt_ramp_end_epoch: int = 40,
        mcpt_disabled_eval: bool = False,
        jpm: bool = False,
        jpm_num_groups: int = 4,
        jpm_shift: int = 5,
        jpm_token_dim: int = 96,
        jpm_num_heads: int = 4,
        jpm_mlp_ratio: float = 4.0,
        jpm_dropout: float = 0.0,
        jpm_id_loss_weight: float = 1.0,
        jpm_metric_loss_weight: float = 1.0,
        evidence_num_roles: int = 8,
        evidence_alignment_loss_weight: float = 0.0,
        evidence_alignment_margin: float = 0.2,
        evidence_sinkhorn_iters: int = 20,
        evidence_sinkhorn_temperature: float = 0.1,
        evidence_rerank_topk: int = 100,
        evidence_null_loss_weight: float = 0.0,
        evidence_diversity_loss_weight: float = 0.0,
        head_pool: str = "avg",
        head_parts: tuple[int, ...] = (1, 2),
        head_type: str = "standard",
        multiscale_channel_alpha: float = 0.5,
        body_slot_mode: str = "recurrent_read",
        body_slot_alpha: float = 0.45,
        body_slot_visibility_floor: float = 0.05,
        part_pooling: str = "stripes",
        num_part_tokens: int = 4,
        decouple_patterns: bool = False,
        pattern_adapter_dim: int = 128,
        stripe_visibility: bool = False,
        drop_global_aux: bool = False,
        drop_global_aux_ratio: float = 0.25,
        reid_adapter_stages: tuple[int, ...] = (),
        reid_adapter_reduction: int = 4,
        reid_adapter_suppression_tau: float = 0.0,
        head_warmup_epochs: int = 0,
        head_warmup_lr_mult: float = 2.0,
        explicit_hparams: Iterable[str] | None = None,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_dir = str(data_dir)
        self.data_specs = tuple(self._normalize_data_spec(spec) for spec in (data_specs or ()))
        self._data_roots_by_name = {self._dataset_lookup_key(spec["name"]): spec["root"] for spec in self.data_specs}
        self.loss_type = loss_type.lower()
        self.preprocess = preprocess
        self.img_size = tuple(int(value) for value in img_size)
        self.eval_batch_size = int(batch_size)
        self.batch_size = self.eval_batch_size  # Backward-compatible alias.
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.eval_interval = int(eval_interval)
        self.p = int(p)
        self.k = int(k)
        self.source_balance = str(source_balance or "").strip()
        self.source_balance_groups = parse_source_balance(self.source_balance) if self.source_balance else ()
        self.pk_steps_per_epoch = int(pk_steps_per_epoch)
        self.camera_aware_sampler = bool(camera_aware_sampler)
        self.margin = float(margin)
        self.label_smooth = float(label_smooth)
        self.classifier_loss = classifier_loss.lower()
        self.triplet_soft_margin = triplet_soft_margin
        self.arcface_scale = arcface_scale
        self.arcface_margin = arcface_margin
        self.cosface_scale = cosface_scale
        self.cosface_margin = cosface_margin
        self.center_loss_weight = float(center_loss_weight)
        self.id_loss_weight = float(id_loss_weight)
        self.metric_loss_weight = float(metric_loss_weight)
        self.adasp_loss_weight = float(adasp_loss_weight)
        self.adasp_temperature = float(adasp_temperature)
        self.adasp_scale = float(adasp_scale)
        self.coarse_branch_ce_weight = float(coarse_branch_ce_weight)
        self.fine_branch_ce_weight = float(fine_branch_ce_weight)
        self.part_relation_weight = float(part_relation_weight)
        self.part_to_global_weight = float(part_to_global_weight)
        self.part_relation_teacher_momentum = float(part_relation_teacher_momentum)
        self.part_relation_temperature = float(part_relation_temperature)
        self.return_auxiliary_features = self.part_relation_weight > 0 or self.part_to_global_weight > 0
        self.compact_metric_loss_weight = float(compact_metric_loss_weight)
        self.compact_cosine_distill_weight = float(compact_cosine_distill_weight)
        self.compact_pairwise_distill_weight = float(compact_pairwise_distill_weight)
        self.csmm_loss_weight = float(csmm_loss_weight)
        self.csmm_margin = float(csmm_margin)
        self.csmm_temperature = float(csmm_temperature)
        self.csmm_topk_negatives = int(csmm_topk_negatives)
        self.csmm_start_epoch = int(csmm_start_epoch)
        self.csmm_ramp_end_epoch = int(csmm_ramp_end_epoch)
        self.treeboost_loss_weight = float(treeboost_loss_weight)
        self.treeboost_coarse_coefficient = float(treeboost_coarse_coefficient)
        self.treeboost_fine_coefficient = float(treeboost_fine_coefficient)
        self.treeboost_node_coefficient = float(treeboost_node_coefficient)
        self.treeboost_regression_coefficient = float(treeboost_regression_coefficient)
        self.treeboost_difficulty_floor = float(treeboost_difficulty_floor)
        self.treeboost_regression_tolerance = float(treeboost_regression_tolerance)
        self.treeboost_temperature = float(treeboost_temperature)
        self.treeboost_start_epoch = int(treeboost_start_epoch)
        self.treeboost_ramp_end_epoch = int(treeboost_ramp_end_epoch)
        self.global_ap_loss_weight = float(global_ap_loss_weight)
        self.global_ap_temperature = float(global_ap_temperature)
        self.global_ap_topk = int(global_ap_topk)
        self.global_ap_memory_size = int(global_ap_memory_size)
        self.global_ap_momentum = float(global_ap_momentum)
        self.global_ap_max_age = int(global_ap_max_age)
        self.global_ap_start_epoch = int(global_ap_start_epoch)
        self.global_ap_ramp_end_epoch = int(global_ap_ramp_end_epoch)
        self.global_ap_decay_start_epoch = int(global_ap_decay_start_epoch)
        self.global_ap_decay_end_epoch = int(global_ap_decay_end_epoch)
        self.hpgrd_cache_dir = None if hpgrd_cache_dir is None else str(hpgrd_cache_dir)
        self.hpgrd_global_weight = float(hpgrd_global_weight)
        self.hpgrd_part_weight = float(hpgrd_part_weight)
        self.hpgrd_background_weight = float(hpgrd_background_weight)
        self.hpgrd_part_drop_weight = float(hpgrd_part_drop_weight)
        self.hpgrd_part_drop_probability = float(hpgrd_part_drop_probability)
        self.hpgrd_gradient_fraction = float(hpgrd_gradient_fraction)
        self.hpgrd_min_confidence = float(hpgrd_min_confidence)
        self._global_ap = None
        self._privileged_graph_cache = None
        self._privileged_graph_loss = None
        self._hpgrd_manifest_sha256 = None
        self._retrieval_dataset_sha256 = None
        self.early_id_loss_weight = float(early_id_loss_weight)
        self.early_id_loss_epochs = int(early_id_loss_epochs)
        self.center_loss_ramp_start_epoch = int(center_loss_ramp_start_epoch)
        self.center_loss_ramp_end_epoch = int(center_loss_ramp_end_epoch)
        self.aux_ce_weight = float(aux_ce_weight)
        self.aux_ce_drop_epoch = int(aux_ce_drop_epoch)
        self.branch_loss_agg = branch_loss_agg.lower()
        if self.branch_loss_agg not in {"mean", "sum"}:
            raise ValueError("branch_loss_agg must be 'mean' or 'sum'")
        self.eta_min = float(eta_min)
        self.pretrained = pretrained
        self.pretrained_weights = (
            None if pretrained_weights is None else str(pretrained_weights)
        )
        self.device = torch.device(device)
        self.project = Path(project)
        self.name = name
        self.requested_num_workers = int(num_workers)
        if self.requested_num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.num_workers = self.requested_num_workers
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.eval_datasets = eval_datasets or []
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        self.gaussian_blur = gaussian_blur
        self.random_grayscale = random_grayscale
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing
        self.random_patch = random_patch
        self.random_crop_scale = float(random_crop_scale)
        self.color_augmentation = color_augmentation
        self.background_mosaic = bool(background_mosaic)
        self.background_mosaic_mask_dir = (
            None if background_mosaic_mask_dir is None else str(background_mosaic_mask_dir)
        )
        self.background_mosaic_probability = float(background_mosaic_probability)
        self.background_mosaic_start_epoch = int(background_mosaic_start_epoch)
        self.background_mosaic_ramp_end_epoch = int(background_mosaic_ramp_end_epoch)
        self.background_mosaic_min_foreground_ratio = float(background_mosaic_min_foreground_ratio)
        self.background_mosaic_max_foreground_ratio = float(background_mosaic_max_foreground_ratio)
        self.background_mosaic_feather = float(background_mosaic_feather)
        self.background_mosaic_dilation = int(background_mosaic_dilation)
        self.background_mosaic_occluder_probability = float(background_mosaic_occluder_probability)
        self.background_mosaic_occluder_min_area = float(background_mosaic_occluder_min_area)
        self.background_mosaic_occluder_max_area = float(background_mosaic_occluder_max_area)
        self.same_id_part_mosaic = bool(same_id_part_mosaic)
        self.same_id_part_mosaic_probability = float(same_id_part_mosaic_probability)
        self.same_id_part_mosaic_max_regions = int(same_id_part_mosaic_max_regions)
        self.same_id_part_mosaic_min_area = float(same_id_part_mosaic_min_area)
        self.same_id_part_mosaic_max_area = float(same_id_part_mosaic_max_area)
        self.same_id_part_mosaic_boundary_jitter = float(same_id_part_mosaic_boundary_jitter)
        self.same_id_part_mosaic_cross_camera_rate = float(same_id_part_mosaic_cross_camera_rate)
        self.same_id_part_mosaic_min_unaltered = float(same_id_part_mosaic_min_unaltered)
        self.pav_mosaic = bool(pav_mosaic)
        self.pav_metadata_dir = None if pav_metadata_dir is None else str(pav_metadata_dir)
        self.pav_mosaic_probability = float(pav_mosaic_probability)
        self.pav_mosaic_max_parts = int(pav_mosaic_max_parts)
        self.pav_mosaic_max_foreground_replacement = float(pav_mosaic_max_foreground_replacement)
        self.pav_mosaic_cross_camera_rate = float(pav_mosaic_cross_camera_rate)
        self.pav_mosaic_different_pose_rate = float(pav_mosaic_different_pose_rate)
        self.pav_mosaic_min_keypoint_confidence = float(pav_mosaic_min_keypoint_confidence)
        self.pav_mosaic_min_unaltered = float(pav_mosaic_min_unaltered)
        self.pav_mosaic_warmup_epochs = int(pav_mosaic_warmup_epochs)
        self.pav_mosaic_decay_start_epoch = int(pav_mosaic_decay_start_epoch)
        self.pav_mosaic_final_probability_scale = float(pav_mosaic_final_probability_scale)
        self.pav_consistency_weight = float(pav_consistency_weight)
        self.clean_student_consistency_weight = float(clean_student_consistency_weight)
        self.anatomical_auxiliary = bool(anatomical_auxiliary)
        self.anatomical_metadata_dir = None if anatomical_metadata_dir is None else str(anatomical_metadata_dir)
        self.anatomical_person_mask_dir = (
            None if anatomical_person_mask_dir is None else str(anatomical_person_mask_dir)
        )
        self.anatomical_min_keypoint_confidence = float(anatomical_min_keypoint_confidence)
        self.anatomical_token_dim = int(anatomical_token_dim)
        self.anatomical_distill_weight = float(anatomical_distill_weight)
        self.anatomical_attention_weight = float(anatomical_attention_weight)
        self.anatomical_foreground_weight = float(anatomical_foreground_weight)
        self.anatomical_semantic_part_weight = float(anatomical_semantic_part_weight)
        self.anatomical_visibility_weight = float(anatomical_visibility_weight)
        self.anatomical_contrastive_weight = float(anatomical_contrastive_weight)
        self.anatomical_descriptor_distill_weight = float(anatomical_descriptor_distill_weight)
        self.anatomical_branch_distill_weight = float(anatomical_branch_distill_weight)
        self.anatomical_branch_global_coefficient = float(anatomical_branch_global_coefficient)
        self.anatomical_branch_coarse_coefficient = float(anatomical_branch_coarse_coefficient)
        self.anatomical_branch_fine_coefficient = float(anatomical_branch_fine_coefficient)
        self.anatomical_pose_teacher_weight = float(anatomical_pose_teacher_weight)
        self.anatomical_query_distill_weight = float(anatomical_query_distill_weight)
        self.anatomical_query_relational_distill_weight = float(anatomical_query_relational_distill_weight)
        self.anatomical_query_diversity_weight = float(anatomical_query_diversity_weight)
        self.anatomical_query_diversity_margin = float(anatomical_query_diversity_margin)
        self.anatomical_part_triplet_weight = float(anatomical_part_triplet_weight)
        self.anatomical_target_type = str(anatomical_target_type).lower()
        self.anatomical_teacher_momentum = float(anatomical_teacher_momentum)
        self.anatomical_multiscale = bool(anatomical_multiscale)
        self.anatomical_accessory_query = bool(anatomical_accessory_query)
        self.anatomical_deployment = bool(anatomical_deployment)
        self.anatomical_deployment_dim = int(anatomical_deployment_dim)
        self.anatomical_deployment_alpha = float(anatomical_deployment_alpha)
        self.anatomical_deployment_id_weight = float(anatomical_deployment_id_weight)
        self.anatomical_deployment_metric_weight = float(anatomical_deployment_metric_weight)
        self.anatomical_local_scale_weight = float(anatomical_local_scale_weight)
        self.anatomical_fine_scale_weight = float(anatomical_fine_scale_weight)
        self.anatomical_cross_scale_weight = float(anatomical_cross_scale_weight)
        self.anatomical_pose_only_reliability = float(anatomical_pose_only_reliability)
        self.anatomical_min_effective_coverage = float(
            anatomical_min_effective_coverage
        )
        self.anatomical_student_start_epoch = int(anatomical_student_start_epoch)
        self.anatomical_student_ramp_end_epoch = int(anatomical_student_ramp_end_epoch)
        self.anatomical_query_start_epoch = int(anatomical_query_start_epoch)
        self.anatomical_query_ramp_end_epoch = int(anatomical_query_ramp_end_epoch)
        self.anatomical_fine_start_epoch = int(anatomical_fine_start_epoch)
        self.anatomical_fine_ramp_end_epoch = int(anatomical_fine_ramp_end_epoch)
        self.anatomical_decay_start_epoch = int(anatomical_decay_start_epoch)
        self.anatomical_decay_end_epoch = int(anatomical_decay_end_epoch)
        self.anatomical_temperature = float(anatomical_temperature)
        self.flip_tta = flip_tta
        self.resume = resume
        self.metric_feature = str(metric_feature).lower()
        self.inference_feature = str(inference_feature).lower()
        self.feature_fusion = str(feature_fusion).lower()
        self.pyramid_resize_mode = str(pyramid_resize_mode).lower()
        self.spatial_conv_mode = str(spatial_conv_mode).lower()
        self.post_fusion_mixer = self._normalize_post_fusion_mixer(post_fusion_mixer)
        self.post_fusion_mixer_reduction = int(post_fusion_mixer_reduction)
        self.post_fusion_mixer_kernel = self._normalize_int_pair(post_fusion_mixer_kernel)
        self.post_fusion_mixer_gamma_init = float(post_fusion_mixer_gamma_init)
        self.feat_dim = int(feat_dim)
        self.neck_dim = int(neck_dim)
        self.drop_path_rate = float(drop_path_rate)
        self.timm_model_name = str(timm_model_name).strip()
        self.timm_head_mode = str(timm_head_mode).lower()
        self.mobilenetv4_last_stride = int(mobilenetv4_last_stride)
        self.mobilenetv4_neck_mode = str(mobilenetv4_neck_mode).lower()
        self.attention_window_layout = str(attention_window_layout).lower()
        self.attention_bias = str(attention_bias).lower()
        self.interpolate_pretrained_attention_bias = bool(interpolate_pretrained_attention_bias)
        self.attention_mask = bool(attention_mask)
        self.attention_shift = bool(attention_shift)
        self.stage3_global = bool(stage3_global)
        self.stage3_downsample = bool(stage3_downsample)
        self.stage2_width_merge_after = int(stage2_width_merge_after)
        self.stage2_mlp_ratio = float(stage2_mlp_ratio)
        self.stage3_mlp_ratio = float(stage3_mlp_ratio)
        self.stage2_depth = int(stage2_depth)
        self.stage3_depth = int(stage3_depth)
        self.width_first_hierarchy = bool(width_first_hierarchy)
        self.identity_registers = bool(identity_registers)
        self.identity_register_count = int(identity_register_count)
        self.identity_register_dim = int(identity_register_dim)
        self.identity_register_num_heads = int(identity_register_num_heads)
        self.identity_register_dropout = float(identity_register_dropout)
        self.identity_register_gate_init = float(identity_register_gate_init)
        self.identity_register_diversity_weight = float(identity_register_diversity_weight)
        self.identity_register_diversity_margin = float(identity_register_diversity_margin)
        self.native_branch_widths = bool(native_branch_widths)
        self.fine_map_dim = int(fine_map_dim)
        self.compact_deployment_head = bool(compact_deployment_head)
        self.vit_lr_profile = str(vit_lr_profile).lower()
        self.layer_decay = float(layer_decay)
        self.backbone_lr_mult = float(backbone_lr_mult)
        self.backbone_freeze_epochs = int(backbone_freeze_epochs)
        self.gradual_unfreeze = bool(gradual_unfreeze)
        self.gradual_unfreeze_head_epochs = int(gradual_unfreeze_head_epochs)
        self.gradual_unfreeze_stage_epochs = int(gradual_unfreeze_stage_epochs)
        self.gradual_unfreeze_backbone_lr_mult = float(gradual_unfreeze_backbone_lr_mult)
        self.gradual_unfreeze_backbone_lr_epochs = int(gradual_unfreeze_backbone_lr_epochs)
        self.branch_aware_metric = bool(branch_aware_metric)
        self.branch_metric_part_weight = float(branch_metric_part_weight)
        self.scale_balanced_branches = bool(scale_balanced_branches)
        self.multilevel_suppression = bool(multilevel_suppression)
        self.multilevel_suppression_ratio = float(multilevel_suppression_ratio)
        self.multilevel_suppression_loss_weight = float(
            multilevel_suppression_loss_weight
        )
        self.multilevel_suppression_start_epoch = int(
            multilevel_suppression_start_epoch
        )
        self.multilevel_suppression_ramp_end_epoch = int(
            multilevel_suppression_ramp_end_epoch
        )
        self.multilevel_suppression_decay_start_epoch = int(
            multilevel_suppression_decay_start_epoch
        )
        self.multilevel_suppression_decay_end_epoch = int(
            multilevel_suppression_decay_end_epoch
        )
        self.hierarchical_branch_attention = bool(hierarchical_branch_attention)
        self.branch_attention_token_dim = int(branch_attention_token_dim)
        self.branch_attention_num_heads = int(branch_attention_num_heads)
        self.branch_attention_num_layers = int(branch_attention_num_layers)
        self.branch_attention_mlp_ratio = float(branch_attention_mlp_ratio)
        self.branch_attention_dropout = float(branch_attention_dropout)
        self.branch_set_attention = bool(branch_set_attention)
        self.branch_set_attention_token_dim = int(branch_set_attention_token_dim)
        self.branch_set_attention_num_heads = int(branch_set_attention_num_heads)
        self.branch_set_attention_num_layers = int(branch_set_attention_num_layers)
        self.branch_set_attention_mlp_ratio = float(branch_set_attention_mlp_ratio)
        self.branch_set_attention_dropout = float(branch_set_attention_dropout)
        self.multiscale_query_decoder = bool(multiscale_query_decoder)
        self.query_decoder_dim = int(query_decoder_dim)
        self.query_decoder_num_heads = int(query_decoder_num_heads)
        self.query_decoder_num_layers = int(query_decoder_num_layers)
        self.query_decoder_mlp_ratio = float(query_decoder_mlp_ratio)
        self.query_decoder_dropout = float(query_decoder_dropout)
        self.hierarchical_late_interaction = bool(hierarchical_late_interaction)
        self.late_interaction_dim = int(late_interaction_dim)
        self.late_interaction_num_heads = int(late_interaction_num_heads)
        self.late_interaction_num_layers = int(late_interaction_num_layers)
        self.late_interaction_sinkhorn_iters = int(late_interaction_sinkhorn_iters)
        self.late_interaction_null_tokens = int(late_interaction_null_tokens)
        self.late_interaction_negative_identities = int(late_interaction_negative_identities)
        self.late_interaction_rerank_topk = int(late_interaction_rerank_topk)
        self.late_interaction_base_score_init = float(late_interaction_base_score_init)
        self.late_interaction_loss_weight = float(late_interaction_loss_weight)
        self.late_interaction_distill_weight = float(late_interaction_distill_weight)
        self.late_interaction_temperature = float(late_interaction_temperature)
        self.late_interaction_start_epoch = int(late_interaction_start_epoch)
        self.late_interaction_ramp_end_epoch = int(late_interaction_ramp_end_epoch)
        self.mcpt_mode = str(mcpt_mode).lower()
        self.mcpt_hidden_dim = int(mcpt_hidden_dim)
        self.mcpt_max_displacement = float(mcpt_max_displacement)
        self.mcpt_smoothness_weight = float(mcpt_smoothness_weight)
        self.mcpt_identity_weight = float(mcpt_identity_weight)
        self.mcpt_identity_decay_epoch = int(mcpt_identity_decay_epoch)
        self.mcpt_lr_multiplier = float(mcpt_lr_multiplier)
        self.mcpt_start_epoch = int(mcpt_start_epoch)
        self.mcpt_ramp_end_epoch = int(mcpt_ramp_end_epoch)
        self.mcpt_disabled_eval = bool(mcpt_disabled_eval)
        self.jpm = bool(jpm)
        self.jpm_num_groups = int(jpm_num_groups)
        self.jpm_shift = int(jpm_shift)
        self.jpm_token_dim = int(jpm_token_dim)
        self.jpm_num_heads = int(jpm_num_heads)
        self.jpm_mlp_ratio = float(jpm_mlp_ratio)
        self.jpm_dropout = float(jpm_dropout)
        self.jpm_id_loss_weight = float(jpm_id_loss_weight)
        self.jpm_metric_loss_weight = float(jpm_metric_loss_weight)
        self.evidence_num_roles = int(evidence_num_roles)
        self.evidence_alignment_loss_weight = float(evidence_alignment_loss_weight)
        self.evidence_alignment_margin = float(evidence_alignment_margin)
        self.evidence_sinkhorn_iters = int(evidence_sinkhorn_iters)
        self.evidence_sinkhorn_temperature = float(evidence_sinkhorn_temperature)
        self.evidence_rerank_topk = int(evidence_rerank_topk)
        self.evidence_null_loss_weight = float(evidence_null_loss_weight)
        self.evidence_diversity_loss_weight = float(evidence_diversity_loss_weight)
        self.head_pool = str(head_pool).lower()
        self.head_parts = self._normalize_head_parts(head_parts)
        self.head_type = str(head_type).lower()
        self.multiscale_channel_alpha = float(multiscale_channel_alpha)
        self.body_slot_mode = str(body_slot_mode).lower()
        self.body_slot_alpha = float(body_slot_alpha)
        self.body_slot_visibility_floor = float(body_slot_visibility_floor)
        self.part_pooling = str(part_pooling).lower()
        if self.part_pooling in {"soft_stripes", "overlapping_stripes"}:
            self.part_pooling = "overlap_stripes"
        if self.part_pooling in {"semantic", "semantic_tokens", "semantic_visibility"}:
            self.part_pooling = "semantic_parts"
        normalize_selector("part_pooling", self.part_pooling)
        self.num_part_tokens = int(num_part_tokens)
        if self.num_part_tokens < 1:
            raise ValueError("num_part_tokens must be positive")
        self.decouple_patterns = bool(decouple_patterns)
        self.pattern_adapter_dim = int(pattern_adapter_dim)
        if self.pattern_adapter_dim < 1:
            raise ValueError("pattern_adapter_dim must be positive")
        self.stripe_visibility = bool(stripe_visibility)
        self.drop_global_aux = bool(drop_global_aux)
        self.drop_global_aux_ratio = float(drop_global_aux_ratio)
        self.reid_adapter_stages = self._normalize_adapter_stages(reid_adapter_stages)
        self.reid_adapter_reduction = int(reid_adapter_reduction)
        self.reid_adapter_suppression_tau = float(reid_adapter_suppression_tau)
        self.head_warmup_epochs = int(head_warmup_epochs)
        self.head_warmup_lr_mult = float(head_warmup_lr_mult)
        self.explicit_hparams = set(explicit_hparams or ())
        self._validate_config()
        self._train_generator = torch.Generator()
        self._train_generator.manual_seed(self.seed)
        self.checkpoint_manager = CheckpointManager(
            metadata_factory=self._checkpoint_metadata,
            rng_state_factory=self._capture_rng_state,
            classifier_loss=self.classifier_loss,
        )
