"""Configuration validation for the ReID trainer."""

from __future__ import annotations

import math

from boxmot.reid.backbones.anatomical_registry import (
    ANATOMICAL_TARGET_TYPES,
    EMA_ANATOMICAL_TARGET_TYPES,
    SEMANTIC_ANATOMICAL_TARGET_TYPES,
    V8_ANATOMICAL_TARGET_TYPE,
)
from boxmot.reid.backbones.families.csl_tinyvit.fusion import (
    CSLTinyViTFeatureFusion,
)
from boxmot.reid.backbones.families.csl_tinyvit.transport import MCPT_MODES
from boxmot.reid.backbones.head_registry import (
    MULTI_BRANCH_HEAD_TYPES,
    get_reid_head_spec,
)
from boxmot.reid.backbones.option_registry import normalize_selector
from boxmot.reid.datasets.anatomical import (
    ANATOMICAL_CANONICAL_CELLS,
)
from boxmot.reid.training.ablation import resolve_csl_tinyvit_ablation
from boxmot.reid.training.augmentations import (
    augmentation_config_from_options,
    validate_augmentation_config,
)
from boxmot.reid.training.losses import (
    METRIC_LOSS_REGISTRY,
)

_CSL_TINYVIT_11M_MODELS = frozenset(
    {
        "csl_tinyvit_11m",
        "csl_tinyvit_11m_v20",
    }
)


class _ConfigurationMixin:
    def _validate_config(self) -> None:
        """Reject invalid or ambiguous training configurations before setup."""
        if self.loss_type not in {"softmax", *METRIC_LOSS_REGISTRY}:
            raise ValueError(
                f"Unsupported loss_type={self.loss_type!r}; expected one of "
                f"{sorted({'softmax', *METRIC_LOSS_REGISTRY})}"
            )
        if self.classifier_loss not in {"ce", "arcface", "cosface"}:
            raise ValueError("classifier_loss must be one of: ce, arcface, cosface")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.p <= 0 or self.k <= 0:
            raise ValueError("p and k must be positive")
        if self.pk_steps_per_epoch < 0:
            raise ValueError("pk_steps_per_epoch must be non-negative")
        if self.source_balance_groups and (self.pk_steps_per_epoch or self.camera_aware_sampler):
            raise ValueError("fixed-step and camera-aware sampling are not supported with source_balance")
        for group in self.source_balance_groups:
            if group.p <= 0 or group.k <= 0:
                raise ValueError("source_balance p and k values must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("batch_size (evaluation batch size) must be positive")
        if len(self.img_size) != 2 or any(value <= 0 for value in self.img_size):
            raise ValueError("img_size must contain two positive integers")
        if self.preprocess not in {"resize", "resize_pad"}:
            raise ValueError("preprocess must be one of: resize, resize_pad")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0 or self.eta_min < 0:
            raise ValueError("weight_decay and eta_min must be non-negative")
        if self.eta_min > self.lr:
            raise ValueError("eta_min must not exceed lr")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if not 0 <= self.label_smooth < 1:
            raise ValueError("label_smooth must satisfy 0 <= label_smooth < 1")
        for name in (
            "center_loss_weight",
            "id_loss_weight",
            "metric_loss_weight",
            "adasp_loss_weight",
            "adasp_scale",
            "coarse_branch_ce_weight",
            "fine_branch_ce_weight",
            "part_relation_weight",
            "part_to_global_weight",
            "compact_metric_loss_weight",
            "compact_cosine_distill_weight",
            "compact_pairwise_distill_weight",
            "csmm_loss_weight",
            "treeboost_loss_weight",
            "global_ap_loss_weight",
            "hpgrd_global_weight",
            "hpgrd_part_weight",
            "hpgrd_background_weight",
            "hpgrd_part_drop_weight",
            "treeboost_coarse_coefficient",
            "treeboost_fine_coefficient",
            "treeboost_node_coefficient",
            "treeboost_regression_coefficient",
            "treeboost_regression_tolerance",
            "late_interaction_loss_weight",
            "late_interaction_distill_weight",
            "early_id_loss_weight",
            "aux_ce_weight",
            "jpm_id_loss_weight",
            "jpm_metric_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.early_id_loss_epochs < 0 or self.early_id_loss_epochs > self.epochs:
            raise ValueError("early_id_loss_epochs must satisfy 0 <= value <= epochs")
        if self.center_loss_ramp_start_epoch < 0 or self.center_loss_ramp_end_epoch < 0:
            raise ValueError("center_loss_ramp_* epochs must be non-negative")
        if self.center_loss_ramp_end_epoch > 0:
            if self.center_loss_ramp_end_epoch <= self.center_loss_ramp_start_epoch:
                raise ValueError("center_loss_ramp_end_epoch must be > center_loss_ramp_start_epoch")
            if self.center_loss_ramp_end_epoch > self.epochs:
                raise ValueError("center_loss_ramp_end_epoch must be <= epochs")
        if self.csmm_margin < 0:
            raise ValueError("csmm_margin must be non-negative")
        if self.csmm_temperature <= 0:
            raise ValueError("csmm_temperature must be positive")
        if self.csmm_topk_negatives < 1:
            raise ValueError("csmm_topk_negatives must be positive")
        if self.csmm_loss_weight > 0:
            if not 0 <= self.csmm_start_epoch < self.csmm_ramp_end_epoch <= self.epochs:
                raise ValueError("Enabled CSMM requires 0 <= start epoch < ramp end epoch <= epochs")
        if not 0 <= self.treeboost_difficulty_floor <= 1:
            raise ValueError("treeboost_difficulty_floor must satisfy 0 <= value <= 1")
        if self.treeboost_temperature <= 0:
            raise ValueError("treeboost_temperature must be positive")
        if self.treeboost_loss_weight > 0:
            if not 0 <= self.treeboost_start_epoch < self.treeboost_ramp_end_epoch <= self.epochs:
                raise ValueError("Enabled TreeBoost-AP requires 0 <= start epoch < ramp end epoch <= epochs")
        if self.global_ap_temperature <= 0:
            raise ValueError("global_ap_temperature must be positive")
        if self.global_ap_topk < 1 or self.global_ap_memory_size < 1:
            raise ValueError("global_ap_topk and global_ap_memory_size must be positive")
        if self.global_ap_topk > self.global_ap_memory_size:
            raise ValueError("global_ap_topk must not exceed global_ap_memory_size")
        if not 0 <= self.global_ap_momentum < 1:
            raise ValueError("global_ap_momentum must satisfy 0 <= value < 1")
        if self.global_ap_max_age < 0:
            raise ValueError("global_ap_max_age must be non-negative")
        hpgrd_active = any(
            weight > 0
            for weight in (
                self.hpgrd_global_weight,
                self.hpgrd_part_weight,
                self.hpgrd_background_weight,
                self.hpgrd_part_drop_weight,
            )
        )
        if (self.global_ap_loss_weight > 0 or hpgrd_active) and not (
            0
            <= self.global_ap_start_epoch
            < self.global_ap_ramp_end_epoch
            <= self.global_ap_decay_start_epoch
            < self.global_ap_decay_end_epoch
            <= self.epochs
        ):
            raise ValueError("Enabled GlobalAP/HP-GRD requires 0 <= start < ramp <= decay_start < decay_end <= epochs")
        if self.global_ap_loss_weight > 0 and self.inference_feature != "norm_concat_bn":
            raise ValueError("GlobalAP must supervise inference_feature='norm_concat_bn'")
        if not 0 <= self.hpgrd_part_drop_probability <= 1:
            raise ValueError("hpgrd_part_drop_probability must satisfy 0 <= value <= 1")
        if not 0 < self.hpgrd_gradient_fraction <= 1:
            raise ValueError("hpgrd_gradient_fraction must satisfy 0 < value <= 1")
        if not 0 <= self.hpgrd_min_confidence <= 1:
            raise ValueError("hpgrd_min_confidence must satisfy 0 <= value <= 1")
        if hpgrd_active and not self.hpgrd_cache_dir:
            raise ValueError("HP-GRD losses require hpgrd_cache_dir")
        if hpgrd_active and self.inference_feature != "norm_concat_bn":
            raise ValueError("HP-GRD must supervise inference_feature='norm_concat_bn'")
        hpgrd_parts_active = self.hpgrd_part_weight > 0 or self.hpgrd_part_drop_weight > 0
        if hpgrd_parts_active:
            if not self.anatomical_metadata_dir:
                raise ValueError("HP-GRD part supervision requires anatomical_metadata_dir")
            if not self.model_name.startswith("csl_tinyvit_") or self.head_type != "standard":
                raise ValueError("HP-GRD fixed part pooling requires a standard CSL-TinyViT head")
        if self.hpgrd_part_drop_weight > 0 and self.hpgrd_part_drop_probability <= 0:
            raise ValueError("HP-GRD part-drop loss requires a positive part-drop probability")
        if self.hpgrd_background_weight > 0 and not self.background_mosaic:
            raise ValueError("HP-GRD background consistency requires background_mosaic")
        if self.aux_ce_drop_epoch < 0 or self.aux_ce_drop_epoch > self.epochs:
            raise ValueError("aux_ce_drop_epoch must satisfy 0 <= value <= epochs")
        if self.adasp_temperature <= 0:
            raise ValueError("adasp_temperature must be positive")
        if self.adasp_loss_weight > 0 and (self.p < 2 or self.k < 2):
            raise ValueError("AdaSP requires P >= 2 identities and K >= 2 instances")
        if not 0 <= self.coarse_branch_ce_weight <= 1:
            raise ValueError("coarse_branch_ce_weight must satisfy 0 <= value <= 1")
        if not 0 <= self.fine_branch_ce_weight <= 1:
            raise ValueError("fine_branch_ce_weight must satisfy 0 <= value <= 1")
        if not 0 <= self.part_relation_teacher_momentum < 1:
            raise ValueError("part_relation_teacher_momentum must satisfy 0 <= value < 1")
        if self.part_relation_temperature <= 0:
            raise ValueError("part_relation_temperature must be positive")
        part_relation_active = self.part_relation_weight > 0 or self.part_to_global_weight > 0
        if part_relation_active or self.coarse_branch_ce_weight != 1.0 or self.fine_branch_ce_weight != 1.0:
            if self.model_name not in {"csl_tinyvit_7m", "csl_tinyvit_7m_v20"}:
                raise ValueError("part-relation supervision is scoped to the csl_tinyvit_7m family")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("part-relation supervision requires the standard stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("part-relation supervision requires scale-balanced head_parts=(1, 2, 4)")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("part-relation supervision preserves raw_concat metric learning")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("part-relation supervision preserves norm_concat_bn inference")
            if self.anatomical_auxiliary or self.mcpt_mode != "none":
                raise ValueError("part-relation supervision must be isolated from pose and MCPT")
        metric_active = self.loss_type != "softmax" and self.metric_loss_weight > 0
        center_active = self.loss_type != "ms" and self.center_loss_weight > 0
        if (
            self.id_loss_weight == 0
            and not metric_active
            and self.adasp_loss_weight == 0
            and self.part_relation_weight == 0
            and self.part_to_global_weight == 0
            and not center_active
            and self.csmm_loss_weight == 0
            and self.treeboost_loss_weight == 0
            and (not self.hierarchical_late_interaction or self.late_interaction_loss_weight == 0)
        ):
            raise ValueError(
                "At least one ID, metric, center, CSMM, TreeBoost, or late-interaction loss weight must be positive"
            )
        validate_augmentation_config(
            augmentation_config_from_options(self),
            epochs=self.epochs,
        )
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        for selector in (
            "metric_feature",
            "inference_feature",
            "feature_fusion",
            "pyramid_resize_mode",
            "spatial_conv_mode",
            "head_pool",
        ):
            normalize_selector(selector, getattr(self, selector))
        if self.post_fusion_mixer_reduction < 1:
            raise ValueError("post_fusion_mixer_reduction must be positive")
        if len(self.post_fusion_mixer_kernel) != 2 or any(value <= 0 for value in self.post_fusion_mixer_kernel):
            raise ValueError("post_fusion_mixer_kernel must contain two positive integers")
        if any(value % 2 == 0 for value in self.post_fusion_mixer_kernel):
            raise ValueError("post_fusion_mixer_kernel values must be odd")
        if not math.isfinite(self.post_fusion_mixer_gamma_init):
            raise ValueError("post_fusion_mixer_gamma_init must be finite")
        head_spec = get_reid_head_spec(self.head_type)
        standard_multiscale_head_types = MULTI_BRANCH_HEAD_TYPES
        if self.model_name.startswith("csl_tinyvit"):
            get_reid_head_spec(self.head_type, family="csl_tinyvit")
        elif self.model_name.startswith("mobilenetv4"):
            get_reid_head_spec(self.head_type, family="mobilenetv4")
        if not 0 <= self.multiscale_channel_alpha <= 1:
            raise ValueError("multiscale_channel_alpha must be in [0, 1]")
        if self.body_slot_mode not in {
            "recurrent_read",
            "recurrent_read_write",
        }:
            raise ValueError("body_slot_mode must be 'recurrent_read' or 'recurrent_read_write'")
        if not 0 < self.body_slot_alpha < 1:
            raise ValueError("body_slot_alpha must satisfy 0 < value < 1")
        if not 0 <= self.body_slot_visibility_floor < 1:
            raise ValueError("body_slot_visibility_floor must satisfy 0 <= value < 1")
        if self.feat_dim <= 0 or self.neck_dim <= 0:
            raise ValueError("feat_dim and neck_dim must be positive")
        if not self.head_parts or 1 not in self.head_parts or any(part <= 0 for part in self.head_parts):
            raise ValueError("head_parts must contain positive values and include the global branch 1")
        if self.head_type == "gpc_lite" and self.part_pooling != "stripes":
            raise ValueError("gpc_lite requires part_pooling='stripes'")
        if self.head_type == "gpc_lite" and self.decouple_patterns:
            raise ValueError("gpc_lite uses a shared backbone and does not support pattern decoupling")
        if self.head_type == "gpc_lite" and self.stripe_visibility:
            raise ValueError("gpc_lite does not support stripe visibility")
        if self.head_type == "body_slot":
            if not self.anatomical_auxiliary:
                raise ValueError("body_slot training requires anatomical_auxiliary")
            if self.anatomical_target_type != "body_slot_privileged_ema":
                raise ValueError("body_slot training requires anatomical_target_type='body_slot_privileged_ema'")
            if not self.scale_balanced_branches:
                raise ValueError("body_slot training requires scale_balanced_branches")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("body_slot training requires metric_feature='raw_concat'")
        elif self.anatomical_target_type == "body_slot_privileged_ema":
            raise ValueError("body_slot_privileged_ema requires head_type='body_slot'")
        if self.scale_balanced_branches:
            if not head_spec.supports_scale_balance:
                raise ValueError("scale_balanced_branches requires a standard multi-scale head")
            if self.part_pooling not in {"stripes", "overlap_stripes"}:
                raise ValueError("scale_balanced_branches requires fixed or overlapping stripe pooling")
            if self.branch_aware_metric:
                raise ValueError("scale_balanced_branches uses one selected metric descriptor")
            if self.classifier_loss != "ce":
                raise ValueError("scale_balanced_branches requires classifier_loss='ce'")
            if self._effective_metric_feature() not in {"raw_concat", "global", "coarse_concat"}:
                raise ValueError(
                    "scale_balanced_branches requires metric_feature='raw_concat', 'global', or 'coarse_concat'"
                )
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("scale_balanced_branches requires inference_feature='norm_concat_bn'")
        if not 0 <= self.multilevel_suppression_ratio < 1:
            raise ValueError("multilevel_suppression_ratio must satisfy 0 <= value < 1")
        if not math.isfinite(self.multilevel_suppression_loss_weight):
            raise ValueError("multilevel_suppression_loss_weight must be finite")
        if self.multilevel_suppression_loss_weight < 0:
            raise ValueError("multilevel_suppression_loss_weight must be non-negative")
        suppression_schedule = (
            self.multilevel_suppression_start_epoch,
            self.multilevel_suppression_ramp_end_epoch,
            self.multilevel_suppression_decay_start_epoch,
            self.multilevel_suppression_decay_end_epoch,
        )
        if any(epoch < 0 for epoch in suppression_schedule):
            raise ValueError("multilevel suppression schedule epochs must be non-negative")
        if not (
            self.multilevel_suppression_start_epoch
            < self.multilevel_suppression_ramp_end_epoch
            <= self.multilevel_suppression_decay_start_epoch
            < self.multilevel_suppression_decay_end_epoch
        ):
            raise ValueError("multilevel suppression requires start < ramp_end <= decay_start < decay_end")
        if self.multilevel_suppression:
            if not self.model_name.startswith("csl_tinyvit"):
                raise ValueError("multilevel suppression is implemented only for CSL-TinyViT")
            if self.multilevel_suppression_ratio <= 0:
                raise ValueError("enabled multilevel suppression requires a positive ratio")
            if self.multilevel_suppression_loss_weight <= 0:
                raise ValueError("enabled multilevel suppression requires a positive loss weight")
            if self.multilevel_suppression_decay_end_epoch > self.epochs:
                raise ValueError("enabled multilevel suppression requires decay_end_epoch <= epochs")
            if self.multilevel_suppression_start_epoch < self.backbone_freeze_epochs:
                raise ValueError("enabled multilevel suppression requires start_epoch >= backbone_freeze_epochs")
            if self.reid_adapter_stages:
                raise ValueError("multilevel suppression is a clean V20 ablation and requires reid_adapter_stages=()")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("multilevel suppression requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("multilevel suppression requires scale-balanced head_parts=(1, 2, 4)")
            if self.classifier_loss != "ce":
                raise ValueError("multilevel suppression requires classifier_loss='ce'")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("multilevel suppression requires the Stage-0 semantic-fine feature map")
        if self.branch_attention_token_dim < 1 or self.branch_attention_num_heads < 1:
            raise ValueError("branch attention token dimension and head count must be positive")
        if self.branch_attention_token_dim % self.branch_attention_num_heads:
            raise ValueError("branch attention token dimension must be divisible by its head count")
        if self.branch_attention_num_layers < 1 or self.branch_attention_mlp_ratio <= 0:
            raise ValueError("branch attention layer count and MLP ratio must be positive")
        if not 0 <= self.branch_attention_dropout < 1:
            raise ValueError("branch attention dropout must satisfy 0 <= value < 1")
        if self.hierarchical_branch_attention:
            if self.model_name not in _CSL_TINYVIT_11M_MODELS:
                raise ValueError("hierarchical branch attention is currently implemented for csl_tinyvit_11m")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("hierarchical branch attention requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("hierarchical branch attention requires scale-balanced head_parts=(1, 2, 4)")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("hierarchical branch attention requires optimized Stage-0 semantic-fine fusion")
            if self.compact_deployment_head:
                raise ValueError("hierarchical branch attention does not support the compact deployment head")
        if self.branch_set_attention_token_dim < 1 or self.branch_set_attention_num_heads < 1:
            raise ValueError("branch-set attention token dimension and head count must be positive")
        if self.branch_set_attention_token_dim % self.branch_set_attention_num_heads:
            raise ValueError("branch-set attention token dimension must be divisible by its head count")
        if self.branch_set_attention_num_layers < 1 or self.branch_set_attention_mlp_ratio <= 0:
            raise ValueError("branch-set attention layer count and MLP ratio must be positive")
        if not 0 <= self.branch_set_attention_dropout < 1:
            raise ValueError("branch-set attention dropout must satisfy 0 <= value < 1")
        if self.branch_set_attention:
            if self.model_name not in _CSL_TINYVIT_11M_MODELS:
                raise ValueError("branch-set attention is currently implemented for csl_tinyvit_11m")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("branch-set attention requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("branch-set attention requires scale-balanced head_parts=(1, 2, 4)")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("branch-set attention requires optimized Stage-0 semantic-fine fusion")
            if self.spatial_conv_mode != "depthwise_separable":
                raise ValueError("branch-set attention must remain rooted in the a11s2 depthwise neck")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("branch-set attention preserves the winning raw_concat triplet descriptor")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("branch-set attention preserves the norm_concat_bn descriptor")
            if self.compact_deployment_head or self.hierarchical_branch_attention:
                raise ValueError("branch-set and tree attention are independent non-compact treatments")
        if self.query_decoder_dim < 1 or self.query_decoder_num_heads < 1:
            raise ValueError("query-decoder dimension and head count must be positive")
        if self.query_decoder_dim % self.query_decoder_num_heads:
            raise ValueError("query-decoder dimension must be divisible by its head count")
        if self.query_decoder_dim % 4:
            raise ValueError("query-decoder dimension must be divisible by four for 2D positions")
        if self.query_decoder_num_layers < 1 or self.query_decoder_mlp_ratio <= 0:
            raise ValueError("query-decoder layer count and MLP ratio must be positive")
        if not 0 <= self.query_decoder_dropout < 1:
            raise ValueError("query-decoder dropout must satisfy 0 <= value < 1")
        if self.multiscale_query_decoder:
            if self.model_name not in _CSL_TINYVIT_11M_MODELS:
                raise ValueError("multi-scale query decoder is currently implemented for csl_tinyvit_11m")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("multi-scale query decoder requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("multi-scale query decoder requires scale-balanced head_parts=(1, 2, 4)")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("multi-scale query decoder requires optimized Stage-0 semantic-fine fusion")
            if self.spatial_conv_mode != "depthwise_separable":
                raise ValueError("multi-scale query decoder must remain rooted in the a11s2 depthwise neck")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("multi-scale query decoder preserves the winning raw_concat triplet descriptor")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("multi-scale query decoder preserves the norm_concat_bn descriptor")
            if self.compact_deployment_head or self.hierarchical_branch_attention or self.branch_set_attention:
                raise ValueError("multi-scale query decoder is an independent non-compact treatment")
        if self.late_interaction_dim < 1 or self.late_interaction_num_heads < 1:
            raise ValueError("late-interaction token dimension and head count must be positive")
        if self.late_interaction_dim % self.late_interaction_num_heads:
            raise ValueError("late-interaction token dimension must be divisible by its head count")
        if self.late_interaction_num_layers < 1 or self.late_interaction_sinkhorn_iters < 1:
            raise ValueError("late-interaction layer and Sinkhorn iteration counts must be positive")
        if self.late_interaction_null_tokens != 1:
            raise ValueError("hierarchical late interaction currently requires exactly one null token")
        if self.late_interaction_negative_identities < 1:
            raise ValueError("late-interaction negative identity count must be positive")
        if self.late_interaction_rerank_topk < 1:
            raise ValueError("late-interaction rerank top-k must be positive")
        if not 0 < self.late_interaction_base_score_init < 1:
            raise ValueError("late-interaction base-score initialization must satisfy 0 < value < 1")
        if self.late_interaction_temperature <= 0:
            raise ValueError("late-interaction temperature must be positive")
        if self.hierarchical_late_interaction:
            if not 0 <= self.late_interaction_start_epoch < self.late_interaction_ramp_end_epoch <= self.epochs:
                raise ValueError(
                    "Enabled hierarchical late interaction requires 0 <= start epoch < ramp end epoch <= epochs"
                )
            if self.late_interaction_loss_weight <= 0:
                raise ValueError("hierarchical late interaction requires a positive matcher loss weight")
            if self.model_name not in _CSL_TINYVIT_11M_MODELS:
                raise ValueError("hierarchical late interaction is currently implemented for csl_tinyvit_11m")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("hierarchical late interaction requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("hierarchical late interaction requires scale-balanced head_parts=(1, 2, 4)")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("hierarchical late interaction requires optimized Stage-0 semantic-fine fusion")
            if self.spatial_conv_mode != "depthwise_separable":
                raise ValueError("hierarchical late interaction must remain rooted in the a11s2 depthwise neck")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("hierarchical late interaction preserves the winning raw_concat triplet descriptor")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("hierarchical late interaction preserves the norm_concat_bn descriptor")
            if (
                self.compact_deployment_head
                or self.hierarchical_branch_attention
                or self.branch_set_attention
                or self.multiscale_query_decoder
            ):
                raise ValueError("hierarchical late interaction is an independent non-compact treatment")
        if self.mcpt_mode not in MCPT_MODES:
            raise ValueError(f"mcpt_mode must be one of {sorted(MCPT_MODES)}, got {self.mcpt_mode!r}")
        if self.mcpt_hidden_dim < 1:
            raise ValueError("mcpt_hidden_dim must be positive")
        if not 0 < self.mcpt_max_displacement < 0.5:
            raise ValueError("mcpt_max_displacement must satisfy 0 < value < 0.5")
        if self.mcpt_smoothness_weight < 0 or self.mcpt_identity_weight < 0:
            raise ValueError("MCPT regularization weights must be non-negative")
        if self.mcpt_lr_multiplier <= 0:
            raise ValueError("mcpt_lr_multiplier must be positive")
        if self.mcpt_mode != "none":
            mcpt_feature_dims = {
                "csl_tinyvit_7m": 384,
                "csl_tinyvit_7m_v20": 384,
                "csl_tinyvit_11m": 512,
                "csl_tinyvit_11m_v20": 512,
                "mobilenetv4_conv_medium": 384,
                "mobilenetv4_hybrid_medium": 384,
                "mobilenetv4_conv_medium_v20": 384,
                "mobilenetv4_hybrid_medium_v20": 384,
            }
            expected_feature_dim = mcpt_feature_dims.get(self.model_name)
            if expected_feature_dim is None:
                raise ValueError("MCPT is implemented for the CSL-TinyViT 7M/11M and MobileNetV4 Medium families")
            if not 0 <= self.mcpt_start_epoch < self.mcpt_ramp_end_epoch <= self.epochs:
                raise ValueError("Enabled MCPT requires 0 <= start epoch < ramp end epoch <= epochs")
            if not self.mcpt_start_epoch < self.mcpt_identity_decay_epoch <= self.epochs:
                raise ValueError("Enabled MCPT requires start epoch < identity decay epoch <= epochs")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("MCPT requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("MCPT requires scale-balanced head_parts=(1, 2, 4)")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("MCPT requires the proven Stage-0 semantic-fine fusion")
            if self.spatial_conv_mode != "depthwise_separable":
                raise ValueError("MCPT preserves the optimized depthwise-separable neck")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("MCPT preserves the raw_concat metric descriptor")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("MCPT preserves the norm_concat_bn inference descriptor")
            if self.feat_dim != expected_feature_dim or self.neck_dim != expected_feature_dim:
                raise ValueError(f"{self.model_name} MCPT requires feat_dim=neck_dim={expected_feature_dim}")
            if self.anatomical_auxiliary:
                supported_pose_teacher = (
                    self.model_name
                    in {
                        "csl_tinyvit_7m",
                        "csl_tinyvit_7m_v20",
                        "mobilenetv4_conv_medium",
                        "mobilenetv4_hybrid_medium",
                        "mobilenetv4_conv_medium_v20",
                        "mobilenetv4_hybrid_medium_v20",
                    }
                    and self.anatomical_target_type == V8_ANATOMICAL_TARGET_TYPE
                    and self.anatomical_multiscale
                    and not self.anatomical_deployment
                    and not self.anatomical_accessory_query
                )
                if not supported_pose_teacher:
                    raise ValueError(
                        "MCPT plus anatomical supervision is supported only "
                        "for the training-only multiscale V8 pose teacher on "
                        "the CSL-TinyViT 7M or MobileNetV4 Medium families"
                    )
            if (
                self.compact_deployment_head
                or self.hierarchical_branch_attention
                or self.branch_set_attention
                or self.multiscale_query_decoder
                or self.hierarchical_late_interaction
            ):
                raise ValueError("MCPT must be isolated from other head treatments")
        if self.jpm_num_groups < 2:
            raise ValueError("jpm_num_groups must be at least two")
        if self.jpm_shift < 0:
            raise ValueError("jpm_shift must be non-negative")
        if self.jpm_token_dim < 1:
            raise ValueError("jpm_token_dim must be positive")
        if self.jpm_num_heads < 1 or self.jpm_token_dim % self.jpm_num_heads:
            raise ValueError("jpm_token_dim must be divisible by a positive jpm_num_heads")
        if self.jpm_mlp_ratio <= 0:
            raise ValueError("jpm_mlp_ratio must be positive")
        if not 0 <= self.jpm_dropout < 1:
            raise ValueError("jpm_dropout must satisfy 0 <= value < 1")
        if self.jpm:
            if self.model_name not in {"csl_tinyvit_7m", "csl_tinyvit_7m_v20"}:
                raise ValueError("JPM is scoped to the csl_tinyvit_7m family")
            if self.loss_type != "triplet" or self.classifier_loss != "ce":
                raise ValueError("JPM requires CE identity and triplet losses")
            if self.jpm_id_loss_weight <= 0 or self.jpm_metric_loss_weight <= 0:
                raise ValueError("Enabled JPM requires positive ID and metric weights")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("JPM requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("JPM requires scale-balanced head_parts=(1, 2, 4)")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("JPM requires the proven Stage-0 semantic-fine fusion")
            if self.spatial_conv_mode != "depthwise_separable":
                raise ValueError("JPM preserves the optimized depthwise-separable neck")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("JPM preserves the raw_concat metric descriptor")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("JPM preserves the norm_concat_bn inference descriptor")
            if self.feat_dim != 384 or self.neck_dim != 384:
                raise ValueError("7M JPM requires feat_dim=neck_dim=384")
            if self.anatomical_auxiliary or self.mcpt_mode != "none":
                raise ValueError("JPM must be isolated from pose and MCPT")
            if any(
                (
                    self.compact_deployment_head,
                    self.hierarchical_branch_attention,
                    self.branch_set_attention,
                    self.multiscale_query_decoder,
                    self.hierarchical_late_interaction,
                    self._part_relation_enabled(),
                )
            ):
                raise ValueError("JPM must be isolated from other head treatments")
        if self.csmm_loss_weight > 0:
            if self.model_name not in _CSL_TINYVIT_11M_MODELS:
                raise ValueError("CSMM is currently implemented for csl_tinyvit_11m")
            if self.head_type != "standard" or self.part_pooling not in {"stripes", "overlap_stripes"}:
                raise ValueError("CSMM requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("CSMM requires scale-balanced head_parts=(1, 2, 4)")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("CSMM preserves the winning full raw_concat triplet descriptor")
        if self.treeboost_loss_weight > 0:
            if self.model_name not in _CSL_TINYVIT_11M_MODELS:
                raise ValueError("TreeBoost-AP is currently implemented for csl_tinyvit_11m")
            if self.head_type != "standard" or self.part_pooling not in {"stripes", "overlap_stripes"}:
                raise ValueError("TreeBoost-AP requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("TreeBoost-AP requires scale-balanced head_parts=(1, 2, 4)")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("TreeBoost-AP preserves the winning full raw_concat triplet descriptor")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("TreeBoost-AP requires the norm_concat_bn inference descriptor")
        if self.head_type in standard_multiscale_head_types - {"standard"}:
            if self.model_name not in _CSL_TINYVIT_11M_MODELS:
                raise ValueError("G/P/C specialist heads are currently validated only for csl_tinyvit_11m")
            if self.feature_fusion not in {
                "global_final_parts_stage0_semantic_fine_reference",
                "global_final_parts_stage0_semantic_fine",
            }:
                raise ValueError("G/P/C specialist heads require Stage-0 semantic-fine fusion")
            if self.head_parts != (1, 2, 4) or self.part_pooling != "stripes":
                raise ValueError("G/P/C specialist heads require stripe head_parts=(1, 2, 4)")
            if not self.scale_balanced_branches:
                raise ValueError("G/P/C specialist heads require scale-balanced main branches")
            if self._effective_metric_feature() != "raw_concat":
                raise ValueError("G/P/C specialist heads require metric_feature='raw_concat'")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("G/P/C specialist heads require norm_concat_bn retrieval")
        if self._effective_metric_feature() == "coarse_concat":
            if self.head_type != "standard":
                raise ValueError("metric_feature='coarse_concat' requires head_type='standard'")
            if self.part_pooling not in {"stripes", "overlap_stripes"} or 2 not in self.head_parts:
                raise ValueError("metric_feature='coarse_concat' requires fixed or overlapping two-stripe pooling")
        if self.drop_global_aux and self.head_type != "standard":
            raise ValueError("drop_global_aux requires head_type='standard'")
        if self.drop_global_aux and self.classifier_loss != "ce":
            raise ValueError("drop_global_aux requires classifier_loss='ce'")
        if self.stripe_visibility:
            local_granularities = tuple(part for part in self.head_parts if part != 1)
            if self.part_pooling != "stripes" or len(local_granularities) != 1:
                raise ValueError("stripe_visibility requires fixed stripes with exactly one local granularity")
        if self.part_pooling == "semantic_parts" and not any(part > 1 for part in self.head_parts):
            raise ValueError("semantic_parts pooling requires at least one local part in head_parts")
        if not 0 < self.drop_global_aux_ratio <= 1:
            raise ValueError("drop_global_aux_ratio must satisfy 0 < value <= 1")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("drop_path_rate must satisfy 0 <= value < 1")
        if self.timm_head_mode not in {
            "pooled",
            "spatial",
            "spatial_adapt_norm",
            "spatial_linear",
            "off",
        }:
            raise ValueError("timm_head_mode must be one of: pooled, spatial, spatial_adapt_norm, spatial_linear, off")
        if self.mobilenetv4_last_stride not in {1, 2}:
            raise ValueError("mobilenetv4_last_stride must be 1 or 2")
        if self.mobilenetv4_neck_mode not in {"cnn", "spatial_ln"}:
            raise ValueError("mobilenetv4_neck_mode must be one of: cnn, spatial_ln")
        if self.attention_window_layout not in {"legacy", "rect"}:
            raise ValueError("Unsupported attention_window_layout")
        if self.stage2_mlp_ratio <= 0 or self.stage3_mlp_ratio <= 0:
            raise ValueError("stage2_mlp_ratio and stage3_mlp_ratio must be positive")
        if self.stage2_depth < 1:
            raise ValueError("stage2_depth must be positive")
        if self.stage3_depth < 1:
            raise ValueError("stage3_depth must be positive")
        if self.identity_register_count < 2:
            raise ValueError("identity_register_count must be at least two")
        if self.identity_register_dim < 1:
            raise ValueError("identity_register_dim must be positive")
        if self.identity_register_num_heads < 1:
            raise ValueError("identity_register_num_heads must be positive")
        if self.identity_register_dim % self.identity_register_num_heads:
            raise ValueError("identity_register_dim must be divisible by identity_register_num_heads")
        if not 0 <= self.identity_register_dropout < 1:
            raise ValueError("identity_register_dropout must be in [0, 1)")
        if self.identity_register_diversity_weight < 0:
            raise ValueError("identity_register_diversity_weight must be non-negative")
        if not -1 <= self.identity_register_diversity_margin <= 1:
            raise ValueError("identity_register_diversity_margin must be in [-1, 1]")
        if self.identity_register_diversity_weight > 0 and not self.identity_registers:
            raise ValueError("identity register diversity requires identity_registers")
        if self.stage2_width_merge_after < 0 or self.stage2_width_merge_after >= self.stage2_depth:
            if self.stage2_width_merge_after != 0:
                raise ValueError("stage2_width_merge_after must be zero or fall before the final Stage-2 block")
        if self.stage2_width_merge_after:
            if self.stage3_downsample:
                raise ValueError("stage2_width_merge_after and stage3_downsample are alternative reductions")
            if self.attention_window_layout != "rect":
                raise ValueError("stage2_width_merge_after requires rectangular attention windows")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("stage2_width_merge_after requires optimized Stage-0 semantic-fine fusion")
        if self.width_first_hierarchy:
            if not self.model_name.startswith("csl_tinyvit"):
                raise ValueError("width_first_hierarchy currently supports CSL-TinyViT")
            if self.attention_window_layout != "rect":
                raise ValueError("width_first_hierarchy requires rectangular windows")
            if self.feature_fusion != ("global_final_parts_stage0_semantic_fine"):
                raise ValueError("width_first_hierarchy requires optimized Stage-0 semantic-fine fusion")
            if self.stage2_width_merge_after or self.stage3_downsample:
                raise ValueError("width_first_hierarchy cannot use later spatial reduction paths")
        if self.identity_registers:
            if not self.model_name.startswith("csl_tinyvit"):
                raise ValueError("identity_registers currently supports CSL-TinyViT")
            if self.stage2_width_merge_after or self.stage3_downsample:
                raise ValueError("identity_registers requires unreduced Stage-2/3 maps")
            if self.head_type != "standard" or self.head_parts != (1, 2, 4) or self.part_pooling != "stripes":
                raise ValueError("identity_registers requires the unchanged standard global/2-stripe/4-stripe head")
        if self.native_branch_widths:
            if not self.stage3_downsample:
                raise ValueError("native_branch_widths requires stage3_downsample")
            if self.feature_fusion != "global_final_parts_stage0_semantic_fine":
                raise ValueError("native_branch_widths requires optimized Stage-0 semantic-fine fusion")
            if self.head_parts != (1, 2, 4) or self.part_pooling != "stripes":
                raise ValueError("native_branch_widths requires stripe head_parts=(1, 2, 4)")
        if self.fine_map_dim < 0:
            raise ValueError("fine_map_dim must be non-negative")
        if self.fine_map_dim:
            if self.feature_fusion not in {
                "global_final_parts_stage0_semantic_fine",
                "global_final_parts_stage0_fine_lite",
            }:
                raise ValueError("fine_map_dim requires optimized or lite Stage-0 fine fusion")
            if self.fine_map_dim > self.neck_dim:
                raise ValueError("fine_map_dim must not exceed neck_dim")
            if self.head_parts != (1, 2, 4) or self.part_pooling != "stripes":
                raise ValueError("fine_map_dim requires stripe head_parts=(1, 2, 4)")
            if self.native_branch_widths:
                raise ValueError("fine_map_dim and native_branch_widths are alternative width controls")
        if self.feature_fusion == "global_final_parts_stage0_pool_first":
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("pool-first fusion requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4):
                raise ValueError("pool-first fusion requires head_parts=(1, 2, 4)")
            if self.post_fusion_mixer != "none":
                raise ValueError("pool-first fusion does not support a shared post-fusion mixer")
        if self.compact_deployment_head:
            if self.model_name not in _CSL_TINYVIT_11M_MODELS:
                raise ValueError("compact_deployment_head is currently validated only for csl_tinyvit_11m")
            if self.head_type != "standard" or self.part_pooling != "stripes":
                raise ValueError("compact_deployment_head requires the standard fixed-stripe head")
            if self.head_parts != (1, 2, 4) or not self.scale_balanced_branches:
                raise ValueError("compact_deployment_head requires scale-balanced head_parts=(1, 2, 4)")
            if self.feature_fusion not in {
                "global_final_parts_stage0_semantic_fine_reference",
                "global_final_parts_stage0_semantic_fine",
            }:
                raise ValueError("compact_deployment_head requires Stage-0 semantic-fine fusion")
        if self.attention_bias not in {"absolute", "signed_factorized"}:
            raise ValueError("Unsupported attention_bias")
        if self.interpolate_pretrained_attention_bias and self.attention_bias != "absolute":
            raise ValueError("interpolate_pretrained_attention_bias requires attention_bias='absolute'")
        if self.vit_lr_profile not in {"layer_decay", "reid_lrd"}:
            raise ValueError("vit_lr_profile must be one of: layer_decay, reid_lrd")
        if not 0 < self.layer_decay <= 1:
            raise ValueError("layer_decay must satisfy 0 < value <= 1")
        if self.backbone_lr_mult <= 0:
            raise ValueError("backbone_lr_mult must be positive")
        if self.backbone_freeze_epochs < 0 or self.backbone_freeze_epochs > self.epochs:
            raise ValueError("backbone_freeze_epochs must satisfy 0 <= value <= epochs")
        if self.gradual_unfreeze_head_epochs < 0:
            raise ValueError("gradual_unfreeze_head_epochs must be non-negative")
        if self.gradual_unfreeze_stage_epochs < 0:
            raise ValueError("gradual_unfreeze_stage_epochs must be non-negative")
        if self.gradual_unfreeze_backbone_lr_epochs < 0:
            raise ValueError("gradual_unfreeze_backbone_lr_epochs must be non-negative")
        if self.gradual_unfreeze_backbone_lr_mult <= 0:
            raise ValueError("gradual_unfreeze_backbone_lr_mult must be positive")
        if self.gradual_unfreeze:
            if self.backbone_freeze_epochs > 0:
                raise ValueError("gradual_unfreeze cannot be combined with backbone_freeze_epochs")
            if self.head_warmup_epochs > 0:
                raise ValueError("gradual_unfreeze cannot be combined with head_warmup_epochs")
            if self.gradual_unfreeze_stage_epochs < self.gradual_unfreeze_head_epochs:
                raise ValueError("gradual_unfreeze_stage_epochs must be >= gradual_unfreeze_head_epochs")
            if self.gradual_unfreeze_stage_epochs > self.epochs:
                raise ValueError("gradual_unfreeze_stage_epochs must be <= epochs")
        if self.branch_metric_part_weight < 0:
            raise ValueError("branch_metric_part_weight must be non-negative")
        if self.evidence_num_roles < 1:
            raise ValueError("evidence_num_roles must be positive")
        for name in (
            "evidence_alignment_loss_weight",
            "evidence_alignment_margin",
            "evidence_sinkhorn_temperature",
            "evidence_null_loss_weight",
            "evidence_diversity_loss_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.evidence_sinkhorn_iters < 1:
            raise ValueError("evidence_sinkhorn_iters must be positive")
        if self.evidence_rerank_topk < 0:
            raise ValueError("evidence_rerank_topk must be non-negative")
        if self.anatomical_token_dim < 2 * ANATOMICAL_CANONICAL_CELLS:
            raise ValueError(
                "anatomical_token_dim must provide at least two channels per canonical anatomical grid cell"
            )
        if self.anatomical_token_dim % ANATOMICAL_CANONICAL_CELLS:
            raise ValueError("anatomical_token_dim must be divisible by the eight canonical anatomical grid cells")
        if not 0 <= self.anatomical_min_keypoint_confidence <= 1:
            raise ValueError("anatomical_min_keypoint_confidence must be in [0, 1]")
        for name in (
            "anatomical_distill_weight",
            "anatomical_attention_weight",
            "anatomical_foreground_weight",
            "anatomical_semantic_part_weight",
            "anatomical_visibility_weight",
            "anatomical_contrastive_weight",
            "anatomical_descriptor_distill_weight",
            "anatomical_branch_distill_weight",
            "anatomical_branch_global_coefficient",
            "anatomical_branch_coarse_coefficient",
            "anatomical_branch_fine_coefficient",
            "anatomical_pose_teacher_weight",
            "anatomical_query_distill_weight",
            "anatomical_query_relational_distill_weight",
            "anatomical_query_diversity_weight",
            "anatomical_part_triplet_weight",
            "anatomical_deployment_id_weight",
            "anatomical_deployment_metric_weight",
            "anatomical_local_scale_weight",
            "anatomical_fine_scale_weight",
            "anatomical_cross_scale_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.anatomical_temperature <= 0:
            raise ValueError("anatomical_temperature must be positive")
        if not -1.0 <= self.anatomical_query_diversity_margin <= 1.0:
            raise ValueError("anatomical_query_diversity_margin must be in [-1, 1]")
        if self.anatomical_deployment_dim < 1:
            raise ValueError("anatomical_deployment_dim must be positive")
        if not 0 < self.anatomical_deployment_alpha <= 1:
            raise ValueError("anatomical_deployment_alpha must satisfy 0 < value <= 1")
        if self.anatomical_target_type not in ANATOMICAL_TARGET_TYPES:
            raise ValueError(
                "anatomical_target_type must be one of "
                f"{sorted(ANATOMICAL_TARGET_TYPES)}, got "
                f"{self.anatomical_target_type!r}"
            )
        if not 0 <= self.anatomical_teacher_momentum < 1:
            raise ValueError("anatomical_teacher_momentum must be in [0, 1)")
        if not 0 <= self.anatomical_pose_only_reliability <= 1:
            raise ValueError("anatomical_pose_only_reliability must be in [0, 1]")
        if not 0 <= self.anatomical_min_effective_coverage <= 1:
            raise ValueError("anatomical_min_effective_coverage must be in [0, 1]")
        for name in (
            "anatomical_student_start_epoch",
            "anatomical_student_ramp_end_epoch",
            "anatomical_query_start_epoch",
            "anatomical_query_ramp_end_epoch",
            "anatomical_fine_start_epoch",
            "anatomical_fine_ramp_end_epoch",
            "anatomical_decay_start_epoch",
            "anatomical_decay_end_epoch",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.anatomical_student_ramp_end_epoch < self.anatomical_student_start_epoch:
            raise ValueError(
                "anatomical_student_ramp_end_epoch must be greater than or equal to anatomical_student_start_epoch"
            )
        if self.anatomical_query_ramp_end_epoch < self.anatomical_query_start_epoch:
            raise ValueError(
                "anatomical_query_ramp_end_epoch must be greater than or equal to anatomical_query_start_epoch"
            )
        if self.anatomical_fine_ramp_end_epoch < self.anatomical_fine_start_epoch:
            raise ValueError(
                "anatomical_fine_ramp_end_epoch must be greater than or equal to anatomical_fine_start_epoch"
            )
        if (
            self.anatomical_fine_start_epoch > 0 or self.anatomical_fine_ramp_end_epoch > 0
        ) and not self.anatomical_multiscale:
            raise ValueError("fine-scale anatomical scheduling requires anatomical_multiscale")
        if bool(self.anatomical_decay_start_epoch) != bool(self.anatomical_decay_end_epoch):
            raise ValueError("anatomical decay start/end epochs must both be zero or both be positive")
        if self.anatomical_decay_end_epoch > 0 and self.anatomical_decay_end_epoch <= self.anatomical_decay_start_epoch:
            raise ValueError("anatomical_decay_end_epoch must be greater than anatomical_decay_start_epoch")
        if (
            self.anatomical_decay_start_epoch > 0
            and self.anatomical_student_ramp_end_epoch > self.anatomical_decay_start_epoch
        ):
            raise ValueError("anatomical_student_ramp_end_epoch must not exceed anatomical_decay_start_epoch")
        if (
            self.anatomical_decay_start_epoch > 0
            and self.anatomical_fine_ramp_end_epoch > self.anatomical_decay_start_epoch
        ):
            raise ValueError("anatomical_fine_ramp_end_epoch must not exceed anatomical_decay_start_epoch")
        if self.anatomical_decay_end_epoch > self.epochs:
            raise ValueError("anatomical_decay_end_epoch must not exceed total epochs")
        if (
            self.anatomical_target_type == "decoupled_pose_parsing_teacher"
            and self.anatomical_decay_start_epoch > 0
            and self.anatomical_query_ramp_end_epoch > self.anatomical_decay_start_epoch
        ):
            raise ValueError("anatomical_query_ramp_end_epoch must not exceed anatomical_decay_start_epoch")
        if self.anatomical_descriptor_distill_weight > 0 and not self.anatomical_auxiliary:
            raise ValueError("anatomical descriptor distillation requires anatomical_auxiliary")
        if self.anatomical_branch_distill_weight > 0:
            if not self.anatomical_auxiliary:
                raise ValueError("anatomical branch distillation requires anatomical_auxiliary")
            if self.head_type != "standard":
                raise ValueError("anatomical branch distillation requires head_type='standard'")
            if self.anatomical_target_type not in EMA_ANATOMICAL_TARGET_TYPES:
                raise ValueError("anatomical branch distillation requires an EMA pose-teacher target type")
            if not self.anatomical_multiscale:
                raise ValueError("anatomical branch distillation requires anatomical_multiscale")
            if not self.scale_balanced_branches:
                raise ValueError("anatomical branch distillation requires scale-balanced branches")
            if self.part_pooling != "stripes":
                raise ValueError("anatomical branch distillation requires fixed stripe pooling")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("anatomical branch distillation requires inference_feature='norm_concat_bn'")
            if self.compact_deployment_head:
                raise ValueError("anatomical branch distillation does not support the compact deployment descriptor")
            coefficient_sum = (
                self.anatomical_branch_global_coefficient
                + self.anatomical_branch_coarse_coefficient
                + self.anatomical_branch_fine_coefficient
            )
            if not math.isclose(
                coefficient_sum,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("anatomical global/coarse/fine branch coefficients must sum to 1")
        if self.anatomical_deployment:
            if not self.anatomical_auxiliary:
                raise ValueError("anatomical deployment requires anatomical_auxiliary")
            if self.head_type != "standard":
                raise ValueError("anatomical deployment requires head_type='standard'")
            if self.anatomical_target_type != "learned_pose_concat_ema":
                raise ValueError("anatomical deployment requires anatomical_target_type='learned_pose_concat_ema'")
            if not self.anatomical_multiscale:
                raise ValueError("anatomical deployment requires anatomical_multiscale")
            if self.anatomical_descriptor_distill_weight > 0:
                raise ValueError("anatomical deployment and descriptor distillation are independent treatments")
            if self.anatomical_branch_distill_weight > 0:
                raise ValueError("anatomical deployment and branch distillation are independent treatments")
            if self.inference_feature != "norm_concat_bn":
                raise ValueError("anatomical deployment requires inference_feature='norm_concat_bn'")
            if self.compact_deployment_head:
                raise ValueError("anatomical deployment does not support the compact deployment head")
            if self.classifier_loss != "ce":
                raise ValueError("anatomical deployment currently requires classifier_loss='ce'")
        if self.anatomical_pose_teacher_weight > 0 and not self.anatomical_auxiliary:
            raise ValueError("pose-guided anatomical teacher requires anatomical_auxiliary")
        if self.anatomical_multiscale:
            if not self.anatomical_auxiliary:
                raise ValueError("multi-scale anatomy requires anatomical_auxiliary")
            if not math.isclose(
                self.anatomical_local_scale_weight + self.anatomical_fine_scale_weight,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("anatomical local/fine scale weights must sum to 1")
            if not CSLTinyViTFeatureFusion.uses_hierarchical_scales(self.feature_fusion) or tuple(self.head_parts) != (
                1,
                2,
                4,
            ):
                raise ValueError(
                    "multi-scale anatomy requires hierarchical head_parts=(1, 2, 4) with a fine feature map"
                )
        if self.anatomical_auxiliary:
            if not self.model_name.startswith(("csl_tinyvit", "mobilenetv4")):
                raise ValueError("anatomical_auxiliary currently supports CSL-TinyViT and MobileNetV4")
            if self.head_type not in {
                "standard",
                "stage2_channel2",
                "multiscale_channel2",
                "body_slot",
            }:
                raise ValueError(
                    "anatomical_auxiliary requires head_type='standard' or a channel-representation control"
                )
            if not self.anatomical_metadata_dir:
                raise ValueError("anatomical_auxiliary requires anatomical_metadata_dir")
            if self.anatomical_target_type == "privileged_mask_pose_attention" and not self.anatomical_person_mask_dir:
                raise ValueError("privileged mask-pose attention requires anatomical_person_mask_dir")
            if self.anatomical_target_type == "decoupled_pose_parsing_teacher":
                if not self.anatomical_person_mask_dir:
                    raise ValueError("decoupled pose-parsing teacher requires anatomical_person_mask_dir")
                if not self.anatomical_multiscale:
                    raise ValueError("decoupled pose-parsing teacher requires anatomical_multiscale")
                if self.anatomical_query_distill_weight <= 0:
                    raise ValueError(
                        "decoupled pose-parsing teacher requires a positive anatomical_query_distill_weight"
                    )
                if self.anatomical_foreground_weight <= 0:
                    raise ValueError(
                        "decoupled pose-parsing teacher requires a positive "
                        "anatomical_foreground_weight to train its private "
                        "parsing adapters"
                    )
                if self.anatomical_query_ramp_end_epoch > self.epochs:
                    raise ValueError("anatomical_query_ramp_end_epoch must not exceed total epochs")
                if (
                    self.anatomical_descriptor_distill_weight > 0
                    or self.anatomical_branch_distill_weight > 0
                    or self.anatomical_deployment
                ):
                    raise ValueError(
                        "decoupled pose-parsing teacher is training-only; "
                        "descriptor, branch, and deployment treatments must "
                        "remain disabled"
                    )
                if (
                    self.part_pooling != "stripes"
                    or not self.scale_balanced_branches
                    or self.inference_feature != "norm_concat_bn"
                ):
                    raise ValueError(
                        "decoupled pose-parsing teacher preserves the v8 "
                        "fixed-stripe, scale-balanced norm_concat_bn path"
                    )
            if self.anatomical_target_type in SEMANTIC_ANATOMICAL_TARGET_TYPES:
                if not self.anatomical_person_mask_dir:
                    raise ValueError("pose-semantic anatomy requires anatomical_person_mask_dir")
                if not self.anatomical_multiscale:
                    raise ValueError("pose-semantic anatomy requires anatomical_multiscale")
                if self.anatomical_semantic_part_weight <= 0:
                    raise ValueError("pose-semantic anatomy requires a positive anatomical_semantic_part_weight")
                if self.anatomical_deployment:
                    raise ValueError(
                        "pose-semantic anatomy is training-only and does not support anatomical_deployment"
                    )
            if self.anatomical_target_type == "body_slot_privileged_ema":
                if self.head_type != "body_slot":
                    raise ValueError("body-slot supervision requires head_type='body_slot'")
                if not self.anatomical_person_mask_dir:
                    raise ValueError("body-slot supervision requires anatomical_person_mask_dir")
                if not self.anatomical_multiscale:
                    raise ValueError("body-slot supervision requires anatomical_multiscale")
                if (
                    self.anatomical_distill_weight <= 0
                    or self.anatomical_attention_weight <= 0
                    or self.anatomical_visibility_weight <= 0
                    or self.anatomical_foreground_weight <= 0
                ):
                    raise ValueError(
                        "body-slot supervision requires positive distill, "
                        "attention, visibility, and foreground/coverage weights"
                    )
                if (
                    self.anatomical_descriptor_distill_weight > 0
                    or self.anatomical_branch_distill_weight > 0
                    or self.anatomical_deployment
                ):
                    raise ValueError(
                        "body slots are the deployed descriptor; legacy "
                        "descriptor, branch, and deployment distillation must "
                        "remain disabled"
                    )
                unused_body_slot_losses = {
                    "anatomical_contrastive_weight": (self.anatomical_contrastive_weight),
                    "anatomical_pose_teacher_weight": (self.anatomical_pose_teacher_weight),
                    "anatomical_semantic_part_weight": (self.anatomical_semantic_part_weight),
                }
                enabled_unused = [name for name, value in unused_body_slot_losses.items() if value > 0]
                if enabled_unused:
                    raise ValueError(
                        "body-slot training uses explicit part-triplet, "
                        "attention, and coverage losses; disable legacy "
                        f"overlapping weights: {enabled_unused}"
                    )
                stage_coefficient_sum = (
                    self.anatomical_branch_global_coefficient
                    + self.anatomical_branch_coarse_coefficient
                    + self.anatomical_branch_fine_coefficient
                )
                if not math.isclose(
                    stage_coefficient_sum,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError("body-slot fine/coarse/global stage coefficients must sum to 1")
            if self.same_id_part_mosaic:
                raise ValueError("anatomical_auxiliary cannot be combined with unaligned same-ID part mosaic")
        if self.anatomical_accessory_query and self.anatomical_target_type not in {
            "decoupled_pose_parsing_teacher",
            "body_slot_privileged_ema",
        }:
            raise ValueError(
                "anatomical_accessory_query is only supported by the "
                "decoupled pose-parsing teacher or body-slot teacher"
            )
        if self.anatomical_query_distill_weight > 0 and self.anatomical_target_type != "decoupled_pose_parsing_teacher":
            raise ValueError(
                "decoupled query distillation requires anatomical_target_type='decoupled_pose_parsing_teacher'"
            )
        if (
            self.anatomical_query_relational_distill_weight > 0
            and self.anatomical_target_type != "decoupled_pose_parsing_teacher"
        ):
            raise ValueError(
                "relational query distillation requires anatomical_target_type='decoupled_pose_parsing_teacher'"
            )
        if self.clean_student_consistency_weight > 0:
            if self.anatomical_target_type != "decoupled_pose_parsing_teacher":
                raise ValueError(
                    "clean-student consistency requires anatomical_target_type='decoupled_pose_parsing_teacher'"
                )
            if self.pav_mosaic or self.pav_consistency_weight > 0:
                raise ValueError(
                    "clean-student consistency and PAV mosaic/consistency are "
                    "independent treatments and cannot be combined"
                )
        if (
            self.anatomical_query_diversity_weight > 0 or self.anatomical_part_triplet_weight > 0
        ) and self.anatomical_target_type not in {
            "decoupled_pose_parsing_teacher",
            "body_slot_privileged_ema",
        }:
            raise ValueError("query diversity and part-triplet losses require decoupled queries or body slots")
        invalid_adapter_stages = [stage for stage in self.reid_adapter_stages if stage not in {1, 2, 3}]
        if invalid_adapter_stages:
            raise ValueError(
                f"reid_adapter_stages must only contain CSL-TinyViT attention stages 1, 2, 3; "
                f"got {invalid_adapter_stages}"
            )
        if self.reid_adapter_reduction < 1:
            raise ValueError("reid_adapter_reduction must be positive")
        if not 0.0 <= self.reid_adapter_suppression_tau <= 1.0:
            raise ValueError("reid_adapter_suppression_tau must be between 0 and 1")
        if self.head_warmup_epochs < 0 or self.head_warmup_epochs > self.epochs:
            raise ValueError("head_warmup_epochs must satisfy 0 <= value <= epochs")
        if self.head_warmup_lr_mult <= 0:
            raise ValueError("head_warmup_lr_mult must be positive")
        if self.model_name.startswith(("csl_tinyvit", "mobilenetv4")):
            resolve_csl_tinyvit_ablation(self)
