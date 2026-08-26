"""Run hyperparameters, history restoration, and timing state."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, List

from boxmot.reid.backbones.families.csl_tinyvit.multilevel_suppression import (
    MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION,
)
from boxmot.reid.training.ablation import resolve_csl_tinyvit_ablation
from boxmot.reid.training.provenance import build_run_provenance
from boxmot.reid.training.resume import (
    contract_fingerprint,
    run_fingerprint,
)
from boxmot.reid.training.trainer_components.types import (
    DatasetBundle,
    LossBundle,
    ModelBundle,
    TrainMetrics,
    ValMetrics,
)
from boxmot.utils import logger as LOGGER


class _RunStateMixin:
    def _write_hparams(
        self,
        save_dir: Path,
        data: DatasetBundle,
        models: ModelBundle,
        losses: LossBundle,
    ) -> None:
        """Persist the effective, post-default training configuration."""
        recipe = self._recipe_for_bundle(models)
        losses_hparams = {
            "loss_type": self.loss_type,
            "classifier_loss": self.classifier_loss,
            "weights": {
                "id_loss_weight": self.id_loss_weight,
                "aux_ce_weight": self.aux_ce_weight,
            },
            "aux_ce_drop_epoch": self.aux_ce_drop_epoch,
        }
        if self.classifier_loss == "ce":
            losses_hparams["label_smooth"] = losses.label_smooth
        if self.loss_type == "triplet":
            losses_hparams["triplet"] = {
                "margin": self.margin,
                "soft_margin": losses.soft_margin,
            }
        if self.loss_type == "wrt":
            losses_hparams["weighted_regularized_triplet"] = {
                "pair_weighting": "softmax",
                "distance": "euclidean",
            }
        if self.classifier_loss == "arcface":
            losses_hparams["arcface"] = {
                "scale": self.arcface_scale,
                "margin": self.arcface_margin,
            }
        if self.classifier_loss == "cosface":
            losses_hparams["cosface"] = {
                "scale": self.cosface_scale,
                "margin": self.cosface_margin,
            }
        if losses.criterion_metric is not None:
            losses_hparams["weights"]["metric_loss_weight"] = self.metric_loss_weight
        if losses.criterion_adasp is not None:
            losses_hparams["adaptive_sparse_pairwise"] = {
                "weight": self.adasp_loss_weight,
                "scale": self.adasp_scale,
                "temperature": self.adasp_temperature,
                "descriptor": self._effective_metric_feature(),
            }
        if self._part_relation_enabled() or self.coarse_branch_ce_weight != 1.0 or self.fine_branch_ce_weight != 1.0:
            losses_hparams["part_relation"] = {
                "weight": self.part_relation_weight,
                "part_to_global_weight": self.part_to_global_weight,
                "teacher_momentum": self.part_relation_teacher_momentum,
                "temperature": self.part_relation_temperature,
                "coarse_branch_ce_weight": self.coarse_branch_ce_weight,
                "fine_branch_ce_weight": self.fine_branch_ce_weight,
                "parts": "four_fine_corresponding_stripes",
                "pairs": "cross_identity",
                "teacher": "training_only_ema",
            }
        if losses.criterion_csmm is not None:
            losses_hparams["cross_scale_majority_margin"] = {
                "weight": self.csmm_loss_weight,
                "margin": self.csmm_margin,
                "temperature": self.csmm_temperature,
                "topk_negatives": self.csmm_topk_negatives,
                "start_epoch": self.csmm_start_epoch,
                "ramp_end_epoch": self.csmm_ramp_end_epoch,
            }
        if losses.criterion_treeboost is not None:
            losses_hparams["treeboost_ap"] = {
                "weight": self.treeboost_loss_weight,
                "coarse_coefficient": self.treeboost_coarse_coefficient,
                "fine_coefficient": self.treeboost_fine_coefficient,
                "node_coefficient": self.treeboost_node_coefficient,
                "regression_coefficient": self.treeboost_regression_coefficient,
                "difficulty_floor": self.treeboost_difficulty_floor,
                "regression_tolerance": self.treeboost_regression_tolerance,
                "temperature": self.treeboost_temperature,
                "start_epoch": self.treeboost_start_epoch,
                "ramp_end_epoch": self.treeboost_ramp_end_epoch,
            }
        if self.global_ap_loss_weight > 0 or self._hpgrd_enabled():
            losses_hparams["global_ap"] = {
                "weight": self.global_ap_loss_weight,
                "temperature": self.global_ap_temperature,
                "topk": self.global_ap_topk,
                "memory_size": self.global_ap_memory_size,
                "momentum": self.global_ap_momentum,
                "max_age": self.global_ap_max_age,
                "start_epoch": self.global_ap_start_epoch,
                "ramp_end_epoch": self.global_ap_ramp_end_epoch,
                "decay_start_epoch": self.global_ap_decay_start_epoch,
                "decay_end_epoch": self.global_ap_decay_end_epoch,
                "descriptor": "norm_concat_bn",
                "label_source": "person_identity",
                "positive_policy": "same_identity_nonself",
                "negative_policy": "different_identity",
                "loss_inputs": ["norm_concat_bn", "sample_indices", "identity_labels"],
                "topk_policy": "hard_negatives_only_all_positives_retained",
                "dataset_sha256": self._retrieval_dataset_sha256,
            }
        if self._hpgrd_enabled():
            losses_hparams["hpgrd"] = {
                "cache_dir": self.hpgrd_cache_dir,
                "manifest_sha256": self._hpgrd_manifest_sha256,
                "dataset_sha256": self._retrieval_dataset_sha256,
                "part_names": list(getattr(self._privileged_graph_cache, "part_names", ())),
                "global_weight": self.hpgrd_global_weight,
                "part_weight": self.hpgrd_part_weight,
                "background_weight": self.hpgrd_background_weight,
                "part_drop_weight": self.hpgrd_part_drop_weight,
                "part_drop_probability": self.hpgrd_part_drop_probability,
                "gradient_fraction": self.hpgrd_gradient_fraction,
                "min_confidence": self.hpgrd_min_confidence,
                "descriptor": "norm_concat_bn",
                "teacher_stop_gradient": True,
                "part_student": "fixed_mask_pool_shared_feature_map",
                "gradient_budget_reference": "shared_late_activation",
                "intervention_bn": "train_domain_restore_buffers",
                "deployment_cost": "none",
                "start_epoch": self.global_ap_start_epoch,
                "ramp_end_epoch": self.global_ap_ramp_end_epoch,
                "decay_start_epoch": self.global_ap_decay_start_epoch,
                "decay_end_epoch": self.global_ap_decay_end_epoch,
            }
        if self.hierarchical_late_interaction:
            losses_hparams["hierarchical_late_interaction"] = {
                "matcher_weight": self.late_interaction_loss_weight,
                "distill_weight": self.late_interaction_distill_weight,
                "negative_identities": self.late_interaction_negative_identities,
                "temperature": self.late_interaction_temperature,
                "start_epoch": self.late_interaction_start_epoch,
                "ramp_end_epoch": self.late_interaction_ramp_end_epoch,
                "positive_policy": "all_cross_camera",
                "negative_policy": "detached_base_top_identity",
                "objective": "multi_positive_listwise",
                "distillation": "matcher_to_base_kl",
            }
        losses_hparams["mcpt"] = {
            "smoothness_weight": self.mcpt_smoothness_weight,
            "identity_weight": self.mcpt_identity_weight,
            "identity_decay_epoch": self.mcpt_identity_decay_epoch,
        }
        losses_hparams["jpm"] = {
            "id_loss_weight": self.jpm_id_loss_weight,
            "metric_loss_weight": self.jpm_metric_loss_weight,
            "reduction": "mean_across_local_groups",
        }
        losses_hparams["multilevel_suppression"] = {
            **({"version": (MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION)} if self.multilevel_suppression else {}),
            "weight": self.multilevel_suppression_loss_weight,
            "start_epoch": self.multilevel_suppression_start_epoch,
            "ramp_end_epoch": self.multilevel_suppression_ramp_end_epoch,
            "decay_start_epoch": self.multilevel_suppression_decay_start_epoch,
            "decay_end_epoch": self.multilevel_suppression_decay_end_epoch,
            "reduction": "half_mean_coarse_plus_half_mean_fine",
        }
        if self.compact_deployment_head:
            losses_hparams["distillation"] = {
                "id_weight": self.id_loss_weight,
                "metric_weight": self.compact_metric_loss_weight,
                "cosine_weight": self.compact_cosine_distill_weight,
                "pairwise_weight": self.compact_pairwise_distill_weight,
                "teacher_stop_gradient": True,
            }
        if self.center_loss_weight > 0:
            losses_hparams["weights"]["center_loss_weight"] = self.center_loss_weight
            losses_hparams["center_loss"] = {
                "center_table": "shared_across_streams",
                "optimizer": "sgd",
                "lr": 0.5,
            }
        loss_schedules = {}
        if self.early_id_loss_weight > 0 and self.early_id_loss_epochs > 0:
            loss_schedules["early_id_loss"] = {
                "weight": self.early_id_loss_weight,
                "epochs": self.early_id_loss_epochs,
            }
        if self.center_loss_ramp_end_epoch > 0:
            loss_schedules["center_loss_ramp"] = {
                "start_epoch": self.center_loss_ramp_start_epoch,
                "end_epoch": self.center_loss_ramp_end_epoch,
            }
        if loss_schedules:
            losses_hparams["schedules"] = loss_schedules

        sampler_hparams = {
            "p": self.p,
            "k": self.k,
            "source_balance": self.source_balance,
        }
        if self.pk_steps_per_epoch:
            sampler_hparams["steps_per_epoch"] = self.pk_steps_per_epoch
        if self.camera_aware_sampler:
            sampler_hparams["camera_aware"] = True

        hparams = {
            "run": {
                "model_name": self.model_name,
                "seed": self.seed,
                "deterministic": self.deterministic,
                "pretrained": self.pretrained,
                "pretrained_weights": self.pretrained_weights,
            },
            "data": {
                "dataset": self.dataset_name,
                "data_dir": str(self.data_dir),
                "img_size": list(self.img_size),
                "preprocess": self.preprocess,
                "num_classes": data.num_classes,
                "batch_size": self.eval_batch_size,
                "train_batch_size": self.train_batch_size,
                "eval_batch_size": self.eval_batch_size,
                "sampler": sampler_hparams,
                "num_workers": self.num_workers,
            },
            "model": {
                "is_vit": models.is_transformer,
                "is_transformer": models.is_transformer,
                "family": recipe.family,
                "training_family": recipe.family,
                "training_recipe": recipe.name,
                "timm_model_name": getattr(models.model, "timm_model_name", None),
                "requested_timm_model_name": self.timm_model_name or None,
                "timm_head_mode": self.timm_head_mode,
                "mobilenetv4_last_stride": self.mobilenetv4_last_stride,
                "mobilenetv4_neck_mode": self.mobilenetv4_neck_mode,
                "feature_fusion": self.feature_fusion,
                "pyramid_resize_mode": self.pyramid_resize_mode,
                "spatial_conv_mode": self.spatial_conv_mode,
                "post_fusion_mixer": {
                    "mode": self.post_fusion_mixer,
                    "reduction": self.post_fusion_mixer_reduction,
                    "kernel": list(self.post_fusion_mixer_kernel),
                    "gamma_init": self.post_fusion_mixer_gamma_init,
                },
                "feat_dim": self.feat_dim,
                "neck_dim": self.neck_dim,
                "attention": {
                    "window_layout": self.attention_window_layout,
                    "bias": self.attention_bias,
                    "interpolate_pretrained_bias": self.interpolate_pretrained_attention_bias,
                    "mask": self.attention_mask,
                    "shift": self.attention_shift,
                    "stage3_global": self.stage3_global,
                },
                "speed": {
                    "stage3_downsample": self.stage3_downsample,
                    "stage2_width_merge_after": self.stage2_width_merge_after,
                    "stage2_mlp_ratio": self.stage2_mlp_ratio,
                    "stage3_mlp_ratio": self.stage3_mlp_ratio,
                    "stage2_depth": self.stage2_depth,
                    "stage3_depth": self.stage3_depth,
                    "native_branch_widths": self.native_branch_widths,
                    "fine_map_dim": self.fine_map_dim,
                },
                "hierarchy": {
                    "width_first": self.width_first_hierarchy,
                },
                "identity_registers": {
                    "enabled": self.identity_registers,
                    "count": self.identity_register_count,
                    "dim": self.identity_register_dim,
                    "num_heads": self.identity_register_num_heads,
                    "dropout": self.identity_register_dropout,
                    "gate_init": self.identity_register_gate_init,
                    "diversity_weight": (self.identity_register_diversity_weight),
                    "diversity_margin": (self.identity_register_diversity_margin),
                },
                "deployment": {
                    "compact_head": self.compact_deployment_head,
                    "descriptor_dim": self.feat_dim if self.compact_deployment_head else None,
                    **(
                        {
                            "anatomical_tokens": True,
                            "anatomical_part_dim": (self.anatomical_deployment_dim),
                            "anatomical_alpha": (self.anatomical_deployment_alpha),
                        }
                        if self.anatomical_deployment
                        else {}
                    ),
                },
                "reid_adapters": {
                    "stages": list(self.reid_adapter_stages),
                    "reduction": self.reid_adapter_reduction,
                    "suppression_tau": self.reid_adapter_suppression_tau,
                },
                "head": {
                    "pool": self.head_pool,
                    "parts": list(self.head_parts),
                    "head_type": self.head_type,
                    "multiscale_channel_alpha": (self.multiscale_channel_alpha),
                    "body_slots": {
                        "mode": self.body_slot_mode,
                        "alpha": self.body_slot_alpha,
                        "visibility_floor": (self.body_slot_visibility_floor),
                    },
                    "part_pooling": self.part_pooling,
                    "coarse_branch_ce_weight": self.coarse_branch_ce_weight,
                    "fine_branch_ce_weight": self.fine_branch_ce_weight,
                    "num_part_tokens": self.num_part_tokens,
                    "evidence_num_roles": self.evidence_num_roles,
                    "anatomical_auxiliary": {
                        "enabled": self.anatomical_auxiliary,
                        "token_dim": self.anatomical_token_dim,
                        "multiscale": self.anatomical_multiscale,
                        "accessory_query": (self.anatomical_accessory_query),
                        "deployment": self.anatomical_deployment,
                        "deployment_dim": self.anatomical_deployment_dim,
                        "deployment_alpha": (self.anatomical_deployment_alpha),
                    },
                    "decouple_patterns": self.decouple_patterns,
                    "pattern_adapter_dim": self.pattern_adapter_dim,
                    "stripe_visibility": self.stripe_visibility,
                    "drop_global_aux": self.drop_global_aux,
                    "drop_global_aux_ratio": self.drop_global_aux_ratio,
                    "multilevel_suppression": {
                        "enabled": self.multilevel_suppression,
                        "ratio": self.multilevel_suppression_ratio,
                        **(
                            {"version": (MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION)}
                            if self.multilevel_suppression
                            else {}
                        ),
                        "deployment": "training_only",
                    },
                    "warmup_epochs": self.head_warmup_epochs,
                    "warmup_lr_mult": self.head_warmup_lr_mult,
                    "hierarchical_attention": {
                        "enabled": self.hierarchical_branch_attention,
                        "token_dim": self.branch_attention_token_dim,
                        "num_heads": self.branch_attention_num_heads,
                        "num_layers": self.branch_attention_num_layers,
                        "mlp_ratio": self.branch_attention_mlp_ratio,
                        "dropout": self.branch_attention_dropout,
                        "mask": "1_to_2_to_4_tree",
                        "output_init": "zero",
                    },
                    "branch_set_attention": {
                        "enabled": self.branch_set_attention,
                        "token_dim": self.branch_set_attention_token_dim,
                        "num_heads": self.branch_set_attention_num_heads,
                        "num_layers": self.branch_set_attention_num_layers,
                        "mlp_ratio": self.branch_set_attention_mlp_ratio,
                        "dropout": self.branch_set_attention_dropout,
                        "mask": "none",
                        "output_init": "zero",
                        "input_location": "post_pool_pre_reduction",
                    },
                    "multiscale_query_decoder": {
                        "enabled": self.multiscale_query_decoder,
                        "token_dim": self.query_decoder_dim,
                        "num_heads": self.query_decoder_num_heads,
                        "num_layers": self.query_decoder_num_layers,
                        "mlp_ratio": self.query_decoder_mlp_ratio,
                        "dropout": self.query_decoder_dropout,
                        "query_seeds": "existing_7_pooled_outputs",
                        "memory": "final_stage2_stage0_maps",
                        "position_encoding": "2d_sine_cosine",
                        "attention_masks": "none",
                        "memory_projection": "shared",
                        "output_init": "zero",
                    },
                    "hierarchical_late_interaction": {
                        "enabled": self.hierarchical_late_interaction,
                        "token_dim": self.late_interaction_dim,
                        "num_heads": self.late_interaction_num_heads,
                        "num_layers": self.late_interaction_num_layers,
                        "sinkhorn_iters": self.late_interaction_sinkhorn_iters,
                        "null_tokens": self.late_interaction_null_tokens,
                        "base_score_init": self.late_interaction_base_score_init,
                        "rerank_topk": self.late_interaction_rerank_topk,
                    },
                    "mcpt": {
                        "mode": self.mcpt_mode,
                        "hidden_dim": self.mcpt_hidden_dim,
                        "max_displacement": self.mcpt_max_displacement,
                        "start_epoch": self.mcpt_start_epoch,
                        "ramp_end_epoch": self.mcpt_ramp_end_epoch,
                        "predictor_source": (
                            "stage2_stage0_foreground_attention"
                            if self.mcpt_mode == "foreground_aware_shared_multiscale"
                            else "stage2_width_mean"
                        ),
                        "foreground_uniform_residual": (
                            0.25 if self.mcpt_mode == "foreground_aware_shared_multiscale" else None
                        ),
                        "fine_fusion_init": (
                            "zero_residual" if self.mcpt_mode == "foreground_aware_shared_multiscale" else None
                        ),
                        "global_unwarped": True,
                    },
                    "jpm": {
                        "enabled": self.jpm,
                        "num_groups": self.jpm_num_groups,
                        "shift": self.jpm_shift,
                        "token_dim": self.jpm_token_dim,
                        "num_heads": self.jpm_num_heads,
                        "mlp_ratio": self.jpm_mlp_ratio,
                        "dropout": self.jpm_dropout,
                        "source": "coarse_24x8_map",
                        "shared_token": "global_image_summary",
                        "deployment": "training_only",
                    },
                },
                "feature_selection": {
                    "metric_feature": self._effective_metric_feature(),
                    "inference_feature": self.inference_feature,
                },
                "branch": {
                    "aware_metric": self.branch_aware_metric,
                    "metric_part_weight": self.branch_metric_part_weight,
                    "loss_agg": self.branch_loss_agg,
                    "scale_balanced": self.scale_balanced_branches,
                },
                "evidence": {
                    "alignment_loss_weight": self.evidence_alignment_loss_weight,
                    "alignment_margin": self.evidence_alignment_margin,
                    "sinkhorn_iters": self.evidence_sinkhorn_iters,
                    "sinkhorn_temperature": self.evidence_sinkhorn_temperature,
                    "rerank_topk": self.evidence_rerank_topk,
                    "null_loss_weight": self.evidence_null_loss_weight,
                    "diversity_loss_weight": self.evidence_diversity_loss_weight,
                },
                "regularization": {
                    "drop_path_rate": self._max_drop_path(models.model),
                },
            },
            "optimization": {
                "epochs": self.epochs,
                "family": recipe.family,
                "training_family": recipe.family,
                "recipe": recipe.name,
                "optimizer": recipe.optimizer_name,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "grad_clip": recipe.grad_clip,
                "vit_lr_profile": self.vit_lr_profile,
                "mcpt_lr_multiplier": self.mcpt_lr_multiplier,
                "layer_decay": recipe.layer_decay(self),
                "backbone_lr_mult": self.backbone_lr_mult,
                "backbone_freeze_epochs": self.backbone_freeze_epochs,
                "gradual_unfreeze": {
                    "enabled": self.gradual_unfreeze,
                    "head_epochs": self.gradual_unfreeze_head_epochs,
                    "stage_epochs": self.gradual_unfreeze_stage_epochs,
                    "backbone_lr_mult": self.gradual_unfreeze_backbone_lr_mult,
                    "backbone_lr_epochs": self.gradual_unfreeze_backbone_lr_epochs,
                },
                "scheduler": {
                    "name": "CosineAnnealingLR",
                    "eta_min": self.eta_min,
                    "warmup_epochs": self.warmup_epochs,
                },
                "ema_decay": self.ema_decay,
                "ema_decay_schedule": ("update_warmup" if self.ema_decay else "none"),
            },
            "losses": losses_hparams,
            "augmentation": {
                "color_jitter": self.color_jitter,
                "gaussian_blur": self.gaussian_blur,
                "random_grayscale": self.random_grayscale,
                "random_erasing": self.random_erasing,
                "random_patch": self.random_patch,
                "random_crop_scale": self.random_crop_scale,
                "color_augmentation": self.color_augmentation,
                "background_mosaic": {
                    "enabled": self.background_mosaic,
                    "mask_dir": self.background_mosaic_mask_dir,
                    "probability": self.background_mosaic_probability,
                    "start_epoch": self.background_mosaic_start_epoch,
                    "ramp_end_epoch": self.background_mosaic_ramp_end_epoch,
                    "min_foreground_ratio": self.background_mosaic_min_foreground_ratio,
                    "max_foreground_ratio": self.background_mosaic_max_foreground_ratio,
                    "feather": self.background_mosaic_feather,
                    "dilation": self.background_mosaic_dilation,
                    "occluder_probability": (self.background_mosaic_occluder_probability),
                    "occluder_min_area": self.background_mosaic_occluder_min_area,
                    "occluder_max_area": self.background_mosaic_occluder_max_area,
                },
                "same_id_part_mosaic": {
                    "enabled": self.same_id_part_mosaic,
                    "probability": self.same_id_part_mosaic_probability,
                    "max_regions": self.same_id_part_mosaic_max_regions,
                    "min_area": self.same_id_part_mosaic_min_area,
                    "max_area": self.same_id_part_mosaic_max_area,
                    "boundary_jitter": self.same_id_part_mosaic_boundary_jitter,
                    "cross_camera_rate": self.same_id_part_mosaic_cross_camera_rate,
                    "min_unaltered": self.same_id_part_mosaic_min_unaltered,
                },
                "pav_mosaic": {
                    "enabled": self.pav_mosaic,
                    "metadata_dir": self.pav_metadata_dir,
                    "probability": self.pav_mosaic_probability,
                    "max_parts": self.pav_mosaic_max_parts,
                    "max_foreground_replacement": (self.pav_mosaic_max_foreground_replacement),
                    "cross_camera_rate": self.pav_mosaic_cross_camera_rate,
                    "different_pose_rate": self.pav_mosaic_different_pose_rate,
                    "min_keypoint_confidence": (self.pav_mosaic_min_keypoint_confidence),
                    "min_unaltered": self.pav_mosaic_min_unaltered,
                    "warmup_epochs": self.pav_mosaic_warmup_epochs,
                    "decay_start_epoch": self.pav_mosaic_decay_start_epoch,
                    "final_probability_scale": (self.pav_mosaic_final_probability_scale),
                    "consistency_weight": self.pav_consistency_weight,
                },
                "anatomical_supervision": {
                    "enabled": self.anatomical_auxiliary,
                    "metadata_dir": self.anatomical_metadata_dir,
                    "person_mask_dir": self.anatomical_person_mask_dir,
                    "min_keypoint_confidence": (self.anatomical_min_keypoint_confidence),
                    "distill_weight": self.anatomical_distill_weight,
                    "attention_weight": self.anatomical_attention_weight,
                    "foreground_weight": self.anatomical_foreground_weight,
                    "semantic_part_weight": (self.anatomical_semantic_part_weight),
                    "visibility_weight": self.anatomical_visibility_weight,
                    "contrastive_weight": (self.anatomical_contrastive_weight),
                    "descriptor_distill_weight": (self.anatomical_descriptor_distill_weight),
                    "branch_distill_weight": (self.anatomical_branch_distill_weight),
                    "branch_global_coefficient": (self.anatomical_branch_global_coefficient),
                    "branch_coarse_coefficient": (self.anatomical_branch_coarse_coefficient),
                    "branch_fine_coefficient": (self.anatomical_branch_fine_coefficient),
                    "pose_teacher_weight": (self.anatomical_pose_teacher_weight),
                    "query_distill_weight": (self.anatomical_query_distill_weight),
                    "query_relational_distill_weight": (self.anatomical_query_relational_distill_weight),
                    "clean_student_consistency_weight": (self.clean_student_consistency_weight),
                    "query_diversity_weight": (self.anatomical_query_diversity_weight),
                    "query_diversity_margin": (self.anatomical_query_diversity_margin),
                    "part_triplet_weight": (self.anatomical_part_triplet_weight),
                    "deployment": self.anatomical_deployment,
                    "deployment_dim": self.anatomical_deployment_dim,
                    "deployment_alpha": (self.anatomical_deployment_alpha),
                    "deployment_id_weight": (self.anatomical_deployment_id_weight),
                    "deployment_metric_weight": (self.anatomical_deployment_metric_weight),
                    "multiscale": self.anatomical_multiscale,
                    "accessory_query": self.anatomical_accessory_query,
                    "local_scale_weight": (self.anatomical_local_scale_weight),
                    "fine_scale_weight": (self.anatomical_fine_scale_weight),
                    "cross_scale_weight": (self.anatomical_cross_scale_weight),
                    "pose_only_reliability": (self.anatomical_pose_only_reliability),
                    "min_effective_coverage": (self.anatomical_min_effective_coverage),
                    "student_start_epoch": (self.anatomical_student_start_epoch),
                    "student_ramp_end_epoch": (self.anatomical_student_ramp_end_epoch),
                    "query_start_epoch": self.anatomical_query_start_epoch,
                    "query_ramp_end_epoch": (self.anatomical_query_ramp_end_epoch),
                    "fine_start_epoch": self.anatomical_fine_start_epoch,
                    "fine_ramp_end_epoch": self.anatomical_fine_ramp_end_epoch,
                    "decay_start_epoch": (self.anatomical_decay_start_epoch),
                    "decay_end_epoch": self.anatomical_decay_end_epoch,
                    "temperature": self.anatomical_temperature,
                    "teacher": self.anatomical_target_type,
                    "teacher_momentum": self.anatomical_teacher_momentum,
                },
            },
            "evaluation": {
                "eval_interval": self.eval_interval,
                "eval_datasets": self.eval_datasets,
                "flip_tta": self.flip_tta if self.flip_tta is not None else recipe.default_flip_tta,
                "mcpt_disabled_eval": self.mcpt_disabled_eval,
            },
            "system": {"device": str(self.device)},
            "provenance": build_run_provenance(
                models.model,
                data.dataset,
                anatomical_metadata=(self._anatomical_metadata_provenance()),
            ),
            "derived": {
                "metric_dim": losses.metric_dim,
                "classifier_dim": losses.classifier_dim,
                "n_params": sum(parameter.numel() for parameter in models.model.parameters()),
            },
        }
        resume_contract = self._resume_contract()
        hparams["resume"] = {
            "contract": resume_contract,
            "fingerprint": contract_fingerprint(resume_contract),
            "run_fingerprint": run_fingerprint(resume_contract, self.epochs),
            "target_epochs": self.epochs,
        }
        if recipe.family == "transformer":
            hparams["model"]["transformer"] = {
                "drop_path_rate": self._max_drop_path(models.model),
                "attention": hparams["model"]["attention"],
                "reid_adapters": hparams["model"]["reid_adapters"],
                "lr_profile": self.vit_lr_profile,
            }
        else:
            hparams["model"][recipe.family] = {
                "timm_model_name": getattr(models.model, "timm_model_name", None),
                "use_timm_head": getattr(models.model, "use_timm_head", None),
                "timm_head_mode": getattr(models.model, "timm_head_mode", None),
                "mobilenetv4_last_stride": getattr(
                    models.model,
                    "mobilenetv4_last_stride",
                    None,
                ),
                "mobilenetv4_neck_mode": getattr(
                    models.model,
                    "mobilenetv4_neck_mode",
                    None,
                ),
                "feature_fusion": self.feature_fusion,
                "drop_path_rate": self._max_drop_path(models.model),
            }
        if self.model_name.startswith(("csl_tinyvit", "mobilenetv4")):
            hparams["model"]["ablation"] = resolve_csl_tinyvit_ablation(self).to_dict()
        reproduction_contract = getattr(models.model, "reproduction_contract", None)
        if reproduction_contract is not None:
            hparams["model"]["reproduction_contract"] = reproduction_contract
        if self.data_specs:
            hparams["data"]["data_specs"] = [dict(spec) for spec in self.data_specs]
        path = save_dir / "hparams.json"
        self._write_json_atomic(path, hparams)
        LOGGER.info(f"Saved hyperparameters to {path}")

    def _restore_history(
        self,
        save_dir: Path,
        start_epoch: int,
    ) -> tuple[List[TrainMetrics], List[ValMetrics]]:
        """Restore persisted metric history before continuing a run."""
        history: List[TrainMetrics] = []
        val_history: List[ValMetrics] = []
        if not self.resume:
            return history, val_history

        metrics_path = save_dir / "metrics.json"
        if not metrics_path.exists():
            return history, val_history
        try:
            previous = json.loads(metrics_path.read_text())
            for train_metrics in previous.get("train", []):
                if train_metrics["epoch"] < start_epoch:
                    history.append(
                        TrainMetrics(
                            epoch=train_metrics["epoch"],
                            loss=train_metrics["loss"],
                            id_loss=train_metrics["id_loss"],
                            triplet_loss=train_metrics["triplet_loss"],
                            center_loss=train_metrics["center_loss"],
                            lr=train_metrics["lr"],
                            elapsed_s=0.0,
                            csmm_loss=train_metrics.get("csmm_loss", 0.0),
                            treeboost_loss=train_metrics.get("treeboost_loss", 0.0),
                            global_ap_loss=train_metrics.get("global_ap_loss", 0.0),
                            hpgrd_loss=train_metrics.get("hpgrd_loss", 0.0),
                            hpgrd_global_loss=train_metrics.get("hpgrd_global_loss", 0.0),
                            hpgrd_part_loss=train_metrics.get("hpgrd_part_loss", 0.0),
                            hpgrd_background_loss=train_metrics.get("hpgrd_background_loss", 0.0),
                            hpgrd_part_drop_loss=train_metrics.get("hpgrd_part_drop_loss", 0.0),
                            hpgrd_gradient_scale=train_metrics.get("hpgrd_gradient_scale", 0.0),
                            late_interaction_loss=train_metrics.get("late_interaction_loss", 0.0),
                            late_interaction_distill_loss=train_metrics.get(
                                "late_interaction_distill_loss",
                                0.0,
                            ),
                            pav_consistency_loss=train_metrics.get(
                                "pav_consistency_loss",
                                0.0,
                            ),
                            clean_student_consistency_loss=train_metrics.get(
                                "clean_student_consistency_loss",
                                0.0,
                            ),
                            anatomical_loss=train_metrics.get(
                                "anatomical_loss",
                                0.0,
                            ),
                            anatomical_distill_loss=train_metrics.get(
                                "anatomical_distill_loss",
                                0.0,
                            ),
                            anatomical_attention_loss=train_metrics.get(
                                "anatomical_attention_loss",
                                0.0,
                            ),
                            anatomical_visibility_loss=train_metrics.get(
                                "anatomical_visibility_loss",
                                0.0,
                            ),
                            anatomical_contrastive_loss=train_metrics.get(
                                "anatomical_contrastive_loss",
                                0.0,
                            ),
                            anatomical_descriptor_distill_loss=(
                                train_metrics.get(
                                    "anatomical_descriptor_distill_loss",
                                    0.0,
                                )
                            ),
                            anatomical_branch_distill_loss=train_metrics.get(
                                "anatomical_branch_distill_loss",
                                0.0,
                            ),
                            anatomical_branch_global_loss=train_metrics.get(
                                "anatomical_branch_global_loss",
                                0.0,
                            ),
                            anatomical_branch_coarse_loss=train_metrics.get(
                                "anatomical_branch_coarse_loss",
                                0.0,
                            ),
                            anatomical_branch_fine_loss=train_metrics.get(
                                "anatomical_branch_fine_loss",
                                0.0,
                            ),
                            anatomical_pose_teacher_loss=train_metrics.get(
                                "anatomical_pose_teacher_loss",
                                0.0,
                            ),
                            anatomical_semantic_foreground_loss=(
                                train_metrics.get(
                                    "anatomical_semantic_foreground_loss",
                                    0.0,
                                )
                            ),
                            anatomical_semantic_part_loss=train_metrics.get(
                                "anatomical_semantic_part_loss",
                                0.0,
                            ),
                            anatomical_local_scale_loss=train_metrics.get(
                                "anatomical_local_scale_loss",
                                0.0,
                            ),
                            anatomical_fine_scale_loss=train_metrics.get(
                                "anatomical_fine_scale_loss",
                                0.0,
                            ),
                            anatomical_cross_scale_loss=train_metrics.get(
                                "anatomical_cross_scale_loss",
                                0.0,
                            ),
                            anatomical_valid_part_fraction=train_metrics.get(
                                "anatomical_valid_part_fraction",
                                0.0,
                            ),
                            anatomical_cross_camera_anchor_fraction=(
                                train_metrics.get(
                                    "anatomical_cross_camera_anchor_fraction",
                                    0.0,
                                )
                            ),
                            anatomical_query_distill_loss=train_metrics.get(
                                "anatomical_query_distill_loss",
                                0.0,
                            ),
                            anatomical_query_relational_distill_loss=(
                                train_metrics.get(
                                    "anatomical_query_relational_distill_loss",
                                    0.0,
                                )
                            ),
                            anatomical_query_diversity_loss=train_metrics.get(
                                "anatomical_query_diversity_loss",
                                0.0,
                            ),
                            anatomical_part_triplet_loss=train_metrics.get(
                                "anatomical_part_triplet_loss",
                                0.0,
                            ),
                            anatomical_accessory_valid_fraction=(
                                train_metrics.get(
                                    "anatomical_accessory_valid_fraction",
                                    0.0,
                                )
                            ),
                            identity_register_diversity_loss=(
                                train_metrics.get(
                                    "identity_register_diversity_loss",
                                    0.0,
                                )
                            ),
                            backbone_lr=train_metrics.get("backbone_lr", 0.0),
                            head_lr=train_metrics.get("head_lr", 0.0),
                            mcpt_loss=train_metrics.get("mcpt_loss", 0.0),
                            mcpt_smoothness=train_metrics.get("mcpt_smoothness", 0.0),
                            mcpt_identity=train_metrics.get("mcpt_identity", 0.0),
                            mcpt_mean_abs_displacement=train_metrics.get("mcpt_mean_abs_displacement", 0.0),
                            mcpt_boundary_1=train_metrics.get("mcpt_boundary_1", 0.25),
                            mcpt_boundary_2=train_metrics.get("mcpt_boundary_2", 0.50),
                            mcpt_boundary_3=train_metrics.get("mcpt_boundary_3", 0.75),
                            mcpt_boundary_std=train_metrics.get("mcpt_boundary_std", 0.0),
                            mcpt_cap_fraction=train_metrics.get("mcpt_cap_fraction", 0.0),
                            mcpt_local_gate=train_metrics.get("mcpt_local_gate", 0.0),
                            mcpt_fine_gate=train_metrics.get("mcpt_fine_gate", 0.0),
                            adasp_loss=train_metrics.get("adasp_loss", 0.0),
                            part_relation_loss=train_metrics.get(
                                "part_relation_loss",
                                0.0,
                            ),
                            part_to_global_loss=train_metrics.get(
                                "part_to_global_loss",
                                0.0,
                            ),
                            jpm_id_loss=train_metrics.get(
                                "jpm_id_loss",
                                0.0,
                            ),
                            jpm_metric_loss=train_metrics.get(
                                "jpm_metric_loss",
                                0.0,
                            ),
                            multilevel_suppression_loss=train_metrics.get(
                                "multilevel_suppression_loss",
                                0.0,
                            ),
                            multilevel_suppression_weight=train_metrics.get(
                                "multilevel_suppression_weight",
                                0.0,
                            ),
                            multilevel_suppression_effective_ratio=(
                                train_metrics.get(
                                    "multilevel_suppression_effective_ratio",
                                    0.0,
                                )
                            ),
                            multilevel_suppression_coarse_erased_fraction=(
                                train_metrics.get(
                                    "multilevel_suppression_coarse_erased_fraction",
                                    0.0,
                                )
                            ),
                            multilevel_suppression_fine_erased_fraction=(
                                train_metrics.get(
                                    "multilevel_suppression_fine_erased_fraction",
                                    0.0,
                                )
                            ),
                            multilevel_suppression_global_cam_active_fraction=(
                                train_metrics.get(
                                    "multilevel_suppression_global_cam_active_fraction",
                                    0.0,
                                )
                            ),
                            multilevel_suppression_coarse_cam_active_fraction=(
                                train_metrics.get(
                                    "multilevel_suppression_coarse_cam_active_fraction",
                                    0.0,
                                )
                            ),
                        )
                    )
            for validation in previous.get("val", []):
                if validation["epoch"] >= start_epoch:
                    continue
                if "mAP" in validation:
                    val_history.append(
                        ValMetrics(
                            epoch=validation["epoch"],
                            mAP=validation["mAP"],
                            rank1=validation["rank1"],
                            rank5=validation.get("rank5", 0.0),
                            rank10=validation.get("rank10", 0.0),
                            dataset=validation.get("dataset", ""),
                            mcpt_disabled_mAP=validation.get("mcpt_disabled_mAP"),
                            mcpt_disabled_rank1=validation.get("mcpt_disabled_rank1"),
                        )
                    )
                    continue
                for dataset_name, metrics in validation.items():
                    if dataset_name == "epoch":
                        continue
                    val_history.append(
                        ValMetrics(
                            epoch=validation["epoch"],
                            mAP=metrics["mAP"],
                            rank1=metrics["rank1"],
                            rank5=metrics.get("rank5", 0.0),
                            rank10=metrics.get("rank10", 0.0),
                            dataset=dataset_name,
                            mcpt_disabled_mAP=metrics.get("mcpt_disabled_mAP"),
                            mcpt_disabled_rank1=metrics.get("mcpt_disabled_rank1"),
                        )
                    )
            LOGGER.info(f"Restored {len(history)} train and {len(val_history)} val entries from prior metrics.json")
        except Exception as exc:
            LOGGER.warning(f"Could not restore prior metrics: {exc}")
        return history, val_history

    @staticmethod
    def _average_duration(values: list[float]) -> float:
        """Return a stable average for optional timing samples."""
        return sum(values) / len(values) if values else 0.0

    def _restore_timing_averages(self, save_dir: Path) -> tuple[float, float]:
        """Load timing fallbacks so a resumed run has an ETA immediately."""
        if not self.resume:
            return 0.0, 0.0
        metrics_path = save_dir / "metrics.json"
        if not metrics_path.exists():
            return 0.0, 0.0
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return (
                float(metrics.get("average_epoch_time_s") or 0.0),
                float(metrics.get("average_eval_time_s") or 0.0),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning(f"Could not restore prior timing estimates: {exc}")
            return 0.0, 0.0

    def _training_phase_for_eta(self, epoch: int) -> str:
        """Return the compute regime used to keep timing samples comparable."""
        gradual_phase = self._gradual_unfreeze_phase(epoch)
        if gradual_phase:
            return f"gradual_{gradual_phase}"
        if self._backbone_freeze_active(epoch):
            return "backbone_frozen"
        if self._head_warmup_active(epoch):
            return "head_warmup"
        return "full_model"

    @staticmethod
    def _format_eta(duration_s: float | None) -> str:
        """Format a wall-time estimate compactly for tqdm."""
        if duration_s is None or not math.isfinite(duration_s) or duration_s < 0:
            return "calculating"
        total_seconds = int(round(duration_s))
        days, remainder = divmod(total_seconds, 24 * 60 * 60)
        hours, remainder = divmod(remainder, 60 * 60)
        minutes, seconds = divmod(remainder, 60)
        if days:
            return f"{days}d {hours}h {minutes}m"
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        """Durably replace one JSON artifact without exposing partial content."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
