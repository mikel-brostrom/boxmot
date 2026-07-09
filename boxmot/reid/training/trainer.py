"""ReID model trainer with training loop, validation, and checkpointing."""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from boxmot.reid.backbones import get_backbone_spec
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.datasets import CombinedReIDDataset, build_combined_dataset, build_dataset
from boxmot.reid.datasets.sampler import PKSampler, SourceBalancedPKSampler, parse_source_balance
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.datasets.transforms import build_test_transforms, build_train_transforms
from boxmot.reid.training.base import BaseTrainer
from boxmot.reid.training.checkpoint import CheckpointManager
from boxmot.reid.training.config import ReIDTrainConfig
from boxmot.reid.training.evaluator import (
    compute_distance_matrix,
    evaluate_ranking,
    extract_features,
    visibility_part_count,
)
from boxmot.reid.training.losses import (
    METRIC_LOSS_REGISTRY,
    ArcFaceLoss,
    CenterLoss,
    CosFaceLoss,
    CrossEntropyLabelSmooth,
)
from boxmot.reid.training.recipes import (
    TrainingRecipe,
    build_training_recipe,
    default_recipe_for_family,
)
from boxmot.utils import logger as LOGGER


def _seed_data_worker(worker_id: int) -> None:
    """Seed every worker-local RNG from the PyTorch DataLoader worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


@dataclass
class TrainMetrics:
    """Metrics collected during a training epoch."""
    epoch: int
    loss: float
    id_loss: float
    triplet_loss: float
    center_loss: float
    lr: float
    elapsed_s: float
    backbone_lr: float = 0.0
    head_lr: float = 0.0


@dataclass
class ValMetrics:
    """Metrics from validation (CMC + mAP)."""
    epoch: int
    mAP: float
    rank1: float
    rank5: float
    rank10: float
    dataset: str = ""


@dataclass
class TrainResult:
    """Final result from a ReID training run."""
    best_epoch: int
    best_mAP: float
    best_rank1: float
    weights_path: Path
    history: List[TrainMetrics] = field(default_factory=list)
    val_history: List[ValMetrics] = field(default_factory=list)


@dataclass
class DatasetBundle:
    """Loaded training dataset and its primary evaluation identity."""

    dataset: Any
    num_classes: int
    default_eval_name: str


@dataclass
class LoaderBundle:
    """Training, primary validation, and cross-domain dataloaders."""

    train: DataLoader
    query: DataLoader
    gallery: DataLoader
    cross_domain: Dict[str, Tuple[DataLoader, DataLoader]]


@dataclass
class ModelBundle:
    """Live, EMA, and validation model references."""

    model: nn.Module
    ema_model: Optional[nn.Module]
    val_model: nn.Module
    is_transformer: bool
    training_family: str = "cnn"
    recipe: Optional[TrainingRecipe] = None


@dataclass
class LossBundle:
    """Loss modules plus their resolved feature dimensions."""

    criterion_id: nn.Module
    criterion_metric: Optional[nn.Module]
    criterion_center: CenterLoss
    label_smooth: float
    soft_margin: bool
    metric_dim: int
    classifier_dim: int


@dataclass
class OptimizationBundle:
    """Optimizers, scheduler, and clipping policy."""

    optimizer: torch.optim.Optimizer
    optimizer_center: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    grad_clip: float


@dataclass
class ResumeState:
    """Mutable progress restored from a checkpoint."""

    start_epoch: int = 1
    best_mAP: float = 0.0
    best_rank1: float = 0.0
    best_epoch: int = 0


class ReIDTrainer(BaseTrainer):
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
        early_id_loss_weight: float = 0.0,
        early_id_loss_epochs: int = 0,
        center_loss_ramp_start_epoch: int = 0,
        center_loss_ramp_end_epoch: int = 0,
        aux_ce_weight: float = 1.0,
        aux_ce_drop_epoch: int = 0,
        branch_loss_agg: str = "mean",
        eta_min: float = 1e-7,
        pretrained: bool = True,
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
        flip_tta: Optional[bool] = None,
        resume: Optional[str] = None,
        metric_feature: str = "auto",
        inference_feature: str = "concat_bn",
        feature_fusion: str = "last3",
        post_fusion_mixer: str = "none",
        post_fusion_mixer_reduction: int = 4,
        post_fusion_mixer_kernel: tuple[int, int] = (5, 3),
        post_fusion_mixer_gamma_init: float = 0.0,
        feat_dim: int = 512,
        neck_dim: int = 512,
        drop_path_rate: float = 0.1,
        attention_window_layout: str = "legacy",
        attention_bias: str = "absolute",
        attention_mask: bool = False,
        attention_shift: bool = False,
        stage3_global: bool = False,
        vit_lr_profile: str = "layer_decay",
        backbone_freeze_epochs: int = 0,
        gradual_unfreeze: bool = False,
        gradual_unfreeze_head_epochs: int = 5,
        gradual_unfreeze_stage_epochs: int = 10,
        gradual_unfreeze_backbone_lr_mult: float = 0.1,
        gradual_unfreeze_backbone_lr_epochs: int = 5,
        branch_aware_metric: bool = False,
        branch_metric_part_weight: float = 0.5,
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
        part_pooling: str = "stripes",
        num_part_tokens: int = 4,
        decouple_patterns: bool = False,
        pattern_adapter_dim: int = 128,
        stripe_visibility: bool = False,
        drop_global_aux: bool = False,
        drop_global_aux_ratio: float = 0.25,
        reid_adapter_stages: tuple[int, ...] = (),
        reid_adapter_reduction: int = 4,
        head_warmup_epochs: int = 0,
        head_warmup_lr_mult: float = 2.0,
        explicit_hparams: Iterable[str] | None = None,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_dir = str(data_dir)
        self.data_specs = tuple(self._normalize_data_spec(spec) for spec in (data_specs or ()))
        self._data_roots_by_name = {
            self._dataset_lookup_key(spec["name"]): spec["root"] for spec in self.data_specs
        }
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
        self.device = torch.device(device)
        self.project = Path(project)
        self.name = name
        self.requested_num_workers = int(num_workers)
        if self.requested_num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.num_workers = 0 if self.device.type in {"cpu", "mps"} else self.requested_num_workers
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
        self.flip_tta = flip_tta
        self.resume = resume
        self.metric_feature = str(metric_feature).lower()
        self.inference_feature = str(inference_feature).lower()
        self.feature_fusion = str(feature_fusion).lower()
        self.post_fusion_mixer = self._normalize_post_fusion_mixer(post_fusion_mixer)
        self.post_fusion_mixer_reduction = int(post_fusion_mixer_reduction)
        self.post_fusion_mixer_kernel = self._normalize_int_pair(post_fusion_mixer_kernel)
        self.post_fusion_mixer_gamma_init = float(post_fusion_mixer_gamma_init)
        self.feat_dim = int(feat_dim)
        self.neck_dim = int(neck_dim)
        self.drop_path_rate = float(drop_path_rate)
        self.attention_window_layout = str(attention_window_layout).lower()
        self.attention_bias = str(attention_bias).lower()
        self.attention_mask = bool(attention_mask)
        self.attention_shift = bool(attention_shift)
        self.stage3_global = bool(stage3_global)
        self.vit_lr_profile = str(vit_lr_profile).lower()
        self.backbone_freeze_epochs = int(backbone_freeze_epochs)
        self.gradual_unfreeze = bool(gradual_unfreeze)
        self.gradual_unfreeze_head_epochs = int(gradual_unfreeze_head_epochs)
        self.gradual_unfreeze_stage_epochs = int(gradual_unfreeze_stage_epochs)
        self.gradual_unfreeze_backbone_lr_mult = float(gradual_unfreeze_backbone_lr_mult)
        self.gradual_unfreeze_backbone_lr_epochs = int(gradual_unfreeze_backbone_lr_epochs)
        self.branch_aware_metric = bool(branch_aware_metric)
        self.branch_metric_part_weight = float(branch_metric_part_weight)
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
        self.part_pooling = str(part_pooling).lower()
        if self.part_pooling in {"soft_stripes", "overlapping_stripes"}:
            self.part_pooling = "overlap_stripes"
        if self.part_pooling in {"semantic", "semantic_tokens", "semantic_visibility"}:
            self.part_pooling = "semantic_parts"
        if self.part_pooling not in {"stripes", "overlap_stripes", "tokens", "semantic_parts"}:
            raise ValueError("part_pooling must be one of: stripes, overlap_stripes, tokens, semantic_parts")
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

    @property
    def train_batch_size(self) -> int:
        """Effective PK-sampled training batch size."""
        if self.source_balance_groups:
            return sum(group.batch_size for group in self.source_balance_groups)
        return self.p * self.k

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
            "early_id_loss_weight",
            "aux_ce_weight",
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
        if self.aux_ce_drop_epoch < 0 or self.aux_ce_drop_epoch > self.epochs:
            raise ValueError("aux_ce_drop_epoch must satisfy 0 <= value <= epochs")
        metric_active = self.loss_type != "softmax" and self.metric_loss_weight > 0
        center_active = self.loss_type != "ms" and self.center_loss_weight > 0
        if self.id_loss_weight == 0 and not metric_active and not center_active:
            raise ValueError("At least one ID, metric, or center loss weight must be positive")
        for name in ("random_grayscale", "random_erasing"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must satisfy 0 <= value <= 1")
        if self.random_crop_scale < 1.0:
            raise ValueError("random_crop_scale must be >= 1.0")
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        valid_metric_features = {
            "auto",
            "global",
            "raw_mean",
            "raw_concat",
            "concat_bn",
            "dse_weighted",
            "dse_mix",
        }
        if self.metric_feature not in valid_metric_features:
            raise ValueError("Unsupported metric_feature")
        if self.inference_feature not in {
            "concat_bn",
            "norm_concat_bn",
            "global",
            "raw_mean",
            "raw_concat",
            "visibility_weighted_parts",
            "evidence_sinkhorn",
            "dse_weighted",
            "dse_mix",
        }:
            raise ValueError("Unsupported inference_feature")
        if self.feature_fusion not in {
            "final",
            "last2",
            "last3",
            "last4_layer0_target",
            "last3_stage2_target",
            "last3_fpn_stage2",
            "last3_pafpn_stage2",
            "last4_fpn_layer0_target",
            "global_final_parts_stage2",
            "late_concat_stage2",
            "weighted_last2",
            "weighted_last3",
            "normpres_last2",
            "normpres_last3",
            "dynamic_last3",
            "dynamic_last3_scale_token",
            "dpt_fpn",
        }:
            raise ValueError("Unsupported feature_fusion")
        if self.post_fusion_mixer_reduction < 1:
            raise ValueError("post_fusion_mixer_reduction must be positive")
        if len(self.post_fusion_mixer_kernel) != 2 or any(value <= 0 for value in self.post_fusion_mixer_kernel):
            raise ValueError("post_fusion_mixer_kernel must contain two positive integers")
        if any(value % 2 == 0 for value in self.post_fusion_mixer_kernel):
            raise ValueError("post_fusion_mixer_kernel values must be odd")
        if not math.isfinite(self.post_fusion_mixer_gamma_init):
            raise ValueError("post_fusion_mixer_gamma_init must be finite")
        if self.head_pool not in {"avg", "gem", "dse", "gelu_gem", "relu_gem", "softplus_gem"}:
            raise ValueError("Unsupported head_pool")
        if self.head_type not in {"standard", "gpc_lite"}:
            raise ValueError("head_type must be one of: standard, gpc_lite")
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
        if self.drop_global_aux and self.head_type != "standard":
            raise ValueError("drop_global_aux requires head_type='standard'")
        if self.drop_global_aux and self.classifier_loss != "ce":
            raise ValueError("drop_global_aux requires classifier_loss='ce'")
        if self.stripe_visibility:
            local_granularities = tuple(part for part in self.head_parts if part != 1)
            if self.part_pooling != "stripes" or len(local_granularities) != 1:
                raise ValueError(
                    "stripe_visibility requires fixed stripes with exactly one local granularity"
                )
        if self.part_pooling == "semantic_parts" and not any(part > 1 for part in self.head_parts):
            raise ValueError("semantic_parts pooling requires at least one local part in head_parts")
        if not 0 < self.drop_global_aux_ratio <= 1:
            raise ValueError("drop_global_aux_ratio must satisfy 0 < value <= 1")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("drop_path_rate must satisfy 0 <= value < 1")
        if self.attention_window_layout not in {"legacy", "rect"}:
            raise ValueError("Unsupported attention_window_layout")
        if self.attention_bias not in {"absolute", "signed_factorized"}:
            raise ValueError("Unsupported attention_bias")
        if self.vit_lr_profile not in {"layer_decay", "reid_lrd"}:
            raise ValueError("vit_lr_profile must be one of: layer_decay, reid_lrd")
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
                raise ValueError(
                    "gradual_unfreeze_stage_epochs must be >= gradual_unfreeze_head_epochs"
                )
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
        invalid_adapter_stages = [stage for stage in self.reid_adapter_stages if stage not in {1, 2, 3}]
        if invalid_adapter_stages:
            raise ValueError(
                f"reid_adapter_stages must only contain CSL-TinyViT attention stages 1, 2, 3; "
                f"got {invalid_adapter_stages}"
            )
        if self.reid_adapter_reduction < 1:
            raise ValueError("reid_adapter_reduction must be positive")
        if self.head_warmup_epochs < 0 or self.head_warmup_epochs > self.epochs:
            raise ValueError("head_warmup_epochs must satisfy 0 <= value <= epochs")
        if self.head_warmup_lr_mult <= 0:
            raise ValueError("head_warmup_lr_mult must be positive")

    def _memory_utilization(self) -> Optional[float]:
        """Return this process's accelerator-memory utilization when available."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(self.device).total_memory
            return torch.cuda.memory_reserved(self.device) / total if total > 0 else None
        if self.device.type == "mps" and torch.backends.mps.is_available():
            total = torch.mps.recommended_max_memory()
            return torch.mps.driver_allocated_memory() / total if total > 0 else None
        return None

    def _clear_memory(self, *, force: bool = False, threshold: Optional[float] = None) -> bool:
        """Collect garbage and clear the accelerator cache on OOM or high utilization."""
        if self.device.type not in {"cuda", "mps"}:
            return False
        if not force:
            utilization = self._memory_utilization()
            if threshold is None or utilization is None or utilization < threshold:
                return False

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        else:
            torch.mps.empty_cache()
        return True

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        """Return whether an exception represents a CUDA/MPS out-of-memory failure."""
        return isinstance(exc, torch.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )

    def _handle_oom(self, exc: BaseException, *optimizers) -> bool:
        """Release gradients and cached memory after an accelerator OOM."""
        if not self._is_oom_error(exc):
            return False
        for optimizer in optimizers:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        self._clear_memory(force=True)
        return True

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed every RNG used by model training and data augmentation."""
        seed = int(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    @staticmethod
    def _configure_determinism(enabled: bool) -> None:
        """Configure PyTorch backends to require or permit nondeterministic algorithms."""
        enabled = bool(enabled)
        if enabled:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = not enabled
            torch.backends.cudnn.deterministic = enabled
        torch.use_deterministic_algorithms(enabled)

    def _seed_training_epoch(self, epoch: int, loader: DataLoader) -> None:
        """Seed one epoch independently so fresh and resumed runs agree."""
        epoch_seed = self.seed + int(epoch)
        self._seed_everything(epoch_seed)
        self._train_generator.manual_seed(epoch_seed)
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    @staticmethod
    def _capture_rng_state() -> dict:
        """Capture process and accelerator RNG states for checkpoint resume."""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            state["mps"] = torch.mps.get_rng_state()
        return state

    @staticmethod
    def _restore_rng_state(state: Optional[dict]) -> None:
        """Restore RNG state saved in a training checkpoint."""
        if not state:
            LOGGER.warning(
                "Checkpoint has no RNG state; the next epoch will use its configured seeded RNG stream"
            )
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])
        if torch.backends.mps.is_available() and state.get("mps") is not None:
            torch.mps.set_rng_state(state["mps"].cpu())

    @staticmethod
    def _normalize_head_parts(head_parts) -> tuple[int, ...]:
        """Normalize CSL-TinyViT head part granularities from CLI/API inputs."""
        if isinstance(head_parts, str):
            parts = [part for part in head_parts.replace(";", ",").split(",") if part.strip()]
            return tuple(int(part) for part in parts)
        if isinstance(head_parts, int):
            return (int(head_parts),)
        return tuple(int(part) for part in head_parts)

    @staticmethod
    def _normalize_int_pair(value) -> tuple[int, int]:
        """Normalize integer-pair CLI/API inputs without deduplicating equal values."""
        if isinstance(value, int):
            return (int(value), int(value))
        if isinstance(value, str):
            tokens = [part for part in value.replace(";", ",").split(",") if part.strip()]
            if len(tokens) == 1:
                tokens = tokens * 2
            if len(tokens) != 2:
                raise ValueError(f"Expected one or two comma-separated integers, got {value!r}")
            return (int(tokens[0]), int(tokens[1]))
        values = tuple(int(part) for part in value)
        if len(values) == 1:
            return (values[0], values[0])
        if len(values) != 2:
            raise ValueError(f"Expected one or two integers, got {value!r}")
        return values

    @staticmethod
    def _normalize_post_fusion_mixer(mixer: str) -> str:
        """Normalize post-fusion local mixer aliases."""
        normalized = str(mixer).lower()
        if normalized in {"", "none", "off", "identity"}:
            return "none"
        if normalized in {"dwconv", "local", "dwconv5x3"}:
            return "dwconv"
        raise ValueError("post_fusion_mixer must be one of: none, dwconv")

    @staticmethod
    def _normalize_adapter_stages(stages) -> tuple[int, ...]:
        """Normalize CSL-TinyViT ReID adapter stage indices from CLI/API inputs."""
        if stages is None:
            return ()
        if isinstance(stages, str):
            if stages.lower() in {"", "none", "off"}:
                return ()
            parts = [part for part in stages.replace(";", ",").split(",") if part.strip()]
        elif isinstance(stages, int):
            parts = [stages]
        else:
            parts = list(stages)
        return tuple(dict.fromkeys(int(part) for part in parts))

    def _prepare_runtime(self) -> None:
        """Initialize deterministic process state and log effective runtime settings."""
        self._configure_determinism(self.deterministic)
        self._seed_everything(self.seed)
        LOGGER.info(
            f"Training reproducibility: seed={self.seed}, deterministic={self.deterministic}"
        )
        if self.num_workers != self.requested_num_workers:
            LOGGER.info(
                f"{self.device.type.upper()} training: forcing dataloader workers "
                f"from {self.requested_num_workers} to 0"
            )
        if self.source_balance:
            LOGGER.info(
                f"Batch sizes: train={self.train_batch_size} "
                f"(source_balance={self.source_balance}), eval={self.eval_batch_size}"
            )
        else:
            LOGGER.info(
                f"Batch sizes: train={self.train_batch_size} (p={self.p} x k={self.k}), "
                f"eval={self.eval_batch_size}"
            )

    @staticmethod
    def _normalize_data_spec(spec: dict[str, Any]) -> dict[str, Any]:
        name = str(spec.get("name") or spec.get("dataset") or "").strip()
        root = str(spec.get("root") or spec.get("path") or "").strip()
        if not name:
            raise ValueError("data_specs entries must define a dataset name")
        if not root:
            raise ValueError(f"data_specs entry for '{name}' must define a root/path")

        normalized = {"name": name, "root": str(Path(root).expanduser().resolve())}
        for key in ("config", "train", "val", "query", "gallery"):
            if spec.get(key) is not None:
                normalized[key] = str(spec[key])
        return normalized

    @staticmethod
    def _dataset_lookup_key(name: str) -> str:
        key = str(name).lower().replace("-", "").replace("_", "")
        if key in {"dukemtmcreid", "dukemtmc", "duke"}:
            return "duke"
        if key in {"mot171501", "mot17market1501"}:
            return "mot171501"
        if key in {"veri776", "veri"}:
            return "veri"
        if key in {"cuhk03", "cuhk03np"}:
            return "cuhk03"
        if key == "msmt17merged":
            return "msmt17merged"
        return key

    def _data_root_for_name(self, name: str) -> str:
        return self._data_roots_by_name.get(self._dataset_lookup_key(name), self.data_dir)

    def _build_dataset_bundle(self) -> DatasetBundle:
        """Load the configured dataset and identify the primary validation split."""
        if self.data_specs:
            dataset_names = [spec["name"] for spec in self.data_specs]
            if len(self.data_specs) > 1:
                LOGGER.info(f"Loading combined dataset from data specs: {dataset_names}")
                datasets = [build_dataset(spec["name"], spec["root"]) for spec in self.data_specs]
                dataset = CombinedReIDDataset(datasets)
            else:
                spec = self.data_specs[0]
                LOGGER.info(f"Loading dataset '{spec['name']}' from {spec['root']}")
                dataset = build_dataset(spec["name"], spec["root"])
            default_eval_name = dataset_names[0].lower()
        else:
            dataset_names = [name.strip() for name in self.dataset_name.split(",") if name.strip()]
            if len(dataset_names) > 1:
                LOGGER.info(f"Loading combined dataset from: {dataset_names}")
                dataset = build_combined_dataset(dataset_names, self.data_dir)
                default_eval_name = dataset_names[0].lower()
            else:
                LOGGER.info(f"Loading dataset '{self.dataset_name}' from {self.data_dir}")
                dataset = build_dataset(self.dataset_name, self.data_dir)
                default_eval_name = self.dataset_name.lower()
        LOGGER.info(dataset.summary())
        return DatasetBundle(
            dataset=dataset,
            num_classes=dataset.num_train_pids,
            default_eval_name=default_eval_name,
        )

    def _build_model_bundle(self, num_classes: int) -> ModelBundle:
        """Build live and optional EMA models with finalized training-family defaults."""
        LOGGER.info(f"Building model '{self.model_name}' with {num_classes} classes, loss='{self.loss_type}'")
        pre_build_recipe = self._resolve_training_recipe_for_model_name()
        if pre_build_recipe is not None:
            pre_build_recipe.apply_pre_build_defaults(self)
            self._validate_config()
        model = self._build_model(num_classes).to(self.device)
        if hasattr(model, "img_size") and model.img_size != self.img_size:
            LOGGER.info(f"Syncing img_size with model architecture: {self.img_size} → {model.img_size}")
            self.img_size = model.img_size

        recipe = self._resolve_training_recipe(model)
        recipe.apply_defaults(self)
        self._validate_config()

        ema_model: Optional[nn.Module] = None
        if self.ema_decay:
            ema_model = copy.deepcopy(model)
            for parameter in ema_model.parameters():
                parameter.requires_grad_(False)
            LOGGER.info(f"EMA model enabled (decay={self.ema_decay})")
        return ModelBundle(
            model=model,
            ema_model=ema_model,
            val_model=ema_model if ema_model is not None else model,
            is_transformer=recipe.family == "transformer",
            training_family=recipe.family,
            recipe=recipe,
        )

    def _build_loader_bundle(self, data: DatasetBundle) -> LoaderBundle:
        """Build train, primary validation, and optional cross-domain loaders."""
        train_loader = self._build_train_loader(data.dataset)
        query_loader, gallery_loader = self._build_test_loaders(data.dataset)
        cross_domain: Dict[str, Tuple[DataLoader, DataLoader]] = {}
        for eval_dataset_name in self.eval_datasets:
            if eval_dataset_name.strip().lower() == data.default_eval_name:
                continue
            try:
                eval_root = self._data_root_for_name(eval_dataset_name)
                eval_dataset = build_dataset(eval_dataset_name, eval_root)
                query, gallery = self._build_test_loaders(eval_dataset)
                cross_domain[eval_dataset_name] = (query, gallery)
                LOGGER.info(
                    f"Cross-domain eval: loaded '{eval_dataset_name}' "
                    f"from {eval_root} ({eval_dataset.query.num_imgs}q / {eval_dataset.gallery.num_imgs}g)"
                )
            except Exception as exc:
                LOGGER.warning(f"Skipping cross-domain eval dataset '{eval_dataset_name}': {exc}")
        return LoaderBundle(
            train=train_loader,
            query=query_loader,
            gallery=gallery_loader,
            cross_domain=cross_domain,
        )

    def _build_loss_bundle(self, model: ModelBundle, num_classes: int) -> LossBundle:
        """Resolve and construct ID, metric, and center-loss modules."""
        recipe = self._recipe_for_bundle(model)
        label_smooth = recipe.resolve_label_smooth(self, self.label_smooth)

        soft_margin = self._use_soft_margin_triplet(recipe.default_triplet_soft_margin)
        criterion_metric = None
        if self.loss_type in METRIC_LOSS_REGISTRY:
            metric_loss_class = METRIC_LOSS_REGISTRY[self.loss_type]
            criterion_metric = (
                metric_loss_class(margin=self.margin, soft_margin=soft_margin)
                if self.loss_type == "triplet"
                else metric_loss_class()
            )
            LOGGER.info(f"Metric loss: {metric_loss_class.__name__}")

        if self.loss_type == "ms" and self.center_loss_weight > 0:
            LOGGER.info("MS loss active: disabling center loss (redundant)")
            self.center_loss_weight = 0

        metric_dim = self._probe_feat_dim(model.model)
        classifier_dim = (
            self._probe_classifier_feat_dim(model.model)
            if self.classifier_loss != "ce"
            else metric_dim
        )
        criterion_id = self._build_classifier_loss(
            num_classes,
            classifier_dim,
            label_smooth,
        ).to(self.device)
        criterion_center = CenterLoss(num_classes, metric_dim).to(self.device)
        return LossBundle(
            criterion_id=criterion_id,
            criterion_metric=criterion_metric,
            criterion_center=criterion_center,
            label_smooth=label_smooth,
            soft_margin=soft_margin,
            metric_dim=metric_dim,
            classifier_dim=classifier_dim,
        )

    def _build_optimization_bundle(
        self,
        model: ModelBundle,
        losses: LossBundle,
    ) -> OptimizationBundle:
        """Build model/center optimizers and the cosine scheduler."""
        classifier_parameters = (
            list(losses.criterion_id.parameters())
            if self.classifier_loss != "ce"
            else []
        )
        recipe = self._recipe_for_bundle(model)
        parameter_groups = recipe.build_param_groups(self, model.model)
        if classifier_parameters:
            parameter_groups.append(recipe.classifier_param_group(self, classifier_parameters))
        optimizer = recipe.build_optimizer(self, parameter_groups)
        if recipe.family == "transformer":
            LOGGER.info(
                f"Transformer training: {recipe.optimizer_name} (lr={self.lr:.1e}, wd={self.weight_decay}), "
                f"lr_profile={self.vit_lr_profile}, grad clip={recipe.grad_clip:.1f}, DropPath enabled"
            )
        else:
            LOGGER.info(
                f"{recipe.family.upper()} training: {recipe.optimizer_name} "
                f"(lr={self.lr:.1e}, wd={self.weight_decay}), "
                f"grad clip={recipe.grad_clip:.1f}"
            )

        optimizer_center = torch.optim.SGD(losses.criterion_center.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=self.eta_min,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["_base_lr"] = parameter_group["lr"]
            if self.warmup_epochs > 0:
                parameter_group["lr"] /= self.warmup_epochs
        return OptimizationBundle(
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            scheduler=scheduler,
            grad_clip=recipe.grad_clip,
        )

    def _resolve_resume_path(self) -> Optional[Path]:
        """Resolve a resume directory to its preferred checkpoint."""
        if not self.resume:
            return None
        resume_path = Path(self.resume)
        if resume_path.is_dir():
            if (resume_path / "last.pt").exists():
                return resume_path / "last.pt"
            if (resume_path / "best.pt").exists():
                LOGGER.warning("last.pt not found, falling back to best.pt (optimizer state will be reset)")
                return resume_path / "best.pt"
            raise FileNotFoundError(f"No checkpoint found in: {resume_path}")
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        return resume_path

    def _restore_if_needed(
        self,
        model: ModelBundle,
        loaders: LoaderBundle,
        losses: LossBundle,
        optimization: OptimizationBundle,
    ) -> ResumeState:
        """Restore live/EMA model state and compatible optimizer state."""
        resume_path = self._resolve_resume_path()
        if resume_path is None:
            return ResumeState()

        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
        resumable = checkpoint.get("resumable", "optimizer" in checkpoint)
        model.model.load_state_dict(checkpoint["state_dict"], strict=resumable)
        if not resumable:
            LOGGER.warning(
                f"{resume_path} is a weights-only checkpoint; optimizer and scheduler state will be reset"
            )
        self._restore_center_loss_state(
            checkpoint,
            losses.criterion_center,
            model.model,
            loaders.train,
            resume_path,
        )
        self._restore_classifier_loss_state(
            checkpoint,
            losses.criterion_id,
            resume_path,
        )
        if "optimizer" in checkpoint:
            optimization.optimizer.load_state_dict(checkpoint["optimizer"])
        if "optimizer_center" in checkpoint:
            optimization.optimizer_center.load_state_dict(checkpoint["optimizer_center"])
        resumed_epoch = int(checkpoint.get("epoch", 0))
        optimization.scheduler = self._build_resume_scheduler(
            optimization.optimizer,
            resumed_epoch,
            resume_path,
            checkpoint,
        )
        if model.ema_model is not None:
            model.ema_model.load_state_dict(
                checkpoint.get("ema_state_dict", checkpoint["state_dict"]),
                strict=resumable,
            )
        self._restore_rng_state(checkpoint.get("rng_state"))
        best_mAP = float(checkpoint.get("best_mAP") or checkpoint.get("mAP", 0.0))
        best_rank1 = float(checkpoint.get("rank1", 0.0))
        LOGGER.info(
            f"Resumed from {resume_path} (epoch {resumed_epoch}, "
            f"mAP={best_mAP:.2%}, R1={best_rank1:.2%})"
        )
        return ResumeState(
            start_epoch=resumed_epoch + 1,
            best_mAP=best_mAP,
            best_rank1=best_rank1,
            best_epoch=resumed_epoch,
        )

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
        if self.center_loss_weight > 0 and self.loss_type != "ms":
            losses_hparams["weights"]["center_loss_weight"] = self.center_loss_weight
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

        hparams = {
            "run": {
                "model_name": self.model_name,
                "seed": self.seed,
                "deterministic": self.deterministic,
                "pretrained": self.pretrained,
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
                "sampler": {
                    "p": self.p,
                    "k": self.k,
                    "source_balance": self.source_balance,
                },
                "num_workers": self.num_workers,
            },
            "model": {
                "is_vit": models.is_transformer,
                "is_transformer": models.is_transformer,
                "family": recipe.family,
                "training_family": recipe.family,
                "training_recipe": recipe.name,
                "timm_model_name": getattr(models.model, "timm_model_name", None),
                "feature_fusion": self.feature_fusion,
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
                    "mask": self.attention_mask,
                    "shift": self.attention_shift,
                    "stage3_global": self.stage3_global,
                },
                "reid_adapters": {
                    "stages": list(self.reid_adapter_stages),
                    "reduction": self.reid_adapter_reduction,
                },
                "head": {
                    "pool": self.head_pool,
                    "parts": list(self.head_parts),
                    "head_type": self.head_type,
                    "part_pooling": self.part_pooling,
                    "num_part_tokens": self.num_part_tokens,
                    "evidence_num_roles": self.evidence_num_roles,
                    "decouple_patterns": self.decouple_patterns,
                    "pattern_adapter_dim": self.pattern_adapter_dim,
                    "stripe_visibility": self.stripe_visibility,
                    "drop_global_aux": self.drop_global_aux,
                    "drop_global_aux_ratio": self.drop_global_aux_ratio,
                    "warmup_epochs": self.head_warmup_epochs,
                    "warmup_lr_mult": self.head_warmup_lr_mult,
                },
                "feature_selection": {
                    "metric_feature": self._effective_metric_feature(),
                    "inference_feature": self.inference_feature,
                },
                "branch": {
                    "aware_metric": self.branch_aware_metric,
                    "metric_part_weight": self.branch_metric_part_weight,
                    "loss_agg": self.branch_loss_agg,
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
                    "drop_path_rate": self._max_drop_path(models.model)
                    if recipe.family == "transformer"
                    else 0.0,
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
                "layer_decay": recipe.layer_decay(self),
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
            },
            "evaluation": {
                "eval_interval": self.eval_interval,
                "eval_datasets": self.eval_datasets,
                "flip_tta": self.flip_tta if self.flip_tta is not None else recipe.default_flip_tta,
            },
            "system": {"device": str(self.device)},
            "derived": {
                "metric_dim": losses.metric_dim,
                "classifier_dim": losses.classifier_dim,
                "n_params": sum(parameter.numel() for parameter in models.model.parameters()),
            },
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
                "feature_fusion": self.feature_fusion,
            }
        if self.data_specs:
            hparams["data"]["data_specs"] = [dict(spec) for spec in self.data_specs]
        path = save_dir / "hparams.json"
        path.write_text(json.dumps(hparams, indent=2))
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
                    history.append(TrainMetrics(
                        epoch=train_metrics["epoch"],
                        loss=train_metrics["loss"],
                        id_loss=train_metrics["id_loss"],
                        triplet_loss=train_metrics["triplet_loss"],
                        center_loss=train_metrics["center_loss"],
                        lr=train_metrics["lr"],
                        elapsed_s=0.0,
                        backbone_lr=train_metrics.get("backbone_lr", 0.0),
                        head_lr=train_metrics.get("head_lr", 0.0),
                    ))
            for validation in previous.get("val", []):
                if validation["epoch"] >= start_epoch:
                    continue
                if "mAP" in validation:
                    val_history.append(ValMetrics(
                        epoch=validation["epoch"],
                        mAP=validation["mAP"],
                        rank1=validation["rank1"],
                        rank5=validation.get("rank5", 0.0),
                        rank10=validation.get("rank10", 0.0),
                        dataset=validation.get("dataset", ""),
                    ))
                    continue
                for dataset_name, metrics in validation.items():
                    if dataset_name == "epoch":
                        continue
                    val_history.append(ValMetrics(
                        epoch=validation["epoch"],
                        mAP=metrics["mAP"],
                        rank1=metrics["rank1"],
                        rank5=metrics.get("rank5", 0.0),
                        rank10=metrics.get("rank10", 0.0),
                        dataset=dataset_name,
                    ))
            LOGGER.info(
                f"Restored {len(history)} train and {len(val_history)} "
                "val entries from prior metrics.json"
            )
        except Exception as exc:
            LOGGER.warning(f"Could not restore prior metrics: {exc}")
        return history, val_history

    @staticmethod
    def _average_duration(values: list[float]) -> float:
        """Return a stable average for optional timing samples."""
        return sum(values) / len(values) if values else 0.0

    def _fit(
        self,
        *,
        save_dir: Path,
        data: DatasetBundle,
        models: ModelBundle,
        loaders: LoaderBundle,
        losses: LossBundle,
        optimization: OptimizationBundle,
        state: ResumeState,
        run_started_at: float,
    ) -> TrainResult:
        """Run epoch orchestration after all setup and restore work is complete."""
        from tqdm import tqdm

        best_weights = save_dir / "best.pt"
        history, val_history = self._restore_history(save_dir, state.start_epoch)
        latest_primary_val = next(
            (
                metrics
                for metrics in reversed(val_history)
                if metrics.dataset == data.default_eval_name
            ),
            None,
        )
        epoch_durations_s: list[float] = []
        eval_durations_s: list[float] = []
        epoch_bar = tqdm(
            range(state.start_epoch, self.epochs + 1),
            desc="Training",
            unit="epoch",
            initial=state.start_epoch - 1,
            total=self.epochs,
        )
        for epoch in epoch_bar:
            try:
                metrics = self._train_epoch(
                    epoch,
                    models.model,
                    loaders.train,
                    losses.criterion_id,
                    losses.criterion_metric,
                    losses.criterion_center,
                    optimization.optimizer,
                    optimization.optimizer_center,
                    optimization.scheduler,
                    ema_model=models.ema_model,
                    grad_clip=optimization.grad_clip,
                )
            except RuntimeError as exc:
                if not self._handle_oom(
                    exc,
                    optimization.optimizer,
                    optimization.optimizer_center,
                ):
                    raise
                raise RuntimeError(
                    f"{self.device.type.upper()} out of memory during training. "
                    "Cached memory was cleared; reduce --batch-size or --p-ids and resume from last.pt."
                ) from exc
            history.append(metrics)
            epoch_durations_s.append(metrics.elapsed_s)
            self._clear_memory(threshold=self.MEMORY_CLEAR_THRESHOLD)
            epoch_bar.set_postfix(
                loss=f"{metrics.loss:.4f}",
                id=f"{metrics.id_loss:.4f}",
                tri=f"{metrics.triplet_loss:.4f}",
                lr=f"{metrics.lr:.6f}",
            )

            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                if models.ema_model is not None:
                    self._calibrate_bn(models.val_model, loaders.train)
                eval_started_at = time.monotonic()
                try:
                    val = self._validate(
                        epoch,
                        models.val_model,
                        loaders.query,
                        loaders.gallery,
                    )
                except RuntimeError as exc:
                    if not self._handle_oom(exc):
                        raise
                    raise RuntimeError(
                        f"{self.device.type.upper()} out of memory during validation. "
                        "Cached memory was cleared; reduce --batch-size and resume from last.pt."
                    ) from exc
                eval_durations_s.append(time.monotonic() - eval_started_at)
                val.dataset = data.default_eval_name
                val_history.append(val)
                latest_primary_val = val
                epoch_bar.set_postfix(
                    loss=f"{metrics.loss:.4f}",
                    mAP=f"{val.mAP:.2%}",
                    R1=f"{val.rank1:.2%}",
                    lr=f"{metrics.lr:.6f}",
                )
                if val.mAP > state.best_mAP:
                    state.best_mAP = val.mAP
                    state.best_rank1 = val.rank1
                    state.best_epoch = epoch
                    self.checkpoint_manager.save_best(
                        best_weights,
                        model=models.val_model,
                        epoch=epoch,
                        val=val,
                        criterion_center=losses.criterion_center,
                        criterion_classifier=losses.criterion_id,
                        best_mAP=state.best_mAP,
                    )
                    tqdm.write(
                        f"  ✓ New best model (mAP={val.mAP:.2%}, "
                        f"R1={val.rank1:.2%}) -> {best_weights}"
                    )

                for dataset_name, (query_loader, gallery_loader) in loaders.cross_domain.items():
                    eval_started_at = time.monotonic()
                    try:
                        cross_domain_val = self._validate(
                            epoch,
                            models.val_model,
                            query_loader,
                            gallery_loader,
                        )
                    except RuntimeError as exc:
                        if not self._handle_oom(exc):
                            raise
                        raise RuntimeError(
                            f"{self.device.type.upper()} out of memory during cross-domain validation. "
                            "Cached memory was cleared; reduce --batch-size and resume from last.pt."
                        ) from exc
                    eval_durations_s.append(time.monotonic() - eval_started_at)
                    cross_domain_val.dataset = dataset_name
                    val_history.append(cross_domain_val)
                    tqdm.write(
                        f"  → {dataset_name}: mAP={cross_domain_val.mAP:.2%}  "
                        f"R1={cross_domain_val.rank1:.2%}  R5={cross_domain_val.rank5:.2%}"
                    )
                models.val_model.train()
                self._clear_memory(threshold=self.MEMORY_CLEAR_THRESHOLD)

            if epoch % 10 == 0 or epoch == self.epochs:
                self.checkpoint_manager.save_last(
                    save_dir / "last.pt",
                    model=models.model,
                    epoch=epoch,
                    val=latest_primary_val,
                    optimizer=optimization.optimizer,
                    optimizer_center=optimization.optimizer_center,
                    criterion_center=losses.criterion_center,
                    criterion_classifier=losses.criterion_id,
                    ema_model=models.ema_model,
                    best_mAP=state.best_mAP,
                )
                self._save_metrics(
                    save_dir,
                    history,
                    val_history,
                    state.best_epoch,
                    state.best_mAP,
                    state.best_rank1,
                    average_epoch_time_s=self._average_duration(epoch_durations_s),
                    average_eval_time_s=self._average_duration(eval_durations_s),
                    total_end_to_end_time_s=time.monotonic() - run_started_at,
                )

        self._save_metrics(
            save_dir,
            history,
            val_history,
            state.best_epoch,
            state.best_mAP,
            state.best_rank1,
            average_epoch_time_s=self._average_duration(epoch_durations_s),
            average_eval_time_s=self._average_duration(eval_durations_s),
            total_end_to_end_time_s=time.monotonic() - run_started_at,
        )
        self._save_training_plots(save_dir, history, val_history)
        LOGGER.info(
            f"Training complete. Best epoch={state.best_epoch}  "
            f"mAP={state.best_mAP:.2%}  R1={state.best_rank1:.2%}"
        )
        return TrainResult(
            best_epoch=state.best_epoch,
            best_mAP=state.best_mAP,
            best_rank1=state.best_rank1,
            weights_path=best_weights,
            history=history,
            val_history=val_history,
        )

    def _build_model(self, num_classes: int) -> nn.Module:
        model = ReIDModelRegistry.build_model(
            name=self.model_name,
            weights=Path(f"{self.model_name}_{self.dataset_name}.pt"),
            num_classes=num_classes,
            loss=self._model_loss_type(),
            pretrained=self.pretrained,
            use_gpu=self.device.type != "cpu",
            img_size=self.img_size,
            inference_feature=self.inference_feature,
            feature_fusion=self.feature_fusion,
            post_fusion_mixer=self.post_fusion_mixer,
            post_fusion_mixer_reduction=self.post_fusion_mixer_reduction,
            post_fusion_mixer_kernel=self.post_fusion_mixer_kernel,
            post_fusion_mixer_gamma_init=self.post_fusion_mixer_gamma_init,
            feat_dim=self.feat_dim,
            neck_dim=self.neck_dim,
            drop_path_rate=self.drop_path_rate,
            attention_window_layout=self.attention_window_layout,
            attention_bias=self.attention_bias,
            attention_mask=self.attention_mask,
            attention_shift=self.attention_shift,
            stage3_global=self.stage3_global,
            reid_adapter_stages=self.reid_adapter_stages,
            reid_adapter_reduction=self.reid_adapter_reduction,
            head_pool=self.head_pool,
            head_parts=self.head_parts,
            head_type=self.head_type,
            part_pooling=self.part_pooling,
            num_part_tokens=self.num_part_tokens,
            decouple_patterns=self.decouple_patterns,
            pattern_adapter_dim=self.pattern_adapter_dim,
            stripe_visibility=self.stripe_visibility,
            drop_global_aux=self.drop_global_aux,
            drop_global_aux_ratio=self.drop_global_aux_ratio,
            evidence_num_roles=self.evidence_num_roles,
            branch_metric=self.branch_aware_metric,
        )
        if hasattr(model, "head") and hasattr(model.head, "metric_feature"):
            model.head.metric_feature = self._effective_metric_feature()
        if hasattr(model, "head") and hasattr(model.head, "set_pooling"):
            model.head.set_pooling(self.head_pool)
        if hasattr(model, "head") and hasattr(model.head, "set_branch_metric"):
            model.head.set_branch_metric(self.branch_aware_metric)
        self._log_csl_tinyvit_config(model)
        return model

    @staticmethod
    def _max_drop_path(model: nn.Module) -> float:
        max_drop = 0.0
        for module in model.modules():
            drop_prob = getattr(module, "drop_prob", None)
            if drop_prob is not None:
                max_drop = max(max_drop, float(drop_prob))
        return max_drop

    def _log_csl_tinyvit_config(self, model: nn.Module) -> None:
        """Log active CSL-TinyViT architecture settings after construction."""
        if not self.model_name.startswith("csl_tinyvit"):
            return
        head = getattr(model, "head", None)
        block_windows = []
        for layer in getattr(model, "layers", []):
            if hasattr(layer, "blocks"):
                layer_windows = [
                    getattr(block, "window_size", None)
                    for block in layer.blocks
                    if hasattr(block, "window_size")
                ]
                if layer_windows:
                    block_windows.append(layer_windows)
        LOGGER.info(
            "CSL-TinyViT active config: "
            f"max_drop_path={self._max_drop_path(model):.3f}, "
            f"metric_feature={getattr(head, 'metric_feature', None)}, "
            f"inference_feature={getattr(head, 'inference_feature', None)}, "
            f"head_type={getattr(model, 'head_type', None)}, "
            f"head_pool={getattr(head, 'head_pool', None)}, "
            f"part_pooling={getattr(head, 'part_pooling', None)}, "
            f"num_part_tokens={getattr(head, 'num_part_tokens', None)}, "
            f"decouple_patterns={getattr(head, 'decouple_patterns', None)}, "
            f"stripe_visibility={getattr(head, 'stripe_visibility', None)}, "
            f"drop_global_aux={getattr(head, 'drop_global_aux_enabled', None)}, "
            f"drop_global_aux_ratio={getattr(head, 'drop_global_aux_ratio', None)}, "
            f"feature_fusion={getattr(model, 'feature_fusion', None)}, "
            f"post_fusion_mixer={getattr(model, 'post_fusion_mixer', None)}, "
            f"post_fusion_mixer_kernel={getattr(model, 'post_fusion_mixer_kernel', None)}, "
            f"post_fusion_mixer_gamma_init={getattr(model, 'post_fusion_mixer_gamma_init', None)}, "
            f"reid_adapter_stages={getattr(model, 'reid_adapter_stages', None)}, "
            f"reid_adapter_reduction={getattr(model, 'reid_adapter_reduction', None)}, "
            f"attention_window_layout={getattr(model, 'attention_window_layout', None)}, "
            f"attention_bias={getattr(model, 'attention_bias', None)}, "
            f"attention_mask={getattr(model, 'attention_mask', None)}, "
            f"attention_shift={getattr(model, 'attention_shift', None)}, "
            f"stage3_global={getattr(model, 'stage3_global', None)}, "
            f"windows={block_windows}"
        )
        match_count = getattr(model, "pretrained_match_count", None)
        total_count = getattr(model, "pretrained_total_count", None)
        if match_count is not None and total_count is not None:
            LOGGER.info(
                f"CSL-TinyViT pretrained tensor match count: {match_count}/{total_count} "
                f"from {getattr(model, 'pretrained_url', None)}"
            )

    def _resume_target_epochs(self, resume_path: Path, ckpt: dict) -> Optional[int]:
        """Return the epoch target saved by the run being resumed."""
        if ckpt.get("epochs") is not None:
            return int(ckpt["epochs"])

        run_dir = resume_path if resume_path.is_dir() else resume_path.parent
        for filename in ("hparams.json", "metrics.json"):
            path = run_dir / filename
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text())
                epochs = raw.get("epochs")
                if epochs is None and isinstance(raw.get("optimization"), dict):
                    epochs = raw["optimization"].get("epochs")
            except Exception:
                continue
            if epochs is not None:
                return int(epochs)
        return None

    def _build_resume_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        resumed_epoch: int,
        resume_path: Path,
        ckpt: dict,
    ) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        """Build a resume scheduler without increasing LR when extending a run."""
        previous_epochs = self._resume_target_epochs(resume_path, ckpt)
        extending_run = (
            previous_epochs is not None
            and self.epochs > previous_epochs
            and resumed_epoch >= self.warmup_epochs
            and "optimizer" in ckpt
        )
        if extending_run:
            remaining_epochs = max(self.epochs - resumed_epoch, 1)
            for group in optimizer.param_groups:
                current_lr = float(group["lr"])
                group["initial_lr"] = current_lr
                group["_base_lr"] = current_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
                eta_min=self.eta_min,
            )
            LOGGER.info(
                f"Extending cosine LR from epoch {resumed_epoch}/{previous_epochs} "
                f"to {self.epochs}: continuing from checkpoint LR over "
                f"{remaining_epochs} epochs"
            )
            return scheduler

        if self.warmup_epochs > 0 and resumed_epoch < self.warmup_epochs:
            warmup_factor = max(resumed_epoch, 1) / self.warmup_epochs
            for group in optimizer.param_groups:
                base_lr = float(group.get("_base_lr", group.get("initial_lr", group["lr"])))
                group["_base_lr"] = base_lr
                group["initial_lr"] = base_lr

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(self.epochs - self.warmup_epochs, 1),
                eta_min=self.eta_min,
            )
            scheduler.last_epoch = 0
            for group in optimizer.param_groups:
                group["lr"] = float(group["_base_lr"]) * warmup_factor
            LOGGER.info(
                f"Resuming linear LR warmup at epoch {resumed_epoch}/{self.warmup_epochs}: "
                f"warmup_factor={warmup_factor:.3f}"
            )
            return scheduler

        # Normal resume within the active cosine schedule. PyTorch's
        # last_epoch param has off-by-one issues with the incremental get_lr()
        # formula, so set last_epoch and LR via the closed-form cosine.
        cosine_epoch = max(resumed_epoch - self.warmup_epochs, 0)
        new_T_max = max(self.epochs - self.warmup_epochs, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=new_T_max,
            eta_min=self.eta_min,
        )
        scheduler.last_epoch = cosine_epoch
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = self.eta_min + (base_lr - self.eta_min) * (
                1 + math.cos(math.pi * cosine_epoch / new_T_max)
            ) / 2
        return scheduler

    def _build_train_loader(self, dataset) -> DataLoader:
        transform = build_train_transforms(
            self.img_size,
            preprocess=self.preprocess,
            color_jitter=self.color_jitter,
            gaussian_blur=self.gaussian_blur,
            random_grayscale=self.random_grayscale,
            random_erasing=self.random_erasing,
            random_patch=self.random_patch,
            random_crop_scale=self.random_crop_scale,
            color_augmentation=self.color_augmentation,
        )
        torch_ds = ReIDImageDataset(dataset.train.samples, transform=transform)
        if self.source_balance_groups:
            sampler = SourceBalancedPKSampler(
                dataset.train.samples,
                self.source_balance_groups,
                seed=self.seed,
            )
            batch_size = sampler.batch_size
        else:
            sampler = PKSampler(dataset.train.samples, p=self.p, k=self.k, seed=self.seed)
            batch_size = self.p * self.k
        return DataLoader(
            torch_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=_seed_data_worker,
            generator=self._train_generator,
        )

    def _build_test_loaders(self, dataset) -> Tuple[DataLoader, DataLoader]:
        transform = build_test_transforms(self.img_size, preprocess=self.preprocess)
        query_ds = ReIDImageDataset(dataset.query.samples, transform=transform)
        gallery_ds = ReIDImageDataset(dataset.gallery.samples, transform=transform)
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            shuffle=False,
            persistent_workers=False,
        )
        return DataLoader(query_ds, **loader_kwargs), DataLoader(gallery_ds, **loader_kwargs)

    def _probe_feat_dim(self, model: nn.Module) -> int:
        """Run a dummy forward to determine the training embedding dimension."""
        was_training = model.training
        model.train()
        dummy = torch.randn(2, 3, *self.img_size, device=self.device)
        with torch.no_grad():
            out = model(dummy)
        if not was_training:
            model.eval()
        _, features = self._split_model_output(out)
        if isinstance(features, dict):
            return features["global"].shape[1]
        if isinstance(features, (list, tuple)) and len(features) > 0:
            return features[0].shape[1]
        if isinstance(features, torch.Tensor):
            return features.shape[1]
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            return out[0].shape[1]  # multi-branch softmax: list of logits
        return out.shape[1]

    def _probe_classifier_feat_dim(self, model: nn.Module) -> int:
        """Run a dummy forward to determine the margin-classifier feature dimension."""
        was_training = model.training
        model.train()
        dummy = torch.randn(2, 3, *self.img_size, device=self.device)
        with torch.no_grad():
            out = model(dummy)
        if not was_training:
            model.eval()
        _, features = self._split_model_output(out)
        classifier_features = self._classification_features(features)
        if classifier_features is None:
            raise RuntimeError(f"classifier_loss={self.classifier_loss} requires embedding features")
        return classifier_features.shape[1]

    @staticmethod
    def _split_model_output(output):
        """Unpack training output into (logits, features) across backbone contracts."""
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0], output[1]
        if isinstance(output, list) and len(output) == 2:
            return output[0], output[1]
        return output, None

    # ------------------------------------------------------------------
    # Training-family helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_vit(model: nn.Module) -> bool:
        """Check if the model is a transformer-like variant."""
        return hasattr(model, "blocks") and hasattr(model, "patch_embed")

    def _backbone_spec(self):
        """Return the registered backbone spec for this trainer, when available."""
        try:
            return get_backbone_spec(self.model_name)
        except KeyError:
            return None

    @classmethod
    def _detect_training_family(cls, model: nn.Module) -> str:
        """Return the canonical training family for a model instance."""
        explicit_family = getattr(model, "training_family", None)
        if explicit_family is not None:
            return cls._normalize_training_family(explicit_family)
        return "transformer" if cls._is_vit(model) else "cnn"

    @staticmethod
    def _normalize_training_family(family: str) -> str:
        family = str(family).lower()
        if family in {"vit", "transformer"}:
            return "transformer"
        if family in {"cnn", "convnet"}:
            return "cnn"
        if family in {"hybrid", "legacy"}:
            return family
        raise ValueError(f"Unsupported training_family={family!r}")

    @classmethod
    def _training_recipe_for_family(cls, family: str) -> TrainingRecipe:
        """Build the recipe object for a canonical training family."""
        return default_recipe_for_family(cls._normalize_training_family(family))

    def _resolve_training_recipe_for_model_name(self) -> TrainingRecipe | None:
        """Resolve recipes that need defaults before model construction."""
        spec = self._backbone_spec()
        return build_training_recipe(spec=spec) if spec is not None else None

    def _resolve_training_recipe(self, model: nn.Module) -> TrainingRecipe:
        """Resolve the model's family-specific training recipe."""
        explicit_recipe = getattr(model, "training_recipe", None)
        if explicit_recipe is not None:
            return build_training_recipe(str(explicit_recipe))
        spec = self._backbone_spec()
        if spec is not None:
            return build_training_recipe(spec=spec)
        return self._training_recipe_for_family(self._detect_training_family(model))

    def _recipe_for_bundle(self, model: ModelBundle) -> TrainingRecipe:
        """Resolve the recipe stored on, or implied by, a model bundle."""
        if model.recipe is not None:
            return model.recipe
        family = "transformer" if model.is_transformer else model.training_family
        return self._training_recipe_for_family(family)

    def _model_loss_type(self) -> str:
        """Choose the backbone output contract needed by the configured losses."""
        if self.loss_type == "softmax" and self.classifier_loss == "ce":
            return "softmax"
        if self.loss_type == "ms":
            return "ms"
        return "triplet"

    def _use_soft_margin_triplet(self, default_soft_margin: bool) -> bool:
        """Resolve hard-margin vs softplus batch-hard triplet behavior."""
        if self.triplet_soft_margin is not None:
            return bool(self.triplet_soft_margin)
        return default_soft_margin

    def _build_classifier_loss(self, num_classes: int, feat_dim: int, label_smooth: float) -> nn.Module:
        """Build the ID-classification criterion."""
        if self.classifier_loss == "ce":
            return CrossEntropyLabelSmooth(num_classes, epsilon=label_smooth)
        if self.classifier_loss == "arcface":
            return ArcFaceLoss(
                feat_dim=feat_dim,
                num_classes=num_classes,
                scale=self.arcface_scale,
                margin=self.arcface_margin,
            )
        if self.classifier_loss == "cosface":
            return CosFaceLoss(
                feat_dim=feat_dim,
                num_classes=num_classes,
                scale=self.cosface_scale,
                margin=self.cosface_margin,
            )
        raise ValueError(f"Unsupported classifier_loss: {self.classifier_loss}")

    def _effective_metric_feature(self) -> str:
        """Resolve the metric feature mode for multi-branch models."""
        if self.metric_feature != "auto":
            return self.metric_feature
        return "concat_bn" if self.loss_type == "ms" else "raw_mean"

    def _aux_ce_weight_for_epoch(self, epoch: int) -> float:
        """Return the active auxiliary classifier CE weight for this epoch."""
        if self.aux_ce_drop_epoch > 0 and epoch > self.aux_ce_drop_epoch:
            return 0.0
        return self.aux_ce_weight

    def _apply_vit_training_defaults(self) -> None:
        """Apply transformer training conveniences unless the caller set values explicitly."""
        # AdamW uses decoupled weight decay: effective WD = lr x wd.
        # The default wd=5e-4 (calibrated for Adam L2-reg) gives negligible
        # regularization with AdamW, so use the transformer recipe default unless the
        # caller intentionally passed a lower value for an ablation.
        if "weight_decay" not in self.explicit_hparams and self.weight_decay < 0.01:
            self.weight_decay = 0.1

        if "warmup_epochs" not in self.explicit_hparams and self.warmup_epochs <= 10:
            self.warmup_epochs = 20

        # Transformer-style ReID backbones with AdamW need ~2x higher LR than CNNs with Adam. Preserve
        # explicit lower LRs so LR sweeps test the requested value.
        if "lr" not in self.explicit_hparams and self.lr <= 3.5e-4:
            self.lr = 7e-4

        # Transformer-style ReID backbones need stronger center loss to tighten positive clusters. Preserve
        # explicit zero so loss ablations can remove center loss.
        if (
            "center_loss_weight" not in self.explicit_hparams
            and self.loss_type != "ms"
            and self.center_loss_weight <= 5e-4
        ):
            self.center_loss_weight = 5e-3

    def _vit_layer_id_for_param(self, name: str, depth: int) -> int:
        """Map parameter name to layer index (0=patch/stem, depth+1=head/new modules)."""
        if name.startswith(("patch_embed", "cls_token", "pos_embed")):
            return 0
        # "blocks." is the standard ViT naming; "layers." is used by
        # CSL-TinyViT (self.layers registered first, self.blocks alias).
        if name.startswith(("blocks.", "layers.")):
            return int(name.split(".")[1]) + 1
        return depth + 1

    def _vit_lr_scale_for_param(self, name: str, depth: int) -> float:
        """Return the LR scale for a transformer parameter under the active LR profile."""
        if self._is_reid_adaptation_param(name):
            return 1.0

        layer_id = self._vit_layer_id_for_param(name, depth)
        if self.vit_lr_profile == "reid_lrd":
            if name.startswith("patch_embed") or name.startswith(("blocks.0.", "layers.0.")):
                return 0.05
            if name.startswith(("blocks.1.", "layers.1.")):
                return 0.10
            if name.startswith(("blocks.2.", "layers.2.")):
                return 0.25
            if name.startswith(("blocks.3.", "layers.3.")):
                return 0.50
            return 1.0

        layer_decay = 0.95
        return layer_decay ** (depth + 1 - layer_id)

    def _build_vit_param_groups(self, model: nn.Module) -> list:
        """Build parameter groups with layer-decay LR and no-WD filtering.

        Layer-decay assigns geometrically decreasing LR to earlier blocks:
            lr_scale = layer_decay ** (depth - layer_idx)
        Patch embed and pos_embed get the lowest LR; the classifier head
        gets the base LR.

        No weight decay is applied to: bias, LayerNorm, InstanceNorm,
        cls_token, pos_embed, and AIN gate parameters.
        """
        depth = getattr(model, "depth", len(model.blocks))

        # Identify parameters that should not have weight decay.
        # "bn" covers BatchNorm gamma in hybrid CNN-Transformer models
        # (e.g. CSL-TinyViT's Conv2d_BN and BNNeck3 modules).
        no_wd_keywords = {
            "bias", "cls_token", "pos_embed",
            "norm", "ln", "bn", "in_norm", "gate",
        }

        def _no_wd(name: str) -> bool:
            return any(kw in name for kw in no_wd_keywords)

        param_groups: dict[str, dict] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            lr_scale = self._vit_lr_scale_for_param(name, depth)
            wd = 0.0 if _no_wd(name) else self.weight_decay

            is_reid_adaptation = self._is_reid_adaptation_param(name)
            group_key = f"lr_{lr_scale:.6g}_wd_{wd}"
            if self.gradual_unfreeze:
                role = "reid" if is_reid_adaptation else "backbone"
                group_key = f"{role}_{group_key}"
            if group_key not in param_groups:
                param_groups[group_key] = {
                    "params": [],
                    "lr": self.lr * lr_scale,
                    "weight_decay": wd,
                    "is_head": is_reid_adaptation,
                    "is_backbone": not is_reid_adaptation,
                    "lr_scale": lr_scale,
                }
            else:
                param_groups[group_key]["is_head"] |= is_reid_adaptation
                param_groups[group_key]["is_backbone"] |= not is_reid_adaptation
            param_groups[group_key]["params"].append(param)

        LOGGER.info(
            f"Transformer param groups: {len(param_groups)} groups, "
            f"lr_profile={self.vit_lr_profile}, depth={depth}"
        )
        return list(param_groups.values())

    def _build_cnn_param_groups(self, model: nn.Module) -> list[dict]:
        """Build Adam parameter groups for CNN models.

        Keep the historical single-group behavior unless a staged freeze/warmup
        schedule needs backbone and ReID-specific parameters to have separate LR
        state.
        """
        if not (self.gradual_unfreeze or self.head_warmup_epochs > 0 or self.backbone_freeze_epochs > 0):
            return [{"params": model.parameters()}]

        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if self._is_reid_adaptation_param(name):
                head_params.append(param)
            else:
                backbone_params.append(param)

        parameter_groups = []
        if backbone_params:
            parameter_groups.append({
                "params": backbone_params,
                "is_head": False,
                "is_backbone": True,
            })
        if head_params:
            parameter_groups.append({
                "params": head_params,
                "is_head": True,
                "is_backbone": False,
            })
        return parameter_groups

    def _build_mobilenetv4_param_groups(self, model: nn.Module) -> list[dict]:
        """Build AdamW groups for MobileNetV4 with no decay on norm/bias params."""
        grouped: dict[tuple[bool, bool], dict] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_head = self._is_reid_adaptation_param(name)
            no_decay = self._is_cnn_no_weight_decay_param(name, param)
            key = (is_head, no_decay)
            if key not in grouped:
                grouped[key] = {
                    "params": [],
                    "lr": self.lr,
                    "weight_decay": 0.0 if no_decay else self.weight_decay,
                    "is_head": is_head,
                    "is_backbone": not is_head,
                    "no_weight_decay": no_decay,
                }
            grouped[key]["params"].append(param)

        return list(grouped.values())

    @staticmethod
    def _is_cnn_no_weight_decay_param(name: str, param: nn.Parameter) -> bool:
        lower_name = name.lower()
        return (
            name.endswith(".bias")
            or param.ndim <= 1
            or ".bn" in lower_name
            or "batchnorm" in lower_name
            or "norm" in lower_name
        )

    def _head_warmup_active(self, epoch: int) -> bool:
        """Return whether this epoch should train only neck/head parameters.

        If backbone freeze is configured, the head-only phase starts after the
        freeze warm-start instead of being silently overlapped and skipped.
        """
        if self.head_warmup_epochs <= 0:
            return False
        start_epoch = self.backbone_freeze_epochs if self.backbone_freeze_epochs > 0 else 0
        return start_epoch < epoch <= start_epoch + self.head_warmup_epochs

    def _head_warmup_start_epoch(self) -> int:
        """Return the first epoch of the effective head-only warmup phase."""
        return self.backbone_freeze_epochs + 1 if self.backbone_freeze_epochs > 0 else 1

    def _backbone_freeze_active(self, epoch: int) -> bool:
        """Return whether this epoch should keep pretrained backbone stages frozen."""
        return self.backbone_freeze_epochs > 0 and epoch <= self.backbone_freeze_epochs

    def _gradual_unfreeze_phase(self, epoch: int) -> str | None:
        """Return the active staged-unfreeze phase for this epoch."""
        if not self.gradual_unfreeze:
            return None
        if epoch <= self.gradual_unfreeze_head_epochs:
            return "head"
        if epoch <= self.gradual_unfreeze_stage_epochs:
            return "stage"
        return "full"

    def _gradual_backbone_lr_active(self, epoch: int) -> bool:
        """Return whether trainable backbone groups should use the temporary LR drop."""
        return (
            self.gradual_unfreeze
            and self.gradual_unfreeze_backbone_lr_epochs > 0
            and epoch > self.gradual_unfreeze_head_epochs
            and epoch <= self.gradual_unfreeze_stage_epochs + self.gradual_unfreeze_backbone_lr_epochs
        )

    def _effective_id_loss_weight(self, epoch: int) -> float:
        """Return ID loss weight after applying any temporary early CE boost."""
        if self.early_id_loss_epochs > 0 and epoch <= self.early_id_loss_epochs:
            return self.early_id_loss_weight if self.early_id_loss_weight > 0 else self.id_loss_weight
        return self.id_loss_weight

    def _effective_center_loss_weight(self, epoch: int) -> float:
        """Return center loss weight after applying the optional epoch ramp."""
        if self.center_loss_weight <= 0 or self.center_loss_ramp_end_epoch <= 0:
            return self.center_loss_weight
        if epoch <= self.center_loss_ramp_start_epoch:
            return 0.0
        if epoch <= self.center_loss_ramp_end_epoch:
            span = self.center_loss_ramp_end_epoch - self.center_loss_ramp_start_epoch
            return self.center_loss_weight * ((epoch - self.center_loss_ramp_start_epoch) / span)
        return self.center_loss_weight

    @staticmethod
    def _is_head_or_neck_param(name: str) -> bool:
        return name == "fusion_logits" or name.startswith(
            (
                "head.",
                "neck.",
                "spatial_neck.",
                "fpn_projections.",
                "output_norms.",
                "post_fusion_mixer_module.",
            )
        )

    @staticmethod
    def _is_reid_adaptation_param(name: str) -> bool:
        return (
            name == "fusion_logits"
            or name.startswith(
                (
                    "head.",
                    "neck.",
                    "spatial_neck.",
                    "feature_fusion_module.",
                    "fpn_projections.",
                    "output_norms.",
                    "post_fusion_mixer_module.",
                )
            )
            or ".reid_adapters." in name
        )

    @staticmethod
    def _last_vit_stage_index(model: nn.Module) -> int | None:
        layers = getattr(model, "layers", None)
        if layers is not None:
            return len(layers) - 1
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            return len(blocks) - 1
        return None

    @staticmethod
    def _is_last_vit_stage_param(name: str, stage_index: int | None) -> bool:
        if stage_index is None or stage_index < 0:
            return False
        return name.startswith((f"layers.{stage_index}.", f"blocks.{stage_index}."))

    @staticmethod
    def _last_backbone_stage_prefixes(model: nn.Module) -> tuple[str, ...]:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return ()
        for attr_name in ("stages", "blocks", "features"):
            container = getattr(backbone, attr_name, None)
            if isinstance(container, (nn.ModuleList, nn.Sequential)) and len(container) > 0:
                return (f"backbone.{attr_name}.{len(container) - 1}.",)
        return ()

    def _is_last_stage_param(self, model: nn.Module, name: str, stage_index: int | None) -> bool:
        return self._is_last_vit_stage_param(name, stage_index) or name.startswith(
            self._last_backbone_stage_prefixes(model)
        )

    @staticmethod
    def _is_attention_param(name: str) -> bool:
        return ".attn." in name

    def _set_head_warmup_trainability(self, model: nn.Module, enabled: bool) -> None:
        """Freeze/unfreeze backbone parameters for head-only warmup."""
        for name, param in model.named_parameters():
            param.requires_grad_(not enabled or self._is_head_or_neck_param(name))

    def _set_backbone_freeze_trainability(self, model: nn.Module, enabled: bool) -> None:
        """Freeze pretrained backbone parameters while keeping ReID-specific modules trainable."""
        for name, param in model.named_parameters():
            param.requires_grad_(not enabled or self._is_reid_adaptation_param(name))
        if not enabled:
            return
        for module_name in ("patch_embed", "layers"):
            module = getattr(model, module_name, None)
            if module is not None:
                module.eval()
        if getattr(model, "layers", None) is None:
            blocks = getattr(model, "blocks", None)
            if blocks is not None:
                blocks.eval()
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            backbone.eval()

    def _set_gradual_unfreeze_trainability(self, model: nn.Module, phase: str) -> None:
        """Apply staged CSL-TinyViT unfreeze trainability for the active phase."""
        last_stage_index = self._last_vit_stage_index(model)
        for name, param in model.named_parameters():
            if phase == "head":
                trainable = self._is_reid_adaptation_param(name)
            elif phase == "stage":
                trainable = (
                    self._is_reid_adaptation_param(name)
                    or (
                        self._is_last_stage_param(model, name, last_stage_index)
                        and not self._is_attention_param(name)
                    )
                )
            else:
                trainable = True
            param.requires_grad_(trainable)

        if phase == "full":
            return

        patch_embed = getattr(model, "patch_embed", None)
        if patch_embed is not None:
            patch_embed.eval()
        layers = getattr(model, "layers", None)
        if layers is not None:
            if phase == "head":
                layers.eval()
                return
            for index, layer in enumerate(layers):
                layer.train(index == last_stage_index)
            return
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            if phase == "head":
                blocks.eval()
                return
            for index, block in enumerate(blocks):
                block.train(index == last_stage_index)
            return

        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return
        if phase == "head":
            backbone.eval()
            return
        for attr_name in ("stages", "blocks", "features"):
            container = getattr(backbone, attr_name, None)
            if isinstance(container, (nn.ModuleList, nn.Sequential)) and len(container) > 0:
                backbone.eval()
                last_index = len(container) - 1
                for index, stage in enumerate(container):
                    stage.train(index == last_index)
                return

    def _apply_head_warmup_lrs(self, optimizer) -> list[float]:
        """Use zero LR for backbone groups and a boosted LR for head groups."""
        original_lrs = [group["lr"] for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            base_lr = group.get("_base_lr", group.get("lr", self.lr))
            group["lr"] = base_lr * self.head_warmup_lr_mult if group.get("is_head", False) else 0.0
        return original_lrs

    @staticmethod
    def _group_has_trainable_params(group: dict) -> bool:
        """Return whether an optimizer group currently owns any trainable parameter."""
        return any(param.requires_grad for param in group.get("params", ()))

    def _apply_gradual_backbone_lrs(self, optimizer, epoch: int | None = None) -> list[float]:
        """Zero frozen groups and temporarily reduce trainable backbone LR groups."""
        original_lrs = [group["lr"] for group in optimizer.param_groups]
        backbone_lr_active = True if epoch is None else self._gradual_backbone_lr_active(epoch)
        for group in optimizer.param_groups:
            if not self._group_has_trainable_params(group):
                group["lr"] = 0.0
            elif backbone_lr_active and group.get("is_backbone", False):
                group["lr"] *= self.gradual_unfreeze_backbone_lr_mult
        return original_lrs

    def _optimizer_lr_summary(self, optimizer) -> tuple[float, float, float]:
        """Return max active LR plus active backbone/head LR summaries."""
        active_lrs = []
        backbone_lrs = []
        head_lrs = []
        for group in optimizer.param_groups:
            if not self._group_has_trainable_params(group):
                continue
            lr = float(group["lr"])
            if lr <= 0:
                continue
            active_lrs.append(lr)
            if group.get("is_backbone", False):
                backbone_lrs.append(lr)
            if group.get("is_head", False):
                head_lrs.append(lr)
        fallback = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
        return (
            max(active_lrs, default=fallback),
            max(backbone_lrs, default=0.0),
            max(head_lrs, default=0.0),
        )

    @staticmethod
    def _trainable_parameter_summary(model: nn.Module) -> tuple[int, int]:
        """Return trainable and total parameter counts."""
        total = 0
        trainable = 0
        for param in model.parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
        return trainable, total

    def _metric_loss_for_features(self, criterion_metric, features, pids: torch.Tensor) -> torch.Tensor:
        """Compute metric loss for a tensor feature or branch feature dict."""
        if isinstance(features, dict):
            if not self.branch_aware_metric:
                key = self._effective_metric_feature()
                selected = features.get(key, features["raw_mean"])
                return criterion_metric(F.normalize(selected, p=2, dim=1), pids)

            part_supervision_weights = self._part_supervision_weights(features)
            if torch.is_tensor(part_supervision_weights):
                global_features = features.get("global", features.get("raw_mean"))
                global_loss = criterion_metric(F.normalize(global_features, p=2, dim=1), pids)
                part_losses = []
                part_weights = []
                for index, key in enumerate(self._sorted_part_feature_keys(features)):
                    if key not in features or index >= part_supervision_weights.shape[1]:
                        continue
                    branch_features = F.normalize(features[key], p=2, dim=1)
                    part_losses.append(criterion_metric(branch_features, pids))
                    part_weights.append(part_supervision_weights[:, index].mean().clamp(min=0.0))
                if not part_losses:
                    return global_loss
                weights = torch.stack(part_weights)
                part_loss = sum(loss * weight for loss, weight in zip(part_losses, weights)) / weights.sum().clamp(
                    min=1e-12
                )
                return global_loss + self.branch_metric_part_weight * part_loss

            weighted_losses = []
            total_weight = 0.0
            branch_weights = [("global", 1.0)]
            metric_key = self._effective_metric_feature()
            if metric_key == "raw_concat" and metric_key in features:
                branch_weights.append((metric_key, 1.0))
            branch_weights += [
                (key, self.branch_metric_part_weight)
                for key in self._sorted_part_feature_keys(features)
            ]
            for key, weight in branch_weights:
                if key in features and weight > 0:
                    branch_features = F.normalize(features[key], p=2, dim=1)
                    weighted_losses.append(criterion_metric(branch_features, pids) * weight)
                    total_weight += weight
            if weighted_losses and total_weight > 0:
                return sum(weighted_losses) / total_weight
            return torch.zeros((), device=self.device, requires_grad=True)

        if isinstance(features, (list, tuple)):
            valid_features = [feat for feat in features if isinstance(feat, torch.Tensor)]
            if not valid_features:
                return torch.zeros((), device=self.device, requires_grad=True)
            losses = [criterion_metric(F.normalize(feat, p=2, dim=1), pids) for feat in valid_features]
            return self._reduce_branch_losses(losses)

        return criterion_metric(F.normalize(features, p=2, dim=1), pids)

    def _evidence_auxiliary_loss(self, features, pids: torch.Tensor) -> torch.Tensor:
        """Compute IET evidence alignment, null-token, and diversity losses."""
        if not isinstance(features, dict):
            return torch.zeros((), device=self.device)

        total = torch.zeros((), device=self.device)
        if self.evidence_alignment_loss_weight > 0:
            total = total + self.evidence_alignment_loss_weight * self._evidence_alignment_loss(features, pids)
        if self.evidence_null_loss_weight > 0:
            total = total + self.evidence_null_loss_weight * self._evidence_null_loss(features)
        if self.evidence_diversity_loss_weight > 0:
            total = total + self.evidence_diversity_loss_weight * self._evidence_diversity_loss(features)
        return total

    def _evidence_alignment_loss(self, features: dict, pids: torch.Tensor) -> torch.Tensor:
        """Batch evidence-set OT loss over same-ID positives and different-ID negatives."""
        part_keys = self._sorted_part_feature_keys(features)
        required = ("_visibility", "_rarity", "_role_logits", "_nullness")
        if not part_keys or any(key not in features for key in required):
            return torch.zeros((), device=self.device)

        part_features = torch.stack(
            [F.normalize(features[key], p=2, dim=1) for key in part_keys],
            dim=1,
        )
        visibility = features["_visibility"].to(dtype=part_features.dtype)
        rarity = features["_rarity"].to(dtype=part_features.dtype)
        role_probs = F.softmax(features["_role_logits"].to(dtype=part_features.dtype), dim=-1)
        nullness = features["_nullness"].to(dtype=part_features.dtype)
        if visibility.shape[1] != part_features.shape[1]:
            return torch.zeros((), device=self.device)

        evidence_mass = (visibility * rarity * (1.0 - nullness)).clamp_min(1e-6)
        evidence_mass = evidence_mass / evidence_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)

        part_similarity = torch.einsum("bkd,cld->bckl", part_features, part_features)
        role_compatibility = torch.einsum("bkr,clr->bckl", role_probs, role_probs)
        scores = part_similarity * role_compatibility
        alignment = self._sinkhorn_alignment(
            scores,
            evidence_mass,
            iters=self.evidence_sinkhorn_iters,
            temperature=self.evidence_sinkhorn_temperature,
        )

        same_id = pids[:, None].eq(pids[None, :])
        eye = torch.eye(pids.shape[0], device=pids.device, dtype=torch.bool)
        positive_mask = same_id & ~eye
        negative_mask = ~same_id

        losses = []
        if positive_mask.any():
            losses.append((1.0 - alignment[positive_mask]).mean())
        if negative_mask.any():
            losses.append(F.relu(alignment[negative_mask] - self.evidence_alignment_margin).mean())
        if not losses:
            return torch.zeros((), device=self.device)
        return sum(losses) / len(losses)

    @staticmethod
    def _sinkhorn_alignment(
        scores: torch.Tensor,
        evidence_mass: torch.Tensor,
        *,
        iters: int,
        temperature: float,
    ) -> torch.Tensor:
        """Return pairwise optimal-transport evidence alignment scores."""
        eps = torch.finfo(scores.dtype).eps
        temperature = max(float(temperature), float(eps))
        logits = (scores - scores.amax(dim=(-1, -2), keepdim=True)) / temperature
        kernel = logits.exp().clamp_min(eps)
        row_mass = evidence_mass[:, None, :]
        col_mass = evidence_mass[None, :, :]
        u = torch.ones_like(row_mass).expand(scores.shape[0], scores.shape[1], scores.shape[2])
        v = torch.ones_like(col_mass).expand(scores.shape[0], scores.shape[1], scores.shape[3])
        for _ in range(max(int(iters), 1)):
            u = row_mass / torch.einsum("bckl,bcl->bck", kernel, v).clamp_min(eps)
            v = col_mass / torch.einsum("bckl,bck->bcl", kernel, u).clamp_min(eps)
        plan = u[..., :, None] * kernel * v[..., None, :]
        return (plan * scores).sum(dim=(-1, -2))

    def _evidence_null_loss(self, features: dict) -> torch.Tensor:
        """Supervise the final semantic evidence token as the explicit null slot."""
        nullness = features.get("_nullness")
        if not torch.is_tensor(nullness) or nullness.shape[1] < 2:
            return torch.zeros((), device=self.device)
        target = torch.zeros_like(nullness)
        target[:, -1] = 1.0
        return F.binary_cross_entropy(nullness.clamp(1e-6, 1.0 - 1e-6), target)

    def _evidence_diversity_loss(self, features: dict) -> torch.Tensor:
        """Encourage evidence roles and descriptors to avoid token collapse."""
        part_keys = self._sorted_part_feature_keys(features)
        role_logits = features.get("_role_logits")
        if len(part_keys) < 2 or not torch.is_tensor(role_logits):
            return torch.zeros((), device=self.device)
        role_probs = F.softmax(role_logits, dim=-1)
        role_overlap = torch.einsum("bkr,blr->bkl", role_probs, role_probs)
        part_features = torch.stack(
            [F.normalize(features[key], p=2, dim=1) for key in part_keys],
            dim=1,
        )
        feature_overlap = torch.einsum("bkd,bld->bkl", part_features, part_features).clamp_min(0.0)
        mask = ~torch.eye(len(part_keys), device=role_overlap.device, dtype=torch.bool)
        return 0.5 * (role_overlap[:, mask].mean() + feature_overlap[:, mask].mean())

    def _reduce_branch_losses(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate branch losses using mean (default) or sum."""
        if not losses:
            return torch.zeros((), device=self.device, requires_grad=True)
        if self.branch_loss_agg == "sum":
            return sum(losses)
        return sum(losses) / len(losses)

    def _classification_loss_for_logits(
        self,
        criterion_id: nn.Module,
        logits,
        pids: torch.Tensor,
        epoch: int,
        features=None,
    ) -> torch.Tensor:
        """Compute global CE plus relatively weighted auxiliary-head CE."""
        if not isinstance(logits, list):
            return criterion_id(logits, pids)
        losses = [criterion_id(logit, pids) for logit in logits]
        if len(losses) == 1:
            return losses[0]
        aux_weight = self._aux_ce_weight_for_epoch(epoch)
        weights = [torch.ones((), device=losses[0].device, dtype=losses[0].dtype)]
        part_supervision_weights = self._part_supervision_weights(features) if isinstance(features, dict) else None
        if torch.is_tensor(part_supervision_weights):
            for index in range(1, len(losses)):
                part_index = index - 1
                if part_index < part_supervision_weights.shape[1]:
                    weights.append(aux_weight * part_supervision_weights[:, part_index].mean().clamp(min=0.0))
                else:
                    weights.append(torch.as_tensor(aux_weight, device=losses[0].device, dtype=losses[0].dtype))
        else:
            weights.extend(
                torch.as_tensor(aux_weight, device=losses[0].device, dtype=losses[0].dtype)
                for _ in losses[1:]
            )
        weighted = sum(loss * weight for loss, weight in zip(losses, weights))
        normalizer = torch.stack(weights).sum().clamp(min=1e-12)
        return weighted / normalizer

    @staticmethod
    def _part_supervision_weights(features: dict) -> torch.Tensor | None:
        """Detached part weights for auxiliary CE/metric supervision.

        Visibility decides whether a part should contribute. Nullness removes
        explicit null/background evidence tokens from identity supervision.
        Detaching prevents the part metadata heads from minimizing ID/metric
        losses by simply lowering their own weights.
        """
        visibility = features.get("_visibility")
        if not torch.is_tensor(visibility):
            return None
        weights = visibility
        nullness = features.get("_nullness")
        if torch.is_tensor(nullness) and nullness.shape == visibility.shape:
            weights = weights * (1.0 - nullness)
        return weights.detach().clamp(min=0.0)

    @staticmethod
    def _sorted_part_feature_keys(features: dict) -> list[str]:
        """Return part feature keys sorted by numeric suffix: part0, part1, ..."""
        def part_index(key: str) -> int:
            try:
                return int(key[4:])
            except ValueError:
                return 10**9

        return sorted(
            (key for key in features if key.startswith("part") and key[4:].isdigit()),
            key=part_index,
        )

    @staticmethod
    def _center_features(features):
        """Use the global raw branch for center loss when branch metrics are enabled."""
        if isinstance(features, dict):
            return features.get("global", features.get("raw_mean"))
        if isinstance(features, (list, tuple)):
            return features[0] if len(features) > 0 else None
        return features

    def _classification_features(self, features):
        """Select embeddings for margin-based classifier losses."""
        if isinstance(features, dict):
            key = self._effective_metric_feature()
            return features.get(key, features.get("raw_mean", features.get("global")))
        if isinstance(features, (list, tuple)):
            return features[0] if len(features) > 0 else None
        return features

    def _restore_classifier_loss_state(self, ckpt: dict, criterion_id: nn.Module, resume_path: Path) -> None:
        """Restore train-only margin classifier weights when resuming."""
        if self.classifier_loss == "ce":
            return

        state = ckpt.get("classifier_loss_state_dict")
        if state is None:
            LOGGER.warning(
                f"{resume_path} has no classifier_loss_state_dict; "
                f"initializing {self.classifier_loss} classifier from scratch"
            )
            return

        try:
            criterion_id.load_state_dict(state)
            LOGGER.info(f"Restored {self.classifier_loss} classifier state from checkpoint")
        except RuntimeError as exc:
            LOGGER.warning(f"Could not restore {self.classifier_loss} classifier state from {resume_path}: {exc}")

    def _restore_center_loss_state(
        self,
        ckpt: dict,
        criterion_center: CenterLoss,
        model: nn.Module,
        train_loader: DataLoader,
        resume_path: Path,
    ) -> None:
        """Restore center-loss centers, or initialize them for older checkpoints."""
        if self.center_loss_weight <= 0:
            return

        center_state = ckpt.get("center_loss_state_dict")
        if center_state is not None:
            try:
                criterion_center.load_state_dict(center_state)
                LOGGER.info("Restored center loss state from checkpoint")
                return
            except RuntimeError as exc:
                LOGGER.warning(f"Could not restore center loss state from {resume_path}: {exc}")

        if "optimizer_center" in ckpt:
            LOGGER.warning(
                f"{resume_path} has optimizer_center but no center_loss_state_dict; "
                "initializing center-loss centers from resumed model features"
            )
        self._initialize_center_loss_from_features(model, criterion_center, train_loader)

    def _initialize_center_loss_from_features(
        self,
        model: nn.Module,
        criterion_center: CenterLoss,
        train_loader: DataLoader,
    ) -> None:
        """Initialize missing center-loss centers from per-class feature means."""
        was_training = model.training
        model.eval()

        centers = torch.zeros_like(criterion_center.centers.data)
        counts = torch.zeros(criterion_center.num_classes, device=self.device)

        try:
            with torch.no_grad():
                for imgs, pids, _ in train_loader:
                    imgs = imgs.to(self.device)
                    pids = pids.to(self.device)
                    output = model(imgs)
                    _, features = self._split_model_output(output)
                    center_features = self._center_features(features)
                    if center_features is None:
                        continue

                    center_features = center_features.detach()
                    if center_features.shape[1] != criterion_center.feat_dim:
                        raise RuntimeError(
                            "Center feature dimension does not match center-loss checkpoint state: "
                            f"{center_features.shape[1]} != {criterion_center.feat_dim}"
                        )

                    valid = (pids >= 0) & (pids < criterion_center.num_classes)
                    if not valid.any():
                        continue

                    valid_pids = pids[valid].long()
                    centers.index_add_(0, valid_pids, center_features[valid])
                    counts.index_add_(0, valid_pids, torch.ones_like(valid_pids, dtype=counts.dtype))
        finally:
            if was_training:
                model.train()

        seen = counts > 0
        if not seen.any():
            LOGGER.warning("Could not initialize center-loss centers: no valid class features found")
            return

        centers[seen] = centers[seen] / counts[seen].unsqueeze(1)
        criterion_center.centers.data[seen] = centers[seen]
        LOGGER.info(
            f"Initialized center-loss centers from resumed model features "
            f"({int(seen.sum().item())}/{criterion_center.num_classes} classes)"
        )

    def _train_epoch(
        self, epoch, model, loader, criterion_id, criterion_metric, criterion_center,
        optimizer, optimizer_center, scheduler, *, ema_model=None, grad_clip: float = 0.0,
    ) -> TrainMetrics:
        from tqdm import tqdm

        self._seed_training_epoch(epoch, loader)
        model.train()
        backbone_freeze_active = self._backbone_freeze_active(epoch)
        gradual_unfreeze_phase = self._gradual_unfreeze_phase(epoch)
        gradual_backbone_original_lrs = None
        head_warmup_original_lrs = None
        head_warmup_active = False
        if gradual_unfreeze_phase:
            self._set_gradual_unfreeze_trainability(model, gradual_unfreeze_phase)
            if epoch == 1:
                LOGGER.info(
                    "Gradual unfreeze enabled: "
                    f"ReID modules through epoch {self.gradual_unfreeze_head_epochs}, "
                    f"last stage through epoch {self.gradual_unfreeze_stage_epochs}, "
                    "then full backbone"
                )
                if self.gradual_unfreeze_backbone_lr_epochs > 0:
                    LOGGER.info(
                        "Gradual unfreeze backbone LR drop active from epoch "
                        f"{self.gradual_unfreeze_head_epochs + 1} through epoch "
                        f"{self.gradual_unfreeze_stage_epochs + self.gradual_unfreeze_backbone_lr_epochs} "
                        f"(backbone_lr_mult={self.gradual_unfreeze_backbone_lr_mult:g})"
                    )
        elif backbone_freeze_active:
            self._set_backbone_freeze_trainability(model, True)
            if epoch == 1:
                LOGGER.info(
                    f"Backbone freeze warm-start enabled for {self.backbone_freeze_epochs} epochs; "
                    "training neck, feature fusion, adapters, and head"
                )
        else:
            requested_head_warmup = self._head_warmup_active(epoch)
            head_warmup_supported = any(group.get("is_head", False) for group in optimizer.param_groups)
            head_warmup_active = requested_head_warmup and head_warmup_supported
            if requested_head_warmup and not head_warmup_supported and epoch == 1:
                LOGGER.warning(
                    "Head warmup requested, but optimizer has no separate head parameter group; ignoring"
                )
            self._set_head_warmup_trainability(model, head_warmup_active)

        if gradual_unfreeze_phase:
            gradual_backbone_original_lrs = self._apply_gradual_backbone_lrs(optimizer, epoch)
            if epoch in {
                1,
                self.gradual_unfreeze_head_epochs + 1,
                self.gradual_unfreeze_stage_epochs + 1,
            }:
                trainable, total = self._trainable_parameter_summary(model)
                active_lr, backbone_lr, head_lr = self._optimizer_lr_summary(optimizer)
                LOGGER.info(
                    f"Gradual unfreeze phase '{gradual_unfreeze_phase}': "
                    f"trainable_params={trainable}/{total}, "
                    f"active_lr={active_lr:.2e}, backbone_lr={backbone_lr:.2e}, head_lr={head_lr:.2e}"
                )
        elif head_warmup_active:
            head_warmup_original_lrs = self._apply_head_warmup_lrs(optimizer)
            if epoch == self._head_warmup_start_epoch():
                LOGGER.info(
                    f"Head warmup enabled for {self.head_warmup_epochs} epochs "
                    f"(head_lr_mult={self.head_warmup_lr_mult:g})"
                )
        epoch_lr, epoch_backbone_lr, epoch_head_lr = self._optimizer_lr_summary(optimizer)

        id_loss_weight = self._effective_id_loss_weight(epoch)
        center_loss_weight = self._effective_center_loss_weight(epoch)

        running_losses = torch.zeros(4, device=self.device)
        n_batches = 0
        t0 = time.monotonic()

        # AMP: mixed precision on CUDA for ~2x throughput, skip on CPU/MPS
        use_amp = self.device.type == "cuda"
        if not hasattr(self, "_scaler"):
            self._scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scaler = self._scaler

        batch_bar = tqdm(
            loader, desc=f"  Epoch {epoch}/{self.epochs}",
            leave=False, unit="batch",
        )
        for imgs, pids, _ in batch_bar:
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(imgs)
                logits, features = self._split_model_output(output)

                # ID loss — CE uses model logits; margin classifiers use embeddings.
                if self.classifier_loss == "ce":
                    loss_id = self._classification_loss_for_logits(criterion_id, logits, pids, epoch, features)
                else:
                    cls_features = self._classification_features(features)
                    if cls_features is None:
                        raise RuntimeError(
                            f"classifier_loss={self.classifier_loss} requires embedding features; "
                            f"model loss contract is {self._model_loss_type()}"
                        )
                    loss_id = criterion_id(cls_features, pids)
                loss = id_loss_weight * loss_id

                # Triplet loss — L2-normalize features so Euclidean distance in
                # triplet loss aligns with cosine distance used at evaluation.
                loss_tri = torch.tensor(0.0, device=self.device)
                if criterion_metric is not None and features is not None:
                    loss_tri = self._metric_loss_for_features(criterion_metric, features, pids)
                    loss = loss + self.metric_loss_weight * loss_tri

                loss_evidence = self._evidence_auxiliary_loss(features, pids)
                loss = loss + loss_evidence

                # Center loss — only on embeddings, never on logits
                center_features = self._center_features(features)
                loss_cen = torch.tensor(0.0, device=self.device)
                if center_features is not None and center_loss_weight > 0 and not head_warmup_active:
                    loss_cen = criterion_center(center_features, pids) * center_loss_weight
                    loss = loss + loss_cen

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, batch {n_batches + 1}: "
                    f"loss={loss.detach().item():.6g}, "
                    f"id_loss={loss_id.detach().item():.6g}, "
                    f"triplet_loss={loss_tri.detach().item():.6g}, "
                    f"evidence_loss={loss_evidence.detach().item():.6g}, "
                    f"center_loss={loss_cen.detach().item():.6g}"
                )

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping (transformer training stability)
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)

            # Center loss has its own optimizer with special LR
            if center_features is not None and center_loss_weight > 0 and not head_warmup_active:
                scaler.unscale_(optimizer_center)
                for param in criterion_center.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / center_loss_weight)
                scaler.step(optimizer_center)

            scaler.update()

            # EMA update (parameters + buffers)
            # Float buffers (BN running_mean/var) are EMA'd so their
            # statistics match the EMA model's feature distribution.
            # Integer buffers (num_batches_tracked, index tensors) are copied.
            if ema_model is not None:
                decay = self.ema_decay
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)
                for ema_b, model_b in zip(ema_model.buffers(), model.buffers()):
                    if ema_b.is_floating_point():
                        ema_b.data.mul_(decay).add_(model_b.data, alpha=1.0 - decay)
                    else:
                        ema_b.data.copy_(model_b.data)

            running_losses.add_(
                torch.stack(
                    (
                        loss.detach(),
                        loss_id.detach(),
                        loss_tri.detach(),
                        loss_cen.detach(),
                    )
                ).float()
            )
            n_batches += 1
            if n_batches % 20 == 0:
                batch_bar.set_postfix(loss=f"{(running_losses[0] / n_batches).item():.4f}")

        if gradual_backbone_original_lrs is not None:
            for group, lr in zip(optimizer.param_groups, gradual_backbone_original_lrs):
                group["lr"] = lr
        if head_warmup_original_lrs is not None:
            for group, lr in zip(optimizer.param_groups, head_warmup_original_lrs):
                group["lr"] = lr

        # Scheduler step
        if epoch > self.warmup_epochs:
            scheduler.step()
        elif self.warmup_epochs > 0:
            # Linear warmup — respect per-group base LR (layer-decay)
            warmup_factor = epoch / self.warmup_epochs
            for pg in optimizer.param_groups:
                base_lr = pg.get("_base_lr", self.lr)
                pg["lr"] = base_lr * warmup_factor

        elapsed = time.monotonic() - t0
        average_losses = (running_losses / max(n_batches, 1)).cpu().tolist()
        return TrainMetrics(
            epoch=epoch,
            loss=average_losses[0],
            id_loss=average_losses[1],
            triplet_loss=average_losses[2],
            center_loss=average_losses[3],
            lr=epoch_lr,
            elapsed_s=elapsed,
            backbone_lr=epoch_backbone_lr,
            head_lr=epoch_head_lr,
        )

    @torch.no_grad()
    def _calibrate_bn(self, model, data_loader, num_batches: int = 50):
        """Run forward passes to calibrate BN running stats for the EMA model."""
        model.train()
        with torch.no_grad():
            for i, (imgs, _, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                imgs = imgs.to(self.device)
                model(imgs)
        model.eval()

    def _validate(self, epoch, model, query_loader, gallery_loader) -> ValMetrics:
        recipe = self._resolve_training_recipe(model)
        use_flip = self.flip_tta if self.flip_tta is not None else recipe.default_flip_tta
        visibility_distance = self.inference_feature == "visibility_weighted_parts"
        evidence_distance = self.inference_feature == "evidence_sinkhorn"
        structured_distance = visibility_distance or evidence_distance
        q_feats, q_pids, q_camids = extract_features(
            model,
            query_loader,
            self.device,
            desc="Query",
            flip_tta=use_flip,
            normalize=not structured_distance,
        )
        g_feats, g_pids, g_camids = extract_features(
            model,
            gallery_loader,
            self.device,
            desc="Gallery",
            flip_tta=use_flip,
            normalize=not structured_distance,
        )
        distmat = compute_distance_matrix(
            q_feats,
            g_feats,
            metric=(
                "evidence_sinkhorn"
                if evidence_distance
                else "visibility_weighted_parts"
                if visibility_distance
                else "cosine"
            ),
            part_dim=self.feat_dim if structured_distance else None,
            part_count=visibility_part_count(self.head_parts) if structured_distance else None,
            role_count=self.evidence_num_roles if evidence_distance else None,
            beta=self.branch_metric_part_weight,
            topk=self.evidence_rerank_topk if evidence_distance else None,
            sinkhorn_iters=self.evidence_sinkhorn_iters,
            sinkhorn_temperature=self.evidence_sinkhorn_temperature,
        )
        del q_feats, g_feats
        cmc, mAP = evaluate_ranking(distmat, q_pids, g_pids, q_camids, g_camids)
        del distmat, q_pids, g_pids, q_camids, g_camids
        return ValMetrics(
            epoch=epoch,
            mAP=mAP,
            rank1=float(cmc[0]) if len(cmc) > 0 else 0.0,
            rank5=float(cmc[4]) if len(cmc) > 4 else 0.0,
            rank10=float(cmc[9]) if len(cmc) > 9 else 0.0,
        )

    @staticmethod
    def _get_num_classes(model) -> int:
        """Infer num_classes from model's classifier layer."""
        if hasattr(model, "classifier"):
            return model.classifier.out_features
        # Multi-branch models (e.g. CSLTinyViT with BNNeck3 head)
        for name, module in model.named_modules():
            if name.endswith("classifier") and hasattr(module, "out_features"):
                return module.out_features
        return -1

    def _checkpoint_metadata(self, model: nn.Module) -> dict[str, Any]:
        """Return stable model/training metadata shared by all checkpoint types."""
        recipe = self._resolve_training_recipe(model)
        metadata = {
            "model_name": self.model_name,
            "model_family": recipe.family,
            "training_family": recipe.family,
            "training_recipe": recipe.name,
            "is_vit": recipe.family == "transformer",
            "is_transformer": recipe.family == "transformer",
            "dataset": self.dataset_name,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "eta_min": self.eta_min,
            "num_classes": self._get_num_classes(model),
            "preprocess": self.preprocess,
            "loss_type": self.loss_type,
            "classifier_loss": self.classifier_loss,
            "inference_feature": self.inference_feature,
            "timm_model_name": getattr(model, "timm_model_name", None),
            "use_timm_head": getattr(model, "use_timm_head", None),
            "feature_fusion": self.feature_fusion,
            "post_fusion_mixer": self.post_fusion_mixer,
            "post_fusion_mixer_reduction": self.post_fusion_mixer_reduction,
            "post_fusion_mixer_kernel": list(self.post_fusion_mixer_kernel),
            "post_fusion_mixer_gamma_init": self.post_fusion_mixer_gamma_init,
            "feat_dim": self.feat_dim,
            "neck_dim": self.neck_dim,
            "drop_path_rate": self.drop_path_rate,
            "attention_window_layout": self.attention_window_layout,
            "attention_bias": self.attention_bias,
            "attention_mask": self.attention_mask,
            "attention_shift": self.attention_shift,
            "stage3_global": self.stage3_global,
            "vit_lr_profile": self.vit_lr_profile,
            "optimizer_name": recipe.optimizer_name,
            "grad_clip": recipe.grad_clip,
            "backbone_freeze_epochs": self.backbone_freeze_epochs,
            "gradual_unfreeze": self.gradual_unfreeze,
            "gradual_unfreeze_head_epochs": self.gradual_unfreeze_head_epochs,
            "gradual_unfreeze_stage_epochs": self.gradual_unfreeze_stage_epochs,
            "gradual_unfreeze_backbone_lr_mult": self.gradual_unfreeze_backbone_lr_mult,
            "gradual_unfreeze_backbone_lr_epochs": self.gradual_unfreeze_backbone_lr_epochs,
            "reid_adapter_stages": list(self.reid_adapter_stages),
            "reid_adapter_reduction": self.reid_adapter_reduction,
            "head_pool": self.head_pool,
            "head_parts": list(self.head_parts),
            "head_type": self.head_type,
            "part_pooling": self.part_pooling,
            "num_part_tokens": self.num_part_tokens,
            "evidence_num_roles": self.evidence_num_roles,
            "decouple_patterns": self.decouple_patterns,
            "pattern_adapter_dim": self.pattern_adapter_dim,
            "stripe_visibility": self.stripe_visibility,
            "drop_global_aux": self.drop_global_aux,
            "drop_global_aux_ratio": self.drop_global_aux_ratio,
            "branch_aware_metric": self.branch_aware_metric,
            "branch_metric_part_weight": self.branch_metric_part_weight,
            "evidence_alignment_loss_weight": self.evidence_alignment_loss_weight,
            "evidence_alignment_margin": self.evidence_alignment_margin,
            "evidence_sinkhorn_iters": self.evidence_sinkhorn_iters,
            "evidence_sinkhorn_temperature": self.evidence_sinkhorn_temperature,
            "evidence_rerank_topk": self.evidence_rerank_topk,
            "evidence_null_loss_weight": self.evidence_null_loss_weight,
            "evidence_diversity_loss_weight": self.evidence_diversity_loss_weight,
            "head_warmup_epochs": self.head_warmup_epochs,
            "head_warmup_lr_mult": self.head_warmup_lr_mult,
            "early_id_loss_weight": self.early_id_loss_weight,
            "early_id_loss_epochs": self.early_id_loss_epochs,
            "center_loss_ramp_start_epoch": self.center_loss_ramp_start_epoch,
            "center_loss_ramp_end_epoch": self.center_loss_ramp_end_epoch,
            "aux_ce_weight": self.aux_ce_weight,
            "aux_ce_drop_epoch": self.aux_ce_drop_epoch,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }
        metadata["model"] = {
            "family": recipe.family,
            "training_family": recipe.family,
            "training_recipe": recipe.name,
            "is_vit": recipe.family == "transformer",
            "is_transformer": recipe.family == "transformer",
            "feature_fusion": self.feature_fusion,
            "timm_model_name": getattr(model, "timm_model_name", None),
        }
        if recipe.family == "transformer":
            metadata["model"]["transformer"] = {
                "drop_path_rate": self._max_drop_path(model),
                "attention": {
                    "window_layout": self.attention_window_layout,
                    "bias": self.attention_bias,
                    "mask": self.attention_mask,
                    "shift": self.attention_shift,
                    "stage3_global": self.stage3_global,
                },
                "reid_adapters": {
                    "stages": list(self.reid_adapter_stages),
                    "reduction": self.reid_adapter_reduction,
                },
                "lr_profile": self.vit_lr_profile,
            }
        else:
            metadata["model"][recipe.family] = {
                "timm_model_name": getattr(model, "timm_model_name", None),
                "use_timm_head": getattr(model, "use_timm_head", None),
                "feature_fusion": self.feature_fusion,
            }
        metadata["optimization"] = {
            "recipe": recipe.name,
            "optimizer": recipe.optimizer_name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "grad_clip": recipe.grad_clip,
            "layer_decay": recipe.layer_decay(self),
        }
        if self.data_specs:
            metadata["data_specs"] = [dict(spec) for spec in self.data_specs]
        return metadata

    def _save_checkpoint(
        self, model, path: Path, epoch: int, val: Optional[ValMetrics],
        optimizer=None, optimizer_center=None,
        criterion_center: Optional[CenterLoss] = None,
        criterion_classifier: Optional[nn.Module] = None,
        ema_model=None, best_mAP: float = 0.0,
    ):
        self.checkpoint_manager.save(
            path,
            model=model,
            epoch=epoch,
            val=val,
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            criterion_center=criterion_center,
            criterion_classifier=criterion_classifier,
            ema_model=ema_model,
            best_mAP=best_mAP,
            resumable=optimizer is not None,
        )

    def _save_metrics(
        self, save_dir: Path,
        history: List[TrainMetrics],
        val_history: List[ValMetrics],
        best_epoch: int, best_mAP: float, best_rank1: float,
        average_epoch_time_s: float = 0.0,
        average_eval_time_s: float = 0.0,
        total_end_to_end_time_s: float = 0.0,
    ):
        """Persist full training & validation history to metrics.json."""
        # Group val entries by epoch
        from collections import OrderedDict
        val_by_epoch: OrderedDict[int, dict] = OrderedDict()
        for v in val_history:
            if v.epoch not in val_by_epoch:
                val_by_epoch[v.epoch] = {"epoch": v.epoch}
            val_by_epoch[v.epoch][v.dataset] = {
                "mAP": round(v.mAP, 4), "rank1": round(v.rank1, 4),
                "rank5": round(v.rank5, 4), "rank10": round(v.rank10, 4),
            }

        data = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "epochs": self.epochs,
            "best_epoch": best_epoch,
            "best_mAP": round(best_mAP, 4),
            "best_rank1": round(best_rank1, 4),
            "average_epoch_time_s": round(average_epoch_time_s, 4),
            "average_eval_time_s": round(average_eval_time_s, 4),
            "total_end_to_end_time_s": round(total_end_to_end_time_s, 4),
            "train": [
                {
                    "epoch": m.epoch, "loss": round(m.loss, 5),
                    "id_loss": round(m.id_loss, 5),
                    "triplet_loss": round(m.triplet_loss, 5),
                    "center_loss": round(m.center_loss, 5),
                    "lr": round(m.lr, 8),
                    "backbone_lr": round(m.backbone_lr, 8),
                    "head_lr": round(m.head_lr, 8),
                }
                for m in history
            ],
            "val": list(val_by_epoch.values()),
        }
        path = save_dir / "metrics.json"
        path.write_text(json.dumps(data, indent=2))
        LOGGER.info(f"Saved training metrics to {path}")

    def _save_training_plots(
        self,
        save_dir: Path,
        history: List[TrainMetrics],
        val_history: List[ValMetrics],
    ) -> None:
        """Plot training losses and primary validation metrics after training."""
        if not history:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            LOGGER.warning(f"Could not generate training plots: {exc}")
            return

        train_epochs = [m.epoch for m in history]
        primary_dataset = self.dataset_name.split(",")[0].strip().lower()
        val_by_epoch: dict[int, ValMetrics] = {}
        for val in val_history:
            val_ds = val.dataset.strip().lower()
            if val_ds == primary_dataset and val.epoch not in val_by_epoch:
                val_by_epoch[val.epoch] = val

        if not val_by_epoch:
            for val in val_history:
                val_by_epoch.setdefault(val.epoch, val)

        val_epochs = sorted(val_by_epoch)
        mAP = [val_by_epoch[epoch].mAP for epoch in val_epochs]
        rank1 = [val_by_epoch[epoch].rank1 for epoch in val_epochs]
        rank5 = [val_by_epoch[epoch].rank5 for epoch in val_epochs]
        rank10 = [val_by_epoch[epoch].rank10 for epoch in val_epochs]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=140)

        loss_ax = axes[0]
        loss_ax.plot(train_epochs, [m.loss for m in history], label="loss", linewidth=2)
        loss_ax.plot(train_epochs, [m.id_loss for m in history], label="id_loss")
        loss_ax.plot(train_epochs, [m.triplet_loss for m in history], label="triplet_loss")
        loss_ax.plot(train_epochs, [m.center_loss for m in history], label="center_loss")
        loss_ax.set_title("Training Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.grid(True, alpha=0.3)
        loss_ax.legend()

        metrics_ax = axes[1]
        if val_epochs:
            metrics_ax.plot(val_epochs, mAP, label="mAP", linewidth=2)
            metrics_ax.plot(val_epochs, rank1, label="Rank-1")
            metrics_ax.plot(val_epochs, rank5, label="Rank-5")
            metrics_ax.plot(val_epochs, rank10, label="Rank-10")
            metrics_ax.set_ylim(0.0, 1.0)
        metrics_ax.set_title(f"Validation Metrics ({primary_dataset})")
        metrics_ax.set_xlabel("Epoch")
        metrics_ax.set_ylabel("Score")
        metrics_ax.grid(True, alpha=0.3)
        if val_epochs:
            metrics_ax.legend()

        fig.tight_layout()
        path = save_dir / "training_curves.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info(f"Saved training curves to {path}")

    def _make_save_dir(self) -> Path:
        base = self.project / self.name
        if base.exists():
            idx = 1
            while (self.project / f"{self.name}_{idx}").exists():
                idx += 1
            base = self.project / f"{self.name}_{idx}"
        base.mkdir(parents=True, exist_ok=True)
        return base
