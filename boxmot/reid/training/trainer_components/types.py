"""Shared value objects used by the ReID training workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from boxmot.reid.training.losses import (
    AdaSPLoss,
    CenterLoss,
    CrossScaleMajorityMarginLoss,
    TreeBoostAPLoss,
)
from boxmot.reid.training.recipes import TrainingRecipe


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
    csmm_loss: float = 0.0
    treeboost_loss: float = 0.0
    global_ap_loss: float = 0.0
    hpgrd_loss: float = 0.0
    hpgrd_global_loss: float = 0.0
    hpgrd_part_loss: float = 0.0
    hpgrd_background_loss: float = 0.0
    hpgrd_part_drop_loss: float = 0.0
    hpgrd_gradient_scale: float = 0.0
    late_interaction_loss: float = 0.0
    late_interaction_distill_loss: float = 0.0
    pav_consistency_loss: float = 0.0
    clean_student_consistency_loss: float = 0.0
    anatomical_loss: float = 0.0
    anatomical_distill_loss: float = 0.0
    anatomical_attention_loss: float = 0.0
    anatomical_visibility_loss: float = 0.0
    anatomical_contrastive_loss: float = 0.0
    anatomical_descriptor_distill_loss: float = 0.0
    anatomical_branch_distill_loss: float = 0.0
    anatomical_branch_global_loss: float = 0.0
    anatomical_branch_coarse_loss: float = 0.0
    anatomical_branch_fine_loss: float = 0.0
    anatomical_pose_teacher_loss: float = 0.0
    anatomical_semantic_foreground_loss: float = 0.0
    anatomical_semantic_part_loss: float = 0.0
    anatomical_query_distill_loss: float = 0.0
    anatomical_query_relational_distill_loss: float = 0.0
    anatomical_query_diversity_loss: float = 0.0
    anatomical_part_triplet_loss: float = 0.0
    anatomical_accessory_valid_fraction: float = 0.0
    identity_register_diversity_loss: float = 0.0
    anatomical_local_scale_loss: float = 0.0
    anatomical_fine_scale_loss: float = 0.0
    anatomical_cross_scale_loss: float = 0.0
    anatomical_valid_part_fraction: float = 0.0
    anatomical_cross_camera_anchor_fraction: float = 0.0
    forward_elapsed_s: float = 0.0
    backbone_lr: float = 0.0
    head_lr: float = 0.0
    mcpt_loss: float = 0.0
    mcpt_smoothness: float = 0.0
    mcpt_identity: float = 0.0
    mcpt_mean_abs_displacement: float = 0.0
    mcpt_boundary_1: float = 0.25
    mcpt_boundary_2: float = 0.5
    mcpt_boundary_3: float = 0.75
    mcpt_boundary_std: float = 0.0
    mcpt_cap_fraction: float = 0.0
    mcpt_local_gate: float = 0.0
    mcpt_fine_gate: float = 0.0
    adasp_loss: float = 0.0
    part_relation_loss: float = 0.0
    part_to_global_loss: float = 0.0
    jpm_id_loss: float = 0.0
    jpm_metric_loss: float = 0.0
    multilevel_suppression_loss: float = 0.0
    multilevel_suppression_weight: float = 0.0
    multilevel_suppression_effective_ratio: float = 0.0
    multilevel_suppression_coarse_erased_fraction: float = 0.0
    multilevel_suppression_fine_erased_fraction: float = 0.0
    multilevel_suppression_global_cam_active_fraction: float = 0.0
    multilevel_suppression_coarse_cam_active_fraction: float = 0.0


@dataclass
class ValMetrics:
    """Metrics from validation (CMC + mAP)."""

    epoch: int
    mAP: float
    rank1: float
    rank5: float
    rank10: float
    dataset: str = ""
    mcpt_disabled_mAP: float | None = None
    mcpt_disabled_rank1: float | None = None


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
    criterion_csmm: Optional[CrossScaleMajorityMarginLoss] = None
    criterion_treeboost: Optional[TreeBoostAPLoss] = None
    criterion_adasp: Optional[AdaSPLoss] = None


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


@dataclass
class _TrainingTimeEstimator:
    """Estimate remaining wall time without blending epochs and evaluations."""

    total_epochs: int
    eval_interval: int
    epoch_window: int = 20
    eval_window: int = 3
    fallback_epoch_s: float = 0.0
    fallback_eval_s: float = 0.0
    epoch_samples_s: list[float] = field(default_factory=list)
    eval_samples_s: list[float] = field(default_factory=list)
    training_phase: str | None = None

    @staticmethod
    def _valid_duration(value: float) -> bool:
        return math.isfinite(value) and value > 0

    @staticmethod
    def _rolling_median(values: list[float], window: int) -> float:
        samples = values[-max(int(window), 1) :]
        return float(median(samples)) if samples else 0.0

    def add_epoch(self, duration_s: float, *, phase: str) -> None:
        """Record one epoch, resetting samples after a freeze/unfreeze transition."""
        if phase != self.training_phase:
            self.training_phase = phase
            self.epoch_samples_s.clear()
        if self._valid_duration(duration_s):
            self.epoch_samples_s.append(float(duration_s))

    def add_evaluation(self, duration_s: float) -> None:
        """Record one complete evaluation event, including all configured datasets."""
        if self._valid_duration(duration_s):
            self.eval_samples_s.append(float(duration_s))

    @property
    def epoch_duration_s(self) -> float:
        measured = self._rolling_median(self.epoch_samples_s, self.epoch_window)
        return measured or self.fallback_epoch_s

    @property
    def evaluation_duration_s(self) -> float:
        measured = self._rolling_median(self.eval_samples_s, self.eval_window)
        return measured or self.fallback_eval_s

    def remaining_evaluations(self, completed_epoch: int) -> int:
        """Count scheduled evaluation events strictly after a completed epoch."""
        if completed_epoch >= self.total_epochs:
            return 0
        interval = max(int(self.eval_interval), 1)
        scheduled = set(range(interval, self.total_epochs + 1, interval))
        scheduled.add(self.total_epochs)
        return sum(epoch > completed_epoch for epoch in scheduled)

    def estimate_remaining_s(self, completed_epoch: int) -> float | None:
        """Return estimated remaining wall time after ``completed_epoch``."""
        epoch_duration = self.epoch_duration_s
        if not self._valid_duration(epoch_duration):
            return None
        evaluations_left = self.remaining_evaluations(completed_epoch)
        evaluation_duration = self.evaluation_duration_s
        if evaluations_left and not self._valid_duration(evaluation_duration):
            return None
        epochs_left = max(self.total_epochs - completed_epoch, 0)
        return epochs_left * epoch_duration + evaluations_left * evaluation_duration
