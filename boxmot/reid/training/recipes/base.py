# BoxMOT AGPL-3.0 license

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from boxmot.reid.training.trainer import ReIDTrainer


@dataclass
class TrainingRecipe:
    """Optimization and training defaults for a backbone family."""

    family: str
    name: str
    optimizer_name: str
    grad_clip: float
    default_flip_tta: bool
    default_triplet_soft_margin: bool

    def apply_defaults(self, trainer: ReIDTrainer) -> None:
        """Apply recipe defaults after the model has been built."""
        del trainer

    def apply_pre_build_defaults(self, trainer: ReIDTrainer) -> None:
        """Apply recipe defaults that must be resolved before model construction."""
        del trainer

    def resolve_label_smooth(self, trainer: ReIDTrainer, label_smooth: float) -> float:
        """Return the effective CE label smoothing for this recipe."""
        del trainer
        return label_smooth

    def build_param_groups(self, trainer: ReIDTrainer, model: nn.Module) -> list[dict]:
        """Build optimizer parameter groups for this recipe."""
        raise NotImplementedError

    def classifier_param_group(self, trainer: ReIDTrainer, params: list[nn.Parameter]) -> dict:
        """Build the optional margin-classifier parameter group."""
        raise NotImplementedError

    def build_optimizer(
        self,
        trainer: ReIDTrainer,
        parameter_groups: list[dict],
    ) -> torch.optim.Optimizer:
        """Build the model optimizer for this recipe."""
        raise NotImplementedError

    def layer_decay(self, trainer: ReIDTrainer) -> float:
        """Return the effective layer-decay scalar for hparams metadata."""
        del trainer
        return 1.0
