# BoxMOT AGPL-3.0 license

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from boxmot.reid.training.recipes.base import TrainingRecipe

if TYPE_CHECKING:
    from boxmot.reid.training.trainer import ReIDTrainer


class CNNReIDRecipe(TrainingRecipe):
    """CNN-native ReID training defaults."""

    def __init__(self) -> None:
        super().__init__(
            family="cnn",
            name="cnn_reid",
            optimizer_name="Adam",
            grad_clip=0.0,
            default_flip_tta=False,
            default_triplet_soft_margin=False,
        )

    def build_param_groups(self, trainer: ReIDTrainer, model: nn.Module) -> list[dict]:
        return trainer._build_cnn_param_groups(model)

    def classifier_param_group(self, trainer: ReIDTrainer, params: list[nn.Parameter]) -> dict:
        return {
            "params": params,
            "is_head": True,
            "is_backbone": False,
        }

    def build_optimizer(
        self,
        trainer: ReIDTrainer,
        parameter_groups: list[dict],
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            parameter_groups,
            lr=trainer.lr,
            weight_decay=trainer.weight_decay,
        )
