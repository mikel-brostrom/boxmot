# BoxMOT AGPL-3.0 license

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from boxmot.reid.training.recipes.base import TrainingRecipe
from boxmot.utils import logger as LOGGER

if TYPE_CHECKING:
    from boxmot.reid.training.trainer import ReIDTrainer


class TransformerReIDRecipe(TrainingRecipe):
    """Transformer-style ReID defaults with AdamW and layer-wise LR decay."""

    def __init__(self) -> None:
        super().__init__(
            family="transformer",
            name="transformer_reid",
            optimizer_name="AdamW",
            grad_clip=1.0,
            default_flip_tta=True,
            default_triplet_soft_margin=True,
        )

    def apply_defaults(self, trainer: ReIDTrainer) -> None:
        trainer._apply_vit_training_defaults()
        if trainer.ema_decay is None:
            trainer.ema_decay = 0
            LOGGER.info("Transformer recipe: leaving EMA disabled by default")

    def resolve_label_smooth(self, trainer: ReIDTrainer, label_smooth: float) -> float:
        if label_smooth > 0 and "label_smooth" not in trainer.explicit_hparams:
            label_smooth = 0.05
            LOGGER.info(
                f"Transformer recipe: reducing label smoothing to {label_smooth} "
                f"(was {trainer.label_smooth})"
            )
        return label_smooth

    def build_param_groups(self, trainer: ReIDTrainer, model: nn.Module) -> list[dict]:
        return trainer._build_vit_param_groups(model)

    def classifier_param_group(self, trainer: ReIDTrainer, params: list[nn.Parameter]) -> dict:
        return {
            "params": params,
            "lr": trainer.lr,
            "weight_decay": 0.0,
            "is_head": True,
        }

    def build_optimizer(
        self,
        trainer: ReIDTrainer,
        parameter_groups: list[dict],
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            parameter_groups,
            lr=trainer.lr,
            weight_decay=trainer.weight_decay,
        )

    def layer_decay(self, trainer: ReIDTrainer) -> float:
        return 0.95 if trainer.vit_lr_profile == "layer_decay" else 1.0
