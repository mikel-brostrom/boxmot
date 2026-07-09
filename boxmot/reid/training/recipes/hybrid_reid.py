# BoxMOT AGPL-3.0 license

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from boxmot.reid.training.recipes.base import TrainingRecipe

if TYPE_CHECKING:
    from boxmot.reid.training.trainer import ReIDTrainer


class HybridReIDRecipe(TrainingRecipe):
    """Hybrid CNN/attention ReID defaults with AdamW and no-decay groups."""

    def __init__(self) -> None:
        super().__init__(
            family="hybrid",
            name="hybrid_reid",
            optimizer_name="AdamW",
            grad_clip=1.0,
            default_flip_tta=False,
            default_triplet_soft_margin=True,
        )

    def apply_pre_build_defaults(self, trainer: ReIDTrainer) -> None:
        self._apply_hybrid_defaults(trainer, include_architecture=True)

    def apply_defaults(self, trainer: ReIDTrainer) -> None:
        self._apply_hybrid_defaults(trainer, include_architecture=False)

    @staticmethod
    def _apply_hybrid_defaults(
        trainer: ReIDTrainer,
        *,
        include_architecture: bool,
    ) -> None:
        if trainer.resume or "recipe" in trainer.explicit_hparams:
            return

        explicit = trainer.explicit_hparams

        def set_if_default(name: str, value, defaults: tuple, aliases: tuple[str, ...] = ()) -> None:
            explicit_name = name in explicit or any(alias in explicit for alias in aliases)
            if not explicit_name and getattr(trainer, name) in defaults:
                setattr(trainer, name, value)

        set_if_default("weight_decay", 1e-4, (5e-4,))
        set_if_default("epochs", 120, (200, 250))
        set_if_default("eta_min", 1e-7, (1e-6,))
        if (
            "triplet_soft_margin" not in explicit
            and "soft_margin_triplet" not in explicit
            and trainer.triplet_soft_margin in {None, False}
        ):
            trainer.triplet_soft_margin = True
        set_if_default("ema_decay", 0.999, (None, 0, 0.0))
        set_if_default("random_erasing", 0.35, (0.5,))
        set_if_default("random_patch", False, (True,))
        set_if_default("color_jitter", False, (True,))
        set_if_default("gaussian_blur", False, (True,))
        set_if_default("random_grayscale", 0.0, (0.1,))
        set_if_default("drop_path_rate", 0.0, (0.1,))
        set_if_default("backbone_freeze_epochs", min(10, trainer.epochs), (0,))
        set_if_default("gradual_unfreeze", False, (True,))
        set_if_default("gradual_unfreeze_head_epochs", 0, (5,))
        set_if_default("gradual_unfreeze_stage_epochs", 0, (10, 20))
        set_if_default("gradual_unfreeze_backbone_lr_mult", 1.0, (0.1,))
        set_if_default("gradual_unfreeze_backbone_lr_epochs", 0, (5,))
        set_if_default("head_warmup_epochs", 0, (5,))
        set_if_default("head_warmup_lr_mult", 2.0, (2.0,))

        if not include_architecture:
            return

        set_if_default("img_size", (384, 128), ((256, 128),), ("imgsz",))
        set_if_default("batch_size", 64, (128,))
        set_if_default("feature_fusion", "final", ("last2", "last3", "global_final_parts_stage2"))
        set_if_default("head_pool", "avg", ("gelu_gem", "gem"))
        set_if_default("head_parts", (1,), ((1, 2),))
        set_if_default("metric_feature", "auto", ("raw_concat",))
        set_if_default("inference_feature", "concat_bn", ("norm_concat_bn",))

    def build_param_groups(self, trainer: ReIDTrainer, model: nn.Module) -> list[dict]:
        return trainer._build_mobilenetv4_param_groups(model)

    def classifier_param_group(self, trainer: ReIDTrainer, params: list[nn.Parameter]) -> dict:
        return {
            "params": params,
            "lr": trainer.lr,
            "weight_decay": 0.0,
            "is_head": True,
            "is_backbone": False,
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
