# BoxMOT AGPL-3.0 license

from __future__ import annotations

from boxmot.reid.training.recipes.base import TrainingRecipe
from boxmot.reid.training.recipes.cnn_reid import CNNReIDRecipe


class LMBNReIDRecipe(CNNReIDRecipe):
    """LMBN recipe using the CNN optimization defaults."""

    def __init__(self) -> None:
        TrainingRecipe.__init__(
            self,
            family="cnn",
            name="lmbn_reid",
            optimizer_name="Adam",
            grad_clip=0.0,
            default_flip_tta=False,
            default_triplet_soft_margin=False,
        )
