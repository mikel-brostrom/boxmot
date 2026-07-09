# BoxMOT AGPL-3.0 license

from __future__ import annotations

from collections.abc import Callable

from boxmot.reid.backbones.common.typing import RecipeName
from boxmot.reid.backbones.registry import BackboneSpec
from boxmot.reid.training.recipes.base import TrainingRecipe
from boxmot.reid.training.recipes.cnn_reid import CNNReIDRecipe
from boxmot.reid.training.recipes.hybrid_reid import HybridReIDRecipe
from boxmot.reid.training.recipes.legacy_reid import LegacyReIDRecipe
from boxmot.reid.training.recipes.lmbn_reid import LMBNReIDRecipe
from boxmot.reid.training.recipes.transformer_reid import TransformerReIDRecipe

RecipeBuilder = Callable[[], TrainingRecipe]

TRAINING_RECIPE_REGISTRY: dict[str, RecipeBuilder] = {
    "cnn_reid": CNNReIDRecipe,
    "transformer_reid": TransformerReIDRecipe,
    "hybrid_reid": HybridReIDRecipe,
    "legacy_reid": LegacyReIDRecipe,
    "lmbn_reid": LMBNReIDRecipe,
}

FAMILY_DEFAULT_RECIPES: dict[str, RecipeName] = {
    "cnn": "cnn_reid",
    "transformer": "transformer_reid",
    "hybrid": "hybrid_reid",
    "legacy": "legacy_reid",
}


def build_training_recipe(
    name: str | None = None,
    *,
    spec: BackboneSpec | None = None,
) -> TrainingRecipe:
    """Build a training recipe from an explicit name or backbone spec."""
    recipe_name = str(name or (spec.default_recipe if spec is not None else "cnn_reid")).lower()
    try:
        return TRAINING_RECIPE_REGISTRY[recipe_name]()
    except KeyError as exc:
        available = ", ".join(sorted(TRAINING_RECIPE_REGISTRY))
        raise KeyError(f"Unknown training recipe '{recipe_name}'. Available: {available}") from exc


def default_recipe_for_family(family: str) -> TrainingRecipe:
    """Build the default recipe for a backbone family."""
    normalized_family = str(family).lower()
    if normalized_family == "vit":
        normalized_family = "transformer"
    if normalized_family == "convnet":
        normalized_family = "cnn"
    try:
        return build_training_recipe(FAMILY_DEFAULT_RECIPES[normalized_family])
    except KeyError as exc:
        available = ", ".join(sorted(FAMILY_DEFAULT_RECIPES))
        raise ValueError(f"Unsupported training_family={family!r}; expected one of: {available}") from exc


__all__ = (
    "CNNReIDRecipe",
    "FAMILY_DEFAULT_RECIPES",
    "HybridReIDRecipe",
    "LMBNReIDRecipe",
    "LegacyReIDRecipe",
    "TRAINING_RECIPE_REGISTRY",
    "TrainingRecipe",
    "TransformerReIDRecipe",
    "build_training_recipe",
    "default_recipe_for_family",
)
