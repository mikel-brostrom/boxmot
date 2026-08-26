"""Promoted model-to-recipe selections for ReID training."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from boxmot.reid.training.config import ReIDTrainConfig, train_hparams_to_args
from boxmot.utils.config import load_yaml_mapping

TRAIN_DEFAULTS_PATH = Path(__file__).resolve().parent / "configs" / "defaults.yaml"
TRAINING_RECIPES_DIR = Path(__file__).resolve().parent / "configs" / "recipes"

_FIELD_ALIASES = {
    "dataset_name": "dataset",
    "img_size": "imgsz",
    "p": "p_ids",
    "k": "k_instances",
    "model_name": "model",
    "loss_type": "loss",
}

_DEFAULT_RECIPES = {
    "csl_tinyvit_11m": "csl_tinyvit_11m",
    "csl_tinyvit_11m_v20": "csl_tinyvit_11m",
    "csl_tinyvit_7m_v20": "csl_tinyvit_7m_v20",
    "mobilenetv4_conv_medium_v20": "mobilenetv4_conv_medium_v20",
    "mobilenetv4_hybrid_medium_v20": "mobilenetv4_hybrid_medium_v20",
}


def default_training_recipe_for_model(model: str | Path | None) -> str | None:
    """Return the promoted recipe for a model, if one is defined."""
    if model in {None, ""}:
        return None
    return _DEFAULT_RECIPES.get(Path(str(model)).stem.lower())


def _component_defaults(component: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in fields(component):
        if item.name in {"explicit_hparams", "resume"}:
            continue
        values[_FIELD_ALIASES.get(item.name, item.name)] = getattr(component, item.name)
    return values


def schema_training_defaults() -> dict[str, Any]:
    """Flatten defaults from the canonical typed ReID training schema."""
    config = ReIDTrainConfig()
    defaults: dict[str, Any] = {}
    for component in (
        config.run,
        config.data,
        config.model,
        config.loss,
        config.optimization,
        config.augmentation,
        config.evaluation,
    ):
        defaults.update(_component_defaults(component))
    return defaults


def load_training_defaults() -> dict[str, Any]:
    """Return typed schema defaults overlaid by promoted train defaults."""
    defaults = schema_training_defaults()
    defaults.update(load_yaml_mapping(TRAIN_DEFAULTS_PATH))
    return defaults


def normalize_training_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize nested recipes or flat configs to public train argument keys."""
    return train_hparams_to_args(dict(values))


def load_training_recipe(name: str) -> dict[str, Any]:
    """Load a named ReID training recipe."""
    recipe_path = TRAINING_RECIPES_DIR / f"{name}.yaml"
    if not recipe_path.exists():
        available = list_training_recipes()
        raise FileNotFoundError(
            f"Training recipe '{name}' not found at {recipe_path}. "
            f"Available recipes: {', '.join(available) or '(none)'}"
        )
    return normalize_training_values(load_yaml_mapping(recipe_path))


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load an explicit ReID training config or saved hparams mapping."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    return normalize_training_values(load_yaml_mapping(config_path))


def list_training_recipes() -> list[str]:
    """Return sorted names of available ReID training recipes."""
    if not TRAINING_RECIPES_DIR.is_dir():
        return []
    return sorted(path.stem for path in TRAINING_RECIPES_DIR.glob("*.yaml"))


__all__ = (
    "TRAIN_DEFAULTS_PATH",
    "TRAINING_RECIPES_DIR",
    "default_training_recipe_for_model",
    "list_training_recipes",
    "load_training_config",
    "load_training_defaults",
    "load_training_recipe",
    "normalize_training_values",
    "schema_training_defaults",
)
