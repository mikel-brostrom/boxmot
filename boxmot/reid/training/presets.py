"""Promoted model-to-recipe selections for ReID training."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

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


def _normalize_int_tuple(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        parts = [part for part in values.replace(";", ",").split(",") if part.strip()]
        return tuple(int(part) for part in parts)
    if isinstance(values, int):
        return (int(values),)
    return tuple(int(value) for value in values)


def _normalize_int_pair(value: Any, default: tuple[int, int] = (5, 3)) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, str):
        parts = [part for part in value.replace(";", ",").split(",") if part.strip()]
        if len(parts) == 1:
            parts *= 2
        if len(parts) != 2:
            raise ValueError(f"Expected one or two comma-separated integers, got {value!r}")
        return (int(parts[0]), int(parts[1]))
    values = tuple(int(part) for part in value)
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) != 2:
        raise ValueError(f"Expected one or two integers, got {value!r}")
    return values


def build_training_namespace(
    payload: Mapping[str, Any],
    *,
    explicit_keys: Iterable[str] | None = None,
) -> SimpleNamespace:
    """Resolve defaults, recipes, config files, and explicit train overrides."""

    explicit = set(explicit_keys or ())
    values = load_training_defaults()
    values.update(dict(payload))

    config_values: dict[str, Any] | None = None
    config_path = values.pop("cfg", None)
    if config_path is not None:
        config_values = load_training_config(config_path)

    recipe_name = values.pop("recipe", None)
    if config_values is not None and "recipe" not in explicit and config_values.get("recipe") is not None:
        recipe_name = config_values["recipe"]

    effective_model = values.get("model")
    if config_values is not None and "model" not in explicit and config_values.get("model") is not None:
        effective_model = config_values["model"]
    if recipe_name is None:
        recipe_name = default_training_recipe_for_model(effective_model)

    if recipe_name is not None:
        for key, value in load_training_recipe(recipe_name).items():
            if key not in explicit:
                values[key] = value
    if config_values is not None:
        for key, value in config_values.items():
            if key != "recipe" and key not in explicit:
                values[key] = value

    if values.get("project") is not None:
        values["project"] = Path(values["project"])
    image_size = values.get("imgsz")
    if isinstance(image_size, (list, tuple)):
        values["imgsz"] = tuple(image_size)
    elif isinstance(image_size, int):
        values["imgsz"] = (image_size, image_size // 2)
    values["head_parts"] = _normalize_int_tuple(values.get("head_parts", (1, 2)))
    values["reid_adapter_stages"] = _normalize_int_tuple(values.get("reid_adapter_stages", ()))
    values["post_fusion_mixer_kernel"] = _normalize_int_pair(
        values.get("post_fusion_mixer_kernel", (5, 3))
    )
    eval_datasets = values.get("eval_datasets", ())
    if isinstance(eval_datasets, str):
        eval_datasets = [item.strip() for item in eval_datasets.split(",") if item.strip()]
    values["eval_datasets"] = list(eval_datasets)
    if "backbone_freeze_epochs" not in explicit:
        epochs = int(values.get("epochs", 0) or 0)
        freeze_epochs = int(values.get("backbone_freeze_epochs", 0) or 0)
        if epochs >= 0 and freeze_epochs > epochs:
            values["backbone_freeze_epochs"] = epochs
    values.setdefault("train_explicit_keys", tuple(sorted(explicit)))
    return SimpleNamespace(**values)


__all__ = (
    "TRAIN_DEFAULTS_PATH",
    "TRAINING_RECIPES_DIR",
    "build_training_namespace",
    "default_training_recipe_for_model",
    "list_training_recipes",
    "load_training_config",
    "load_training_defaults",
    "load_training_recipe",
    "normalize_training_values",
    "schema_training_defaults",
)
