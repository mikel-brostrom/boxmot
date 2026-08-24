from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from boxmot.configs import CONFIG_ROOT
from boxmot.utils.config import ConfigurationError, load_yaml_mapping, resolve_config_path, validate_config_id

REID_CONFIGS_DIR = CONFIG_ROOT / "reid"


def _required_mapping(payload: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f'{context} must define a "{key}" mapping.')
    return dict(value)


def _required_text(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if value in (None, ""):
        raise ConfigurationError(f'{context} must define "{key}".')
    return str(value)


def iter_reid_config_paths() -> list[Path]:
    """Return all built-in ReID runtime profile paths."""
    return sorted(REID_CONFIGS_DIR.glob("**/*.yaml"))


def resolve_reid_config_path(reference: str | Path) -> Path:
    """Resolve a ReID runtime profile by id, filename, or explicit path."""
    return resolve_config_path(REID_CONFIGS_DIR, reference, "ReID")


def load_reid_config(reference: str | Path) -> dict[str, Any]:
    """Load and validate one ReID runtime profile."""
    path = resolve_reid_config_path(reference)
    raw = load_yaml_mapping(path)
    context = f'ReID config "{path}"'
    reid_id = validate_config_id(_required_text(raw, "id", context), path=path, label="ReID")
    weights = _required_mapping(raw, "weights", context)
    runtime = _required_mapping(raw, "runtime", context)
    preprocessing = _required_mapping(raw, "preprocessing", context)
    model_path = _required_text(weights, "path", context)
    device = str(runtime.get("device") or "auto")
    precision = str(runtime.get("precision") or "fp32").lower()
    if precision not in {"fp16", "fp32", "bf16"}:
        raise ConfigurationError(f"{context} precision must be fp16, fp32, or bf16.")
    preprocess = _required_text(preprocessing, "mode", context)
    image_size = preprocessing.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ConfigurationError(f"{context} preprocessing.image_size must contain exactly two positive integers.")
    return {
        "id": reid_id,
        "model": model_path,
        "uri": str(weights.get("uri") or ""),
        "device": device,
        "precision": precision,
        "preprocess": preprocess,
        "image_size": list(image_size),
        "config_path": path,
    }


def reid_config_to_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a validated ReID profile to the runtime mapping contract."""
    device = "" if config["device"] == "auto" else config["device"]
    return {
        "id": config["id"],
        "model": config["model"],
        "default_model": config["model"],
        "uri": config["uri"],
        "url": config["uri"],
        "model_url": config["uri"],
        "device": device,
        "precision": config["precision"],
        "half": config["precision"] == "fp16",
        "preprocess": config["preprocess"],
        "imgsz": config["image_size"],
    }


def load_reid_profile(reference: str | Path) -> dict[str, Any]:
    """Load one ReID profile in runtime-friendly form."""
    return reid_config_to_runtime(load_reid_config(reference))


def find_reid_config_for_model(model: str | Path) -> Path | None:
    """Return the ReID profile whose model filename matches ``model``."""
    target_name = Path(str(model)).name.lower()
    for config_path in iter_reid_config_paths():
        try:
            config = load_reid_config(config_path)
        except (ConfigurationError, FileNotFoundError, OSError):
            continue
        if Path(str(config["model"])).name.lower() == target_name:
            return config_path
    return None


def load_runtime_reid_config(reference: str | Path | None) -> dict[str, Any]:
    """Load a ReID runtime profile by id/path or model filename."""
    if reference in (None, ""):
        return {}
    try:
        return load_reid_profile(str(reference))
    except FileNotFoundError:
        pass

    config_path = find_reid_config_for_model(str(reference))
    return load_reid_profile(config_path) if config_path is not None else {}


__all__ = (
    "ConfigurationError",
    "REID_CONFIGS_DIR",
    "find_reid_config_for_model",
    "iter_reid_config_paths",
    "load_reid_config",
    "load_reid_profile",
    "load_runtime_reid_config",
    "reid_config_to_runtime",
    "resolve_reid_config_path",
)
