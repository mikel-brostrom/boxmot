from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

from boxmot.utils import CONFIGS, NUM_THREADS, ROOT, WEIGHTS
from boxmot.utils.misc import resolve_model_path


DEFAULT_DETECTOR = WEIGHTS / "yolov8n.pt"
DEFAULT_REID = WEIGHTS / "osnet_x0_25_msmt17.pt"
RUNTIME_DEFAULTS_PATH = CONFIGS / "runtime" / "default.yaml"

ALIAS_KEYS = {
    "tracking_method": "tracker",
    "yolo_model": "detector",
    "reid_model": "reid",
    "batch": "batch_size",
}


@lru_cache(maxsize=1)
def load_runtime_defaults() -> dict[str, Any]:
    """Load shared runtime defaults for CLI and Python entry points."""
    with open(RUNTIME_DEFAULTS_PATH, "r") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    common = dict(raw_cfg.get("common") or {})
    modes = {key: dict(value or {}) for key, value in dict(raw_cfg.get("modes") or {}).items()}

    if common.get("n_threads") is None:
        common["n_threads"] = NUM_THREADS

    project = common.get("project", "runs")
    common["project"] = (ROOT / str(project)).resolve() if not isinstance(project, Path) else project

    return {"common": common, "modes": modes}


def get_mode_defaults(mode: str) -> dict[str, Any]:
    """Return merged common + mode-specific defaults."""
    defaults = deepcopy(load_runtime_defaults()["common"])
    defaults.update(deepcopy(load_runtime_defaults()["modes"].get(mode, {})))
    return defaults


def get_mode_default(mode: str, key: str) -> Any:
    """Return one shared default value for a mode."""
    return deepcopy(get_mode_defaults(mode).get(key))


def ensure_model_extension(model_path: str | Path | None) -> Path | None:
    """Resolve bare model names to ``.pt`` files under the weights directory."""
    if model_path is None:
        return None

    path = Path(model_path)
    if not path.suffix and "openvino" not in path.name:
        path = path.with_suffix(".pt")
    return resolve_model_path(path)


def normalize_model_argument(
    model_path: str | Path | list[str | Path] | tuple[str | Path, ...] | set[str | Path] | None,
) -> Path | list[Path] | None:
    """Normalize a model path or collection of model paths."""
    if model_path is None:
        return None

    if isinstance(model_path, (list, tuple, set)):
        return [
            path
            for item in model_path
            if (path := ensure_model_extension(item)) is not None
        ]

    return ensure_model_extension(model_path)


def ensure_model_list(model_path: Any, default: Path) -> list[Path]:
    """Return a list of resolved model paths, preserving multi-model inputs."""
    normalized = normalize_model_argument(model_path)
    if normalized is None:
        return [default]
    if isinstance(normalized, list):
        return normalized or [default]
    return [normalized]


def ensure_model_single(model_path: Any, default: Path) -> Path:
    """Return one resolved model path for single-model modes."""
    normalized = normalize_model_argument(model_path)
    if normalized is None:
        return default
    if isinstance(normalized, list):
        return normalized[0] if normalized else default
    return normalized


def parse_classes_value(classes: Any) -> list[int] | None:
    """Normalize class filters from ints, iterables, or comma/space separated strings."""
    if classes is None:
        return None

    if isinstance(classes, str):
        cleaned = classes.replace(",", " ").strip()
        if not cleaned:
            return None
        return [int(token) for token in cleaned.split()]

    if isinstance(classes, (list, tuple, set)):
        return [int(value) for value in classes] or None

    return [int(classes)]


def parse_imgsz_value(imgsz: Any) -> int | tuple[int, int] | None:
    """Accept integer or H,W image sizes in string or sequence form."""
    if imgsz is None:
        return None

    if isinstance(imgsz, int):
        return imgsz

    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return int(imgsz[0])
        if len(imgsz) == 2:
            return int(imgsz[0]), int(imgsz[1])
        raise ValueError(f"Invalid imgsz: {imgsz}")

    if isinstance(imgsz, str):
        cleaned = imgsz.replace(",", " ").strip()
        if not cleaned:
            return None
        parts = cleaned.split()
        if len(parts) == 1:
            return int(parts[0])
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])

    raise ValueError(f"Invalid imgsz: {imgsz}")


def parse_string_list(values: Any) -> list[str]:
    """Normalize strings or iterables into a list of strings."""
    if values is None:
        return []

    if isinstance(values, str):
        cleaned = values.replace(",", " ").strip()
        return cleaned.split() if cleaned else []

    if isinstance(values, (list, tuple, set)):
        return [str(value) for value in values]

    return [str(values)]


def normalize_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize aliases and normalize Python/CLI input values."""
    normalized: dict[str, Any] = {}

    for key, value in overrides.items():
        canonical = ALIAS_KEYS.get(key, key)
        if canonical in normalized and normalized[canonical] != value:
            raise ValueError(f"Conflicting values received for '{canonical}'.")

        if canonical in {"detector", "reid"}:
            normalized[canonical] = normalize_model_argument(value)
        elif canonical == "weights":
            normalized[canonical] = ensure_model_extension(value)
        elif canonical == "tracker":
            normalized[canonical] = str(value).lower()
        elif canonical == "classes":
            normalized[canonical] = parse_classes_value(value)
        elif canonical == "imgsz":
            normalized[canonical] = parse_imgsz_value(value)
        elif canonical in {"objectives", "maximize", "minimize", "include"}:
            normalized[canonical] = parse_string_list(value)
        elif canonical == "project":
            normalized[canonical] = Path(value)
        elif canonical == "n_trials":
            normalized[canonical] = int(value)
        elif canonical == "postprocessing":
            normalized[canonical] = str(value).lower()
        elif canonical == "source":
            normalized[canonical] = None if value is None else str(value)
        else:
            normalized[canonical] = value

    return normalized


def normalize_explicit_keys(explicit_keys: set[str] | None) -> set[str]:
    """Canonicalize explicit CLI/Python override names."""
    if not explicit_keys:
        return set()
    return {ALIAS_KEYS.get(key, key) for key in explicit_keys}


def build_mode_namespace(
    mode: str,
    overrides: Mapping[str, Any],
    *,
    explicit_keys: set[str] | None = None,
) -> SimpleNamespace:
    """Build a runtime namespace for one BoxMOT mode from shared defaults."""
    merged = get_mode_defaults(mode)
    merged.update(normalize_overrides(overrides))

    explicit = normalize_explicit_keys(explicit_keys)

    if mode == "export":
        params = {
            key: value
            for key, value in merged.items()
            if key not in {"detector", "reid", "tracker"}
        }
        include = parse_string_list(params.get("include"))
        params["include"] = tuple(include or parse_string_list(get_mode_default(mode, "include")))
        params["weights"] = ensure_model_single(
            params.get("weights") or merged.get("reid"),
            DEFAULT_REID,
        )
        params["device_explicit"] = "device" in explicit
        params["half_explicit"] = "half" in explicit
        params["weights_explicit"] = "weights" in explicit or "reid" in explicit
        return SimpleNamespace(**params)

    params = {
        key: value
        for key, value in merged.items()
        if key not in {"detector", "reid", "tracker"}
    }
    params["tracking_method"] = merged.get("tracker", get_mode_default(mode, "tracker"))
    params["yolo_model_explicit"] = "detector" in explicit
    params["reid_model_explicit"] = "reid" in explicit
    params["device_explicit"] = "device" in explicit
    params["half_explicit"] = "half" in explicit

    if mode in {"generate", "eval", "tune"}:
        params["yolo_model"] = ensure_model_list(merged.get("detector"), DEFAULT_DETECTOR)
        params["reid_model"] = ensure_model_list(merged.get("reid"), DEFAULT_REID)
    else:
        params["yolo_model"] = ensure_model_single(merged.get("detector"), DEFAULT_DETECTOR)
        params["reid_model"] = ensure_model_single(merged.get("reid"), DEFAULT_REID)

    return SimpleNamespace(**params)


__all__ = [
    "ALIAS_KEYS",
    "DEFAULT_DETECTOR",
    "DEFAULT_REID",
    "build_mode_namespace",
    "ensure_model_extension",
    "ensure_model_list",
    "ensure_model_single",
    "get_mode_default",
    "get_mode_defaults",
    "load_runtime_defaults",
    "normalize_explicit_keys",
    "normalize_model_argument",
    "normalize_overrides",
    "parse_classes_value",
    "parse_imgsz_value",
    "parse_string_list",
]
