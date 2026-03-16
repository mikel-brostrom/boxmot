from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from boxmot.utils import DATASET_CONFIGS, MODEL_CONFIGS, TRACKEVAL, WEIGHTS
from boxmot.utils.download import download_eval_data, download_file
from boxmot.utils.misc import resolve_model_path


def _resolve_yaml_path(config_dir: Path, name: str | Path) -> Path:
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"} and path.exists():
        return path.resolve()

    file_name = path.name if path.suffix else f"{path.name}.yaml"
    exact = (config_dir / file_name).resolve()
    if exact.exists():
        return exact

    stem = Path(file_name).stem.lower()
    matches = sorted(p.resolve() for p in config_dir.glob("*.yaml") if p.stem.lower() == stem)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Config not found for '{name}' in {config_dir}")


def resolve_dataset_cfg_path(name: str | Path) -> Path:
    """Resolve a dataset config by stem or YAML filename."""
    return _resolve_yaml_path(DATASET_CONFIGS, name)


def resolve_model_cfg_path(name: str | Path) -> Path:
    """Resolve a detector+ReID model config by stem or YAML filename."""
    return _resolve_yaml_path(MODEL_CONFIGS, name)


def resolve_benchmark_cfg_path(name: str | Path) -> Path:
    """Backward-compatible alias for dataset configs."""
    return resolve_dataset_cfg_path(name)


def _normalize_dataset_download(cfg: dict[str, Any]) -> dict[str, Any]:
    download_cfg = dict(cfg.get("download") or {})
    dataset_url = download_cfg.get("dataset") or download_cfg.get("dataset_url") or ""
    runs_url = download_cfg.get("runs") or download_cfg.get("runs_url") or ""
    normalized = {
        "dataset": str(dataset_url) if dataset_url else "",
        "runs": str(runs_url) if runs_url else "",
    }
    if download_cfg.get("dataset_dest"):
        normalized["dataset_dest"] = download_cfg["dataset_dest"]
    return normalized


def _normalize_dataset_cfg(raw_cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    """Normalize dataset configs while preserving accessors used by the current runtime."""
    cfg = dict(raw_cfg or {})
    cfg.setdefault("id", cfg_path.stem.lower())

    legacy_benchmark = dict(cfg.get("benchmark") or {})
    storage_cfg = dict(cfg.get("storage") or {})
    evaluation_cfg = dict(cfg.get("evaluation") or {})
    class_cfg = dict(evaluation_cfg.get("classes") or {})

    path_value = cfg.get("path") or storage_cfg.get("root") or legacy_benchmark.get("source") or ""
    split_name = str(
        cfg.get("split")
        or storage_cfg.get("split")
        or legacy_benchmark.get("split")
        or ""
    )

    train_value = cfg.get("train")
    val_value = cfg.get("val")
    test_value = cfg.get("test")

    if not split_name:
        if val_value:
            split_name = "val"
        elif train_value:
            split_name = "train"
        elif test_value:
            split_name = "test"
        else:
            split_name = "train"

    split_paths = {
        "train": train_value,
        "val": val_value,
        "test": test_value,
    }
    if split_paths.get(split_name) is None:
        legacy_split_path = storage_cfg.get("split") or legacy_benchmark.get("split")
        split_paths[split_name] = legacy_split_path or split_name

    layout = cfg.get("layout") or evaluation_cfg.get("layout") or legacy_benchmark.get("layout") or "mot"
    box_type = cfg.get("box_type") or evaluation_cfg.get("box_type") or legacy_benchmark.get("box_type") or "aabb"
    trackeval_name = (
        cfg.get("trackeval")
        or evaluation_cfg.get("tracker_eval")
        or legacy_benchmark.get("tracker_eval")
        or "mot_challenge"
    )

    names = dict(cfg.get("names") or class_cfg.get("eval") or legacy_benchmark.get("eval_classes") or {})
    distractors = dict(
        cfg.get("distractors") or class_cfg.get("distractor") or legacy_benchmark.get("distractor_classes") or {}
    )
    class_map = dict(cfg.get("class_map") or class_cfg.get("mapping") or legacy_benchmark.get("class_mapping") or {})

    download_cfg = _normalize_dataset_download(cfg)
    models_ref = cfg.get("models") or cfg.get("model_config")

    normalized = {
        "id": cfg["id"],
        "path": str(path_value),
        "split": split_name,
        "train": split_paths.get("train"),
        "val": split_paths.get("val"),
        "test": split_paths.get("test"),
        "layout": str(layout),
        "box_type": str(box_type).lower(),
        "trackeval": str(trackeval_name),
        "names": names,
        "distractors": distractors,
        "class_map": class_map,
        "download": download_cfg,
        "models": str(models_ref) if models_ref else None,
    }

    normalized["storage"] = {
        "root": normalized["path"],
        "split": str(split_paths.get(split_name) or split_name),
    }
    normalized["evaluation"] = {
        "box_type": normalized["box_type"],
        "layout": normalized["layout"],
        "tracker_eval": normalized["trackeval"],
        "classes": {
            "eval": names,
            "distractor": distractors,
            "mapping": class_map,
        },
    }
    normalized["benchmark"] = {
        "source": normalized["path"],
        "split": str(split_paths.get(split_name) or split_name),
        "box_type": normalized["box_type"],
        "layout": normalized["layout"],
        "tracker_eval": normalized["trackeval"],
        "eval_classes": names,
        "distractor_classes": distractors,
        "class_mapping": class_map,
    }
    return normalized


def _normalize_model_cfg(raw_cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    """Normalize detector+ReID model configs."""
    cfg = dict(raw_cfg or {})
    cfg.setdefault("id", cfg_path.stem.lower())

    detector_cfg = dict(cfg.get("detector") or {})
    if "default_model" not in detector_cfg and "model" in detector_cfg:
        detector_cfg["default_model"] = detector_cfg["model"]
    if "model" not in detector_cfg and "default_model" in detector_cfg:
        detector_cfg["model"] = detector_cfg["default_model"]
    if "model_url" not in detector_cfg and "url" in detector_cfg:
        detector_cfg["model_url"] = detector_cfg["url"]
    if "url" not in detector_cfg and "model_url" in detector_cfg:
        detector_cfg["url"] = detector_cfg["model_url"]

    reid_cfg = dict(cfg.get("reid") or {})
    if "default_model" not in reid_cfg and "model" in reid_cfg:
        reid_cfg["default_model"] = reid_cfg["model"]
    if "model" not in reid_cfg and "default_model" in reid_cfg:
        reid_cfg["model"] = reid_cfg["default_model"]
    if "model_url" not in reid_cfg and "url" in reid_cfg:
        reid_cfg["model_url"] = reid_cfg["url"]
    if "url" not in reid_cfg and "model_url" in reid_cfg:
        reid_cfg["url"] = reid_cfg["model_url"]

    return {
        "id": cfg["id"],
        "detector": detector_cfg,
        "reid": reid_cfg,
    }


def load_dataset_cfg(name: str | Path) -> dict[str, Any]:
    """Load a dataset config YAML."""
    cfg_path = resolve_dataset_cfg_path(name)
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    return _normalize_dataset_cfg(raw_cfg, cfg_path)


def load_model_cfg(name: str | Path) -> dict[str, Any]:
    """Load a detector+ReID config YAML."""
    cfg_path = resolve_model_cfg_path(name)
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    return _normalize_model_cfg(raw_cfg, cfg_path)


def _combine_dataset_and_model_cfg(
    dataset_cfg: dict[str, Any],
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(dataset_cfg)
    cfg["detector"] = dict((model_cfg or {}).get("detector") or {})
    cfg["reid"] = dict((model_cfg or {}).get("reid") or {})
    if model_cfg and model_cfg.get("id"):
        cfg["model_config_id"] = model_cfg["id"]
    return cfg


def load_benchmark_cfg(name: str | Path, models: str | Path | None = None) -> dict[str, Any]:
    """Load a dataset config and merge in its default or overridden model config."""
    dataset_cfg = load_dataset_cfg(name)
    model_ref = models or dataset_cfg.get("models")
    if not model_ref:
        return _combine_dataset_and_model_cfg(dataset_cfg, {})
    model_cfg = load_model_cfg(model_ref)
    return _combine_dataset_and_model_cfg(dataset_cfg, model_cfg)


def get_benchmark_detector_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return detector settings from a combined dataset+model config."""
    detector_cfg = cfg.get("detector", {})
    return dict(detector_cfg) if isinstance(detector_cfg, dict) else {}


def get_benchmark_reid_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return ReID settings from a combined dataset+model config."""
    reid_cfg = cfg.get("reid", {})
    return dict(reid_cfg) if isinstance(reid_cfg, dict) else {}


def _normalize_google_drive_url(url: str) -> str:
    """Normalize common Google Drive share URLs to the canonical ``uc?id=...`` form."""
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return url

    query = parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return f"https://drive.google.com/uc?id={query['id'][0]}"

    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return f"https://drive.google.com/uc?id={match.group(1)}"

    return url


def get_benchmark_detector_url(cfg: dict[str, Any]) -> str | None:
    """Return the configured detector download URL, if present."""
    detector_cfg = get_benchmark_detector_cfg(cfg)
    model_url = detector_cfg.get("model_url") or detector_cfg.get("url")
    return _normalize_google_drive_url(str(model_url)) if model_url else None


def get_benchmark_reid_url(cfg: dict[str, Any]) -> str | None:
    """Return the configured ReID download URL, if present."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    model_url = reid_cfg.get("model_url") or reid_cfg.get("url")
    return _normalize_google_drive_url(str(model_url)) if model_url else None


def resolve_required_yolo_model(cfg: dict[str, Any]) -> Path | None:
    """Return the detector model path configured for the active dataset/model bundle."""
    detector_cfg = get_benchmark_detector_cfg(cfg)
    model = detector_cfg.get("default_model") or detector_cfg.get("model")
    if model:
        return Path(model)
    return None


def resolve_required_reid_model(cfg: dict[str, Any]) -> Path | None:
    """Return the ReID model path configured for the active dataset/model bundle."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    model = reid_cfg.get("default_model") or reid_cfg.get("model")
    return Path(model) if model else None


def ensure_benchmark_detector_model(cfg: dict[str, Any], overwrite: bool = False) -> Path | None:
    """Ensure the configured detector model exists locally and return its path."""
    model_path = resolve_required_yolo_model(cfg)
    if model_path is None:
        return None

    resolved_path = resolve_model_path(model_path)
    if resolved_path.exists() or overwrite:
        if overwrite and get_benchmark_detector_url(cfg):
            download_file(get_benchmark_detector_url(cfg), resolved_path, overwrite=True)
        return resolved_path

    model_url = get_benchmark_detector_url(cfg)
    if model_url:
        download_file(model_url, resolved_path, overwrite=False)
        return resolved_path
    return resolved_path


def ensure_benchmark_reid_model(cfg: dict[str, Any], overwrite: bool = False) -> Path | None:
    """Ensure the configured ReID model exists locally and return its path."""
    model_path = resolve_required_reid_model(cfg)
    if model_path is None:
        return None

    resolved_path = resolve_model_path(model_path)
    if resolved_path.exists() or overwrite:
        if overwrite and get_benchmark_reid_url(cfg):
            download_file(get_benchmark_reid_url(cfg), resolved_path, overwrite=True)
        return resolved_path

    model_url = get_benchmark_reid_url(cfg)
    if model_url:
        download_file(model_url, resolved_path, overwrite=False)
        return resolved_path
    return resolved_path


def should_use_benchmark_detector(args: Any, cfg: dict[str, Any]) -> bool:
    """Return True when the model config should provide the active detector."""
    benchmark_model = resolve_required_yolo_model(cfg)
    if benchmark_model is None:
        return False

    current_model = getattr(args, "yolo_model", None)
    if current_model is None:
        return False

    if isinstance(current_model, (list, tuple)):
        if not current_model:
            return False
        current_model = current_model[0]

    resolved_current = resolve_model_path(current_model)
    resolved_benchmark = resolve_model_path(benchmark_model)
    if resolved_current == resolved_benchmark:
        return True
    if Path(current_model).name.lower() == Path(benchmark_model).name.lower():
        return True

    if getattr(args, "yolo_model_explicit", None) is True:
        return False

    default_name = (WEIGHTS / "yolov8n.pt").name.lower()
    return Path(current_model).name.lower() == default_name


def should_use_benchmark_reid(args: Any, cfg: dict[str, Any]) -> bool:
    """Return True when the model config should provide the active ReID model."""
    benchmark_model = resolve_required_reid_model(cfg)
    if benchmark_model is None:
        return False

    current_model = getattr(args, "reid_model", None)
    if current_model is None:
        return False

    if isinstance(current_model, (list, tuple)):
        if not current_model:
            return False
        current_model = current_model[0]

    resolved_current = resolve_model_path(current_model)
    resolved_benchmark = resolve_model_path(benchmark_model)
    if resolved_current == resolved_benchmark:
        return True
    if Path(current_model).name.lower() == Path(benchmark_model).name.lower():
        return True

    if getattr(args, "reid_model_explicit", None) is True:
        return False

    default_name = (WEIGHTS / "osnet_x0_25_msmt17.pt").name.lower()
    return Path(current_model).name.lower() == default_name


def _resolve_benchmark_dest(cfg: dict[str, Any], benchmark_name: str, source_root: Path | None) -> Path:
    download_cfg = dict(cfg.get("download") or {})
    dataset_dest = download_cfg.get("dataset_dest")
    if dataset_dest:
        return Path(dataset_dest)

    dataset_url = download_cfg.get("dataset") or download_cfg.get("dataset_url") or ""
    if source_root is not None:
        if str(dataset_url).startswith("hf://"):
            return source_root
        if dataset_url:
            return source_root.parent / f"{source_root.name}.zip"
        return source_root

    if str(dataset_url).startswith("hf://"):
        return TRACKEVAL / "data" / benchmark_name
    if dataset_url:
        return TRACKEVAL / "data" / f"{benchmark_name}.zip"
    return Path(f"assets/{benchmark_name}")


def _resolve_runtime_benchmark_name(cfg: dict[str, Any], source_root: Path | None, cfg_path: Path) -> str:
    """Resolve a stable runtime benchmark name for cache and results paths."""
    benchmark_id = str(cfg.get("id") or cfg_path.stem)
    root_name = source_root.name if source_root is not None else ""
    if root_name and root_name.lower() == benchmark_id.lower():
        return root_name
    return benchmark_id


def _resolve_active_split_path(cfg: dict[str, Any]) -> str:
    split = str(cfg.get("split") or "train")
    split_path = cfg.get(split)
    if split_path is None:
        split_path = split
    return str(split_path)


def _apply_benchmark_config_ref(args: Any, benchmark_ref: str | Path | None, overwrite: bool = False) -> dict[str, Any] | None:
    """Apply a dataset YAML referenced by ``benchmark_ref`` to the current args namespace."""
    if not benchmark_ref:
        return None

    try:
        cfg_path = resolve_dataset_cfg_path(benchmark_ref)
        dataset_cfg = load_dataset_cfg(cfg_path)
    except FileNotFoundError:
        return None

    if not getattr(args, "models", None) and dataset_cfg.get("models"):
        args.models = dataset_cfg["models"]

    cfg = load_benchmark_cfg(cfg_path, getattr(args, "models", None))

    download_cfg = dict(cfg.get("download") or {})
    path_str = str(cfg.get("path") or "")
    source_root = Path(path_str) if path_str else None
    benchmark_name = _resolve_runtime_benchmark_name(cfg, source_root, cfg_path)
    benchmark_dest = _resolve_benchmark_dest(cfg, benchmark_name, source_root)
    split_path = _resolve_active_split_path(cfg)

    download_eval_data(
        runs_url=download_cfg.get("runs", "") or download_cfg.get("runs_url", ""),
        dataset_url=download_cfg.get("dataset", "") or download_cfg.get("dataset_url", ""),
        dataset_dest=benchmark_dest,
        overwrite=overwrite,
    )

    args.benchmark_id = cfg.get("id", benchmark_name)
    args.dataset_id = args.benchmark_id
    args.benchmark = benchmark_name
    args.split = str(cfg.get("split") or "train")
    args.source = (source_root / split_path) if source_root is not None else (benchmark_dest / split_path)

    box_type = cfg.get("box_type")
    if box_type:
        args.eval_box_type = str(box_type).lower()

    detector_cfg = get_benchmark_detector_cfg(cfg)
    if detector_cfg:
        args.dataset_detector_cfg = detector_cfg

    required_yolo_model = resolve_required_yolo_model(cfg)
    if required_yolo_model:
        args.required_yolo_model = required_yolo_model

    required_reid_model = resolve_required_reid_model(cfg)
    if required_reid_model:
        args.required_reid_model = required_reid_model

    if getattr(args, "models", None):
        args.model_config_id = getattr(args, "models")

    return cfg


def apply_benchmark_config(args: Any, overwrite: bool = False) -> dict[str, Any] | None:
    """Apply a dataset YAML referenced via ``args.data`` to the current args namespace."""
    return _apply_benchmark_config_ref(args, getattr(args, "data", None), overwrite=overwrite)


# Backward-compatible aliases for external imports.
def apply_dataset_benchmark_config(args: Any, overwrite: bool = False) -> dict[str, Any] | None:
    """Backward-compatible dataset resolver using ``args.data`` or legacy ``args.source``."""
    dataset_ref = getattr(args, "data", None) or getattr(args, "source", None)
    return _apply_benchmark_config_ref(args, dataset_ref, overwrite=overwrite)


get_dataset_detector_cfg = get_benchmark_detector_cfg
should_use_dataset_detector = should_use_benchmark_detector


__all__ = [
    "apply_benchmark_config",
    "apply_dataset_benchmark_config",
    "ensure_benchmark_detector_model",
    "ensure_benchmark_reid_model",
    "get_benchmark_detector_cfg",
    "get_benchmark_detector_url",
    "get_benchmark_reid_cfg",
    "get_benchmark_reid_url",
    "get_dataset_detector_cfg",
    "load_benchmark_cfg",
    "load_dataset_cfg",
    "load_model_cfg",
    "resolve_benchmark_cfg_path",
    "resolve_dataset_cfg_path",
    "resolve_model_cfg_path",
    "resolve_required_reid_model",
    "resolve_required_yolo_model",
    "should_use_benchmark_detector",
    "should_use_benchmark_reid",
    "should_use_dataset_detector",
]
