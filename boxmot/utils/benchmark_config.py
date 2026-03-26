from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from boxmot.utils import (
    BENCHMARK_CONFIGS,
    DATASET_CONFIGS,
    DETECTOR_CONFIGS,
    REID_CONFIGS,
    TRACKEVAL,
    WEIGHTS,
)
from boxmot.utils.download import download_eval_data, download_file
from boxmot.utils.misc import resolve_model_path


def _resolve_yaml_path(config_dir: Path, name: str | Path) -> Path:
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"} and path.exists():
        return path.resolve()

    file_name = path.name if path.suffix else f"{path.name}.yaml"
    # Exact path (including subdirectory, e.g. "ultralytics/default.yaml")
    exact = (config_dir / file_name).resolve()
    if exact.exists():
        return exact

    # Recursive search by stem
    stem = Path(file_name).stem.lower()
    matches = sorted(p.resolve() for p in config_dir.glob("**/*.yaml") if p.stem.lower() == stem)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Config not found for '{name}' in {config_dir}")


def _load_yaml_cfg(cfg_path: Path) -> dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_dataset_cfg_path(name: str | Path) -> Path:
    """Resolve a dataset config by stem or YAML filename."""
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"} and path.exists():
        return path.resolve()
    return _resolve_yaml_path(DATASET_CONFIGS, name)


def resolve_benchmark_cfg_path(name: str | Path) -> Path:
    """Resolve a benchmark config by stem or YAML filename."""
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"} and path.exists():
        return path.resolve()
    return _resolve_yaml_path(BENCHMARK_CONFIGS, name)


def resolve_detector_cfg_path(name: str | Path) -> Path:
    """Resolve a detector config by stem or YAML filename."""
    return _resolve_yaml_path(DETECTOR_CONFIGS, name)


def resolve_reid_cfg_path(name: str | Path) -> Path:
    """Resolve a ReID config by stem or YAML filename."""
    return _resolve_yaml_path(REID_CONFIGS, name)


def _normalize_dataset_download(cfg: dict[str, Any]) -> dict[str, Any]:
    download_cfg = dict(cfg.get("download") or {})
    dataset_url = download_cfg.get("dataset") or ""
    runs_url = download_cfg.get("runs") or ""
    normalized = {
        "dataset": str(dataset_url) if dataset_url else "",
        "runs": str(runs_url) if runs_url else "",
    }
    dataset_dest = download_cfg.get("dataset_dest")
    if dataset_dest:
        normalized["dataset_dest"] = dataset_dest
    return normalized


def _trackeval_adapter_for_box_type(box_type: str) -> str:
    """Map the configured box type to the TrackEval adapter used at runtime."""
    normalized = str(box_type or "aabb").lower()
    return "mmot_rgb" if normalized == "obb" else "mot_challenge"


def _normalize_benchmark_cfg(raw_cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    """Normalize dataset-like configs while preserving the current runtime accessors."""
    cfg = dict(raw_cfg or {})
    cfg.setdefault("id", cfg_path.stem.lower())

    path_value = cfg.get("path") or ""
    split_name = str(cfg.get("split") or "")

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
        split_paths[split_name] = split_name

    box_type = cfg.get("box_type") or "aabb"
    layout = cfg.get("layout") or "mot"
    trackeval_name = _trackeval_adapter_for_box_type(str(box_type).lower())

    names = dict(cfg.get("names") or {})
    distractors = dict(cfg.get("distractors") or {})
    class_map = dict(cfg.get("class_map") or {})

    download_cfg = _normalize_dataset_download(cfg)
    detector_ref = cfg.get("detector")
    reid_ref = cfg.get("reid")

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
        "detector_config": str(detector_ref) if detector_ref else None,
        "reid_config": str(reid_ref) if reid_ref else None,
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


def _merge_download_cfg(base_download: dict[str, Any], overlay_download: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_download or {})
    if overlay_download.get("dataset"):
        merged["dataset"] = overlay_download["dataset"]
    if overlay_download.get("runs"):
        merged["runs"] = overlay_download["runs"]
    if overlay_download.get("dataset_dest"):
        merged["dataset_dest"] = overlay_download["dataset_dest"]
    return merged


def _merge_benchmark_bundle_cfg(
    raw_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    cfg_path: Path,
) -> dict[str, Any]:
    """Merge a thin benchmark bundle onto its referenced dataset config."""
    cfg = dict(raw_cfg or {})
    payload = dict(dataset_cfg)

    payload["id"] = str(cfg.get("id") or cfg_path.stem.lower())
    dataset_ref = cfg.get("dataset") or dataset_cfg.get("id")
    if dataset_ref:
        payload["dataset_config"] = str(dataset_ref)

    detector_ref = cfg.get("detector") or dataset_cfg.get("detector_config")
    reid_ref = cfg.get("reid") or dataset_cfg.get("reid_config")
    if detector_ref:
        payload["detector"] = str(detector_ref)
    if reid_ref:
        payload["reid"] = str(reid_ref)

    payload["download"] = _merge_download_cfg(
        dict(dataset_cfg.get("download") or {}),
        _normalize_dataset_download(cfg),
    )

    normalized = _normalize_benchmark_cfg(payload, cfg_path)
    normalized["dataset_config"] = str(dataset_ref) if dataset_ref else None
    return normalized


def _normalize_component_cfg(raw_cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    """Normalize standalone detector/ReID component configs."""
    cfg = dict(raw_cfg or {})
    cfg.setdefault("id", cfg_path.stem.lower())
    if "default_model" not in cfg and "model" in cfg:
        cfg["default_model"] = cfg["model"]
    if "model" not in cfg and "default_model" in cfg:
        cfg["model"] = cfg["default_model"]
    if "model_url" not in cfg and "url" in cfg:
        cfg["model_url"] = cfg["url"]
    if "url" not in cfg and "model_url" in cfg:
        cfg["url"] = cfg["model_url"]
    return cfg


def load_dataset_cfg(name: str | Path) -> dict[str, Any]:
    """Load a dataset config YAML."""
    cfg_path = resolve_dataset_cfg_path(name)
    raw_cfg = _load_yaml_cfg(cfg_path)
    return _normalize_benchmark_cfg(raw_cfg, cfg_path)


def load_benchmark_only_cfg(name: str | Path) -> dict[str, Any]:
    """Load a benchmark bundle and merge in its referenced dataset config."""
    cfg_path = resolve_benchmark_cfg_path(name)
    raw_cfg = _load_yaml_cfg(cfg_path)
    dataset_ref = raw_cfg.get("dataset")
    if not dataset_ref:
        return _normalize_benchmark_cfg(raw_cfg, cfg_path)
    dataset_cfg = load_dataset_cfg(dataset_ref)
    return _merge_benchmark_bundle_cfg(raw_cfg, dataset_cfg, cfg_path)


def load_detector_component_cfg(name: str | Path) -> dict[str, Any]:
    """Load a detector component config YAML."""
    cfg_path = resolve_detector_cfg_path(name)
    raw_cfg = _load_yaml_cfg(cfg_path)
    return _normalize_component_cfg(raw_cfg, cfg_path)


def load_reid_component_cfg(name: str | Path) -> dict[str, Any]:
    """Load a ReID component config YAML."""
    cfg_path = resolve_reid_cfg_path(name)
    raw_cfg = _load_yaml_cfg(cfg_path)
    return _normalize_component_cfg(raw_cfg, cfg_path)


def load_runtime_reid_component_cfg(name: str | Path | None) -> dict[str, Any]:
    """Load a ReID component config by model/config reference, returning ``{}`` when unmatched."""
    if name in (None, ""):
        return {}
    try:
        return load_reid_component_cfg(name)
    except FileNotFoundError:
        return {}


def _combine_benchmark_and_component_cfg(
    benchmark_cfg: dict[str, Any],
    detector_cfg: dict[str, Any] | None = None,
    reid_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(benchmark_cfg)
    cfg["detector"] = dict(detector_cfg or {})
    cfg["reid"] = dict(reid_cfg or {})
    if detector_cfg and detector_cfg.get("id"):
        cfg["detector_config_id"] = detector_cfg["id"]
    if reid_cfg and reid_cfg.get("id"):
        cfg["reid_config_id"] = reid_cfg["id"]
    return cfg


def load_benchmark_cfg(name: str | Path) -> dict[str, Any]:
    """Load a benchmark config and merge in its detector/ReID defaults."""
    benchmark_cfg = load_benchmark_only_cfg(name)

    detector_ref = benchmark_cfg.get("detector_config")
    reid_ref = benchmark_cfg.get("reid_config")
    if detector_ref or reid_ref:
        detector_cfg = load_detector_component_cfg(detector_ref) if detector_ref else {}
        reid_cfg = load_reid_component_cfg(reid_ref) if reid_ref else {}
        return _combine_benchmark_and_component_cfg(benchmark_cfg, detector_cfg=detector_cfg, reid_cfg=reid_cfg)

    return _combine_benchmark_and_component_cfg(benchmark_cfg, {}, {})


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


def resolve_required_reid_device(cfg: dict[str, Any]) -> str | None:
    """Return the ReID device configured for the active dataset/model bundle."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    device = reid_cfg.get("device")
    if device is None:
        return None
    device = str(device).strip()
    return device or None


def resolve_required_reid_half(cfg: dict[str, Any]) -> bool | None:
    """Return the ReID half-precision preference configured for the active dataset/model bundle."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    if "half" not in reid_cfg:
        return None
    return bool(reid_cfg["half"])


def apply_reid_runtime_defaults(args: Any, cfg: dict[str, Any], use_config: bool = True) -> None:
    """Populate ``args.reid_device`` and ``args.reid_half`` from config when CLI did not override them."""
    fallback_device = getattr(args, "device", "")
    fallback_half = bool(getattr(args, "half", False))

    reid_device = fallback_device
    if use_config and not getattr(args, "device_explicit", False):
        configured_device = resolve_required_reid_device(cfg)
        if configured_device is not None:
            reid_device = configured_device

    reid_half = fallback_half
    if use_config and not getattr(args, "half_explicit", False):
        configured_half = resolve_required_reid_half(cfg)
        if configured_half is not None:
            reid_half = configured_half

    args.reid_device = reid_device
    args.reid_half = reid_half


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
    if (
        Path(current_model).stem.lower().replace("-", "").replace("_", "")
        == Path(benchmark_model).stem.lower().replace("-", "").replace("_", "")
    ):
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

    dataset_url = download_cfg.get("dataset") or ""
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
    return str(cfg.get("id") or cfg_path.stem)


def _normalize_path_match_key(path_like: str | Path) -> str:
    return Path(str(path_like)).as_posix().lower().rstrip("/")


def _resolve_active_split_path(cfg: dict[str, Any]) -> str:
    split = str(cfg.get("split") or "train")
    split_path = cfg.get(split)
    if split_path is None:
        split_path = split
    return str(split_path)


def _apply_benchmark_config_ref(args: Any, benchmark_ref: str | Path | None, overwrite: bool = False) -> dict[str, Any] | None:
    """Apply a benchmark YAML referenced by ``benchmark_ref`` to the current args namespace."""
    if not benchmark_ref:
        return None

    try:
        cfg_path = resolve_benchmark_cfg_path(benchmark_ref)
    except FileNotFoundError:
        return None

    cfg = load_benchmark_cfg(cfg_path)

    download_cfg = dict(cfg.get("download") or {})
    path_str = str(cfg.get("path") or "")
    source_root = Path(path_str) if path_str else None
    benchmark_name = _resolve_runtime_benchmark_name(cfg, source_root, cfg_path)
    benchmark_dest = _resolve_benchmark_dest(cfg, benchmark_name, source_root)
    split_path = _resolve_active_split_path(cfg)

    runs_check_path = Path("runs") / "dets_n_embs" / benchmark_name
    download_eval_data(
        runs_url=download_cfg.get("runs", ""),
        dataset_url=download_cfg.get("dataset", ""),
        dataset_dest=benchmark_dest,
        overwrite=overwrite,
        runs_check_path=runs_check_path,
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

    return cfg


def find_dataset_cfg_for_source(source: str | Path | None) -> dict[str, Any] | None:
    """Return the dataset config whose configured root best matches ``source``."""
    if not source:
        return None

    source_key = _normalize_path_match_key(source)
    best_match = None
    best_len = -1

    for cfg_path in sorted(DATASET_CONFIGS.glob("*.yaml")):
        try:
            cfg = load_dataset_cfg(cfg_path)
        except Exception:
            continue

        root = cfg.get("path") or ""
        if not root:
            continue

        root_key = _normalize_path_match_key(root)
        if source_key == root_key or source_key.startswith(root_key + "/"):
            if len(root_key) > best_len:
                best_match = cfg
                best_len = len(root_key)

    return best_match


def ensure_dataset_source_available(args: Any, overwrite: bool = False) -> dict[str, Any] | None:
    """Download a configured dataset when ``args.source`` targets a missing dataset path."""
    source = getattr(args, "source", None)
    if not source:
        return None

    source_path = Path(source)
    if source_path.exists():
        return None

    cfg = find_dataset_cfg_for_source(source)
    if cfg is None:
        return None

    download_cfg = dict(cfg.get("download") or {})
    source_root = Path(str(cfg.get("path") or "")) if cfg.get("path") else None
    dataset_name = str(cfg.get("id") or (source_root.name if source_root is not None else "dataset"))
    dataset_dest = _resolve_benchmark_dest(cfg, dataset_name, source_root)

    download_eval_data(
        runs_url=download_cfg.get("runs", ""),
        dataset_url=download_cfg.get("dataset", ""),
        dataset_dest=dataset_dest,
        overwrite=overwrite,
        runs_check_path=None,
    )

    args.dataset_id = cfg.get("id", dataset_name)
    box_type = cfg.get("box_type")
    if box_type:
        args.eval_box_type = str(box_type).lower()

    return cfg


def apply_benchmark_config(args: Any, overwrite: bool = False) -> dict[str, Any] | None:
    """Apply a benchmark YAML referenced via ``args.data`` to the current args namespace."""
    return _apply_benchmark_config_ref(args, getattr(args, "data", None), overwrite=overwrite)


__all__ = [
    "apply_benchmark_config",
    "ensure_dataset_source_available",
    "ensure_benchmark_detector_model",
    "ensure_benchmark_reid_model",
    "find_dataset_cfg_for_source",
    "apply_reid_runtime_defaults",
    "get_benchmark_detector_cfg",
    "get_benchmark_detector_url",
    "get_benchmark_reid_cfg",
    "get_benchmark_reid_url",
    "load_benchmark_only_cfg",
    "load_benchmark_cfg",
    "load_detector_component_cfg",
    "load_dataset_cfg",
    "load_reid_component_cfg",
    "load_runtime_reid_component_cfg",
    "resolve_benchmark_cfg_path",
    "resolve_detector_cfg_path",
    "resolve_dataset_cfg_path",
    "resolve_reid_cfg_path",
    "resolve_required_reid_device",
    "resolve_required_reid_half",
    "resolve_required_reid_model",
    "resolve_required_yolo_model",
    "should_use_benchmark_detector",
    "should_use_benchmark_reid",
]
