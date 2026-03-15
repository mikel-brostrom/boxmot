from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from boxmot.utils import BENCHMARK_CONFIGS, TRACKEVAL, WEIGHTS
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

    raise FileNotFoundError(f"Benchmark config not found for '{name}' in {config_dir}")


def resolve_benchmark_cfg_path(name: str | Path) -> Path:
    """Resolve a benchmark config by stem or YAML filename."""
    return _resolve_yaml_path(BENCHMARK_CONFIGS, name)


def _normalize_benchmark_cfg(raw_cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    """Normalize benchmark configs to the new schema while preserving legacy accessors."""
    cfg = dict(raw_cfg or {})
    cfg.setdefault("id", cfg_path.stem.lower())

    legacy_benchmark = dict(cfg.get("benchmark") or {})
    storage_cfg = dict(cfg.get("storage") or {})
    evaluation_cfg = dict(cfg.get("evaluation") or {})
    class_cfg = dict(evaluation_cfg.get("classes") or {})

    if not storage_cfg and legacy_benchmark:
        storage_cfg = {
            "root": legacy_benchmark.get("source", ""),
            "split": legacy_benchmark.get("split", "train"),
        }

    if not evaluation_cfg and legacy_benchmark:
        evaluation_cfg = {
            "box_type": legacy_benchmark.get("box_type", "aabb"),
            "layout": legacy_benchmark.get("layout", "mot"),
            "tracker_eval": legacy_benchmark.get("tracker_eval", "mot_challenge"),
            "classes": {
                "eval": legacy_benchmark.get("eval_classes", {}),
                "distractor": legacy_benchmark.get("distractor_classes", {}),
                "mapping": legacy_benchmark.get("class_mapping", {}),
            },
        }
        class_cfg = dict(evaluation_cfg.get("classes") or {})

    if storage_cfg:
        storage_cfg.setdefault("root", legacy_benchmark.get("source", ""))
        storage_cfg.setdefault("split", legacy_benchmark.get("split", "train"))

    if evaluation_cfg:
        evaluation_cfg.setdefault("box_type", legacy_benchmark.get("box_type", "aabb"))
        evaluation_cfg.setdefault("layout", legacy_benchmark.get("layout", "mot"))
        evaluation_cfg.setdefault("tracker_eval", legacy_benchmark.get("tracker_eval", "mot_challenge"))
        class_cfg = dict(evaluation_cfg.get("classes") or {})
        class_cfg.setdefault("eval", legacy_benchmark.get("eval_classes", {}))
        class_cfg.setdefault("distractor", legacy_benchmark.get("distractor_classes", {}))
        class_cfg.setdefault("mapping", legacy_benchmark.get("class_mapping", {}))
        evaluation_cfg["classes"] = class_cfg

    if storage_cfg:
        cfg["storage"] = storage_cfg
    if evaluation_cfg:
        cfg["evaluation"] = evaluation_cfg

    detector_cfg = dict(cfg.get("detector") or {})
    if "default_model" not in detector_cfg and "model" in detector_cfg:
        detector_cfg["default_model"] = detector_cfg["model"]
    if "model" not in detector_cfg and "default_model" in detector_cfg:
        detector_cfg["model"] = detector_cfg["default_model"]
    if not detector_cfg and legacy_benchmark.get("required_yolo_model"):
        detector_cfg = {
            "default_model": legacy_benchmark["required_yolo_model"],
            "model": legacy_benchmark["required_yolo_model"],
        }
    if detector_cfg:
        cfg["detector"] = detector_cfg

    if storage_cfg or evaluation_cfg:
        cfg["benchmark"] = {
            "source": storage_cfg.get("root", legacy_benchmark.get("source", "")),
            "split": storage_cfg.get("split", legacy_benchmark.get("split", "train")),
            "box_type": evaluation_cfg.get("box_type", legacy_benchmark.get("box_type", "aabb")),
            "layout": evaluation_cfg.get("layout", legacy_benchmark.get("layout", "mot")),
            "tracker_eval": evaluation_cfg.get("tracker_eval", legacy_benchmark.get("tracker_eval", "mot_challenge")),
            "eval_classes": class_cfg.get("eval", legacy_benchmark.get("eval_classes", {})),
            "distractor_classes": class_cfg.get("distractor", legacy_benchmark.get("distractor_classes", {})),
            "class_mapping": class_cfg.get("mapping", legacy_benchmark.get("class_mapping", {})),
        }

    return cfg


def load_benchmark_cfg(name: str | Path) -> dict[str, Any]:
    """Load a benchmark config YAML."""
    cfg_path = resolve_benchmark_cfg_path(name)
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    return _normalize_benchmark_cfg(raw_cfg, cfg_path)


def get_benchmark_detector_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return detector settings embedded in a benchmark config, if present."""
    detector_cfg = cfg.get("detector", {})
    return dict(detector_cfg) if isinstance(detector_cfg, dict) else {}


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
    """Return the benchmark detector download URL, if configured."""
    detector_cfg = get_benchmark_detector_cfg(cfg)
    model_url = detector_cfg.get("model_url") or detector_cfg.get("url")
    return _normalize_google_drive_url(str(model_url)) if model_url else None


def resolve_required_yolo_model(cfg: dict[str, Any]) -> Path | None:
    """Return the benchmark-required detector model path, if configured."""
    detector_cfg = get_benchmark_detector_cfg(cfg)
    model = detector_cfg.get("default_model") or detector_cfg.get("model")
    if model:
        return Path(model)

    benchmark_cfg = cfg.get("benchmark", {})
    required_yolo_model = benchmark_cfg.get("required_yolo_model")
    if required_yolo_model:
        return Path(required_yolo_model)
    return None


def ensure_benchmark_detector_model(cfg: dict[str, Any], overwrite: bool = False) -> Path | None:
    """Ensure the benchmark-default detector model exists locally and return its path."""
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


def should_use_benchmark_detector(args: Any, cfg: dict[str, Any]) -> bool:
    """Return True when benchmark detector settings should supply the active detector."""
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


def _resolve_benchmark_dest(cfg: dict[str, Any], benchmark_name: str, source_root: Path) -> Path:
    download_cfg = cfg.get("download", {})
    dataset_dest = download_cfg.get("dataset_dest")
    if dataset_dest:
        return Path(dataset_dest)

    dataset_url = download_cfg.get("dataset_url", "") or ""
    if source_root:
        if dataset_url.startswith("hf://"):
            return source_root
        if dataset_url:
            return source_root.parent / f"{source_root.name}.zip"
        return source_root

    if dataset_url.startswith("hf://"):
        return TRACKEVAL / "data" / benchmark_name
    if dataset_url:
        return TRACKEVAL / "data" / f"{benchmark_name}.zip"
    return Path(f"assets/{benchmark_name}")


def _resolve_runtime_benchmark_name(cfg: dict[str, Any], source_root: Path, cfg_path: Path) -> str:
    """Resolve a stable runtime benchmark name for cache and results paths."""
    benchmark_id = str(cfg.get("id") or cfg_path.stem)
    root_name = source_root.name
    if root_name and root_name.lower() == benchmark_id.lower():
        return root_name
    return benchmark_id


def _apply_benchmark_config_ref(args: Any, benchmark_ref: str | Path | None, overwrite: bool = False) -> dict[str, Any] | None:
    """Apply a benchmark YAML referenced by ``benchmark_ref`` to the current args namespace."""
    if not benchmark_ref:
        return None

    try:
        cfg_path = resolve_benchmark_cfg_path(benchmark_ref)
        cfg = load_benchmark_cfg(cfg_path)
    except FileNotFoundError:
        return None

    storage_cfg = cfg.get("storage", {})
    eval_cfg = cfg.get("evaluation", {})
    download_cfg = cfg.get("download", {})
    source_root = Path(storage_cfg.get("root", ""))
    benchmark_name = _resolve_runtime_benchmark_name(cfg, source_root, cfg_path)
    benchmark_dest = _resolve_benchmark_dest(cfg, benchmark_name, source_root)

    download_eval_data(
        runs_url=download_cfg.get("runs_url", ""),
        dataset_url=download_cfg.get("dataset_url", ""),
        dataset_dest=benchmark_dest,
        overwrite=overwrite,
    )

    args.benchmark_id = cfg.get("id", benchmark_name)
    # Backward-compatible alias for existing callers.
    args.dataset_id = args.benchmark_id
    args.benchmark = benchmark_name
    args.split = storage_cfg.get("split", "train")
    args.source = source_root / args.split if source_root else benchmark_dest / args.split

    box_type = eval_cfg.get("box_type")
    if box_type:
        args.eval_box_type = str(box_type).lower()

    detector_cfg = get_benchmark_detector_cfg(cfg)
    if detector_cfg:
        args.dataset_detector_cfg = detector_cfg

    required_yolo_model = resolve_required_yolo_model(cfg)
    if required_yolo_model:
        args.required_yolo_model = required_yolo_model

    return cfg


def apply_benchmark_config(args: Any, overwrite: bool = False) -> dict[str, Any] | None:
    """Apply a benchmark YAML referenced via ``args.data`` to the current args namespace."""
    return _apply_benchmark_config_ref(args, getattr(args, "data", None), overwrite=overwrite)


# Backward-compatible aliases for external imports.
resolve_dataset_cfg_path = resolve_benchmark_cfg_path
load_dataset_cfg = load_benchmark_cfg
get_dataset_detector_cfg = get_benchmark_detector_cfg
should_use_dataset_detector = should_use_benchmark_detector
def apply_dataset_benchmark_config(args: Any, overwrite: bool = False) -> dict[str, Any] | None:
    """Backward-compatible benchmark resolver using ``args.data`` or legacy ``args.source``."""
    benchmark_ref = getattr(args, "data", None) or getattr(args, "source", None)
    return _apply_benchmark_config_ref(args, benchmark_ref, overwrite=overwrite)


__all__ = [
    "apply_benchmark_config",
    "apply_dataset_benchmark_config",
    "ensure_benchmark_detector_model",
    "get_benchmark_detector_cfg",
    "get_benchmark_detector_url",
    "get_dataset_detector_cfg",
    "load_benchmark_cfg",
    "load_dataset_cfg",
    "resolve_benchmark_cfg_path",
    "resolve_dataset_cfg_path",
    "resolve_required_yolo_model",
    "should_use_benchmark_detector",
    "should_use_dataset_detector",
]
