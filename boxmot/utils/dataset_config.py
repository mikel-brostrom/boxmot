from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from boxmot.utils import DATASET_CONFIGS, TRACKEVAL, WEIGHTS
from boxmot.utils.download import download_eval_data
from boxmot.utils.misc import resolve_model_path


def _resolve_yaml_path(config_dir: Path, name: str | Path) -> Path:
    path = Path(name)
    if path.is_absolute() and path.exists():
        return path.resolve()

    file_name = path.name if path.suffix else f"{path.name}.yaml"
    exact = (config_dir / file_name).resolve()
    if exact.exists():
        return exact

    stem = Path(file_name).stem.lower()
    matches = sorted(p.resolve() for p in config_dir.glob("*.yaml") if p.stem.lower() == stem)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Dataset config not found for '{name}' in {config_dir}")


def resolve_dataset_cfg_path(name: str | Path) -> Path:
    """Resolve a dataset config by stem or YAML filename."""
    return _resolve_yaml_path(DATASET_CONFIGS, name)


def load_dataset_cfg(name: str | Path) -> dict[str, Any]:
    """Load a dataset benchmark config YAML."""
    with open(resolve_dataset_cfg_path(name), "r") as f:
        return yaml.safe_load(f) or {}


def get_dataset_detector_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return detector settings embedded in a dataset config, if present."""
    detector_cfg = cfg.get("detector", {})
    return dict(detector_cfg) if isinstance(detector_cfg, dict) else {}


def merge_detector_cfg(base_cfg: dict[str, Any] | None, override_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay dataset-specific detector settings on top of a detector-model config."""
    merged = dict(base_cfg or {})
    if not isinstance(override_cfg, dict):
        return merged
    for key, value in override_cfg.items():
        if key == "model":
            continue
        merged[key] = value
    return merged


def resolve_required_yolo_model(cfg: dict[str, Any]) -> Path | None:
    """Return the benchmark-required detector model path, if configured."""
    detector_cfg = get_dataset_detector_cfg(cfg)
    model = detector_cfg.get("model")
    if model:
        return Path(model)

    benchmark_cfg = cfg.get("benchmark", {})
    required_yolo_model = benchmark_cfg.get("required_yolo_model")
    if required_yolo_model:
        return Path(required_yolo_model)
    return None


def should_use_dataset_detector(args: Any, cfg: dict[str, Any]) -> bool:
    """Return True when benchmark detector settings should supply the active detector."""
    dataset_model = resolve_required_yolo_model(cfg)
    if dataset_model is None:
        return False

    current_model = getattr(args, "yolo_model", None)
    if current_model is None:
        return False

    if isinstance(current_model, (list, tuple)):
        if not current_model:
            return False
        current_model = current_model[0]

    resolved_current = resolve_model_path(current_model)
    resolved_dataset = resolve_model_path(dataset_model)
    if resolved_current == resolved_dataset:
        return True
    if Path(current_model).name.lower() == Path(dataset_model).name.lower():
        return True

    if getattr(args, "yolo_model_explicit", None) is True:
        return False

    default_name = (WEIGHTS / "yolov8n.pt").name.lower()
    return Path(current_model).name.lower() == default_name


def _resolve_dataset_dest(cfg: dict[str, Any], benchmark_name: str, source_root: Path) -> Path:
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


def apply_dataset_benchmark_config(args: Any, overwrite: bool = False) -> dict[str, Any] | None:
    """Apply a benchmark YAML referenced via ``args.source`` to the current args namespace."""
    try:
        cfg = load_dataset_cfg(args.source)
    except FileNotFoundError:
        return None

    bench_cfg = cfg.get("benchmark", {})
    download_cfg = cfg.get("download", {})
    source_root = Path(bench_cfg.get("source", ""))
    benchmark_name = source_root.name or Path(resolve_dataset_cfg_path(args.source)).stem
    dataset_dest = _resolve_dataset_dest(cfg, benchmark_name, source_root)

    download_eval_data(
        runs_url=download_cfg.get("runs_url", ""),
        dataset_url=download_cfg.get("dataset_url", ""),
        dataset_dest=dataset_dest,
        overwrite=overwrite,
    )

    args.benchmark = benchmark_name
    args.split = bench_cfg.get("split", "train")
    args.source = source_root / args.split if source_root else dataset_dest / args.split

    box_type = bench_cfg.get("box_type")
    if box_type:
        args.eval_box_type = str(box_type).lower()

    detector_cfg = get_dataset_detector_cfg(cfg)
    if detector_cfg:
        args.dataset_detector_cfg = detector_cfg

    required_yolo_model = resolve_required_yolo_model(cfg)
    if required_yolo_model:
        args.required_yolo_model = required_yolo_model

    return cfg
