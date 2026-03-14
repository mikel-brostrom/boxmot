from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from boxmot.utils import DATASET_CONFIGS, TRACKEVAL
from boxmot.utils.download import download_eval_data


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


def _resolve_dataset_dest(cfg: dict[str, Any], benchmark_name: str) -> Path:
    download_cfg = cfg.get("download", {})
    dataset_dest = download_cfg.get("dataset_dest")
    if dataset_dest:
        return Path(dataset_dest)

    dataset_url = download_cfg.get("dataset_url", "") or ""
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
    dataset_dest = _resolve_dataset_dest(cfg, benchmark_name)

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

    required_yolo_model = bench_cfg.get("required_yolo_model")
    if required_yolo_model:
        args.required_yolo_model = Path(required_yolo_model)

    return cfg
