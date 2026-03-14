"""Backward-compatible re-exports for the benchmark config helpers."""

from boxmot.utils.benchmark_config import (
    apply_benchmark_config,
    apply_dataset_benchmark_config,
    get_benchmark_detector_cfg,
    get_dataset_detector_cfg,
    load_benchmark_cfg,
    load_dataset_cfg,
    resolve_benchmark_cfg_path,
    resolve_dataset_cfg_path,
    resolve_required_yolo_model,
    should_use_benchmark_detector,
    should_use_dataset_detector,
)

__all__ = [
    "apply_benchmark_config",
    "apply_dataset_benchmark_config",
    "get_benchmark_detector_cfg",
    "get_dataset_detector_cfg",
    "load_benchmark_cfg",
    "load_dataset_cfg",
    "resolve_benchmark_cfg_path",
    "resolve_dataset_cfg_path",
    "resolve_required_yolo_model",
    "should_use_benchmark_detector",
    "should_use_dataset_detector",
]
