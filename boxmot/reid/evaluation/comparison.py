"""Reusable ReID checkpoint comparison workflow.

Builds a model-by-target dataset evaluation matrix and delegates each pair to
the canonical single-checkpoint evaluator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from boxmot.reid.evaluation import runner
from boxmot.utils import logger as LOGGER


@dataclass(frozen=True, slots=True)
class EvalTarget:
    """Evaluation dataset and root path."""

    dataset: str
    data_dir: Path


@dataclass(frozen=True, slots=True)
class ReIDComparisonConfig:
    """Resolved inputs for a checkpoint-by-dataset ReID comparison."""

    weights: tuple[str | Path, ...]
    targets: tuple[EvalTarget, ...]
    labels: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    include_same_dataset: bool = False
    preprocess: str | None = None
    imgsz: int | tuple[int, int] | list[int] | None = None
    inference_feature: str | None = None
    flip_tta: bool | None = None
    device: str = "cpu"
    batch_size: int = 64
    num_workers: int = 4
    latency_warmup: int = 5
    latency_iters: int = 30
    continue_on_error: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        LOGGER.warning(f"Could not parse {path}; falling back to checkpoint metadata")
        return {}


def _nested_value(mapping: dict[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _normalize_dataset_name(name: str | None) -> str | None:
    """Normalize dataset aliases to the same keys used by the dataset registry."""
    if not name:
        return None
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key in {"dukemtmcreid", "dukemtmc", "duke"}:
        return "duke"
    if key in {"mot171501", "mot17market1501"}:
        return "mot171501"
    if key in {"veri776", "veri"}:
        return "veri"
    if key in {"cuhk03", "cuhk03np"}:
        return "cuhk03"
    if key == "msmt17merged":
        return "msmt17merged"
    return key


def _dataset_keys(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_names = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_names = [str(part).strip() for part in value]
    else:
        raw_names = [str(value).strip()]
    return {key for name in raw_names if (key := _normalize_dataset_name(name))}


def _trained_dataset_keys(weights_path: Path) -> set[str]:
    """Return normalized training dataset keys from hparams/checkpoint metadata."""
    hparams = _read_json(weights_path.parent / "hparams.json")
    dataset = (
        _nested_value(hparams, "data", "dataset")
        or hparams.get("dataset")
        or _nested_value(hparams, "run", "dataset")
    )
    if dataset:
        return _dataset_keys(dataset)

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        LOGGER.warning(f"Could not inspect {weights_path} for training dataset: {exc}")
        return set()
    if isinstance(checkpoint, dict):
        return _dataset_keys(checkpoint.get("dataset"))
    return set()


def _default_label(weights_path: Path) -> str:
    if weights_path.name in {"best.pt", "last.pt"} and weights_path.parent.name:
        return weights_path.parent.name
    return weights_path.stem


def _safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return safe.strip("._") or "model"


def _resolve_labels(weights: tuple[Path, ...], labels: tuple[str, ...]) -> tuple[str, ...]:
    if labels and len(labels) != len(weights):
        raise ValueError(f"Expected either 0 or {len(weights)} --label values, got {len(labels)}.")

    base_labels = list(labels) if labels else [_default_label(path) for path in weights]
    counts: dict[str, int] = {}
    resolved: list[str] = []
    for label in base_labels:
        safe = _safe_label(label)
        counts[safe] = counts.get(safe, 0) + 1
        suffix = f"_{counts[safe]}" if counts[safe] > 1 else ""
        resolved.append(f"{safe}{suffix}")
    return tuple(resolved)


def _resolve_models(models: tuple[str, ...], count: int) -> tuple[str | None, ...]:
    if not models:
        return tuple(None for _ in range(count))
    if len(models) == 1:
        return tuple(models[0] for _ in range(count))
    if len(models) != count:
        raise ValueError(f"Expected either 0, 1, or {count} --model values, got {len(models)}.")
    return tuple(models)


def _result_row(
    *,
    label: str,
    weights: Path,
    train_datasets: set[str],
    target: EvalTarget,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    eval_key = _normalize_dataset_name(target.dataset)
    row = {
        "label": label,
        "weights": str(weights),
        "train_dataset": ",".join(sorted(train_datasets)) if train_datasets else None,
        "eval_dataset": target.dataset,
        "eval_dataset_key": eval_key,
        "data_dir": str(target.data_dir),
        "cross_domain": None if not train_datasets else eval_key not in train_datasets,
        "status": status,
    }
    if result:
        row.update(result)
        row["weights"] = str(row["weights"])
    if error:
        row["error"] = error
    return row


def _evaluation_config(
    config: ReIDComparisonConfig,
    *,
    weights: Path,
    model: str | None,
    target: EvalTarget,
) -> runner.ReIDEvaluationConfig:
    return runner.ReIDEvaluationConfig(
        weights=weights,
        model=model,
        dataset=target.dataset,
        data_dir=target.data_dir,
        preprocess=config.preprocess,
        imgsz=config.imgsz,
        inference_feature=config.inference_feature,
        flip_tta=config.flip_tta,
        device=config.device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        latency_warmup=config.latency_warmup,
        latency_iters=config.latency_iters,
    )


def compare_reid(config: ReIDComparisonConfig) -> dict[str, Any]:
    """Compare an arbitrary set of ReID checkpoints across target datasets."""
    weights = tuple(Path(path).expanduser() for path in config.weights)
    targets = tuple(config.targets)
    if not weights:
        raise ValueError("compare-reid requires at least one --weights checkpoint.")
    if not targets:
        raise ValueError("compare-reid requires at least one --target DATASET=DATA_DIR.")

    labels = _resolve_labels(
        weights,
        tuple(str(label) for label in config.labels),
    )
    models = _resolve_models(
        tuple(str(model) for model in config.models),
        len(weights),
    )
    include_same_dataset = bool(config.include_same_dataset)
    continue_on_error = bool(config.continue_on_error)
    results: list[dict[str, Any]] = []
    for weights_path in weights:
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {weights_path}")
    trained_by_model = {weights_path: _trained_dataset_keys(weights_path) for weights_path in weights}

    for weights_path, label, model_name in zip(weights, labels, models):
        train_datasets = trained_by_model[weights_path]
        for target in targets:
            eval_key = _normalize_dataset_name(target.dataset)
            if train_datasets and eval_key in train_datasets and not include_same_dataset:
                results.append(
                    _result_row(
                        label=label,
                        weights=weights_path,
                        train_datasets=train_datasets,
                        target=target,
                        status="skipped_same_dataset",
                    )
                )
                continue

            evaluation_config = _evaluation_config(
                config,
                weights=weights_path,
                model=model_name,
                target=target,
            )
            try:
                result = runner.evaluate_reid(evaluation_config)
            except Exception as exc:
                if not continue_on_error:
                    raise
                results.append(
                    _result_row(
                        label=label,
                        weights=weights_path,
                        train_datasets=train_datasets,
                        target=target,
                        status="failed",
                        error=str(exc),
                    )
                )
                continue

            results.append(
                _result_row(
                    label=label,
                    weights=weights_path,
                    train_datasets=train_datasets,
                    target=target,
                    status="ok",
                    result=result,
                )
            )

    summary = {
        "models": len(weights),
        "targets": len(targets),
        "rows": len(results),
        "evaluated": sum(row["status"] == "ok" for row in results),
        "skipped": sum(row["status"] == "skipped_same_dataset" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "cross_domain_only": not include_same_dataset,
    }
    return {
        "summary": summary,
        "results": results,
    }


__all__ = ("EvalTarget", "ReIDComparisonConfig", "compare_reid")
