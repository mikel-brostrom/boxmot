"""ReID checkpoint comparison workflow.

Invoked by the CLI ``compare-reid`` subcommand via ``main(args)``.
Builds a model-by-target dataset evaluation matrix and delegates each pair to
the canonical single-checkpoint evaluator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from boxmot.engine.reid import evaluator
from boxmot.utils import logger as LOGGER


@dataclass(frozen=True, slots=True)
class EvalTarget:
    """Evaluation dataset and root path."""

    dataset: str
    data_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        LOGGER.warning(f"Could not parse {path}; falling back to checkpoint metadata")
        return {}


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (value,)
    return tuple(value)


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


def _parse_target(raw: str) -> EvalTarget:
    """Parse ``DATASET=DATA_DIR`` target specifications."""
    if "=" in raw:
        dataset, data_dir = raw.split("=", 1)
    elif ":" in raw:
        dataset, data_dir = raw.split(":", 1)
    else:
        raise ValueError(f"Invalid target '{raw}'. Expected DATASET=DATA_DIR.")

    dataset = dataset.strip()
    data_path = Path(data_dir.strip()).expanduser()
    if not dataset:
        raise ValueError(f"Invalid target '{raw}': dataset name is empty.")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset root for target '{dataset}' does not exist: {data_path}")
    return EvalTarget(dataset=dataset, data_dir=data_path)


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
    if error:
        row["error"] = error
    return row


def _markdown_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "Model",
        "Train",
        "Eval",
        "mAP",
        "Rank-1",
        "Latency ms/img",
        "Device",
        "Feature",
        "Status",
    ]
    lines = [
        "# ReID Model Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_value(row.get("label")),
                    _markdown_value(row.get("train_dataset")),
                    _markdown_value(row.get("eval_dataset")),
                    _markdown_value(row.get("mAP")),
                    _markdown_value(row.get("rank1")),
                    _markdown_value(row.get("latency_ms_per_image")),
                    _markdown_value(row.get("latency_device")),
                    _markdown_value(row.get("inference_feature")),
                    _markdown_value(row.get("status")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_map_latency_plot(results: list[dict[str, Any]], path: Path) -> Path | None:
    """Write mAP-vs-latency scatter plot for successfully evaluated rows."""
    rows = [
        row
        for row in results
        if row.get("status") == "ok"
        and row.get("mAP") is not None
        and row.get("latency_ms_per_image") is not None
    ]
    if not rows:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except Exception as exc:
        LOGGER.warning(f"Could not create mAP/latency plot because matplotlib is unavailable: {exc}")
        return None

    x_values = [float(row["latency_ms_per_image"]) for row in rows]
    y_values = [float(row["mAP"]) for row in rows]
    devices = {str(row.get("latency_device") or "") for row in rows if row.get("latency_device")}
    device_label = devices.pop() if len(devices) == 1 else "selected device"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, s=52, alpha=0.85)
    for row, x, y in zip(rows, x_values, y_values):
        label = str(row.get("label") or Path(str(row.get("weights", "model"))).stem)
        if row.get("eval_dataset"):
            label = f"{label} / {row['eval_dataset']}"
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)

    x_pad = max(0.05, (max(x_values) - min(x_values)) * 0.08)
    y_pad = max(0.005, (max(y_values) - min(y_values)) * 0.08)
    ax.set_xlim(max(0.0, min(x_values) - x_pad), max(x_values) + x_pad)
    ax.set_ylim(max(0.0, min(y_values) - y_pad), min(1.0, max(y_values) + y_pad))
    ax.set_xlabel(f"Inference latency on {device_label} (ms/image)")
    ax.set_ylabel("mAP")
    ax.set_title("ReID mAP vs inference latency")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _evaluation_args(args, *, weights: Path, model: str | None, target: EvalTarget, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        weights=str(weights),
        model=model,
        dataset=target.dataset,
        data_dir=str(target.data_dir),
        preprocess=getattr(args, "preprocess", None),
        imgsz=getattr(args, "imgsz", None),
        inference_feature=getattr(args, "inference_feature", None),
        flip_tta=getattr(args, "flip_tta", None),
        device=getattr(args, "device", "cpu"),
        batch_size=getattr(args, "batch_size", 64),
        num_workers=getattr(args, "num_workers", 4),
        latency_warmup=getattr(args, "latency_warmup", 5),
        latency_iters=getattr(args, "latency_iters", 30),
        output=str(output),
    )


def main(args) -> dict[str, Any]:
    """Compare an arbitrary set of ReID checkpoints across target datasets."""
    weights = tuple(Path(path).expanduser() for path in _as_tuple(getattr(args, "weights", ())))
    targets = tuple(_parse_target(raw) for raw in _as_tuple(getattr(args, "target", ())))
    if not weights:
        raise ValueError("compare-reid requires at least one --weights checkpoint.")
    if not targets:
        raise ValueError("compare-reid requires at least one --target DATASET=DATA_DIR.")

    labels = _resolve_labels(
        weights,
        tuple(str(label) for label in _as_tuple(getattr(args, "label", ()))),
    )
    models = _resolve_models(
        tuple(str(model) for model in _as_tuple(getattr(args, "model", ()))),
        len(weights),
    )
    include_same_dataset = bool(getattr(args, "include_same_dataset", False))
    continue_on_error = bool(getattr(args, "continue_on_error", False))
    output_dir = Path(getattr(args, "output", None) or "runs/reid_cross_domain").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for weights_path in weights:
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {weights_path}")
    trained_by_model = {weights_path: _trained_dataset_keys(weights_path) for weights_path in weights}

    for weights_path, label, model_name in zip(weights, labels, models):
        train_datasets = trained_by_model[weights_path]
        model_output = output_dir / label
        model_output.mkdir(parents=True, exist_ok=True)

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

            eval_args = _evaluation_args(
                args,
                weights=weights_path,
                model=model_name,
                target=target,
                output=model_output,
            )
            try:
                result = evaluator.main(eval_args)
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
    plot_path = _write_map_latency_plot(results, output_dir / "map_vs_latency.png")
    if plot_path is not None:
        summary["map_latency_plot"] = str(plot_path)

    payload = {
        "summary": summary,
        "results": results,
    }
    json_path = output_dir / "cross_domain_results.json"
    md_path = output_dir / "cross_domain_results.md"
    json_path.write_text(json.dumps(payload, indent=2))
    _write_markdown(results, md_path)
    LOGGER.info(f"Saved cross-domain comparison JSON to {json_path}")
    LOGGER.info(f"Saved cross-domain comparison table to {md_path}")
    return payload
