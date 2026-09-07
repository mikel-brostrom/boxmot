"""CLI adapter for cross-dataset ReID checkpoint comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from boxmot.engine.commands._support import _run_engine_workflow
from boxmot.engine.commands.reid._options import reid_evaluation_options


def _parse_target(value: str):
    """Parse a ``DATASET=DATA_DIR`` comparison target."""

    from boxmot.reid.evaluation.comparison import EvalTarget

    if "=" in value:
        dataset, data_dir = value.split("=", 1)
    elif ":" in value:
        dataset, data_dir = value.split(":", 1)
    else:
        raise ValueError(f"Invalid target {value!r}; expected DATASET=DATA_DIR.")
    dataset = dataset.strip()
    path = Path(data_dir.strip()).expanduser()
    if not dataset or not str(path):
        raise ValueError(f"Invalid target {value!r}; expected DATASET=DATA_DIR.")
    if not path.exists():
        raise FileNotFoundError(f"Target data directory does not exist: {path}")
    return EvalTarget(dataset=dataset, data_dir=path)


def main(args: Any) -> dict[str, Any]:
    """Run domain comparison and persist aggregate and per-model reports."""

    from boxmot.reid.evaluation.comparison import ReIDComparisonConfig
    from boxmot.reid.evaluation.comparison import compare_reid as run_comparison
    from boxmot.utils import logger as LOGGER

    config = ReIDComparisonConfig(
        weights=tuple(args.weights),
        targets=tuple(_parse_target(value) for value in args.target),
        labels=tuple(getattr(args, "label", ()) or ()),
        models=tuple(getattr(args, "model", ()) or ()),
        include_same_dataset=bool(getattr(args, "include_same_dataset", False)),
        preprocess=getattr(args, "preprocess", None),
        imgsz=getattr(args, "imgsz", None),
        inference_feature=getattr(args, "inference_feature", None),
        flip_tta=getattr(args, "flip_tta", None),
        device=getattr(args, "device", "cpu"),
        batch_size=getattr(args, "batch_size", 64),
        num_workers=getattr(args, "num_workers", 4),
        latency_warmup=getattr(args, "latency_warmup", 5),
        latency_iters=getattr(args, "latency_iters", 30),
        continue_on_error=bool(getattr(args, "continue_on_error", False)),
    )
    payload = run_comparison(config)
    output_dir = Path(getattr(args, "output", "runs/reid_cross_domain")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_evaluation_results(payload["results"], output_dir)

    plot_path = _write_map_latency_plot(payload["results"], output_dir / "map_vs_latency.png")
    if plot_path is not None:
        payload["summary"]["map_latency_plot"] = str(plot_path)

    json_path = output_dir / "cross_domain_results.json"
    markdown_path = output_dir / "cross_domain_results.md"
    json_path.write_text(json.dumps(payload, indent=2))
    _write_markdown(payload["results"], markdown_path)
    LOGGER.info(f"Saved cross-domain comparison JSON to {json_path}")
    LOGGER.info(f"Saved cross-domain comparison table to {markdown_path}")
    return payload


_COMPARISON_ROW_FIELDS = {
    "label",
    "train_dataset",
    "eval_dataset",
    "eval_dataset_key",
    "data_dir",
    "cross_domain",
    "status",
    "error",
}


def _evaluation_filename(result: dict[str, Any]) -> str:
    name = f"eval_{result['model']}_{result['dataset']}"
    if result.get("inference_feature"):
        name += f"_{result['inference_feature']}"
    return f"{name}.json"


def _write_evaluation_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Persist each successful pair in eval-reid's JSON format."""

    for label in dict.fromkeys(str(row["label"]) for row in results):
        (output_dir / label).mkdir(parents=True, exist_ok=True)
    for row in results:
        if row.get("status") != "ok":
            continue
        evaluation = {key: value for key, value in row.items() if key not in _COMPARISON_ROW_FIELDS}
        output_path = output_dir / str(row["label"]) / _evaluation_filename(evaluation)
        output_path.write_text(json.dumps(evaluation, indent=2))


def _markdown_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    headers = (
        "Model",
        "Train",
        "Eval",
        "mAP",
        "Rank-1",
        "Latency ms/img",
        "Device",
        "Feature",
        "Status",
    )
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
                _markdown_value(row.get(key))
                for key in (
                    "label",
                    "train_dataset",
                    "eval_dataset",
                    "mAP",
                    "rank1",
                    "latency_ms_per_image",
                    "latency_device",
                    "inference_feature",
                    "status",
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_map_latency_plot(results: list[dict[str, Any]], path: Path) -> Path | None:
    """Write an mAP-vs-latency scatter plot for successful rows."""

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
        from boxmot.utils import logger as LOGGER

        LOGGER.warning(f"Could not create mAP/latency plot because matplotlib is unavailable: {exc}")
        return None

    x_values = [float(row["latency_ms_per_image"]) for row in rows]
    y_values = [float(row["mAP"]) for row in rows]
    devices = {str(row.get("latency_device") or "") for row in rows if row.get("latency_device")}
    device_label = devices.pop() if len(devices) == 1 else "selected device"
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.scatter(x_values, y_values, s=52, alpha=0.85)
    for row, x_value, y_value in zip(rows, x_values, y_values):
        label = str(row.get("label") or Path(str(row.get("weights", "model"))).stem)
        if row.get("eval_dataset"):
            label = f"{label} / {row['eval_dataset']}"
        axes.annotate(label, (x_value, y_value), xytext=(5, 4), textcoords="offset points", fontsize=8)

    x_pad = max(0.05, (max(x_values) - min(x_values)) * 0.08)
    y_pad = max(0.005, (max(y_values) - min(y_values)) * 0.08)
    axes.set_xlim(max(0.0, min(x_values) - x_pad), max(x_values) + x_pad)
    axes.set_ylim(max(0.0, min(y_values) - y_pad), min(1.0, max(y_values) + y_pad))
    axes.set_xlabel(f"Inference latency on {device_label} (ms/image)")
    axes.set_ylabel("mAP")
    axes.set_title("ReID mAP vs inference latency")
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


@click.command(name="compare-reid", help="Compare ReID checkpoints across target datasets")
@click.option(
    "--weights",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
    help="Path to a trained ReID checkpoint. Repeat for multiple models.",
)
@click.option(
    "--target",
    multiple=True,
    required=True,
    help="Evaluation target as DATASET=DATA_DIR. Repeat for multiple target datasets.",
)
@click.option("--label", multiple=True, help="Optional display/output label for each --weights entry.")
@click.option(
    "--model",
    multiple=True,
    help="Model architecture override. Pass once for all weights or once per checkpoint.",
)
@click.option(
    "--include-same-dataset/--cross-domain-only",
    default=False,
    show_default=True,
    help="Also evaluate models on their training dataset when checkpoint metadata is available.",
)
@reid_evaluation_options
@click.option(
    "--continue-on-error/--fail-fast",
    default=False,
    show_default=True,
    help="Record failed pairs and continue instead of stopping at the first failure.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="runs/reid_cross_domain",
    show_default=True,
    help="Directory to save aggregate comparison files and per-model eval JSONs",
)
def compare_reid(**kwargs: Any) -> Any:
    """Compare trained ReID checkpoints across target datasets."""

    from types import SimpleNamespace

    return _run_engine_workflow(__name__, SimpleNamespace(**kwargs))


__all__ = ("compare_reid", "main")
