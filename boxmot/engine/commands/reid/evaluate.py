"""CLI adapter for standalone ReID checkpoint evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from boxmot.engine.commands._support import _run_engine_workflow
from boxmot.engine.commands.reid._options import reid_evaluation_options


def main(args: Any) -> dict[str, Any]:
    """Run domain evaluation and persist its JSON report."""

    from boxmot.reid.evaluation.runner import ReIDEvaluationConfig, evaluate_reid
    from boxmot.utils import logger as LOGGER

    config = ReIDEvaluationConfig(
        weights=args.weights,
        model=getattr(args, "model", None),
        dataset=args.dataset,
        data_dir=args.data_dir,
        preprocess=getattr(args, "preprocess", None),
        imgsz=getattr(args, "imgsz", None),
        inference_feature=getattr(args, "inference_feature", None),
        flip_tta=getattr(args, "flip_tta", None),
        device=getattr(args, "device", "cpu"),
        batch_size=getattr(args, "batch_size", 64),
        num_workers=getattr(args, "num_workers", 4),
        latency_warmup=getattr(args, "latency_warmup", 5),
        latency_iters=getattr(args, "latency_iters", 30),
    )
    result = evaluate_reid(config)

    feature_name = result.get("inference_feature")
    output = getattr(args, "output", None)
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"eval_{result['model']}_{result['dataset']}"
    else:
        output_dir = Path(args.weights).parent
        filename = f"eval_{result['dataset']}"
    if feature_name:
        filename += f"_{feature_name}"
    json_path = output_dir / f"{filename}.json"
    json_path.write_text(json.dumps(result, indent=2))
    LOGGER.info(f"Saved evaluation results to {json_path}")
    return result


@click.command(name="eval-reid", help="Evaluate a trained ReID model on query/gallery")
@click.option(
    "--weights",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained ReID checkpoint (.pt)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model architecture (auto-detected from checkpoint if omitted)",
)
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="Evaluation dataset (e.g. market1501, duke, msmt17)",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    required=True,
    help="Root directory of the dataset",
)
@reid_evaluation_options
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Directory to save eval JSON (default: next to weights)",
)
def eval_reid(**kwargs: Any) -> Any:
    """Evaluate a trained ReID checkpoint."""

    from types import SimpleNamespace

    return _run_engine_workflow(__name__, SimpleNamespace(**kwargs))


__all__ = ("eval_reid", "main")
