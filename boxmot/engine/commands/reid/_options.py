"""Shared, import-light Click options for ReID evaluation commands."""

from __future__ import annotations

from typing import Callable

import click

from boxmot.engine.commands._options import _parse_imgsz

_INFERENCE_FEATURES = (
    "concat_bn",
    "norm_concat_bn",
    "global",
    "raw_mean",
    "raw_concat",
    "visibility_weighted_parts",
    "evidence_sinkhorn",
    "dse_weighted",
    "dse_mix",
)


def reid_evaluation_options(func: Callable) -> Callable:
    """Attach preprocessing, inference, runtime, and latency options."""

    options = [
        click.option(
            "--preprocess",
            type=click.Choice(("resize", "resize_pad"), case_sensitive=False),
            default=None,
            help="Crop preprocessing method (default: checkpoint/hparams value)",
        ),
        click.option(
            "--imgsz",
            callback=_parse_imgsz,
            type=str,
            default=None,
            help="Image size as H,W (default: hparams value, fallback 256,128)",
        ),
        click.option(
            "--inference-feature",
            type=click.Choice(_INFERENCE_FEATURES, case_sensitive=False),
            default=None,
            help="Override CSL-TinyViT eval embedding without retraining",
        ),
        click.option(
            "--flip-tta/--no-flip-tta",
            default=None,
            help="Use horizontal flip test-time augmentation (default: hparams value)",
        ),
        click.option("--device", default="cpu", help="Device: cpu, mps, or cuda index"),
        click.option(
            "--batch-size",
            type=int,
            default=64,
            show_default=True,
            help="Batch size for feature extraction",
        ),
        click.option(
            "--num-workers",
            type=int,
            default=4,
            show_default=True,
            help="Dataloader workers",
        ),
        click.option(
            "--latency-warmup",
            type=int,
            default=5,
            show_default=True,
            help="Warmup forward passes before measuring ReID inference latency",
        ),
        click.option(
            "--latency-iters",
            type=int,
            default=30,
            show_default=True,
            help="Timed forward passes for ReID inference latency; 0 disables latency measurement",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


__all__ = ("reid_evaluation_options",)
