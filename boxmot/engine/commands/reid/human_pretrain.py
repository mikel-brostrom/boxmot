"""Command adapter for offline human-privileged ReID pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _parse_img_size(value: str) -> tuple[int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image size must be H,W, for example 384,128")
    return parts


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone human-pretraining parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", dest="model_name", default="csl_tinyvit_7m_v20")
    parser.add_argument("--imgsz", type=_parse_img_size, default=(384, 128))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.50)
    parser.add_argument("--consistency-weight", type=float, default=0.50)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=1.0)
    parser.add_argument("--background-weight", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-weights", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None):
    """Run human pretraining from command-line arguments."""

    arguments = vars(build_parser().parse_args(argv))

    from boxmot.reid.training.human_pretraining_runner import (
        HumanPretrainConfig,
        run_human_pretraining,
    )

    arguments["img_size"] = arguments.pop("imgsz")
    return run_human_pretraining(HumanPretrainConfig(**arguments))


__all__ = ("build_parser", "main")


if __name__ == "__main__":
    main()
