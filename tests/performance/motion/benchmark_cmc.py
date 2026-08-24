"""Benchmark camera-motion-compensation methods on a fixed image pair.

Run from the repository root::

    uv run python -m tests.performance.motion.benchmark_cmc

Use ``--help`` to select methods, images, iteration counts, and JSON output.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from boxmot.motion.cmc.base_cmc import BaseCMC
from boxmot.motion.cmc.ecc import ECC
from boxmot.motion.cmc.orb import ORB
from boxmot.motion.cmc.sift import SIFT
from boxmot.motion.cmc.sof import SOF
from boxmot.utils import ROOT

DEFAULT_PREVIOUS_IMAGE = ROOT / "assets/MOT17-mini/train/MOT17-04-FRCNN/img1/000001.jpg"
DEFAULT_CURRENT_IMAGE = ROOT / "assets/MOT17-mini/train/MOT17-04-FRCNN/img1/000005.jpg"
DEFAULT_METHODS = ("ecc", "orb", "sift", "sof")
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 100
DETECTIONS = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)

CMC_METHODS: dict[str, type[BaseCMC]] = {
    "ecc": ECC,
    "orb": ORB,
    "sift": SIFT,
    "sof": SOF,
}


@dataclass(frozen=True)
class BenchmarkResult:
    """Serializable result for one CMC implementation."""

    method: str
    status: str
    iterations: int
    apply_calls: int
    elapsed_s: float | None = None
    mean_ms_per_apply: float | None = None
    applies_per_second: float | None = None
    error: str | None = None


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    if not methods:
        raise argparse.ArgumentTypeError("at least one CMC method is required")
    invalid = sorted(set(methods) - CMC_METHODS.keys())
    if invalid:
        available = ", ".join(CMC_METHODS)
        raise argparse.ArgumentTypeError(f"unknown CMC method(s): {', '.join(invalid)}; choose from {available}")
    return methods


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=DEFAULT_METHODS,
        metavar="LIST",
        help=f"comma-separated methods (default: {','.join(DEFAULT_METHODS)})",
    )
    parser.add_argument(
        "--previous-image",
        type=Path,
        default=DEFAULT_PREVIOUS_IMAGE,
        help=f"first image in the alternating benchmark pair (default: {DEFAULT_PREVIOUS_IMAGE})",
    )
    parser.add_argument(
        "--current-image",
        type=Path,
        default=DEFAULT_CURRENT_IMAGE,
        help=f"second image in the alternating benchmark pair (default: {DEFAULT_CURRENT_IMAGE})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"untimed image-pair iterations per method (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"timed image-pair iterations per method (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument("--json", type=Path, dest="json_path", help="optional path for JSON results")
    return parser


def _load_image(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} image does not exist: {path}. Pass --{label.replace(' ', '-')}-image with a readable image path."
        )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode the {label} image: {path}")
    return image


def _validate_image_pair(previous_image: np.ndarray, current_image: np.ndarray) -> None:
    if previous_image.shape != current_image.shape:
        raise ValueError(
            "benchmark images must have matching shapes; "
            f"got previous={previous_image.shape} and current={current_image.shape}"
        )


def _validate_transform(method: str, transform: np.ndarray) -> None:
    if not BaseCMC.is_valid_transform(transform):
        raise RuntimeError(f"{method} returned an invalid transform with shape {np.asarray(transform).shape}")


def _benchmark_method(
    method: str,
    previous_image: np.ndarray,
    current_image: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    apply_calls = iterations * 2
    try:
        cmc = CMC_METHODS[method]()
        for _ in range(warmup):
            cmc.apply(previous_image, DETECTIONS)
            cmc.apply(current_image, DETECTIONS)

        started = time.perf_counter()
        for _ in range(iterations):
            previous_transform = cmc.apply(previous_image, DETECTIONS)
            current_transform = cmc.apply(current_image, DETECTIONS)
        elapsed_s = time.perf_counter() - started

        _validate_transform(method, previous_transform)
        _validate_transform(method, current_transform)
    except Exception as exc:
        return BenchmarkResult(
            method=method,
            status="error",
            iterations=iterations,
            apply_calls=apply_calls,
            error=f"{type(exc).__name__}: {exc}",
        )

    return BenchmarkResult(
        method=method,
        status="ok",
        iterations=iterations,
        apply_calls=apply_calls,
        elapsed_s=round(elapsed_s, 6),
        mean_ms_per_apply=round(elapsed_s * 1000.0 / apply_calls, 4),
        applies_per_second=round(apply_calls / elapsed_s, 2) if elapsed_s > 0 else None,
    )


def _print_results(results: Sequence[BenchmarkResult], console: Console) -> None:
    table = Table(title="BoxMOT CMC performance")
    table.add_column("Method", style="bold")
    table.add_column("Apply calls", justify="right")
    table.add_column("Elapsed (s)", justify="right")
    table.add_column("Mean (ms/apply)", justify="right")
    table.add_column("Applies/s", justify="right")
    table.add_column("Status")
    for result in results:
        table.add_row(
            result.method,
            str(result.apply_calls),
            f"{result.elapsed_s:.4f}" if result.elapsed_s is not None else "-",
            f"{result.mean_ms_per_apply:.4f}" if result.mean_ms_per_apply is not None else "-",
            f"{result.applies_per_second:.2f}" if result.applies_per_second is not None else "-",
            "[green]OK[/green]" if result.status == "ok" else f"[red]{result.error}[/red]",
        )
    console.print(table)


def _write_json(path: Path, results: Sequence[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    try:
        previous_image = _load_image(args.previous_image, "previous")
        current_image = _load_image(args.current_image, "current")
        _validate_image_pair(previous_image, current_image)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    results = [
        _benchmark_method(
            method,
            previous_image,
            current_image,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for method in args.methods
    ]
    console = Console()
    _print_results(results, console)

    if args.json_path is not None:
        try:
            _write_json(args.json_path, results)
        except OSError as exc:
            console.print(f"[red]Could not write JSON results to {args.json_path}: {exc}[/red]")
            return 1
        console.print(f"Saved JSON results to {args.json_path}")

    return 1 if any(result.status != "ok" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
