"""Measure CMC latency and known-motion accuracy on a real image texture.

Run from the repository root::

    uv run --no-sync python -m tests.performance.trackers.motion.benchmark_cmc_latency \
        --widths 1280 1920 3840 --json /tmp/cmc-latency.json

Frames are preloaded. Each timed call includes preprocessing, estimation, and
state refresh. Accuracy uses a known translation and similarity transform of
the first image, in both directions, so returning identity cannot appear as a
successful optimization. The real image pair has no ground-truth transform.
The similarity case intentionally exceeds the translation-only ECC model.
Timing is a host-specific benchmark, not a portable unit-test assertion.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from boxmot.trackers.common.motion.cmc.base import BaseCMC
from boxmot.trackers.common.motion.cmc.registry import available_cmc_methods, create_cmc
from boxmot.utils import ROOT

DEFAULT_IMAGE_DIRECTORY = ROOT / "assets/MOT17-mini/train/MOT17-04-FRCNN/img1"
DEFAULT_METHODS = ("ecc", "sof")


@dataclass(frozen=True)
class LatencyResult:
    """A warmed estimator's per-call latency and optional geometric error."""

    method: str
    scenario: str
    width: int
    height: int
    apply_calls: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    maximum_ms: float
    mean_error_px: float | None
    p95_error_px: float | None
    identity_fraction: float
    p95_within_budget: bool


def _read_image(path: Path) -> np.ndarray:
    """Read one BGR image, failing before any measurements if unavailable."""
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def _point_error(estimated: np.ndarray, expected: np.ndarray, points: np.ndarray) -> float:
    """Return mean mapping error on a fixed interior image grid in pixels."""
    actual = points @ estimated.T
    if actual.shape[1] == 3:
        actual = actual[:, :2] / actual[:, 2:]
    reference = points @ expected.T
    return float(np.linalg.norm(actual - reference[:, :2], axis=1).mean())


def _benchmark_pair(
    method: str,
    scenario: str,
    images: tuple[np.ndarray, np.ndarray],
    forward_transform: np.ndarray | None,
    *,
    warmup: int,
    iterations: int,
    budget_ms: float,
) -> LatencyResult:
    """Measure alternating forward/backward pairs without decoding or scoring."""
    cmc = create_cmc(method)
    if cmc is None:
        raise ValueError(f"Cannot benchmark disabled CMC method: {method}")
    detections = np.empty((0, 4), dtype=np.float32)
    cmc.apply(images[0], detections)
    # Begin and end on images[0], making the first measured call a forward warp.
    for _ in range(warmup):
        cmc.apply(images[1], detections)
        cmc.apply(images[0], detections)

    elapsed_ms: list[float] = []
    transforms: list[np.ndarray] = []
    for _ in range(iterations):
        for image in (images[1], images[0]):
            started = time.perf_counter_ns()
            transform = cmc.apply(image, detections)
            elapsed_ms.append((time.perf_counter_ns() - started) / 1e6)
            transforms.append(transform.copy())

    height, width = images[0].shape[:2]
    grid = np.stack(
        np.meshgrid(np.linspace(0.1 * width, 0.9 * width, 9), np.linspace(0.1 * height, 0.9 * height, 7)),
        axis=-1,
    ).reshape(-1, 2)
    points = np.column_stack((grid, np.ones(len(grid))))
    errors: list[float] = []
    identities = 0
    expected = (
        None
        if forward_transform is None
        else (
            forward_transform,
            cv2.invertAffineTransform(forward_transform),
        )
    )
    for index, transform in enumerate(transforms):
        if not BaseCMC.is_valid_transform(transform):
            raise RuntimeError(f"{method} returned an invalid transform in {scenario}: {transform}")
        identities += bool(np.allclose(transform, np.eye(*transform.shape), atol=1e-6))
        if expected is not None:
            errors.append(_point_error(transform, expected[index % 2], points))

    p95_ms = float(np.percentile(elapsed_ms, 95))
    return LatencyResult(
        method=method,
        scenario=scenario,
        width=width,
        height=height,
        apply_calls=len(elapsed_ms),
        mean_ms=float(np.mean(elapsed_ms)),
        median_ms=float(np.median(elapsed_ms)),
        p95_ms=p95_ms,
        maximum_ms=max(elapsed_ms),
        mean_error_px=float(np.mean(errors)) if errors else None,
        p95_error_px=float(np.percentile(errors, 95)) if errors else None,
        identity_fraction=identities / len(transforms),
        p95_within_budget=p95_ms <= budget_ms,
    )


def _print_results(results: Sequence[LatencyResult], budget_ms: float) -> None:
    """Display latency and quality together to make tradeoffs visible."""
    table = Table(title=f"CMC latency and accuracy ({budget_ms:g} ms p95 budget)")
    for heading in ("Size", "Method", "Scenario", "Mean ms", "p50 ms", "p95 ms", "Error px", "Identity %", "Budget"):
        table.add_column(heading)
    for result in results:
        table.add_row(
            f"{result.width}x{result.height}",
            result.method,
            result.scenario,
            f"{result.mean_ms:.3f}",
            f"{result.median_ms:.3f}",
            f"{result.p95_ms:.3f}",
            "-" if result.mean_error_px is None else f"{result.mean_error_px:.3f}",
            f"{100 * result.identity_fraction:.0f}",
            "pass" if result.p95_within_budget else "over",
        )
    Console(width=120).print(table)


def main(argv: Sequence[str] | None = None) -> int:
    """Run reproducible benchmarks with an explicit OpenCV thread count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-image", type=Path, default=DEFAULT_IMAGE_DIRECTORY / "000001.jpg")
    parser.add_argument("--current-image", type=Path, default=DEFAULT_IMAGE_DIRECTORY / "000005.jpg")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=available_cmc_methods(),
        default=DEFAULT_METHODS,
        help="methods to benchmark; orb/sift are slower opt-ins (default: ecc sof)",
    )
    parser.add_argument(
        "--widths", nargs="+", type=int, help="resize both images to these widths, preserving aspect ratio"
    )
    parser.add_argument("--warmup", type=int, default=5, help="untimed pairs per method/scenario")
    parser.add_argument("--iterations", type=int, default=100, help="timed pairs per method/scenario")
    parser.add_argument("--threads", type=int, default=1, help="OpenCV threads (default: 1)")
    parser.add_argument("--budget-ms", type=float, default=1.0, help="per-call p95 latency budget")
    parser.add_argument(
        "--check-budget", action="store_true", help="exit nonzero if any measured p95 exceeds the budget"
    )
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)
    if args.warmup < 0 or min(args.iterations, args.threads, args.budget_ms) <= 0:
        parser.error("warmup must be nonnegative; iterations, threads, and budget must be positive")
    if args.widths is not None and min(args.widths) <= 0:
        parser.error("widths must be positive")
    try:
        previous = _read_image(args.previous_image)
        current = _read_image(args.current_image)
        if previous.shape != current.shape:
            raise ValueError(f"Image shapes must match: {previous.shape} != {current.shape}")
    except ValueError as exc:
        parser.error(str(exc))

    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(0)
    results: list[LatencyResult] = []
    for width in args.widths or [previous.shape[1]]:
        height = round(previous.shape[0] * width / previous.shape[1])
        image_a = cv2.resize(previous, (width, height))
        image_b = cv2.resize(current, (width, height))
        translation = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, -4.0]], dtype=np.float32)
        similarity = cv2.getRotationMatrix2D((width / 2, height / 2), 0.5, 1.002)
        similarity[:, 2] += (6.0, -4.0)
        scenarios = [
            ("real", image_b, None),
            ("translation", cv2.warpAffine(image_a, translation, (width, height)), translation),
            ("similarity", cv2.warpAffine(image_a, similarity, (width, height)), similarity),
        ]
        for method in args.methods:
            for scenario, second_image, expected in scenarios:
                results.append(
                    _benchmark_pair(
                        method,
                        scenario,
                        (image_a, second_image),
                        expected,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        budget_ms=args.budget_ms,
                    )
                )
    _print_results(results, args.budget_ms)
    if args.json_path is not None:
        metadata = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "opencv_version": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "previous_image": str(args.previous_image.resolve()),
            "current_image": str(args.current_image.resolve()),
            "warmup_pairs": args.warmup,
            "budget_ms": args.budget_ms,
            "results": [asdict(result) for result in results],
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return int(args.check_budget and any(not result.p95_within_budget for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
