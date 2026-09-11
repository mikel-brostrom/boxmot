"""Time batched CMC state transforms against a saved scalar implementation.

Run with ``python -m tests.performance.trackers.motion.benchmark_cmc_states
--baseline <directory-containing-boxmot> --json <results.json>``. Input creation
and scalar/batch parity checks are excluded from timing. Results measure state
transforms, including covariance, rather than camera-transform estimation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from boxmot.trackers.common.motion.cmc.state import (
    transform_aabb_kalman_states,
    transform_obb_kalman_states,
)
from boxmot.trackers.common.motion.models import create_motion_model


def _elapsed(call: Callable, repeat: int, iterations: int) -> float:
    """Return median seconds per call after one discarded warmup."""
    call()
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        for _ in range(iterations):
            call()
        times.append((time.perf_counter() - started) / iterations)
    return statistics.median(times)


def main() -> None:
    """Exercise shared AABB and OBB state formats with correlated covariances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sizes", default="1,10,100")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warp", choices=("similarity", "affine", "homography"), default="similarity")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or min(*sizes, args.repeat, args.iterations) < 1:
        parser.error("Sizes, repeat, and iterations must be positive")
    path = args.baseline / "boxmot/trackers/common/geometry/obb.py"
    spec = importlib.util.spec_from_file_location("_cmc_scalar_reference", path)
    if spec is None or spec.loader is None:
        parser.error(f"Cannot load scalar CMC implementation from {path}")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    angle = 0.08
    transform = np.array(
        [[1.01 * np.cos(angle), -1.01 * np.sin(angle), 5.0], [1.01 * np.sin(angle), 1.01 * np.cos(angle), -3.0]],
    )
    if args.warp != "similarity":
        transform[0, 1] += 0.03
        transform[1, 1] -= 0.04
    if args.warp == "homography":
        transform = np.vstack((transform, [1e-5, -2e-5, 1.0]))
    rows = []
    for kind, obb in (
        ("xywh", False),
        ("xyah", False),
        ("xysr", False),
        ("xywh", True),
        ("xysr", True),
        ("xyhr", True),
    ):
        adapter = create_motion_model(kind, is_obb=obb)
        scalar = reference.transform_obb_kalman_state if obb else reference.transform_aabb_kalman_state
        batch = transform_obb_kalman_states if obb else transform_aabb_kalman_states
        velocity_indices = (0, 1, 2, 4) if kind == "xysr" and obb else tuple(range(adapter.dim_x - adapter.dim_z))
        for count in sizes:
            rng = np.random.default_rng(1234)
            boxes = np.column_stack((rng.uniform(50, 500, (count, 2)), rng.uniform(10, 100, (count, 2))))
            if obb:
                boxes = np.column_stack((boxes, rng.uniform(-np.pi, np.pi, count)))
            else:
                boxes[:, :2] -= boxes[:, 2:] / 2
                boxes[:, 2:] += boxes[:, :2]
            means = rng.normal(size=(count, adapter.dim_x))
            means[:, : adapter.dim_z] = [adapter.to_measurement(box, column=False) for box in boxes]
            noise = rng.normal(size=(count, adapter.dim_x, adapter.dim_x))
            covariances = noise @ noise.swapaxes(-1, -2) + np.eye(adapter.dim_x)

            def run_scalar():
                return [
                    scalar(
                        mean,
                        covariance,
                        transform,
                        measurement_to_box=lambda values: adapter.to_box(values)[0],
                        box_to_measurement=lambda box: adapter.to_measurement(box, column=False),
                        velocity_measurement_indices=velocity_indices,
                    )
                    for mean, covariance in zip(means, covariances)
                ]

            def run_batch():
                return batch(
                    means,
                    covariances,
                    transform,
                    measurement_to_box=adapter.to_boxes,
                    box_to_measurement=adapter.to_measurements,
                    velocity_measurement_indices=velocity_indices,
                )

            expected = run_scalar()
            actual_mean, actual_covariance = run_batch()
            expected_mean = np.asarray([item[0] for item in expected])
            expected_covariance = np.asarray([item[1] for item in expected])
            np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(actual_covariance, expected_covariance, rtol=1e-6, atol=1e-6)
            before = _elapsed(run_scalar, args.repeat, args.iterations)
            after = _elapsed(run_batch, args.repeat, args.iterations)
            row = {
                "model": kind,
                "geometry": "obb" if obb else "aabb",
                "tracks": count,
                "before_ms": 1000 * before,
                "after_ms": 1000 * after,
                "speedup": before / after,
                "max_mean_error": float(np.max(np.abs(actual_mean - expected_mean))),
                "max_covariance_error": float(np.max(np.abs(actual_covariance - expected_covariance))),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps({"warp": args.warp, "repeat": args.repeat, "iterations": args.iterations, "results": rows}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
