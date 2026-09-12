"""Time scalar and batched Kalman prediction/correction with identical inputs.

Run from the repository root::

    uv run --no-sync python -m tests.performance.trackers.motion.benchmark_kalman \
        --sizes 1,10,100,1000 --repeat 7 --iterations 20 --json /tmp/kalman.json

Use ``--baseline-root /path/to/prior/checkout`` to additionally compare existing
XYAH/XYWH batch prediction with that checkout. Preparation, state reset and
parity checks are untimed. Stateful timings include gathering/scattering arrays
and ordinary observation bookkeeping; they exclude missing-observation replay.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from rich.console import Console
from rich.table import Table
from threadpoolctl import threadpool_info, threadpool_limits

from boxmot.trackers.common.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.trackers.common.motion.kalman_filters.xyhr import KalmanFilterXYHR
from boxmot.trackers.common.motion.kalman_filters.xyscr import KalmanFilterXYSCR
from boxmot.trackers.common.motion.kalman_filters.xysr import KalmanFilterXYSR
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.eagermot.motion import Kalman3D

VARIANTS = (
    "xyah",
    "xyah-obb",
    "xywh",
    "xywh-obb",
    "xysr",
    "xysr-obb",
    "xyscr",
    "xyhr",
    "xyhr-obb",
    "eagermot",
    "eagermot-angular",
)
THREAD_ENVIRONMENT = ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass
class Case:
    """Untimed resets and equivalent numerical operations for one experiment."""

    scalar: Callable
    batched: Callable
    reset: Callable
    snapshot: Callable
    prior_batch: Callable | None = None


def _state_data(name: str, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate finite boxes and distinct correlated positive-definite covariances."""
    rng = np.random.default_rng(seed)
    if name.startswith("eagermot"):
        measurement = np.tile([10.0, 2.0, 25.0, 0.2, 4.0, 1.8, 1.6], (count, 1))
        dimension = 11 if name.endswith("angular") else 10
    else:
        values = {
            "xyah": [100.0, 150.0, 1.5, 40.0],
            "xywh": [100.0, 150.0, 60.0, 40.0],
            "xysr": [100.0, 150.0, 2400.0, 1.5],
            "xyscr": [100.0, 150.0, 2400.0, 0.8, 1.5],
            "xyhr": [100.0, 150.0, 40.0, 1.5],
        }[name.split("-")[0]]
        if name.endswith("obb"):
            values = [*values, 0.2]
        measurement = np.tile(values, (count, 1))
        dimension = 2 * measurement.shape[1] - (1 if name.startswith(("xysr", "xyscr")) else 0)
    measurement[:, :2] += rng.uniform(-8.0, 8.0, (count, 2))
    mean = np.zeros((count, dimension))
    mean[:, : measurement.shape[1]] = measurement
    mean[:, measurement.shape[1] :] = rng.normal(0.0, 0.1, (count, dimension - measurement.shape[1]))
    measurements = measurement.copy()
    measurements[:, :2] += rng.normal(0.0, 0.3, (count, 2))
    matrix = rng.normal(0.0, 0.3, (count, dimension, dimension))
    covariance = matrix @ matrix.swapaxes(-1, -2) + np.eye(dimension) * 2.0
    return mean, covariance, measurements


def _case(name: str, count: int, operation: str, seed: int, dt: float | None, prior=None) -> Case:
    """Build benchmark inputs once, exposing resets outside the timed interval."""
    means, covariances, measurements = _state_data(name, count, seed)
    obb = name.endswith("obb")
    if name.startswith(("xyah", "xywh")):
        factory = KalmanFilterXYAH if name.startswith("xyah") else KalmanFilterXYWH
        kf = factory(ndim=5 if obb else 4)
        scalar_method = kf.predict if operation == "predict" else kf.update
        batch_method = kf.multi_predict if operation == "predict" else kf.multi_update

        def scalar():
            result = [
                scalar_method(mean, covariance, dt=dt)
                if operation == "predict"
                else scalar_method(mean, covariance, measurement)
                for mean, covariance, measurement in zip(means, covariances, measurements)
            ]
            return tuple(np.stack(values) for values in zip(*result))

        def batched():
            if operation == "predict":
                return batch_method(means, covariances, dt=dt)
            return batch_method(means, covariances, measurements)

        prior_batch = None
        if prior is not None and operation == "predict":
            prior_filter = prior(ndim=5 if obb else 4)

            def prior_batch():
                return prior_filter.multi_predict(means, covariances, dt=dt)

        return Case(scalar, batched, lambda: None, lambda result: result, prior_batch)

    if name.startswith("eagermot"):
        models = [Kalman3D(z, is_angular=name.endswith("angular")) for z in measurements]

        def reset():
            for model, mean, covariance in zip(models, means, covariances):
                model.state, model.covariance = mean.copy(), covariance.copy()

        def scalar():
            for model, measurement in zip(models, measurements):
                model.predict() if operation == "predict" else model.update(measurement)

        def batched():
            if operation == "predict":
                Kalman3D.multi_predict(models)
            else:
                Kalman3D.multi_update(models, measurements)

        def snapshot(_result):
            return np.stack([model.state for model in models]), np.stack([model.covariance for model in models])

        return Case(scalar, batched, reset, snapshot)

    if name.startswith("xysr"):
        models = [KalmanFilterXYSR(dim_x=9 if obb else 7, dim_z=5 if obb else 4) for _ in range(count)]
    elif name == "xyscr":
        models = [KalmanFilterXYSCR() for _ in range(count)]
    else:
        models = [KalmanFilterXYHR(z) for z in measurements]
    factory = type(models[0])

    def reset():
        for model, mean, covariance in zip(models, means, covariances):
            model.x = mean.copy() if name.startswith("xyhr") else mean[:, None].copy()
            model.P = covariance.copy()
            model.observed = True
            model.history_obs.clear()
            model._last_observed_measurement = None
            model._prediction_origin = None
            model._prediction_steps.clear()
            model._time_aware = False
            model._unrecorded_prediction = False
            model._prediction_history_overflowed = False

    def scalar():
        for model, measurement in zip(models, measurements):
            model.predict(dt=dt) if operation == "predict" else model.update(measurement)

    def batched():
        if operation == "predict":
            factory.predict_many(models, dt=dt)
        else:
            factory.update_many(models, measurements)

    def snapshot(_result):
        return np.stack([model.x.reshape(-1) for model in models]), np.stack([model.P for model in models])

    return Case(scalar, batched, reset, snapshot)


def _measure(case: Case, *, warmup: int, repeat: int, iterations: int) -> dict:
    """Check parity then alternate methods, reporting medians of repeated means."""
    methods = {"scalar": case.scalar, "batch": case.batched}
    if case.prior_batch is not None:
        methods["prior_batch"] = case.prior_batch
    reference = None
    for name, operation in methods.items():
        case.reset()
        snapshot = case.snapshot(operation())
        if reference is None:
            reference = snapshot
        else:
            for expected, observed in zip(reference, snapshot):
                np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-9, err_msg=f"{name} parity")
        for _ in range(warmup):
            case.reset()
            operation()

    samples = {name: [] for name in methods}
    for repetition in range(repeat):
        order = list(methods) if repetition % 2 == 0 else list(reversed(methods))
        for name in order:
            elapsed = 0
            operation = methods[name]
            for _ in range(iterations):
                case.reset()
                started = time.perf_counter_ns()
                operation()
                elapsed += time.perf_counter_ns() - started
            samples[name].append(elapsed / iterations / 1e6)
    medians = {name: float(np.median(values)) for name, values in samples.items()}
    result = {
        "parity": "passed",
        "median_ms": medians,
        "repeat_ms": samples,
        "speedup": medians["scalar"] / medians["batch"],
    }
    if "prior_batch" in medians:
        result["existing_batch_speedup"] = medians["prior_batch"] / medians["batch"]
    return result


def _prior_factories(root: Path | None) -> dict:
    """Load a supplied checkout's batch filters without replacing live modules."""
    if root is None:
        return {}
    directory = root / "boxmot/trackers/common/motion/kalman_filters"
    module_name = "_kalman_benchmark_prior_base"
    spec = importlib.util.spec_from_file_location(module_name, directory / "base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    factories = {}
    for name in ("xyah", "xywh"):
        path = directory / f"{name}.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "boxmot.trackers.common.motion.kalman_filters.base":
                node.module = module_name
        loaded = types.ModuleType(f"_kalman_benchmark_prior_{name}")
        exec(compile(tree, str(path), "exec"), loaded.__dict__)
        factories[name] = getattr(loaded, f"KalmanFilter{name.upper()}")
    return factories


def _metadata() -> dict:
    """Record software, processor and the effective BLAS thread limits."""
    processor = platform.processor()
    if platform.system() == "Darwin":
        query = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, capture_output=True, check=False
        )
        if query.returncode == 0:
            processor = query.stdout.strip()
    return {
        "platform": platform.platform(),
        "processor": processor,
        "logical_cpus": os.cpu_count(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threadpools": threadpool_info(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "timer": "perf_counter_ns",
    }


def main() -> None:
    """Run the explicitly requested numerical microbenchmarks and write JSON."""
    if any(os.environ.get(name) != "1" for name in THREAD_ENVIRONMENT):
        # Accelerate is not exposed by threadpoolctl. Configure every backend
        # before importing NumPy in a fresh child so its limit is effective.
        environment = {**os.environ, **dict.fromkeys(THREAD_ENVIRONMENT, "1")}
        command = [sys.executable, "-m", "tests.performance.trackers.motion.benchmark_kalman", *sys.argv[1:]]
        raise SystemExit(subprocess.run(command, env=environment, check=False).returncode)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1,10,100,1000", help="comma-separated track counts")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated filter/geometry names")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20, help="calls per method per repetition")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--dt", type=float, help="explicit elapsed interval for 2D filters; 3D always uses one frame")
    parser.add_argument(
        "--baseline-root", type=Path, help="optional prior checkout for existing XYAH/XYWH batch prediction"
    )
    parser.add_argument("--json", type=Path, required=True, dest="json_path")
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    variants = args.variants.split(",")
    if any(value < 1 for value in sizes) or min(args.repeat, args.iterations) < 1 or args.warmup < 0:
        parser.error("sizes, repeat and iterations must be positive; warmup must be nonnegative")
    if set(variants) - set(VARIANTS):
        parser.error(f"variants must be selected from {', '.join(VARIANTS)}")
    prior = _prior_factories(args.baseline_root)
    console = Console()
    results = []
    with threadpool_limits(limits=1):
        metadata = _metadata()
        for variant in variants:
            for count in sizes:
                for operation in ("predict", "update"):
                    case = _case(variant, count, operation, args.seed, args.dt, prior.get(variant.split("-")[0]))
                    result = {"filter": variant, "tracks": count, "operation": operation}
                    result.update(_measure(case, warmup=args.warmup, repeat=args.repeat, iterations=args.iterations))
                    results.append(result)
            console.print(f"Completed {variant}")
    payload = {
        "metadata": metadata,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": (
            "Numerical KF steps, including stateful gather/scatter and matched observation bookkeeping; "
            "no tracking association or gap replay."
        ),
        "results": results,
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2) + "\n")
    table = Table("Filter", "Tracks", "Step", "Scalar ms", "Batch ms", "Speedup")
    for result in results:
        table.add_row(
            result["filter"],
            str(result["tracks"]),
            result["operation"],
            f"{result['median_ms']['scalar']:.4f}",
            f"{result['median_ms']['batch']:.4f}",
            f"{result['speedup']:.2f}x",
        )
    console.print(table)
    console.print(f"Results: {args.json_path}")


if __name__ == "__main__":
    main()
