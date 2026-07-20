"""Compare ReID inference latency across PyTorch and ONNX runtimes.

The defaults reproduce the CSL-TinyViT-11M FPN versus LMBN-n comparison::

    uv run python -m tests.performance.benchmark_reid_inference

The benchmark exports static FP32 ONNX graphs when they are missing or stale,
then measures PyTorch CPU/MPS and ONNX Runtime CPU/CoreML. ONNX Runtime does
not expose an MPS execution provider; CoreML with ``CPUAndGPU`` is the closest
Apple GPU path and is reported separately from true PyTorch MPS.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

RESULT_PREFIX = "BOXMOT_REID_BENCHMARK_RESULT="
MODULE_NAME = "tests.performance.benchmark_reid_inference"
DEFAULT_RUNTIMES = ("pytorch-cpu", "pytorch-mps", "onnx-cpu", "onnx-coreml")
VALID_RUNTIMES = frozenset(DEFAULT_RUNTIMES)


@dataclass(frozen=True)
class ModelSpec:
    """One model participating in the benchmark."""

    label: str
    weights: Path

    @property
    def onnx(self) -> Path:
        return self.weights.with_suffix(".onnx")


DEFAULT_MODELS = (
    ModelSpec(
        "CSL-TinyViT-11M FPN",
        Path("runs/ablation_csl_tinyvit_11m_stage1_concat/d_last3_fpn_stage1_split_feat512/best.pt"),
    ),
    ModelSpec("LMBN-n", Path("models/lmbn_n_market.pt")),
)


def _parse_imgsz(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image size must be H,W, for example 384,128")
    try:
        height, width = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("image dimensions must be integers") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("image dimensions must be positive")
    return height, width


def _parse_runtimes(value: str) -> tuple[str, ...]:
    runtimes = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    invalid = sorted(set(runtimes) - VALID_RUNTIMES)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown runtime(s): {', '.join(invalid)}; choose from {', '.join(DEFAULT_RUNTIMES)}"
        )
    if not runtimes:
        raise argparse.ArgumentTypeError("at least one runtime is required")
    return runtimes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        action="append",
        type=Path,
        help="PyTorch checkpoint to compare; repeat for multiple models (defaults to CSL and LMBN)",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="display label matching each --weights argument; repeat in the same order",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--imgsz", type=_parse_imgsz, default=(384, 128), metavar="H,W")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--runtimes",
        type=_parse_runtimes,
        default=DEFAULT_RUNTIMES,
        metavar="LIST",
        help="comma-separated runtimes: " + ",".join(DEFAULT_RUNTIMES),
    )
    parser.add_argument(
        "--export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export missing/stale static ONNX graphs before benchmarking",
    )
    parser.add_argument("--force-export", action="store_true", help="re-export ONNX even when it is current")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset used during export")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--coreml-model-format",
        choices=("NeuralNetwork", "MLProgram"),
        default="NeuralNetwork",
    )
    parser.add_argument(
        "--coreml-compute-units",
        choices=("ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly"),
        default="CPUAndGPU",
    )

    # Each runtime is isolated because a CoreML compiler/runtime failure can
    # terminate the process instead of raising a recoverable Python exception.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-runtime", choices=tuple(DEFAULT_RUNTIMES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-weight", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-onnx", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", help=argparse.SUPPRESS)
    return parser


def _resolve_models(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    if not args.weights:
        return DEFAULT_MODELS

    labels = args.label or [path.parent.name if path.stem == "best" else path.stem for path in args.weights]
    if len(labels) != len(args.weights):
        raise ValueError(f"received {len(args.weights)} weights but {len(labels)} labels")
    return tuple(ModelSpec(label, path) for label, path in zip(labels, args.weights))


def _onnx_input_shape(path: Path) -> tuple[int, ...] | None:
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        dims = model.graph.input[0].type.tensor_type.shape.dim
        values = tuple(int(dim.dim_value) for dim in dims)
        return values if all(values) else None
    except Exception:
        return None


def _onnx_is_current(model: ModelSpec, expected_shape: tuple[int, ...]) -> bool:
    if not model.onnx.is_file() or model.onnx.stat().st_mtime < model.weights.stat().st_mtime:
        return False
    return _onnx_input_shape(model.onnx) == expected_shape


def _export_model(model: ModelSpec, args: argparse.Namespace) -> None:
    expected_shape = (args.batch_size, 3, *args.imgsz)
    if not args.force_export and _onnx_is_current(model, expected_shape):
        print(f"ONNX current: {model.onnx}")
        return

    command = [
        sys.executable,
        "-m",
        "boxmot.engine.cli",
        "export",
        "--weights",
        str(model.weights),
        "--include",
        "onnx",
        "--batch-size",
        str(args.batch_size),
        "--imgsz",
        f"{args.imgsz[0]},{args.imgsz[1]}",
        "--device",
        "cpu",
        "--opset",
        str(args.opset),
        # The modern torch.export path is required by CSL's learned GeM
        # exponent; the export command selects it when verbose is enabled.
        "--verbose",
    ]
    print(f"Exporting {model.label}: {model.weights}", flush=True)
    subprocess.run(command, check=True)


def _synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _measure(
    forward: Callable[[], Any],
    *,
    synchronize: Callable[[], None],
    warmup: int,
    iterations: int,
) -> tuple[float, Any]:
    output: Any = None
    for _ in range(warmup):
        output = forward()
    synchronize()

    started = time.perf_counter()
    for _ in range(iterations):
        output = forward()
    synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations, output


def _benchmark_pytorch(args: argparse.Namespace, device_name: str) -> dict[str, Any]:
    import torch

    from boxmot.reid.core import ReID
    from boxmot.reid.exporters.base_exporter import as_inference_export_model

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available")

    device = torch.device(device_name)
    rng = np.random.default_rng(args.seed)
    input_array = rng.standard_normal((args.batch_size, 3, *args.imgsz), dtype=np.float32)

    created = time.perf_counter()
    reid = ReID(weights=args.worker_weight, device=device, half=False)
    model = as_inference_export_model(reid.model.model.eval())
    inputs = torch.from_numpy(input_array).to(device)
    session_create_ms = (time.perf_counter() - created) * 1000.0

    with torch.inference_mode():
        mean_ms, output = _measure(
            lambda: model(inputs),
            synchronize=lambda: _synchronize(device),
            warmup=args.warmup,
            iterations=args.iterations,
        )

    return _result(
        args,
        provider=f"PyTorch {device_name.upper()}",
        mean_ms=mean_ms,
        session_create_ms=session_create_ms,
        output_shape=list(output.shape),
    )


def _benchmark_onnx(args: argparse.Namespace, *, coreml: bool) -> dict[str, Any]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if coreml:
        if "CoreMLExecutionProvider" not in available:
            raise RuntimeError(f"CoreMLExecutionProvider is unavailable; installed providers: {available}")
        providers: list[Any] = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": args.coreml_model_format,
                    "MLComputeUnits": args.coreml_compute_units,
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "1",
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 2 if coreml else 3
    created = time.perf_counter()
    session = ort.InferenceSession(
        str(args.worker_onnx),
        sess_options=session_options,
        providers=providers,
    )
    session_create_ms = (time.perf_counter() - created) * 1000.0

    rng = np.random.default_rng(args.seed)
    inputs = rng.standard_normal((args.batch_size, 3, *args.imgsz), dtype=np.float32)
    feed = {session.get_inputs()[0].name: inputs}
    mean_ms, output = _measure(
        lambda: session.run(None, feed),
        synchronize=lambda: None,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    provider = "+".join(session.get_providers())
    if coreml:
        provider += f" ({args.coreml_model_format}, {args.coreml_compute_units})"

    return _result(
        args,
        provider=provider,
        mean_ms=mean_ms,
        session_create_ms=session_create_ms,
        output_shape=list(output[0].shape),
    )


def _result(
    args: argparse.Namespace,
    *,
    provider: str,
    mean_ms: float,
    session_create_ms: float,
    output_shape: list[int],
) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": args.worker_label,
        "weights": str(args.worker_weight),
        "runtime": args.worker_runtime,
        "provider": provider,
        "batch_size": args.batch_size,
        "imgsz": list(args.imgsz),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "session_create_ms": round(session_create_ms, 4),
        "mean_ms_batch": round(mean_ms, 4),
        "mean_ms_image": round(mean_ms / args.batch_size, 4),
        "images_per_second": round(args.batch_size * 1000.0 / mean_ms, 2),
        "output_shape": output_shape,
    }


def _run_worker(args: argparse.Namespace) -> int:
    try:
        if args.worker_runtime == "pytorch-cpu":
            result = _benchmark_pytorch(args, "cpu")
        elif args.worker_runtime == "pytorch-mps":
            result = _benchmark_pytorch(args, "mps")
        elif args.worker_runtime == "onnx-cpu":
            result = _benchmark_onnx(args, coreml=False)
        elif args.worker_runtime == "onnx-coreml":
            result = _benchmark_onnx(args, coreml=True)
        else:
            raise ValueError(f"unsupported worker runtime: {args.worker_runtime}")
    except Exception as exc:
        result = {
            "status": "error",
            "model": args.worker_label,
            "weights": str(args.worker_weight),
            "runtime": args.worker_runtime,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ok" else 1


def _worker_command(model: ModelSpec, runtime: str, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--worker",
        "--worker-runtime",
        runtime,
        "--worker-weight",
        str(model.weights.resolve()),
        "--worker-onnx",
        str(model.onnx.resolve()),
        "--worker-label",
        model.label,
        "--batch-size",
        str(args.batch_size),
        "--imgsz",
        f"{args.imgsz[0]},{args.imgsz[1]}",
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--coreml-model-format",
        args.coreml_model_format,
        "--coreml-compute-units",
        args.coreml_compute_units,
    ]


def _extract_worker_result(process: subprocess.CompletedProcess[str], model: ModelSpec, runtime: str) -> dict[str, Any]:
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line.removeprefix(RESULT_PREFIX))
            break
    else:
        detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "no worker result"
        result = {
            "status": "error",
            "model": model.label,
            "weights": str(model.weights),
            "runtime": runtime,
            "error": f"worker exited {process.returncode}: {detail}",
        }

    partition_match = re.search(
        r"number of partitions supported by CoreML: (\d+) number of nodes in the graph: (\d+) "
        r"number of nodes supported by CoreML: (\d+)",
        process.stderr,
    )
    if partition_match:
        result["coreml_partitions"] = int(partition_match.group(1))
        result["onnx_nodes"] = int(partition_match.group(2))
        result["coreml_supported_nodes"] = int(partition_match.group(3))
    return result


def _run_benchmarks(models: tuple[ModelSpec, ...], args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for runtime in args.runtimes:
        for model in models:
            if runtime.startswith("onnx") and not model.onnx.is_file():
                results.append(
                    {
                        "status": "error",
                        "model": model.label,
                        "weights": str(model.weights),
                        "runtime": runtime,
                        "error": f"missing ONNX graph: {model.onnx}",
                    }
                )
                continue

            print(f"Benchmarking {model.label} on {runtime}...", flush=True)
            process = subprocess.run(
                _worker_command(model, runtime, args),
                capture_output=True,
                text=True,
            )
            result = _extract_worker_result(process, model, runtime)
            results.append(result)
            if result["status"] == "ok":
                print(
                    f"  {result['mean_ms_image']:.4f} ms/image "
                    f"({result['images_per_second']:.2f} images/s)"
                )
            else:
                print(f"  ERROR: {result['error']}")
    return results


def _print_results(results: list[dict[str, Any]]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="ReID inference speed (model forward only)")
    table.add_column("Model", style="bold")
    table.add_column("Runtime")
    table.add_column("Provider")
    table.add_column("ms/batch", justify="right")
    table.add_column("ms/image", justify="right")
    table.add_column("images/s", justify="right")
    table.add_column("Status")
    for result in results:
        if result["status"] == "ok":
            table.add_row(
                result["model"],
                result["runtime"],
                result["provider"],
                f"{result['mean_ms_batch']:.4f}",
                f"{result['mean_ms_image']:.4f}",
                f"{result['images_per_second']:.2f}",
                "OK",
            )
        else:
            table.add_row(
                result["model"],
                result["runtime"],
                "-",
                "-",
                "-",
                "-",
                result["error"],
            )
    Console().print(table)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("batch-size and iterations must be positive; warmup must be non-negative")

    if args.worker:
        return _run_worker(args)

    try:
        models = _resolve_models(args)
    except ValueError as exc:
        parser.error(str(exc))

    missing = [str(model.weights) for model in models if not model.weights.is_file()]
    if missing:
        parser.error("missing weights: " + ", ".join(missing))

    if args.export:
        for model in models:
            _export_model(model, args)

    print(
        f"\nInput: batch={args.batch_size}, shape=3x{args.imgsz[0]}x{args.imgsz[1]}, "
        f"warmup={args.warmup}, iterations={args.iterations}, FP32, no TTA"
    )
    if "onnx-coreml" in args.runtimes:
        print(
            "Note: ONNX Runtime has no MPS provider; onnx-coreml uses CoreML "
            f"{args.coreml_model_format}/{args.coreml_compute_units} with CPU fallback."
        )

    results = _run_benchmarks(models, args)
    _print_results(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved JSON results to {args.output}")
    return 0 if any(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
