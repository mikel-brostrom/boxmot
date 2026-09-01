"""Compare ReID inference latency across PyTorch, native Core ML and ONNX.

The defaults reproduce the CSL-TinyViT-11M FPN versus LMBN-n comparison::

    uv run --no-sync python -m tests.performance.reid.benchmark_inference

The benchmark exports missing/stale artifacts, then measures PyTorch CPU/MPS,
native Core ML MLProgram, and ONNX Runtime CPU. ONNX Runtime's legacy Core ML
execution-provider path remains available as an explicitly guarded diagnostic,
but native MLProgram is the supported Apple deployment path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from boxmot.reid.exporters.process import (
    ProcessMemoryGuard,
    run_limited,
    terminate_process_tree,
)

RESULT_PREFIX = "BOXMOT_REID_BENCHMARK_RESULT="
PROGRESS_PREFIX = "BOXMOT_REID_BENCHMARK_PROGRESS="
MODULE_NAME = "tests.performance.reid.benchmark_inference"
VALID_RUNTIMES = frozenset(("pytorch-cpu", "pytorch-mps", "onnx-cpu", "coreml", "onnx-coreml"))
DEFAULT_RUNTIMES = (
    ("pytorch-cpu", "pytorch-mps", "onnx-cpu", "coreml") if sys.platform == "darwin" else ("pytorch-cpu", "onnx-cpu")
)


@dataclass(frozen=True)
class ModelSpec:
    """One model participating in the benchmark."""

    label: str
    weights: Path

    @property
    def onnx(self) -> Path:
        return self.weights.with_suffix(".onnx")

    @property
    def coreml(self) -> Path:
        return self.weights.parent / f"{self.weights.stem}_coreml_model"


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
            f"unknown runtime(s): {', '.join(invalid)}; choose from {', '.join(sorted(VALID_RUNTIMES))}"
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
        help="export missing/stale ONNX/Core ML artifacts before benchmarking",
    )
    parser.add_argument("--force-export", action="store_true", help="re-export artifacts even when current")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset used during export")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--coreml-model-format",
        choices=("NeuralNetwork", "MLProgram"),
        default="MLProgram",
        help="legacy ONNX Core ML EP model format; native coreml always uses MLProgram",
    )
    parser.add_argument(
        "--coreml-compute-units",
        choices=("ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly"),
        default="CPUAndGPU",
    )
    parser.add_argument(
        "--coreml-batch-buckets",
        default="1,8,16,32",
        help="static native MLProgram batch buckets (maximum 32)",
    )
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=600.0,
        help="maximum seconds for each export or benchmark worker",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=16.0,
        help="aggregate resident-RAM limit for each export or benchmark worker",
    )
    parser.add_argument(
        "--allow-unsafe-onnx-coreml",
        action="store_true",
        help="allow legacy ONNX Core ML EP at batch sizes above 8 (can consume extreme RAM)",
    )

    # Each runtime is isolated because a CoreML compiler/runtime failure can
    # terminate the process instead of raising a recoverable Python exception.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-runtime", choices=tuple(sorted(VALID_RUNTIMES)), help=argparse.SUPPRESS)
    parser.add_argument("--worker-weight", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-onnx", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-coreml", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", help=argparse.SUPPRESS)
    return parser


def _resolve_models(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    if not args.weights:
        return DEFAULT_MODELS

    labels = args.label or [path.parent.name if path.stem == "best" else path.stem for path in args.weights]
    if len(labels) != len(args.weights):
        raise ValueError(f"received {len(args.weights)} weights but {len(labels)} labels")
    return tuple(ModelSpec(label, path) for label, path in zip(labels, args.weights))


def _onnx_is_current(
    model: ModelSpec,
    expected_shape: tuple[int, ...],
    *,
    opset: int = 18,
) -> bool:
    """Validate benchmark exports against source, code, dependencies, and settings."""
    from boxmot.reid.core.artifacts import export_content_fingerprint
    from boxmot.reid.exporters.onnx_exporter import (
        _onnx_export_contract,
        _onnx_export_is_current,
    )

    contract = _onnx_export_contract(
        SimpleNamespace(shape=expected_shape),
        opset=opset,
        dynamic=False,
        half=False,
        simplify=False,
    )
    expected_fingerprint = export_content_fingerprint(model.weights, contract) if model.weights.is_file() else None
    return _onnx_export_is_current(
        model.weights,
        model.onnx,
        dynamic=False,
        expected_fingerprint=expected_fingerprint,
    )


def _coreml_is_current(model: ModelSpec, args: argparse.Namespace) -> bool:
    manifest_path = model.coreml / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_mtime < model.weights.stat().st_mtime:
        return False
    try:
        from boxmot.reid.core.artifacts import file_sha256

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_buckets = sorted(
            {int(value.strip()) for value in args.coreml_batch_buckets.split(",") if value.strip()}
        )
        source_sha256 = manifest.get("source_sha256")
        source_matches = bool(source_sha256) and file_sha256(model.weights) == source_sha256
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        source_matches
        and manifest.get("format") == "coreml_mlprogram"
        and manifest.get("batch_buckets") == expected_buckets
        and manifest.get("input_shape") == [3, *args.imgsz]
    )


def _run_export_command(command: list[str], args: argparse.Namespace) -> None:
    process = run_limited(
        command,
        timeout_s=args.worker_timeout,
        max_memory_gb=args.max_memory_gb,
    )
    if process.returncode != 0:
        lines = process.stdout.strip().splitlines()
        detail = lines[-1] if lines else f"exit code {process.returncode}"
        raise RuntimeError(f"export failed: {detail}")


def _export_onnx_model(model: ModelSpec, args: argparse.Namespace) -> None:
    expected_shape = (args.batch_size, 3, *args.imgsz)
    if not args.force_export and _onnx_is_current(model, expected_shape, opset=args.opset):
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
    _run_export_command(command, args)


def _export_coreml_model(model: ModelSpec, args: argparse.Namespace) -> None:
    if not args.force_export and _coreml_is_current(model, args):
        print(f"Core ML current: {model.coreml}")
        return

    command = [
        sys.executable,
        "-m",
        "boxmot.engine.cli",
        "export",
        "--weights",
        str(model.weights),
        "--include",
        "coreml",
        "--batch-size",
        "1",
        "--imgsz",
        f"{args.imgsz[0]},{args.imgsz[1]}",
        "--device",
        "cpu",
        "--coreml-batch-buckets",
        args.coreml_batch_buckets,
        "--coreml-compute-units",
        args.coreml_compute_units,
        "--coreml-timeout",
        str(args.worker_timeout),
        "--coreml-max-memory-gb",
        str(args.max_memory_gb),
        "--verbose",
    ]
    print(f"Exporting native Core ML {model.label}: {model.weights}", flush=True)
    _run_export_command(command, args)


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
    def report(phase: str, current: int, total: int) -> None:
        print(
            PROGRESS_PREFIX + json.dumps({"phase": phase, "current": current, "total": total}, sort_keys=True),
            flush=True,
        )

    output: Any = None
    report("warmup", 0, warmup)
    for _ in range(warmup):
        output = forward()
    synchronize()
    report("warmup", warmup, warmup)

    report("measure", 0, iterations)
    started = time.perf_counter()
    for _ in range(iterations):
        output = forward()
    synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / iterations
    report("measure", iterations, iterations)
    return elapsed_ms, output


def _benchmark_pytorch(args: argparse.Namespace, device_name: str) -> dict[str, Any]:
    import torch

    from boxmot.reid import ReID
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


def _benchmark_coreml(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from boxmot.reid import ReID

    rng = np.random.default_rng(args.seed)
    input_array = rng.standard_normal((args.batch_size, 3, *args.imgsz), dtype=np.float32)

    previous_compute_units = os.environ.get("BOXMOT_COREML_COMPUTE_UNITS")
    os.environ["BOXMOT_COREML_COMPUTE_UNITS"] = args.coreml_compute_units
    created = time.perf_counter()
    try:
        reid = ReID(weights=args.worker_coreml, device="cpu", half=False)
    finally:
        if previous_compute_units is None:
            os.environ.pop("BOXMOT_COREML_COMPUTE_UNITS", None)
        else:
            os.environ["BOXMOT_COREML_COMPUTE_UNITS"] = previous_compute_units
    model = reid.model
    inputs = torch.from_numpy(input_array)
    session_create_ms = (time.perf_counter() - created) * 1000.0
    compute_units = str(getattr(model, "_compute_units_name", args.coreml_compute_units))

    mean_ms, output = _measure(
        lambda: model.forward(inputs),
        synchronize=lambda: None,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    return _result(
        args,
        provider=f"Core ML MLProgram ({compute_units})",
        mean_ms=mean_ms,
        session_create_ms=session_create_ms,
        output_shape=list(output.shape),
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
        elif args.worker_runtime == "coreml":
            result = _benchmark_coreml(args)
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
        "--worker-coreml",
        str(model.coreml.resolve()),
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
        diagnostics = "\n".join(part for part in (process.stdout, process.stderr) if part).strip()
        detail = diagnostics.splitlines()[-1] if diagnostics else "no worker result"
        result = {
            "status": "error",
            "model": model.label,
            "weights": str(model.weights),
            "runtime": runtime,
            "error": f"worker exited {process.returncode}: {detail}",
        }

    diagnostics = "\n".join(part for part in (process.stdout, process.stderr) if part)
    partition_match = re.search(
        r"number of partitions supported by CoreML: (\d+) number of nodes in the graph: (\d+) "
        r"number of nodes supported by CoreML: (\d+)",
        diagnostics,
    )
    if partition_match:
        result["coreml_partitions"] = int(partition_match.group(1))
        result["onnx_nodes"] = int(partition_match.group(2))
        result["coreml_supported_nodes"] = int(partition_match.group(3))
    return result


def _run_worker_with_progress(
    command: list[str],
    on_progress: Callable[[dict[str, Any]], None],
    *,
    timeout_s: float,
    max_memory_gb: float,
) -> subprocess.CompletedProcess[str]:
    """Run one isolated worker while forwarding its structured progress events."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_lines: list[str] = []
    timed_out = threading.Event()
    memory_guard = ProcessMemoryGuard(process, max_memory_gb).start()

    def terminate_on_timeout() -> None:
        timed_out.set()
        terminate_process_tree(process)

    timer = threading.Timer(timeout_s, terminate_on_timeout) if timeout_s > 0 else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.strip()
            if stripped.startswith(PROGRESS_PREFIX):
                try:
                    on_progress(json.loads(stripped.removeprefix(PROGRESS_PREFIX)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    output_lines.append(line)
                continue
            output_lines.append(line)
        returncode = process.wait()
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        if timer is not None:
            timer.cancel()
        memory_guard.stop()
    if timed_out.is_set():
        output_lines.append(f"\nworker exceeded {timeout_s:.0f}s timeout and was terminated\n")
    if memory_guard.exceeded.is_set():
        peak_gb = memory_guard.peak_rss_bytes / 1024**3
        output_lines.append(
            f"\nworker exceeded {max_memory_gb:.1f} GB RAM (observed at least {peak_gb:.1f} GB) and was terminated\n"
        )
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout="".join(output_lines),
        stderr="",
    )


def _run_benchmarks(models: tuple[ModelSpec, ...], args: argparse.Namespace) -> list[dict[str, Any]]:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

    results: list[dict[str, Any]] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    with progress:
        for runtime in args.runtimes:
            for model in models:
                description = f"{model.label} on {runtime}"
                task_id = progress.add_task(f"{description} · starting", total=args.warmup + args.iterations)
                if runtime.startswith("onnx") and not model.onnx.is_file():
                    result = {
                        "status": "error",
                        "model": model.label,
                        "weights": str(model.weights),
                        "runtime": runtime,
                        "error": f"missing ONNX graph: {model.onnx}",
                    }
                    results.append(result)
                    progress.update(
                        task_id,
                        completed=args.warmup + args.iterations,
                        description=f"{description} · error",
                    )
                    progress.console.print(f"  ERROR: {result['error']}")
                    continue
                if runtime == "coreml" and not model.coreml.is_dir():
                    result = {
                        "status": "error",
                        "model": model.label,
                        "weights": str(model.weights),
                        "runtime": runtime,
                        "error": f"missing Core ML bundle: {model.coreml}",
                    }
                    results.append(result)
                    progress.update(
                        task_id,
                        completed=args.warmup + args.iterations,
                        description=f"{description} · error",
                    )
                    progress.console.print(f"  ERROR: {result['error']}")
                    continue

                def update_progress(event: dict[str, Any]) -> None:
                    phase = str(event.get("phase", "measure"))
                    current = int(event.get("current", 0))
                    total = int(event.get("total", 0))
                    if phase == "warmup":
                        completed = min(current, args.warmup)
                        phase_label = "warmup"
                    else:
                        completed = args.warmup + min(current, args.iterations)
                        phase_label = "measuring"
                    progress.update(
                        task_id,
                        completed=completed,
                        description=f"{description} · {phase_label} {current}/{total}",
                    )

                process = _run_worker_with_progress(
                    _worker_command(model, runtime, args),
                    update_progress,
                    timeout_s=args.worker_timeout,
                    max_memory_gb=args.max_memory_gb,
                )
                result = _extract_worker_result(process, model, runtime)
                results.append(result)
                progress.update(
                    task_id,
                    completed=args.warmup + args.iterations,
                    description=f"{description} · {'done' if result['status'] == 'ok' else 'error'}",
                )
                if result["status"] == "ok":
                    progress.console.print(
                        f"  {result['mean_ms_image']:.4f} ms/image ({result['images_per_second']:.2f} images/s)"
                    )
                else:
                    progress.console.print(f"  ERROR: {result['error']}")
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("batch-size and iterations must be positive; warmup must be non-negative")
    if args.worker_timeout <= 0 or args.max_memory_gb <= 0:
        parser.error("worker-timeout and max-memory-gb must be positive")

    from boxmot.reid.exporters.coreml_exporter import parse_coreml_buckets

    try:
        buckets = parse_coreml_buckets(args.coreml_batch_buckets)
    except ValueError as exc:
        parser.error(str(exc))
    args.coreml_batch_buckets = ",".join(str(value) for value in buckets)

    if args.worker:
        return _run_worker(args)

    if "onnx-coreml" in args.runtimes and args.batch_size > 8 and not args.allow_unsafe_onnx_coreml:
        parser.error(
            "legacy onnx-coreml is blocked above batch 8 because Core ML EP graph "
            "compilation can consume extreme RAM. Use native 'coreml', or pass "
            "--allow-unsafe-onnx-coreml to accept that risk."
        )

    try:
        models = _resolve_models(args)
    except ValueError as exc:
        parser.error(str(exc))

    missing = [str(model.weights) for model in models if not model.weights.is_file()]
    if missing:
        parser.error("missing weights: " + ", ".join(missing))

    if args.export:
        try:
            for model in models:
                if any(runtime.startswith("onnx") for runtime in args.runtimes):
                    _export_onnx_model(model, args)
                if "coreml" in args.runtimes:
                    _export_coreml_model(model, args)
        except (MemoryError, RuntimeError, subprocess.TimeoutExpired) as exc:
            parser.error(str(exc))

    print(
        f"\nInput: batch={args.batch_size}, shape=3x{args.imgsz[0]}x{args.imgsz[1]}, "
        f"warmup={args.warmup}, iterations={args.iterations}, FP32, no TTA"
    )
    if "onnx-coreml" in args.runtimes:
        print(
            "Warning: onnx-coreml is the legacy ONNX Runtime execution-provider path. "
            "It is not MPS and may compile unsupported graph fragments with CPU fallback."
        )
    if "coreml" in args.runtimes:
        print(
            "Native Core ML uses FP16 MLProgram static buckets "
            f"[{args.coreml_batch_buckets}] with {args.coreml_compute_units}; "
            "larger batches are chunked and compiled packages are loaded lazily."
        )

    results = _run_benchmarks(models, args)
    _print_results(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved JSON results to {args.output}")
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
