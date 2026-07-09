"""Engine entry point for ReID model export."""

import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from boxmot.engine.workflows.results import ExportResult
from boxmot.reid.core import ReID, export_formats
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.utils import WEIGHTS
from boxmot.utils.rich.reporters.export import ExportWorkflowReporter
from boxmot.utils.torch_utils import select_device

__all__ = [
    "ExportWorkflowReporter",
    "main",
    "run_export",
]


@dataclass(frozen=True)
class ExportTask:
    exporter_class: type[BaseExporter]
    kwargs: dict[str, Any]
    report: bool = True

    def export(self):
        exporter = self.exporter_class(**self.kwargs)
        return exporter.export()


def validate_export_formats(include):
    available_formats = tuple(export_formats()["Argument"][1:])
    include_lower = [fmt.lower() for fmt in include]
    flags = [fmt in include_lower for fmt in available_formats]
    if sum(flags) != len(include_lower):
        raise AssertionError(f"ERROR: Invalid --include {include}, valid arguments are {available_formats}")
    return tuple(flags)


@contextmanager
def _suppress_export_noise(enabled: bool):
    if not enabled:
        yield
        return

    boxmot_logger = logging.getLogger("boxmot")
    previous_boxmot_level = boxmot_logger.level
    boxmot_logger.setLevel(logging.ERROR)

    # Several export backends (torch.export, openvino, onnx) emit warnings
    # through their own Python loggers that write directly to ``sys.stderr``.
    # When the Rich workflow Live region is active on stderr, those stray
    # writes corrupt the panel and cause it to be redrawn / duplicated.
    noisy_logger_names = (
        "torch",
        "torch._dynamo",
        "torch._inductor",
        "torch.export",
        "torch._export",
        "torch.onnx",
        "openvino",
        "openvino.tools",
        "onnx",
        "onnxruntime",
        "nncf",
    )
    noisy_loggers = []
    for name in noisy_logger_names:
        target = logging.getLogger(name)
        noisy_loggers.append((target, target.level, target.propagate))
        target.setLevel(logging.ERROR)
        target.propagate = False

    try:
        # NOTE: ``sys.stdout`` is intentionally NOT redirected here. Rich's
        # ``Console`` reads ``sys.stdout`` lazily, so any swap would silently
        # divert the workflow Live region's writes (including in-panel
        # progress bars for downloads) into a discarded buffer. The noisy
        # Python loggers above already cover the bulk of the unwanted
        # output from the export backends.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*treespec.*")
            yield
    finally:
        boxmot_logger.setLevel(previous_boxmot_level)
        for target, level, propagate in noisy_loggers:
            target.setLevel(level)
            target.propagate = propagate


def setup_model(args):
    args.device = select_device(args.device)
    include = tuple(str(fmt).lower() for fmt in (getattr(args, "include", ()) or ()))
    cpu_fp16_graph_export = bool(args.half and args.device.type == "cpu" and "engine" not in include)
    if args.half and args.device.type == "cpu" and not cpu_fp16_graph_export:
        raise AssertionError("--half TensorRT export requires GPU, use --device 0")

    backend_half = bool(args.half and not cpu_fp16_graph_export)
    reid = ReID(weights=args.weights, device=args.device, half=backend_half)
    model_name = ReIDModelRegistry.get_model_name(args.weights)
    model = reid.model.model.eval()

    if args.optimize and args.device.type != "cpu":
        raise AssertionError("--optimize not compatible with CUDA devices, use --device cpu")

    if "vehicleid" in args.weights.name or "veri" in args.weights.name:
        args.imgsz = (256, 256)
    elif "lmbn" in model_name or "csl_tinyvit" in model_name or "mobilenetv4" in model_name:
        args.imgsz = (384, 128)
    elif "hacnn" in model_name:
        args.imgsz = (160, 64)
    else:
        args.imgsz = (256, 128)

    if backend_half:
        model = model.half()

    first_param = next(model.parameters(), None)
    if first_param is not None:
        model_dtype = first_param.dtype
    else:
        first_buffer = next(model.buffers(), None)
        model_dtype = first_buffer.dtype if first_buffer is not None else torch.float32
    dummy_input = torch.empty(
        args.batch_size,
        3,
        args.imgsz[0],
        args.imgsz[1],
        device=args.device,
        dtype=model_dtype,
    )
    for _ in range(2):
        _ = model(dummy_input)

    return model, dummy_input


def create_export_tasks(args, model, dummy_input):
    torchscript_flag, onnx_flag, openvino_flag, engine_flag, tflite_flag = validate_export_formats(args.include)
    tasks = {}
    common_kwargs = {
        "model": model,
        "im": dummy_input,
        "file": args.weights,
    }
    onnx_kwargs = {
        **common_kwargs,
        "opset": args.opset,
        "dynamic": args.dynamic,
        "half": args.half,
        "simplify": args.simplify,
        "verbose": args.verbose,
    }

    if torchscript_flag:
        from boxmot.reid.exporters.torchscript_exporter import TorchScriptExporter

        tasks["torchscript"] = ExportTask(
            TorchScriptExporter,
            {
                **common_kwargs,
                "optimize": args.optimize,
                "verbose": args.verbose,
            },
        )

    if onnx_flag or engine_flag or openvino_flag:
        from boxmot.reid.exporters.onnx_exporter import ONNXExporter

        tasks["onnx"] = ExportTask(
            ONNXExporter,
            dict(onnx_kwargs),
            report=onnx_flag,
        )

    if engine_flag:
        from boxmot.reid.exporters.tensorrt_exporter import EngineExporter

        tasks["engine"] = ExportTask(
            EngineExporter,
            {
                **onnx_kwargs,
                "workspace": args.workspace,
            },
        )

    if tflite_flag:
        from boxmot.reid.exporters.tflite_exporter import TFLiteExporter

        tasks["tflite"] = ExportTask(
            TFLiteExporter,
            {
                **common_kwargs,
                "opset": args.opset,
                "dynamic": args.dynamic,
                "half": args.half,
                "simplify": args.simplify,
                "quantize": getattr(args, "tflite_quantize", "none"),
                "calibration_data": getattr(args, "tflite_calibration_data", None),
                "calibration_samples": getattr(args, "tflite_calibration_samples", 256),
                "calibration_preprocess": getattr(args, "tflite_calibration_preprocess", "resize"),
                "calibration_seed": getattr(args, "tflite_calibration_seed", 0),
                "calibration_update": getattr(args, "tflite_calibration_update", "minmax"),
                "static_activation_bits": getattr(args, "tflite_static_activation_bits", 16),
                "verbose": args.verbose,
            },
        )

    if openvino_flag:
        from boxmot.reid.exporters.openvino_exporter import OpenVINOExporter

        tasks["openvino"] = ExportTask(
            OpenVINOExporter,
            dict(onnx_kwargs),
        )

    return tasks


def perform_exports(export_tasks):
    exported_files = {}
    for fmt, task in export_tasks.items():
        exported = task.export()
        if task.report:
            exported_files[fmt] = exported
    return exported_files


def _prepare_export(args):
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    args.weights = _resolve_export_weights(args.weights)

    with _suppress_export_noise(not args.verbose):
        model, dummy_input = setup_model(args)
    return model, dummy_input


def _resolve_export_weights(weights: str | Path) -> Path:
    """Keep explicit checkpoint paths; otherwise resolve bare names in WEIGHTS."""
    path = Path(weights)
    if path.exists():
        return path
    return WEIGHTS / path.name


def _execute_export(args, model, dummy_input):
    export_tasks = create_export_tasks(args, model, dummy_input)
    with _suppress_export_noise(not args.verbose):
        exported_files = perform_exports(export_tasks)
    return exported_files


def run_export(args) -> ExportResult:
    model, dummy_input = _prepare_export(args)
    exported_files = _execute_export(args, model, dummy_input)
    parity_report: dict[str, dict[str, Any]] = {}
    if exported_files:
        with _suppress_export_noise(not args.verbose):
            parity_report = _verify_export_parity(args, model, dummy_input, exported_files)
    return ExportResult(
        weights=args.weights,
        files=exported_files,
        parity=parity_report,
        half=bool(getattr(args, "half", False)),
    )


def _verify_export_parity(
    args,
    model,
    dummy_input,
    exported_files: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Compare each exported model's output to the original PyTorch model.

    Two complementary metrics are reported per format:

    1. **Strict numerical parity** (``parity_ok``): element-wise
       ``np.allclose`` against the source PyTorch model using the standard
       cross-framework tolerances

       - FP32 export → ``rtol=1e-3, atol=1e-5``
       - FP16 export → ``rtol=1e-2, atol=1e-3``

       Failing this typically points to a real conversion bug (wrong
       layout, missing op fusion, precision truncation).

    2. **Embedding parity** (``embedding_ok``): mean cosine similarity
       ≥ ``0.999`` between exported and reference output. ReID models
       are consumed by cosine matching, so two embeddings that align in
       direction are functionally equivalent even if their absolute
       values drift slightly (e.g. from ``ov.convert_model`` graph
       fusions).

    The combined ``ok`` flag is ``True`` when either metric passes:
    strict parity is preferred, but embedding parity is sufficient for
    correctness of the downstream tracker.

    Returns a mapping ``{format: {"max_abs": float, "mean_abs": float,
    "cosine": float, "parity_ok": bool, "embedding_ok": bool,
    "ok": bool}}``. Failures are logged but do not raise — exporting is
    the primary goal, parity is informational.
    """
    import numpy as np

    if bool(getattr(args, "half", False)):
        rtol, atol = 1e-2, 1e-3
    else:
        rtol, atol = 1e-3, 1e-5

    # ``dummy_input`` was created via ``torch.empty`` for shape inference, so
    # its values are uninitialised and can be NaN/inf — propagating those
    # through the model gives spurious parity failures. Use a deterministic
    # random tensor of the same shape/dtype/device for the comparison.
    torch.manual_seed(0)
    sample = torch.rand_like(dummy_input.float()).to(dtype=dummy_input.dtype, device=dummy_input.device)

    with torch.inference_mode():
        ref_output = model(sample)
    if isinstance(ref_output, (tuple, list)):
        ref_output = ref_output[0]
    ref_np = ref_output.detach().to(torch.float32).cpu().numpy()

    cpu_input = sample.detach().to("cpu", dtype=torch.float32)

    report: dict[str, dict[str, float]] = {}
    for fmt, fpath in exported_files.items():
        if fmt == "engine":
            # TensorRT requires the matching CUDA runtime. Skip parity here.
            continue
        try:
            if fmt == "tflite":
                out = _run_tflite_for_parity(fpath, cpu_input)
            else:
                # OpenVINO exporters return the .xml path, but ReID's suffix
                # check expects the ``_openvino_model`` directory. Pass the
                # directory so the parity load doesn't emit a stray warning.
                load_path = fpath
                if fmt == "openvino":
                    parent = Path(fpath).parent
                    if parent.name.endswith("_openvino_model"):
                        load_path = str(parent)
                reid = ReID(weights=load_path, device="cpu", half=False)
                out = reid.model.forward(cpu_input)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if hasattr(out, "detach"):
                out_np = out.detach().to(torch.float32).cpu().numpy()
            else:
                out_np = np.asarray(out, dtype=np.float32)

            if out_np.shape != ref_np.shape:
                report[fmt] = {
                    "max_abs": float("nan"),
                    "mean_abs": float("nan"),
                    "cosine": float("nan"),
                    "parity_ok": False,
                    "embedding_ok": False,
                    "ok": False,
                    "error": f"shape mismatch: {out_np.shape} vs {ref_np.shape}",
                }
                continue

            diff = np.abs(out_np - ref_np)
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            parity_ok = bool(np.allclose(out_np, ref_np, rtol=rtol, atol=atol))

            # Embedding-aware metric: ReID features are consumed by
            # cosine similarity, so two embeddings that point in the
            # same direction are functionally equivalent. Flatten any
            # spatial dims and average the per-sample cosine.
            ref_flat = ref_np.reshape(ref_np.shape[0], -1)
            out_flat = out_np.reshape(out_np.shape[0], -1)
            denom = np.linalg.norm(ref_flat, axis=1) * np.linalg.norm(out_flat, axis=1)
            denom = np.where(denom == 0, 1.0, denom)
            cosine = float(((ref_flat * out_flat).sum(axis=1) / denom).mean())
            embedding_ok = cosine >= 0.999

            report[fmt] = {
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "cosine": cosine,
                "parity_ok": parity_ok,
                "embedding_ok": embedding_ok,
                "ok": parity_ok or embedding_ok,
            }
        except Exception as exc:  # pragma: no cover - defensive
            report[fmt] = {
                "max_abs": float("nan"),
                "mean_abs": float("nan"),
                "cosine": float("nan"),
                "parity_ok": False,
                "embedding_ok": False,
                "ok": False,
                "error": str(exc),
            }

    return report


def _run_tflite_for_parity(fpath: str | Path, cpu_input: torch.Tensor):
    litert = import_module("ai_edge_litert.interpreter")
    from boxmot.reid.backends.tflite_backend import TFLiteBackend

    interpreter = litert.Interpreter(model_path=str(fpath))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details or not output_details:
        raise RuntimeError("TFLite model is missing input or output tensors.")

    input_detail = input_details[0]
    input_array = cpu_input.detach().cpu().numpy().astype("float32", copy=False)
    expected_shape = tuple(int(dim) for dim in input_detail["shape"])
    input_array = _match_tflite_input_layout(input_array, expected_shape)

    if tuple(input_array.shape) != expected_shape:
        interpreter.resize_tensor_input(input_detail["index"], list(input_array.shape))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()

    input_array = TFLiteBackend._quantize_input(input_array, input_detail)
    interpreter.set_tensor(input_detail["index"], input_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return TFLiteBackend._dequantize_output(output, output_details[0])


def _match_tflite_input_layout(input_array, expected_shape: tuple[int, ...]):
    if input_array.ndim == 4 and len(expected_shape) == 4:
        nchw_shape = tuple(input_array.shape)
        nhwc_shape = (nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1])
        if expected_shape == nhwc_shape:
            return input_array.transpose(0, 2, 3, 1)
    return input_array


def main(args):
    Export = ExportWorkflowReporter
    pipeline = Export(args).pipeline()
    start_time = time.time()
    with pipeline:
        pipeline.update("Loading ReID model...")
        model, dummy_input = _prepare_export(args)

        output = model(dummy_input)
        output_tensor = output[0] if isinstance(output, tuple) else output
        output_shape = tuple(output_tensor.shape)
        pipeline.update(
            (
                f"Input shape:  {tuple(dummy_input.shape)}\n"
                f"Output shape: {output_shape} "
                f"({BaseExporter.file_size(args.weights):.1f} MB)"
            ),
        )
        pipeline.advance("Exporting model...")

        formats = list(getattr(args, "include", []) or [])
        pipeline.update(
            f"Exporting to {len(formats)} format(s): {', '.join(formats) if formats else 'none'}",
        )
        exported_files = _execute_export(args, model, dummy_input)
        parity_report: dict[str, dict[str, Any]] = {}
        if exported_files:
            with _suppress_export_noise(not args.verbose):
                parity_report = _verify_export_parity(args, model, dummy_input, exported_files)
        result = ExportResult(
            weights=args.weights,
            files=exported_files,
            parity=parity_report,
            half=bool(getattr(args, "half", False)),
        )

        elapsed_time = time.time() - start_time
        if result.files:
            lines = [
                f"Time: {elapsed_time:.1f}s",
                f"Saved to: {args.weights.parent.resolve()}",
                "",
                "Files:",
            ]
            for fmt, fpath in result.files.items():
                stats = parity_report.get(fmt)
                fname = Path(fpath).name
                if stats is None:
                    suffix = " — parity: skipped"
                elif "error" in stats:
                    suffix = f" — parity: error ({stats['error']})"
                else:
                    parity_str = "OK" if stats.get("parity_ok") else "MISMATCH"
                    embed_str = "OK" if stats.get("embedding_ok") else "MISMATCH"
                    suffix = (
                        f" — parity {parity_str} (maxΔ={stats['max_abs']:.1e}), "
                        f"embedding {embed_str} (cos={stats['cosine']:.4f})"
                    )
                lines.append(f"  • {fmt}: {fname}{suffix}")
            lines.append("")
            verdict = "acceptable" if result.parity_ok else "out of tolerance"
            lines.append(f"Overall parity: {verdict}")
            lines.append("")
            lines.append("Visualize: https://netron.app")
            pipeline.update("\n".join(lines))
        else:
            pipeline.update(f"Export complete in {elapsed_time:.1f}s")
        pipeline.finish()
        return result


if __name__ == "__main__":
    main()
