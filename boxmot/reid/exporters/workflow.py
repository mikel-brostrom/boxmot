"""Reusable workflow for ReID model export."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from boxmot.reid.core import export_formats
from boxmot.reid.core.formats import ReIDFormat, resolve_export_formats
from boxmot.reid.core.runtime import ReID
from boxmot.reid.exporters.config import resolve_export_weights
from boxmot.reid.exporters.model_setup import prepare_export_model
from boxmot.reid.exporters.registry import get_exporter_class
from boxmot.utils import WEIGHTS

if TYPE_CHECKING:
    from boxmot.reid.exporters.backends.base import BaseExporter

@dataclass(frozen=True)
class ExportTask:
    exporter_class: type[BaseExporter]
    kwargs: dict[str, Any]
    report: bool = True

    def export(self):
        exporter = self.exporter_class(**self.kwargs)
        return exporter.export()


@dataclass(slots=True)
class ReIDExportResult:
    """Artifacts and parity information produced by a ReID export."""

    weights: Path
    files: dict[str, Any]
    parity: dict[str, dict[str, Any]] = field(default_factory=dict)
    half: bool = False

    @property
    def parity_ok(self) -> bool:
        """Return whether every checked format is within tolerance."""

        return all(stats.get("ok", False) for stats in self.parity.values())

    @property
    def embedding_weights(self) -> Path:
        """Return the preferred exported artifact for embedding inference."""

        if "onnx" in self.files:
            return Path(self.files["onnx"])
        if "coreml" in self.files:
            return Path(self.files["coreml"])
        return Path(self.weights)


def validate_export_formats(include):
    try:
        return resolve_export_formats(include)
    except ValueError as exc:
        available_formats = tuple(export_formats()["Argument"][1:])
        raise AssertionError(
            f"ERROR: Invalid --include {include}, valid arguments are {available_formats}"
        ) from exc


@contextmanager
def suppress_export_noise(enabled: bool):
    """Temporarily quiet third-party exporter logs and compatibility warnings."""

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


def create_export_tasks(args, model, dummy_input):
    selected_formats: tuple[ReIDFormat, ...] = validate_export_formats(args.include)
    selected_ids = {format_.id for format_ in selected_formats}
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

    if "torchscript" in selected_ids:
        tasks["torchscript"] = ExportTask(
            get_exporter_class("torchscript"),
            {
                **common_kwargs,
                "optimize": args.optimize,
                "verbose": args.verbose,
            },
        )

    needs_onnx = bool(selected_ids & {"onnx", "tensorrt", "openvino"})
    if needs_onnx:
        tasks["onnx"] = ExportTask(
            get_exporter_class("onnx"),
            dict(onnx_kwargs),
            report="onnx" in selected_ids,
        )

    if "tensorrt" in selected_ids:
        tasks["engine"] = ExportTask(
            get_exporter_class("tensorrt"),
            {
                **onnx_kwargs,
                "workspace": args.workspace,
            },
        )

    if "coreml" in selected_ids:
        tasks["coreml"] = ExportTask(
            get_exporter_class("coreml"),
            {
                **common_kwargs,
                "batch_buckets": getattr(args, "coreml_batch_buckets", (1, 8, 16, 32)),
                "minimum_deployment_target": getattr(
                    args,
                    "coreml_minimum_deployment_target",
                    "macOS15",
                ),
                "compute_units": getattr(args, "coreml_compute_units", "CPUAndGPU"),
                "timeout_s": getattr(args, "coreml_timeout", 600.0),
                "max_memory_gb": getattr(args, "coreml_max_memory_gb", 16.0),
                "verbose": args.verbose,
            },
        )

    if "tflite" in selected_ids:
        tasks["tflite"] = ExportTask(
            get_exporter_class("tflite"),
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

    if "openvino" in selected_ids:
        tasks["openvino"] = ExportTask(
            get_exporter_class("openvino"),
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


def prepare_export(args):
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    args.weights = resolve_export_weights(args.weights)

    with suppress_export_noise(not args.verbose):
        model, dummy_input = prepare_export_model(args)
    return model, dummy_input


def execute_export(args, model, dummy_input):
    export_tasks = create_export_tasks(args, model, dummy_input)
    with suppress_export_noise(not args.verbose):
        exported_files = perform_exports(export_tasks)
    return exported_files


def export_reid(args) -> ReIDExportResult:
    """Prepare, export, and parity-check a ReID checkpoint."""

    model, dummy_input = prepare_export(args)
    exported_files = execute_export(args, model, dummy_input)
    parity_report: dict[str, dict[str, Any]] = {}
    if exported_files:
        with suppress_export_noise(not args.verbose):
            parity_report = verify_export_parity(args, model, dummy_input, exported_files)
    return ReIDExportResult(
        weights=args.weights,
        files=exported_files,
        parity=parity_report,
        half=bool(getattr(args, "half", False)),
    )


def verify_export_parity(
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
            format_input = cpu_input
            format_ref_np = ref_np
            # Native Core ML loads MLPrograms lazily. Verifying with the B=1
            # bucket proves graph correctness without compiling a larger
            # package solely for this informational parity check.
            if fmt == "coreml" and cpu_input.shape[0] > 1:
                format_input = cpu_input[:1]
                format_ref_np = ref_np[:1]
            if fmt == "tflite":
                out = _run_tflite_for_parity(fpath, format_input)
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
                out = reid.model.forward(format_input)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if hasattr(out, "detach"):
                out_np = out.detach().to(torch.float32).cpu().numpy()
            else:
                out_np = np.asarray(out, dtype=np.float32)

            if out_np.shape != format_ref_np.shape:
                report[fmt] = {
                    "max_abs": float("nan"),
                    "mean_abs": float("nan"),
                    "cosine": float("nan"),
                    "parity_ok": False,
                    "embedding_ok": False,
                    "ok": False,
                    "error": f"shape mismatch: {out_np.shape} vs {format_ref_np.shape}",
                }
                continue

            diff = np.abs(out_np - format_ref_np)
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            parity_ok = bool(np.allclose(out_np, format_ref_np, rtol=rtol, atol=atol))

            # Embedding-aware metric: ReID features are consumed by
            # cosine similarity, so two embeddings that point in the
            # same direction are functionally equivalent. Flatten any
            # spatial dims and average the per-sample cosine.
            ref_flat = format_ref_np.reshape(format_ref_np.shape[0], -1)
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


__all__ = (
    "ExportTask",
    "ReIDExportResult",
    "create_export_tasks",
    "execute_export",
    "export_reid",
    "perform_exports",
    "prepare_export",
    "suppress_export_noise",
    "validate_export_formats",
    "verify_export_parity",
)
