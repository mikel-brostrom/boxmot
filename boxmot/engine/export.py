#!/usr/bin/env python3
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from boxmot.engine.workflow_results import ExportResult
from boxmot.reid.core import ReID, export_formats
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.utils import WEIGHTS, logger as LOGGER
from boxmot.utils.download import set_download_status_fn
from boxmot.utils.rich.export_reporting import (
    EXPORT_RUN_STEP,
    EXPORT_SETUP_STEP,
    ExportWorkflowReporter,
    log_export_pipeline_intro,
)
from boxmot.utils.rich.reporting import WorkflowDetailCallback
from boxmot.utils.torch_utils import select_device


__all__ = [
    "EXPORT_SETUP_STEP",
    "EXPORT_RUN_STEP",
    "ExportWorkflowReporter",
    "log_export_pipeline_intro",
    "main",
    "run_export",
]


def validate_export_formats(include):
    available_formats = tuple(export_formats()["Argument"][1:])
    include_lower = [fmt.lower() for fmt in include]
    flags = [fmt in include_lower for fmt in available_formats]
    if sum(flags) != len(include_lower):
        raise AssertionError(
            f"ERROR: Invalid --include {include}, valid arguments are {available_formats}"
        )
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
        yield
    finally:
        boxmot_logger.setLevel(previous_boxmot_level)
        for target, level, propagate in noisy_loggers:
            target.setLevel(level)
            target.propagate = propagate


def setup_model(args):
    args.device = select_device(args.device)
    if args.half and args.device.type == "cpu":
        raise AssertionError("--half only compatible with GPU export, use --device 0 for GPU")

    reid = ReID(weights=args.weights, device=args.device, half=args.half)
    model_name = ReIDModelRegistry.get_model_name(args.weights)
    model = reid.model.model.eval()

    if args.optimize and args.device.type != "cpu":
        raise AssertionError("--optimize not compatible with CUDA devices, use --device cpu")

    if "vehicleid" in args.weights.name or "veri" in args.weights.name:
        args.imgsz = (256, 256)
    elif "lmbn" in model_name:
        args.imgsz = (384, 128)
    elif "hacnn" in model_name:
        args.imgsz = (160, 64)
    else:
        args.imgsz = (256, 128)

    if args.half:
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

    if torchscript_flag:
        from boxmot.reid.exporters.torchscript_exporter import \
            TorchScriptExporter
        tasks["torchscript"] = (
            True,
            TorchScriptExporter,
            (model, dummy_input, args.weights, args.optimize),
        )

    if onnx_flag:
        from boxmot.reid.exporters.onnx_exporter import ONNXExporter
        tasks["onnx"] = (
            True,
            ONNXExporter,
            (model, dummy_input, args.weights, args.opset, args.dynamic, args.half, args.simplify, args.verbose),
        )

    if engine_flag:
        from boxmot.reid.exporters.tensorrt_exporter import EngineExporter
        tasks["engine"] = (
            True,
            EngineExporter,
            (model, dummy_input, args.weights, args.half, args.dynamic, args.simplify, args.verbose),
        )

    if tflite_flag:
        from boxmot.reid.exporters.tflite_exporter import TFLiteExporter
        tasks["tflite"] = (
            True,
            TFLiteExporter,
            (model, dummy_input, args.weights, args.opset, args.dynamic, args.half, args.simplify),
        )

    if openvino_flag:
        from boxmot.reid.exporters.openvino_exporter import OpenVINOExporter
        tasks["openvino"] = (
            True,
            OpenVINOExporter,
            (model, dummy_input, args.weights, args.half),
        )

    return tasks



def perform_exports(export_tasks):
    exported_files = {}
    for fmt, (flag, exporter_class, exp_args) in export_tasks.items():
        if flag:
            exporter = exporter_class(*exp_args)
            # Exporters can optionally declare:
            #   group="...", or extra="...", and extra_args=["--upgrade"]
            # The BaseExporter decorator will auto-install them.
            exported_files[fmt] = exporter.export()
    return exported_files


def _prepare_export(args):
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    args.weights = WEIGHTS / Path(args.weights).name

    with _suppress_export_noise(not args.verbose):
        model, dummy_input = setup_model(args)
    return model, dummy_input


def _execute_export(args, model, dummy_input):
    export_tasks = create_export_tasks(args, model, dummy_input)
    with _suppress_export_noise(not args.verbose):
        exported_files = perform_exports(export_tasks)
    return exported_files


def run_export(args) -> ExportResult:
    model, dummy_input = _prepare_export(args)
    exported_files = _execute_export(args, model, dummy_input)
    return ExportResult(weights=args.weights, files=exported_files)


def _verify_export_parity(
    args,
    model,
    dummy_input,
    exported_files: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Compare each exported model's output to the original PyTorch model.

    Returns a mapping ``{format: {"max_abs": float, "mean_abs": float, "ok": bool}}``.
    Failures are logged but do not raise — exporting is the primary goal,
    parity is informational.

    Per-format tolerances:
    - torchscript / onnx: tight (~1e-3) — bit-exact FP32 is expected.
    - openvino: looser (~5e-2) — OpenVINO's ``convert_model`` performs
      BN folding and other graph rewrites at conversion time that drift
      a few percent even with ``INFERENCE_PRECISION_HINT=f32``.
    """
    import numpy as np

    tolerances: dict[str, tuple[float, float]] = {
        "torchscript": (1e-2, 1e-3),
        "onnx": (1e-2, 1e-3),
        "openvino": (5e-2, 5e-2),
    }

    # ``dummy_input`` was created via ``torch.empty`` for shape inference, so
    # its values are uninitialised and can be NaN/inf — propagating those
    # through the model gives spurious parity failures. Use a deterministic
    # random tensor of the same shape/dtype/device for the comparison.
    torch.manual_seed(0)
    sample = torch.rand_like(dummy_input.float()).to(
        dtype=dummy_input.dtype, device=dummy_input.device
    )

    with torch.inference_mode():
        ref_output = model(sample)
    if isinstance(ref_output, (tuple, list)):
        ref_output = ref_output[0]
    ref_np = ref_output.detach().to(torch.float32).cpu().numpy()

    cpu_input = sample.detach().to("cpu", dtype=torch.float32)

    report: dict[str, dict[str, float]] = {}
    for fmt, fpath in exported_files.items():
        if fmt in ("engine", "tflite"):
            # TensorRT requires the matching CUDA runtime; TFLite needs the
            # tflite-runtime + Edge ops shim. Skip parity for those here.
            continue
        try:
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
                    "ok": False,
                    "error": f"shape mismatch: {out_np.shape} vs {ref_np.shape}",
                }
                continue

            diff = np.abs(out_np - ref_np)
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            rtol, atol = tolerances.get(fmt, (1e-2, 1e-3))
            ok = bool(np.allclose(out_np, ref_np, rtol=rtol, atol=atol))
            report[fmt] = {"max_abs": max_abs, "mean_abs": mean_abs, "ok": ok}
        except Exception as exc:  # pragma: no cover - defensive
            report[fmt] = {
                "max_abs": float("nan"),
                "mean_abs": float("nan"),
                "ok": False,
                "error": str(exc),
            }

    return report


def main(args):
    workflow = log_export_pipeline_intro(args)
    start_time = time.time()
    setup_status_fn = WorkflowDetailCallback(workflow, EXPORT_SETUP_STEP)
    set_download_status_fn(setup_status_fn)
    try:
        workflow.set_detail(EXPORT_SETUP_STEP, "Loading ReID model...")
        model, dummy_input = _prepare_export(args)

        output = model(dummy_input)
        output_tensor = output[0] if isinstance(output, tuple) else output
        output_shape = tuple(output_tensor.shape)
        workflow.set_detail(
            EXPORT_SETUP_STEP,
            (
                f"Input shape:  {tuple(dummy_input.shape)}\n"
                f"Output shape: {output_shape} "
                f"({BaseExporter.file_size(args.weights):.1f} MB)"
            ),
        )
        workflow.complete(EXPORT_SETUP_STEP, render=False)
        workflow.activate(EXPORT_RUN_STEP)

        formats = list(getattr(args, "include", []) or [])
        workflow.set_detail(
            EXPORT_RUN_STEP,
            f"Exporting to {len(formats)} format(s): {', '.join(formats) if formats else 'none'}",
        )
        exported_files = _execute_export(args, model, dummy_input)
        result = ExportResult(weights=args.weights, files=exported_files)

        elapsed_time = time.time() - start_time
        parity_report: dict[str, dict[str, float]] = {}
        if result.files:
            with _suppress_export_noise(not args.verbose):
                parity_report = _verify_export_parity(
                    args, model, dummy_input, result.files
                )
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
                    status = "OK" if stats["ok"] else "MISMATCH"
                    suffix = (
                        f" — parity {status} (maxΔ={stats['max_abs']:.1e})"
                    )
                lines.append(f"  • {fmt}: {fname}{suffix}")
            lines.append("")
            lines.append("Visualize: https://netron.app")
            workflow.set_detail(EXPORT_RUN_STEP, "\n".join(lines))
        else:
            workflow.set_detail(EXPORT_RUN_STEP, f"Export complete in {elapsed_time:.1f}s")
        workflow.complete(EXPORT_RUN_STEP, render=False)
        return result
    except BaseException as exc:
        workflow.fail(error=exc)
        raise
    finally:
        set_download_status_fn(None)
        workflow.stop()


if __name__ == "__main__":
    main()
