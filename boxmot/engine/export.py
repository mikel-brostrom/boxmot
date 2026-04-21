#!/usr/bin/env python3
import io
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import torch

from boxmot.reid.core import ReID, export_formats
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.utils import WEIGHTS
from boxmot.utils import configure_logging as _configure_boxmot_logging, logger as LOGGER
from boxmot.utils.torch_utils import select_device


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

    original_stderr = io.StringIO()
    LOGGER.remove()
    LOGGER.add(
        sys.stderr,
        level="ERROR",
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
        format="<level>{level: <8}</level> | <level>{message}</level>",
    )
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(original_stderr):
            yield
    finally:
        _configure_boxmot_logging(main_only=True)


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


def main(args):
    start_time = time.time()

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    args.weights = WEIGHTS / args.weights.name
    
    if args.verbose:
        LOGGER.info("")
        LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
        LOGGER.opt(colors=True).info("<bold><cyan>🚀 BoxMOT ReID Export</cyan></bold>")
        LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
        LOGGER.opt(colors=True).info(f"<bold>Weights:</bold>    <cyan>{args.weights}</cyan>")
        LOGGER.opt(colors=True).info(f"<bold>Formats:</bold>    <cyan>{', '.join(args.include)}</cyan>")
        LOGGER.opt(colors=True).info(f"<bold>Device:</bold>     <cyan>{args.device}</cyan>")
        LOGGER.opt(colors=True).info(f"<bold>Half:</bold>       <cyan>{args.half}</cyan>")
        LOGGER.opt(colors=True).info("<blue>" + "-"*60 + "</blue>")
        LOGGER.opt(colors=True).info("<cyan>[1/3]</cyan> Setting up model...")

    with _suppress_export_noise(not args.verbose):
        model, dummy_input = setup_model(args)

    output = model(dummy_input)
    output_tensor = output[0] if isinstance(output, tuple) else output
    output_shape = tuple(output_tensor.shape)
    if args.verbose:
        LOGGER.opt(colors=True).info(
            f"<bold>Input shape:</bold>  <cyan>{tuple(dummy_input.shape)}</cyan>"
        )
        LOGGER.opt(colors=True).info(
            f"<bold>Output shape:</bold> <cyan>{output_shape}</cyan> "
            f"({BaseExporter.file_size(args.weights):.1f} MB)"
        )
        LOGGER.opt(colors=True).info("<cyan>[2/3]</cyan> Exporting to formats...")

    export_tasks = create_export_tasks(args, model, dummy_input)
    with _suppress_export_noise(not args.verbose):
        exported_files = perform_exports(export_tasks)

    if exported_files and args.verbose:
        elapsed_time = time.time() - start_time
        LOGGER.opt(colors=True).info("<cyan>[3/3]</cyan> Export complete!")
        LOGGER.info("")
        LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
        LOGGER.opt(colors=True).info("<bold><cyan>✅ Export Summary</cyan></bold>")
        LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
        LOGGER.opt(colors=True).info(f"<bold>Time:</bold>       <cyan>{elapsed_time:.1f}s</cyan>")
        LOGGER.opt(colors=True).info(f"<bold>Saved to:</bold>   <cyan>{args.weights.parent.resolve()}</cyan>")
        for fmt, fpath in exported_files.items():
            LOGGER.opt(colors=True).info(f"<bold>  • {fmt}:</bold> <cyan>{fpath}</cyan>")
        LOGGER.opt(colors=True).info("<bold>Visualize:</bold>  <cyan>https://netron.app</cyan>")
        LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")


if __name__ == "__main__":
    main()
