#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch

from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.reid.core import export_formats
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.utils import WEIGHTS
from boxmot.utils import logger as LOGGER
from boxmot.utils.torch_utils import select_device
from boxmot.utils.checks import RequirementsChecker


def validate_export_formats(include):
    available_formats = tuple(export_formats()["Argument"][1:])
    include_lower = [fmt.lower() for fmt in include]
    flags = [fmt in include_lower for fmt in available_formats]
    if sum(flags) != len(include_lower):
        raise AssertionError(
            f"ERROR: Invalid --include {include}, valid arguments are {available_formats}"
        )
    return tuple(flags)


def setup_model(args):
    args.device = select_device(args.device)
    if args.half and args.device.type == "cpu":
        raise AssertionError("--half only compatible with GPU export, use --device 0 for GPU")

    auto_backend = ReidAutoBackend(weights=args.weights, device=args.device, half=args.half)
    model_name = ReIDModelRegistry.get_model_name(args.weights)
    model = auto_backend.model.model.eval()

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

    dummy_input = torch.empty(args.batch_size, 3, args.imgsz[0], args.imgsz[1]).to(args.device)
    for _ in range(2):
        _ = model(dummy_input)

    if args.half:
        dummy_input = dummy_input.half()
        model = model.half()

    return model, dummy_input


def create_export_tasks(args, model, dummy_input):
    torchscript_flag, onnx_flag, openvino_flag, engine_flag, tflite_flag = validate_export_formats(args.include)
    tasks = {}

    if torchscript_flag:
        from boxmot.reid.exporters.torchscript_exporter import TorchScriptExporter
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
            (model, dummy_input, args.weights, args.opset, args.dynamic, args.half, args.simplify),
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
            (model, dummy_input, args.weights),
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

    WEIGHTS.mkdir(parents=False, exist_ok=True)

    model, dummy_input = setup_model(args)

    output = model(dummy_input)
    output_tensor = output[0] if isinstance(output, tuple) else output
    output_shape = tuple(output_tensor.shape)
    LOGGER.info(
        f"\nStarting from {args.weights} with output shape {output_shape} "
        f"({BaseExporter.file_size(args.weights):.1f} MB)"
    )

    export_tasks = create_export_tasks(args, model, dummy_input)
    exported_files = perform_exports(export_tasks)

    if exported_files:
        elapsed_time = time.time() - start_time
        LOGGER.info(
            f"\nExport complete ({elapsed_time:.1f}s)"
            f"\nResults saved to {args.weights.parent.resolve()}"
            f"\nVisualize: https://netron.app"
        )


if __name__ == "__main__":
    main()
