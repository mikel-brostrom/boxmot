#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch

from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.appearance.exporters.onnx_exporter import ONNXExporter
from boxmot.appearance.exporters.openvino_exporter import OpenVINOExporter
from boxmot.appearance.exporters.tflite_exporter import TFLiteExporter
from boxmot.appearance.exporters.torchscript_exporter import TorchScriptExporter
from boxmot.appearance.exporters.tensorrt_exporter import EngineExporter
from boxmot.appearance.reid import export_formats
from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.appearance.reid.registry import ReIDModelRegistry
from boxmot.utils import WEIGHTS, logger as LOGGER
from boxmot.utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description="ReID export")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--imgsz", "--img", "--img-size",
        nargs="+",
        type=int,
        default=[256, 128],
        help="Image size (height, width)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="CUDA device, e.g. '0', '0,1,2,3' or 'cpu'"
    )
    parser.add_argument("--optimize", action="store_true", help="Optimize TorchScript for mobile")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: enable dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS / "osnet_x0_25_msmt17.pt",
        help="Path to the model weights (.pt file)"
    )
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export (GPU only)")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="List of export formats: torchscript, onnx, openvino, engine, tflite"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # Ensure the weights directory exists
    WEIGHTS.mkdir(parents=False, exist_ok=True)

    # Validate export formats
    include = [fmt.lower() for fmt in args.include]
    available_formats = tuple(export_formats()["Argument"][1:])
    flags = [fmt in include for fmt in available_formats]
    if sum(flags) != len(include):
        raise AssertionError(
            f"ERROR: Invalid --include {include}, valid arguments are {available_formats}"
        )
    jit, onnx, openvino, engine, tflite = flags

    # Select device and validate half precision usage
    args.device = select_device(args.device)
    if args.half and args.device.type == "cpu":
        raise AssertionError("--half only compatible with GPU export, use --device 0 for GPU")

    # Initialize backend model using the auto backend
    auto_backend = ReidAutoBackend(weights=args.weights, device=args.device, half=args.half)
    model = auto_backend.get_backend()

    # Build and load the ReID model
    model = ReIDModelRegistry.build_model(
        ReIDModelRegistry.get_model_name(args.weights),
        num_classes=ReIDModelRegistry.get_nr_classes(args.weights),
        pretrained=not (args.weights and args.weights.is_file() and args.weights.suffix == ".pt"),
        use_gpu=args.device,
    ).to(args.device)
    ReIDModelRegistry.load_pretrained_weights(model, args.weights)
    model.eval()

    # Ensure --optimize is only used with CPU exports
    if args.optimize and args.device.type != "cpu":
        raise AssertionError("--optimize not compatible with CUDA devices, use --device cpu")

    # Adjust image size if specific weight type is detected
    if "lmbn" in str(args.weights):
        args.imgsz = [384, 128]

    # Create a dummy input tensor and warm-up the model
    im = torch.empty(args.batch_size, 3, args.imgsz[0], args.imgsz[1]).to(args.device)
    for _ in range(2):
        output = model(im)
    if args.half:
        im = im.half()
        model = model.half()

    # Log model output shape and file size
    output_shape = tuple((output[0] if isinstance(output, tuple) else output).shape)
    LOGGER.info(
        f"\nStarting from {args.weights} with output shape {output_shape} "
        f"({BaseExporter.file_size(args.weights):.1f} MB)"
    )

    # Create mapping for each export format including associated flag, exporter class, and parameters
    export_tasks = {
        "torchscript": (
            jit,
            TorchScriptExporter,
            (model, im, args.weights, args.optimize)
        ),
        "engine": (
            engine,
            EngineExporter,
            (model, im, args.weights, args.half, args.dynamic, args.simplify, args.verbose)
        ),
        "onnx": (
            onnx,
            ONNXExporter,
            (model, im, args.weights, args.opset, args.dynamic, args.half, args.simplify)
        ),
        "tflite": (
            tflite,
            TFLiteExporter,
            (model, im, args.weights)
        ),
        "openvino": (
            openvino,
            OpenVINOExporter,
            (model, im, args.weights, args.half)
        )
    }

    exported_files = {}

    # Iterate over each export task and perform export if the flag is enabled
    for fmt, (flag, exporter_class, exp_args) in export_tasks.items():
        if flag:
            exporter = exporter_class(*exp_args)
            export_result = exporter.export()
            exported_files[fmt] = export_result

    # Log export completion details if any export was performed
    if exported_files:
        LOGGER.info(
            f"\nExport complete ({time.time() - start_time:.1f}s)"
            f"\nResults saved to {args.weights.parent.resolve()}"
            f"\nVisualize: https://netron.app"
        )


if __name__ == "__main__":
    main()
