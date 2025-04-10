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
    """
    Parse command-line arguments for the ReID export script.
    """
    parser = argparse.ArgumentParser(description="ReID Export Script")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export")
    parser.add_argument("--imgsz", "--img", "--img-size",
                        nargs="+", type=int, default=[256, 128],
                        help="Image size in the format: height width")
    parser.add_argument("--device", default="cpu",
                        help="CUDA device (e.g., '0', '0,1,2,3', or 'cpu')")
    parser.add_argument("--optimize", action="store_true",
                        help="Optimize TorchScript for mobile (CPU export only)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Enable dynamic axes for ONNX/TF/TensorRT export")
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify ONNX model")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version")
    parser.add_argument("--workspace", type=int, default=4,
                        help="TensorRT workspace size (GB)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging for TensorRT")
    parser.add_argument("--weights", type=Path,
                        default=WEIGHTS / "osnet_x0_25_msmt17.pt",
                        help="Path to the model weights (.pt file)")
    parser.add_argument("--half", action="store_true",
                        help="Enable FP16 half-precision export (GPU only)")
    parser.add_argument("--include", nargs="+",
                        default=["torchscript"],
                        help=("Export formats to include. Options: torchscript, onnx, "
                              "openvino, engine, tflite"))
    return parser.parse_args()


def validate_export_formats(include):
    """
    Validate the provided export formats and return corresponding flags.

    Args:
        include (list): List of export formats provided via the command line.

    Returns:
        tuple: Boolean flags for each export format in the order:
               (torchscript, onnx, openvino, engine, tflite)
    """
    available_formats = tuple(export_formats()["Argument"][1:])
    include_lower = [fmt.lower() for fmt in include]
    flags = [fmt in include_lower for fmt in available_formats]
    if sum(flags) != len(include_lower):
        raise AssertionError(
            f"ERROR: Invalid --include {include}, valid arguments are {available_formats}"
        )
    return tuple(flags)


def setup_model(args):
    """
    Initialize and prepare the ReID model for export.

    Args:
        args: Parsed command-line arguments.

    Returns:
        tuple: (model (torch.nn.Module), dummy_input (torch.Tensor))
    """
    # Select the correct device
    args.device = select_device(args.device)
    if args.half and args.device.type == "cpu":
        raise AssertionError("--half only compatible with GPU export, use --device 0 for GPU")

    # Initialize backend model using the auto backend
    auto_backend = ReidAutoBackend(weights=args.weights, device=args.device, half=args.half)
    _ = auto_backend.get_backend()  # Backend model is managed internally

    # Build and load the ReID model from the registry
    model_name = ReIDModelRegistry.get_model_name(args.weights)
    nr_classes = ReIDModelRegistry.get_nr_classes(args.weights)
    pretrained = not (args.weights and args.weights.is_file() and args.weights.suffix == ".pt")
    model = ReIDModelRegistry.build_model(
        model_name,
        num_classes=nr_classes,
        pretrained=pretrained,
        use_gpu=args.device,
    ).to(args.device)
    ReIDModelRegistry.load_pretrained_weights(model, args.weights)
    model.eval()

    # Ensure --optimize is only used with CPU exports
    if args.optimize and args.device.type != "cpu":
        raise AssertionError("--optimize not compatible with CUDA devices, use --device cpu")

    # Adjust image size if a specific weight type is detected
    if "lmbn" in str(args.weights):
        args.imgsz = [384, 128]

    # Create dummy input tensor for warming up the model
    dummy_input = torch.empty(args.batch_size, 3, args.imgsz[0], args.imgsz[1]).to(args.device)
    for _ in range(2):
        _ = model(dummy_input)

    # Convert to half precision if required
    if args.half:
        dummy_input = dummy_input.half()
        model = model.half()

    return model, dummy_input


def create_export_tasks(args, model, dummy_input):
    """
    Create a mapping of export tasks with associated flags, exporter classes, and parameters.

    Args:
        args: Parsed command-line arguments.
        model: Prepared ReID model.
        dummy_input: Dummy input tensor.

    Returns:
        dict: Mapping of export format to a tuple (flag, exporter_class, export_args)
    """
    torchscript_flag, onnx_flag, openvino_flag, engine_flag, tflite_flag = validate_export_formats(args.include)
    return {
        "torchscript": (
            torchscript_flag,
            TorchScriptExporter,
            (model, dummy_input, args.weights, args.optimize)
        ),
        "engine": (
            engine_flag,
            EngineExporter,
            (model, dummy_input, args.weights, args.half, args.dynamic, args.simplify, args.verbose)
        ),
        "onnx": (
            onnx_flag,
            ONNXExporter,
            (model, dummy_input, args.weights, args.opset, args.dynamic, args.half, args.simplify)
        ),
        "tflite": (
            tflite_flag,
            TFLiteExporter,
            (model, dummy_input, args.weights)
        ),
        "openvino": (
            openvino_flag,
            OpenVINOExporter,
            (model, dummy_input, args.weights, args.half)
        )
    }


def perform_exports(export_tasks):
    """
    Iterate over export tasks and perform export for enabled formats.

    Args:
        export_tasks (dict): Mapping of export tasks.

    Returns:
        dict: Mapping of export format to export results.
    """
    exported_files = {}
    for fmt, (flag, exporter_class, exp_args) in export_tasks.items():
        if flag:
            exporter = exporter_class(*exp_args)
            export_result = exporter.export()
            exported_files[fmt] = export_result
    return exported_files


def main():
    """Main function to execute the ReID export process."""
    args = parse_args()
    start_time = time.time()

    # Ensure the weights directory exists
    WEIGHTS.mkdir(parents=False, exist_ok=True)

    # Setup model and create a dummy input tensor
    model, dummy_input = setup_model(args)

    # Log model output shape and file size
    output = model(dummy_input)
    output_tensor = output[0] if isinstance(output, tuple) else output
    output_shape = tuple(output_tensor.shape)
    LOGGER.info(
        f"\nStarting from {args.weights} with output shape {output_shape} "
        f"({BaseExporter.file_size(args.weights):.1f} MB)"
    )

    # Create export tasks
    export_tasks = create_export_tasks(args, model, dummy_input)

    # Perform exports for enabled formats
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
