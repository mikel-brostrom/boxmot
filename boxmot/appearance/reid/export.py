import argparse
import time
from pathlib import Path

import torch

from boxmot.appearance.reid import export_formats
from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.appearance.reid.registry import ReIDModelRegistry
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.appearance.exporters.torchscript_exporter import TorchScriptExporter
from boxmot.appearance.exporters.onnx_exporter import ONNXExporter
from boxmot.appearance.exporters.openvino_exporter import OpenVINOExporter
from boxmot.appearance.exporters.tflite_exporter import TFLiteExporter
from boxmot.appearance.exporters.tensorrt_exporter import EngineExporter
from boxmot.utils import WEIGHTS, logger as LOGGER
from boxmot.utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description="ReID Model Exporter")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[256, 128],
        help="Image size (height, width)",
    )
    parser.add_argument(
        "--device", default="cpu", help="CUDA device, e.g., 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Optimize TorchScript export for mobile"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes for ONNX/TF/TensorRT export",
    )
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--workspace", type=int, default=4, help="TensorRT workspace size (GB)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for TensorRT"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS / "osnet_x0_25_msmt17.pt",
        help="Path to model weights (model.pt)",
    )
    parser.add_argument(
        "--half", action="store_true", help="Export in FP16 half-precision"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="Export formats to include: torchscript, onnx, openvino, engine, tflite",
    )
    return parser.parse_args()


def load_model(args):
    """
    Load and prepare the pt ReID model.
    """

    model_name = ReIDModelRegistry.get_model_name(args.weights)
    nr_classes = ReIDModelRegistry.get_nr_classes(args.weights)
    pretrained = not (args.weights and args.weights.is_file() and args.weights.suffix == ".pt")

    model = ReIDModelRegistry.build_model(
        model_name, num_classes=nr_classes, pretrained=pretrained, use_gpu=args.device
    ).to(args.device)
    ReIDModelRegistry.load_pretrained_weights(model, args.weights)
    model.eval()
    return model


def prepare_input(args, device):
    """
    Prepare the dummy input tensor for model export.
    Adjust image size if a specific model is used.
    """
    if "lmbn" in str(args.weights):
        args.imgsz = [384, 128]
    return torch.empty(args.batch_size, 3, args.imgsz[0], args.imgsz[1]).to(device)


def warmup_model(model, im, half):
    """
    Run a couple of forward passes to warm up the model.
    Optionally convert input and model to half-precision.
    """
    for _ in range(2):
        y = model(im)
    if half:
        im = im.half()
        model = model.half()
    return y, im, model


def export_models(model, im, args, flags):
    """
    Export the model in each of the requested formats.
    `flags` is a dict containing export type flags.
    Returns a dictionary of export results.
    """
    export_results = {}

    # TorchScript exporter
    if flags.get("torchscript", False):
        exporter = TorchScriptExporter(model, im, args.weights, args.optimize)
        export_results["torchscript"] = exporter.export()

    # TensorRT engine exporter
    if flags.get("engine", False):
        exporter = EngineExporter(
            model, im, args.weights, args.half, args.dynamic, args.simplify, args.verbose
        )
        export_results["engine"] = exporter.export()

    # ONNX exporter
    if flags.get("onnx", False):
        exporter = ONNXExporter(
            model, im, args.weights, args.opset, args.dynamic, args.half, args.simplify
        )
        export_results["onnx"] = exporter.export()

    # TFLite exporter (does not return a file path)
    if flags.get("tflite", False):
        exporter = TFLiteExporter(model, im, args.weights)
        exporter.export()

    # OpenVINO exporter
    if flags.get("openvino", False):
        exporter = OpenVINOExporter(model, im, args.weights, args.half)
        export_results["openvino"] = exporter.export()

    return export_results


def main():
    args = parse_args()
    start_time = time.time()

    # Create output directory if not existing
    args.weights.parent.mkdir(parents=True, exist_ok=True)

    # Determine which export formats were requested
    valid_formats = tuple(export_formats()["Argument"][1:])  # e.g., ('torchscript', 'onnx', 'openvino', 'engine', 'tflite')
    requested_formats = [x.lower() for x in args.include]
    flags_list = [fmt in requested_formats for fmt in valid_formats]
    if sum(flags_list) != len(requested_formats):
        raise ValueError(
            f"ERROR: Invalid --include {requested_formats}. Valid options are {valid_formats}"
        )
    flags = dict(zip(valid_formats, flags_list))

    # Device selection and validations
    args.device = select_device(args.device)
    if args.half and args.device.type == "cpu":
        raise ValueError("--half export is only supported for GPU. Please use a GPU device (e.g., --device 0).")
    if args.optimize and args.device.type != "cpu":
        raise ValueError("--optimize is only compatible with CPU export. Please use --device cpu.")

    # Load model and prepare dummy input
    model = load_model(args)
    im = prepare_input(args, args.device)

    # Warm up the model before export
    y, im, model = warmup_model(model, im, args.half)
    output_tensor = y[0] if isinstance(y, tuple) else y
    output_shape = tuple(output_tensor.shape)
    file_size_mb = BaseExporter.file_size(args.weights)
    LOGGER.info(
        f"\nStarting from {args.weights} with output shape {output_shape} ({file_size_mb:.1f} MB)"
    )

    # Export the model in each requested format
    export_results = export_models(model, im, args, flags)

    elapsed = time.time() - start_time
    if export_results:
        LOGGER.info(
            f"\nExport complete ({elapsed:.1f}s)"
            f"\nResults saved to {args.weights.parent.resolve()}"
            f"\nVisualize models at https://netron.app"
        )


if __name__ == "__main__":
    main()
