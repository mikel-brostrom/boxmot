#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch

from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.appearance.reid import export_formats
from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.appearance.reid.registry import ReIDModelRegistry
from boxmot.utils import WEIGHTS
from boxmot.utils import logger as LOGGER
from boxmot.utils.torch_utils import select_device
from boxmot.utils.checks import RequirementsChecker


def parse_args():
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
    # Optional convenience flags to auto-install from project extras/groups
    parser.add_argument("--dep-group", type=str, default=None,
                        help="Install a uv dependency group before export (requires uv).")
    parser.add_argument("--dep-extra", type=str, default=None,
                        help="Install a project extra before export (e.g., 'openvino').")
    parser.add_argument("--dep-args", nargs="*", default=None,
                        help="Extra args for the installer, e.g. --upgrade")
    return parser.parse_args()


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
    _ = auto_backend.get_backend()

    model_name = ReIDModelRegistry.get_model_name(args.weights)
    nr_classes = ReIDModelRegistry.get_nr_classes(args.weights)
    pretrained = not (args.weights and args.weights.is_file() and args.weights.suffix == ".pt")
    model = ReIDModelRegistry.build_model(
        model_name, num_classes=nr_classes, pretrained=pretrained, use_gpu=args.device
    ).to(args.device)
    ReIDModelRegistry.load_pretrained_weights(model, args.weights)
    model.eval()

    if args.optimize and args.device.type != "cpu":
        raise AssertionError("--optimize not compatible with CUDA devices, use --device cpu")

    if "lmbn" in str(args.weights):
        args.imgsz = [384, 128]

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
        from boxmot.appearance.exporters.torchscript_exporter import TorchScriptExporter
        tasks["torchscript"] = (
            True,
            TorchScriptExporter,
            (model, dummy_input, args.weights, args.optimize),
        )

    if onnx_flag:
        from boxmot.appearance.exporters.onnx_exporter import ONNXExporter
        tasks["onnx"] = (
            True,
            ONNXExporter,
            (model, dummy_input, args.weights, args.opset, args.dynamic, args.half, args.simplify),
        )

    if engine_flag:
        from boxmot.appearance.exporters.tensorrt_exporter import EngineExporter
        tasks["engine"] = (
            True,
            EngineExporter,
            (model, dummy_input, args.weights, args.half, args.dynamic, args.simplify, args.verbose),
        )

    if tflite_flag:
        from boxmot.appearance.exporters.tflite_exporter import TFLiteExporter
        tasks["tflite"] = (
            True,
            TFLiteExporter,
            (model, dummy_input, args.weights),
        )

    if openvino_flag:
        from boxmot.appearance.exporters.openvino_exporter import OpenVINOExporter
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


def main():
    args = parse_args()
    start_time = time.time()

    # Optional: preflight dependency install from CLI flags
    if args.dep_group or args.dep_extra:
        RequirementsChecker().sync_group_or_extra(
            group=args.dep_group, extra=args.dep_extra, extra_args=args.dep_args
        )

    # Or, as a fallback, ensure basic reqs from a requirements file:
    # RequirementsChecker().check_requirements_file()

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
