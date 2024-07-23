import argparse
import time
import torch
from pathlib import Path
from boxmot.appearance import export_formats
from boxmot.utils.torch_utils import select_device
from boxmot.appearance.reid_model_factory import get_model_name, load_pretrained_weights, build_model, get_nr_classes
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.utils import WEIGHTS, logger as LOGGER

from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.appearance.exporters.torchscript_exporter import TorchScriptExporter
from boxmot.appearance.exporters.onnx_exporter import ONNXExporter
from boxmot.appearance.exporters.openvino_exporter import OpenVINOExporter
from boxmot.appearance.exporters.tflite_exporter import TFLiteExporter
from boxmot.appearance.exporters.tensorrt_exporter import EngineExporter

def parse_args():
    parser = argparse.ArgumentParser(description="ReID export")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[256, 128], help="image (h, w)")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--weights", type=Path, default=WEIGHTS / "osnet_x0_25_msmt17.pt", help="model.pt path(s)")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--include", nargs="+", default=["torchscript"], help="torchscript, onnx, openvino, engine")
    return parser.parse_args()

def main():
    args = parse_args()

    t = time.time()
    WEIGHTS.mkdir(parents=False, exist_ok=True)

    include = [x.lower() for x in args.include]
    fmts = tuple(export_formats()["Argument"][1:])
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    jit, onnx, openvino, engine, tflite = flags

    args.device = select_device(args.device)
    if args.half:
        assert args.device.type != "cpu", "--half only compatible with GPU export, i.e. use --device 0"

    rab = ReidAutoBackend(weights=args.weights, device=args.device, half=args.half)
    model = rab.get_backend()

    model = build_model(
        get_model_name(args.weights),
        num_classes=get_nr_classes(args.weights),
        pretrained=not (args.weights and args.weights.is_file() and args.weights.suffix == ".pt"),
        use_gpu=args.device,
    ).to(args.device)
    load_pretrained_weights(model, args.weights)
    model.eval()

    if args.optimize:
        assert args.device.type == "cpu", "--optimize not compatible with cuda devices, i.e. use --device cpu"

    if "lmbn" in str(args.weights):
        args.imgsz = (384, 128)

    im = torch.empty(args.batch_size, 3, args.imgsz[0], args.imgsz[1]).to(args.device)
    for _ in range(2):
        y = model(im)
    if args.half:
        im, model = im.half(), model.half()
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)
    LOGGER.info(f"\nStarting from {args.weights} with output shape {shape} ({BaseExporter.file_size(args.weights):.1f} MB)")

    f = [""] * len(fmts)
    if jit:
        exporter = TorchScriptExporter(model, im, args.weights, args.optimize)
        f[0] = exporter.export()
    if engine:
        exporter = EngineExporter(model, im, args.weights, args.half, args.dynamic, args.simplify, args.verbose)
        f[1] = exporter.export()
    if onnx:
        exporter = ONNXExporter(model, im, args.weights, args.opset, args.dynamic, args.half, args.simplify)
        f[2] = exporter.export()
    if tflite:
        exporter = TFLiteExporter(model, im, args.weights)
        exporter.export()
    if openvino:
        exporter = OpenVINOExporter(model, im, args.weights, args.half)
        f[3] = exporter.export()

    f = [str(x) for x in f if x]
    if any(f):
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {args.weights.parent.resolve()}"
            f"\nVisualize:       https://netron.app"
        )

if __name__ == "__main__":
    main()