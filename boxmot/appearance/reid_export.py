# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from boxmot.appearance import export_formats
from boxmot.appearance.backbones import build_model, get_nr_classes
from boxmot.appearance.reid_model_factory import (get_model_name,
                                                  load_pretrained_weights)
from boxmot.utils import WEIGHTS
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import TestRequirements
from boxmot.utils.torch_utils import select_device

__tr = TestRequirements()


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / 1e6
    else:
        return 0.0


def export_torchscript(model, im, file, optimize):
    try:
        LOGGER.info(f"\nStarting export with torch {torch.__version__}...")
        f = file.with_suffix(".torchscript")
        print(f)
        ts = torch.jit.trace(model, im, strict=False)
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f))
        else:
            ts.save(str(f))

        LOGGER.info(f"Export success, saved as {f} ({file_size(f):.1f} MB)")
        return f
    except Exception as e:
        LOGGER.info(f"Export failure: {e}")


def export_onnx(model, im, file, opset, dynamic, fp16, simplify):
    # ONNX export
    try:
        __tr.check_packages(("onnx",))
        import onnx

        f = file.with_suffix(".onnx")
        LOGGER.info(f"\nStarting export with onnx {onnx.__version__}...")

        if dynamic:
            # input --> shape(N, 3, h, w), output --> shape(N, feat_size)
            dynamic = {"images": {0: "batch"}, "output": {0: "batch"}}

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes=dynamic or None,
        )
        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                cuda = torch.cuda.is_available()
                __tr.check_packages(
                    (
                        "onnxruntime-gpu" if cuda else "onnxruntime",
                        "onnx-simplifier>=0.4.1",
                    )
                )
                import onnxsim

                LOGGER.info(
                    f"simplifying with onnx-simplifier {onnxsim.__version__}..."
                )
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, "assert check failed"
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f"simplifier failure: {e}")
        LOGGER.info(f"Export success, saved as {f} ({file_size(f):.1f} MB)")
        return f
    except Exception as e:
        LOGGER.info(f"export failure: {e}")


def export_openvino(file, half):
    __tr.check_packages(
        ("openvino-dev",)
    )  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.runtime as ov  # noqa
    from openvino.tools import mo  # noqa

    f = str(file).replace(file.suffix, f"_openvino_model{os.sep}")
    f_onnx = file.with_suffix(".onnx")
    f_ov = str(Path(f) / file.with_suffix(".xml").name)
    try:
        LOGGER.info(f"\nStarting export with openvino {ov.__version__}...")
        # subprocess.check_output(cmd.split())  # export
        ov_model = mo.convert_model(
            f_onnx,
            model_name=file.with_suffix(".xml"),
            framework="onnx",
            compress_to_fp16=half,
        )  # export
        ov.serialize(ov_model, f_ov)  # save
    except Exception as e:
        LOGGER.info(f"export failure: {e}")
    LOGGER.info(f"Export success, saved as {f_ov} ({file_size(f_ov):.1f} MB)")
    return f


def export_tflite(file):
    try:
        __tr.check_packages(
            ("onnx2tf>=1.15.4", "tensorflow", "onnx_graphsurgeon>=0.3.26", "sng4onnx>=1.0.1"),
            cmds='--extra-index-url https://pypi.ngc.nvidia.com'
        )  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import onnx2tf

        LOGGER.info(f"\nStarting {file} export with onnx2tf {onnx2tf.__version__}")
        f = str(file).replace(".onnx", f"_saved_model{os.sep}")
        cmd = f"onnx2tf -i {file} -o {f} -osd -coion --non_verbose"
        print(cmd.split())
        subprocess.check_output(cmd.split())  # export
        LOGGER.info(f"Export success, results saved in {f} ({file_size(f):.1f} MB)")
        return f
    except Exception as e:
        LOGGER.info(f"\nExport failure: {e}")


def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False):
    try:
        assert (
            im.device.type != "cpu"
        ), "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
        try:
            import tensorrt as trt
        except Exception:
            if platform.system() == "Linux":
                __tr.check_packages(
                    ["nvidia-tensorrt"],
                    cmds=("-U --index-url https://pypi.ngc.nvidia.com",),
                )
            import tensorrt as trt

        export_onnx(model, im, file, 12, dynamic, half, simplify)  # opset 13
        onnx = file.with_suffix(".onnx")

        LOGGER.info(f"\nStarting export with TensorRT {trt.__version__}...")
        assert onnx.exists(), f"failed to export ONNX file: {onnx}"
        f = file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f"failed to load ONNX file: {onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        logger.info("Network Description:")
        for inp in inputs:
            logger.info(
                f'\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}'
            )
        for out in outputs:
            logger.info(
                f'\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}'
            )

        if dynamic:
            if im.shape[0] <= 1:
                logger.warning(
                    "WARNING: --dynamic model requires maximum --batch-size argument"
                )
            profile = builder.create_optimization_profile()
            for inp in inputs:
                if half:
                    inp.dtype = trt.float16
                profile.set_shape(
                    inp.name,
                    (1, *im.shape[1:]),
                    (max(1, im.shape[0] // 2), *im.shape[1:]),
                    im.shape,
                )
            config.add_optimization_profile(profile)

        logger.info(
            f"Building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine in {f}"
        )
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
            config.default_device_type = trt.DeviceType.GPU
        with builder.build_engine(network, config) as engine, open(f, "wb") as t:
            t.write(engine.serialize())
        logger.info(f"Export success, saved as {f} ({file_size(f):.1f} MB)")
        return f
    except Exception as e:
        logger.info(f"\nexport failure: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID export")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[256, 128],
        help="image (h, w)",
    )
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="TorchScript: optimize for mobile"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes"
    )
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument(
        "--workspace", type=int, default=4, help="TensorRT: workspace size (GB)"
    )
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--half", action="store_true", help="FP16 half-precision export"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="torchscript, onnx, openvino, engine",
    )
    args = parser.parse_args()

    t = time.time()

    include = [x.lower() for x in args.include]  # to lowercase
    fmts = tuple(export_formats()["Argument"][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(
        include
    ), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    jit, onnx, openvino, engine, tflite = flags  # export booleans

    args.device = select_device(args.device)
    if args.half:
        assert (
            args.device.type != "cpu"
        ), "--half only compatible with GPU export, i.e. use --device 0"

    model = build_model(
        get_model_name(args.weights),
        num_classes=get_nr_classes(args.weights),
        pretrained=not (
            args.weights and args.weights.is_file() and args.weights.suffix == ".pt"
        ),
        use_gpu=args.device,
    ).to(args.device)
    load_pretrained_weights(model, args.weights)
    model.eval()

    if args.optimize:
        assert (
            args.device.type == "cpu"
        ), "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # adapt input shapes for lmbn models
    if "lmbn" in str(args.weights):
        args.imgsz = (384, 128)

    im = torch.empty(args.batch_size, 3, args.imgsz[0], args.imgsz[1]).to(
        args.device
    )  # image size(1,3,640,480) BCHW iDetection
    for _ in range(2):
        y = model(im)  # dry runs
    if args.half:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    LOGGER.info(
        f"\nStarting from {args.weights} with output shape {shape} ({file_size(args.weights):.1f} MB)"
    )

    # Exports
    f = [""] * len(fmts)  # exported filenames
    if jit:
        f[0] = export_torchscript(model, im, args.weights, args.optimize)  # opset 12
    if engine:  # TensorRT required before ONNX
        f[1] = export_engine(
            model,
            im,
            args.weights,
            args.half,
            args.dynamic,
            args.simplify,
            args.workspace,
            args.verbose,
        )
    if onnx:  # OpenVINO requires ONNX
        f[2] = export_onnx(
            model, im, args.weights, args.opset, args.dynamic, args.half, args.simplify
        )  # opset 12
    if tflite:
        export_tflite(f[2])
    if openvino:
        f[3] = export_openvino(args.weights, args.half)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {args.weights.parent.resolve()}"
            f"\nVisualize:       https://netron.app"
        )
