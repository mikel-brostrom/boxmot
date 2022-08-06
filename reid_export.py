import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import pandas as pd
import subprocess
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import LOGGER, colorstr, check_requirements
from strong_sort.deep.reid.torchreid.utils.feature_extractor import FeatureExtractor
from strong_sort.deep.reid.torchreid.models import build_model
from strong_sort.deep.reid_model_factory import get_model_name

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0


def export_formats():
    # YOLOv5 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
    ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def export_onnx(model, im, file, opset, train=False, dynamic=True, simplify=False):
    # ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        f = file.with_suffix('.onnx')
        LOGGER.info(f'\nstarting export with onnx {onnx.__version__}...')

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                },  # shape(x,3,256,128)
                'output': {
                    0: 'batch',
                }  # shape(x,2048)
            } if dynamic else None
        )
        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                cuda = torch.cuda.is_available()
                check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
                import onnxsim

                LOGGER.info(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'t0': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'simplifier failure: {e}')
        LOGGER.info(f'export success, saved as {f} ({file_size(f):.1f} MB)')
        LOGGER.info(f"run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
    except Exception as e:
        LOGGER.info(f'export failure: {e}')
    return f
        
        
def export_openvino(file, dynamic, half, prefix=colorstr('OpenVINO:')):
    f = str(file).replace('.onnx', f'_openvino_model{os.sep}')
    # YOLOv5 OpenVINO export
    try:
        check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.inference_engine as ie

        LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
        f = str(file).replace('.onnx', f'_openvino_model{os.sep}')
        dyn_shape = [-1,3,256,128] if dynamic else None
        cmd = f"mo \
            --input_model {file} \
            --output_dir {f} \
            --data_type {'FP16' if half else 'FP32'}"
        
        if dyn_shape is not None:
            cmd + f"--input_shape {dyn_shape}"

        subprocess.check_output(cmd.split())  # export

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
    return f
        

def export_tflite(file, half, prefix=colorstr('TFLite:')):
    # YOLOv5 OpenVINO export
    try:
        check_requirements(('openvino2tensorflow', 'tensorflow', 'tensorflow_datasets'))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.inference_engine as ie
        LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
        output = Path(str(file).replace(f'_openvino_model{os.sep}', f'_tflite_model{os.sep}'))
        modelxml = list(Path(file).glob('*.xml'))[0]
        cmd = f"openvino2tensorflow \
            --model_path {modelxml} \
            --model_output_path {output} \
            --output_pb \
            --output_saved_model \
            --output_no_quant_float32_tflite \
            --output_dynamic_range_quant_tflite"
        subprocess.check_output(cmd.split())  # export

        LOGGER.info(f'{prefix} export success, results saved in {output} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CPHD train")
    parser.add_argument(
        "-d",
        "--dynamic",
        action="store_true",
        help="dynamic model input",
    )
    parser.add_argument(
        "-p",
        "--weights",
        type=Path,
        default="./weight/mobilenetv2_x1_0_msmt17.pt",
        help="Path to weights",
    )
    parser.add_argument(
        "-hp",
        "--half_precision",
        action="store_true",
        help="transform model to half precision",
    )
    parser.add_argument(
        '--imgsz', '--img', '--img-size',
        nargs='+',
        type=int,
        default=[256, 128],
        help='image (h, w)'
    )
    parser.add_argument('--include',
                        nargs='+',
                        default=['onnx', 'openvino', 'tflite'],
                        help='onnx, openvino, tflite')
    args = parser.parse_args()

    # Build model
    extractor = FeatureExtractor(
        # get rid of dataset information DeepSort model name
        model_name=get_model_name(args.weights),
        model_path=args.weights,
        device=str('cpu')
    )

    include = [x.lower() for x in args.include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    onnx, openvino, tflite = flags  # export booleans
    
    im = torch.zeros(1, 3, args.imgsz[0], args.imgsz[1]).to('cpu')  # image size(1,3,640,480) BCHW iDetection
    if onnx:
        f = export_onnx(extractor.model.eval(), im, args.weights, 12, train=False, dynamic=args.dynamic, simplify=True)  # opset 12
    if openvino:
        f = export_openvino(f, dynamic=args.dynamic, half=False)
    if tflite:
        export_tflite(f, False)
