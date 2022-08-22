import argparse
import time
import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

sys.path.append('../../')

from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.data.data_load import create_dataloader
from yolov6.utils.config import Config

from tools.partial_quantization.eval import EvalerWrapper
from tools.partial_quantization.utils import module_quant_enable, module_quant_disable, model_quant_disable
from tools.partial_quantization.utils import quant_sensitivity_save, quant_sensitivity_load
from tools.partial_quantization.ptq import do_ptq, load_ptq

from pytorch_quantization import nn as quant_nn


def quant_sensitivity_analyse(model_ptq, evaler):
    # disable all quantable layer
    model_quant_disable(model_ptq)

    # analyse each quantable layer
    quant_sensitivity = list()
    for k, m in model_ptq.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or \
           isinstance(m, quant_nn.QuantConvTranspose2d) or \
           isinstance(m, quant_nn.MaxPool2d):
            module_quant_enable(model_ptq, k)
        else:
            # module can not be quantized, continue
            continue

        eval_result = evaler.eval(model_ptq)
        print(eval_result)
        print("Quantize Layer {}, result mAP0.5 = {:0.4f}, mAP0.5:0.95 = {:0.4f}".format(k,
                                                                                          eval_result[0],
                                                                                          eval_result[1]))
        quant_sensitivity.append((k, eval_result[0], eval_result[1]))
        # disable this module sensitivity, anlayse next module
        module_quant_disable(model_ptq, k)

    return quant_sensitivity

def get_yolov6_config(key):
    # hard code
    config_dict = {'yolov6s_reopt.pt': '../../configs/repopt/yolov6s_opt.py'}
    return config_dict[key]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6s.pt', help='weights path')
    parser.add_argument('--data-root', type=str, default=None, help='train data path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--batch-number', type=int, default=1, help='batch number')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0, 1, 2, 3 or cpu')
    parser.add_argument('--calib-weights', type=str, default=None, help='weights with calibration parameter')
    parser.add_argument('--sensitivity-file', type=str, default=None, help='quantization sensitivity file')
    parser.add_argument('--data-yaml', type=str, default='../../data/coco.yaml', help='data config')
    parser.add_argument('--eval-yaml', type=str, default='./eval.yaml', help='evaluation config')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    yolov6_evaler = EvalerWrapper(eval_cfg=load_yaml(args.eval_yaml))
    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    model.eval()

    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace

    orig_mAP = yolov6_evaler.eval(model)
    print("Full Precision model mAP0.5={:.4f}, mAP0.5_0.95={:0.4f}".format(orig_mAP[0], orig_mAP[1]))

    cfg = Config.fromfile(get_yolov6_config(os.path.basename(args.weights)))
    data_cfg = load_yaml(args.data_yaml)
    train_loader, _ = create_dataloader(
        args.data_root,
        img_size=args.img_size[0],
        batch_size=args.batch_size,
        stride=32,
        hyp=dict(cfg.data_aug),
        augment=True,
        shuffle=True,
        data_dict=data_cfg)

    # Step1: do post training quantization
    if args.calib_weights is None:
        model_ptq= do_ptq(model, train_loader, args.batch_number, device)
        torch.save({'model': model_ptq}, args.weights.replace('.pt', '_calib.pt'))
    else:
        model_ptq = load_ptq(model, args.calib_weights, device)
    quant_mAP = yolov6_evaler.eval(model_ptq)
    print("Post Training Quantization model mAP0.5={:.4f}, mAP0.5_0.95={:0.4f}".format(quant_mAP[0], quant_mAP[1]))
    # Step2: do sensitivity analysis and save sensistivity results
    if args.sensitivity_file is None:
        quant_sensitivity = quant_sensitivity_analyse(model_ptq, yolov6_evaler)
        qfile = "{}_quant_sensitivity_{}_calib.txt".format(os.path.basename(args.weights).split('.')[0],
                                                         args.batch_size * args.batch_number)
        quant_sensitivity.sort(key=lambda tup: tup[2], reverse=True)
        quant_sensitivity_save(quant_sensitivity, qfile)
    else:
        quant_sensitivity = quant_sensitivity_load(args.sensitivity_file)

    quant_sensitivity.sort(key=lambda tup: tup[2], reverse=True)
    for sensitivity in quant_sensitivity:
        print(sensitivity)
