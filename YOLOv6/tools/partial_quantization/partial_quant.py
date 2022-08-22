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

from tools.partial_quantization.eval import EvalerWrapper
from tools.partial_quantization.utils import get_module, concat_quant_amax_fuse, quant_sensitivity_load
from tools.partial_quantization.ptq import load_ptq, partial_quant

from pytorch_quantization import nn as quant_nn

# concat_fusion_list = [
#     ('backbone.ERBlock_5.2.m', 'backbone.ERBlock_5.2.cv2.conv'),
#     ('backbone.ERBlock_5.0.rbr_reparam', 'neck.Rep_p4.conv1.rbr_reparam'),
#     ('backbone.ERBlock_4.0.rbr_reparam', 'neck.Rep_p3.conv1.rbr_reparam'),
#     ('neck.upsample1.upsample_transpose', 'neck.Rep_n3.conv1.rbr_reparam'),
#     ('neck.upsample0.upsample_transpose', 'neck.Rep_n4.conv1.rbr_reparam')
# ]

opt_concat_fusion_list = [
    ('backbone.ERBlock_5.2.m', 'backbone.ERBlock_5.2.cv2.conv'),
    ('backbone.ERBlock_5.0.conv', 'neck.Rep_p4.conv1.conv'),
    ('backbone.ERBlock_4.0.conv', 'neck.Rep_p3.conv1.conv'),
    ('neck.upsample1.upsample_transpose', 'neck.Rep_n3.conv1.conv'),
    ('neck.upsample0.upsample_transpose', 'neck.Rep_n4.conv1.conv')
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6s_reopt.pt', help='weights path')
    parser.add_argument('--calib-weights', type=str, default='./yolov6s_reopt_calib.pt', help='calib weights path')
    parser.add_argument('--data-root', type=str, default=None, help='train data path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--export-batch-size', type=int, default=None, help='export batch size')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0, 1, 2, 3 or cpu')
    parser.add_argument('--sensitivity-file', type=str, default=None, help='quantization sensitivity file')
    parser.add_argument('--quant-boundary', type=int, default=None, help='quantization boundary')
    parser.add_argument('--eval-yaml', type=str, default='./eval.yaml', help='evaluation config')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    model.eval()
    yolov6_evaler = EvalerWrapper(eval_cfg=load_yaml(args.eval_yaml))
    orig_mAP = yolov6_evaler.eval(model)

    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace

    model_ptq = load_ptq(model, args.calib_weights, device)

    quant_sensitivity = quant_sensitivity_load(args.sensitivity_file)
    quant_sensitivity.sort(key=lambda tup: tup[2], reverse=True)
    boundary = args.quant_boundary
    quantable_ops = [qops[0] for qops in quant_sensitivity[:boundary+1]]
    # only quantize ops in quantable_ops list
    partial_quant(model_ptq, quantable_ops=quantable_ops)
    # concat amax fusion
    for sub_fusion_list in opt_concat_fusion_list:
        ops = [get_module(model_ptq, op_name) for op_name in sub_fusion_list]
        concat_quant_amax_fuse(ops)

    part_mAP = yolov6_evaler.eval(model_ptq)
    print(part_mAP)
    # ONNX export
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    if args.export_batch_size is None:
        img = torch.zeros(1, 3, *args.img_size).to(device)
        export_file = args.weights.replace('.pt', '_partial_dynamic.onnx')  # filename
        dynamic_axes = {"image_arrays": {0: "batch"}, "outputs": {0: "batch"}}
        torch.onnx.export(model_ptq,
                          img,
                          export_file,
                          verbose=False,
                          opset_version=13,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['image_arrays'],
                          output_names=['outputs'],
                          dynamic_axes=dynamic_axes
                         )
    else:
        img = torch.zeros(args.export_batch_size, 3, *args.img_size).to(device)
        export_file = args.weights.replace('.pt', '_partial_bs{}.onnx'.format(args.export_batch_size))  # filename
        torch.onnx.export(model_ptq,
                          img,
                          export_file,
                          verbose=False,
                          opset_version=13,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['image_arrays'],
                          output_names=['outputs']
                          )
