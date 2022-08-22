import torch
import torch.nn as nn
import copy

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from tools.partial_quantization.utils import set_module, module_quant_disable

def collect_stats(model, data_loader, batch_number, device='cuda'):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, data_tuple in enumerate(data_loader):
        image = data_tuple[0]
        image = image.float()/255.0
        model(image.to(device))
        if i + 1 >= batch_number:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")

def quantable_op_check(k, quantable_ops):
    if quantable_ops is None:
        return True

    if k in quantable_ops:
        return True
    else:
        return False

def quant_model_init(model, device):

    model_ptq = copy.deepcopy(model)
    model_ptq.eval()
    model_ptq.to(device)
    # print(model)
    conv2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    conv2d_input_default_desc = QuantDescriptor(num_bits=8, calib_method='histogram')

    convtrans2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    convtrans2d_input_default_desc = QuantDescriptor(num_bits=8, calib_method='histogram')

    for k, m in model_ptq.named_modules():
        # print(k, m)
        if isinstance(m, nn.Conv2d):
            # print("in_channel = {}".format(m.in_channels))
            # print("out_channel = {}".format(m.out_channels))
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = quant_nn.QuantConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              quant_desc_input = conv2d_input_default_desc,
                                              quant_desc_weight = conv2d_weight_default_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model_ptq, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            # print("in_channel = {}".format(m.in_channels))
            # print("out_channel = {}".format(m.out_channels))
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                       out_channels,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       quant_desc_input = convtrans2d_input_default_desc,
                                                       quant_desc_weight = convtrans2d_weight_default_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(model_ptq, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            # print("dilation = {}".format(m.dilation))
            # print("ceil mode = {}".format(m.ceil_mode))
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      ceil_mode,
                                                      quant_desc_input = conv2d_input_default_desc)
            set_module(model_ptq, k, quant_maxpool2d)
        else:
            # module can not be quantized, continue
            continue

    return model_ptq.to(device)

def do_ptq(model, train_loader, batch_number, device):
    model_ptq = quant_model_init(model, device)
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model_ptq, train_loader, batch_number)
        compute_amax(model_ptq, method='entropy')
    return model_ptq

def load_ptq(model, calib_path, device):
    model_ptq = quant_model_init(model, device)
    model_ptq.load_state_dict(torch.load(calib_path)['model'].state_dict())
    return model_ptq

def partial_quant(model_ptq, quantable_ops=None):
    # ops not in quantable_ops will reserve full-precision.
    for k, m in model_ptq.named_modules():
        if quantable_op_check(k, quantable_ops):
            continue
        # enable full-precision
        if isinstance(m, quant_nn.QuantConv2d) or \
            isinstance(m, quant_nn.QuantConvTranspose2d) or \
            isinstance(m, quant_nn.QuantMaxPool2d):
            module_quant_disable(model_ptq, k)
