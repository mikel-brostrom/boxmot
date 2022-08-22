#
#  QAT_quantizer.py
#  YOLOv6
#
#  Created by Meituan on 2022/06/24.
#  Copyright © 2022
#

from absl import logging
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

# Call this function before defining the model
def tensorrt_official_qat():
    # Quantization Aware Training is based on Straight Through Estimator (STE) derivative approximation.
    # It is some time known as “quantization aware training”.

    # PyTorch-Quantization is a toolkit for training and evaluating PyTorch models with simulated quantization.
    # Quantization can be added to the model automatically, or manually, allowing the model to be tuned for accuracy and performance.
    # Quantization is compatible with NVIDIAs high performance integer kernels which leverage integer Tensor Cores.
    # The quantized model can be exported to ONNX and imported by TensorRT 8.0 and later.
    # https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization/examples/finetune_quant_resnet50.ipynb

    # The example to export the
    # model.eval()
    # quant_nn.TensorQuantizer.use_fb_fake_quant = True # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    # opset_version = 13

    # Export ONNX for multiple batch sizes
    # print("Creating ONNX file: " + onnx_filename)
    # dummy_input = torch.randn(batch_onnx, 3, 224, 224, device='cuda') #TODO: switch input dims by model
    # torch.onnx.export(model, dummy_input, onnx_filename, verbose=False, opset_version=opset_version, enable_onnx_checker=False, do_constant_folding=True)
    try:
        quant_modules.initialize()
    except NameError:
        logging.info("initialzation error for quant_modules")

# def QAT_quantizer():
# coming soon
