import os
from pytorch_quantization import nn as quant_nn

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def get_module(model, submodule_key):
    sub_tokens = submodule_key.split('.')
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

def module_quant_disable(model, k):
    cur_module = get_module(model, k)
    if hasattr(cur_module, '_input_quantizer'):
        cur_module._input_quantizer.disable()
    if hasattr(cur_module, '_weight_quantizer'):
        cur_module._weight_quantizer.disable()

def module_quant_enable(model, k):
    cur_module = get_module(model, k)
    if hasattr(cur_module, '_input_quantizer'):
        cur_module._input_quantizer.enable()
    if hasattr(cur_module, '_weight_quantizer'):
        cur_module._weight_quantizer.enable()

def model_quant_disable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()

def model_quant_enable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()

def concat_quant_amax_fuse(ops_list):
    if len(ops_list) <= 1:
        return

    amax = -1
    for op in ops_list:
        if hasattr(op, '_amax'):
            op_amax = op._amax.detach().item()
        elif hasattr(op, '_input_quantizer'):
            op_amax = op._input_quantizer._amax.detach().item()
        else:
            print("Not quantable op")
            exit(0)
        print("op amax = {:7.4f}, amax = {:7.4f}".format(op_amax, amax))
        if amax < op_amax:
            amax = op_amax

    print("amax = {:7.4f}".format(amax))
    for op in ops_list:
        if hasattr(op, '_amax'):
            op._amax.fill_(amax)
        elif hasattr(op, '_input_quantizer'):
            op._input_quantizer._amax.fill_(amax)

def quant_sensitivity_load(file):
    assert os.path.exists(file), print("File {} does not exist".format(file))
    quant_sensitivity = list()
    with open(file, 'r') as qfile:
        lines = qfile.readlines()
        for line in lines:
            layer, mAP1, mAP2 = line.strip('\n').split(' ')
            quant_sensitivity.append((layer, float(mAP1), float(mAP2)))

    return quant_sensitivity

def quant_sensitivity_save(quant_sensitivity, file):
    with open(file, 'w') as qfile:
        for item in quant_sensitivity:
            name, mAP1, mAP2 = item
            line = name + " " + "{:0.4f}".format(mAP1) + " " + "{:0.4f}".format(mAP2) + "\n"
            qfile.write(line)
